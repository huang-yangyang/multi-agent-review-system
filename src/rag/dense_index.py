"""Qdrant 稠密检索索引（带元数据 pre-filter）。

使用 Qdrant 本地模式（磁盘存储）管理稠密向量，payload 携带 doc_path / file_name /
visibility / domain 等元数据。检索时通过 filter 实现 pre-filter：先按文档权限与领域过滤
候选集，再取 top_k，从根本上解决「先取 top_k 再剔权限」的 post-filter 在大体量下
RRF=0 的召回空洞问题。

与旧 FAISS 实现的区别：
- 不再用文件型 FAISS（faiss.index / faiss_meta.json），改为 Qdrant 集合。
- 向量检索在「权限可访问」的子集内进行，密度不随总量增长而稀释。
- 持久化由 Qdrant 本地引擎负责，无需手动 save/load 二进制。
"""

import logging
import uuid
from pathlib import Path
from typing import List, Tuple, Optional, Set

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
    FilterSelector,
)

logger = logging.getLogger(__name__)

COLLECTION_NAME = "doc_chunks"

# 稳定的命名空间，用于将业务 doc_id 映射为确定性的 UUID point id
_POINT_NS = uuid.UUID("a1b2c3d4-0000-4000-8000-000000000001")


def _point_id(doc_id: str) -> str:
    """将业务 doc_id 映射为稳定的 UUID（Qdrant point id）。"""
    return str(uuid.uuid5(_POINT_NS, doc_id))


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else arr


class DenseIndex:
    """Qdrant 稠密向量索引（pre-filter 元数据过滤）。"""

    def __init__(self, dim: int, storage_path: str = None):
        self.dim = dim
        if storage_path:
            Path(storage_path).mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=storage_path)
        else:
            # 内存模式（测试用）
            self._client = QdrantClient(location=":memory:")
        self._ensure_collection()

    def _ensure_collection(self):
        if not self._client.collection_exists(COLLECTION_NAME):
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )
            # 为过滤字段建立 payload 索引，加速元数据过滤
            for field in ("doc_path", "file_name", "doc_prefix", "visibility", "domain"):
                try:
                    self._client.create_payload_index(COLLECTION_NAME, field, "keyword")
                except Exception as e:  # 已存在或版本不支持时忽略
                    logger.debug(f"Qdrant payload index 创建跳过 {field}: {e}")
            logger.info("Qdrant 集合已创建: %s", COLLECTION_NAME)
        else:
            logger.info("Qdrant 集合已存在: %s", COLLECTION_NAME)

    # ── 写入 ──

    def add(
        self,
        vectors: np.ndarray,
        chunks: List[str],
        file_names: List[str],
        doc_ids: List[str],
        doc_paths: Optional[List[str]] = None,
        visibility: str = "admin",
        domain: str = "general",
    ):
        if len(vectors) == 0:
            return

        points = []
        for i in range(len(doc_ids)):
            doc_id = doc_ids[i]
            doc_path = (doc_paths[i] if doc_paths else file_names[i])
            points.append(PointStruct(
                id=_point_id(doc_id),
                vector=_normalize(vectors[i]).tolist(),
                payload={
                    "doc_id": doc_id,
                    "chunk": chunks[i],
                    "file_name": file_names[i],
                    "doc_path": doc_path,
                    "doc_prefix": "::".join(doc_id.split("::")[:2]),
                    "visibility": visibility,
                    "domain": domain,
                },
            ))

        self._client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info("DenseIndex(Qdrant) 已添加 %d 条", len(points))

    # ── 过滤条件构造 ──

    @staticmethod
    def _build_filter(
        accessible_paths: Optional[Set[str]],
        domain: Optional[str],
    ) -> Optional[Filter]:
        """构造 pre-filter。

        - accessible_paths 为 None → 不过滤（管理员/看全部）。
        - accessible_paths 为空集 → MatchAny([])，返回空集（无权限）。
        - accessible_paths 为非空集 → 仅匹配这些文档路径。
        - domain 非空且非 general → 额外按领域过滤。
        """
        conditions = []
        if accessible_paths is not None:
            conditions.append(
                FieldCondition(
                    key="doc_path",
                    match=MatchAny(any=list(accessible_paths)),
                )
            )
        if domain and domain != "general":
            conditions.append(
                FieldCondition(key="domain", match=MatchValue(value=domain))
            )
        if not conditions:
            return None
        return Filter(must=conditions)

    # ── 检索（pre-filter）──

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int = 10,
        accessible_paths: Optional[Set[str]] = None,
        domain: Optional[str] = None,
    ) -> List[Tuple[str, str, float]]:
        qfilter = self._build_filter(accessible_paths, domain)
        resp = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=_normalize(query_vec).tolist(),
            query_filter=qfilter,
            limit=top_k,
            with_payload=True,
        )
        results = []
        for h in resp.points:
            payload = h.payload or {}
            results.append(
                (payload.get("doc_id", ""), payload.get("chunk", ""), float(h.score))
            )
        return results

    # ── 删除 ──

    def remove_by_prefix(self, prefix: str):
        """删除指定 doc_prefix 下的所有点（整文档粒度）。"""
        flt = Filter(must=[
            FieldCondition(key="doc_prefix", match=MatchValue(value=prefix))
        ])
        try:
            self._client.delete(
                COLLECTION_NAME,
                points_selector=FilterSelector(filter=flt),
            )
            logger.info("DenseIndex remove_by_prefix: 已删除 prefix=%s", prefix)
        except Exception as e:
            logger.warning("DenseIndex remove_by_prefix 失败: %s", e)

    # ── 统计 ──

    def count(self) -> int:
        try:
            return self._client.count(COLLECTION_NAME).count
        except Exception:
            return 0

    @property
    def ntotal(self) -> int:
        return self.count()
