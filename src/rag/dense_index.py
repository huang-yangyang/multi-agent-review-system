"""FAISS 稠密检索索引。

使用 FAISS IndexFlatIP（内积）存储稠密向量，配合余弦相似度检索。
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    raise RuntimeError("faiss-cpu 未安装，请执行: pip install faiss-cpu")

logger = logging.getLogger(__name__)


class DenseIndex:
    """FAISS 稠密向量索引。"""

    def __init__(self, dim: int):
        self.dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._chunks: List[str] = []
        self._file_names: List[str] = []
        self._doc_ids: List[str] = []

    def add(self, vectors: np.ndarray, chunks: List[str], file_names: List[str], doc_ids: List[str]):
        if len(vectors) == 0:
            return

        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)

        self._index.add(vectors)
        self._chunks.extend(chunks)
        self._file_names.extend(file_names)
        self._doc_ids.extend(doc_ids)
        logger.info(f"DenseIndex 已添加 {len(vectors)} 条，总计 {self._index.ntotal} 条")

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> List[Tuple[str, str, float]]:
        if self._index.ntotal == 0:
            return []

        query_vec = query_vec.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, min(top_k, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self._doc_ids):
                results.append((self._doc_ids[idx], self._chunks[idx], float(score)))
        return results

    def remove_by_prefix(self, prefix: str):
        keep_indices = [i for i, did in enumerate(self._doc_ids) if not did.startswith(prefix)]
        removed = len(self._doc_ids) - len(keep_indices)
        if removed == 0:
            logger.info(f"DenseIndex remove_by_prefix: 无匹配 '{prefix}'")
            return

        self._chunks = [self._chunks[i] for i in keep_indices]
        self._file_names = [self._file_names[i] for i in keep_indices]
        self._doc_ids = [self._doc_ids[i] for i in keep_indices]

        if self._chunks:
            new_index = faiss.IndexFlatIP(self.dim)
            kept_vectors = self._index.reconstruct_n(0, self._index.ntotal)
            kept_vectors = kept_vectors[keep_indices].astype(np.float32)
            new_index.add(kept_vectors)
            self._index = new_index
        else:
            self._index = faiss.IndexFlatIP(self.dim)

        logger.info(f"DenseIndex remove_by_prefix: 移除 {removed} 条，剩余 {self._index.ntotal} 条")

    def save(self, index_path: str, meta_path: str):
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, index_path)

        meta = {
            "dim": self.dim,
            "chunks": self._chunks,
            "file_names": self._file_names,
            "doc_ids": self._doc_ids,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        logger.info(f"DenseIndex 已保存: {index_path}, {meta_path}")

    def save_index(self, index_path: str):
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, index_path)
        logger.info(f"DenseIndex FAISS 二进制已保存: {index_path}")

    def to_meta_dict(self) -> dict:
        return {
            "dim": self.dim,
            "chunks": self._chunks,
            "file_names": self._file_names,
            "doc_ids": self._doc_ids,
        }

    @classmethod
    def load(cls, index_path: str, meta_path: str) -> "DenseIndex":
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        inst = cls(dim=meta["dim"])
        inst._index = faiss.read_index(index_path)
        inst._chunks = meta["chunks"]
        inst._file_names = meta["file_names"]
        inst._doc_ids = meta["doc_ids"]
        logger.info(f"DenseIndex 已加载: {index_path} ({inst._index.ntotal} 条)")
        return inst
