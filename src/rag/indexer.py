"""全局索引编排器。

流程: 解析 → 分块 → embedding → 双路索引（BM25 + FAISS）
持久化: SQLite (indexes.db) + FAISS 二进制文件
支持 SHA256 去重、增量添加/移除、RRF 混合检索。
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .parser import parse_document
from .chunker import chunk_text
from .embedder import Embedder
from .bm25_index import BM25Index
from .dense_index import DenseIndex
from .hybrid_index import rrf_fuse
from .reranker import Reranker

logger = logging.getLogger(__name__)

# RRF 默认参数
RRF_K = 60
RERANK_TOP_K = 20
RERANK_FINAL_K = 10


class Indexer:
    """全局索引编排器。"""

    def __init__(self, uploads_dir: str, indexes_dir: str):
        self.uploads_dir = Path(uploads_dir)
        self.indexes_dir = Path(indexes_dir)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = self.indexes_dir / "indexes.db"
        self.bm25_path = self.indexes_dir / "bm25.pkl"
        self.faiss_index_path = self.indexes_dir / "faiss.index"
        self.faiss_meta_path = self.indexes_dir / "faiss_meta.json"

        self._lock = threading.Lock()

        # Embedding
        self.embedder = Embedder()
        self.dim = self.embedder.dim

        # 双路索引
        self.bm25 = BM25Index()
        self.dense = DenseIndex(dim=self.dim)

        # 精排器
        self.reranker = Reranker()

        # 领域映射：{doc_prefix → domain}
        self._doc_domain_map: Dict[str, str] = {}

        # 数据库初始化
        self._init_db()

        # 从磁盘恢复
        self._restore()

    def _init_db(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL UNIQUE,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER,
                    chunk_count INTEGER,
                    status TEXT DEFAULT 'indexed',
                    domain TEXT DEFAULT 'general',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    doc_id TEXT UNIQUE NOT NULL,
                    file_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)
            """)

            # ── 向后兼容迁移：如果旧表缺少 domain 列，自动补加 ──
            try:
                conn.execute("SELECT domain FROM files LIMIT 0")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE files ADD COLUMN domain TEXT DEFAULT 'general'")
                logger.info("数据库迁移：files 表已添加 domain 列（默认值 'general'）")

            conn.commit()

    def _restore(self):
        if self.bm25_path.exists():
            try:
                self.bm25 = BM25Index.load(str(self.bm25_path))
                logger.info("BM25 索引已恢复")
            except Exception as e:
                logger.warning(f"BM25 索引恢复失败: {e}")

        if self.faiss_index_path.exists() and self.faiss_meta_path.exists():
            try:
                self.dense = DenseIndex.load(str(self.faiss_index_path), str(self.faiss_meta_path))
                logger.info("FAISS 索引已恢复")
            except Exception as e:
                logger.warning(f"FAISS 索引恢复失败: {e}")

        # 从 SQLite 重建领域映射
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("SELECT domain FROM files LIMIT 0")  # 探测列是否存在
                rows = conn.execute(
                    "SELECT file_name, file_hash, domain FROM files WHERE status='indexed'"
                ).fetchall()
            for name, fhash, domain in rows:
                if domain:
                    prefix = f"{name}::{fhash}"
                    self._doc_domain_map[prefix] = domain
            logger.info(f"领域映射已恢复: {len(self._doc_domain_map)} 条")
        except Exception as e:
            logger.warning(f"领域映射恢复失败: {e}")

    def _persist(self):
        self.bm25.save(str(self.bm25_path))
        self.dense.save(str(self.faiss_index_path), str(self.faiss_meta_path))
        logger.info("索引已持久化")

    def _compute_hash(self, file_path: str) -> str:
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    def check_duplicate(self, file_path: str) -> Optional[dict]:
        file_hash = self._compute_hash(file_path)
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT file_name, file_path FROM files WHERE file_hash = ?",
                (file_hash,)
            ).fetchone()
        if row:
            existing_path = row[1]
            if Path(existing_path).exists():
                return {"duplicate": True, "hash": file_hash, "existing_file": existing_path}
            # 陈旧记录：DB 有记录但磁盘文件已不存在，清理并放行
            logger.warning(f"检测到陈旧索引记录，自动清理: {existing_path}")
            conn.execute("DELETE FROM chunks WHERE file_path = ?", (existing_path,))
            conn.execute("DELETE FROM files WHERE file_path = ?", (existing_path,))
            conn.commit()
        return None

    def index_file(self, file_path: str, domain: str = "general") -> dict:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        try:
            text = parse_document(file_path)
        except Exception as e:
            raise RuntimeError(f"文档解析失败: {e}")

        is_md = path.suffix.lower() == ".md"
        chunks = chunk_text(text, is_markdown=is_md)

        if not chunks:
            raise RuntimeError("文档分块结果为空")

        # Doc ID 前缀
        doc_prefix = f"{path.name}::{self._compute_hash(file_path)}"
        doc_ids = [f"{doc_prefix}::{i}" for i in range(len(chunks))]
        file_names = [path.name] * len(chunks)

        with self._lock:

            try:
                vectors = self.embedder.encode(chunks)
            except Exception as e:
                raise RuntimeError(f"Embedding 生成失败: {e}")

            self.bm25.add(chunks, file_names, doc_ids)
            self.dense.add(vectors, chunks, file_names, doc_ids)

            # 写入 chunks 表
            with sqlite3.connect(str(self.db_path)) as conn:
                for i, (did, c) in enumerate(zip(doc_ids, chunks)):
                    conn.execute(
                        "INSERT OR REPLACE INTO chunks (doc_id, file_path, chunk_index, content) VALUES (?, ?, ?, ?)",
                        (did, str(path), i, c)
                    )
                conn.commit()

            # 写入 files 表
            file_hash = self._compute_hash(file_path)
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO files (file_name, file_path, file_hash, file_size, chunk_count, status, domain) VALUES (?, ?, ?, ?, ?, 'indexed', ?)",
                    (path.name, str(path), file_hash, path.stat().st_size, len(chunks), domain)
                )
                conn.commit()

            # 维护内存领域映射
            doc_prefix = f"{path.name}::{file_hash}"
            self._doc_domain_map[doc_prefix] = domain

            self._persist()

        logger.info(f"索引完成: {path.name} ({len(chunks)} 块, domain={domain})")
        return {
            "file_name": path.name,
            "status": "indexed",
            "chunk_count": len(chunks),
        }

    def remove_file(self, file_path: str) -> dict:
        path = Path(file_path)
        name = path.name

        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT file_path, file_hash FROM files WHERE file_name = ? OR file_path = ?",
                (name, str(path))
            ).fetchone()

        if not row:
            return {"status": "not_found", "file": name}

        actual_path, file_hash = row
        doc_prefix = f"{name}::{file_hash}"

        with self._lock:
            self.bm25.remove_by_prefix(doc_prefix)
            self.dense.remove_by_prefix(doc_prefix)

            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("DELETE FROM chunks WHERE file_path = ?", (actual_path,))
                conn.execute("DELETE FROM files WHERE file_path = ?", (actual_path,))
                conn.commit()

            # 清理领域映射
            self._doc_domain_map.pop(doc_prefix, None)

            self._persist()

        logger.info(f"已移除: {name}")
        return {"status": "removed", "file": name}

    def _get_domain(self, doc_id: str) -> str:
        """根据 doc_id 反查领域标签。

        doc_id 格式: {file_name}::{file_hash}::{chunk_index}
        提取前缀后从 _doc_domain_map 查询。
        """
        prefix = "::".join(doc_id.split("::")[:2])
        return self._doc_domain_map.get(prefix, "general")

    def _filter_by_domain(
        self,
        candidates: List[Dict[str, Any]],
        domain: Optional[str],
    ) -> List[Dict[str, Any]]:
        """按领域过滤候选 chunk。

        Args:
            candidates: [{"id": doc_id, "chunk": ..., "score": ...}, ...]
            domain: 目标领域标签，None 或 "general" 时不过滤（返回全部）。

        Returns:
            过滤后的候选列表，保留原顺序。
        """
        if not domain or domain == "general":
            return candidates
        return [c for c in candidates if self._get_domain(c["id"]) == domain]

    def search(
        self,
        query: str,
        top_k: int = 10,
        domain: Optional[str] = None,
    ) -> dict:
        with self._lock:
            bm25_candidates_raw = self.bm25.search(query, top_k=20)
            bm25_candidates = [
                {"id": doc_id, "chunk": chunk, "score": score}
                for doc_id, chunk, score in bm25_candidates_raw
            ]

            query_vec = self.embedder.encode_single(query)
            dense_candidates_raw = self.dense.search(query_vec, top_k=20)
            dense_candidates = [
                {"id": doc_id, "chunk": chunk, "score": float(score)}
                for doc_id, chunk, score in dense_candidates_raw
            ]

        # ── 领域后过滤 ──
        if domain and domain != "general":
            dense_candidates = self._filter_by_domain(dense_candidates, domain)
            bm25_candidates = self._filter_by_domain(bm25_candidates, domain)
            logger.debug(f"领域过滤后: dense={len(dense_candidates)}, bm25={len(bm25_candidates)}")

        # RRF 融合
        fused = rrf_fuse(dense_candidates, bm25_candidates, k=RRF_K)

        # 精排
        rerank_input = fused[:RERANK_TOP_K]
        reranked = self.reranker.rerank(query, rerank_input, top_k=top_k)

        # 构建最终结果
        results = []
        for item in reranked:
            results.append({
                "doc_id": item["id"],
                "chunk": item["chunk"],
                "score": item.get("rerank_score", item.get("rrf_score", 0)),
            })

        stats = {
            "dense_hits": len(dense_candidates),
            "bm25_hits": len(bm25_candidates),
            "fused_hits": len(fused),
            "final_hits": len(results),
        }

        return {"results": results, "stats": stats}

    def list_files(self) -> List[dict]:
        with sqlite3.connect(str(self.db_path)) as conn:
            # 兼容迁移：旧库可能无 domain 列
            try:
                rows = conn.execute(
                    "SELECT id, file_name, file_path, file_size, chunk_count, status, domain, created_at FROM files ORDER BY id DESC"
                ).fetchall()
            except sqlite3.OperationalError:
                rows = conn.execute(
                    "SELECT id, file_name, file_path, file_size, chunk_count, status, 'general', created_at FROM files ORDER BY id DESC"
                ).fetchall()

        return [
            {
                "id": r[0],
                "file_name": r[1],
                "file_path": r[2],
                "file_size": r[3],
                "chunk_count": r[4],
                "status": r[5],
                "domain": r[6] if len(r) > 6 else "general",
                "created_at": r[7] if len(r) > 7 else None,
            }
            for r in rows
        ]

    def get_stats(self) -> dict:
        with sqlite3.connect(str(self.db_path)) as conn:
            file_count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            total_size = conn.execute("SELECT COALESCE(SUM(file_size), 0) FROM files").fetchone()[0]

            # 领域分布统计（兼容旧库）
            try:
                domain_rows = conn.execute(
                    "SELECT domain, COUNT(*) FROM files GROUP BY domain"
                ).fetchall()
                domain_counts = {d: c for d, c in domain_rows}
            except sqlite3.OperationalError:
                domain_counts = {"general": file_count}

        return {
            "file_count": file_count,
            "chunk_count": chunk_count,
            "total_size_bytes": total_size,
            "bm25_docs": len(self.bm25._chunks),
            "dense_docs": self.dense._index.ntotal if self.dense._index else 0,
            "domains": domain_counts,
        }

    def get_full_document(self, name_fragment: str) -> Optional[str]:
        """获取已索引文档的完整文本（所有 chunks 按顺序拼接）。

        通过文件名模糊匹配找到文档，从 chunks 表取出所有分块，
        按 chunk_index 排序后拼接为完整文本。

        Args:
            name_fragment: 文件名片段，支持模糊匹配（LIKE %name_fragment%）。
                           例如 "信用风险评估操作规程" 可匹配
                           "信用风险评估操作规程_1783002054394.md"。

        Returns:
            完整文档文本，未找到或已删除则返回 None。
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # 模糊匹配文件名，取第一条已索引记录
                row = conn.execute(
                    "SELECT file_name, file_path FROM files "
                    "WHERE file_name LIKE ? AND status = 'indexed' "
                    "ORDER BY created_at DESC LIMIT 1",
                    (f"%{name_fragment}%",)
                ).fetchone()

                if not row:
                    logger.info(
                        f"get_full_document: no indexed file matching '{name_fragment}'"
                    )
                    return None

                matched_name, file_path = row

                # 取出所有 chunks，按 chunk_index 排序
                chunk_rows = conn.execute(
                    "SELECT chunk_index, content FROM chunks "
                    "WHERE file_path = ? "
                    "ORDER BY chunk_index",
                    (file_path,)
                ).fetchall()

                if not chunk_rows:
                    logger.warning(
                        f"get_full_document: no chunks found for '{matched_name}'"
                    )
                    return None

                full_text = "\n\n".join(row[1] for row in chunk_rows)
                logger.info(
                    f"get_full_document: '{matched_name}' — "
                    f"{len(chunk_rows)} chunks, {len(full_text)} chars"
                )
                return full_text

        except Exception as e:
            logger.error(
                f"get_full_document failed for '{name_fragment}': {e}",
                exc_info=True,
            )
            return None

    def find_file_by_name(self, name_fragment: str) -> Optional[str]:
        """根据文件名片段查找已索引文件的存储路径。

        Args:
            name_fragment: 文件名片段，支持模糊匹配。

        Returns:
            文件的完整路径，未找到返回 None。
        """
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                row = conn.execute(
                    "SELECT file_path FROM files "
                    "WHERE file_name LIKE ? AND status = 'indexed' "
                    "ORDER BY created_at DESC LIMIT 1",
                    (f"%{name_fragment}%",)
                ).fetchone()
                return row[0] if row else None
        except Exception as e:
            logger.error(
                f"find_file_by_name failed for '{name_fragment}': {e}",
                exc_info=True,
            )
            return None


_indexer_instance: Optional[Indexer] = None


def get_indexer(uploads_dir: str = "", indexes_dir: str = "") -> Indexer:
    global _indexer_instance
    if _indexer_instance is None:
        _indexer_instance = Indexer(uploads_dir=uploads_dir, indexes_dir=indexes_dir)
    return _indexer_instance
