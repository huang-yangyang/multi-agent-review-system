"""语义缓存 — embedding 向量 + FAISS 余弦相似度，业界主流方案。

流程：问题 → embedding → FAISS IndexFlatIP.search() → 余弦阈值命中
附加：源文件新鲜度校验（命中后比对 mtime，过期自动剔除）
"""

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .config import config

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────
_DEFAULT_THRESHOLD = float(os.environ.get("SEMANTIC_CACHE_THRESHOLD", "0.75"))
_PRIMARY_MODEL = "BAAI/bge-small-zh-v1.5"       # 512 维，中文优化，与 RAG 统一
_FALLBACK_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384 维，回退


# ── 缓存类 ────────────────────────────────────────────

class SemanticCache:
    """语义缓存 — embedding 向量 + FAISS 余弦匹配。

    写入时：embedding(question) → 归一化 → FAISS IndexFlatIP.add()
    查询时：embedding(question) → 归一化 → FAISS IndexFlatIP.search(k=1) → 阈值判定
    命中后：源文件 mtime 校验，过期自动剔除。
    """

    def __init__(self, dimension: int = 512, threshold: float = _DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._dim = dimension
        self._index = None
        self._questions: list[str] = []
        self._answers: list[str] = []
        self._source_files: list[str] = []       # | 分隔的多文件路径
        self._indexed_at: list[float] = []        # 最大 mtime
        self._domains: list[str] = []              # 领域分类: finance/contract/law/general
        self._model = None
        self._cache_dir = Path(config.rag.indexes_dir) / "semantic_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._index_fingerprint: str = ""
        self._try_load()
        self._try_load_model()

    # ── 属性 ──────────────────────────────────────

    @property
    def size(self) -> int:
        return len(self._questions)

    # ── 模型管理 ──────────────────────────────────

    def _compute_index_fingerprint(self) -> str:
        indexes_dir = Path(config.rag.indexes_dir)
        parts = []
        for fname in ("faiss.index", "meta.json", "indexes.db"):
            fp = indexes_dir / fname
            if fp.exists():
                parts.append(fname + ":" + str(fp.stat().st_mtime))
        if not parts:
            return ""
        raw = "|".join(parts)
        return hashlib.md5(raw.encode()).hexdigest()

    def _is_fingerprint_valid(self) -> bool:
        current = self._compute_index_fingerprint()
        if not current:
            return True
        if self._index_fingerprint and self._index_fingerprint != current:
            logger.info("索引指纹变更，清空语义缓存")
            with self._lock:
                self._index = None
                self._questions.clear()
                self._answers.clear()
                self._source_files.clear()
                self._indexed_at.clear()
                self._domains.clear()
            self._index_fingerprint = current
            return False
        if not self._index_fingerprint:
            self._index_fingerprint = current
        return True

    # ── 源文件新鲜度校验 ─────────────────────────

    def _is_entry_fresh(self, idx: int) -> bool:
        """检查第 idx 条缓存引用的源文件是否未被修改。

        支持多文件：source_file 以 ``|`` 分隔多个路径，indexed_at 存储最大 mtime。
        """
        if idx >= len(self._source_files) or not self._source_files[idx]:
            return True

        source_paths = [p for p in self._source_files[idx].split("|") if p.strip()]
        if not source_paths:
            return True

        stored_mtime = self._indexed_at[idx] if idx < len(self._indexed_at) else 0.0

        for sp in source_paths:
            fp = Path(sp.strip())
            if not fp.exists():
                logger.debug(f"语义缓存: 源文件已删除 idx={idx} path={fp}")
                return False
            current_mtime = fp.stat().st_mtime
            if current_mtime > stored_mtime + 1e-6:
                logger.debug(
                    f"语义缓存: 源文件已修改 idx={idx} path={fp} "
                    f"stored={stored_mtime} current={current_mtime}"
                )
                return False
        return True

    def _invalidate_entry(self, idx: int):
        """从所有存储中移除第 idx 条缓存条目。"""
        import faiss
        with self._lock:
            if self._index is not None and self._index.ntotal > 0:
                n = self._index.ntotal
                keep_mask = np.ones(n, dtype=bool)
                keep_mask[idx] = False
                if keep_mask.any():
                    vecs = np.vstack([self._index.reconstruct(i)
                                     for i in range(n) if keep_mask[i]])
                    new_index = faiss.IndexFlatIP(self._dim)
                    new_index.add(vecs)
                    self._index = new_index
                else:
                    self._index = None

            self._questions.pop(idx)
            self._answers.pop(idx)
            self._source_files.pop(idx)
            self._indexed_at.pop(idx)
            if idx < len(self._domains):
                self._domains.pop(idx)
        logger.info(f"语义缓存: 已剔除过期条目 idx={idx} (剩余 {self.size})")
        self._dump()  # 剔除后持久化

    def _try_load_model(self):
        import sentence_transformers
        for model_name in (_PRIMARY_MODEL, _FALLBACK_MODEL):
            try:
                self._model = sentence_transformers.SentenceTransformer(model_name)
                self._dim = self._model.get_embedding_dimension()
                logger.info(f"语义缓存模型加载: {model_name} (dim={self._dim})")
                return
            except Exception as e:
                logger.warning(f"语义缓存模型 {model_name} 加载失败: {e}")
        logger.warning("语义缓存: 所有模型加载失败，缓存降级不生效")

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def _embed(self, text: str) -> Optional[np.ndarray]:
        if self._model is None:
            return None
        try:
            vec = self._model.encode([text])[0].astype(np.float32)
            return self._normalize(vec)
        except Exception as e:
            logger.warning(f"语义缓存 embed 失败: {e}")
            return None

    # ── 搜索（FAISS 一步匹配） ────────────────────

    def search(self, question: str, domain: str = "", source_file: str = "") -> Optional[Tuple[str, float]]:
        """FAISS 内积搜索 → 余弦阈值判定 → 领域过滤 → 源文件新鲜度校验。

        Args:
            question: 用户问题文本。
            domain: 可选领域过滤。非空时只匹配同领域缓存。
            source_file: 可选源文件标识。用于构建文档感知的缓存 key。

        Returns:
            (answer, similarity_score) 或 None
        """
        if self._index is None or self._index.ntotal == 0:
            return None

        # 文档感知：question + source_file 组成复合 key
        search_key = question
        if source_file:
            search_key = question + " |doc:" + str(hash(source_file))

        vec = self._embed(search_key)
        if vec is None:
            return None

        # 搜索 top-k（领域过滤时需要多个候选项）
        k = 5 if domain else 1
        with self._lock:
            scores, ids = self._index.search(vec.reshape(1, -1), k=min(k, self._index.ntotal))

        for i in range(k):
            idx = int(ids[0][i])
            if idx < 0:
                continue
            score = float(scores[0][i])
            if score < self.threshold:
                continue
            if idx >= len(self._answers):
                continue

            # ── 领域过滤 ──
            if domain:
                entry_domain = self._domains[idx] if idx < len(self._domains) else "general"
                if entry_domain != domain:
                    continue

            # ── 源文件新鲜度校验 ──
            if not self._is_entry_fresh(idx):
                logger.info(f"语义缓存: 源文件已变更，剔除过期条目 idx={idx}")
                self._invalidate_entry(idx)
                continue

            logger.info(f"语义缓存命中: cos={score:.4f} idx={idx} domain={domain or 'any'}")
            return (self._answers[idx], score)

        return None

    # ── 写入 ─────────────────────────────────────

    def add(self, question: str, answer: str,
            source_file: str = "", indexed_at: float = 0.0,
            domain: str = "general"):
        """添加缓存条目。

        Args:
            question: 用户问题文本。
            answer: 系统回答文本。
            source_file: | 分隔的源文件路径。
            indexed_at: 源文件最大 mtime。
            domain: 领域分类（finance/contract/law/general/policy/tech）。
        """
        if self._model is None or not answer.strip():
            if self._model is None:
                logger.warning("语义缓存: 模型未加载，跳过写入")
            return

        # 文档感知缓存 key：question + source_file 共同决定 embedding
        cache_key = question
        if source_file:
            cache_key = question + " |doc:" + str(hash(source_file))
        vec = self._embed(cache_key)
        if vec is None:
            return

        source_path = str(source_file) if source_file else ""
        indexed_at = float(indexed_at) if indexed_at else 0.0

        with self._lock:
            # 去重：对已有条目做余弦相似度检查，超过阈值则覆盖
            if self._index is not None and self._index.ntotal > 0:
                scores, ids = self._index.search(vec.reshape(1, -1), k=1)
                if ids[0][0] >= 0 and float(scores[0][0]) >= self.threshold:
                    dup_idx = int(ids[0][0])
                    self._answers[dup_idx] = answer
                    self._source_files[dup_idx] = source_path
                    self._indexed_at[dup_idx] = indexed_at
                    if dup_idx < len(self._domains):
                        self._domains[dup_idx] = domain
                    logger.debug(f"语义缓存: 余弦重复(cos={scores[0][0]:.4f})，覆盖 idx={dup_idx}")
                    return

            self._ensure_index()
            self._index.add(vec.reshape(1, -1))
            self._questions.append(question)
            self._answers.append(answer)
            self._source_files.append(source_path)
            self._indexed_at.append(indexed_at)
            self._domains.append(domain)

            logger.info(f"语义缓存写入: size={self.size} domain={domain} src={source_path[:80]}")
            self._dump()  # 每次写入后立即持久化

    def _ensure_index(self):
        if self._index is None:
            import faiss
            self._index = faiss.IndexFlatIP(self._dim)

    # ── 持久化 ───────────────────────────────────

    def _dump(self):
        """内部持久化：自动在每次变更后调用。"""
        self.dump()

    def dump(self, path: Optional[str] = None):
        """持久化缓存到磁盘。"""
        path = path or str(self._cache_dir)
        os.makedirs(path, exist_ok=True)

        import faiss
        if self._index is not None and self._index.ntotal > 0:
            faiss.write_index(self._index, os.path.join(path, "index.faiss"))
        else:
            # 空缓存：删除旧的持久化文件
            for fname in ("index.faiss", "meta.json"):
                fp = os.path.join(path, fname)
                if os.path.exists(fp):
                    os.remove(fp)
            return

        meta = {
            "questions": self._questions,
            "answers": self._answers,
            "source_files": self._source_files,
            "indexed_at": self._indexed_at,
            "domains": self._domains,
            "fingerprint": self._index_fingerprint,
        }
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        logger.debug(f"语义缓存已持久化: {path} (entries={self._index.ntotal})")

    def _try_load(self):
        index_path = self._cache_dir / "index.faiss"
        meta_path = self._cache_dir / "meta.json"

        if not index_path.exists() or not meta_path.exists():
            return

        import faiss
        try:
            self._index = faiss.read_index(str(index_path))

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self._questions = meta.get("questions", [])
            self._answers = meta.get("answers", [])
            self._source_files = meta.get("source_files", [])
            self._indexed_at = meta.get("indexed_at", [])
            self._domains = meta.get("domains", ["general"] * len(self._questions))
            self._index_fingerprint = meta.get("fingerprint", "")

            logger.info(f"语义缓存已加载: {self._cache_dir} (entries={self._index.ntotal})")
        except Exception as e:
            logger.warning(f"语义缓存加载失败: {e}")

    # ── 缓存列表查询 ────────────────────────────

    def list_entries(self) -> list[dict]:
        """返回所有缓存条目的结构化数据（按 domain 分类）。"""
        entries = []
        for i in range(len(self._questions)):
            answer = self._answers[i] if i < len(self._answers) else ""
            source_files_raw = self._source_files[i] if i < len(self._source_files) else ""
            source_files = [p.strip() for p in source_files_raw.split("|") if p.strip()]
            domain = self._domains[i] if i < len(self._domains) else "general"

            entries.append({
                "id": i,
                "question": self._questions[i][:200],
                "answer_preview": answer[:150] if answer else "",
                "source_files": source_files,
                "indexed_at": self._indexed_at[i] if i < len(self._indexed_at) else 0.0,
                "domain": domain,
            })
        return entries

    def clear(self) -> int:
        """清空全部缓存条目。返回清除的条目数。线程安全。"""
        count = len(self._questions)
        with self._lock:
            self._index = None
            self._questions.clear()
            self._answers.clear()
            self._source_files.clear()
            self._indexed_at.clear()
            self._domains.clear()
        logger.info(f"语义缓存已全部清空 ({count} 条)")
        self._dump()
        return count

    def remove_by_domain(self, domain: str) -> int:
        """删除指定领域的所有缓存条目。返回删除数。"""
        import faiss
        removed = 0
        with self._lock:
            n = len(self._questions)
            keep_mask = [True] * n
            for i in range(n):
                d = self._domains[i] if i < len(self._domains) else "general"
                if d == domain:
                    keep_mask[i] = False
                    removed += 1

            if removed == 0:
                return 0

            # 重建 index
            keep_indices = [i for i in range(n) if keep_mask[i]]
            if keep_indices and self._index is not None and self._index.ntotal > 0:
                vecs = np.vstack([self._index.reconstruct(i) for i in keep_indices])
                new_index = faiss.IndexFlatIP(self._dim)
                new_index.add(vecs)
                self._index = new_index
            else:
                self._index = None

            self._questions = [self._questions[i] for i in keep_indices]
            self._answers = [self._answers[i] for i in keep_indices]
            self._source_files = [self._source_files[i] for i in keep_indices]
            self._indexed_at = [self._indexed_at[i] for i in keep_indices]
            self._domains = [self._domains[i] for i in keep_indices if i < len(self._domains)]

        logger.info(f"语义缓存: 已删除 domain={domain} 的 {removed} 条")
        self._dump()
        return removed

    def stats_by_domain(self) -> dict:
        """按领域统计缓存条目数。"""
        counts = {}
        for i in range(len(self._questions)):
            d = self._domains[i] if i < len(self._domains) else "general"
            counts[d] = counts.get(d, 0) + 1
        return {
            "total": self.size,
            "by_domain": counts,
            "cosine_threshold": self.threshold,
            "model_loaded": self._model is not None,
        }

    def remove_entry(self, idx: int) -> bool:
        """删除指定索引的缓存条目。线程安全。"""
        if idx < 0 or idx >= len(self._questions):
            return False
        self._invalidate_entry(idx)
        return True

    # ── 统计 ─────────────────────────────────────

    def stats(self) -> dict:
        """返回缓存统计信息。"""
        return {
            "size": self.size,
            "cosine_threshold": self.threshold,
            "dimension": self._dim,
            "model_loaded": self._model is not None,
        }


# ── 单例 ──────────────────────────────────────────────

_cache_instance = None
_cache_lock = threading.Lock()


def get_cache() -> SemanticCache:
    global _cache_instance
    if _cache_instance is None:
        with _cache_lock:
            if _cache_instance is None:
                _cache_instance = SemanticCache()
    return _cache_instance
