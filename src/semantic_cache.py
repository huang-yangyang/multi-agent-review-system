"""语义缓存 — embedding 向量 + FAISS 余弦相似度，业界主流方案。

流程：问题 → embedding → FAISS IndexFlatIP.search() → 余弦阈值命中
附加：源文件新鲜度校验（命中后比对 mtime，过期自动剔除）

权限模型（共享 + 安全）：
- 缓存按「问题语义」共享：相同问题，任何用户都能命中他人已缓存的答案（B 问过，A 直接用）。
- 每条缓存记录「引用了哪些文档 (referenced_docs)」：
  - 纯网络/通用答案（无引用文档）→ 公开，所有人可用。
  - 引用了文档 → 只有「对这些文档都有访问权限」的用户才能命中，
    从而管理员专属文档的缓存不会被普通用户读到（防泄漏）。
- 每条缓存记录「哪些用户问过 (user_names)」，用于区分与审计。
"""

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .config import config

logger = logging.getLogger(__name__)

# ── 常量 ──────────────────────────────────────────────
_DEFAULT_THRESHOLD = float(os.environ.get("SEMANTIC_CACHE_THRESHOLD", "0.75"))
_PRIMARY_MODEL = "BAAI/bge-small-zh-v1.5"       # 512 维，中文优化，与 RAG 统一
_FALLBACK_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # 384 维，回退


# ── 缓存类 ────────────────────────────────────────────

class SemanticCache:
    """语义缓存 — embedding 向量 + FAISS 余弦匹配（跨用户共享 + 权限安全）。

    写入时：embedding(question) → 归一化 → FAISS IndexFlatIP.add()
    查询时：embedding(question) → 归一化 → FAISS IndexFlatIP.search(k) →
            阈值判定 → 领域过滤 → 权限过滤（引用文档可访问性）→ 源文件新鲜度校验。
    """

    def __init__(self, dimension: int = 512, threshold: float = _DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._dim = dimension
        self._index = None
        self._questions: List[str] = []
        self._answers: List[str] = []
        self._source_files: List[str] = []       # | 分隔的多文件路径（新鲜度校验 + 按文档失效）
        self._indexed_at: List[float] = []        # 最大 mtime
        self._domains: List[str] = []              # 领域分类: finance/contract/law/general
        self._referenced_docs: List[List[str]] = []  # 每条缓存引用的文档 basename 列表（权限过滤用）
        self._user_names: List[List[str]] = []       # 每条缓存贡献过的用户（区分/审计用）
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
        """计算 RAG 索引指纹，用于在索引变更时自动清空语义缓存。

        D 方案下稠密向量由 Qdrant 管理（qdrant_storage），所有索引变更都会同步
        反映到 indexes.db（files / chunks 表），因此以 indexes.db 的 mtime 作为
        索引变更的唯一真相源，避免引用已被移除的 faiss.index / faiss_meta.json。
        """
        indexes_dir = Path(config.rag.indexes_dir)
        fp = indexes_dir / "indexes.db"
        if not fp.exists():
            return ""
        raw = "indexes.db:" + str(fp.stat().st_mtime)
        return hashlib.md5(raw.encode()).hexdigest()

    def _is_fingerprint_valid(self) -> bool:
        """索引指纹变更时自动清空语义缓存，避免返回过期答案。"""
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
                self._referenced_docs.clear()
                self._user_names.clear()
            self._index_fingerprint = current
            self._dump()
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
            if idx < len(self._referenced_docs):
                self._referenced_docs.pop(idx)
            if idx < len(self._user_names):
                self._user_names.pop(idx)
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

    # ── 权限过滤 ──────────────────────────────────

    def _entry_accessible_to(self, referenced_docs: List[str], reader_user: str) -> bool:
        """判断某条缓存（引用了 referenced_docs）是否对 reader_user 可见。

        - 无引用文档（纯网络/通用答案）→ 公开，所有人可见。
        - reader_user 为空 → 视为公开（调用方未传入用户时不误杀）。
        - 管理员（get_user_accessible_doc_paths 返回 None）→ 可见全部。
        - 否则要求 referenced_docs 全部在该用户可访问文档集合中。
        """
        if not referenced_docs:
            return True
        if not reader_user:
            return True
        try:
            from src.permissions import get_user_accessible_doc_paths
            accessible = get_user_accessible_doc_paths(reader_user)
            if accessible is None:
                return True  # 管理员看全部
            allowed = {os.path.basename(p) for p in accessible}
            needed = {os.path.basename(d) for d in referenced_docs}
            return needed.issubset(allowed)
        except Exception:
            return True  # 降级：允许

    # ── 搜索（FAISS 一步匹配，跨用户共享） ─────────

    def search(
        self,
        question: str,
        domain: str = "",
        reader_user: str = "",
    ) -> Optional[Tuple[str, float]]:
        """FAISS 内积搜索 → 余弦阈值 → 领域过滤 → 权限过滤 → 源文件新鲜度校验。

        缓存跨用户共享：相同问题，任何用户都能命中他人已缓存的答案。
        仅当缓存答案「引用了当前用户无权访问的文档」时才跳过（防泄漏）。

        Args:
            question: 用户问题文本（embedding key 仅由问题本身构成）。
            domain: 可选领域过滤。非空时只匹配同领域缓存。
            reader_user: 读取者用户名，用于权限过滤。

        Returns:
            (answer, similarity_score) 或 None
        """
        self._is_fingerprint_valid()

        if self._index is None or self._index.ntotal == 0:
            return None

        # embedding key 仅由问题本身构成 —— 保证跨用户共享：
        # 同一问题无论谁问、检索到哪些文档，都映射到同一个向量。
        vec = self._embed(question)
        if vec is None:
            return None

        k = 5  # 多取候选项，便于在权限/领域过滤后仍能命中
        with self._lock:
            scores, ids = self._index.search(vec.reshape(1, -1), k=min(k, self._index.ntotal))

        for i in range(len(ids[0])):
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

            # ── 权限过滤：引用了当前用户无权访问的文档则跳过 ──
            entry_docs = self._referenced_docs[idx] if idx < len(self._referenced_docs) else []
            if not self._entry_accessible_to(entry_docs, reader_user):
                continue

            # ── 源文件新鲜度校验 ──
            if not self._is_entry_fresh(idx):
                logger.info(f"语义缓存: 源文件已变更，剔除过期条目 idx={idx}")
                self._invalidate_entry(idx)
                continue

            logger.info(
                f"语义缓存命中: cos={score:.4f} idx={idx} domain={domain or 'any'} "
                f"reader={reader_user or 'public'}"
            )
            return (self._answers[idx], score)

        return None

    # ── 写入 ─────────────────────────────────────

    def add(
        self,
        question: str,
        answer: str,
        source_file: str = "",
        indexed_at: float = 0.0,
        domain: str = "general",
        contributor_user: str = "",
        referenced_docs: Optional[List[str]] = None,
    ):
        """添加（或更新）缓存条目。

        缓存跨用户共享：
        - embedding key 仅由问题构成，不绑定用户或具体文档。
        - 同一问题若已存在缓存：
            * 且当前贡献者有权访问该旧条目 → 覆盖更新（合并贡献用户、取文档并集）。
            * 且旧条目引用了当前贡献者无权访问的文档（如管理员专属）→
              不覆盖，另存一条新条目（避免普通用户覆盖/污染管理员缓存）。
        - 记录 contributor_user 到 user_names，用于区分「谁问过」。

        Args:
            question: 用户问题文本。
            answer: 系统回答文本。
            source_file: | 分隔的源文件路径（新鲜度校验 + 按文档失效）。
            indexed_at: 源文件最大 mtime。
            domain: 领域分类。
            contributor_user: 提问/贡献该答案的用户名。
            referenced_docs: 答案引用到的文档 basename 列表（权限过滤依据）。
                             为 None 时从 source_file 推导。
        """
        self._is_fingerprint_valid()

        if self._model is None or not answer.strip():
            if self._model is None:
                logger.warning("语义缓存: 模型未加载，跳过写入")
            return

        vec = self._embed(question)
        if vec is None:
            return

        if referenced_docs is None:
            referenced_docs = [
                os.path.basename(p.strip())
                for p in source_file.split("|") if p.strip()
            ]
        referenced_docs = list(dict.fromkeys(referenced_docs))  # 去重保序

        source_path = str(source_file) if source_file else ""
        indexed_at = float(indexed_at) if indexed_at else 0.0

        with self._lock:
            # 去重：对已有条目做余弦相似度检查
            if self._index is not None and self._index.ntotal > 0:
                scores, ids = self._index.search(vec.reshape(1, -1), k=1)
                if ids[0][0] >= 0 and float(scores[0][0]) >= self.threshold:
                    dup_idx = int(ids[0][0])
                    dup_docs = (
                        self._referenced_docs[dup_idx]
                        if dup_idx < len(self._referenced_docs) else []
                    )
                    # 仅当贡献者能访问旧条目时才覆盖（防止普通用户覆盖管理员专属缓存）
                    if self._entry_accessible_to(dup_docs, contributor_user):
                        self._answers[dup_idx] = answer
                        self._source_files[dup_idx] = source_path
                        self._indexed_at[dup_idx] = indexed_at
                        if dup_idx < len(self._domains):
                            self._domains[dup_idx] = domain
                        if dup_idx < len(self._referenced_docs):
                            merged = list(dict.fromkeys(dup_docs + referenced_docs))
                            self._referenced_docs[dup_idx] = merged
                        if dup_idx < len(self._user_names):
                            if contributor_user and contributor_user not in self._user_names[dup_idx]:
                                self._user_names[dup_idx].append(contributor_user)
                        logger.debug(
                            f"语义缓存: 余弦重复(cos={scores[0][0]:.4f})，覆盖更新 idx={dup_idx}"
                        )
                        self._dump()
                        return
                    # 否则：旧条目更受限，另存新条目（下方 append）

            self._ensure_index()
            self._index.add(vec.reshape(1, -1))
            self._questions.append(question)
            self._answers.append(answer)
            self._source_files.append(source_path)
            self._indexed_at.append(indexed_at)
            self._domains.append(domain)
            self._referenced_docs.append(referenced_docs)
            self._user_names.append([contributor_user] if contributor_user else [])

            logger.info(
                f"语义缓存写入: size={self.size} domain={domain} "
                f"contributor={contributor_user or '?'} docs={referenced_docs[:3]}"
            )
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
            "referenced_docs": self._referenced_docs,
            "user_names": self._user_names,
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
            self._referenced_docs = meta.get(
                "referenced_docs", [[] for _ in self._questions]
            )
            self._user_names = meta.get(
                "user_names", [[] for _ in self._questions]
            )
            self._index_fingerprint = meta.get("fingerprint", "")

            logger.info(f"语义缓存已加载: {self._cache_dir} (entries={self._index.ntotal})")
        except Exception as e:
            logger.warning(f"语义缓存加载失败: {e}")

    # ── 缓存列表查询 ────────────────────────────

    def list_entries(self) -> List[dict]:
        """返回所有缓存条目的结构化数据（含贡献用户与引用文档）。"""
        entries = []
        for i in range(len(self._questions)):
            answer = self._answers[i] if i < len(self._answers) else ""
            source_files_raw = self._source_files[i] if i < len(self._source_files) else ""
            source_files = [p.strip() for p in source_files_raw.split("|") if p.strip()]
            domain = self._domains[i] if i < len(self._domains) else "general"
            referenced = self._referenced_docs[i] if i < len(self._referenced_docs) else []
            users = self._user_names[i] if i < len(self._user_names) else []

            entries.append({
                "id": i,
                "question": self._questions[i][:200],
                "answer_preview": answer[:150] if answer else "",
                "source_files": source_files,
                "referenced_docs": referenced,
                "contributors": users,
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
            self._referenced_docs.clear()
            self._user_names.clear()
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
            self._referenced_docs = [
                self._referenced_docs[i] for i in keep_indices if i < len(self._referenced_docs)
            ]
            self._user_names = [
                self._user_names[i] for i in keep_indices if i < len(self._user_names)
            ]

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

    def remove_by_doc_path(self, doc_path: str) -> int:
        """删除所有引用了指定文档路径的缓存条目。

        在文档权限变更（如 visibility 从 all 改为 admin）时调用，
        确保旧缓存不会返回变更前的答案。支持多文件缓存条目（| 分隔）。

        Args:
            doc_path: 文档的完整路径或文件名，两者都会匹配。

        Returns:
            被删除的条目数。
        """
        import faiss
        doc_name = os.path.basename(doc_path)
        removed = 0
        with self._lock:
            n = len(self._questions)
            keep_mask = [True] * n
            for i in range(n):
                # 1) 检查 source_files（路径匹配）
                hit = False
                if i < len(self._source_files):
                    sources = [p.strip() for p in self._source_files[i].split("|") if p.strip()]
                    for sp in sources:
                        if (doc_path in sp or doc_name in sp
                                or sp in doc_path or doc_name in os.path.basename(sp)):
                            hit = True
                            break
                # 2) 检查 referenced_docs（basename 匹配）
                if not hit and i < len(self._referenced_docs):
                    for rd in self._referenced_docs[i]:
                        if doc_name in rd or rd in doc_name:
                            hit = True
                            break
                if hit:
                    keep_mask[i] = False
                    removed += 1

            if removed == 0:
                return 0

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
            self._referenced_docs = [
                self._referenced_docs[i] for i in keep_indices if i < len(self._referenced_docs)
            ]
            self._user_names = [
                self._user_names[i] for i in keep_indices if i < len(self._user_names)
            ]

        logger.info(f"语义缓存: 文档权限变更，已删除引用 {doc_name} 的 {removed} 条缓存")
        self._dump()
        return removed

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
