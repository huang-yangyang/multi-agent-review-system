"""BM25 稀疏检索索引（基于 rank_bm25 + jieba 中文分词）。"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> List[str]:
    return list(jieba.cut(text))


class BM25Index:
    """BM25 稀疏检索索引（jieba 分词 + BM25Okapi）。"""

    def __init__(self):
        self._chunks: List[str] = []
        self._file_names: List[str] = []
        self._doc_ids: List[str] = []
        self._tokenized_corpus: List[List[str]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: List[str], file_names: List[str], doc_ids: List[str]):
        self._chunks = list(chunks)
        self._file_names = list(file_names)
        self._doc_ids = list(doc_ids)

        self._tokenized_corpus = [_tokenize(c) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(f"BM25 索引构建完成: {len(self._chunks)} 条")

    def add(self, chunks: List[str], file_names: List[str], doc_ids: List[str]):
        self._chunks.extend(chunks)
        self._file_names.extend(file_names)
        self._doc_ids.extend(doc_ids)

        new_tokenized = [_tokenize(c) for c in chunks]
        self._tokenized_corpus.extend(new_tokenized)
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        logger.info(f"BM25 增量添加 {len(chunks)} 条，总计 {len(self._chunks)} 条")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, str, float]]:
        if self._bm25 is None:
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in indices:
            if scores[idx] > 0:
                results.append((self._doc_ids[idx], self._chunks[idx], float(scores[idx])))
        return results

    def remove_by_prefix(self, prefix: str):
        keep_indices = [i for i, did in enumerate(self._doc_ids) if not did.startswith(prefix)]
        removed = len(self._doc_ids) - len(keep_indices)
        if removed == 0:
            logger.info(f"BM25 remove_by_prefix: 无匹配 '{prefix}'")
            return

        self._chunks = [self._chunks[i] for i in keep_indices]
        self._file_names = [self._file_names[i] for i in keep_indices]
        self._doc_ids = [self._doc_ids[i] for i in keep_indices]
        self._tokenized_corpus = [self._tokenized_corpus[i] for i in keep_indices]

        if self._tokenized_corpus:
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None

        logger.info(f"BM25 remove_by_prefix: 移除 {removed} 条，剩余 {len(self._chunks)} 条")

    def save(self, path: str):
        data = {
            "chunks": self._chunks,
            "file_names": self._file_names,
            "doc_ids": self._doc_ids,
            "tokenized_corpus": self._tokenized_corpus,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"BM25 索引已保存: {path}")

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)

        inst = cls()
        inst._chunks = data["chunks"]
        inst._file_names = data["file_names"]
        inst._doc_ids = data["doc_ids"]
        inst._tokenized_corpus = data["tokenized_corpus"]
        if inst._tokenized_corpus:
            inst._bm25 = BM25Okapi(inst._tokenized_corpus)
        logger.info(f"BM25 索引已加载: {path} ({len(inst._chunks)} 块)")
        return inst

    def to_dict(self) -> dict:
        return {
            "chunks": self._chunks,
            "file_names": self._file_names,
            "doc_ids": self._doc_ids,
            "tokenized_corpus": self._tokenized_corpus,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BM25Index":
        inst = cls()
        inst._chunks = data["chunks"]
        inst._file_names = data["file_names"]
        inst._doc_ids = data["doc_ids"]
        inst._tokenized_corpus = data["tokenized_corpus"]
        if inst._tokenized_corpus:
            inst._bm25 = BM25Okapi(inst._tokenized_corpus)
        return inst
