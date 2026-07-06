"""RAG 文档上传与混合索引模块。

提供文档解析、分块、双路索引（BM25 + FAISS）及 RRF 混合检索能力。
"""

from .parser import parse_document
from .chunker import chunk_text
from .embedder import Embedder
from .bm25_index import BM25Index
from .dense_index import DenseIndex
from .hybrid_index import rrf_fuse
from .indexer import Indexer, get_indexer

__all__ = [
    "parse_document",
    "chunk_text",
    "Embedder",
    "BM25Index",
    "DenseIndex",
    "rrf_fuse",
    "Indexer",
    "get_indexer",
]
