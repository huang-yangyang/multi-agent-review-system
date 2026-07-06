"""Conversation-level Long-Term Memory.

跨会话语义检索：将历史对话摘要向量化存储，新对话自动注入相关背景。

架构：
  - 使用 ChromaDB (复用现有知识库) 存储对话摘要向量
  - 每个会话结束后自动生成摘要并存入
  - 新对话开始时检索相似历史注入上下文
"""

import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import config

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "conversation_memory"
_embeddings_model: Optional[HuggingFaceEmbeddings] = None
_vector_store: Optional[Chroma] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = HuggingFaceEmbeddings(
            model_name=config.rag.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_model


def _get_store() -> Chroma:
    global _vector_store
    if _vector_store is None:
        persist_dir = str(Path(config.rag.indexes_dir) / _COLLECTION_NAME)
        _vector_store = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=_get_embeddings(),
            persist_directory=persist_dir,
        )
    return _vector_store


def _conversation_hash(conv_id: str) -> str:
    return hashlib.md5(conv_id.encode()).hexdigest()[:16]


def save_conversation_memory(
    conv_id: str,
    question: str,
    answer: str,
    summary: str = "",
) -> None:
    """存储一个会话的摘要到长期记忆。

    Args:
        conv_id: 会话 ID
        question: 用户问题
        answer: Agent 回答
        summary: 可选的手动摘要，不传则自动用 question[:200]
    """
    try:
        store = _get_store()
        doc_id = f"mem_{_conversation_hash(conv_id)}_{int(datetime.now().timestamp())}"
        text = summary or question[:300]
        metadata = {
            "conv_id": conv_id,
            "question": question[:500],
            "answer_preview": answer[:300],
            "timestamp": datetime.now().isoformat(),
            "source": "conversation_memory",
        }
        store.add_texts(
            texts=[text],
            metadatas=[metadata],
            ids=[doc_id],
        )
        logger.info(
            f"Long-term memory: saved for conv {conv_id[:12]} ({len(text)} chars)",
            extra={"component": "memory", "conv_id": conv_id[:12]},
        )
    except Exception as e:
        logger.error(
            f"Long-term memory save failed: {e}",
            extra={"component": "memory"},
            exc_info=True,
        )


def retrieve_similar_conversations(
    question: str,
    top_k: int = 3,
    similarity_threshold: float = 0.5,
) -> List[Dict[str, str]]:
    """检索与当前问题相似的历史对话。

    Args:
        question: 当前用户问题
        top_k: 返回最大条数
        similarity_threshold: 相似度阈值 (0~1, 越低越宽松)

    Returns:
        相关历史对话列表，每项含 question / answer_preview / conv_id
    """
    try:
        store = _get_store()
        results = store.similarity_search_with_score(
            question,
            k=top_k,
        )

        if not results:
            return []

        similar: List[Dict[str, str]] = []
        for doc, score in results:
            if score > similarity_threshold:
                continue
            similar.append({
                "question": doc.metadata.get("question", "")[:300],
                "answer_preview": doc.metadata.get("answer_preview", "")[:300],
                "conv_id": doc.metadata.get("conv_id", "")[:12],
                "similarity": f"{1 - score:.3f}",
            })

        if similar:
            logger.info(
                f"Long-term memory: found {len(similar)} related conversations",
                extra={"component": "memory", "count": len(similar)},
            )

        return similar
    except Exception as e:
        logger.error(
            f"Long-term memory retrieval failed: {e}",
            extra={"component": "memory"},
            exc_info=True,
        )
        return []


def build_long_term_context(question: str, max_results: int = 2) -> str:
    """构建可注入 LLM 的长期记忆上下文文本。

    用于 _build_initial_state 时将相关历史注入 state。

    Args:
        question: 当前用户问题
        max_results: 最多注入几条历史

    Returns:
        格式化的上下文文本，无匹配时返回空字符串
    """
    similar = retrieve_similar_conversations(question, top_k=max_results)

    if not similar:
        return ""

    lines = ["【历史相关对话】"]
    for i, item in enumerate(similar, 1):
        lines.append(
            f"--- 历史对话 {i} ---\n"
            f"用户问题: {item['question']}\n"
            f"AI 回答摘要: {item['answer_preview']}"
        )

    return "\n".join(lines)


def forget_conversation(conv_id: str) -> bool:
    """删除指定会话的长期记忆（按 conv_id 元数据过滤删除）。"""
    try:
        store = _get_store()
        results = store.get(
            where={"conv_id": conv_id},
        )
        ids = results.get("ids", [])
        if ids:
            store.delete(ids=ids)
            logger.info(
                f"Long-term memory: deleted {len(ids)} entries for conv {conv_id[:12]}",
                extra={"component": "memory", "conv_id": conv_id[:12]},
            )
            return True
        return False
    except Exception as e:
        logger.error(
            f"Long-term memory forget failed: {e}",
            extra={"component": "memory"},
            exc_info=True,
        )
        return False
