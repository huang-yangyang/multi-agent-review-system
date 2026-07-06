"""CrossEncoder 精排器。

使用 sentence-transformers CrossEncoder 对候选文档精排。
主模型: BAAI/bge-reranker-base，回退 cross-encoder/ms-marco-MiniLM-L-6-v2。

加载失败时自动降级：rerank() 返回原始候选，不影响检索主流程。
"""

import logging
import os
from typing import List

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "BAAI/bge-reranker-base"
_FALLBACK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """CrossEncoder 精排器，单例+线程安全。

    设计原则：加载失败不阻塞索引初始化。
    available=False 时，rerank() 返回原始候选。
    """

    _instance = None

    def __new__(cls, model_name: str | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str | None = None):
        if self._initialized:
            return

        self.model_name = model_name or _DEFAULT_MODEL
        self._model = None
        self.available = False
        self._try_load_model()
        self._initialized = True

    def _try_load_model(self):
        from sentence_transformers import CrossEncoder

        # 确保离线模式（从 manage.py 或 .env 继承）
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

        for attempt, model in enumerate([self.model_name, _FALLBACK_MODEL]):
            try:
                logger.info(f"加载精排模型: {model}")
                self._model = CrossEncoder(
                    model,
                    max_length=512,
                    local_files_only=True,  # 强制离线
                )
                self.available = True
                self.model_name = model
                logger.info(f"精排模型加载完成: {model}")
                return
            except Exception as e:
                logger.warning(f"精排模型 {model} 加载失败: {e}")

        # 两个模型都失败 → 优雅降级
        logger.warning(
            "所有精排模型加载失败，检索将使用 RRF 分数（无 CrossEncoder 精排）。"
            "如需精排，请在有网络时下载模型文件到 HF_HOME 目录。"
        )
        self.available = False

    def rerank(self, query: str, candidates: List[dict], top_k: int = 10) -> List[dict]:
        if not candidates:
            return []

        if not self.available or self._model is None:
            # 降级：按 RRF 分数排序
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            return candidates[:top_k]

        chunks = [c["chunk"] for c in candidates]
        try:
            pairs = [(query, chunk) for chunk in chunks]
            scores = self._model.predict(pairs)

            for i, c in enumerate(candidates):
                c["rerank_score"] = float(scores[i])

            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            return candidates[:top_k]

        except Exception as e:
            logger.warning(f"精排失败，降级返回原始候选: {e}")
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            return candidates[:top_k]
