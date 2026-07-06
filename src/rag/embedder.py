"""嵌入模型。

使用 sentence-transformers 生成文本向量，模型已下载到本地缓存。
默认模型: bge-small-zh（中文优先），回退 all-MiniLM-L6-v2。
"""

import logging
import os
from typing import List

import numpy as np

# ── 强制完全离线模式（用 = 不是 setdefault，覆盖可能已被其他导入设置的值）──
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "5"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "5"

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_FALLBACK_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """文本嵌入器，单例模式。"""

    _instance = None

    def __new__(cls, model_name: str | None = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str | None = None):
        if self._initialized:
            return

        self.model_name = model_name or _DEFAULT_MODEL_NAME
        self._model = None
        self._try_load_model()
        self._initialized = True

    def _try_load_model(self):
        from sentence_transformers import SentenceTransformer

        try:
            logger.info(f"加载嵌入模型(本地): {self.model_name}")
            self._model = SentenceTransformer(
                self.model_name,
                local_files_only=True,
            )
            dim = self._model.get_embedding_dimension()
            logger.info(f"模型加载完成，向量维度: {dim}")

        except Exception as e:
            logger.warning(f"模型 {self.model_name} 加载失败: {e}")
            if self.model_name != _FALLBACK_MODEL_NAME:
                logger.info(f"回退到: {_FALLBACK_MODEL_NAME}")
                self.model_name = _FALLBACK_MODEL_NAME
                try:
                    self._model = SentenceTransformer(
                        self.model_name,
                        local_files_only=True,
                    )
                    logger.info(f"回退模型加载完成")
                except Exception as e2:
                    raise RuntimeError(f"无法加载任何嵌入模型: {e2}")
            else:
                raise RuntimeError(f"无法加载任何嵌入模型: {e}")

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([], dtype=np.float32)

        safe_texts = [t if t.strip() else " " for t in texts]
        return self._model.encode(safe_texts, convert_to_numpy=True, show_progress_bar=False)

    def encode_single(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()
