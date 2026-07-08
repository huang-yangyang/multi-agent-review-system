"""Configuration management for the Multi-Agent System.

Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = os.getenv("LLM_PROVIDER", "openai")

    # OpenAI
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # DeepSeek
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    # Local
    local_model_url: str = os.getenv("LOCAL_MODEL_URL", "http://localhost:11434/v1")
    local_model_name: str = os.getenv("LOCAL_MODEL_NAME", "llama3")

    # 通用 LLM 参数
    lite_model_name: str = os.getenv("LITE_MODEL_NAME", "deepseek-chat")
    llm_timeout: float = 60.0
    llm_max_retries: int = 2
    request_timeout: int = 30
    react_max_iterations: int = 15

    # 模型 Fallback 配置
    fallback_models: list = field(default_factory=lambda: os.getenv("FALLBACK_MODELS", "deepseek-chat").split(","))
    model_cooldown_seconds: int = int(os.getenv("MODEL_COOLDOWN_SECONDS", "30"))
    circuit_failure_threshold: int = int(os.getenv("CIRCUIT_FAILURE_THRESHOLD", "5"))
    circuit_recovery_timeout: int = int(os.getenv("CIRCUIT_RECOVERY_TIMEOUT", "60"))

    @property
    def effective_api_key(self) -> str:
        """Return the API key for the configured provider."""
        mapping = {
            "openai": self.openai_api_key,
            "deepseek": self.deepseek_api_key,
            "local": "not-needed",
        }
        return mapping.get(self.provider, "")

    @property
    def effective_model(self) -> str:
        """Return the model name for the configured provider."""
        mapping = {
            "openai": self.openai_model,
            "deepseek": self.deepseek_model,
            "local": self.local_model_name,
        }
        return mapping.get(self.provider, "gpt-4o")

    @property
    def effective_base_url(self) -> str:
        """Return the base URL for the configured provider."""
        mapping = {
            "openai": self.openai_base_url,
            "deepseek": "https://api.deepseek.com/v1",
            "local": self.local_model_url,
        }
        return mapping.get(self.provider, "https://api.openai.com/v1")


@dataclass
class BDIConfig:
    """BDI cognitive architecture parameters."""
    max_goals: int = int(os.getenv("BDI_MAX_GOALS", "20"))
    belief_ttl: int = int(os.getenv("BDI_BELIEF_TTL", "86400"))
    max_plan_steps: int = int(os.getenv("BDI_MAX_PLAN_STEPS", "15"))


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", "6379"))
    db: int = int(os.getenv("REDIS_DB", "0"))
    password: str = os.getenv("REDIS_PASSWORD", "")


@dataclass
class ChromaConfig:
    """ChromaDB vector store configuration."""
    persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_data"))


@dataclass
class RAGConfig:
    """RAG 文档检索配置。"""
    uploads_dir: str = os.getenv("RAG_UPLOADS_DIR", str(BASE_DIR / "uploads"))
    indexes_dir: str = os.getenv("RAG_INDEXES_DIR", str(BASE_DIR / "indexes"))
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
    reranker_model: str = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
    rrf_k: int = int(os.getenv("RAG_RRF_K", "60"))
    rerank_top_k: int = int(os.getenv("RAG_RERANK_TOP_K", "20"))
    rerank_final_k: int = int(os.getenv("RAG_RERANK_FINAL_K", "10"))


@dataclass
class PathConfig:
    """项目路径配置。"""
    project_root: str = str(BASE_DIR)
    uploads_dir: str = os.getenv("RAG_UPLOADS_DIR", str(BASE_DIR / "uploads"))
    indexes_dir: str = os.getenv("RAG_INDEXES_DIR", str(BASE_DIR / "indexes"))
    checkpoints_db: str = os.getenv("CHECKPOINTS_DB", str(BASE_DIR / "indexes" / "checkpoints.db"))


@dataclass
class APIConfig:
    """FastAPI server configuration."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    workers: int = int(os.getenv("API_WORKERS", "4"))


@dataclass
class AppConfig:
    """Application-wide configuration aggregator."""
    llm: LLMConfig = field(default_factory=LLMConfig)
    bdi: BDIConfig = field(default_factory=BDIConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    chroma: ChromaConfig = field(default_factory=ChromaConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    path: PathConfig = field(default_factory=PathConfig)
    api: APIConfig = field(default_factory=APIConfig)

    # ── Environment ──────────────────────────────────────

    environment: str = field(
        default_factory=lambda: os.getenv("MAS_ENV", "dev")
    )

    @property
    def debug(self) -> bool:
        """True when running in development mode."""
        return self.environment == "dev"

    def __post_init__(self) -> None:
        """Apply environment-specific overrides after dataclass init."""
        if self.environment == "production":
            # Stricter timeouts and disable debug-oriented defaults
            self.llm.llm_timeout = 30.0
            self.llm.llm_max_retries = 1
            self.llm.request_timeout = 15

    @classmethod
    def from_profile(cls, environment: str) -> "AppConfig":
        """Create a pre-configured instance for the given environment.

        Args:
            environment: One of "dev", "staging", "production".

        Returns:
            An AppConfig with environment-specific defaults applied.
        """
        valid_envs = ("dev", "staging", "production")
        if environment not in valid_envs:
            raise ValueError(
                f"Invalid environment '{environment}'. Must be one of {valid_envs}"
            )
        os.environ["MAS_ENV"] = environment
        return cls()


# Singleton config instance
config = AppConfig()
