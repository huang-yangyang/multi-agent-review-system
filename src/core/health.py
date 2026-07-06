"""System health check — layered, fast-fail, framework-agnostic.

Design principles:
- Layered: checks cascade from critical (LLM) to optional (Tavily).
- Fast-fail: each check has a short timeout; unhealthy checks don't block.
- Independent: no dependency on FastAPI/Django; pure functions + dataclasses.

Usage::

    from src.core.health import check_health
    report = await check_health()
    print(report.status)   # "healthy" | "degraded" | "unhealthy"
"""

import asyncio
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.config import config
from src.core.logging_config import get_logger

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DISK_WARN_THRESHOLD_MB = 100  # Warn when free space drops below 100 MB


# ── Data Types ────────────────────────────────────────────


@dataclass
class HealthCheckResult:
    """Result of a single health check item.

    Attributes:
        status: "healthy", "degraded", or "unhealthy".
        message: Human-readable description.
        details: Optional diagnostic key-value pairs.
    """

    status: str  # "healthy" | "degraded" | "unhealthy"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Aggregated health report.

    Attributes:
        status: Overall status — "healthy" if all checks are healthy,
                "unhealthy" if any critical check fails, "degraded" otherwise.
        checks: Per-component health results keyed by component name.
        timestamp: UTC timestamp of the report generation.
    """

    status: str
    checks: Dict[str, HealthCheckResult]
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ── Check Functions ───────────────────────────────────────


async def _check_llm_availability() -> HealthCheckResult:
    """Check that at least one LLM provider is configured."""
    api_key = config.llm.effective_api_key
    if api_key and api_key != "not-needed":
        return HealthCheckResult(
            status="healthy",
            message=f"LLM provider '{config.llm.provider}' configured.",
            details={"provider": config.llm.provider, "model": config.llm.effective_model},
        )
    # Local model doesn't need API key
    if config.llm.provider == "local":
        return HealthCheckResult(
            status="healthy",
            message="Local LLM provider configured.",
            details={"provider": "local", "url": config.llm.local_model_url},
        )
    return HealthCheckResult(
        status="unhealthy",
        message=f"No valid API key for provider '{config.llm.provider}'.",
        details={"provider": config.llm.provider},
    )


async def _check_rag_indexer() -> HealthCheckResult:
    """Check that the RAG indexer can be initialized."""
    try:
        from src.rag.indexer import get_indexer

        indexer = get_indexer(
            uploads_dir=config.rag.uploads_dir,
            indexes_dir=config.rag.indexes_dir,
        )
        return HealthCheckResult(
            status="healthy",
            message="RAG indexer is available.",
            details={"uploads_dir": config.rag.uploads_dir, "indexes_dir": config.rag.indexes_dir},
        )
    except Exception as exc:
        return HealthCheckResult(
            status="degraded",
            message=f"RAG indexer unavailable: {exc}",
            details={"error": str(exc)},
        )


async def _check_disk_space() -> HealthCheckResult:
    """Check free disk space on the project root volume."""
    try:
        usage = shutil.disk_usage(str(PROJECT_ROOT))
        free_mb = usage.free / (1024 * 1024)
        if free_mb < DISK_WARN_THRESHOLD_MB:
            return HealthCheckResult(
                status="degraded",
                message=f"Low disk space: {free_mb:.1f} MB free.",
                details={"free_mb": round(free_mb, 1), "threshold_mb": DISK_WARN_THRESHOLD_MB},
            )
        return HealthCheckResult(
            status="healthy",
            message=f"Disk space OK: {free_mb:.1f} MB free.",
            details={"free_mb": round(free_mb, 1)},
        )
    except Exception as exc:
        return HealthCheckResult(
            status="degraded",
            message=f"Disk check failed: {exc}",
            details={"error": str(exc)},
        )


async def _check_optional_dependencies() -> HealthCheckResult:
    """Check availability of optional external API keys."""
    details = {}
    missing = []

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key:
        details["tavily"] = "configured"
    else:
        details["tavily"] = "missing"
        missing.append("Tavily")

    baidu_key = os.getenv("BAIDU_API_KEY", "")
    if baidu_key:
        details["baidu"] = "configured"
    else:
        details["baidu"] = "missing"
        missing.append("Baidu")

    if missing:
        return HealthCheckResult(
            status="degraded",
            message=f"Optional APIs not configured: {', '.join(missing)}.",
            details=details,
        )
    return HealthCheckResult(
        status="healthy",
        message="All optional API keys configured.",
        details=details,
    )


# ── Aggregation ───────────────────────────────────────────


async def check_health() -> HealthReport:
    """Run all health checks and aggregate results.

    Checks are executed concurrently where possible. The overall status
    is derived as:
    - "unhealthy" if any critical check (LLM) fails.
    - "degraded" if any non-critical check reports unhealthy or degraded.
    - "healthy" otherwise.

    Returns:
        A HealthReport with per-check results and overall status.
    """
    # Run checks concurrently
    llm, rag, disk, deps = await asyncio.gather(
        _check_llm_availability(),
        _check_rag_indexer(),
        _check_disk_space(),
        _check_optional_dependencies(),
    )

    checks: Dict[str, HealthCheckResult] = {
        "llm": llm,
        "rag_indexer": rag,
        "disk_space": disk,
        "optional_deps": deps,
    }

    # Determine overall status
    statuses = {c.status for c in checks.values()}
    if "unhealthy" in statuses:
        overall = "unhealthy"
    elif "degraded" in statuses:
        overall = "degraded"
    else:
        overall = "healthy"

    report = HealthReport(status=overall, checks=checks)
    logger.info(
        f"Health check complete: {overall}",
        extra={"component": "health", "details": {k: v.status for k, v in checks.items()}},
    )
    return report
