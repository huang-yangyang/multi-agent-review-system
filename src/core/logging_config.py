"""Structured logging configuration for the Multi-Agent System.

Design principles:
- JSON format for machine-readability (production); text for dev.
- Daily log rotation via TimedRotatingFileHandler.
- Environment-aware: LOG_LEVEL / LOG_FORMAT / LOG_FILE env vars.
- Zero dependency on Django/Flask — pure stdlib + optional json.
"""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ── Constants ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"
_logging_initialized = False


# ── JSON Formatter ────────────────────────────────────────


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
        }
        # Inject structured extras
        for key in ("component", "agent_id", "trace_id"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        # Inject all custom extras (set via extra={...})
        for key, value in record.__dict__.items():
            if key not in {
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "levelname", "levelno", "lineno",
                "module", "msecs", "message", "msg", "name", "pathname",
                "process", "processName", "relativeCreated", "stack_info",
                "thread", "threadName",
            }:
                if key not in log_entry:
                    log_entry[key] = value
        # Exception info
        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable coloured log format for development."""

    GREY = "\x1b[90m"
    RESET = "\x1b[0m"

    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extras = []
        for key in ("component", "agent_id", "trace_id"):
            value = getattr(record, key, None)
            if value:
                extras.append(f"{key}={value}")
        if extras:
            base += f"  {self.GREY}[{', '.join(extras)}]{self.RESET}"
        return base


# ── Public API ────────────────────────────────────────────


def setup_logging(
    level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Initialize structured logging for the application.

    Called once at app startup. Idempotent — subsequent calls are no-ops.

    Args:
        level: Log level override (defaults to LOG_LEVEL env or "INFO").
        log_format: "json" or "text" (defaults to LOG_FORMAT env or "json").
        log_file: Log file path override (defaults to LOG_FILE env or
                  ``logs/mas_YYYY-MM-DD.log`` under PROJECT_ROOT).
    """
    global _logging_initialized
    if _logging_initialized:
        return

    # Resolve level
    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, level, logging.INFO)

    # Resolve format
    fmt = (log_format or os.getenv("LOG_FORMAT", "json")).lower()
    if fmt == "text":
        formatter = TextFormatter()
    else:
        formatter = JSONFormatter()

    # Resolve log file
    today = datetime.now().strftime("%Y-%m-%d")
    if log_file is None:
        log_file = os.getenv("LOG_FILE", "")
    if not log_file:
        DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = str(DEFAULT_LOG_DIR / f"mas_{today}.log")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear any existing handlers (idempotent)
    root_logger.handlers.clear()

    # File handler with daily rotation
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_file,
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Also log to stderr if LOG_FORMAT is text (dev mode) or LOG_FILE is empty
    if fmt == "text" or not log_file:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(
            TextFormatter() if fmt == "text" else JSONFormatter()
        )
        root_logger.addHandler(console_handler)

    _logging_initialized = True
    root_logger.info(
        "Logging initialized", extra={"component": "logging", "level": level, "format": fmt}
    )


def get_logger(name: str) -> logging.Logger:
    """Convenience: return a logger for the given module name.

    Usage::

        from src.core.logging_config import get_logger
        logger = get_logger(__name__)
        logger.info("Something happened", extra={"agent_id": "researcher-1"})
    """
    return logging.getLogger(name)
