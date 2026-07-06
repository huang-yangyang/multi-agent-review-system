"""Enterprise-grade exception hierarchy for the Multi-Agent System.

Design principles:
- Traceable: every exception carries a unique 5-digit error_code (MAS-XXXXX)
- Categorizable: typed hierarchy enables fine-grained catch/retry logic
- Degradable: recoverable flag + RetryableError mixin support graceful fallback
"""

from typing import Any, Dict, Optional


class MASException(Exception):
    """Root exception for all MAS framework errors.

    Attributes:
        error_code: Unique 5-digit code (e.g., "MAS-10001").
        detail: Human-readable context dict for debugging/tracing.
        recoverable: Whether the system can self-heal from this error.
    """

    def __init__(
        self,
        message: str,
        error_code: str = "MAS-10000",
        detail: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
    ):
        super().__init__(message)
        self.error_code = error_code
        self.detail = detail or {}
        self.recoverable = recoverable

    def __str__(self) -> str:
        return f"[{self.error_code}] {super().__str__()}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_code": self.error_code,
            "message": super().__str__(),
            "detail": self.detail,
            "recoverable": self.recoverable,
        }


class RetryableError:
    """Mixin marking an exception as eligible for automatic retry.

    Classes that inherit from both MASException and RetryableError
    signal to callers (circuit breakers, retry decorators) that
    re-attempting the operation after a delay may succeed.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure subclasses default to recoverable=True when using this mixin
        original_init = cls.__init__

        def _patched_init(self, *args, recoverable: bool = True, **kw):
            original_init(self, *args, recoverable=recoverable, **kw)

        cls.__init__ = _patched_init


# ═══════════════════════════════════════════════════════════
# Concrete Exceptions
# ═══════════════════════════════════════════════════════════


class ConfigError(MASException):
    """Configuration missing, invalid, or malformed. Non-recoverable.

    Raised when required environment variables are absent or config
    values fail validation. The service cannot start without fixing
    the underlying configuration.
    """

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="MAS-10001",
            detail=detail,
            recoverable=False,
        )


class AgentExecutionError(MASException):
    """Agent execution failed. May be recoverable depending on cause.

    Wraps failures inside agent.act() / agent.run() so the orchestrator
    can decide whether to retry or route to a fallback agent.
    """

    def __init__(
        self,
        message: str,
        detail: Optional[Dict[str, Any]] = None,
        recoverable: bool = True,
    ):
        super().__init__(
            message=message,
            error_code="MAS-10002",
            detail=detail,
            recoverable=recoverable,
        )


class ToolExecutionError(MASException, RetryableError):
    """Tool invocation failed. Degradable — callers may retry or fall back.

    Wraps exceptions from individual tool calls (search, calculate, etc.).
    """

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="MAS-10003",
            detail=detail,
            recoverable=True,
        )


class SearchError(MASException, RetryableError):
    """Search backend unavailable or returned empty. Degradable.

    Raised when both internal KB and external web search fail.
    Callers should handle gracefully (e.g., respond with partial results).
    """

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="MAS-10004",
            detail=detail,
            recoverable=True,
        )


class CircuitBreakerOpenError(MASException, RetryableError):
    """Circuit breaker is open — requests are being rejected.

    The caller should wait for the recovery timeout to elapse before
    retrying. No further attempts should be made until the circuit
    transitions to HALF_OPEN.
    """

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="MAS-10005",
            detail=detail,
            recoverable=True,
        )


class ModelUnavailableError(MASException, RetryableError):
    """All configured LLM models are unavailable. Recoverable after cooldown.

    Raised by the model fallback chain when every model has been tried
    and all are in cooldown. The caller should wait and retry.
    """

    def __init__(self, message: str, detail: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="MAS-10006",
            detail=detail,
            recoverable=True,
        )
