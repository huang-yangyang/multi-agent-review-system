"""熔断器 + 模型 Fallback + 指数退避重试 模块。

提供：
  1. CircuitBreaker — 状态机 CLOSED → OPEN → HALF_OPEN → CLOSED
  2. ModelFallback — 多模型优先级降级，失败自动切换
  3. with_retry — 异步指数退避重试装饰器
  4. 全局实例 — llm_circuit / baidu_circuit / tavily_circuit / model_fallback

设计原则：
  - 熔断器是可选增强，降级优于报错
  - 线程安全，支持 async 和 sync 两种调用上下文
"""

import asyncio
import enum
import logging
import threading
import time
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

logger = logging.getLogger(__name__)


class CircuitState(enum.Enum):
    """熔断器三态。"""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenError(Exception):
    """熔断器开启时抛出。"""
    pass


class CircuitBreaker:
    """熔断器：保护外部依赖免受级联故障。"""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        name: str = "",
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.name = name or self.__class__.__name__
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        return self._state

    def is_open(self) -> bool:
        return self._state == CircuitState.OPEN

    def _should_attempt_recovery(self) -> bool:
        return time.time() - self._last_failure_time >= self.recovery_timeout

    def _before_call(self):
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._transition_to(CircuitState.HALF_OPEN)
                    logger.info(f"[{self.name}] 状态转换: OPEN → HALF_OPEN")
                else:
                    raise CircuitBreakerOpenError(
                        f"[{self.name}] 熔断器开启，拒绝调用"
                    )
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpenError(
                        f"[{self.name}] 半开状态超过最大探测调用数"
                    )
                self._half_open_calls += 1

    def _on_success(self):
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    self._transition_to(CircuitState.CLOSED)
                    logger.info(f"[{self.name}] 状态转换: HALF_OPEN → CLOSED")
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0

    def _on_failure(self):
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)
                logger.warning(f"[{self.name}] 状态转换: HALF_OPEN → OPEN")
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
                    logger.warning(
                        f"[{self.name}] 状态转换: CLOSED → OPEN "
                        f"(连续失败 {self._failure_count}/{self.failure_threshold})"
                    )

    def _transition_to(self, new_state: CircuitState):
        old_state = self._state
        self._state = new_state
        if new_state in (CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN):
            self._failure_count = 0
            self._success_count = 0
            self._half_open_calls = 0
        logger.debug(f"[{self.name}] {old_state.value} → {new_state.value}")

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        self._before_call()
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerOpenError:
            raise
        except Exception:
            self._on_failure()
            raise

    async def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        self._before_call()
        try:
            result = await asyncio.to_thread(func, *args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerOpenError:
            raise
        except Exception:
            self._on_failure()
            raise

    def call_blocking(self, func: Callable, *args, **kwargs) -> Any:
        self._before_call()
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except CircuitBreakerOpenError:
            raise
        except Exception:
            self._on_failure()
            raise


class ModelFallback:
    """多模型优先级降级管理。"""

    def __init__(
        self,
        models: Optional[list] = None,
        cooldown_seconds: float = 30.0,
    ):
        self.models = models or ["deepseek-chat", "deepseek-reasoner", "qwen-plus", "gpt-4o-mini"]
        self.cooldown_seconds = cooldown_seconds
        self._cooldown_until: dict[str, float] = {}
        self._healthy_model: Optional[str] = None
        self._lock = threading.Lock()

    def _is_cooled_down(self, model: str) -> bool:
        deadline = self._cooldown_until.get(model, 0)
        return time.time() < deadline

    def _mark_failed(self, model: str):
        with self._lock:
            self._cooldown_until[model] = time.time() + self.cooldown_seconds
            if self._healthy_model == model:
                self._healthy_model = None
            logger.warning(f"模型 {model} 进入冷却 ({self.cooldown_seconds}s)")

    def _mark_healthy(self, model: str):
        with self._lock:
            self._healthy_model = model
            self._cooldown_until.pop(model, None)

    def _get_ordered_models(self) -> list[str]:
        result = []
        if self._healthy_model and not self._is_cooled_down(self._healthy_model):
            result.append(self._healthy_model)
        for m in self.models:
            if m not in result and not self._is_cooled_down(m):
                result.append(m)
        if not result and self.models:
            logger.warning("所有模型都在冷却中，强制尝试优先级最高的模型")
            result.append(self.models[0])
        return result

    async def invoke_with_fallback(
        self,
        llm_factory: Callable,
        messages: list,
        circuit: Optional[CircuitBreaker] = None,
        **kwargs,
    ) -> tuple:
        models_to_try = self._get_ordered_models()
        last_error: Optional[Exception] = None
        for model in models_to_try:
            if self._is_cooled_down(model):
                logger.debug(f"模型 {model} 冷却中，跳过")
                continue
            try:
                llm = llm_factory(model)
                async def _invoke():
                    return await llm.ainvoke(messages, **kwargs)
                if circuit:
                    response = await circuit.call(_invoke)
                else:
                    response = await _invoke()
                self._mark_healthy(model)
                logger.info(f"模型 {model} 调用成功")
                return response, model
            except CircuitBreakerOpenError as e:
                logger.warning(f"熔断器开启，跳过模型 {model}")
                last_error = e
                continue
            except Exception as e:
                self._mark_failed(model)
                logger.warning(f"模型 {model} 调用失败: {e}")
                last_error = e
                continue
        raise Exception(f"所有模型均不可用，最后错误: {last_error}")


# 全局实例
llm_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=60, half_open_max_calls=3, name="LLM")
baidu_circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=30, half_open_max_calls=2, name="BaiduSearch")
tavily_circuit = CircuitBreaker(failure_threshold=3, recovery_timeout=30, half_open_max_calls=2, name="TavilySearch")
model_fallback = ModelFallback(
    models=["deepseek-chat", "deepseek-reasoner", "qwen-plus", "gpt-4o-mini"],
    cooldown_seconds=30,
)


# ── 指数退避重试 ──────────────────────────────────────


F = TypeVar("F", bound=Callable[..., Any])


def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """异步指数退避重试装饰器。

    Args:
        max_retries: 最大重试次数 (默认 3, 即最多总执行 4 次)
        base_delay: 初始延迟秒数 (默认 1.0)
        max_delay: 最大延迟秒数 (默认 30.0)
        exceptions: 捕获的异常类型元组 (默认所有 Exception)

    用法::

        @retry(max_retries=2, base_delay=0.5)
        async def my_llm_call(...):
            return await llm.ainvoke(...)

    重试延迟: base_delay * 2^attempt, 上限 max_delay
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt == max_retries:
                        break
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} for {func.__name__}: "
                        f"{e} — 等待 {delay:.1f}s"
                    )
                    await asyncio.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper  # type: ignore[return-value]
    return decorator
