"""Application lifecycle management for the Multi-Agent System.

Design principles:
- Self-bootstrapping: loads config, initializes logging, discovers components.
- Graceful shutdown: stops components in reverse order, releases resources.
- Health-aware: validates component health before marking the app ready.

Provides:
- MASApplication: the main application orchestrator.
- LifecycleComponent: protocol for startable/stoppable components.
- ComponentHealth: health status of a single component.
- create_app() / run_app(): factory and convenience entry points.
"""

import asyncio
import logging
import signal
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from src.config import AppConfig, config
from src.core.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


# ── Health ────────────────────────────────────────────────


@dataclass
class ComponentHealth:
    """Health status for a single lifecycle component.

    Attributes:
        status: "healthy", "degraded", or "unhealthy".
        details: Arbitrary key-value pairs for diagnostics.
    """

    status: str  # "healthy" | "degraded" | "unhealthy"
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_healthy(self) -> bool:
        return self.status == "healthy"


# ── Lifecycle Protocol ────────────────────────────────────


@runtime_checkable
class LifecycleComponent(Protocol):
    """Protocol for components that participate in app lifecycle.

    Any object implementing start() / stop() / health() can be
    registered as a lifecycle component.
    """

    async def start(self) -> None:
        """Initialize and start this component."""
        ...

    async def stop(self) -> None:
        """Gracefully shut down this component."""
        ...

    async def health(self) -> ComponentHealth:
        """Return current health status."""
        ...


# ── MASApplication ────────────────────────────────────────


class MASApplication:
    """Main application orchestrator.

    Lifecycle::

        app = MASApplication()
        app.start()   # init components, validate health
        app.run()     # start → wait signal → stop
    """

    def __init__(self, app_config: Optional[AppConfig] = None) -> None:
        """Create a new MAS application instance.

        Args:
            app_config: Application configuration. Uses global singleton
                        if not provided.
        """
        self.config = app_config or config

        # Step 1: Initialize structured logging
        setup_logging()
        logger.info("MASApplication instance created", extra={"component": "app"})

        # Step 2: Component registry
        self._components: Dict[str, LifecycleComponent] = {}
        self._ready: bool = False
        self._shutdown_event: Optional[asyncio.Event] = None

    # ── Component Registration ────────────────────────

    def register_component(self, name: str, component: LifecycleComponent) -> None:
        """Register a lifecycle component.

        Args:
            name: Unique component name (e.g., "agent_registry").
            component: Object conforming to LifecycleComponent protocol.
        """
        if name in self._components:
            logger.warning(
                f"Component '{name}' already registered, overwriting.",
                extra={"component": "app"},
            )
        self._components[name] = component
        logger.info(f"Registered component: {name}", extra={"component": "app"})

    def unregister_component(self, name: str) -> None:
        """Remove a previously registered component."""
        self._components.pop(name, None)

    # ── Lifecycle ─────────────────────────────────────

    async def start(self) -> None:
        """Start all registered components in registration order.

        After all components have started, performs a health check
        and marks the application as ready.
        """
        logger.info("MASApplication starting...", extra={"component": "app"})

        for name, comp in self._components.items():
            logger.info(f"Starting component: {name}", extra={"component": "app"})
            try:
                await comp.start()
            except Exception as exc:
                logger.error(
                    f"Failed to start component '{name}': {exc}",
                    extra={"component": "app"},
                    exc_info=True,
                )
                raise

        # Validate health
        all_healthy = True
        for name, comp in self._components.items():
            try:
                health = await comp.health()
                if not health.is_healthy:
                    all_healthy = False
                    logger.warning(
                        f"Component '{name}' is {health.status}: {health.details}",
                        extra={"component": "app"},
                    )
            except Exception as exc:
                all_healthy = False
                logger.error(
                    f"Health check failed for '{name}': {exc}",
                    extra={"component": "app"},
                )

        self._ready = all_healthy
        if self._ready:
            logger.info("MASApplication ready — all components healthy", extra={"component": "app"})
        else:
            logger.warning("MASApplication started with degraded components", extra={"component": "app"})

    async def stop(self) -> None:
        """Gracefully shut down all components in reverse registration order."""
        logger.info("MASApplication shutting down...", extra={"component": "app"})

        for name in reversed(list(self._components.keys())):
            comp = self._components[name]
            logger.info(f"Stopping component: {name}", extra={"component": "app"})
            try:
                await comp.stop()
            except Exception as exc:
                logger.error(
                    f"Error stopping component '{name}': {exc}",
                    extra={"component": "app"},
                    exc_info=True,
                )

        self._ready = False
        logger.info("MASApplication stopped.", extra={"component": "app"})

    async def run(self) -> None:
        """Full lifecycle: start → wait for signal → stop.

        Listens for SIGINT and SIGTERM to trigger graceful shutdown.
        """
        await self.start()

        # Set up signal handling
        self._shutdown_event = asyncio.Event()
        loop = asyncio.get_running_loop()

        def _handle_signal(sig):
            logger.info(f"Received signal {sig.name}, initiating shutdown...", extra={"component": "app"})
            self._shutdown_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: _handle_signal(s))
            except NotImplementedError:
                # Windows doesn't support add_signal_handler for SIGTERM
                pass

        # Wait for shutdown signal
        await self._shutdown_event.wait()
        await self.stop()


# ── Factory Functions ─────────────────────────────────────


def create_app(app_config: Optional[AppConfig] = None) -> MASApplication:
    """Factory: create and return a MASApplication instance.

    Args:
        app_config: Optional override configuration.

    Returns:
        A new MASApplication ready to start.
    """
    return MASApplication(app_config=app_config)


def run_app(app_config: Optional[AppConfig] = None) -> None:
    """Convenience: create and run the application synchronously.

    Suitable for use in ``if __name__ == "__main__":`` blocks.

    Args:
        app_config: Optional override configuration.
    """
    app = create_app(app_config=app_config)
    asyncio.run(app.run())
