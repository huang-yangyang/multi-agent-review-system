"""Agent Registry — type-safe, auto-discovery, lazy-initialization.

Design principles:
- Auto-discovery: scans ``src/agents/`` for ``*_agent.py`` files on init.
- Type-safe: factory enforces that registered classes inherit BaseAgent.
- Lazy: agent classes are imported only when first needed.
- Singleton: one global ``agent_registry`` instance for the whole process.
"""

import importlib
import inspect
import logging
import pkgutil
from pathlib import Path
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


class AgentRegistry:
    """Central registry for Agent types.

    Usage::

        registry = AgentRegistry()
        registry.register("research", ResearchAgent)
        agent = registry.create("research", agent_id="r1")
        types = registry.list_types()
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type] = {}
        self._discovered: bool = False

    # ── Registration ──────────────────────────────────

    def register(self, agent_type: str, agent_class: Type) -> None:
        """Register an Agent class under a type name.

        Args:
            agent_type: Unique type key (e.g., "research", "analysis").
            agent_class: Agent class (must be a BaseAgent subclass).

        Raises:
            TypeError: if agent_class is not a BaseAgent subclass.
        """
        from src.agents.base_agent import BaseAgent

        if not issubclass(agent_class, BaseAgent):
            raise TypeError(
                f"Cannot register '{agent_type}': {agent_class.__name__} is not a BaseAgent subclass."
            )

        if agent_type in self._registry:
            logger.warning(
                f"Agent type '{agent_type}' already registered, overwriting.",
            )
        self._registry[agent_type] = agent_class
        logger.debug(f"Registered agent type: {agent_type} → {agent_class.__name__}")

    # ── Factory ───────────────────────────────────────

    def create(self, agent_type: str, **kwargs: Any) -> Any:
        """Factory: instantiate an agent by type name.

        Args:
            agent_type: Registered type key.
            **kwargs: Forwarded to the agent class constructor.

        Returns:
            A new agent instance.

        Raises:
            KeyError: if agent_type is not registered.
        """
        cls = self.get_class(agent_type)
        return cls(**kwargs)

    # ── Introspection ─────────────────────────────────

    def list_types(self) -> list:
        """Return all registered agent type names."""
        return sorted(self._registry.keys())

    def get_class(self, agent_type: str) -> Type:
        """Get the class registered under agent_type.

        Raises:
            KeyError: if agent_type is not found.
        """
        if agent_type not in self._registry:
            available = ", ".join(sorted(self._registry.keys())) or "(none)"
            raise KeyError(
                f"Unknown agent type '{agent_type}'. Available: {available}"
            )
        return self._registry[agent_type]

    # ── Auto-Discovery ────────────────────────────────

    def discover_agents(self) -> None:
        """Scan ``src/agents/`` for ``*_agent.py`` files and auto-register.

        Each module is imported and inspected for BaseAgent subclasses.
        The agent_type is derived from the module name (everything before
        ``_agent``), e.g., ``research_agent.py`` → ``"research"``.

        Already-registered types are not overwritten.
        """
        if self._discovered:
            logger.debug("Agent discovery already performed, skipping.")
            return

        from src.agents.base_agent import BaseAgent

        agents_dir = Path(__file__).resolve().parent
        logger.info(f"Discovering agents in: {agents_dir}")

        for finder, module_name, is_pkg in pkgutil.iter_modules(
            [str(agents_dir)]
        ):
            if not module_name.endswith("_agent"):
                continue
            if module_name in ("base_agent",):
                continue

            # Derive agent type from module name
            agent_type = module_name.replace("_agent", "")

            if agent_type in self._registry:
                logger.debug(
                    f"Agent type '{agent_type}' already registered, skipping auto-discovery."
                )
                continue

            try:
                module = importlib.import_module(f"src.agents.{module_name}")
            except Exception as exc:
                logger.warning(
                    f"Failed to import agent module '{module_name}': {exc}"
                )
                continue

            # Find BaseAgent subclasses in the module
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if not issubclass(obj, BaseAgent) or obj is BaseAgent:
                    continue
                # Only register classes defined in this module (not imported)
                if obj.__module__ != module.__name__:
                    continue
                self.register(agent_type, obj)
                logger.info(f"Auto-discovered agent: {agent_type} → {obj.__name__}")
                break  # one class per module

        self._discovered = True
        logger.info(
            f"Agent discovery complete. Registered types: {self.list_types()}"
        )


# ── Global Singleton ──────────────────────────────────────

agent_registry = AgentRegistry()
