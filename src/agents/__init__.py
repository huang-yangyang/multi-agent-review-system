"""Specialized BDI Agents."""

from src.agents.registry import agent_registry, AgentRegistry
from src.core.logging_config import get_logger

_logger = get_logger(__name__)

# Auto-discover and register all agent types on import
agent_registry.discover_agents()

_logger.info(f"Agent auto-discovery complete. Available types: {agent_registry.list_types()}")
