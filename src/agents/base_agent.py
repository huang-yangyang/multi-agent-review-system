"""Base Agent class implementing the BDI cognitive architecture.

Every specialized agent inherits from this base, which provides:
- Belief: Knowledge base (BeliefBase)
- Desire: Goal-driven behavior (GoalManager)
- Intention: Plan execution (PlanExecutor)

The cognitive loop: perceive → deliberate → plan → act.
"""

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.config import config
from src.core.belief_base import BeliefBase
from src.core.goal_manager import Goal, GoalManager, GoalPriority, GoalStatus
from src.core.message_bus import MessageBus, Message
from src.core.plan_executor import PlanExecutor


class BaseAgent(ABC):
    """Abstract BDI Agent base class.

    Attributes:
        agent_id: Unique identifier for this agent.
        agent_type: Human-readable type label (e.g., "research", "analysis").
        belief_base: Knowledge storage and retrieval.
        goal_manager: Goal lifecycle management.
        plan_executor: Plan generation and step execution.
        message_bus: Inter-agent communication bus (optional).
    """

    def __init__(
        self,
        agent_id: Optional[str] = None,
        agent_type: str = "base",
        message_bus: Optional[MessageBus] = None,
    ):
        """Initialize the BDI agent.

        Args:
            agent_id: Unique agent ID. Auto-generated if not provided.
            agent_type: Type label for this agent.
            message_bus: Shared message bus for communication.
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type
        self.belief_base = BeliefBase()
        self.goal_manager = GoalManager()
        self.plan_executor = PlanExecutor()
        self.message_bus = message_bus

        # Registration with message bus
        if self.message_bus:
            self.message_bus.register_agent(self.agent_id, self._handle_message)

    # ------------------------------------------------------------------
    # BDI Cognitive Loop
    # ------------------------------------------------------------------

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the full BDI cognitive loop.

        perceive → deliberate → plan → act

        Args:
            state: Input state dict with task context.

        Returns:
            Updated state dict with agent outputs.
        """
        # 1. Perceive: observe environment and update beliefs
        await self.perceive(state)

        # 2. Deliberate: choose goals based on beliefs and input
        await self.deliberate(state)

        # 3. Plan: generate execution steps for selected goals
        await self.plan(state)

        # 4. Act: execute steps and produce output
        result = await self.act(state)

        return result

    async def perceive(self, state: Dict[str, Any]) -> None:
        """Perceive the environment and update beliefs.

        Override in subclasses to add domain-specific perception.

        Args:
            state: Current world state.
        """
        question = state.get("question", "") or state.get("raw_input", "")
        if question:
            self.belief_base.add_knowledge(
                content=f"User query: {question}",
                category="perception",
            )

    async def deliberate(self, state: Dict[str, Any]) -> None:
        """Choose goals based on current beliefs and input.

        Override in subclasses for domain-specific goal selection.

        Args:
            state: Current world state.
        """
        question = state.get("question", "") or state.get("raw_input", "")
        self.goal_manager.create_goal(
            description=f"Process user request: {question[:100]}",
            priority=GoalPriority.HIGH,
        )

    async def plan(self, state: Dict[str, Any]) -> None:
        """Generate execution plan for active goals.

        Override in subclasses for domain-specific planning.

        Args:
            state: Current world state.
        """
        next_goal = self.goal_manager.get_next_goal()
        if next_goal:
            self.goal_manager.activate_goal(next_goal.goal_id)
            self.plan_executor.generate_plan(
                goal_id=next_goal.goal_id,
                goal_description=next_goal.description,
            )

    @abstractmethod
    async def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the plan steps and produce output.

        Must be implemented by each specialized agent.

        Args:
            state: Current world state.

        Returns:
            Updated state with agent-specific outputs.
        """
        ...

    # ------------------------------------------------------------------
    # Message Bus Integration
    # ------------------------------------------------------------------

    async def _handle_message(self, message: Message) -> None:
        """Handle incoming messages from the message bus.

        Args:
            message: Received message envelope.
        """
        # Default: log to belief base
        self.belief_base.add_knowledge(
            content=f"[MSG from {message.sender}]: {json.dumps(message.payload, default=str)}",
            category="message",
        )

    async def send_to(self, recipient: str, payload: Dict[str, Any]) -> bool:
        """Send a P2P message to another agent.

        Args:
            recipient: Target agent ID.
            payload: Message content.

        Returns:
            True if delivered.
        """
        if not self.message_bus:
            return False
        return await self.message_bus.send_p2p(self.agent_id, recipient, payload)

    async def request_from(
        self,
        recipient: str,
        payload: Dict[str, Any],
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Send a request and wait for response.

        Args:
            recipient: Target agent ID.
            payload: Request content.
            timeout: Response timeout in seconds.

        Returns:
            Response payload or None.
        """
        if not self.message_bus:
            return None
        return await self.message_bus.request(self.agent_id, recipient, payload, timeout)

    async def broadcast(self, payload: Dict[str, Any]) -> int:
        """Broadcast to all agents.

        Args:
            payload: Message content.

        Returns:
            Number of recipients.
        """
        if not self.message_bus:
            return 0
        return await self.message_bus.broadcast(self.agent_id, payload)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_query_text(self, state: Dict[str, Any]) -> str:
        """Extract query text from state regardless of key name."""
        return (
            state.get("question", "")
            or state.get("raw_input", "")
            or state.get("task_description", "")
        )


# json import for message handler
import json
