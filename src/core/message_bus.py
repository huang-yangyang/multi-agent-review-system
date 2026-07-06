"""Asynchronous Message Bus for inter-agent communication.

Supports four communication patterns:
- P2P: Direct point-to-point between two agents
- PubSub: Topic-based publish-subscribe
- RequestResponse: Synchronous request with response
- Broadcast: Send to all agents

All communication is asynchronous using asyncio.
"""

import asyncio
import hashlib
import hmac
import json
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class MessageType(str, Enum):
    """Types of messages supported by the bus."""
    P2P = "p2p"
    PUBLISH = "publish"
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"


@dataclass
class Message:
    """A message envelope for inter-agent communication.

    Attributes:
        msg_id: Unique message identifier.
        msg_type: Type of communication pattern.
        sender: Agent ID of the sender.
        recipient: Target agent ID (for P2P/Request).
        topic: Topic string (for PubSub).
        payload: Message body (any serializable data).
        correlation_id: Links request to response.
        timestamp: Unix timestamp of creation.
        signature: HMAC signature for message integrity.
    """
    msg_id: str
    msg_type: MessageType
    sender: str
    recipient: str = ""
    topic: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    correlation_id: str = ""
    timestamp: float = field(default_factory=time.time)
    signature: str = ""

    def to_json(self) -> str:
        """Serialize to JSON string (excluding signature)."""
        return json.dumps({
            "msg_id": self.msg_id,
            "msg_type": self.msg_type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "topic": self.topic,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
        }, ensure_ascii=False)

    @classmethod
    def from_json(cls, data: str) -> "Message":
        """Deserialize from JSON string."""
        d = json.loads(data)
        return cls(
            msg_id=d["msg_id"],
            msg_type=MessageType(d["msg_type"]),
            sender=d["sender"],
            recipient=d.get("recipient", ""),
            topic=d.get("topic", ""),
            payload=d.get("payload", {}),
            correlation_id=d.get("correlation_id", ""),
            timestamp=d.get("timestamp", time.time()),
        )


class MessageBus:
    """Async message bus for agent communication.

    Responsibilities:
    - Route messages between agents using four patterns
    - Maintain subscriptions for pub/sub
    - Handle request-response correlation
    - Optional message signing and verification
    - In-memory delivery with configurable timeouts
    """

    def __init__(self, secret_key: str = ""):
        """Initialize the message bus.

        Args:
            secret_key: Optional HMAC secret for message signing.
        """
        self._secret_key = secret_key
        self._subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._agent_handlers: Dict[str, Callable] = {}
        self._response_timeout: float = 30.0

    # ------------------------------------------------------------------
    # Agent Registration
    # ------------------------------------------------------------------

    def register_agent(self, agent_id: str, handler: Callable) -> None:
        """Register an agent's message handler.

        Args:
            agent_id: Unique agent identifier.
            handler: Async callable that receives Message and returns response.
        """
        self._agent_handlers[agent_id] = handler

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the bus."""
        self._agent_handlers.pop(agent_id, None)
        # Remove from all subscriptions
        for topic in self._subscriptions:
            self._subscriptions[topic].discard(agent_id)

    # ------------------------------------------------------------------
    # PubSub
    # ------------------------------------------------------------------

    def subscribe(self, agent_id: str, topic: str) -> None:
        """Subscribe an agent to a topic.

        Args:
            agent_id: The agent to subscribe.
            topic: The topic string.
        """
        self._subscriptions[topic].add(agent_id)

    def unsubscribe(self, agent_id: str, topic: str) -> None:
        """Unsubscribe an agent from a topic.

        Args:
            agent_id: The agent to unsubscribe.
            topic: The topic string.
        """
        self._subscriptions[topic].discard(agent_id)

    def get_subscribers(self, topic: str) -> Set[str]:
        """Get all agent IDs subscribed to a topic."""
        return self._subscriptions.get(topic, set())

    # ------------------------------------------------------------------
    # Message Creation Helpers
    # ------------------------------------------------------------------

    def _create_message(
        self,
        msg_type: MessageType,
        sender: str,
        payload: Dict[str, Any],
        recipient: str = "",
        topic: str = "",
        correlation_id: str = "",
    ) -> Message:
        """Create and optionally sign a message."""
        msg = Message(
            msg_id=str(uuid.uuid4()),
            msg_type=msg_type,
            sender=sender,
            recipient=recipient,
            topic=topic,
            payload=payload,
            correlation_id=correlation_id,
        )
        if self._secret_key:
            raw = msg.to_json()
            msg.signature = hmac.new(
                self._secret_key.encode(),
                raw.encode(),
                hashlib.sha256,
            ).hexdigest()
        return msg

    def verify_signature(self, msg: Message) -> bool:
        """Verify the HMAC signature of a message.

        Args:
            msg: The message to verify.

        Returns:
            True if signature is valid or signing is disabled.
        """
        if not self._secret_key:
            return True
        if not msg.signature:
            return False
        raw = msg.to_json()
        expected = hmac.new(
            self._secret_key.encode(),
            raw.encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(expected, msg.signature)

    # ------------------------------------------------------------------
    # Communication Patterns
    # ------------------------------------------------------------------

    async def send_p2p(
        self,
        sender: str,
        recipient: str,
        payload: Dict[str, Any],
    ) -> bool:
        """Send a point-to-point message.

        Args:
            sender: Sending agent ID.
            recipient: Target agent ID.
            payload: Message content.

        Returns:
            True if delivered successfully.
        """
        msg = self._create_message(MessageType.P2P, sender, payload, recipient=recipient)
        return await self._deliver(recipient, msg)

    async def publish(
        self,
        sender: str,
        topic: str,
        payload: Dict[str, Any],
    ) -> int:
        """Publish a message to all subscribers of a topic.

        Args:
            sender: Publishing agent ID.
            topic: The topic string.
            payload: Message content.

        Returns:
            Number of agents the message was delivered to.
        """
        msg = self._create_message(MessageType.PUBLISH, sender, payload, topic=topic)
        subscribers = self._subscriptions.get(topic, set())
        delivered = 0
        for agent_id in list(subscribers):
            if await self._deliver(agent_id, msg):
                delivered += 1
        return delivered

    async def request(
        self,
        sender: str,
        recipient: str,
        payload: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Send a request and wait for a response.

        Args:
            sender: Requesting agent ID.
            recipient: Target agent ID.
            payload: Request content.
            timeout: Response timeout in seconds.

        Returns:
            Response payload, or None on timeout/error.
        """
        correlation_id = str(uuid.uuid4())
        msg = self._create_message(
            MessageType.REQUEST,
            sender,
            payload,
            recipient=recipient,
            correlation_id=correlation_id,
        )

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_requests[correlation_id] = future

        delivered = await self._deliver(recipient, msg)
        if not delivered:
            self._pending_requests.pop(correlation_id, None)
            return None

        try:
            effective_timeout = timeout or self._response_timeout
            response_msg = await asyncio.wait_for(future, timeout=effective_timeout)
            return response_msg.payload
        except asyncio.TimeoutError:
            return None
        finally:
            self._pending_requests.pop(correlation_id, None)

    async def send_response(
        self,
        sender: str,
        correlation_id: str,
        recipient: str,
        payload: Dict[str, Any],
    ) -> bool:
        """Send a response to a pending request.

        Args:
            sender: Responding agent ID.
            correlation_id: Matches the original request.
            recipient: The original requesting agent.
            payload: Response content.

        Returns:
            True if the response was correlated and delivered.
        """
        msg = self._create_message(
            MessageType.RESPONSE,
            sender,
            payload,
            recipient=recipient,
            correlation_id=correlation_id,
        )

        # First try to resolve pending request future
        future = self._pending_requests.get(correlation_id)
        if future and not future.done():
            future.set_result(msg)
            return True

        # Fallback: deliver directly to recipient
        return await self._deliver(recipient, msg)

    async def broadcast(
        self,
        sender: str,
        payload: Dict[str, Any],
    ) -> int:
        """Broadcast a message to all registered agents.

        Args:
            sender: Broadcasting agent ID.
            payload: Message content.

        Returns:
            Number of agents delivered to.
        """
        msg = self._create_message(MessageType.BROADCAST, sender, payload)
        delivered = 0
        for agent_id in list(self._agent_handlers.keys()):
            if agent_id != sender and await self._deliver(agent_id, msg):
                delivered += 1
        return delivered

    # ------------------------------------------------------------------
    # Internal Delivery
    # ------------------------------------------------------------------

    async def _deliver(self, agent_id: str, msg: Message) -> bool:
        """Deliver a message to a registered agent handler.

        Args:
            agent_id: Target agent.
            msg: The message to deliver.

        Returns:
            True if handler was found and invoked.
        """
        handler = self._agent_handlers.get(agent_id)
        if handler is None:
            return False
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(msg)
            else:
                handler(msg)
            return True
        except Exception:
            return False
