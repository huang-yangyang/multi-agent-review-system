"""Goal Manager: Goal creation, prioritization, and lifecycle tracking.

Implements the Desire component of BDI architecture.
Manages a priority queue of goals with status tracking.
"""

import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from src.config import config


class GoalStatus(str, Enum):
    """Lifecycle states for a goal."""
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class GoalPriority(int, Enum):
    """Priority levels for goal scheduling."""
    HIGH = 1
    MEDIUM = 2
    LOW = 3


@dataclass
class Goal:
    """A single goal in the BDI architecture.

    Attributes:
        goal_id: Unique identifier.
        description: Human-readable goal description.
        priority: Scheduling priority (HIGH/MEDIUM/LOW).
        status: Current lifecycle status.
        parent_goal_id: Optional parent goal for hierarchical decomposition.
        sub_goals: List of child goal descriptions.
        metadata: Arbitrary structured metadata.
        created_at: Unix timestamp of creation.
        updated_at: Unix timestamp of last update.
    """
    goal_id: str
    description: str
    priority: GoalPriority = GoalPriority.MEDIUM
    status: GoalStatus = GoalStatus.PENDING
    parent_goal_id: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class GoalManager:
    """Manages goals for a BDI agent.

    Features:
    - Goal creation with automatic ID generation
    - Priority-based scheduling
    - Lifecycle tracking (pending -> active -> completed/failed)
    - Hierarchical goal decomposition (parent/child)
    - Capacity limit enforcement
    """

    def __init__(self, max_goals: Optional[int] = None):
        """Initialize the goal manager.

        Args:
            max_goals: Maximum concurrent goals. Defaults to config value.
        """
        self._max_goals = max_goals or config.bdi.max_goals
        self._goals: Dict[str, Goal] = {}
        self._priority_queues: Dict[GoalPriority, List[str]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Goal Creation
    # ------------------------------------------------------------------

    def create_goal(
        self,
        description: str,
        priority: GoalPriority = GoalPriority.MEDIUM,
        parent_goal_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        sub_goals: Optional[List[str]] = None,
    ) -> Optional[Goal]:
        """Create a new goal and add it to the goal pool.

        Args:
            description: Human-readable goal description.
            priority: Scheduling priority.
            parent_goal_id: Optional parent goal ID.
            metadata: Optional structured metadata.
            sub_goals: Optional list of sub-goal descriptions.

        Returns:
            The created Goal, or None if capacity reached.
        """
        if len(self._goals) >= self._max_goals:
            return None

        goal = Goal(
            goal_id=str(uuid.uuid4()),
            description=description,
            priority=priority,
            status=GoalStatus.PENDING,
            parent_goal_id=parent_goal_id,
            sub_goals=sub_goals or [],
            metadata=metadata or {},
        )
        self._goals[goal.goal_id] = goal
        self._priority_queues[priority].append(goal.goal_id)
        return goal

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def get_next_goal(self) -> Optional[Goal]:
        """Get the highest-priority pending goal.

        Returns:
            The next Goal to activate, or None if queue is empty.
        """
        for priority in (GoalPriority.HIGH, GoalPriority.MEDIUM, GoalPriority.LOW):
            queue = self._priority_queues[priority]
            while queue:
                goal_id = queue[0]
                goal = self._goals.get(goal_id)
                if goal and goal.status == GoalStatus.PENDING:
                    return goal
                # Remove stale entries
                queue.pop(0)
        return None

    def activate_goal(self, goal_id: str) -> bool:
        """Mark a goal as active.

        Args:
            goal_id: The goal to activate.

        Returns:
            True if activation succeeded.
        """
        goal = self._goals.get(goal_id)
        if goal is None or goal.status != GoalStatus.PENDING:
            return False
        goal.status = GoalStatus.ACTIVE
        goal.updated_at = time.time()
        return True

    # ------------------------------------------------------------------
    # Status Updates
    # ------------------------------------------------------------------

    def update_goal_status(
        self,
        goal_id: str,
        status: GoalStatus,
    ) -> bool:
        """Update the lifecycle status of a goal.

        Args:
            goal_id: The target goal.
            status: New status to set.

        Returns:
            True if the update succeeded.
        """
        goal = self._goals.get(goal_id)
        if goal is None:
            return False
        goal.status = status
        goal.updated_at = time.time()

        # If completed, also try to complete parent if all children done
        if status == GoalStatus.COMPLETED and goal.parent_goal_id:
            self._check_parent_completion(goal.parent_goal_id)

        return True

    def _check_parent_completion(self, parent_id: str) -> None:
        """Check if all children of a parent goal are completed."""
        parent = self._goals.get(parent_id)
        if parent is None:
            return
        all_done = all(
            self._goals[gid].status == GoalStatus.COMPLETED
            for gid in self._goals
            if self._goals[gid].parent_goal_id == parent_id
        )
        if all_done:
            parent.status = GoalStatus.COMPLETED
            parent.updated_at = time.time()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Retrieve a goal by ID."""
        return self._goals.get(goal_id)

    def list_goals(
        self,
        status: Optional[GoalStatus] = None,
        priority: Optional[GoalPriority] = None,
    ) -> List[Goal]:
        """List goals with optional filters.

        Args:
            status: Filter by lifecycle status.
            priority: Filter by priority level.

        Returns:
            Filtered list of goals.
        """
        goals = self._goals.values()
        if status is not None:
            goals = [g for g in goals if g.status == status]
        if priority is not None:
            goals = [g for g in goals if g.priority == priority]
        return sorted(goals, key=lambda g: (g.priority.value, g.created_at))

    def count_by_status(self) -> Dict[str, int]:
        """Return counts grouped by status."""
        counts: Dict[str, int] = defaultdict(int)
        for g in self._goals.values():
            counts[g.status.value] += 1
        return dict(counts)

    def clear_completed(self) -> int:
        """Remove all completed goals. Returns count of removed goals."""
        completed_ids = [
            gid for gid, g in self._goals.items()
            if g.status == GoalStatus.COMPLETED
        ]
        for gid in completed_ids:
            del self._goals[gid]
        # Rebuild priority queues
        self._priority_queues.clear()
        for g in self._goals.values():
            self._priority_queues[g.priority].append(g.goal_id)
        return len(completed_ids)
