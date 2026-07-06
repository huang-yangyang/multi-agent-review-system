"""Plan Executor: Goal decomposition, step planning, and execution.

Implements the Intention component of BDI architecture.
Breaks goals into ordered step sequences with dependency management.
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from src.config import config


class StepStatus(str, Enum):
    """Execution status for a plan step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step within an execution plan.

    Attributes:
        step_id: Unique identifier.
        description: Human-readable step description.
        status: Current execution status.
        dependencies: Set of step_ids that must complete before this step.
        result: Output data from step execution.
        agent: Optional agent type to execute this step.
        metadata: Arbitrary structured metadata.
    """
    step_id: str
    description: str
    status: StepStatus = StepStatus.PENDING
    dependencies: Set[str] = field(default_factory=set)
    result: Any = None
    agent: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    """An execution plan composed of ordered steps.

    Attributes:
        plan_id: Unique identifier.
        goal_id: Associated goal ID.
        steps: Ordered list of plan steps.
        created_at: Unix timestamp.
    """
    plan_id: str
    goal_id: str
    steps: List[PlanStep] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


class PlanExecutor:
    """Generates and executes plans by decomposing goals into steps.

    Features:
    - Goal -> step decomposition
    - Dependency graph for ordering
    - Step-by-step execution with status tracking
    - Parallel-ready step batches (same-depth steps)
    - Progress reporting
    """

    def __init__(self, max_steps: Optional[int] = None):
        """Initialize the plan executor.

        Args:
            max_steps: Maximum steps per plan. Defaults to config value.
        """
        self._max_steps = max_steps or config.bdi.max_plan_steps
        self._plans: Dict[str, Plan] = {}
        self._step_callbacks: Dict[str, Callable] = {}

    # ------------------------------------------------------------------
    # Plan Generation
    # ------------------------------------------------------------------

    def generate_plan(
        self,
        goal_id: str,
        goal_description: str,
        step_descriptions: Optional[List[str]] = None,
        dependencies: Optional[List[Dict[str, Any]]] = None,
    ) -> Plan:
        """Generate an execution plan for a goal.

        If step_descriptions is provided, uses them directly.
        Otherwise, generates a simple linear plan from the description.

        Args:
            goal_id: The goal this plan serves.
            goal_description: Text description of the goal.
            step_descriptions: Optional explicit step descriptions.
            dependencies: Optional list of {"step_index": int, "depends_on": [int]}.

        Returns:
            A Plan ready for execution.
        """
        plan = Plan(
            plan_id=str(uuid.uuid4()),
            goal_id=goal_id,
        )

        if step_descriptions:
            descriptions = step_descriptions[:self._max_steps]
        else:
            # Simple heuristic decomposition: split by semicolons or numbered items
            parts = [s.strip() for s in goal_description.replace("\n", ";").split(";") if s.strip()]
            descriptions = parts[:self._max_steps] if parts else [goal_description]

        for i, desc in enumerate(descriptions):
            step = PlanStep(
                step_id=f"{plan.plan_id}_step_{i}",
                description=desc,
            )
            plan.steps.append(step)

        # Apply explicit dependencies if provided
        if dependencies:
            for dep_entry in dependencies:
                idx = dep_entry.get("step_index", -1)
                depends_on = dep_entry.get("depends_on", [])
                if 0 <= idx < len(plan.steps):
                    for dep_idx in depends_on:
                        if 0 <= dep_idx < len(plan.steps):
                            plan.steps[idx].dependencies.add(plan.steps[dep_idx].step_id)
        else:
            # Default: linear chain
            for i in range(1, len(plan.steps)):
                plan.steps[i].dependencies.add(plan.steps[i - 1].step_id)

        self._plans[plan.plan_id] = plan
        return plan

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_step(
        self,
        plan_id: str,
        step_id: str,
        executor_fn: Optional[Callable] = None,
        agent: str = "",
    ) -> Optional[Any]:
        """Execute a single step in a plan.

        Args:
            plan_id: The plan containing the step.
            step_id: The step to execute.
            executor_fn: Optional callable to perform the step.
            agent: Optional agent type executing this step.

        Returns:
            Result of the step execution, or None if step can't run.
        """
        plan = self._plans.get(plan_id)
        if plan is None:
            return None

        step = next((s for s in plan.steps if s.step_id == step_id), None)
        if step is None:
            return None

        # Check if dependencies are satisfied
        if not self._dependencies_met(plan, step):
            return None

        step.status = StepStatus.RUNNING
        step.agent = agent

        try:
            if executor_fn:
                result = executor_fn(step)
            else:
                result = {"status": "executed", "description": step.description}
            step.result = result
            step.status = StepStatus.COMPLETED
        except Exception as e:
            step.result = {"error": str(e)}
            step.status = StepStatus.FAILED

        return step.result

    def _dependencies_met(self, plan: Plan, step: PlanStep) -> bool:
        """Check if all dependencies of a step are completed."""
        for dep_id in step.dependencies:
            dep_step = next((s for s in plan.steps if s.step_id == dep_id), None)
            if dep_step is None or dep_step.status != StepStatus.COMPLETED:
                return False
        return True

    def get_ready_steps(self, plan_id: str) -> List[PlanStep]:
        """Get all steps ready for parallel execution (dependencies satisfied).

        Args:
            plan_id: The plan to query.

        Returns:
            List of steps that can be executed now.
        """
        plan = self._plans.get(plan_id)
        if plan is None:
            return []
        return [
            s for s in plan.steps
            if s.status == StepStatus.PENDING and self._dependencies_met(plan, s)
        ]

    # ------------------------------------------------------------------
    # Progress & Queries
    # ------------------------------------------------------------------

    def get_progress(self, plan_id: str) -> Dict[str, Any]:
        """Get execution progress for a plan.

        Args:
            plan_id: The plan to query.

        Returns:
            Dict with total, completed, failed, pending, and percentage.
        """
        plan = self._plans.get(plan_id)
        if plan is None:
            return {"error": "Plan not found"}

        total = len(plan.steps)
        completed = sum(1 for s in plan.steps if s.status == StepStatus.COMPLETED)
        failed = sum(1 for s in plan.steps if s.status == StepStatus.FAILED)
        pending = total - completed - failed
        return {
            "plan_id": plan_id,
            "total_steps": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "progress_pct": round(completed / total * 100, 1) if total > 0 else 0,
        }

    def get_plan(self, plan_id: str) -> Optional[Plan]:
        """Retrieve a plan by ID."""
        return self._plans.get(plan_id)

    def get_step(self, plan_id: str, step_id: str) -> Optional[PlanStep]:
        """Retrieve a specific step."""
        plan = self._plans.get(plan_id)
        if plan is None:
            return None
        return next((s for s in plan.steps if s.step_id == step_id), None)
