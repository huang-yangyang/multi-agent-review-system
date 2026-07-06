"""Tests for the multi-agent system core components.

Validates:
1. BDI core components (BeliefBase, GoalManager, PlanExecutor)
2. Agent instantiation and basic operation
3. LangGraph orchestrator compilation
4. Message bus communication patterns
"""

import asyncio
import pytest

from src.core.belief_base import BeliefBase
from src.core.goal_manager import GoalManager, GoalPriority, GoalStatus
from src.core.plan_executor import PlanExecutor, StepStatus
from src.core.message_bus import MessageBus, MessageType
from src.core.state_manager import StateManager
from src.agents.base_agent import BaseAgent
from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.workflows.orchestrator import build_graph


# ------------------------------------------------------------------
# BeliefBase Tests
# ------------------------------------------------------------------

class TestBeliefBase:
    """Test the Belief component of BDI architecture."""

    def test_add_and_query(self):
        bb = BeliefBase()
        bb.add_knowledge("Python is a programming language", category="facts")
        bb.add_knowledge("Machine learning uses neural networks", category="ai")

        results = bb.query("Python")
        assert len(results) > 0
        assert any("Python" in r["content"] for r in results)

    def test_get_context(self):
        bb = BeliefBase()
        bb.add_knowledge("Apples are fruits", category="food")
        bb.add_knowledge("Bananas are yellow", category="food")

        ctx = bb.get_context("fruit")
        assert "Apples" in ctx or "Bananas" in ctx

    def test_get_by_id(self):
        bb = BeliefBase()
        bid = bb.add_knowledge("Test knowledge")
        entry = bb.get_by_id(bid)
        assert entry is not None
        assert entry["content"] == "Test knowledge"

    def test_list_by_category(self):
        bb = BeliefBase()
        # Clean any left-over test data from previous runs (persistent ChromaDB)
        for entry in bb.list_by_category("test_cat"):
            bb.delete(entry["id"])

        bb.add_knowledge("Item 1", category="test_cat")
        bb.add_knowledge("Item 2", category="test_cat")
        bb.add_knowledge("Item 3", category="other")

        items = bb.list_by_category("test_cat")
        assert len(items) == 2

    def test_delete(self):
        bb = BeliefBase()
        bid = bb.add_knowledge("To be deleted")
        assert bb.delete(bid) is True
        assert bb.get_by_id(bid) is None

    def test_count(self):
        bb = BeliefBase()
        initial = bb.count()
        bb.add_knowledge("New entry")
        assert bb.count() == initial + 1


# ------------------------------------------------------------------
# GoalManager Tests
# ------------------------------------------------------------------

class TestGoalManager:
    """Test the Desire component of BDI architecture."""

    def test_create_goal(self):
        gm = GoalManager()
        goal = gm.create_goal("Test goal", priority=GoalPriority.HIGH)
        assert goal is not None
        assert goal.description == "Test goal"
        assert goal.priority == GoalPriority.HIGH
        assert goal.status == GoalStatus.PENDING

    def test_get_next_goal(self):
        gm = GoalManager()
        gm.create_goal("Low priority", priority=GoalPriority.LOW)
        gm.create_goal("High priority", priority=GoalPriority.HIGH)

        next_goal = gm.get_next_goal()
        assert next_goal is not None
        assert next_goal.description == "High priority"

    def test_activate_and_complete(self):
        gm = GoalManager()
        goal = gm.create_goal("Complete me")
        assert gm.activate_goal(goal.goal_id) is True
        assert gm.get_goal(goal.goal_id).status == GoalStatus.ACTIVE

        assert gm.update_goal_status(goal.goal_id, GoalStatus.COMPLETED) is True
        assert gm.get_goal(goal.goal_id).status == GoalStatus.COMPLETED

    def test_count_by_status(self):
        gm = GoalManager()
        gm.create_goal("Goal A")
        gm.create_goal("Goal B")
        counts = gm.count_by_status()
        assert counts.get("pending", 0) == 2

    def test_max_goals(self):
        gm = GoalManager(max_goals=2)
        assert gm.create_goal("Goal 1") is not None
        assert gm.create_goal("Goal 2") is not None
        assert gm.create_goal("Goal 3") is None  # Should be rejected

    def test_clear_completed(self):
        gm = GoalManager()
        g1 = gm.create_goal("Done")
        g2 = gm.create_goal("Pending")
        gm.update_goal_status(g1.goal_id, GoalStatus.COMPLETED)

        removed = gm.clear_completed()
        assert removed == 1
        assert gm.get_goal(g1.goal_id) is None
        assert gm.get_goal(g2.goal_id) is not None


# ------------------------------------------------------------------
# PlanExecutor Tests
# ------------------------------------------------------------------

class TestPlanExecutor:
    """Test the Intention component of BDI architecture."""

    def test_generate_plan(self):
        pe = PlanExecutor()
        plan = pe.generate_plan(
            goal_id="g1",
            goal_description="Step 1; Step 2; Step 3",
        )
        assert len(plan.steps) == 3
        assert plan.steps[0].status == StepStatus.PENDING

    def test_linear_dependencies(self):
        pe = PlanExecutor()
        plan = pe.generate_plan("g1", "A; B; C")
        # Step B should depend on A, C on B
        assert plan.steps[1].step_id in plan.steps[2].dependencies

    def test_execute_step(self):
        pe = PlanExecutor()
        plan = pe.generate_plan("g1", "Single step")

        def dummy_executor(step):
            return {"done": True}

        result = pe.execute_step(plan.plan_id, plan.steps[0].step_id, executor_fn=dummy_executor)
        assert result == {"done": True}
        assert pe.get_step(plan.plan_id, plan.steps[0].step_id).status == StepStatus.COMPLETED

    def test_get_progress(self):
        pe = PlanExecutor()
        plan = pe.generate_plan("g1", "A; B; C")
        pe.execute_step(plan.plan_id, plan.steps[0].step_id, executor_fn=lambda s: None)
        progress = pe.get_progress(plan.plan_id)
        assert progress["completed"] == 1
        assert progress["pending"] == 2

    def test_ready_steps_parallel(self):
        pe = PlanExecutor()
        plan = pe.generate_plan("g1", "A; B; C", dependencies=[
            {"step_index": 2, "depends_on": [0]},
        ])
        # Steps 0 and 1 should be ready (no deps), step 2 depends on 0
        ready = pe.get_ready_steps(plan.plan_id)
        ready_ids = [s.step_id for s in ready]
        assert plan.steps[0].step_id in ready_ids
        assert plan.steps[1].step_id in ready_ids
        assert plan.steps[2].step_id not in ready_ids


# ------------------------------------------------------------------
# StateManager Tests
# ------------------------------------------------------------------

class TestStateManager:
    """Test the distributed state manager."""

    def test_set_and_get(self):
        sm = StateManager()
        sm.set("test_key", {"value": 42})
        assert sm.get("test_key") == {"value": 42}

    def test_default_value(self):
        sm = StateManager()
        assert sm.get("missing", "default") == "default"

    def test_exists(self):
        sm = StateManager()
        sm.set("exists", "yes")
        assert sm.exists("exists") is True
        assert sm.exists("nope") is False

    def test_delete(self):
        sm = StateManager()
        sm.set("del_me", "value")
        assert sm.delete("del_me") is True
        assert sm.exists("del_me") is False

    def test_version_control(self):
        sm = StateManager()
        sm.set("ver_key", "v1")
        v1 = sm.get_version("ver_key")
        sm.set("ver_key", "v2")
        assert sm.get_version("ver_key") == v1 + 1

    def test_compare_and_set(self):
        sm = StateManager()
        sm.set("cas_key", "initial")
        v = sm.get_version("cas_key")
        # Should succeed with correct version
        assert sm.compare_and_set("cas_key", v, "updated") is True
        assert sm.get("cas_key") == "updated"
        # Should fail with stale version
        assert sm.compare_and_set("cas_key", v, "stale") is False

    def test_prefix_operations(self):
        sm = StateManager()
        sm.set("ns:a", 1)
        sm.set("ns:b", 2)
        sm.set("other", 3)

        result = sm.get_by_prefix("ns:")
        assert len(result) == 2
        assert "ns:a" in result

        deleted = sm.delete_by_prefix("ns:")
        assert deleted == 2
        assert not sm.exists("ns:a")


# ------------------------------------------------------------------
# MessageBus Tests
# ------------------------------------------------------------------

class TestMessageBus:
    """Test the async message bus."""

    def test_agent_registration(self):
        bus = MessageBus()
        received = []

        async def handler(msg):
            received.append(msg)

        bus.register_agent("agent1", handler)
        assert "agent1" in bus._agent_handlers
        bus.unregister_agent("agent1")
        assert "agent1" not in bus._agent_handlers

    def test_p2p_delivery(self):
        bus = MessageBus()
        received = []

        async def handler(msg):
            received.append(msg)

        bus.register_agent("receiver", handler)
        asyncio.run(bus.send_p2p("sender", "receiver", {"data": "hello"}))
        assert len(received) == 1
        assert received[0].payload == {"data": "hello"}

    def test_publish_subscribe(self):
        bus = MessageBus()
        received_a, received_b = [], []

        async def handler_a(msg):
            received_a.append(msg)

        async def handler_b(msg):
            received_b.append(msg)

        bus.register_agent("a", handler_a)
        bus.register_agent("b", handler_b)
        bus.subscribe("a", "news")
        bus.subscribe("b", "news")

        asyncio.run(bus.publish("pub", "news", {"title": "Breaking"}))
        assert len(received_a) == 1
        assert len(received_b) == 1

    def test_broadcast(self):
        bus = MessageBus()
        results = []

        async def make_handler(name):
            async def h(msg):
                results.append((name, msg.payload))
            return h

        bus.register_agent("a1", asyncio.run(make_handler("a1")))
        bus.register_agent("a2", asyncio.run(make_handler("a2")))

        asyncio.run(bus.broadcast("sender", {"alert": "test"}))
        # Both should receive
        assert len(results) == 2

    def test_request_response(self):
        bus = MessageBus()

        async def server_handler(msg):
            await bus.send_response(
                "server", msg.correlation_id, msg.sender,
                {"answer": "42"},
            )

        bus.register_agent("server", server_handler)

        response = asyncio.run(
            bus.request("client", "server", {"question": "life"}, timeout=2.0)
        )
        assert response == {"answer": "42"}

    def test_signature_verification(self):
        bus = MessageBus(secret_key="my-secret")
        msg = bus._create_message(MessageType.P2P, "s", {"x": 1}, recipient="r")
        assert bus.verify_signature(msg) is True

        # Tamper with payload
        msg.payload = {"x": 2}
        assert bus.verify_signature(msg) is False


# ------------------------------------------------------------------
# Agent Instantiation Tests
# ------------------------------------------------------------------

class TestAgents:
    """Test agent instantiation and basic operation."""

    def test_research_agent_instantiation(self):
        agent = ResearchAgent()
        assert agent.agent_type == "research"
        assert agent.belief_base is not None

    def test_research_agent_run(self):
        agent = ResearchAgent()
        state = {"question": "Tell me about AI"}
        result = asyncio.run(agent.run(state))
        assert "research_report" in result
        assert len(result["research_report"]) > 0

    def test_analysis_agent_instantiation(self):
        agent = AnalysisAgent()
        assert agent.agent_type == "analysis"

    def test_analysis_agent_run(self):
        agent = AnalysisAgent()
        state = {"question": "Analyze revenue data"}
        result = asyncio.run(agent.run(state))
        assert "analysis_result" in result
        assert len(result["analysis_result"]) > 0
        assert "analysis_visualization" in result

    def test_customer_service_agent_instantiation(self):
        assert agent.agent_type == "customer_service"

    def test_customer_service_agent_run(self):
        state = {"question": "My app keeps crashing!"}
        result = asyncio.run(agent.run(state))
        assert "customer_response" in result
        assert "sentiment" in result

    def test_cs_sentiment_positive(self):
        state = {"question": "Thank you, your service is great!"}
        result = asyncio.run(agent.run(state))

    def test_cs_role_switching(self):
        # Technical issue -> troubleshooter
        asyncio.run(agent.run({"question": "The app crashed"}))

        # Complaint -> escalation
        asyncio.run(agent.run({"question": "This is terrible! I want a refund!"}))

    def test_cs_conversation_history(self):
        asyncio.run(agent.run({"question": "Hello"}))
        history = agent.get_conversation_history()
        assert len(history) == 2  # user + assistant

        agent.reset_conversation()
        assert len(agent.get_conversation_history()) == 0


# ------------------------------------------------------------------
# Orchestrator Tests
# ------------------------------------------------------------------

class TestOrchestrator:
    """Test LangGraph workflow compilation and execution."""

    def test_build_graph(self):
        graph = build_graph()
        assert graph is not None
        # Verify graph compiles (build_graph already compiles)

    def test_graph_structure(self):
        graph = build_graph()
        nodes = graph.get_graph().nodes
        expected_nodes = {
            "decomposer_node", "router_node",
            "research_node", "analysis_node", "customer_service_node",
            "aggregator_node",
        }
        assert expected_nodes.issubset(set(nodes.keys()))

    def test_research_workflow(self):
        graph = build_graph()
        state = {
            "question": "Research about artificial intelligence",
            "raw_input": "Research about artificial intelligence",
            "thread_id": "test-thread-1",
            "messages": [],
            "intent": "research",
            "task_description": "",
            "sub_tasks": [],
            "retrieved_context": [],
            "plan": [],
            "current_step_index": 0,
            "quality_check_passed": False,
            "loop_count": 0,
            "final_response": "",
        }
        config = {"configurable": {"thread_id": "test-thread-1"}}
        result = asyncio.run(graph.ainvoke(state, config))
        assert "final_response" in result
        assert result.get("quality_check_passed") is True
        assert "research_report" in result or result.get("intent") == "research"

    def test_customer_service_workflow(self):
        graph = build_graph()
        state = {
            "question": "I need help with my account",
            "raw_input": "I need help with my account",
            "thread_id": "test-thread-2",
            "messages": [],
            "intent": "customer_service",
            "task_description": "",
            "sub_tasks": [],
            "retrieved_context": [],
            "plan": [],
            "current_step_index": 0,
            "quality_check_passed": False,
            "loop_count": 0,
            "final_response": "",
        }
        config = {"configurable": {"thread_id": "test-thread-2"}}
        result = asyncio.run(graph.ainvoke(state, config))
        assert "final_response" in result
        assert result.get("quality_check_passed") is True

    def test_empty_query(self):
        graph = build_graph()
        state = {
            "question": "",
            "raw_input": "",
            "thread_id": "test-thread-3",
            "messages": [],
            "intent": "",
            "task_description": "",
            "sub_tasks": [],
            "retrieved_context": [],
            "plan": [],
            "current_step_index": 0,
            "quality_check_passed": False,
            "loop_count": 0,
            "final_response": "",
        }
        config = {"configurable": {"thread_id": "test-thread-3"}}
        result = asyncio.run(graph.ainvoke(state, config))
        assert "final_response" in result or "error" in result


# ------------------------------------------------------------------
# Integration Smoke Test
# ------------------------------------------------------------------

def test_all_imports():
    """Verify all core components can be imported successfully."""
    from src.core.belief_base import BeliefBase
    from src.core.goal_manager import GoalManager
    from src.core.plan_executor import PlanExecutor
    from src.core.message_bus import MessageBus
    from src.core.state_manager import StateManager
    from src.agents.base_agent import BaseAgent
    from src.agents.research_agent import ResearchAgent
    from src.agents.analysis_agent import AnalysisAgent
    from src.workflows.orchestrator import build_graph
    from src.config import config
    from src.state import AgentState
    from src.api.routes import router

    bb = BeliefBase()
    gm = GoalManager()
    pe = PlanExecutor()
    bus = MessageBus()
    sm = StateManager()
    ra = ResearchAgent()
    aa = AnalysisAgent()
    graph = build_graph()

    assert bb is not None
    assert gm is not None
    assert pe is not None
    assert bus is not None
    assert sm is not None
    assert ra is not None
    assert aa is not None
    assert cs is not None
    assert graph is not None
