"""Global state definition using TypedDict for LangGraph.

Defines the shared state schema used across all workflow nodes.
Agent nodes read/write into this state, and LangGraph manages field-level
merging via the TypedDict reducer semantics.
"""

from typing import Annotated, Any, Dict, List, Optional, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    """Shared state flowing through the LangGraph orchestrator.

    Fields use total=False so nodes can set only the fields they need.
    LangGraph TypedDict subclass uses field-level merging (not full replacement).
    """

    # --- Input ---
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    raw_input: str
    user_name: str           # 当前用户名，用于文档权限过滤

    # --- Orchestration ---
    intent: str                    # "research" | "analysis"
    complexity: str                # "simple" | "complex" — determines fast path vs agentic path
    domain: Optional[str]          # 知识库领域标签: "finance" | "contract" | "law" | "general"
    current_agent: str             # Currently assigned agent id
    task_description: str          # Decomposed task description
    sub_tasks: List[str]           # List of sub-task strings

    # --- Memory ---
    long_term_context: str         # 跨会话的长期记忆上下文，由 memory.long_term 注入

    # --- Agent Outputs ---
    retrieved_context: List[Dict[str, Any]]  # Retrieved documents with doc_id, text, score
    research_report: str           # Structured research output
    analysis_result: str           # Statistical / analytical output
    analysis_visualization: str    # Path or JSON for visuals
    kb_search_error: str           # Knowledge base search error if any

    # --- Flow Control ---
    plan: List[Dict[str, Any]]     # Execution plan steps
    current_step_index: int
    quality_check_passed: bool
    loop_count: int
    final_response: str
    error: str

    # --- Tracing ---
    trace_id: str  # Request trace ID for log correlation

    # --- Review Extraction (deterministic pre-extraction for credit review tasks) ---
    review_extraction_context: str   # 系统自动提取的量化指标比对结果，注入 agent 上下文

    # --- Checkpointing ---
    thread_id: str
