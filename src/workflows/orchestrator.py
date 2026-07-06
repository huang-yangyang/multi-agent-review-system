"""LangGraph Orchestrator: Workflow engine for multi-agent coordination.

Dual-path architecture (no-HITL, straight-through execution):

    decomposer_node        ← complexity classification (simple / complex)
        |
    knowledge_retriever_node
        |
    router_node
     /   |   \              ← simple → fast path agents
research analysis customer_service
    |      |         |
    ├──────┼─────────┤
    |      |         |
    |   agentic_research_node  ← complex → agentic path (Tool Calling)
    |      |         |
     \     |         /
    aggregator_node
        |
       END

Checkpointer: AsyncSqliteSaver (persistent across sessions).
"""

import json
import re
import traceback
from pathlib import Path as _Path
from typing import Dict, List, Literal, Optional, Tuple

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.config import get_stream_writer
from src.semantic_cache import get_cache
from src.state import AgentState
from src.config import config
from src.agents.research_agent import ResearchAgent
from src.agents.analysis_agent import AnalysisAgent
from src.tools import knowledge_search
from src.prompts import get_prompt
from src.core.logging_config import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 授信报告审查专用 System Prompt（6 项系统性改进封装）
# ══════════════════════════════════════════════════════════════════════════════

REVIEW_SYSTEM_PROMPT = get_prompt("review_finance_system")  # 从 prompts.py 加载


# ── 审查任务检测关键词 ──
_REVIEW_KEYWORDS = [
    "审查", "审核", "检查", "审阅", "核查", "复核",
    "review", "audit", "check",
]
# 扩展：合同/法律领域的风险分析也应走 Map-Reduce
_CONTRACT_REVIEW_KEYWORDS = [
    "风险", "分析", "评估", "合规", "条款",
]
# ── 审查输出结构标记（用于代码级护栏校验） ──
# 若 LLM 输出中缺少任一标记，_validate_review_output 将触发降级补充
# 更新：覆盖 6 项系统性缺陷的强制检查——
#   [1] 量化指标对照表  [2] 抵押率对照  [3] 审批权限核查
#   [4] 阶段分界         [5] 来源分区标记  [6] 禁止模糊词汇
_REVIEW_REQUIRED_MARKERS = [
    r"量化指标逐项比对",
    r"指标名称|资产负债率|流动比率|速动比率",
    r"抵押率与规程对照",
    r"抵押率",
    r"审批权限逐级核查",
    r"审批层级|审批权限",
    r"深度问题分析",
    r"📄",
]

# ── 审查输出禁止使用的模糊词汇 ──
_REVIEW_BANNED_FUZZY_PATTERNS = [
    r'表面上看',
    r'似乎',
    r'可能(?!.*未提供)',  # 允许"报告中未提供"，禁止其他"可能"
    r'大概',
    r'貌似',
    r'或许',
    r'基本上是',
    r'大致上',
]


def _validate_review_output(report: str, trace_id: str = "") -> str:
    """代码级护栏：校验审查输出是否包含强制结构标记 + 禁止模糊词汇。

    即使 LLM 忽略了 prompt 中的结构化指令，本函数在代码层面检测输出
    是否包含「第一阶段」「指标名称」「第二阶段」「抵押率」「审批层级」「📄」等强制标记。
    缺少任一标记时，记录 warning 日志并在输出末尾追加结构化的指标对照
    框架作为降级补充，确保最终输出至少具备最低限度的结构化审查框架。

    同时检测是否出现「表面上看」「似乎」「可能」「大概」等禁止的模糊词汇，
    出现时追加醒目警告并列出检测到的违规词汇及上下文。

    Args:
        report: LLM 生成的审查报告全文。
        trace_id: 用于日志关联的 trace ID。

    Returns:
        校验通过则返回原始 report；否则返回追加补充框架后的 report。
    """
    missing = [m for m in _REVIEW_REQUIRED_MARKERS if not re.search(m, report)]

    # ── 检测禁止的模糊词汇 ──
    fuzzy_hits: List[str] = []
    for pattern in _REVIEW_BANNED_FUZZY_PATTERNS:
        for match in re.finditer(pattern, report):
            start = max(0, match.start() - 15)
            end = min(len(report), match.end() + 15)
            ctx = report[start:end].replace("\n", " ").strip()
            fuzzy_hits.append(f"  · 「{match.group()}」→ …{ctx}…")

    supplement_parts: List[str] = []

    if missing:
        logger.warning(
            f"_validate_review_output: LLM output missing mandatory markers: {missing}",
            extra={"component": "orchestrator", "trace_id": trace_id},
        )
        supplement_parts.append(
            "\n\n---\n"
            "## ⚠️ 审查范式补正（系统自动补充）\n\n"
            "LLM 未按规定的结构化审查范式输出，以下为强制补充的指标对照框架：\n\n"
            "### 量化指标逐项比对（第一阶段 · 任务 A）\n\n"
            "| 指标名称 | 规程阈值 | 报告实际值 | 判定（达标/不达标/报告中未提供） | 依据 |\n"
            "|---------|---------|-----------|--------------------------|------|\n"
            "| （请结合上述分析内容，逐项填写规程与报告中的量化指标） | ... | ... | ... | ... |\n\n"
            "### 抵押率与规程对照（第一阶段 · 任务 B）\n\n"
            "| 抵押物类型 | 规程抵押率上限 | 报告使用抵押率 | 是否合规 | 按规程重算的可担保额度 |\n"
            "|-----------|-------------|-------------|---------|-------------------|\n"
            "| ... | ... | ... | ... | ... |\n\n"
            "### 审批权限逐级核查（第一阶段 · 任务 C）\n\n"
            "| 报送机构 | 敞口金额 | 规程要求审批层级 | 实际审批层级 | 是否合规 |\n"
            "|---------|---------|----------------|------------|---------|\n"
            "| ... | ... | ... | ... | ... |\n\n"
            "### 深度问题分析（第二阶段）\n\n"
            "（基于第一阶段完整比对后，进行关联互保等复杂推理）\n"
        )

    if fuzzy_hits:
        logger.warning(
            f"_validate_review_output: LLM output contains banned fuzzy words: {len(fuzzy_hits)} hit(s)",
            extra={"component": "orchestrator", "trace_id": trace_id},
        )
        supplement_parts.append(
            "\n\n---\n"
            "## ⚠️ 表述规范补正（系统自动检测）\n\n"
            "审查输出中检测到以下**禁止使用的模糊词汇**，应替换为「达标 / 不达标 / 合规 / 不合规」等明确结论：\n\n"
            + "\n".join(fuzzy_hits[:10])  # 最多展示 10 条
            + "\n\n> 请将所有模糊判定改为明确的二值结论。\n"
        )

    if not supplement_parts:
        return report

    return report + "".join(supplement_parts)


async def _reformat_review_output(raw_report: str, trace_id: str = "") -> str:
    """第二遍调用 LLM，专门将 agent 原始输出按模板重新格式化。

    使用独立的 LLM 调用（非 ReAct），上下文纯净，仅包含：
    1. 简短的格式化 system prompt
    2. 原始报告作为 user message

    LLM 只做格式化，不做新的分析。
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.config import config

    formatting_system_prompt = """你是授信报告格式化助手。你的唯一任务是将原始审查内容重组为规定格式。不要新增分析，只做格式转换。

输出结构：
## 一、量化指标逐项比对
（逐项列出：资产负债率、流动比率、速动比率、EBITDA利息保障倍数、存货周转率、应收账款周转率、销售增长率、总资产报酬率。每个指标单独一行，格式：| 指标 | 规程阈值 | 实际值 | 判定 |）

## 二、抵押率与规程对照
（以规程规定的抵押率上限重新计算可担保额度）

## 三、审批权限逐级核查
（核对该金额在审批层级表中的位置）

## 四、深度问题分析
（关联互保、贷后预警等）

## 五、总结

来源分区：依据操作规程判断的归📄，纯外部常识归🤖，网络信息归🌐。禁止使用"表面上看/似乎/可能/大概/貌似/或许"等模糊词。"""

    try:
        llm = ChatOpenAI(
            model=config.llm.deepseek_model,
            api_key=config.llm.effective_api_key,
            base_url=config.llm.effective_base_url,
            temperature=0.1,
            max_tokens=8192,
        )
        response = await llm.ainvoke([
            SystemMessage(content=formatting_system_prompt),
            HumanMessage(content=f"请将以下审查报告重新格式化：\n\n{raw_report}"),
        ])
        formatted = response.content
        logger.info(
            f"_reformat_review_output: completed, {len(raw_report)} -> {len(formatted)} chars",
            extra={"component": "orchestrator", "trace_id": trace_id},
        )
        return formatted
    except Exception as e:
        logger.warning(
            f"_reformat_review_output failed: {e}, falling back to raw report",
            extra={"component": "orchestrator", "trace_id": trace_id},
        )
        return raw_report


# ── 确定性前置提取：附件解析 ──

def _parse_attachment_text(question: str) -> Tuple[Optional[str], Optional[str], str]:
    """从 question 中解析第一个【附件：xxx】块。

    Returns:
        (attachment_name, attachment_content, clean_question)
        - attachment_name: 附件文件名（不含【附件：】标记）
        - attachment_content: 附件全文内容
        - clean_question: 去除附件块后的问题文本
    """
    if not question:
        return None, None, question

    pattern = r'【附件：(.+?)】\n'
    match = re.search(pattern, question)
    if not match:
        return None, None, question

    name = match.group(1)
    start = match.end()

    # 找到下一个附件块或文本结束
    next_block = re.search(pattern, question[start:])
    if next_block:
        content = question[start:start + next_block.start()].strip()
    else:
        content = question[start:].strip()

    # 去除附件块，重组问题文本
    clean = question[:match.start()].strip()
    remaining = question[start + len(content):].strip()
    # 跳过剩余部分的开头空白
    if remaining.startswith(content):
        remaining = remaining[len(content):].strip()
    if remaining:
        # 如果剩余部分包含下一个附件，递归处理
        clean = f"{clean}\n\n{remaining}" if clean else remaining

    logger.debug(
        f"_parse_attachment_text: name='{name}', content_len={len(content)}, clean_len={len(clean)}",
        extra={"component": "orchestrator"},
    )
    return name, content, clean


# ── 确定性前置提取：审查提取编排 ──

async def _run_review_extraction(question: str, trace_id: str) -> str:
    """编排授信报告审查的确定性前置提取流程。

    步骤：
    1. 从 question 解析附件（报告）文本
    2. 从 indexer 获取操作规程全文
    3. 运行 review_extractor 提取并比对
    4. 返回格式化后的上下文

    Returns:
        格式化的比对结果 Markdown 字符串，失败时返回空字符串。
    """
    from src.review_extractor import extract_all, format_context
    from src.rag.indexer import get_indexer

    try:
        # Step 1: 解析附件
        att_name, report_text, _clean_q = _parse_attachment_text(question)
        if not att_name or not report_text:
            logger.info(
                "_run_review_extraction: no attachment in question, skipping",
                extra={"component": "orchestrator", "trace_id": trace_id},
            )
            return ""

        logger.info(
            f"_run_review_extraction: found attachment '{att_name}', {len(report_text)} chars",
            extra={"component": "orchestrator", "trace_id": trace_id},
        )

        # Step 2: 获取操作规程全文
        indexer = get_indexer(
            uploads_dir=config.rag.uploads_dir,
            indexes_dir=config.rag.indexes_dir,
        )

        procedure_text = None
        for name_pattern in [
            "信用风险评估操作规程",
            "风险评估操作规程",
            "操作规程",
        ]:
            procedure_text = indexer.get_full_document(name_pattern)
            if procedure_text:
                logger.info(
                    f"_run_review_extraction: found procedure doc via '{name_pattern}', "
                    f"{len(procedure_text)} chars",
                    extra={"component": "orchestrator", "trace_id": trace_id},
                )
                break

        if not procedure_text:
            logger.warning(
                "_run_review_extraction: procedure document not found in indexer, skipping",
                extra={"component": "orchestrator", "trace_id": trace_id},
            )
            return ""

        # Step 3: 执行提取和比对
        result = extract_all(report_text, procedure_text)

        if result.extraction_failed:
            logger.warning(
                f"_run_review_extraction: extraction failed: {result.extraction_warnings}",
                extra={"component": "orchestrator", "trace_id": trace_id},
            )
            return ""

        if not result.success and not result.financial_indicators:
            logger.warning(
                "_run_review_extraction: no indicators extracted, skipping",
                extra={"component": "orchestrator", "trace_id": trace_id},
            )
            return ""

        # Step 4: 格式化
        context = format_context(result)
        logger.info(
            f"_run_review_extraction: extracted {len(result.financial_indicators)} indicators, "
            f"{len(result.indicator_comparisons)} comparisons, "
            f"context={len(context)} chars",
            extra={"component": "orchestrator", "trace_id": trace_id},
        )
        return context

    except Exception as e:
        logger.warning(
            f"_run_review_extraction: unexpected error (graceful degradation): {e}",
            extra={"component": "orchestrator", "trace_id": trace_id},
            exc_info=True,
        )
        return ""


# ── 领域检测：关键词 → domain 标签映射 ──
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "finance": [
        "授信", "风控", "风险", "贷款", "信贷", "评级", "金融", "担保", "抵押", "预警",
        "审批", "资产负债", "负债率", "流动比率", "EBITDA", "现金流", "利率", "还本", "敞口",
        "征信", "逾期", "操作规程", "审批权限", "授信额度", "还本付息", "抵押率",
        "资产", "负债", "贷后", "额度", "利息保障",
    ],
    "contract": [
        "合同", "协议", "条款", "签约", "违约", "履约", "仲裁", "保密", "竞业",
        "价款", "交付", "验收", "质保", "不可抗力", "知识产权", "转让", "终止",
    ],
    "law": [
        "法律", "法规", "诉讼", "法院", "判决", "执行", "辩护", "法条",
        "司法解释", "立法", "被告", "原告", "上诉", "当事人", "管辖",
        "官司", "胜诉", "败诉", "罪名", "刑期", "赔偿", "侵权", "承担",
        "责任", "合同纠纷", "劳动仲裁",
    ],
}


def _detect_domain(text: str) -> str:
    """基于关键词匹配检测查询所属领域。

    Args:
        text: 用户输入的原始文本。

    Returns:
        "finance" | "contract" | "law" | "general"
    """
    scores: Dict[str, int] = {"finance": 0, "contract": 0, "law": 0}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text)
    best = max(scores, key=scores.get)
    # 需要至少 2 个关键词命中，或者 best 是唯一命中且 ≥ 1
    if scores[best] >= 2:
        return best
    only_match = scores[best] == 1 and sum(1 for v in scores.values() if v > 0) == 1
    return best if only_match else "general"


# ------------------------------------------------------------------
# Node Functions
# ------------------------------------------------------------------

async def decomposer_node(state: AgentState) -> Dict:
    """Decompose user input into structured task description and sub-tasks."""
    trace_id = state.get("trace_id", "")
    logger.info(
        "decomposer_node: starting task decomposition",
        extra={"component": "orchestrator", "trace_id": trace_id},
    )

    question = state.get("question", "") or state.get("raw_input", "")
    if not question:
        return {"error": "No input provided", "task_description": ""}

    # ── Extract user's real question ──
    # Attachments are injected as 【附件：xxx】 blocks into the question.
    # Intent classification must act on the user's actual question only,
    # not on keywords accidentally present in attached documents
    # (e.g., "数据分析" in a contract → misclassified as analysis intent).
    attachment_marker = "【附件："
    user_question = question.split(attachment_marker)[0].strip() if attachment_marker in question else question

    question_lower = user_question.lower()
    if any(w in question_lower for w in ["analyze", "analysis", "statistics", "data", "分析", "统计"]):
        intent = "analysis"
    elif any(w in question_lower for w in ["research", "search", "find", "study", "研究", "搜索", "查找"]):
        intent = "research"
    else:
        intent = "research"

    # ── Complexity classification ──
    # Simple: fact Q&A, single-entity lookup → fast path (zero overhead)
    # Complex: comparison, multi-step reasoning, multi-entity, analysis → agentic path
    complex_signals = [
        "比较", "对比", "区别", "哪个更好", "优缺点", "优势劣势", "利弊",
        "分析", "评估", "方案", "建议", "推荐",
        "vs", " versus ",
    ]
    is_complex = any(s in question_lower for s in complex_signals)
    # Also mark as complex if the query is long (multi-sentence / compound)
    sentence_count = user_question.count("。") + user_question.count("？") + user_question.count("?") + user_question.count(".")
    if sentence_count >= 2:
        is_complex = True

    complexity = "complex" if is_complex else "simple"

    task_desc = f"Process user request with intent='{intent}': {question}"
    sub_tasks = [f"Understand query: {question[:60]}..."]
    if complexity == "simple":
        sub_tasks.append(f"Execute {intent} pipeline — internal KB + external web search")
    else:
        sub_tasks.append(f"Agentic {intent} pipeline — LLM Tool Calling loop")
    sub_tasks.append("Aggregate and format results")

    logger.info(
        f"decomposer_node: intent={intent}, complexity={complexity}",
        extra={"component": "orchestrator", "trace_id": trace_id, "intent": intent, "complexity": complexity},
    )

    # ── Domain detection（先看问题，再看附件） ──
    domain = _detect_domain(user_question)

    # 如果问题本身未命中任何领域关键词，但用户上传了附件，
    # 则扫描附件内容做二次领域检测（解决"帮我看看这个文档"类问题）
    if domain == "general" and attachment_marker in question:
        attachment_content = question.split(attachment_marker, 1)[1] if attachment_marker in question else ""
        if attachment_content:
            domain = _detect_domain(attachment_content[:2000])  # 只扫描前2000字符
            logger.info(
                f"decomposer_node: domain upgraded to {domain} via attachment scan",
                extra={"component": "orchestrator", "trace_id": trace_id, "domain": domain},
            )

    logger.info(
        f"decomposer_node: domain={domain}",
        extra={"component": "orchestrator", "trace_id": trace_id, "domain": domain},
    )

    # Emit execution trace event
    writer = get_stream_writer()
    writer({
        "phase": "decomposer_done",
        "message": f"意图: {intent}, 复杂度: {complexity}, 领域: {domain}",
        "intent": intent,
        "complexity": complexity,
        "domain": domain,
    })

    return {
        "intent": intent,
        "complexity": complexity,
        "domain": domain,
        "task_description": task_desc,
        "sub_tasks": sub_tasks,
        "retrieved_context": state.get("retrieved_context", []),
    }


async def knowledge_retriever_node(state: AgentState) -> Dict:
    """Pre-search internal knowledge base. Enriches state with context."""
    trace_id = state.get("trace_id", "")
    logger.info(
        "knowledge_retriever_node: pre-searching KB",
        extra={"component": "orchestrator", "trace_id": trace_id},
    )

    question = state.get("question", "") or state.get("raw_input", "")
    domain = state.get("domain")
    if not question:
        writer = get_stream_writer()
        writer({"phase": "kb_presearch_done", "message": "知识库预检索跳过（无问题文本）"})
        return {"retrieved_context": []}

    try:
        raw = knowledge_search(question, top_k=5, domain=domain)
        result = json.loads(raw)
        contexts = []
        if result.get("found"):
            for r in result.get("results", []):
                contexts.append({
                    "doc_id": r.get("doc_id", ""),
                    "text": r.get("text", "")[:500],
                    "score": r.get("score", 0),
                })
        writer = get_stream_writer()
        domain_info = f" (领域: {domain})" if domain and domain != "general" else ""
        writer({"phase": "kb_presearch_done", "message": f"知识库预检索完成 ({len(contexts)} 条命中){domain_info}"})
        return {"retrieved_context": contexts}
    except Exception as e:
        logger.error(
            f"knowledge_retriever_node failed: {e}",
            extra={"component": "orchestrator", "trace_id": trace_id},
            exc_info=True,
        )
        writer = get_stream_writer()
        writer({"phase": "kb_presearch_done", "message": f"知识库预检索失败: {e}"})
        return {"retrieved_context": [], "kb_search_error": str(e)}


# ── Agent Execution Nodes ────────────────────────────

def router_node(state: AgentState) -> Dict:
    """Route to the appropriate agent node based on detected intent."""
    trace_id = state.get("trace_id", "")
    intent = state.get("intent", "research")
    complexity = state.get("complexity", "simple")
    # Choose visible path label
    if intent == "research" and complexity == "complex":
        path_label = f"研究节点 (自主搜索)"
    else:
        path_label = f"{intent} 节点 (快速通道)"

    logger.info(
        f"router_node: routing to '{intent}'",
        extra={"component": "orchestrator", "trace_id": trace_id, "intent": intent},
    )

    writer = get_stream_writer()
    writer({
        "phase": "routing_done",
        "message": f"路由 → {path_label}",
        "intent": intent,
        "complexity": complexity,
    })

    return {"current_agent": intent}


def _collect_source_info(retrieved_context: List[Dict]) -> tuple:
    """从 retrieved_context 提取源文件路径及最大 mtime。

    返回 (source_files_str, max_mtime)，供语义缓存绑定。
    source_files_str 格式: "path1|path2|..."
    """
    uploads_dir = _Path(config.rag.uploads_dir)
    seen = set()
    max_mtime = 0.0
    for ctx in retrieved_context or []:
        doc_id = ctx.get("doc_id", "")
        if "::" in doc_id:
            filename = doc_id.split("::")[0]  # doc_id = filename::hash::chunk_idx
            fpath = uploads_dir / filename
            if fpath not in seen and fpath.exists():
                seen.add(fpath)
                mtime = fpath.stat().st_mtime
                if mtime > max_mtime:
                    max_mtime = mtime
    source_files_str = "|".join(str(p) for p in seen)
    return source_files_str, max_mtime


async def research_node(state: AgentState) -> Dict:
    """Execute the Research Agent pipeline (fast path: fixed 3-phase with streaming).

    Emits custom SSE events for each phase so the frontend can show progress.
    Cache-first: check semantic cache before running full pipeline.
    """
    trace_id = state.get("trace_id", "")
    question = state.get("question", "") or state.get("raw_input", "")
    writer = get_stream_writer()

    logger.info(
        "research_node: executing research pipeline (fast path)",
        extra={"component": "orchestrator", "trace_id": trace_id},
    )
    try:
        agent = ResearchAgent(agent_id="research_orchestrator")

        # Phase 0: pre-retrieved context from orchestrator
        pre_contexts = state.get("retrieved_context", []) or []
        pre_texts = [c.get("text", "") for c in pre_contexts if c.get("text")]

        # Phase 1: Internal KB search —─ emit status
        writer({"phase": "kb_search", "message": "正在搜索知识库..."})
        kb_texts = list(pre_texts)
        if not kb_texts:
            kb_texts = await agent._search_internal_texts(question, trace_id)
        writer({
            "phase": "kb_search_done",
            "message": f"知识库匹配到 {len(kb_texts)} 条相关片段",
            "count": len(kb_texts),
        })

        # Phase 2: External web search —─ emit status
        writer({"phase": "web_search", "message": "正在联网搜索..."})
        web_texts = agent._search_external_texts(question, trace_id)
        writer({
            "phase": "web_search_done",
            "message": f"联网搜索到 {len(web_texts)} 条结果",
            "count": len(web_texts),
        })

        # Extract conversation history
        raw_messages = state.get("messages", []) or []
        history_messages: List[Dict] = []
        for m in raw_messages:
            role = "user" if m.__class__.__name__ == "HumanMessage" else "assistant"
            history_messages.append({"role": role, "content": getattr(m, "content", "")})

        long_term_context = state.get("long_term_context", "") or ""

        # ── 审查任务检测（finance 领域专属，严禁泄漏到 contract/law 等其他领域） ──
        # 注入条件：domain == "finance" AND 问题包含审查关键词
        # 领域隔离保证：domain 由 _detect_domain() 基于 finance 领域关键词（授信/风控/
        # 贷款/资产负债等）匹配生成，contract/law/general 领域不会命中 finance，
        # 从而确保 REVIEW_SYSTEM_PROMPT 不会跨领域泄漏。
        domain = state.get("domain", "general")

        # ── 缓存查：语义指纹匹配（含领域过滤 + 文档感知） ──
        cache = get_cache()
        if question:
            source_files_str, _ = _collect_source_info(pre_contexts)
            cached = cache.search(question, domain=domain, source_file=source_files_str)
            if cached:
                answer, score = cached
                logger.info(
                    f"research_node: semantic cache hit (score={score:.4f})",
                    extra={"component": "orchestrator", "trace_id": trace_id},
                )
                writer({"phase": "cache_hit", "message": f"命中语义缓存 (相似度 {score:.2f})"})
                return {
                    "research_report": answer,
                    "retrieved_context": [],
                    "cache_hit": True,
                }

        question_lower = question.lower()
        is_review_task = (
            domain == "finance"
            and any(kw in question_lower for kw in _REVIEW_KEYWORDS)
        )
        system_prompt_override = REVIEW_SYSTEM_PROMPT if is_review_task else None
        if is_review_task:
            logger.info(
                "research_node: detected finance review task, injecting REVIEW_SYSTEM_PROMPT",
                extra={"component": "orchestrator", "trace_id": trace_id},
            )
            writer({"phase": "review_mode", "message": "检测到授信报告审查任务，启用专用审查流程"})

        # ── 确定性前置提取：从报告和规程中自动提取并比对量化指标 ──
        if is_review_task:
            review_extraction_context = await _run_review_extraction(question, trace_id)
            if review_extraction_context:
                # 将提取结果插入 kb_texts 头部，作为 LLM 的优先参考
                kb_texts.insert(0, review_extraction_context)
                writer({"phase": "review_extraction", "message": "已完成量化指标自动提取与比对"})

        # Phase 3: LLM synthesis —─ streaming token-by-token

        writer({"phase": "synthesize", "message": "正在生成回答..."})
        full_report = ""
        async for token in agent._synthesize_stream(
            question, kb_texts, web_texts, trace_id,
            history_messages, long_term_context,
            system_prompt_override=system_prompt_override,
        ):
            full_report += token
            writer({"phase": "token", "content": token})

        writer({"phase": "synthesize_done", "message": f"回答生成完成 ({len(full_report)} 字符)"})

        # ── 代码级护栏：审查输出结构校验 ──
        if is_review_task and full_report:
            full_report = _validate_review_output(full_report, trace_id)

        # ── 两步法第二步：独立 LLM 调用重新格式化 ──
        if is_review_task and full_report:
            full_report = await _reformat_review_output(full_report, trace_id)

        # ── 缓存写 ──
        if question and full_report:
            source_files, max_mtime = _collect_source_info(pre_contexts)
            cache.add(question, full_report, source_file=source_files, indexed_at=max_mtime, domain=domain)

        return {
            "research_report": full_report,
            "retrieved_context": [],
            "cache_hit": False,
        }
    except Exception:
        logger.error(
            "research_node: execution failed",
            extra={"component": "orchestrator", "trace_id": trace_id},
            exc_info=True,
        )
        writer({"phase": "error", "message": "研究节点执行异常，返回原始上下文"})
        return {
            "research_report": "",
            "retrieved_context": state.get("retrieved_context", []),
            "error": traceback.format_exc(),
        }


async def agentic_research_node(state: AgentState) -> Dict:
    """Execute the Research Agent in agentic mode (Tool Calling loop).

    LLM autonomously decides: which tool to call, how many times, when to stop.
    Uses bind_tools([kb_search_tool, web_search_tool, calculate_tool]) + LangGraph ReAct.
    Emits status events for tool usage visualization.
    """
    trace_id = state.get("trace_id", "")
    writer = get_stream_writer()
    logger.info(
        "agentic_research_node: executing agentic research pipeline",
        extra={"component": "orchestrator", "trace_id": trace_id},
    )

    # ── 审查任务检测（finance 领域专属，严禁泄漏到 contract/law 等其他领域） ──
    # 注入条件：domain == "finance" AND 问题包含审查关键词
    # 领域隔离保证：domain 由 _detect_domain() 基于 finance 领域关键词（授信/风控/
    # 贷款/资产负债等）匹配生成，contract/law/general 领域不会命中 finance，
    # 从而确保 REVIEW_SYSTEM_PROMPT 不会跨领域泄漏。
    question = state.get("question", "") or state.get("raw_input", "")
    domain = state.get("domain", "general")
    question_lower = question.lower()
    is_review_task = (
        domain == "finance"
        and any(kw in question_lower for kw in _REVIEW_KEYWORDS)
    )
    system_prompt_override = REVIEW_SYSTEM_PROMPT if is_review_task else None
    if is_review_task:
        logger.info(
            "agentic_research_node: detected finance review task, injecting REVIEW_SYSTEM_PROMPT",
            extra={"component": "orchestrator", "trace_id": trace_id},
        )
        writer({"phase": "review_mode", "message": "检测到授信报告审查任务，启用专用审查流程"})

        # ── 确定性前置提取：从报告和规程中自动提取并比对量化指标 ──
        if is_review_task:
            review_extraction_context = await _run_review_extraction(question, trace_id)
            if review_extraction_context:
                state["review_extraction_context"] = review_extraction_context
                writer({"phase": "review_extraction", "message": "已完成量化指标自动提取与比对"})

    try:
        writer({"phase": "agentic_start", "message": "启动自主搜索 (Tool Calling 模式)"})
        agent = ResearchAgent(agent_id="agentic_research_orchestrator")

        result = await agent._agentic_search(state, system_prompt_override=system_prompt_override)
        report = result.get("research_report", "")
        # ── 代码级护栏：审查输出结构校验 ──
        if is_review_task and report:
            report = _validate_review_output(report, trace_id)
        # ── 两步法第二步：独立 LLM 调用重新格式化 ──
        if is_review_task and report:
            report = await _reformat_review_output(report, trace_id)
        writer({"phase": "agentic_done", "message": f"自主搜索完成 ({len(report)} 字符)"})
        return {
            "research_report": report,
            "retrieved_context": result.get("retrieved_context", []),
        }
    except Exception:
        logger.error(
            "agentic_research_node: execution failed",
            extra={"component": "orchestrator", "trace_id": trace_id},
            exc_info=True,
        )
        writer({"phase": "error", "message": "自主搜索失败，降级到快速通道"})
        return {
            "research_report": "",
            "retrieved_context": state.get("retrieved_context", []),
            "error": traceback.format_exc(),
        }


async def review_pipeline_node(state: AgentState) -> Dict:
    """Map-Reduce 审查管线 — 审查任务的专用路径（非 ReAct）。

    Map 阶段（确定性代码）：提取 7 项指标 + 规程阈值 + 逐项比对
    Reduce 阶段（单次 LLM）：基于 Map 结果做深度定性分析

    不经过 ReAct 循环，不使用工具调用。
    Map 失败时自动降级为纯 LLM 审查。
    """
    trace_id = state.get("trace_id", "")
    question = state.get("question", "") or state.get("raw_input", "")
    writer = get_stream_writer()

    logger.info(
        "review_pipeline_node: starting Map-Reduce review pipeline",
        extra={"component": "orchestrator", "trace_id": trace_id},
    )

    try:
        domain = state.get("domain", "general")

        # ── 获取参考文档全文（按领域选择不同的搜索词） ──
        from src.rag.indexer import get_indexer
        indexer = get_indexer(
            uploads_dir=config.rag.uploads_dir,
            indexes_dir=config.rag.indexes_dir,
        )

        # 按领域选择参考文档搜索词
        if domain == "contract":
            doc_patterns = ["劳动法知识库", "合同风险知识库", "合同知识库", "合同风险"]
        elif domain == "law":
            doc_patterns = ["法律知识库", "法规", "法律法规"]
        else:  # finance (default)
            doc_patterns = ["信用风险评估操作规程", "风险评估操作规程", "操作规程"]

        # 收集所有匹配的参考文档（合同领域可能匹配多个KB）
        reference_parts = []
        for name_pattern in doc_patterns:
            doc_text = indexer.get_full_document(name_pattern) or ""
            if doc_text:
                logger.info(
                    f"review_pipeline_node: found reference doc via '{name_pattern}' for domain={domain}",
                    extra={"component": "orchestrator", "trace_id": trace_id},
                )
                reference_parts.append(doc_text)
                if domain != "contract":
                    break  # 非合同领域只需第一个匹配
        reference_text = "\n\n".join(reference_parts) if reference_parts else ""

        if not reference_text:
            writer({"phase": "review_warning", "message": f"知识库中未找到{domain}领域的参考文档，使用 ReAct 模式"})
            return await agentic_research_node(state)

        # ── 按领域执行对应的 Map-Reduce 管线 ──
        if domain == "contract":
            from src.review_pipeline import execute_contract_review_pipeline
            writer({"phase": "review_map_start", "message": "Map 阶段：正在提取合同条款并匹配风险模式..."})
            final_report, map_succeeded = await execute_contract_review_pipeline(
                question, reference_text, trace_id,
            )
            writer({"phase": "review_map_done" if map_succeeded else "review_fallback",
                    "message": "Map 完成" if map_succeeded else "Map 失败，LLM 降级"})
        elif domain == "finance":
            from src.review_pipeline import execute_review_pipeline
            writer({"phase": "review_map_start", "message": "Map 阶段：正在自动提取并比对量化指标..."})
            final_report, map_succeeded = await execute_review_pipeline(
                question, reference_text, trace_id,
            )
            writer({"phase": "review_map_done" if map_succeeded else "review_fallback",
                    "message": "Map 完成" if map_succeeded else "Map 失败，LLM 降级"})
        elif domain == "law":
            from src.review_pipeline import execute_labor_review_pipeline
            writer({"phase": "review_map_start", "message": "Map 阶段：正在提取劳动合同条款并匹配风险模式..."})
            final_report, map_succeeded = await execute_labor_review_pipeline(
                question, reference_text, trace_id,
            )
            writer({"phase": "review_map_done" if map_succeeded else "review_fallback",
                    "message": "Map 完成" if map_succeeded else "Map 失败，LLM 降级"})
        else:
            # general — 通用 Map-Reduce：全文提取 + LLM 逐项审查
            from src.review_pipeline import run_fallback_review
            from src.workflows.orchestrator import _parse_attachment_text
            _, doc_text, _ = _parse_attachment_text(question)
            writer({"phase": "review_map_start", "message": f"通用审查模式（{domain}领域）：正在分析文档..."})
            if doc_text and reference_text:
                final_report = await run_fallback_review(doc_text, reference_text, trace_id)
                map_succeeded = False  # fallback 不是严格 Map
            else:
                final_report = "无法提取文档内容，请确认文件已上传。"
                map_succeeded = False
            writer({"phase": "review_done" if final_report else "review_error",
                    "message": "审查完成" if final_report else "审查失败"})

        writer({"phase": "review_done", "message": f"审查完成 ({len(final_report)} 字符)"})

        # ── 护栏校验 ──
        final_report = _validate_review_output(final_report, trace_id)

        # ── 缓存写 ──
        if question and final_report:
            cache.add(question, final_report, domain=domain,
                      source_file=_collect_source_info(state.get("retrieved_context", []))[0])

        return {
            "research_report": final_report,
            "review_extraction_context": "",
            "retrieved_context": [],
        }

    except Exception:
        logger.error(
            "review_pipeline_node: execution failed, falling back to agentic node",
            extra={"component": "orchestrator", "trace_id": trace_id},
            exc_info=True,
        )
        writer({"phase": "review_error", "message": "审查管线异常，降级到 ReAct 模式"})
        return await agentic_research_node(state)


async def analysis_node(state: AgentState) -> Dict:
    """Execute the Analysis Agent pipeline with status events."""
    trace_id = state.get("trace_id", "")
    writer = get_stream_writer()
    logger.info(
        "analysis_node: executing analysis pipeline",
        extra={"component": "orchestrator", "trace_id": trace_id},
    )
    try:
        writer({"phase": "analysis_start", "message": "正在执行数据分析..."})
        agent = AnalysisAgent(agent_id="analysis_orchestrator")
        result = await agent.run(state)
        writer({"phase": "analysis_done", "message": "数据分析完成"})
        return {
            "analysis_result": result.get("analysis_result", ""),
            "analysis_visualization": result.get("analysis_visualization", ""),
        }
    except Exception:
        logger.error(
            "analysis_node: execution failed",
            extra={"component": "orchestrator", "trace_id": trace_id},
            exc_info=True,
        )
        return {
            "analysis_result": "",
            "analysis_visualization": "",
            "error": traceback.format_exc(),
        }




async def aggregator_node(state: AgentState) -> Dict:
    """Aggregate results from all upstream agents into final_response."""
    trace_id = state.get("trace_id", "")
    logger.info(
        "aggregator_node: aggregating results",
        extra={"component": "orchestrator", "trace_id": trace_id},
    )

    intent = state.get("intent", "research")

    final_parts = []
    if intent == "research":
        report = state.get("research_report", "")
        final_parts.append(report if report else "Research completed but no report generated.")
    elif intent == "analysis":
        result = state.get("analysis_result", "")
        final_parts.append(result if result else "Analysis completed but no results generated.")
    else:
        final_parts.append(f"Processed request. Intent: {intent}")

    final_response = "\n\n".join(final_parts)

    writer = get_stream_writer()
    response_len = len(final_response)
    writer({
        "phase": "aggregating_done",
        "message": f"结果聚合完成 ({response_len} 字符)",
    })

    return {"final_response": final_response}


# ------------------------------------------------------------------
# Conditional Routing Functions
# ------------------------------------------------------------------

def route_by_intent(state: AgentState) -> Literal["review_pipeline_node", "agentic_research_node", "research_node", "analysis_node"]:
    """Conditional edge: route based on intent + complexity + domain.

    审查任务 → review_pipeline_node（Map-Reduce，领域自适应）：
      - finance + 审查关键词（审查/审核/检查/核查）
      - contract + 审查关键词（审查/审核/检查/风险/分析/评估/合规）
      - law + 审查关键词
    research + complex → agentic_research_node (ReAct)
    research + simple  → research_node (fast path)
    analysis           → analysis_node
    """
    intent = state.get("intent", "research")
    complexity = state.get("complexity", "simple")
    domain = state.get("domain", "general")
    question = state.get("question", "") or state.get("raw_input", "")
    question_lower = question.lower()

    # ── 审查任务 → Map-Reduce（领域自适应） ──
    # 金融：显式审查关键词
    is_finance_review = (
        domain == "finance"
        and any(kw in question_lower for kw in _REVIEW_KEYWORDS)
    )
    # 合同/法律：审查关键词 OR 风险分析关键词 OR 有附件时默认审查
    is_contract_review = (
        domain == "contract"
        and any(kw in question_lower for kw in _REVIEW_KEYWORDS + _CONTRACT_REVIEW_KEYWORDS)
    )
    is_law_review = (
        domain == "law"
        and any(kw in question_lower for kw in _REVIEW_KEYWORDS + _CONTRACT_REVIEW_KEYWORDS)
    )
    # 未检测到领域但有附件 + 审查意图 → 通用 Map-Reduce
    has_attachment = "【附件：" in question
    is_generic_review = (
        domain == "general"
        and has_attachment
        and any(kw in question_lower for kw in _REVIEW_KEYWORDS + _CONTRACT_REVIEW_KEYWORDS)
    )
    if is_finance_review or is_contract_review or is_law_review or is_generic_review:
        return "review_pipeline_node"

    if intent == "research" and complexity == "complex":
        return "agentic_research_node"

    routing = {
        "research": "research_node",
        "analysis": "analysis_node",
    }
    return routing.get(intent, "research_node")


def after_human_confirm(state: AgentState) -> Literal["decomposer_node", END]:
    """After answer confirmation: if rejected, loop back to decomposer for rework."""
    if state.get("human_confirmed", False):
        return END
    # Rejected or needs rework → loop back
    return "decomposer_node"


# ------------------------------------------------------------------
# Graph Construction
# ------------------------------------------------------------------

async def build_graph() -> StateGraph:
    r"""Build and compile the LangGraph StateGraph (no HITL, straight-through).

    Graph structure::

        decomposer_node
            |
        knowledge_retriever_node
            |
        router_node
         /   |   |   \
    review research analysis customer_service
    (Map-Reduce)  |         |
         \   |   |   /
        aggregator_node
            |
           END
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("decomposer_node", decomposer_node)
    graph.add_node("knowledge_retriever_node", knowledge_retriever_node)
    graph.add_node("router_node", router_node)
    graph.add_node("review_pipeline_node", review_pipeline_node)
    graph.add_node("research_node", research_node)
    graph.add_node("agentic_research_node", agentic_research_node)
    graph.add_node("analysis_node", analysis_node)
    graph.add_node("aggregator_node", aggregator_node)

    # Entry
    graph.set_entry_point("decomposer_node")

    # decomposer → knowledge_retriever → router
    graph.add_edge("decomposer_node", "knowledge_retriever_node")
    graph.add_edge("knowledge_retriever_node", "router_node")

    # router → conditional routing: complexity-aware + review pipeline
    graph.add_conditional_edges(
        "router_node",
        route_by_intent,
        {
            "review_pipeline_node": "review_pipeline_node",
            "agentic_research_node": "agentic_research_node",
            "research_node": "research_node",
            "analysis_node": "analysis_node",

        },
    )

    # All agents → aggregator → END
    graph.add_edge("review_pipeline_node", "aggregator_node")
    graph.add_edge("agentic_research_node", "aggregator_node")
    graph.add_edge("research_node", "aggregator_node")
    graph.add_edge("analysis_node", "aggregator_node")
    graph.add_edge("aggregator_node", END)

    # Compile with async SQLite checkpointer (supports ainvoke)
    # Use direct connection to keep it alive beyond build_graph scope
    import os
    import aiosqlite
    db_path = config.path.checkpoints_db
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = await aiosqlite.connect(db_path)
    checkpointer = AsyncSqliteSaver(conn)
    compiled = graph.compile(checkpointer=checkpointer)

    return compiled


# ------------------------------------------------------------------
# Graph factory (call build_graph() each time for async_to_sync safety)
# ------------------------------------------------------------------


async def get_graph() -> StateGraph:
    """Get a compiled graph instance. Creates fresh per call for async safety."""
    return await build_graph()
