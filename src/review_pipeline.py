"""Map-Reduce 审查管线 — 多领域支持（授信报告 + 合同 + 法律）。

架构：
  Map 阶段（确定性代码）  →  Reduce 阶段（单次 LLM 推理，非 ReAct）
  ─────────────────────     ─────────────────────────────────────
  金融：7项指标提取+规程比对   基于 Map 结果的定性深度分析
  合同：条款提取+风险模式匹配  逐条语义确认+风险分析+修改建议
  法律：法规条文提取+要件比对  法律适用性分析+风险评估

核心设计原则：
1. Map 是穷举保证层（代码级），不依赖 LLM
2. Reduce 是深度推理层（LLM），只做判断不做检索
3. Map 失败时 Reduce 仍可独立运行
4. 领域自动路由：根据 domain 标签选择对应的 Map 提取器和 Reduce prompt
"""

import re
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from src.core.logging_config import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Map 阶段（复用 review_extractor 的确定性提取）
# ══════════════════════════════════════════════════════════════════════════════

async def run_map_phase(
    question: str,
    procedure_text: str,
    trace_id: str = "",
) -> Tuple[Optional[str], Optional[Dict[str, Any]], str]:
    """Map 阶段：确定性提取 + 结构化比对。

    从用户问题中解析附件（报告），结合规程文本，执行代码级指标提取和比对。

    Args:
        question: 用户问题文本（含附件块）。
        procedure_text: 操作规程全文。
        trace_id: 日志追踪 ID。

    Returns:
        (map_context, map_result, error_message)
        - map_context: 格式化后的比对结果 Markdown，失败为 None
        - map_result: ReviewExtractionResult 的 dict，失败为 None
        - error_message: 空字符串表示成功
    """
    from src.review_extractor import extract_all, format_context
    from src.workflows.orchestrator import _parse_attachment_text

    try:
        # 1. 解析附件（报告全文）
        att_name, report_text, _ = _parse_attachment_text(question)
        if not att_name or not report_text:
            logger.info(
                "Map phase: no attachment in question, skipping",
                extra={"component": "review_pipeline", "trace_id": trace_id},
            )
            return None, None, "未在问题中找到附件文档"

        logger.info(
            f"Map phase: attachment '{att_name}', {len(report_text)} chars",
            extra={"component": "review_pipeline", "trace_id": trace_id},
        )

        # 2. 执行确定性提取
        result = extract_all(report_text, procedure_text)

        if result.extraction_failed:
            logger.warning(
                f"Map phase: extraction failed: {result.extraction_warnings}",
                extra={"component": "review_pipeline", "trace_id": trace_id},
            )
            return None, None, f"指标提取失败: {'; '.join(result.extraction_warnings)}"

        if not result.financial_indicators:
            logger.warning(
                "Map phase: no indicators extracted",
                extra={"component": "review_pipeline", "trace_id": trace_id},
            )
            return None, None, "未能从报告中提取到任何财务指标"

        # 3. 格式化输出
        map_context = format_context(result)

        # 4. 转 dict 供 Reduce 使用
        map_result = {
            "company_name": result.company_name,
            "target_rating": result.target_rating,
            "indicator_count": len(result.financial_indicators),
            "comparison_count": len(result.indicator_comparisons),
            "passed_count": sum(1 for c in result.indicator_comparisons if c.verdict == "达标"),
            "failed_count": sum(1 for c in result.indicator_comparisons if c.verdict == "不达标"),
            "failed_indicators": [
                {
                    "name": c.indicator_name,
                    "report_value": c.report_value,
                    "threshold": c.threshold_text,
                    "gap": c.gap_display,
                }
                for c in result.indicator_comparisons if c.verdict == "不达标"
            ],
            "collateral": {
                "is_compliant": result.collateral_check.is_compliant if result.collateral_check else None,
                "report_rate": result.collateral_check.report_rate_pct if result.collateral_check else None,
                "procedure_rate": result.collateral_check.procedure_rate_pct if result.collateral_check else None,
                "report_guarantee": (result.collateral_check.report_guarantee_wan / 10000) if result.collateral_check else None,
                "procedure_guarantee": (result.collateral_check.procedure_guarantee_wan / 10000) if result.collateral_check else None,
            } if result.collateral_check else None,
            "approval": {
                "is_compliant": result.approval_check.is_compliant if result.approval_check else None,
                "branch": result.approval_check.branch if result.approval_check else "",
                "exposure": result.approval_check.exposure_text if result.approval_check else "",
                "correct_level": result.approval_check.correct_level if result.approval_check else "",
            } if result.approval_check else None,
            "warnings": [
                {"name": w.signal_name, "level": w.warning_level, "evidence": w.evidence}
                for w in result.warning_flags
            ],
            "extraction_warnings": result.extraction_warnings,
        }

        logger.info(
            f"Map phase: done — {map_result['indicator_count']} indicators, "
            f"{map_result['passed_count']} pass, {map_result['failed_count']} fail, "
            f"{len(result.warning_flags)} warnings",
            extra={"component": "review_pipeline", "trace_id": trace_id},
        )
        return map_context, map_result, ""

    except Exception as e:
        logger.error(
            f"Map phase: unexpected error: {e}",
            extra={"component": "review_pipeline", "trace_id": trace_id},
            exc_info=True,
        )
        return None, None, str(e)


# ══════════════════════════════════════════════════════════════════════════════
# Reduce 阶段（单次 LLM 调用，非 ReAct，纯推理）
# ══════════════════════════════════════════════════════════════════════════════

REDUCE_SYSTEM_PROMPT = """你是资深信贷审批官，专责对授信调查报告进行终审审查。

## 工作模式

系统已经完成了 Map 阶段的确定性比对（代码级逐项提取），你收到的上下文开头包含「[系统预提取] 量化指标自动比对结果」。
**你不需要重新检索或提取数据**——所有量化指标、规程阈值、比对结论已由系统预先计算。

## 你的任务

基于系统预提取的比对结果，完成以下深度分析：

### 一、总体评价（1-2 段）
- 对报告质量给出总体评价
- 明确指出现有评级（如 AA）是否成立，若指标大面积不达标应直接给出降级建议
- 结论只用"达标/不达标/合规/不合规/建议降级至X级"

### 二、不达标指标逐项分析
对每个 ❌ 不达标 的指标：
1. 说明不达标的程度（差值/偏离幅度）
2. 分析可能的成因（结合报告中的经营信息）
3. 评估对偿债能力和信用风险的影响
4. 给出是否影响评级的明确结论

### 三、抵押率合规分析
- 明确指出报告使用的抵押率与规程上限的差异
- 以规程规定的抵押率上限重新确认可担保额度
- 评估抵押率不合规对担保方案有效性的影响
- 给出整改建议

### 四、审批权限核查
- 明确指出现有审批层级是否具备审批权限
- 若越权，明确应上提至哪一级
- 结合首次授信或评级下調等触发条件，判断是否需要进一步上提

### 五、贷后预警与处置
- 对已触发的预警信号逐条给出处置建议（引用规程具体条款）
- 评估是否触发红色预警
- 给出贷后管理频率建议（对照规程表6.2）

### 六、定性维度补充
在系统预提取的量化指标之外，对以下定性维度进行审查：
- 行业风险（产能过剩、政策变化等）
- 经营风险（客户集中度、供应商依赖等）
- 关联交易风险（关联互保折算、资金挪用风险等）
- 资金用途合规性（流贷是否被挪用至固投）

### 七、总体建议
- 综合评级结论
- 授信方案修改建议
- 补充材料清单
- 是否同意按现有方案审批（同意/有条件同意/退回补充/否决）

## 输出格式要求

1. 使用 Markdown 标题层级组织，从「## 一、总体评价」开始
2. 所有量化判断必须引用系统预提取数据（格式：「系统提取：资产负债率 57.99%，规程 AA 上限 55%」）
3. 来源分区：基于规程条款/表格/数值的判断 → 📄 文档来源；纯经验常识补充 → 🤖 AI 补充
4. 禁止使用"表面上看/似乎/可能/大概/貌似/或许"等模糊词汇
5. 审查结论只用"达标/不达标/合规/不合规"

## 输入说明

系统已提供：
1. [系统预提取] 量化指标自动比对结果（Map 阶段输出）
2. 操作规程原文
3. 授信调查报告原文

请开始审查。
"""


def _build_reduce_user_prompt(
    map_context: str,
    report_text: str,
    procedure_text: str,
) -> str:
    """构建 Reduce 阶段的 user prompt。

    将 Map 结果、报告原文、规程原文组装为结构化输入。
    为控制 token 消耗，报告和规程各截取前 8000 字符。
    """
    report_section = report_text[:8000]
    if len(report_text) > 8000:
        report_section += "\n\n... (报告原文已截断，完整内容见附件)"

    procedure_section = procedure_text[:8000]
    if len(procedure_text) > 8000:
        procedure_section += "\n\n... (规程原文已截断，完整内容见附件)"

    prompt = f"""{map_context}

---

## 授信调查报告原文

{report_section}

---

## 信用风险评估操作规程原文

{procedure_section}

---

请基于以上系统预提取比对结果、报告原文和规程原文，按照你的任务要求（总体评价→不达标指标分析→抵押率合规→审批权限→贷后预警→定性维度→总体建议），逐项输出审查意见。
"""
    return prompt


async def run_reduce_phase(
    map_context: str,
    report_text: str,
    procedure_text: str,
    trace_id: str = "",
) -> str:
    """Reduce 阶段：单次 LLM 调用，不做工具调用。

    将所有 Map 结果和原始文档喂给 LLM，让其进行深度定性分析。

    Args:
        map_context: Map 阶段输出的格式化比对结果。
        report_text: 报告全文。
        procedure_text: 规程全文。
        trace_id: 日志追踪 ID。

    Returns:
        LLM 生成的完整审查报告。
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.config import config

    user_prompt = _build_reduce_user_prompt(map_context, report_text, procedure_text)

    try:
        llm = ChatOpenAI(
            model=config.llm.deepseek_model,
            api_key=config.llm.effective_api_key,
            base_url=config.llm.effective_base_url,
            temperature=0.2,
            max_tokens=8192,
            timeout=120,
        )

        messages = [
            SystemMessage(content=REDUCE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = await llm.ainvoke(messages)
        content = response.content.strip() if hasattr(response, "content") else str(response).strip()

        logger.info(
            f"Reduce phase: LLM generated {len(content)} chars",
            extra={"component": "review_pipeline", "trace_id": trace_id},
        )
        return content

    except Exception as e:
        logger.error(
            f"Reduce phase: LLM call failed: {e}",
            extra={"component": "review_pipeline", "trace_id": trace_id},
            exc_info=True,
        )
        # 降级：返回 Map 结果 + 错误说明
        return (
            f"{map_context}\n\n---\n"
            f"## ⚠️ Reduce 阶段执行失败\n\n"
            f"LLM 深度分析调用失败（{e}），以上为系统自动提取的量化比对结果。"
            f"请人工进行定性分析。"
        )


async def run_reduce_phase_stream(
    map_context: str,
    report_text: str,
    procedure_text: str,
    trace_id: str = "",
) -> AsyncIterator[str]:
    """Reduce 阶段流式版本：逐 token 输出 LLM 推理结果。

    用于前端实时展示审查生成过程。
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.config import config

    user_prompt = _build_reduce_user_prompt(map_context, report_text, procedure_text)

    try:
        llm = ChatOpenAI(
            model=config.llm.deepseek_model,
            api_key=config.llm.effective_api_key,
            base_url=config.llm.effective_base_url,
            temperature=0.2,
            max_tokens=8192,
            timeout=120,
        )

        messages = [
            SystemMessage(content=REDUCE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        async for chunk in llm.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    except Exception as e:
        logger.error(
            f"Reduce phase stream: LLM call failed: {e}",
            extra={"component": "review_pipeline", "trace_id": trace_id},
            exc_info=True,
        )
        fallback = (
            f"\n\n---\n"
            f"## ⚠️ Reduce 阶段执行失败\n\n"
            f"LLM 深度分析调用失败（{e}），以上为系统自动提取的量化比对结果。"
        )
        yield fallback


# ══════════════════════════════════════════════════════════════════════════════
# 降级路径：Map 失败时的纯 LLM 审查（非 ReAct，单次调用）
# ══════════════════════════════════════════════════════════════════════════════

FALLBACK_SYSTEM_PROMPT = """你是资深信贷审批官。系统未能完成自动指标提取，请你基于报告和规程原文进行完整审查。

你必须逐项完成以下工作（不可跳过任何一项）：
1. 从报告中提取全部财务量化指标（资产负债率、流动比率、速动比率、EBITDA利息保障倍数、近三年营收复合增长率、经营现金流/流动负债、有息债务/EBITDA）
2. 从规程表2.2中查找对应评级的阈值
3. 将每项指标与规程阈值逐项比对，以表格形式呈现
4. 检查抵押率是否与规程表5.2一致
5. 核查审批权限是否在规程第4节规定的层级内
6. 检测贷后预警信号（规程表6.1）
7. 完成定性维度分析

输出使用 Markdown 格式，包含量化指标对照表。
所有基于规程的判断归入 📄 文档来源。
严禁使用"表面上看/似乎/可能/大概/貌似/或许"等模糊词汇。
"""


async def run_fallback_review(
    report_text: str,
    procedure_text: str,
    trace_id: str = "",
) -> str:
    """Map 阶段失败时的降级路径：纯 LLM 审查（单次调用，非 ReAct）。

    与 ReAct 路径的关键区别：
    - 不提供工具，不进行多轮循环
    - 一次性接收全部文档内容
    - 要求 LLM 按结构化模板输出
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.config import config

    user_prompt = f"""## 授信调查报告

{report_text[:10000]}

---

## 信用风险评估操作规程

{procedure_text[:10000]}

---

请按照你的任务要求，对上述授信报告进行全面审查。必须逐项完成量化指标提取与比对、抵押率检查、审批权限核查、贷后预警检测。
"""

    try:
        llm = ChatOpenAI(
            model=config.llm.deepseek_model,
            api_key=config.llm.effective_api_key,
            base_url=config.llm.effective_base_url,
            temperature=0.2,
            max_tokens=8192,
            timeout=120,
        )

        response = await llm.ainvoke([
            SystemMessage(content=FALLBACK_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        return response.content.strip() if hasattr(response, "content") else str(response).strip()

    except Exception as e:
        logger.error(
            f"Fallback review: LLM call failed: {e}",
            extra={"component": "review_pipeline", "trace_id": trace_id},
            exc_info=True,
        )
        return f"审查服务暂时不可用（{e}），请稍后重试。"


# ══════════════════════════════════════════════════════════════════════════════
# 主入口：Map-Reduce 审查管线
# ══════════════════════════════════════════════════════════════════════════════

async def execute_review_pipeline(
    question: str,
    procedure_text: str,
    trace_id: str = "",
) -> Tuple[str, bool]:
    """执行完整的 Map-Reduce 审查管线。

    Args:
        question: 用户问题（含附件块）。
        procedure_text: 操作规程全文。
        trace_id: 日志追踪 ID。

    Returns:
        (final_report, map_succeeded)
        - final_report: 完整的审查报告（Map 结果 + Reduce 分析）
        - map_succeeded: Map 阶段是否成功
    """
    # ── 解析报告文本（独立于 Map 阶段，供 Reduce 和降级路径使用） ──
    from src.workflows.orchestrator import _parse_attachment_text
    _, report_text, _ = _parse_attachment_text(question)

    if not report_text:
        return "未能从问题中提取报告文本，请确认已上传授信调查报告。", False

    # ── Map 阶段 ──
    logger.info(
        "Review pipeline: starting Map phase",
        extra={"component": "review_pipeline", "trace_id": trace_id},
    )
    map_context, map_result, map_error = await run_map_phase(
        question, procedure_text, trace_id,
    )

    if map_context and map_result:
        logger.info(
            f"Review pipeline: Map succeeded "
            f"({map_result['indicator_count']} indicators, "
            f"{map_result['failed_count']} failed)",
            extra={"component": "review_pipeline", "trace_id": trace_id},
        )

        # ── Reduce 阶段 ──
        logger.info(
            "Review pipeline: starting Reduce phase",
            extra={"component": "review_pipeline", "trace_id": trace_id},
        )
        reduce_output = await run_reduce_phase(
            map_context, report_text, procedure_text, trace_id,
        )

        # ── 校验员：独立 LLM 二次验证 ──
        logger.info("Review pipeline: starting Reviewer check", extra={"component": "review_pipeline", "trace_id": trace_id})
        try:
            from src.reviewer_agent import run_reviewer_check, format_verdict_block
            verdict = await run_reviewer_check(map_context, reduce_output, trace_id)
            reviewer_block = format_verdict_block(verdict)
        except Exception as e:
            logger.warning(f"Reviewer unavailable: {e}")
            reviewer_block = ""

        # 组装最终报告：Map 结果 + Reduce 分析 + 校验结果
        final_report = map_context + "\n\n---\n\n" + reduce_output + reviewer_block
        return final_report, True

    else:
        # ── Map 失败 → 降级路径 ──
        logger.warning(
            f"Review pipeline: Map failed ({map_error}), falling back to LLM-only review",
            extra={"component": "review_pipeline", "trace_id": trace_id},
        )
        fallback_output = await run_fallback_review(
            report_text, procedure_text, trace_id,
        )
        return fallback_output, False


# ══════════════════════════════════════════════════════════════════════════════
# 合同审查 Map-Reduce 管线
# ══════════════════════════════════════════════════════════════════════════════

CONTRACT_REDUCE_SYSTEM_PROMPT = """你是资深合同审核律师，专责对技术/服务合同进行风险审查。

## 工作模式

系统已完成 Map 阶段的条款提取和风险模式预匹配，你收到的上下文开头包含「[系统预提取] 合同条款与风险模式预匹配结果」。
**你不需要重新搜索或检索**——全部合同条款和知识库风险模式已在预匹配结果中列出。

## 你的任务

基于系统预提取的预匹配结果、合同原文和知识库原文，完成以下审查：

### 一、总体评价（1-2段）
- 对合同的整体公平性给出评价（甲方/乙方视角）
- 指出合同中风险最集中的领域
- 给出整体风险等级（高/中/低）

### 二、逐条风险确认与分析
对系统预匹配结果中的**每一个风险模式**，逐条进行最终语义判定：
1. **确认是否真正匹配**：预匹配说"可能匹配"的 → 你读取合同原文后判定 YES/NO
2. **匹配的** → 给出：原文定位 + 风险分析 + 修改建议
3. **不匹配的** → 一句话说明为什么不适用
4. **预匹配标"未匹配"的** → 快速扫描合同，确认没有隐藏风险后标注"确认无风险"或"发现隐藏风险 → 展开分析"

### 三、补充风险发现
系统预匹配可能遗漏的风险（基于你通读合同全文后的判断），在对应章节补充。

### 四、修改建议优先级排序
将所有风险按严重程度排序（严重/重要/一般），给出谈判优先级建议。

### 五、附件缺失提示
检查合同引用的附件是否完备，给出补充建议。

## 输出格式

1. 使用 Markdown 标题层级组织
2. 每个风险点必须标注：📄 文档来源（基于合同原文/知识库条款）或 🤖 AI 补充
3. 禁止使用模糊词汇
4. 结论明确：适用/不适用/需人工确认
"""


async def execute_contract_review_pipeline(
    question: str,
    kb_text: str,
    trace_id: str = "",
) -> Tuple[str, bool]:
    """执行合同审查 Map-Reduce 管线。

    Args:
        question: 用户问题（含附件块 — 合同全文）。
        kb_text: 合同风险知识库全文。
        trace_id: 日志追踪 ID。

    Returns:
        (final_report, map_succeeded)
    """
    from src.workflows.orchestrator import _parse_attachment_text
    from src.contract_review_extractor import (
        run_contract_map, format_contract_map_context,
    )

    # ── 解析合同文本 ──
    _, contract_text, _ = _parse_attachment_text(question)
    if not contract_text:
        return "未能从问题中提取合同文本，请确认已上传合同文件。", False

    # ── Map 阶段：条款提取 + 风险模式预匹配 ──
    logger.info(
        "Contract review pipeline: starting Map phase",
        extra={"component": "review_pipeline", "trace_id": trace_id},
    )

    map_result = run_contract_map(contract_text, kb_text)

    if not map_result.success:
        logger.warning(
            f"Contract Map failed: {map_result.extraction_warnings}",
            extra={"component": "review_pipeline", "trace_id": trace_id},
        )
        # 降级：纯 LLM 审查
        fallback = await run_fallback_review(contract_text, kb_text, trace_id)
        return fallback, False

    map_context = format_contract_map_context(map_result)

    logger.info(
        f"Contract Map: {map_result.total_clauses} clauses, "
        f"{map_result.total_patterns} patterns, "
        f"{map_result.patterns_with_match} matched",
        extra={"component": "review_pipeline", "trace_id": trace_id},
    )

    # ── Reduce 阶段 ──
    logger.info(
        "Contract review pipeline: starting Reduce phase",
        extra={"component": "review_pipeline", "trace_id": trace_id},
    )

    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.config import config

    user_prompt = f"""{map_context}

---

## 合同原文

{contract_text[:10000]}

---

## 合同风险知识库原文

{kb_text[:8000]}

---

请基于以上系统预提取的预匹配结果、合同原文和知识库原文，逐条确认每个风险模式是否真正适用，对适用的模式给出详细风险分析和修改建议。按「总体评价→逐条确认→补充风险→优先级排序→附件提示」的顺序输出。
"""

    try:
        llm = ChatOpenAI(
            model=config.llm.deepseek_model,
            api_key=config.llm.effective_api_key,
            base_url=config.llm.effective_base_url,
            temperature=0.2,
            max_tokens=8192,
            timeout=120,
        )

        response = await llm.ainvoke([
            SystemMessage(content=CONTRACT_REDUCE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])
        reduce_output = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # ── 校验员 ──
        try:
            from src.reviewer_agent import run_reviewer_check, format_verdict_block
            verdict = await run_reviewer_check(map_context, reduce_output, trace_id)
            reviewer_block = format_verdict_block(verdict)
        except Exception:
            reviewer_block = ""

        final_report = map_context + "\n\n---\n\n" + reduce_output + reviewer_block
        return final_report, True

    except Exception as e:
        logger.error(
            f"Contract Reduce failed: {e}",
            extra={"component": "review_pipeline", "trace_id": trace_id},
            exc_info=True,
        )
        return map_context + f"\n\n---\n\n## ⚠️ Reduce 阶段失败\n\nLLM 分析调用失败（{e}），以上为系统自动提取的条款匹配结果。", True


# ══════════════════════════════════════════════════════════════════════════════
# 劳动法审查 Map-Reduce 管线
# ══════════════════════════════════════════════════════════════════════════════

LABOR_REDUCE_SYSTEM_PROMPT = """你是资深劳动法律师，专责对劳动合同进行审查。

## 工作模式

系统已完成 Map 阶段的条款提取和风险模式预匹配，你收到的上下文开头包含「[系统预提取] 劳动合同条款与劳动法风险模式预匹配结果」。
**你不需要重新搜索或检索**——全部合同条款和劳动法知识库风险模式已在预匹配结果中列出。

## 你的任务

基于系统预提取的预匹配结果、合同原文和知识库原文，完成以下审查：

### 一、总体评价（1-2段）
- 对合同的整体公平性给出评价（劳动者视角）
- 指出合同中最严重的问题
- 给出整体风险等级（高/中/低）

### 二、逐条风险确认与分析
对系统预匹配结果中的**每一个风险模式**，逐条进行最终语义判定：
1. 确认是否真正匹配
2. 匹配的 → 给出：原文引用 + 法律依据（引用具体法条）+ 风险分析 + 修改建议
3. 不匹配的 → 一句话说明原因
4. 预匹配标"未匹配"的 → 快速扫描合同确认

### 三、条款合法性判定
对以下关键条款逐条判定：
- 试用期是否超法定上限
- 违约金是否超出法定范围
- 竞业限制是否约定补偿金
- 解除条件是否合法
- 社保公积金是否依法缴纳

### 四、风险优先级排序
按严重程度排序（🔴违法 / 🟡对劳动者不利 / 🟢建议优化）

### 五、行动建议
- 哪些条款签约前必须修改
- 哪些条款即使签署也可在仲裁中主张无效
- 签约注意事项

## 输出格式

1. 每个风险点须标注：📄 文档来源 + 引用法条
2. 禁止使用模糊词汇
3. 结论明确：合法/不合法/涉嫌违法/建议修改
"""


async def execute_labor_review_pipeline(
    question: str,
    kb_text: str,
    trace_id: str = "",
) -> Tuple[str, bool]:
    """执行劳动法审查 Map-Reduce 管线。"""
    from src.workflows.orchestrator import _parse_attachment_text
    from src.labor_map import (
        run_labor_map, format_labor_map_context,
    )

    _, contract_text, _ = _parse_attachment_text(question)
    if not contract_text:
        return "未能从问题中提取合同文本", False

    logger.info("Labor Map phase starting", extra={"component": "review_pipeline", "trace_id": trace_id})
    map_result = run_labor_map(contract_text, kb_text)

    if not map_result.success:
        logger.warning(f"Labor Map failed: {map_result.extraction_warnings}")
        fallback = await run_fallback_review(contract_text, kb_text, trace_id)
        return fallback, False

    map_context = format_labor_map_context(map_result)
    logger.info(f"Labor Map: {map_result.total_clauses} clauses, {map_result.total_patterns} patterns, {map_result.patterns_with_match} matched")

    # Reduce
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.config import config

    user_prompt = f"""{map_context}

---

## 合同原文

{contract_text[:10000]}

---

## 劳动法知识库原文

{kb_text[:8000]}

---

请基于以上系统预提取的预匹配结果，逐条确认每个风险模式是否真正适用，按「总体评价→逐条确认→合法性判定→优先级排序→行动建议」的顺序输出。
"""

    try:
        llm = ChatOpenAI(model=config.llm.deepseek_model, api_key=config.llm.effective_api_key,
                         base_url=config.llm.effective_base_url, temperature=0.2, max_tokens=8192)
        response = await llm.ainvoke([SystemMessage(content=LABOR_REDUCE_SYSTEM_PROMPT), HumanMessage(content=user_prompt)])
        reduce_output = response.content.strip() if hasattr(response, "content") else ""
        # ── 校验员 ──
        try:
            from src.reviewer_agent import run_reviewer_check, format_verdict_block
            verdict = await run_reviewer_check(map_context, reduce_output, trace_id)
            reviewer_block = format_verdict_block(verdict)
        except Exception:
            reviewer_block = ""
        return map_context + "\n\n---\n\n" + reduce_output + reviewer_block, True
    except Exception as e:
        logger.error(f"Labor Reduce failed: {e}")
        return map_context + f"\n\n---\n\n## ⚠️ Reduce 阶段失败\n\n{e}", True
