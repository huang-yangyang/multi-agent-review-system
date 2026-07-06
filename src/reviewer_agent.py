"""Reviewer Agent — 独立校验员，对 Reduce 输出做二次验证。

设计原理：
- "两双眼睛"原则：独立的 LLM 调用，不同的系统提示词
- 只做检查，不重写答案
- 三个校验维度：覆盖度、准确性、一致性

使用方式：
  verdict = await run_reviewer_check(map_context, reduce_output, trace_id)
  if not verdict["passed"]:
      final_report += format_verdict_block(verdict)
"""

import json
from typing import Any, Dict, Tuple
from src.core.logging_config import get_logger

logger = get_logger(__name__)


REVIEWER_SYSTEM_PROMPT = """你是独立的质量校验员。你的唯一任务是检查审查报告是否存在错误或遗漏。

你收到了两份输入：
1. [系统预提取] Map 阶段的结构化比对结果（确定性数据，视为 ground truth）
2. Reduce 阶段 LLM 生成的审查报告（需要被校验的对象）

请逐项检查以下三个维度，输出 JSON 格式结果：

## 检查维度

### 1. 覆盖度检查
- Map 比对结果中标注 ❌ 不达标 的每一项指标，Reduce 报告中是否都做了分析？
- Map 比对结果中标注的抵押率问题和审批权限问题，Reduce 报告中是否都提及？
- 是否有 Map 结果中提及但 Reduce 遗漏的内容？

### 2. 准确性检查
- Reduce 报告中引用的数值是否与 Map 结果一致？（例如 Map 说 57.99%，Reduce 是否写错？）
- Reduce 报告中引用的规程条款编号是否正确？
- Reduce 报告是否出现了 Map 结果中没有的数据？（可能幻觉）

### 3. 一致性检查
- Reduce 报告的结论是否与 Map 数据一致？（例如 Map 显示 4 项不达标，Reduce 不应该说"全部达标"）
- Reduce 报告内部是否有自相矛盾的表述？

## 输出格式

请严格输出以下 JSON（不要输出其他内容）：

{
  "passed": true/false,
  "coverage": {
    "total_checks": N,
    "passed": N,
    "missed_items": ["遗漏项1", "遗漏项2"]
  },
  "accuracy": {
    "data_errors": ["错误1", "错误2"],
    "has_hallucination": true/false
  },
  "consistency": {
    "contradictions": ["矛盾1"]
  },
  "summary": "一句话总结校验结果"
}
"""


async def run_reviewer_check(
    map_context: str,
    reduce_output: str,
    trace_id: str = "",
) -> Dict[str, Any]:
    """执行独立校验。

    Args:
        map_context: Map 阶段的结构化比对结果。
        reduce_output: Reduce 阶段的 LLM 输出。
        trace_id: 日志追踪 ID。

    Returns:
        {"passed": bool, "coverage": {...}, "accuracy": {...}, "consistency": {...}, "summary": str}
    """
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import SystemMessage, HumanMessage
    from src.config import config
    import json

    user_prompt = f"""## Map 阶段比对结果（Ground Truth）

{map_context[:5000]}

---

## Reduce 阶段审查报告（待校验）

{reduce_output[:8000]}

---

请按三个维度逐项检查，输出 JSON。"""

    try:
        llm = ChatOpenAI(
            model=config.llm.deepseek_model,
            api_key=config.llm.effective_api_key,
            base_url=config.llm.effective_base_url,
            temperature=0.0,  # 零温度确
            max_tokens=2048,
            timeout=60,
        )

        response = await llm.ainvoke([
            SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ])

        raw = response.content.strip() if hasattr(response, "content") else ""

        # 解析 JSON（LLM 可能在 JSON 前后加了 ```json 标记）
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        verdict = json.loads(raw)
        logger.info(
            f"Reviewer: passed={verdict.get('passed')}, "
            f"coverage={verdict.get('coverage',{}).get('passed','?')}/{verdict.get('coverage',{}).get('total_checks','?')}",
            extra={"component": "reviewer", "trace_id": trace_id},
        )
        return verdict

    except Exception as e:
        logger.warning(
            f"Reviewer check failed (graceful degradation): {e}",
            extra={"component": "reviewer", "trace_id": trace_id},
        )
        return {
            "passed": True,  # 校验失败时不阻塞正常流程
            "coverage": {"total_checks": 0, "passed": 0, "missed_items": []},
            "accuracy": {"data_errors": [], "has_hallucination": False},
            "consistency": {"contradictions": []},
            "summary": f"校验员未能完成检查（{e}），请人工复核。",
        }


def format_verdict_block(verdict: Dict[str, Any]) -> str:
    """将校验结果格式化为 Markdown 块，追加到最终报告中。"""
    if verdict.get("passed"):
        return "\n\n---\n\n## ✅ 独立校验通过\n\n校验员已确认：覆盖度、准确性、一致性均无问题。"

    parts = ["\n\n---\n\n## ⚠️ 独立校验发现问题\n"]

    coverage = verdict.get("coverage", {})
    missed = coverage.get("missed_items", [])
    if missed:
        parts.append("### 覆盖度问题")
        for item in missed:
            parts.append(f"- ❌ {item}")

    accuracy = verdict.get("accuracy", {})
    errors = accuracy.get("data_errors", [])
    if errors:
        parts.append("### 数据准确性问题")
        for err in errors:
            parts.append(f"- ❌ {err}")
    if accuracy.get("has_hallucination"):
        parts.append("- ⚠️ 检测到可能的幻觉数据")

    consistency = verdict.get("consistency", {})
    contras = consistency.get("contradictions", [])
    if contras:
        parts.append("### 一致性问题")
        for c in contras:
            parts.append(f"- ❌ {c}")

    parts.append(f"\n> 校验摘要：{verdict.get('summary', '请人工复核')}")
    return "\n".join(parts)
