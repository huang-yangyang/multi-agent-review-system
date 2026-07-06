"""审查质量自动评估框架。

评估维度（对应 D1-D6）：
  D1: 量化指标覆盖度 — 7 项指标是否全部出现
  D2: 抵押率正确性 — 是否检出 60% vs 50%
  D3: 审批权限核查 — 是否检出越级
  D4: 基础指标完整性 — 是否有遗漏
  D5: 来源分区正确性 — 📄/🤖 归属是否正确
  D6: 禁用模糊词汇 — 是否出现"表面上看/似乎/可能/大概"

适用场景：
  - CI 回归测试：每次代码提交自动运行
  - 新 prompt 版本 A/B 对比
  - 新领域审查质量基线
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class EvalResult:
    """单次评估结果。"""
    label: str                     # 评估标签
    passed: bool
    score: float                   # 0.0 - 1.0
    details: List[str] = field(default_factory=list)


@dataclass
class ReviewEvalReport:
    """完整评估报告。"""
    total_checks: int
    passed_checks: int
    pass_rate: float
    d1_d6_results: List[EvalResult] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 评估函数
# ═══════════════════════════════════════════════════════════════

FINANCE_INDICATORS = [
    "资产负债率", "流动比率", "速动比率", "EBITDA利息保障倍数",
    "近三年营收复合增长率", "经营现金流/流动负债", "有息债务/EBITDA",
]

BANNED_FUZZY = [
    r'表面上看', r'似乎', r'大概', r'貌似', r'或许', r'基本上是', r'大致上',
]


def eval_d1_indicator_coverage(output: str) -> EvalResult:
    """D1: 检查 7 项指标是否全部出现在输出中。"""
    found = []
    missing = []
    for ind in FINANCE_INDICATORS:
        if ind in output:
            found.append(ind)
        else:
            missing.append(ind)

    score = len(found) / len(FINANCE_INDICATORS)
    return EvalResult(
        label="D1: 量化指标覆盖度",
        passed=score >= 1.0,
        score=score,
        details=[f"找到 {len(found)}/7: {', '.join(found)}"] +
                ([f"缺失: {', '.join(missing)}"] if missing else []),
    )


def eval_d2_collateral_check(output: str) -> EvalResult:
    """D2: 检查是否检出抵押率 60% vs 规程 50% 的问题。"""
    has_60 = "60%" in output
    has_50 = "50%" in output
    has_noncompliant = "不合规" in output or "不达标" in output

    score = (has_60 + has_50 + has_noncompliant) / 3.0
    return EvalResult(
        label="D2: 抵押率正确性",
        passed=score >= 0.66,
        score=score,
        details=[
            f"提及 60%: {'✅' if has_60 else '❌'}",
            f"提及 50%: {'✅' if has_50 else '❌'}",
            f"判定不合规: {'✅' if has_noncompliant else '❌'}",
        ],
    )


def eval_d3_approval_check(output: str) -> EvalResult:
    """D3: 检查是否核查了审批权限层级。"""
    has_level = bool(re.search(r'(?:三级|省行|审批层级|审批权限)', output))
    has_exposure = bool(re.search(r'(?:4\.2|5\.2|敞口)', output))

    score = (has_level + has_exposure) / 2.0
    return EvalResult(
        label="D3: 审批权限核查",
        passed=score >= 1.0,
        score=score,
        details=[
            f"提及审批层级: {'✅' if has_level else '❌'}",
            f"提及敞口金额: {'✅' if has_exposure else '❌'}",
        ],
    )


def eval_d4_indicator_completeness(output: str) -> EvalResult:
    """D4: 检查是否有结构化表格。"""
    has_table = "|" in output and ("指标" in output or "比对" in output)
    has_all_sections = all(
        kw in output for kw in ["抵押", "审批", "预警"]
    )

    score = (has_table * 0.6 + has_all_sections * 0.4)
    return EvalResult(
        label="D4: 基础指标完整性",
        passed=score >= 0.6,
        score=score,
        details=[
            f"有结构化表格: {'✅' if has_table else '❌'}",
            f"覆盖关键章节: {'✅' if has_all_sections else '❌'}",
        ],
    )


def eval_d5_source_partition(output: str) -> EvalResult:
    """D5: 检查来源分区标记。"""
    has_doc_source = "\U0001F4C4" in output or "文档来源" in output
    # 检查量化指标是否在 📄 下（不在 🤖 下）
    ai_section_start = output.find("\U0001F916") if "\U0001F916" in output else output.find("AI 补充")
    doc_section_start = output.find("\U0001F4C4") if "\U0001F4C4" in output else output.find("文档来源")

    # 如果指标出现在 AI 补充段落之前（即在文档来源段落），则为正确
    indicators_in_right_section = True
    if ai_section_start > 0 and doc_section_start > 0:
        for ind in FINANCE_INDICATORS[:3]:
            ind_pos = output.find(ind)
            if ind_pos > 0 and ind_pos > ai_section_start and ind_pos < doc_section_start:
                indicators_in_right_section = False
                break

    score = (has_doc_source * 0.5 + indicators_in_right_section * 0.5)
    return EvalResult(
        label="D5: 来源分区正确性",
        passed=score >= 0.5,
        score=score,
        details=[
            f"有 📄 标记: {'✅' if has_doc_source else '❌'}",
            f"指标在正确段落: {'✅' if indicators_in_right_section else '⚠️'}",
        ],
    )


def eval_d6_no_fuzzy_words(output: str) -> EvalResult:
    """D6: 检查是否出现禁用模糊词汇。"""
    hits = []
    for pattern in BANNED_FUZZY:
        for match in re.finditer(pattern, output):
            ctx = output[max(0, match.start()-10):match.end()+10]
            hits.append(f"「{match.group()}」→ ...{ctx}...")

    score = 1.0 if not hits else max(0, 1.0 - len(hits) * 0.2)
    return EvalResult(
        label="D6: 禁用模糊词汇",
        passed=len(hits) == 0,
        score=score,
        details=[f"命中 {len(hits)} 个: {hits[:5]}"] if hits else ["未发现禁用词汇"],
    )


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def evaluate_review(review_output: str) -> ReviewEvalReport:
    """对审查输出执行完整的 D1-D6 评估。

    Args:
        review_output: Agent 或 Map-Reduce 产生的审查报告全文。

    Returns:
        ReviewEvalReport。
    """
    evals = [
        eval_d1_indicator_coverage(review_output),
        eval_d2_collateral_check(review_output),
        eval_d3_approval_check(review_output),
        eval_d4_indicator_completeness(review_output),
        eval_d5_source_partition(review_output),
        eval_d6_no_fuzzy_words(review_output),
    ]

    passed = sum(1 for e in evals if e.passed)
    return ReviewEvalReport(
        total_checks=len(evals),
        passed_checks=passed,
        pass_rate=passed / len(evals),
        d1_d6_results=evals,
    )


def compare_versions(
    old_output: str,
    new_output: str,
    old_label: str = "旧版本",
    new_label: str = "新版本",
) -> str:
    """对比两个版本的审查输出。

    Returns:
        Markdown 格式的对比报告。
    """
    old_report = evaluate_review(old_output)
    new_report = evaluate_review(new_output)

    lines = [
        "## 审查质量对比",
        "",
        f"| 维度 | {old_label} | {new_label} | 变化 |",
        "|------|:---:|:---:|:---:|",
    ]

    for old_e, new_e in zip(old_report.d1_d6_results, new_report.d1_d6_results):
        old_icon = "✅" if old_e.passed else "❌"
        new_icon = "✅" if new_e.passed else "❌"
        change = "→ ✅ 修复" if (not old_e.passed and new_e.passed) else \
                 ("→ ❌ 退步" if (old_e.passed and not new_e.passed) else "—")
        lines.append(f"| {old_e.label} | {old_icon} ({old_e.score:.0%}) | {new_icon} ({new_e.score:.0%}) | {change} |")

    lines.append("")
    lines.append(f"**{old_label}**: {old_report.passed_checks}/{old_report.total_checks} ({old_report.pass_rate:.0%})")
    lines.append(f"**{new_label}**: {new_report.passed_checks}/{new_report.total_checks} ({new_report.pass_rate:.0%})")

    return "\n".join(lines)
