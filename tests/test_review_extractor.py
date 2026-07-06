"""金融审查 Map 提取器 — 单元测试 + 回归测试。

测试覆盖：
- 7 项指标提取准确性
- 规程阈值解析完整性
- 比对判定逻辑正确性
- 抵押率/审批/预警检测
- 降级路径（空输入/无规程/无附件）
"""

import pytest
from pathlib import Path

# 添加项目根目录到 path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.review_extractor import (
    extract_financial_indicators,
    extract_procedure_thresholds,
    extract_collateral_rates,
    extract_approval_levels,
    extract_all,
    format_context,
    detect_early_warnings,
    FinancialIndicator,
    ReviewExtractionResult,
)

# ── 测试数据 ──

@pytest.fixture
def report_text():
    path = Path(__file__).resolve().parent.parent / "output" / "宏达实业集团_授信调查报告.md"
    return path.read_text(encoding="utf-8")

@pytest.fixture
def procedure_text():
    path = Path(__file__).resolve().parent.parent / "output" / "信用风险评估操作规程.md"
    return path.read_text(encoding="utf-8")


# ═══════════════════════════════════════════════════════════════
# 指标提取测试
# ═══════════════════════════════════════════════════════════════

class TestIndicatorExtraction:
    """7 项财务指标的提取准确性。"""

    def test_extract_all_seven_indicators(self, report_text):
        indicators = extract_financial_indicators(report_text)
        names = {ind.name for ind in indicators}
        expected = {
            "资产负债率", "流动比率", "速动比率",
            "EBITDA利息保障倍数", "近三年营收复合增长率",
            "经营现金流/流动负债", "有息债务/EBITDA",
        }
        assert names == expected, f"缺失指标: {expected - names}"

    def test_asset_liability_ratio(self, report_text):
        indicators = extract_financial_indicators(report_text)
        ind = next(i for i in indicators if i.name == "资产负债率")
        assert ind.value == pytest.approx(57.99, abs=0.01)
        assert ind.unit == "%"

    def test_current_ratio(self, report_text):
        indicators = extract_financial_indicators(report_text)
        ind = next(i for i in indicators if i.name == "流动比率")
        assert ind.value == pytest.approx(1.49, abs=0.01)

    def test_quick_ratio(self, report_text):
        indicators = extract_financial_indicators(report_text)
        ind = next(i for i in indicators if i.name == "速动比率")
        assert ind.value == pytest.approx(1.05, abs=0.01)

    def test_ebitda_coverage(self, report_text):
        indicators = extract_financial_indicators(report_text)
        ind = next(i for i in indicators if i.name == "EBITDA利息保障倍数")
        assert ind.value == pytest.approx(2.88, abs=0.01)

    def test_revenue_growth(self, report_text):
        indicators = extract_financial_indicators(report_text)
        ind = next(i for i in indicators if i.name == "近三年营收复合增长率")
        assert ind.value == pytest.approx(18.1, abs=0.1)

    def test_operating_cashflow(self, report_text):
        indicators = extract_financial_indicators(report_text)
        ind = next(i for i in indicators if i.name == "经营现金流/流动负债")
        assert ind.value == pytest.approx(-0.065, abs=0.01)

    def test_debt_to_ebitda(self, report_text):
        indicators = extract_financial_indicators(report_text)
        ind = next(i for i in indicators if i.name == "有息债务/EBITDA")
        assert ind.value == pytest.approx(2.93, abs=0.01)


# ═══════════════════════════════════════════════════════════════
# 规程解析测试
# ═══════════════════════════════════════════════════════════════

class TestProcedureParsing:
    """规程阈值表和抵押率表的解析完整性。"""

    def test_all_seven_thresholds_parsed(self, procedure_text):
        thresholds = extract_procedure_thresholds(procedure_text)
        assert len(thresholds) == 7, f"预期 7 组阈值，实际 {len(thresholds)}"

    def test_aa_thresholds_correct(self, procedure_text):
        thresholds = extract_procedure_thresholds(procedure_text)
        aa = thresholds.get("资产负债率", {}).get("AA")
        assert aa is not None
        assert aa.threshold_value == 55.0
        assert aa.operator == "<="

        aa_cr = thresholds.get("流动比率", {}).get("AA")
        assert aa_cr is not None
        assert aa_cr.threshold_value == 1.8
        assert aa_cr.operator == ">="

    def test_collateral_rates(self, procedure_text):
        rates = extract_collateral_rates(procedure_text)
        assert "工业用地及厂房" in rates
        assert rates["工业用地及厂房"] == 50.0  # 关键：规程规定 50%

    def test_approval_levels(self, procedure_text):
        levels = extract_approval_levels(procedure_text)
        assert len(levels) == 5
        assert levels[1]["level"] == "二级"  # 分行贷审会


# ═══════════════════════════════════════════════════════════════
# 比对逻辑测试
# ═══════════════════════════════════════════════════════════════

class TestComparison:
    """逐项比对逻辑。"""

    def test_full_extraction_success(self, report_text, procedure_text):
        result = extract_all(report_text, procedure_text)
        assert result.success
        assert result.company_name == "宏达实业集团有限公司"
        assert result.target_rating == "AA"

    def test_comparison_count(self, report_text, procedure_text):
        result = extract_all(report_text, procedure_text)
        assert len(result.indicator_comparisons) == 7

    def test_failed_indicators(self, report_text, procedure_text):
        """验证 4 项不达标指标被正确识别。"""
        result = extract_all(report_text, procedure_text)
        failed = [c for c in result.indicator_comparisons if c.verdict == "不达标"]
        failed_names = {c.indicator_name for c in failed}
        assert "资产负债率" in failed_names
        assert "流动比率" in failed_names
        assert "EBITDA利息保障倍数" in failed_names
        assert "经营现金流/流动负债" in failed_names
        assert len(failed) == 4

    def test_passed_indicators(self, report_text, procedure_text):
        """验证 3 项达标指标被正确识别。"""
        result = extract_all(report_text, procedure_text)
        passed = [c for c in result.indicator_comparisons if c.verdict == "达标"]
        assert len(passed) == 3

    def test_collateral_noncompliant(self, report_text, procedure_text):
        """抵押率 60% vs 规程 50% → 不合规。"""
        result = extract_all(report_text, procedure_text)
        assert result.collateral_check is not None
        assert not result.collateral_check.is_compliant
        assert result.collateral_check.report_rate_pct == 60.0
        assert result.collateral_check.procedure_rate_pct == 50.0

    def test_approval_noncompliant(self, report_text, procedure_text):
        """二级分行 2亿上限 vs 敞口 4.2亿 → 不合规。"""
        result = extract_all(report_text, procedure_text)
        assert result.approval_check is not None
        assert not result.approval_check.is_compliant
        assert "三级" in result.approval_check.correct_level

    def test_warning_detected(self, report_text, procedure_text):
        result = extract_all(report_text, procedure_text)
        assert len(result.warning_flags) >= 1


# ═══════════════════════════════════════════════════════════════
# 降级路径测试
# ═══════════════════════════════════════════════════════════════

class TestDegradation:
    """优雅降级：空输入 / 无规程 / 无附件。"""

    def test_empty_input(self):
        result = extract_all("", "")
        assert not result.success
        assert result.extraction_failed

    def test_no_procedure(self, report_text):
        result = extract_all(report_text, "")
        assert result.success  # 指标仍能提取
        # 比对全部标为"规程中未找到阈值"
        for c in result.indicator_comparisons:
            assert c.verdict == "规程中未找到阈值"

    def test_no_report(self, procedure_text):
        result = extract_all("", procedure_text)
        assert not result.success


# ═══════════════════════════════════════════════════════════════
# 格式化输出测试
# ═══════════════════════════════════════════════════════════════

class TestFormatting:
    """格式化输出的完整性和关键信息。"""

    def test_format_contains_all_sections(self, report_text, procedure_text):
        result = extract_all(report_text, procedure_text)
        ctx = format_context(result)
        assert "财务指标逐项比对" in ctx
        assert "抵押率" in ctx
        assert "审批权限" in ctx
        assert "预警信号" in ctx

    def test_format_contains_key_values(self, report_text, procedure_text):
        result = extract_all(report_text, procedure_text)
        ctx = format_context(result)
        assert "57.99" in ctx
        assert "1.49" in ctx
        assert "2.88" in ctx
        assert "60%" in ctx
        assert "50%" in ctx
        assert "2.70亿" in ctx  # 规程重算的可担保额度
        assert "不达标" in ctx
        assert "不合规" in ctx
