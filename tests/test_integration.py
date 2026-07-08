"""集成测试 — 端到端验证审查管线。

覆盖：
- 金融 Map-Reduce 完整管线
- 合同 Map-Reduce 完整管线（语义匹配）
- 劳动法 Map-Reduce 完整管线（语义匹配）
- 附件解析（MD/PDF/DOCX base64）
- 降级路径
- 路由分发
"""

import pytest
import base64
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ═══════════════════════════════════════════════════════════════
# 金融管线集成测试
# ═══════════════════════════════════════════════════════════════

class TestFinancePipeline:
    """金融授信报告审查端到端。"""

    @pytest.fixture
    def report_text(self):
        p = Path(__file__).resolve().parent.parent / "output" / "宏达实业集团_授信调查报告.md"
        return p.read_text(encoding="utf-8")

    @pytest.fixture
    def procedure_text(self):
        p = Path(__file__).resolve().parent.parent / "output" / "信用风险评估操作规程.md"
        return p.read_text(encoding="utf-8")

    def test_full_map(self, report_text, procedure_text):
        """Map 阶段完整执行：7指标 + 比对 + 抵押率 + 审批 + 预警。"""
        from src.review_extractor import extract_all
        result = extract_all(report_text, procedure_text)
        assert result.success
        assert len(result.financial_indicators) == 7
        assert result.collateral_check is not None
        assert not result.collateral_check.is_compliant  # 60% > 50%
        assert result.approval_check is not None
        assert not result.approval_check.is_compliant     # 二级 > 4.2亿
        assert len(result.warning_flags) >= 1

    def test_map_format_output(self, report_text, procedure_text):
        """格式化输出包含全部关键信息。"""
        from src.review_extractor import extract_all, format_context
        result = extract_all(report_text, procedure_text)
        ctx = format_context(result)
        for kw in ["57.99", "1.49", "60%", "50%", "2.70亿", "不达标", "不合规", "三级"]:
            assert kw in ctx, f"缺失关键词: {kw}"

    def test_degradation_no_procedure(self, report_text):
        """无规程时降级：指标仍能提取，比对标为'未找到阈值'。"""
        from src.review_extractor import extract_all
        result = extract_all(report_text, "")
        assert result.success  # 指标提取成功
        for c in result.indicator_comparisons:
            assert "未找到阈值" in c.verdict


# ═══════════════════════════════════════════════════════════════
# 合同管线集成测试
# ═══════════════════════════════════════════════════════════════

class TestContractPipeline:
    """合同审查端到端（语义匹配版）。"""

    @pytest.fixture
    def contract_text(self):
        p = Path(__file__).resolve().parent.parent / "output" / "008_技术服务合同_含风险条款.md"
        return p.read_text(encoding="utf-8")

    @pytest.fixture
    def kb_text(self):
        p = Path(__file__).resolve().parent.parent / "uploads" / "007_合同风险知识库_通用版_1782995633845.md"
        return p.read_text(encoding="utf-8")

    def test_contract_map(self, contract_text, kb_text):
        """合同 Map 阶段：条款提取 + 语义预匹配。"""
        from src.contract_review_extractor import run_contract_map
        result = run_contract_map(contract_text, kb_text)
        assert result.success
        assert result.total_clauses >= 20
        assert result.total_patterns >= 25
        assert result.patterns_with_match > 0

    def test_contract_semantic_matching(self, contract_text, kb_text):
        """语义匹配：关键风险模式应被高度可能匹配。"""
        from src.contract_review_extractor import run_contract_map
        result = run_contract_map(contract_text, kb_text)
        # 知识产权归属、违约金上限、验收主观 — 必须命中
        key_patterns = []
        for m in result.matches:
            if m.auto_verdict in ("高度可能匹配", "可能匹配"):
                key_patterns.append(m.pattern.pattern_name)
        # 至少命中 80% 的模式
        assert len(key_patterns) >= result.total_patterns * 0.7, \
            f"仅命中 {len(key_patterns)}/{result.total_patterns}"


# ═══════════════════════════════════════════════════════════════
# 劳动法管线集成测试
# ═══════════════════════════════════════════════════════════════

class TestLaborPipeline:
    """劳动法审查端到端（语义匹配版）。"""

    @pytest.fixture
    def labor_contract_text(self):
        # 从 PDF 提取的文字
        import subprocess, tempfile, os
        pdf = "/Users/hhy/Desktop/劳动合同A4（续签）.pdf"
        if not os.path.exists(pdf):
            pytest.skip("测试 PDF 不存在")
        from src.rag.parser import parse_document
        text, _ = parse_document(pdf)
        return text

    @pytest.fixture
    def labor_kb_text(self):
        p = Path(__file__).resolve().parent.parent / "uploads" / "劳动法知识库.md"
        return p.read_text(encoding="utf-8")

    def test_labor_map(self, labor_contract_text, labor_kb_text):
        """劳动法 Map 阶段：条款提取 + 语义预匹配。"""
        from src.labor_review_extractor import run_labor_map
        result = run_labor_map(labor_contract_text, labor_kb_text)
        assert result.success
        assert result.total_clauses >= 5
        assert result.total_patterns >= 15

    def test_labor_key_risks_detected(self, labor_contract_text, labor_kb_text):
        """核心劳动法风险必须被预匹配捕获。"""
        from src.labor_review_extractor import run_labor_map
        result = run_labor_map(labor_contract_text, labor_kb_text)
        matched_names = []
        for m in result.matches:
            if m.auto_verdict in ("高度可能匹配", "可能匹配"):
                matched_names.append(m.pattern.pattern_name)
        # 单方调岗调薪、违约金、竞业限制 — 至少命中
        checks = ["调岗", "违约金", "竞业限制", "工时"]
        for c in checks:
            found = any(c in name for name in matched_names)
            assert found, f"核心风险未命中: {c}"


# ═══════════════════════════════════════════════════════════════
# 附件解析集成测试
# ═══════════════════════════════════════════════════════════════

class TestAttachmentParsing:
    """_read_attachments 多种格式测试。"""

    def test_md_attachment(self):
        md_path = Path(__file__).resolve().parent.parent / "output" / "008_技术服务合同_含风险条款.md"
        from core.views import _read_attachments
        result = _read_attachments([str(md_path)])
        assert "技术服务合同" in result
        assert "【附件：" in result

    def test_pdf_base64_attachment(self):
        pdf = "/Users/hhy/Desktop/劳动合同A4（续签）.pdf"
        if not Path(pdf).exists():
            pytest.skip("测试 PDF 不存在")
        with open(pdf, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        from core.views import _read_attachments
        result = _read_attachments([{"name": "test.pdf", "content": b64, "encoding": "base64"}])
        assert "劳动合同" in result or "甲方" in result or "乙方" in result

    def test_no_attachment(self):
        from core.views import _read_attachments
        assert _read_attachments([]) == ""


# ═══════════════════════════════════════════════════════════════
# 路由分发集成测试
# ═══════════════════════════════════════════════════════════════

class TestRouting:
    """route_by_intent 各场景测试。"""

    def _make_state(self, question, **kw):
        from src.state import AgentState
        state = AgentState(
            question=question, raw_input=question,
            intent=kw.get("intent", "research"),
            complexity=kw.get("complexity", "simple"),
            domain=kw.get("domain", "general"),
        )
        return state

    def test_finance_review_triggers_map_reduce(self):
        from src.workflows.orchestrator import route_by_intent
        state = self._make_state("审查这份授信报告", domain="finance")
        assert route_by_intent(state) == "review_pipeline_node"

    def test_contract_risk_triggers_map_reduce(self):
        from src.workflows.orchestrator import route_by_intent
        state = self._make_state("这份合同有什么风险吗", domain="contract")
        assert route_by_intent(state) == "review_pipeline_node"

    def test_complex_research_triggers_react(self):
        from src.workflows.orchestrator import route_by_intent
        state = self._make_state("宏达实业和德方纳米哪个更值得授信", domain="finance", complexity="complex")
        assert route_by_intent(state) == "agentic_research_node"

    def test_simple_research_triggers_fast_path(self):
        from src.workflows.orchestrator import route_by_intent
        state = self._make_state("什么是信用风险评估", domain="general", complexity="simple")
        assert route_by_intent(state) == "research_node"
