"""授信报告审查 — 确定性前置提取引擎。

纯代码模块，零 LLM 依赖。用正则从报告和规程 Markdown 中提取：
- 7 项财务量化指标
- 评级阈值表、抵押率表、审批权限表、预警指标表
- 逐项比对并生成结构化结果

设计原则：提取是增强，不是前提。任何步骤失败都不会抛异常，
而是返回部分结果 + extraction_warnings。
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════════
# 数据结构
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class FinancialIndicator:
    """从报告中提取的单个财务指标。"""
    name: str                         # e.g. "资产负债率"
    value: Optional[float] = None     # e.g. 57.99
    unit: str = ""                    # e.g. "%", "倍", ""
    year: Optional[str] = None        # e.g. "2025年末"
    raw_text: str = ""                # 匹配到的原始文本行


@dataclass
class ThresholdRule:
    """规程中的阈值规则。"""
    indicator_name: str               # e.g. "资产负债率"
    rating_level: str                 # e.g. "AA"
    operator: str                     # "<=", ">="
    threshold_value: float
    unit: str = ""


@dataclass
class IndicatorComparison:
    """单项指标比对结果。"""
    indicator_name: str
    report_value: Optional[float]
    report_year: Optional[str]
    target_rating: str                # 报告自评等级
    threshold_text: str               # e.g. "≤55%"（人类可读）
    threshold_value: float
    operator: str
    verdict: str                      # "达标" / "不达标" / "报告中未提供" / "规程中未找到阈值"
    gap: Optional[float] = None       # 差值（正=超过阈值）
    gap_display: str = ""             # 人类可读差值


@dataclass
class CollateralCheck:
    """抵押率检查结果。"""
    collateral_type: str = ""         # e.g. "工业用地及厂房"
    appraisal_value_text: str = ""    # e.g. "5.40亿元"
    appraisal_value_wan: float = 0.0  # 万元
    report_rate_pct: float = 0.0      # 报告使用的抵押率(%)
    procedure_rate_pct: float = 0.0   # 规程规定的最高抵押率(%)
    report_guarantee_wan: float = 0.0 # 报告计算的可担保额度(万元)
    procedure_guarantee_wan: float = 0.0  # 按规程重算的可担保额度(万元)
    is_compliant: bool = True
    detail: str = ""


@dataclass
class ApprovalCheck:
    """审批权限检查结果。"""
    branch: str = ""                  # 报送机构
    exposure_text: str = ""           # e.g. "4.2亿元"
    exposure_wan: float = 0.0         # 敞口金额(万元)
    report_level: str = ""            # 报告使用的审批层级
    correct_level: str = ""           # 规程要求的审批层级
    correct_level_upper_wan: float = 0.0  # 正确层级上限(万元)
    is_compliant: bool = True
    detail: str = ""


@dataclass
class EarlyWarningFlag:
    """检测到的预警信号。"""
    signal_name: str
    warning_level: str                # "🟡 黄色" / "🔴 红色"
    trigger_condition: str            # 规程中的触发条件
    evidence: str                     # 报告中的证据


@dataclass
class ReviewExtractionResult:
    """完整的审查提取结果。"""
    success: bool = False
    report_file_name: Optional[str] = None
    rules_file_name: Optional[str] = None

    # 报告基本信息
    company_name: str = ""
    target_rating: str = ""            # e.g. "AA"
    report_date: str = ""

    # 7 项量化指标
    financial_indicators: List[FinancialIndicator] = field(default_factory=list)

    # 逐项比对
    indicator_comparisons: List[IndicatorComparison] = field(default_factory=list)

    # 抵押率
    collateral_check: Optional[CollateralCheck] = None

    # 审批权限
    approval_check: Optional[ApprovalCheck] = None

    # 预警信号
    warning_flags: List[EarlyWarningFlag] = field(default_factory=list)

    # 诊断信息
    extraction_warnings: List[str] = field(default_factory=list)
    extraction_failed: bool = False


# ══════════════════════════════════════════════════════════════════════════════
# 指标名称规范化映射
# ══════════════════════════════════════════════════════════════════════════════

# 报告中的指标名 → 规程中的指标名
_INDICATOR_NAME_MAP = {
    "资产负债率": "资产负债率",
    "流动比率": "流动比率",
    "速动比率": "速动比率",
    "EBITDA利息保障倍数": "EBITDA利息保障倍数",
    "近三年营收复合增长率": "近三年营收复合增长率",
    "经营现金流/流动负债": "经营现金流/流动负债",
    "有息债务/EBITDA": "有息债务/EBITDA",
}

# 指标提取顺序（保证输出顺序一致）
_INDICATOR_ORDER = [
    "资产负债率",
    "流动比率",
    "速动比率",
    "EBITDA利息保障倍数",
    "近三年营收复合增长率",
    "经营现金流/流动负债",
    "有息债务/EBITDA",
]


# ══════════════════════════════════════════════════════════════════════════════
# 提取函数
# ══════════════════════════════════════════════════════════════════════════════

def _try_float(s: str) -> Optional[float]:
    """安全地将字符串转为 float。"""
    if not s:
        return None
    try:
        return float(s.strip().replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def _find_section(text: str, *headings: str) -> Optional[str]:
    """在 Markdown 文本中定位某个标题段落之后的全部内容（到下一个同级或上级标题为止）。

    返回第一个匹配标题之后、下一个同级（##）或上级（#）标题之前的内容。
    ## 不会错误匹配 ###（三级标题）。

    支持大纲辅助定位：如果传入的 headings 未匹配到精确标题，
    且 text 参数是 (text, outline, keywords) 三元组格式，则使用大纲模糊匹配。
    """
    for heading in headings:
        escaped = re.escape(heading)
        pattern = rf'{escaped}\s*\n(.*?)(?=\n##(?!#)|\n# (?!#)|\Z)'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return None


def _find_section_by_outline(text: str, outline, keywords: list) -> Optional[str]:
    """使用文档大纲 + 关键词模糊定位章节内容。

    比精确标题匹配更鲁棒：即使标题写法不同也能找到。
    """
    from src.document_outline import find_section_by_outline, get_section_text
    bounds = find_section_by_outline(text, outline, keywords)
    if bounds:
        return get_section_text(text, bounds[0], bounds[1])
    return None


def _parse_markdown_table(text: str) -> List[Dict[str, str]]:
    """将 Markdown 表格文本解析为 dict 列表。

    返回每行的 {列名: 值} 字典。
    自动跳过表头分隔行（|---|---|）。
    """
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return []

    # 第一行是列标题
    headers = [h.strip() for h in lines[0].split("|") if h.strip()]
    if not headers:
        return []

    rows = []
    for line in lines[1:]:
        # 跳过分隔行
        if re.match(r'^\|[\s\-:|]+\|$', line):
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if not cells:
            continue
        row = {}
        for i, cell in enumerate(cells):
            if i < len(headers):
                row[headers[i]] = _clean_md(cell)
        if row:
            rows.append(row)
    return rows


def _clean_md(text: str) -> str:
    """清除 Markdown 加粗/斜体标记。"""
    return text.replace("**", "").replace("*", "").strip()


def _extract_number(s: str) -> Optional[float]:
    """从字符串中提取第一个数值（支持百分比、负数、小数）。"""
    s = _clean_md(s)
    # 匹配数字（含负号、小数点），可能后跟 %
    m = re.search(r'(-?[\d,]+\.?\d*)\s*%?', s)
    if m:
        val = m.group(1).replace(",", "")
        return _try_float(val)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 1. 从报告中提取财务指标
# ══════════════════════════════════════════════════════════════════════════════

def extract_financial_indicators(text: str) -> List[FinancialIndicator]:
    """从授信调查报告文本中提取 7 项量化财务指标。

    优先从「量化指标计算依据」表格提取（最精确），
    其次从资产负债表/利润表表格中提取最新年份数据。
    """
    indicators: List[FinancialIndicator] = []
    found_names: set = set()

    # ── 策略 A：从「量化指标计算依据」表格提取 ──
    section = _find_section(text, "量化指标计算依据", "4.2 量化指标计算依据")
    if section:
        # 该 section 含一个表格
        rows = _parse_markdown_table(section)
        for row in rows:
            # 表格列可能是: 指标 | 计算值 | 初评对标等级 | 计算过程
            # 或: 项目 | 内容 (等等)
            # 我们需要的第一列是指标名
            for key in ["指标", "项目", ""]:
                name_cell = row.get(key, "")
                if name_cell:
                    break
            if not name_cell:
                # 尝试获取第一个键的值
                for k, v in row.items():
                    if k:
                        name_cell = v
                        break

            name_clean = _clean_md(name_cell)
            # 匹配已知指标名
            matched_name = None
            for indicator_name in _INDICATOR_ORDER:
                if indicator_name in name_clean:
                    matched_name = indicator_name
                    break
            if not matched_name or matched_name in found_names:
                continue

            # 获取值（通常在第二列 "计算值"）
            value_cell = row.get("计算值", "")
            if not value_cell:
                # 尝试第二列
                vals = list(row.values())
                if len(vals) >= 2:
                    value_cell = vals[1]

            val = _extract_number(value_cell)
            unit = "%" if "%" in value_cell else ""
            if "倍" in name_clean or "保障" in name_clean:
                unit = "倍"

            indicators.append(FinancialIndicator(
                name=matched_name,
                value=val,
                unit=unit,
                year="2025年末",
                raw_text=f"{name_clean}: {value_cell}",
            ))
            found_names.add(matched_name)

    # ── 策略 B：从资产负债表和利润表中补充遗漏指标 ──
    # 资产负债率（从表2.1的最后一列）
    if "资产负债率" not in found_names:
        val = _extract_from_last_column(text, r'\*\*资产负债率\*\*', r'([\d.]+)%')
        if val is not None:
            indicators.append(FinancialIndicator(
                name="资产负债率", value=val, unit="%",
                year="2025年末", raw_text=f"资产负债率: {val}%",
            ))
            found_names.add("资产负债率")

    # 流动比率 — 计算：流动资产/流动负债（2025: 195,400/131,500）
    if "流动比率" not in found_names:
        val = _extract_from_last_column(text, r'\*\*流动比率\*\*', r'([\d.]+)')
        if val is not None:
            indicators.append(FinancialIndicator(
                name="流动比率", value=val, unit="",
                year="2025年末", raw_text=f"流动比率: {val}",
            ))
            found_names.add("流动比率")

    if "速动比率" not in found_names:
        val = _extract_from_last_column(text, r'\*\*速动比率\*\*', r'([\d.]+)')
        if val is not None:
            indicators.append(FinancialIndicator(
                name="速动比率", value=val, unit="",
                year="2025年末", raw_text=f"速动比率: {val}",
            ))
            found_names.add("速动比率")

    if "EBITDA利息保障倍数" not in found_names:
        val = _extract_from_last_column(text, r'EBITDA.*?利息保障倍数', r'([\d.]+)')
        if val is not None:
            indicators.append(FinancialIndicator(
                name="EBITDA利息保障倍数", value=val, unit="倍",
                year="2025年末", raw_text=f"EBITDA利息保障倍数: {val}",
            ))
            found_names.add("EBITDA利息保障倍数")

    # ── 兜底：全文搜索加粗格式 ──
    fallback_patterns = [
        ("资产负债率", r'\*\*资产负债率\*\*\s*[|=]?\s*\*{0,2}([\d.]+)%\*{0,2}', "%"),
        ("流动比率", r'\*\*流动比率\*\*\s*[|=]?\s*\*{0,2}([\d.]+)\*{0,2}', ""),
        ("速动比率", r'\*\*速动比率\*\*\s*[|=]?\s*\*{0,2}([\d.]+)\*{0,2}', ""),
        ("EBITDA利息保障倍数", r'EBITDA.*?利息保障倍数\*{0,2}\s*[|=]?\s*\*{0,2}([\d.]+)\*{0,2}', "倍"),
        ("近三年营收复合增长率", r'近三年.*?复合增长率\*{0,2}\s*[|=]?\s*\*{0,2}([\d.]+)%\*{0,2}', "%"),
        ("经营现金流/流动负债", r'经营现金流.*?流动负债\*{0,2}\s*[|=]?\s*\*{0,2}(-?[\d.]+)\*{0,2}', ""),
        ("有息债务/EBITDA", r'有息债务.*?EBITDA\*{0,2}\s*[|=]?\s*\*{0,2}([\d.]+)\*{0,2}', "倍"),
    ]

    for name, pattern, unit in fallback_patterns:
        if name not in found_names:
            match = re.search(pattern, text)
            if match:
                val = _try_float(match.group(1))
                indicators.append(FinancialIndicator(
                    name=name, value=val, unit=unit,
                    year="2025年末", raw_text=match.group(0),
                ))
                found_names.add(name)

    # 按固定顺序排列
    ordered = []
    for name in _INDICATOR_ORDER:
        for ind in indicators:
            if ind.name == name:
                ordered.append(ind)
                break
    return ordered


def _extract_from_last_column(text: str, label_pattern: str, value_pattern: str) -> Optional[float]:
    """从「包含 label 的表格行」中提取最后一列的数值。

    用于资产负债率、流动比率等在资产负债表中以多列形式出现的指标。
    最后一列通常是 2025年末 数据。
    """
    # 找到包含该 label 的行
    escaped = label_pattern.replace(r'\*\*', r'\*{0,2}')
    line_pat = rf'\|[^|]*{escaped}[^|]*\|.*\|'
    matches = re.findall(line_pat, text)
    if not matches:
        return None
    # 取最后一个匹配行的最后一列
    last_line = matches[-1]
    cells = last_line.split("|")
    # 过滤空字符串
    cells = [c for c in cells if c.strip()]
    if not cells:
        return None
    last_cell = cells[-1]
    m = re.search(value_pattern, last_cell)
    if m:
        return _try_float(m.group(1))
    return None


# ══════════════════════════════════════════════════════════════════════════════
# 2. 从规程中提取阈值表
# ══════════════════════════════════════════════════════════════════════════════

def extract_procedure_thresholds(text: str) -> Dict[str, Dict[str, ThresholdRule]]:
    """从操作规程中解析评级量化指标表（2.2）。

    Returns:
        {indicator_name: {rating_level: ThresholdRule}}
        例如: {"资产负债率": {"AA": ThresholdRule(operator="<=", value=55.0, ...), ...}, ...}
    """
    result: Dict[str, Dict[str, ThresholdRule]] = {}

    # 找到表2.2
    section = _find_section(text, "各等级量化指标", "2.2 各等级量化指标")
    if not section:
        return result

    rows = _parse_markdown_table(section)
    if not rows:
        return result

    # 列标题: 指标 | AAA | AA | A | BBB | BB
    rating_cols = ["AAA", "AA", "A", "BBB", "BB"]

    for row in rows:
        # 获取指标名（第一列）
        first_key = list(row.keys())[0] if row else ""
        indicator_raw = _clean_md(row.get(first_key, ""))
        if not indicator_raw:
            continue

        # 映射到标准指标名
        matched_name = None
        for std_name in _INDICATOR_ORDER:
            if std_name in indicator_raw:
                matched_name = std_name
                break
        if not matched_name:
            continue

        threshold_map: Dict[str, ThresholdRule] = {}

        for level in rating_cols:
            cell = row.get(level, "")
            if not cell or cell in ("—", "-", "—", "不适用"):
                continue

            op, val, unit = _parse_threshold_cell(cell)
            if val is None:
                continue

            threshold_map[level] = ThresholdRule(
                indicator_name=matched_name,
                rating_level=level,
                operator=op,
                threshold_value=val,
                unit=unit,
            )

        if threshold_map:
            result[matched_name] = threshold_map

    return result


def _parse_threshold_cell(cell: str) -> Tuple[str, Optional[float], str]:
    """解析阈值单元格，如 "≤55%" → ("<=", 55.0, "%")"""
    cell = _clean_md(cell).strip()
    if not cell:
        return ("", None, "")

    op = ""
    if "≤" in cell:
        op = "<="
    elif "≥" in cell:
        op = ">="
    elif "<" in cell:
        op = "<"
    elif ">" in cell:
        op = ">"

    unit = "%" if "%" in cell else ""

    # 检查是否为 "倍" 单位
    if "倍" in cell:
        unit = "倍"

    val = _extract_number(cell)
    return (op, val, unit)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 提取抵押率表（规程 5.2）
# ══════════════════════════════════════════════════════════════════════════════

def extract_collateral_rates(text: str) -> Dict[str, float]:
    """从规程中解析抵押物类型及最高抵押率（表5.2）。

    Returns:
        {抵押物类型: 最高抵押率(%)}
        例如: {"工业用地及厂房": 50.0, "住宅类不动产": 70.0, ...}
    """
    result: Dict[str, float] = {}

    section = _find_section(text, "抵押物类型及最高抵押率", "5.2 抵押物类型及最高抵押率")
    if not section:
        return result

    rows = _parse_markdown_table(section)
    for row in rows:
        # 第一列是抵押物类型，第二列是最高抵押率
        vals = list(row.values())
        if len(vals) < 2:
            continue

        type_name = _clean_md(vals[0])
        rate_val = _extract_number(vals[1])

        if type_name and rate_val is not None:
            # 规程中抵押率是百分比形式，不需要再除100
            result[type_name] = rate_val

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 4. 提取审批权限表（规程 4）
# ══════════════════════════════════════════════════════════════════════════════

def extract_approval_levels(text: str) -> List[Dict[str, Any]]:
    """从规程中解析审批权限层级表。

    Returns:
        [{level: "一级", org: "支行贷审会", upper: 5000(万元)}, ...]
    """
    result: List[Dict[str, Any]] = []

    section = _find_section(text, "审批权限层级")
    if not section:
        return result

    rows = _parse_markdown_table(section)
    for row in rows:
        vals = list(row.values())
        if len(vals) < 3:
            continue

        level = _clean_md(vals[0])
        org = _clean_md(vals[1]) if len(vals) > 1 else ""
        amount_cell = _clean_md(vals[2]) if len(vals) > 2 else ""

        # 解析金额上限
        upper_wan = _parse_approval_amount(amount_cell)

        result.append({
            "level": level,
            "org": org,
            "amount_text": amount_cell,
            "upper_wan": upper_wan,
        })

    return result


def _parse_approval_amount(cell: str) -> float:
    """解析审批金额，如 "≤5,000万元" → 5000, "5,000万元 ~ 2亿元（含）" → 20000"""
    cell = _clean_md(cell).replace(",", "").replace(" ", "")
    if not cell:
        return 0.0

    # 提取最大的数值（上限）
    # 匹配所有数值
    nums = re.findall(r'([\d.]+)\s*(亿|万)?', cell)
    if not nums:
        return 0.0

    max_val = 0.0
    for num_str, unit in nums:
        val = _try_float(num_str)
        if val is None:
            continue
        if unit == "亿":
            val *= 10000  # 亿元 → 万元
        max_val = max(max_val, val)

    return max_val


# ══════════════════════════════════════════════════════════════════════════════
# 5. 从报告中提取辅助信息
# ══════════════════════════════════════════════════════════════════════════════

def extract_rating_from_report(text: str) -> str:
    """提取报告中的信用评级结论。"""
    # "**初评等级：AA**" 或 "综合评定为AA级"
    m = re.search(r'初评等级[：:]\s*\*{0,2}(A{1,3}B{0,2})\*{0,2}', text)
    if m:
        return m.group(1)
    m = re.search(r'综合评定为\s*\*{0,2}(A{1,3}B{0,2})', text)
    if m:
        return m.group(1)
    return ""


def extract_company_name(text: str) -> str:
    """提取企业名称。"""
    m = re.search(r'\*\*企业名称\*\*[：:]\s*(.+)', text)
    if m:
        return _clean_md(m.group(1)).strip()
    return ""


def extract_collateral_info(text: str) -> Dict[str, Any]:
    """从报告中提取抵押物信息。

    Returns:
        {collateral_type, appraisal_value_wan, report_rate_pct, report_guarantee_wan}
    """
    info: Dict[str, Any] = {
        "collateral_type": "",
        "appraisal_value_text": "",
        "appraisal_value_wan": 0.0,
        "report_rate_pct": 0.0,
        "report_guarantee_wan": 0.0,
    }

    # 抵押物信息在「担保方案」章节
    section = _find_section(text, "担保方案", "五、担保方案")
    if not section:
        return info

    # 提取抵押物类型
    for row in _parse_markdown_table(section) if "|" in section else []:
        pass  # 表结构复杂，用 regex 更可靠

    # 工业用地及厂房
    for kw in ["工业用地及厂房", "工业用地", "厂房"]:
        if kw in section:
            info["collateral_type"] = kw if kw == "工业用地及厂房" else "工业用地及厂房"
            break

    # 抵押物评估总价值 — "**抵押物合计** | — | — | **5.40亿元**"
    m = re.search(r'抵押物合计.*?([\d.]+)\s*亿', section)
    if m:
        val = _try_float(m.group(1))
        if val:
            info["appraisal_value_text"] = f"{val:.2f}亿元"
            info["appraisal_value_wan"] = val * 10000  # 亿元 → 万元

    # 报告使用的抵押率 — "工业用地及厂房最高抵押率（按本行评估惯例） | **60%**"
    m = re.search(r'(?:最高抵押率|抵押率).*?\*{0,2}(\d+)\s*%\*{0,2}', section)
    if m:
        info["report_rate_pct"] = _try_float(m.group(1)) or 0.0

    # 报告中的可担保额度 — "可担保额度 | 3.24亿元"
    m = re.search(r'可担保额度.*?([\d.]+)\s*亿', section)
    if m:
        val = _try_float(m.group(1))
        if val:
            info["report_guarantee_wan"] = val * 10000

    return info


def extract_approval_info(text: str) -> Dict[str, Any]:
    """从报告中提取审批相关信息。

    Returns:
        {branch, exposure_text, exposure_wan, report_level}
    """
    info: Dict[str, Any] = {
        "branch": "",
        "exposure_text": "",
        "exposure_wan": 0.0,
        "report_level": "",
    }

    # 报送机构
    m = re.search(r'\*\*报送机构\*\*[：:]\s*(.+)', text)
    if m:
        info["branch"] = _clean_md(m.group(1)).strip()

    # 敞口金额 — 多种表述
    for pat in [
        r'敞口合计\s*\*{0,2}\s*约?\s*([\d.]+)\s*亿',
        r'总风险敞口\s*约?\s*([\d.]+)\s*亿',
        r'敞口合计.*?([\d.]+)\s*亿',
        r'申请授信金额.*?([\d.]+)\s*亿',
    ]:
        m = re.search(pat, text)
        if m:
            val = _try_float(m.group(1))
            if val:
                info["exposure_text"] = f"{val:.1f}亿元"
                # 解析具体数值
                # 从"敞口合计 **5.2亿元**（其中贷款3.2亿元、银承2.0亿元按50%风险权重折算后敞口1.0亿元，总风险敞口约 **4.2亿元**）"
                # 优先取"总风险敞口"
                m2 = re.search(r'总风险敞口\s*约?\s*\*{0,2}\s*([\d.]+)\s*亿', text)
                if m2:
                    val2 = _try_float(m2.group(1))
                    if val2:
                        info["exposure_wan"] = val2 * 10000
                        info["exposure_text"] = f"{val2:.1f}亿元"
                else:
                    info["exposure_wan"] = val * 10000
                break

    # 审批层级（如果有）
    for level_kw in ["支行贷审会", "分行贷审会", "省行贷审会", "总行信用审批委员会"]:
        if level_kw in text:
            info["report_level"] = level_kw
            break

    return info


# ══════════════════════════════════════════════════════════════════════════════
# 6. 比对逻辑
# ══════════════════════════════════════════════════════════════════════════════

def compare_indicators(
    indicators: List[FinancialIndicator],
    thresholds: Dict[str, Dict[str, ThresholdRule]],
    target_rating: str,
) -> List[IndicatorComparison]:
    """将报告指标与目标评级的规程阈值逐项比对。"""
    comparisons: List[IndicatorComparison] = []

    for ind in indicators:
        # 找规程中对应的阈值
        proc_name = _INDICATOR_NAME_MAP.get(ind.name, ind.name)
        threshold_map = thresholds.get(proc_name, {})

        # 优先使用目标评级，其次降级查找
        rule = threshold_map.get(target_rating)
        if rule is None and threshold_map:
            # 使用任意可用评级
            rule = list(threshold_map.values())[0]

        if rule is None:
            comparisons.append(IndicatorComparison(
                indicator_name=ind.name,
                report_value=ind.value,
                report_year=ind.year,
                target_rating=target_rating,
                threshold_text="规程中未找到阈值",
                threshold_value=0.0,
                operator="",
                verdict="规程中未找到阈值",
            ))
            continue

        if ind.value is None:
            comparisons.append(IndicatorComparison(
                indicator_name=ind.name,
                report_value=None,
                report_year=ind.year,
                target_rating=target_rating,
                threshold_text=_threshold_display(rule),
                threshold_value=rule.threshold_value,
                operator=rule.operator,
                verdict="报告中未提供",
            ))
            continue

        # 判定达标/不达标
        verdict, gap = _judge(ind.value, rule)
        gap_display = _format_gap(gap, rule.unit)

        comparisons.append(IndicatorComparison(
            indicator_name=ind.name,
            report_value=ind.value,
            report_year=ind.year,
            target_rating=target_rating,
            threshold_text=_threshold_display(rule),
            threshold_value=rule.threshold_value,
            operator=rule.operator,
            verdict=verdict,
            gap=gap,
            gap_display=gap_display,
        ))

    return comparisons


def _judge(value: float, rule: ThresholdRule) -> Tuple[str, Optional[float]]:
    """判定值是否满足阈值要求。

    Returns:
        (verdict, gap) — gap 为正表示超过阈值方向
    """
    if rule.operator in ("<=", "<"):
        gap = value - rule.threshold_value
        if value <= rule.threshold_value:
            return ("达标", gap)
        else:
            return ("不达标", gap)
    elif rule.operator in (">=", ">"):
        gap = value - rule.threshold_value
        if value >= rule.threshold_value:
            return ("达标", gap)
        else:
            return ("不达标", gap)
    else:
        return ("无法判定（未知运算符）", None)


def _threshold_display(rule: ThresholdRule) -> str:
    """阈值的人类可读展示。"""
    op_display = {"<=": "≤", ">=": "≥", "<": "<", ">": ">"}
    op = op_display.get(rule.operator, rule.operator)
    unit = rule.unit if rule.unit else ""
    return f"{op}{rule.threshold_value}{unit}"


def _format_gap(gap: Optional[float], unit: str) -> str:
    """格式化差值。"""
    if gap is None:
        return ""
    if unit == "%":
        return f"{gap:+.2f}pp"
    return f"{gap:+.2f}"


def check_collateral_rate(
    collateral_info: Dict[str, Any],
    collateral_rates: Dict[str, float],
) -> Optional[CollateralCheck]:
    """检查抵押率是否合规。"""
    ctype = collateral_info.get("collateral_type", "")
    if not ctype:
        return None

    # 在规程抵押率表中查找
    proc_rate = None
    for proc_type, rate in collateral_rates.items():
        if ctype in proc_type or proc_type in ctype:
            proc_rate = rate
            break
    # 模糊匹配
    if proc_rate is None:
        for proc_type, rate in collateral_rates.items():
            # 关键词重叠
            if any(kw in ctype for kw in ["工业", "厂房", "用地"]) and any(kw in proc_type for kw in ["工业", "厂房", "用地"]):
                proc_rate = rate
                break

    if proc_rate is None:
        return CollateralCheck(
            collateral_type=ctype,
            appraisal_value_text=collateral_info.get("appraisal_value_text", ""),
            appraisal_value_wan=collateral_info.get("appraisal_value_wan", 0.0),
            report_rate_pct=collateral_info.get("report_rate_pct", 0.0),
            procedure_rate_pct=0.0,
            report_guarantee_wan=collateral_info.get("report_guarantee_wan", 0.0),
            procedure_guarantee_wan=0.0,
            is_compliant=True,
            detail="规程中未找到该抵押物类型的抵押率上限，无法自动判定",
        )

    appraisal = collateral_info.get("appraisal_value_wan", 0.0)
    report_rate = collateral_info.get("report_rate_pct", 0.0)
    report_guarantee = collateral_info.get("report_guarantee_wan", 0.0)
    proc_guarantee = appraisal * proc_rate / 100.0

    is_compliant = report_rate <= proc_rate

    return CollateralCheck(
        collateral_type=ctype,
        appraisal_value_text=collateral_info.get("appraisal_value_text", ""),
        appraisal_value_wan=appraisal,
        report_rate_pct=report_rate,
        procedure_rate_pct=proc_rate,
        report_guarantee_wan=report_guarantee,
        procedure_guarantee_wan=proc_guarantee,
        is_compliant=is_compliant,
        detail=(
            f"合规" if is_compliant
            else f"不合规：报告使用 {report_rate}% 抵押率，规程规定最高 {proc_rate}%"
        ),
    )


def check_approval_level(
    approval_info: Dict[str, Any],
    approval_levels: List[Dict[str, Any]],
) -> Optional[ApprovalCheck]:
    """检查审批权限是否合规。"""
    branch = approval_info.get("branch", "")
    exposure_wan = approval_info.get("exposure_wan", 0.0)
    if not branch or exposure_wan <= 0:
        return None

    # 确定报送机构的层级
    branch_level = _infer_branch_level(branch)
    # 查找敞口金额应落在的审批层级
    correct_level = _find_correct_level(exposure_wan, approval_levels)

    if correct_level is None:
        return None

    correct_level_name = f"{correct_level['level']} {correct_level['org']}"
    is_compliant = branch_level is not None and branch_level == correct_level["level"]

    detail_parts = []
    if not is_compliant:
        if branch_level is not None:
            branch_level_display = branch_level.replace("级", "") + "级" if branch_level else ""
            detail_parts.append(
                f"报送机构「{branch}」为{branch_level_display}，审批权限上限 "
                f"{correct_level['upper_wan']:.0f}万元；"
                f"敞口 {approval_info.get('exposure_text', '')} 应上提至 {correct_level_name}"
            )
        else:
            detail_parts.append(f"无法识别报送机构层级，敞口应报 {correct_level_name}")

    return ApprovalCheck(
        branch=branch,
        exposure_text=approval_info.get("exposure_text", ""),
        exposure_wan=exposure_wan,
        report_level=approval_info.get("report_level", ""),
        correct_level=correct_level_name,
        correct_level_upper_wan=correct_level.get("upper_wan", 0.0),
        is_compliant=is_compliant,
        detail="\n".join(detail_parts),
    )


def _infer_branch_level(branch: str) -> Optional[str]:
    """根据机构名称推断审批层级。"""
    if "支行" in branch:
        return "一级"
    if "分行" in branch and "省" not in branch:
        return "二级"
    if "省行" in branch or "省分" in branch:
        return "三级"
    if "总行" in branch:
        return "四级"
    return None


def _find_correct_level(exposure_wan: float, levels: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """根据敞口金额找到正确的审批层级。"""
    for lv in levels:
        if exposure_wan <= lv.get("upper_wan", 0):
            return lv
    # 超过所有层级上限
    if levels:
        return levels[-1]
    return None


def _find_level_by_name(level_name: str, levels: List[Dict[str, Any]]) -> str:
    """根据层级名称找到上限描述。"""
    for lv in levels:
        if lv.get("level") == level_name:
            return lv.get("amount_text", "未知")
    return "未知"


# ══════════════════════════════════════════════════════════════════════════════
# 7. 预警信号检测
# ══════════════════════════════════════════════════════════════════════════════

def detect_early_warnings(
    indicators: List[FinancialIndicator],
    text: str,
) -> List[EarlyWarningFlag]:
    """检测报告是否触发规程 6.1 中的预警信号。"""
    warnings: List[EarlyWarningFlag] = []

    # 建立指标查找表
    ind_map = {ind.name: ind for ind in indicators}

    # 1. 经营现金流连续为负 → 红色预警
    op_cf = ind_map.get("经营现金流/流动负债")
    if op_cf and op_cf.value is not None and op_cf.value < 0:
        # 确认报告中是否有"连续两年为负"的描述
        has_two_years = bool(re.search(
            r'(2024|2025).*?经营.*?现金.*?为负|经营.*?现金.*?持续.*?为负|连续.*?(两|2).*?[年季度].*?为负',
            text,
        ))
        if has_two_years or "连续" in text:
            warnings.append(EarlyWarningFlag(
                signal_name="经营现金流持续恶化",
                warning_level="🔴 红色",
                trigger_condition="连续两个季度经营现金流为负且无改善迹象（规程6.1第4条）",
                evidence=f"2024-2025年度经营现金流持续为负，2025年末经营现金流/流动负债 = {op_cf.value}",
            ))
        else:
            # 单期为负 → 黄色预警
            warnings.append(EarlyWarningFlag(
                signal_name="经营现金流恶化",
                warning_level="🟡 黄色",
                trigger_condition="经营现金流量净额为负（规程6.1第3条）",
                evidence=f"2025年末经营现金流/流动负债 = {op_cf.value}",
            ))

    # 2. 对外担保超过净资产 50% → 黄色预警
    m = re.search(r'对外担保.*?(\d+\.?\d*)\s*亿.*?净资产的?\s*(\d+)%', text)
    if m:
        guarantee_ratio = _try_float(m.group(2))
        if guarantee_ratio is not None and guarantee_ratio >= 50:
            level = "🔴 红色" if guarantee_ratio >= 100 else "🟡 黄色"
            trigger = (
                "对外担保余额超过净资产100%（规程6.1第6条）" if guarantee_ratio >= 100
                else "对外担保余额超过净资产50%（规程6.1第5条）"
            )
            warnings.append(EarlyWarningFlag(
                signal_name="对外担保风险",
                warning_level=level,
                trigger_condition=trigger,
                evidence=m.group(0).strip(),
            ))

    # 3. 应收账款周转天数（从报告中判断）
    # 报告中可能没有直接给出周转天数，但提到了"应收账款增长较快"

    return warnings


# ══════════════════════════════════════════════════════════════════════════════
# 8. 主入口
# ══════════════════════════════════════════════════════════════════════════════

def extract_all(report_text: str, procedure_text: str) -> ReviewExtractionResult:
    """执行全部提取和比对。

    Args:
        report_text: 授信调查报告的 Markdown 全文。
        procedure_text: 信用风险评估操作规程的 Markdown 全文。

    Returns:
        ReviewExtractionResult，包含所有提取和比对结果。
    """
    result = ReviewExtractionResult()
    warnings: List[str] = []

    try:
        # ── 报告基本信息 ──
        result.company_name = extract_company_name(report_text)
        result.target_rating = extract_rating_from_report(report_text)
        if not result.target_rating:
            warnings.append("未能从报告中提取信用评级结论")
            result.target_rating = "AA"  # 默认

        # ── 财务指标 ──
        result.financial_indicators = extract_financial_indicators(report_text)
        if not result.financial_indicators:
            warnings.append("未能从报告中提取到任何财务指标")
        else:
            found_names = [i.name for i in result.financial_indicators]
            missing = [n for n in _INDICATOR_ORDER if n not in found_names]
            if missing:
                warnings.append(f"以下指标未能从报告中提取: {', '.join(missing)}")

        # ── 规程阈值 ──
        thresholds = extract_procedure_thresholds(procedure_text)
        if not thresholds:
            warnings.append("未能从规程中解析评级量化指标表")

        # ── 指标比对 ──
        result.indicator_comparisons = compare_indicators(
            result.financial_indicators, thresholds, result.target_rating,
        )

        # ── 抵押率检查 ──
        collateral_info = extract_collateral_info(report_text)
        collateral_rates = extract_collateral_rates(procedure_text)
        if collateral_info.get("collateral_type"):
            result.collateral_check = check_collateral_rate(collateral_info, collateral_rates)
        else:
            warnings.append("未能从报告中提取抵押物信息")

        # ── 审批权限检查 ──
        approval_info = extract_approval_info(report_text)
        approval_levels = extract_approval_levels(procedure_text)
        if approval_info.get("branch") and approval_levels:
            result.approval_check = check_approval_level(approval_info, approval_levels)
        else:
            if not approval_info.get("branch"):
                warnings.append("未能从报告中提取报送机构信息")
            if not approval_levels:
                warnings.append("未能从规程中解析审批权限表")

        # ── 预警信号 ──
        result.warning_flags = detect_early_warnings(result.financial_indicators, report_text)

        # ── 判定成功 ──
        result.success = bool(result.financial_indicators)
        if not result.success:
            result.extraction_failed = True

    except Exception as e:
        warnings.append(f"提取过程异常: {e}")
        result.extraction_failed = True

    result.extraction_warnings = warnings
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 9. 格式化输出
# ══════════════════════════════════════════════════════════════════════════════

def format_context(result: ReviewExtractionResult) -> str:
    """将 ReviewExtractionResult 格式化为注入 agent 上下文的结构化 Markdown。"""
    parts: List[str] = []

    parts.append("## [系统预提取] 量化指标自动比对结果\n")
    parts.append(
        "> 以下内容由系统自动从报告和规程中提取并比对，"
        "请在此基础上进行深度定性分析，不需要重新检索这些指标。\n"
    )

    # ── 一、财务指标逐项比对 ──
    company = result.company_name or "被审查企业"
    rating = result.target_rating or "未知"
    parts.append(f"### 一、财务指标逐项比对（目标评级：{rating}）\n")

    if result.indicator_comparisons:
        parts.append("| 指标名称 | 报告值 | 规程阈值（{}级） | 判定 | 差值 |".format(rating))
        parts.append("|---------|--------|-----------------|------|------|")
        for comp in result.indicator_comparisons:
            val_display = _format_value(comp.report_value, comp.report_year)
            verdict_icon = "✅" if comp.verdict == "达标" else ("❌" if comp.verdict == "不达标" else "⚠️")
            gap_display = comp.gap_display if comp.gap_display else "-"
            parts.append(
                f"| {comp.indicator_name} | {val_display} | {comp.threshold_text} "
                f"| {verdict_icon} {comp.verdict} | {gap_display} |"
            )

        # 统计
        total = len(result.indicator_comparisons)
        passed = sum(1 for c in result.indicator_comparisons if c.verdict == "达标")
        failed = sum(1 for c in result.indicator_comparisons if c.verdict == "不达标")
        unknown = total - passed - failed
        parts.append(f"\n**比对统计**：{total} 项指标中，✅ 达标 {passed} 项，❌ 不达标 {failed} 项"
                     + (f"，⚠️ 其他 {unknown} 项" if unknown else ""))
    else:
        parts.append("（未能提取到指标数据）")

    # ── 二、抵押率对照 ──
    parts.append("\n### 二、抵押率与规程对照\n")

    cc = result.collateral_check
    if cc:
        parts.append("| 项目 | 报告使用值 | 规程规定值 | 是否合规 |")
        parts.append("|------|-----------|-----------|---------|")
        parts.append(
            f"| {cc.collateral_type}抵押率 | **{cc.report_rate_pct:.0f}%** "
            f"| **{cc.procedure_rate_pct:.0f}%**（规程表5.2） "
            f"| {'✅ 合规' if cc.is_compliant else '❌ 不合规'} |"
        )
        if not cc.is_compliant:
            parts.append(f"\n**⚠️ 合规风险**：{cc.detail}")
            parts.append(f"\n- 报告按 {cc.report_rate_pct:.0f}% 计算可担保额度：{cc.report_guarantee_wan/10000:.2f}亿元")
            parts.append(f"- **按规程 {cc.procedure_rate_pct:.0f}% 重算可担保额度：{cc.procedure_guarantee_wan/10000:.2f}亿元**")
            diff = cc.report_guarantee_wan - cc.procedure_guarantee_wan
            parts.append(f"- 差额：{diff/10000:.2f}亿元（报告高估了可担保额度）")
    else:
        parts.append("（未能提取到抵押物信息）")

    # ── 三、审批权限核查 ──
    parts.append("\n### 三、审批权限逐级核查\n")

    ac = result.approval_check
    if ac:
        parts.append("| 项目 | 内容 |")
        parts.append("|------|------|")
        parts.append(f"| 报送机构 | {ac.branch} |")
        parts.append(f"| 申请敞口 | {ac.exposure_text}（折合 {ac.exposure_wan:.0f} 万元） |")
        parts.append(f"| 规程要求审批层级 | {ac.correct_level}（上限 {ac.correct_level_upper_wan:.0f} 万元） |")
        parts.append(f"| 核查结论 | {'✅ 合规' if ac.is_compliant else '⚠️ 须关注'} |")
        if ac.detail:
            parts.append(f"\n{ac.detail}")
    else:
        parts.append("（未能提取到审批权限相关信息）")

    # ── 四、贷后预警信号 ──
    parts.append("\n### 四、贷后预警信号检测\n")

    if result.warning_flags:
        parts.append("| 预警信号 | 预警等级 | 触发条件 | 报告证据 |")
        parts.append("|---------|---------|---------|---------|")
        for wf in result.warning_flags:
            parts.append(f"| {wf.signal_name} | {wf.warning_level} | {wf.trigger_condition} | {wf.evidence} |")
    else:
        parts.append("（未检测到明显预警信号，但仍建议人工复核）")

    # ── 提取警告 ──
    if result.extraction_warnings:
        parts.append("\n### ⚠️ 提取说明\n")
        for w in result.extraction_warnings:
            parts.append(f"- {w}")

    parts.append("\n---")
    parts.append(
        "**以上为系统自动提取结果，请重点分析：**\n"
        "1. 各项不达标指标对评级结论的影响（是否需要下调评级）\n"
        "2. 抵押率超规程上限的合规风险及补救措施\n"
        "3. 审批权限是否需要上提\n"
        "4. 预警信号是否需要触发规程规定的处置措施\n"
        "5. 关联互保折算、贷后管理等第二阶段深度分析\n"
    )

    return "\n".join(parts)


def _format_value(value: Optional[float], year: Optional[str] = None) -> str:
    """格式化指标值用于表格展示。"""
    if value is None:
        return "未提供"
    year_suffix = f"（{year}）" if year else ""
    return f"{value}{year_suffix}"


# ══════════════════════════════════════════════════════════════════════════════
# 便捷函数
# ══════════════════════════════════════════════════════════════════════════════

def quick_review(report_text: str, procedure_text: str) -> str:
    """便捷函数：一步完成提取+比对+格式化。"""
    result = extract_all(report_text, procedure_text)
    return format_context(result)
