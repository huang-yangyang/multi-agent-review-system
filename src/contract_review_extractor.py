"""合同审查 Map 提取器 — 确定性逐条匹配。

Map 阶段（纯代码，零 LLM）：
1. 从合同中提取全部条款（按条号结构化）
2. 从知识库中提取全部风险模式（按章节归类）
3. 关键词级预匹配：每个风险模式 → 可能相关的合同条款
4. 输出结构化清单供 Reduce 阶段做最终语义判定

设计原则：穷举保证 — 知识库 27 个风险模式全部列出，一个不漏。
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ═══════════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════════

@dataclass
class ContractClause:
    """合同中的一条。"""
    article_num: str        # "第四条"
    article_title: str      # "知识产权条款"
    section_num: str        # "4.1"
    section_title: str      # "成果归属"
    content: str            # 条款全文
    raw_text: str           # 原始 Markdown 块


@dataclass
class RiskPattern:
    """知识库中的一个风险模式。"""
    chapter: str            # "第2章 知识产权条款风险"
    pattern_id: str         # "2.1"
    pattern_name: str       # "委托开发成果归属约定不明"
    typical_wording: str    # 典型话术
    risk_consequence: str   # 风险后果
    suggestion: str         # 修改建议
    keywords: List[str]     # 匹配关键词
    source_lines: str       # 知识库中的原始行范围


@dataclass
class ClausePatternMatch:
    """一个风险模式与合同条款的预匹配结果。"""
    pattern: RiskPattern
    matched_clauses: List[ContractClause]  # 关键词匹配到的条款
    match_score: int                       # 关键词命中数
    auto_verdict: str                      # "高度可能匹配" / "可能匹配" / "未匹配到相关条款"


@dataclass
class ContractReviewMapResult:
    """合同审查 Map 阶段完整输出。"""
    success: bool
    contract_name: str
    total_clauses: int                     # 提取到的条款数
    total_patterns: int                    # 知识库风险模式总数
    matches: List[ClausePatternMatch]      # 每个风险模式的匹配结果
    patterns_with_match: int               # 至少匹配到1条条款的模式数
    patterns_without_match: int            # 未匹配到任何条款的模式数
    extraction_warnings: List[str]


# ═══════════════════════════════════════════════════════════════
# 1. 合同条款提取
# ═══════════════════════════════════════════════════════════════

def extract_contract_clauses(text: str) -> List[ContractClause]:
    """从合同 Markdown 中提取全部条款。

    按「第X条」标题分块，再按「X.Y」子标题分子条款。
    跳过签章页等非条款内容。
    """
    clauses: List[ContractClause] = []

    # 按「第X条」分块
    article_blocks = re.split(r'\n(?=## 第[一二三四五六七八九十百千]+条\s)', text)

    for block in article_blocks:
        # 提取条标题
        article_match = re.match(r'##\s*(第[一二三四五六七八九十百千]+条)\s*(.*)', block)
        if not article_match:
            continue

        article_num = article_match.group(1)
        article_title = article_match.group(2).strip()

        # 跳过签章页
        if '签章' in article_title or '签章' in block[:100]:
            continue

        # 按「X.Y」子标题分块
        section_pattern = rf'(?:###\s*({re.escape(article_num)}\.?\s*[\.\d]*)\s*(.*?)\n)(.*?)(?=\n###\s*{re.escape(article_num)}\.?\s*[\.\d]|\n##\s*第|\Z)'
        # 简化：直接找 ### 标题
        sub_blocks = re.split(r'\n(?=###\s)', block)

        if len(sub_blocks) == 1:
            # 没有子标题，整条作为一个条款
            clauses.append(ContractClause(
                article_num=article_num,
                article_title=article_title,
                section_num="",
                section_title="",
                content=block.strip(),
                raw_text=block.strip(),
            ))
        else:
            # 第一个块是条标题后的导语
            for sub in sub_blocks[1:]:
                sub_match = re.match(r'###\s*([\d.]+)\s*(.*)', sub)
                if sub_match:
                    section_num = sub_match.group(1)
                    section_title = sub_match.group(2).strip()
                    # 提取内容（到下一个 ### 或 ## 之前）
                    content = sub[sub_match.end():].strip()
                    clauses.append(ContractClause(
                        article_num=article_num,
                        article_title=article_title,
                        section_num=section_num,
                        section_title=section_title,
                        content=content[:500],  # 截取前500字符
                        raw_text=sub.strip(),
                    ))

    return clauses


# ═══════════════════════════════════════════════════════════════
# 2. 知识库风险模式提取
# ═══════════════════════════════════════════════════════════════

def extract_kb_risk_patterns(text: str) -> List[RiskPattern]:
    """从合同风险知识库中提取全部风险模式。

    按「第X章」→「X.Y 风险点」层级解析。
    """
    patterns: List[RiskPattern] = []

    # 按章分块
    chapter_blocks = re.split(r'\n(?=## 第\d+章\s)', text)

    for ch_block in chapter_blocks:
        ch_match = re.match(r'##\s*(第\d+章\s*.+)', ch_block)
        if not ch_match:
            continue
        chapter = ch_match.group(1).strip()

        # 按风险点分块
        risk_blocks = re.split(r'\n(?=###\s*\d+\.\d+\s)', ch_block)

        for r_block in risk_blocks[1:]:  # 跳过章导语
            r_match = re.match(r'###\s*(\d+\.\d+)\s*(.+)', r_block)
            if not r_match:
                continue

            pattern_id = r_match.group(1)
            pattern_name = r_match.group(2).strip()

            # 提取典型话术
            wording = _extract_section(r_block, '典型表现', '风险后果')
            consequence = _extract_section(r_block, '风险后果', '修改建议')
            suggestion = _extract_section(r_block, '修改建议', None)

            # 生成关键词
            keywords = _generate_keywords(pattern_name, wording, chapter)

            # 源行范围
            source_lines = r_block[:100].replace('\n', ' ')

            patterns.append(RiskPattern(
                chapter=chapter,
                pattern_id=pattern_id,
                pattern_name=pattern_name,
                typical_wording=wording[:300],
                risk_consequence=consequence[:300],
                suggestion=suggestion[:300],
                keywords=keywords,
                source_lines=source_lines,
            ))

    return patterns


def _extract_section(text: str, start_marker: str, end_marker: Optional[str]) -> str:
    """提取两个标记之间的文本。"""
    pattern = rf'{start_marker}.*?\n(.*?)'
    if end_marker:
        pattern += rf'(?=\n\*\*{end_marker}|$)'
    else:
        pattern += r'(?=\n---|\n##|\Z)'
    m = re.search(pattern, text, re.DOTALL)
    return m.group(1).strip()[:500] if m else ""


def _generate_keywords(name: str, wording: str, chapter: str) -> List[str]:
    """从风险模式文本中提取匹配关键词。"""
    combined = f"{name} {wording} {chapter}"
    kw_map = {
        '知识产权': ['知识产权', '著作权', '专利权', '源代码', '技术成果', '归属', '共有', '开源'],
        '付款': ['付款', '首付款', '支付', '款项', '发票', '违约金', '逾期'],
        '违约金': ['违约金', '千分之', '逾期', '赔偿', '责任上限', '累计赔偿'],
        '违约': ['违约', '赔偿', '责任', '不对等', '间接损失'],
        '验收': ['验收', '交付物', '缺陷', '测试', '视为', '满意', '主观'],
        '保密': ['保密', '秘密', '泄露', '永久', '期限'],
        '争议': ['争议', '管辖', '法院', '仲裁', '诉讼'],
        '数据': ['数据', '个人信息', '隐私', '跨境', '匿名'],
        '期限': ['期限', '终止', '续期', '到期', '解除'],
        '不可抗力': ['不可抗力', '自然灾害', '网络攻击', '技术人员离职'],
        '价格': ['价格', '调价', '单方', '调整', '上涨'],
        '陈述': ['据.*所知', '据卖方所知', '据乙方所知', '保证', '陈述'],
        '衍生': ['衍生数据', '运营数据', '业务数据'],
    }

    all_kw = []
    for category, kws in kw_map.items():
        if any(k in combined for k in [category]):
            all_kw.extend(kws)

    # 去重并限制
    seen = set()
    unique = []
    for k in all_kw:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique[:8]


# ═══════════════════════════════════════════════════════════════
# 3. 关键词预匹配
# ═══════════════════════════════════════════════════════════════

# ── Embedding 语义匹配（替代关键词匹配） ──

_embedder = None

def _get_embedder():
    """懒加载 embedding 模型（复用 RAG 的 bge-small-zh-v1.5，512维）。"""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        _embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)
    return _embedder


def pre_match(
    patterns: List[RiskPattern],
    clauses: List[ContractClause],
) -> List[ClausePatternMatch]:
    """对每个风险模式，用 embedding 语义相似度匹配合同条款。

    替代了原来的关键词列表匹配，精度显著提升：
    - "违约金上限过低" 能匹配到 "累计赔偿责任上限为甲方已支付的服务费用"
    - 不再依赖手工维护的关键词列表
    - 中文语义匹配，对同义词、近义词有天然鲁棒性
    """
    import numpy as np

    model = _get_embedder()

    # 预计算所有 clause 的 embedding
    clause_texts = [f"{c.article_title} {c.section_title} {c.content}" for c in clauses]
    clause_vecs = model.encode(clause_texts, normalize_embeddings=True)

    # 预计算所有 pattern 的 embedding（pattern_name + typical_wording + risk_consequence）
    pattern_texts = [f"{p.pattern_name} {p.typical_wording[:200]} {p.risk_consequence[:200]}" for p in patterns]
    pattern_vecs = model.encode(pattern_texts, normalize_embeddings=True)

    results: List[ClausePatternMatch] = []
    SIM_THRESHOLD_HIGH = 0.65   # 高度可能匹配
    SIM_THRESHOLD_LOW = 0.50    # 可能匹配

    for i, pattern in enumerate(patterns):
        # 计算该 pattern 与所有 clause 的余弦相似度
        sims = np.dot(clause_vecs, pattern_vecs[i])  # 已归一化，点积=余弦

        matched_clauses: List[ContractClause] = []
        total_score = 0.0
        for j, sim in enumerate(sims):
            if sim >= SIM_THRESHOLD_LOW:
                matched_clauses.append(clauses[j])
                total_score += float(sim)

        if len(matched_clauses) >= 2 and total_score >= SIM_THRESHOLD_HIGH * 2:
            verdict = "高度可能匹配"
        elif len(matched_clauses) >= 1:
            verdict = "可能匹配"
        else:
            verdict = "未匹配到相关条款"

        results.append(ClausePatternMatch(
            pattern=pattern,
            matched_clauses=matched_clauses,
            match_score=int(total_score * 10),
            auto_verdict=verdict,
        ))

    return results


# ═══════════════════════════════════════════════════════════════
# 4. 主入口
# ═══════════════════════════════════════════════════════════════

def run_contract_map(
    contract_text: str,
    kb_text: str,
) -> ContractReviewMapResult:
    """执行合同审查 Map 阶段。

    Args:
        contract_text: 合同 Markdown 全文。
        kb_text: 合同风险知识库 Markdown 全文。

    Returns:
        ContractReviewMapResult。
    """
    warnings: List[str] = []

    try:
        clauses = extract_contract_clauses(contract_text)
        if not clauses:
            return ContractReviewMapResult(
                success=False, contract_name="", total_clauses=0, total_patterns=0,
                matches=[], patterns_with_match=0, patterns_without_match=0,
                extraction_warnings=["未能从合同中提取到任何条款"],
            )

        patterns = extract_kb_risk_patterns(kb_text)
        if not patterns:
            return ContractReviewMapResult(
                success=False, contract_name="", total_clauses=len(clauses), total_patterns=0,
                matches=[], patterns_with_match=0, patterns_without_match=0,
                extraction_warnings=["未能从知识库中提取到任何风险模式"],
            )

        matches = pre_match(patterns, clauses)

        with_match = sum(1 for m in matches if m.matched_clauses)
        without_match = len(matches) - with_match

        # 提取合同名称
        name_match = re.search(r'#\s*(.+)', contract_text)
        contract_name = name_match.group(1).strip() if name_match else "技术服务合同"

        return ContractReviewMapResult(
            success=True,
            contract_name=contract_name,
            total_clauses=len(clauses),
            total_patterns=len(patterns),
            matches=matches,
            patterns_with_match=with_match,
            patterns_without_match=without_match,
            extraction_warnings=warnings,
        )

    except Exception as e:
        return ContractReviewMapResult(
            success=False, contract_name="", total_clauses=0, total_patterns=0,
            matches=[], patterns_with_match=0, patterns_without_match=0,
            extraction_warnings=[f"Map 阶段异常: {e}"],
        )


# ═══════════════════════════════════════════════════════════════
# 5. 格式化输出
# ═══════════════════════════════════════════════════════════════

def format_contract_map_context(result: ContractReviewMapResult) -> str:
    """将 Map 结果格式化为结构化 Markdown，供 Reduce 阶段 LLM 使用。"""
    if not result.success:
        return f"## [系统预提取] 合同条款匹配失败\n\n{'; '.join(result.extraction_warnings)}"

    parts = [
        "## [系统预提取] 合同条款与风险模式预匹配结果",
        "",
        f"> 合同：**{result.contract_name}** | 提取条款：{result.total_clauses} 条 | "
        f"知识库风险模式：{result.total_patterns} 个 | "
        f"预匹配命中：{result.patterns_with_match} 个 | "
        f"未匹配：{result.patterns_without_match} 个",
        "",
        "以下为知识库全部风险模式与合同条款的关键词预匹配结果。",
        "**请逐条进行最终语义判定**：确认是否真正匹配、给出风险分析和修改建议。",
        '预匹配标注"未匹配到相关条款"的模式，请人工检查合同是否有隐藏的对应风险。',
        "",
    ]

    for m in result.matches:
        p = m.pattern
        icon = {"高度可能匹配": "🔴", "可能匹配": "🟡", "未匹配到相关条款": "⚪"}
        verdict_icon = icon.get(m.auto_verdict, "⚪")

        parts.append(f"### {verdict_icon} [{p.pattern_id}] {p.pattern_name}")
        parts.append(f"**来源**：知识库「{p.chapter}」")
        parts.append(f"**预匹配判定**：{m.auto_verdict}（关键词命中 {m.match_score} 次）")
        parts.append(f"**典型话术**：{p.typical_wording[:200]}")
        parts.append(f"**风险后果**：{p.risk_consequence[:200]}")

        if m.matched_clauses:
            parts.append("**匹配到的合同条款**：")
            for c in m.matched_clauses:
                parts.append(f"- 第{c.article_num} {c.article_title} {c.section_num} {c.section_title}")
                parts.append(f"  > {c.content[:150]}...")
        else:
            parts.append("**匹配到的合同条款**：（无 — 请人工复核是否有隐藏风险）")

        parts.append("")  # 空行分隔

    parts.append("---")
    parts.append(f"**预匹配统计**：{result.total_patterns} 个风险模式中，")
    parts.append(f"- 高度可能匹配：{sum(1 for m in result.matches if m.auto_verdict == '高度可能匹配')} 个")
    parts.append(f"- 可能匹配：{sum(1 for m in result.matches if m.auto_verdict == '可能匹配')} 个")
    parts.append(f"- 未匹配：{result.patterns_without_match} 个（需人工复核）")
    parts.append("")
    parts.append("**请基于以上预匹配结果，逐条确认风险模式是否真正适用，对适用的模式给出详细风险分析和修改建议。**")

    return "\n".join(parts)
