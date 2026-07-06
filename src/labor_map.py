"""劳动法 Map 提取器 — 双格式自适应。"""
import re
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class LaborClause:
    clause_id: str; title: str; content: str; raw_text: str = ""
@dataclass
class LaborRiskPattern:
    chapter: str; pattern_id: str; pattern_name: str
    typical_wording: str; risk_consequence: str; suggestion: str
    keywords: List[str] = field(default_factory=list)
@dataclass
class LaborPatternMatch:
    pattern: LaborRiskPattern; matched_clauses: List[LaborClause]
    match_score: int; auto_verdict: str
@dataclass
class LaborReviewMapResult:
    success: bool; total_clauses: int; total_patterns: int
    matches: List[LaborPatternMatch]; patterns_with_match: int
    patterns_without_match: int; extraction_warnings: List[str]

_LABOR_KW = [
    "合同期限","试用期","工作内容","工作地点","工作时间",
    "劳动报酬","工资","社会保险","劳动保护","规章制度",
    "解除","终止","违约责任","违约金","赔偿责任",
    "保密","竞业限制","知识产权","经济补偿",
    "调岗","调薪","加班","休假","培训","服务期",
    "女职工","孕期","产期","哺乳期",
]

def extract_labor_clauses(text: str) -> List[LaborClause]:
    clauses = []; lines = text.split('\n'); cur, cur_id = "", ""
    for line in lines:
        s = line.strip()
        if not s: continue
        for kw in _LABOR_KW:
            if kw in s:
                if cur and len(cur) > 20: clauses.append(LaborClause(cur_id or kw, kw, cur[:500]))
                cur_id, cur = kw, s; break
        else:
            if cur: cur += "\n" + s
    if cur and len(cur) > 20: clauses.append(LaborClause(cur_id or "其他", cur_id or "其他", cur[:500]))
    return clauses

def extract_labor_risk_patterns(text: str) -> List[LaborRiskPattern]:
    p = _parse_a(text); return p if p else _parse_b(text)

def _parse_a(text: str) -> List[LaborRiskPattern]:
    patterns = []
    for ch in re.split(r'\n(?=## 第\d+章\s)', text):
        m = re.match(r'##\s*(第\d+章\s*.+)', ch)
        chapter = m.group(1).strip() if m else "风险模式"
        for rb in re.split(r'\n(?=###\s*\d+\.\d+\s)', ch)[1:]:
            rm = re.match(r'###\s*(\d+\.\d+)\s*(.+)', rb)
            if not rm: continue
            pid, pname = rm.group(1), rm.group(2).strip()
            w = _fld(rb,'典型表现','风险后果'); c = _fld(rb,'风险后果','修改建议'); s = _fld(rb,'修改建议',None)
            patterns.append(LaborRiskPattern(chapter,pid,pname,w[:300],c[:300],s[:300],_kw(pname,w,chapter)))
    return patterns

def _parse_b(text: str) -> List[LaborRiskPattern]:
    patterns = []
    for ch in re.split(r'\n(?=## 第[一二三四五六七八九十\d]+章\s)', text):
        m = re.match(r'##\s*(第[一二三四五六七八九十\d]+章\s*.+)', ch)
        chapter = m.group(1).strip() if m else "劳动法条款"
        sep = r'\*\*第[一二三四五六七八九十\d]+条\*\*'
        articles = re.findall(sep + r'[\s\S]*?(?=' + sep + r'|\n##|\Z)', ch)
        if not articles:
            for i, p in enumerate([x.strip() for x in ch.split('\n\n') if len(x.strip()) > 50][:30]):
                patterns.append(LaborRiskPattern(chapter,f"{chapter}.{i+1}",p[:80].replace('\n',' '),p[:300],"[需 LLM 推理]","[需 LLM 推理]",_kw(p[:100],p,chapter)))
            continue
        for i, a in enumerate(articles[:30]):
            tm = re.match(sep + r'\s*(.+)', a)
            name = tm.group(1).strip()[:80] if tm else f"第{i+1}条"
            patterns.append(LaborRiskPattern(chapter,f"{chapter}.{i+1}",name,a[:300],"[需 LLM 推理]","[需 LLM 推理]",_kw(name,a,chapter)))
    return patterns

def _fld(text,start,end):
    p = rf'\*\*{start}.*?\*\*\s*\n(.*?)' + (rf'(?=\*\*{end}|$)' if end else r'(?=\n---|\n##|\Z)')
    m = re.search(p, text, re.DOTALL); return m.group(1).strip()[:300] if m else ""

def _kw(name,wording,chapter):
    c = f"{name} {wording} {chapter}"
    m = {'调岗':['调岗','调整.*岗位'],'调薪':['调薪','降薪'],'工时':['不定时','综合计算工时'],'违约金':['违约金','赔偿金'],'竞业':['竞业限制','不得.*从事'],'保密':['保密','泄露'],'知识产权':['知识产权','专利'],'试用期':['试用期','试用.*月'],'社保':['社会保险','社保'],'补偿':['经济补偿'],'解除':['解除.*合同','辞退'],'加班':['加班','加班费'],'工资':['工资','劳动报酬'],'期限':['合同期限','无固定期限'],'女职工':['孕期','产期'],'培训':['培训','服务期'],'劳动合同':['劳动合同','用人单位'],'休息':['休息','年假']}
    kw = []; [kw.extend(ks) for cat,ks in m.items() if cat in c]; return list(dict.fromkeys(kw))[:8]

_embedder = None
def _get_embedder():
    global _embedder
    if _embedder is None:
        import os; os.environ.setdefault("HF_HUB_OFFLINE","1")
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("BAAI/bge-small-zh-v1.5", local_files_only=True)
    return _embedder

def pre_match_labor(patterns, clauses):
    import numpy as np
    m = _get_embedder()
    cv = m.encode([f"{c.title} {c.content}" for c in clauses], normalize_embeddings=True)
    pv = m.encode([f"{p.pattern_name} {p.typical_wording[:200]} {p.risk_consequence[:200]}" for p in patterns], normalize_embeddings=True)
    results = []
    for i, p in enumerate(patterns):
        sims = np.dot(cv, pv[i]); matched, score = [], 0.0
        for j, s in enumerate(sims):
            if s >= 0.50: matched.append(clauses[j]); score += float(s)
        v = "高度可能匹配" if (len(matched) >= 2 and score >= 1.3) else ("可能匹配" if matched else "未匹配到相关条款")
        results.append(LaborPatternMatch(p, matched, int(score*10), v))
    return results

def run_labor_map(contract_text, kb_text):
    try:
        clauses = extract_labor_clauses(contract_text)
        if not clauses: return LaborReviewMapResult(False,0,0,[],0,0,["未能提取条款"])
        patterns = extract_labor_risk_patterns(kb_text)
        if not patterns: return LaborReviewMapResult(False,len(clauses),0,[],0,0,["未能提取风险模式"])
        matches = pre_match_labor(patterns, clauses)
        wm = sum(1 for m in matches if m.matched_clauses)
        return LaborReviewMapResult(True,len(clauses),len(patterns),matches,wm,len(matches)-wm,[])
    except Exception as e:
        return LaborReviewMapResult(False,0,0,[],0,0,[f"异常:{e}"])

def format_labor_map_context(result):
    if not result.success: return f"## [系统预提取] 匹配失败\n\n{'; '.join(result.extraction_warnings)}"
    p = [f"## [系统预提取] 劳动合同条款与风险模式预匹配结果","",
         f"> 条款:{result.total_clauses} | 模式:{result.total_patterns} | 命中:{result.patterns_with_match} | 未匹配:{result.patterns_without_match}","",
         "以下为知识库全部风险模式与合同条款的语义预匹配结果。**请逐条最终判定。**",""]
    icons = {"高度可能匹配":"🔴","可能匹配":"🟡","未匹配到相关条款":"⚪"}
    for m in result.matches:
        ic = icons.get(m.auto_verdict,"⚪"); pp = m.pattern
        p.append(f"### {ic} [{pp.pattern_id}] {pp.pattern_name}\n**来源**: {pp.chapter} | **预匹配**: {m.auto_verdict}({m.match_score}分)")
        if pp.typical_wording: p.append(f"**条款**: {pp.typical_wording[:200]}")
        if m.matched_clauses:
            p.append("**匹配合同条款**:")
            for c in m.matched_clauses: p.append(f"- {c.title}: {c.content[:100]}...")
        else: p.append("**匹配合同条款**: (无)")
        p.append("")
    hi = sum(1 for m in result.matches if m.auto_verdict=="高度可能匹配")
    lo = sum(1 for m in result.matches if m.auto_verdict=="可能匹配")
    p.append(f"---\n**统计**: {result.total_patterns}模式 🔴{hi} 🟡{lo} ⚪{result.patterns_without_match}")
    return "\n".join(p)
