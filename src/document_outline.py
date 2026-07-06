"""文档大纲导航 — 为超长文档构建结构化大纲，指导 Map 阶段精准定位。

解决的问题：
- 旧方案依赖精确标题文本匹配（如 "## 五、担保方案"），不同文档格式不同会失败
- 超长文档（> 5000 字）的 Map 提取需要先了解"文档里有什么"

能力：
1. 解析 Markdown 标题层级 → 构建树形大纲
2. 模糊章节查找：用关键词在大纲中定位最相关的章节
3. 为 Map 提取器提供"导航地图"
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Heading:
    """文档中的一个标题节点。"""
    level: int              # 标题层级（1=#, 2=##, 3=###）
    title: str              # 标题文本（去除标记）
    line_number: int        # 在原文中的行号
    char_offset: int        # 在原文中的字符偏移
    parent: Optional['Heading'] = None   # 父标题
    children: List['Heading'] = field(default_factory=list)


@dataclass
class DocumentOutline:
    """文档的完整大纲结构。"""
    headings: List[Heading]          # 所有标题（扁平列表）
    root: Heading                    # 虚拟根节点
    total_chars: int                 # 文档总字符数
    total_headings: int              # 标题总数


def build_outline(text: str) -> DocumentOutline:
    """从 Markdown 文本构建文档大纲。

    识别 # / ## / ### / #### 四级标题，构建父子层级关系。
    兼容以下中文编号格式：
    - "## 五、担保方案"
    - "## 第3章 违约责任"
    - "### 5.2 抵押率计算"
    """
    headings: List[Heading] = []
    lines = text.split('\n')

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith('#'):
            continue

        # 计算标题层级
        level = 0
        for ch in stripped:
            if ch == '#':
                level += 1
            else:
                break

        if level < 1 or level > 4:
            continue

        title = stripped[level:].strip()
        if not title:
            continue

        char_offset = sum(len(l) + 1 for l in lines[:i])

        headings.append(Heading(
            level=level,
            title=title,
            line_number=i + 1,
            char_offset=char_offset,
        ))

    # 构建父子关系
    root = Heading(level=0, title="(root)", line_number=0, char_offset=0)
    stack = [root]

    for h in headings:
        # 弹出比当前层级深或同级的节点
        while stack and stack[-1].level >= h.level:
            stack.pop()
        if stack:
            h.parent = stack[-1]
            stack[-1].children.append(h)
        stack.append(h)

    return DocumentOutline(
        headings=headings,
        root=root,
        total_chars=len(text),
        total_headings=len(headings),
    )


def find_section_by_outline(
    text: str,
    outline: DocumentOutline,
    keywords: List[str],
) -> Optional[Tuple[int, int]]:
    """通过大纲 + 关键词模糊定位章节内容。

    比 _find_section() 的精确标题匹配更鲁棒：
    - 支持多个关键词的语义组合
    - 即使标题写法不同（如 "五、担保方案" vs "第五节 担保方案"）也能找到

    Returns:
        (start_char, end_char) 或 None
    """
    # 在大纲中查找包含最多关键词的标题
    best_heading = None
    best_score = 0

    for h in outline.headings:
        score = sum(1 for kw in keywords if kw in h.title)
        if score > best_score:
            best_score = score
            best_heading = h

    if not best_heading or best_score == 0:
        return None

    # 确定结束位置：下一个同级或上级标题之前
    start = best_heading.char_offset
    end = len(text)

    for h in outline.headings:
        if h.char_offset > start and h.level <= best_heading.level:
            end = h.char_offset
            break

    return (start, end)


def get_section_text(text: str, start: int, end: int) -> str:
    """根据字符偏移提取章节文本。"""
    return text[start:end].strip()


def format_outline_tree(outline: DocumentOutline) -> str:
    """将大纲格式化为树形文本（用于注入 LLM 上下文）。"""
    lines = ["## 文档大纲", "", f"共 {outline.total_headings} 个标题，{outline.total_chars} 字符", ""]

    def _render(heading: Heading, indent: int = 0):
        prefix = "  " * indent + ("- " if heading.level > 0 else "")
        if heading.level > 0:
            level_mark = "#" * heading.level
            lines.append(f"{prefix}[行{heading.line_number}] {heading.title}")
        for child in heading.children:
            _render(child, indent + 1)

    for child in outline.root.children:
        _render(child, 0)

    return "\n".join(lines)


def get_outline_context(outline: DocumentOutline, max_items: int = 30) -> str:
    """生成精简的大纲上下文（用于注入 LLM prompt）。

    只保留前 max_items 个最重要的标题。
    """
    items = []
    for h in outline.headings[:max_items]:
        indent = "  " * (h.level - 1)
        items.append(f"{indent}- {h.title} (行{h.line_number})")
    return "\n".join(items)
