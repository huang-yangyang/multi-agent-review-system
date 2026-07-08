"""文本分块器。

支持 Markdown 结构感知分块 + 递归字符分割。
- 非 MD 文本：按分隔符递归切分 + 滑动窗口重叠
- MD 文本：在标题处分段，段内递归切分，每个子块保留标题锚点
- 输出前剥离 Markdown 内联语法噪声
"""

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", " "]

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

_INLINE_CLEANERS = [
    (re.compile(r"!\[([^\]]*)\]\([^\)]+\)"), r"\1"),
    (re.compile(r"\[([^\]]+)\]\([^\)]+\)"), r"\1"),
    (re.compile(r"\*\*([^\*]+)\*\*"), r"\1"),
    (re.compile(r"__([^_]+)__"), r"\1"),
    (re.compile(r"(?<!\*)\*([^\*]+)\*(?!\*)"), r"\1"),
    (re.compile(r"(?<!_)_([^_]+)_(?!_)"), r"\1"),
    (re.compile(r"`{1,3}[^`]*`{1,3}"), ""),
    (re.compile(r"~~([^~]+)~~"), r"\1"),
    (re.compile(r"^[>\s]*>+\s?", re.MULTILINE), ""),
    (re.compile(r"^[\-\*\+]\s+", re.MULTILINE), ""),
    (re.compile(r"^\d+\.\s+", re.MULTILINE), ""),
    (re.compile(r"\|\s*[-:]+\s*\|"), ""),
    (re.compile(r"\|"), " "),
    (re.compile(r"\n{3,}"), "\n\n"),
]


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: List[str] | None = None,
    is_markdown: bool = False,
) -> List[str]:
    text = text.strip()
    if not text:
        return []

    if separators is None:
        separators = DEFAULT_SEPARATORS

    if is_markdown:
        raw_chunks = _split_markdown_by_headings(text, chunk_size, chunk_overlap)
    else:
        raw_chunks = _split_recursive(text, separators, chunk_size)
        if chunk_overlap > 0 and len(raw_chunks) > 1:
            raw_chunks = _apply_sliding_window(raw_chunks, chunk_overlap)

    cleaned_chunks = []
    for chunk in raw_chunks:
        cleaned = _strip_markdown_inline(chunk).strip()
        if cleaned:
            cleaned_chunks.append(cleaned)

    logger.info(f"分块完成: {len(cleaned_chunks)} 块 (size={chunk_size}, md={is_markdown})")
    return cleaned_chunks


def _split_markdown_by_headings(text: str, chunk_size: int, chunk_overlap: int = 0) -> List[str]:
    """Markdown 结构感知分块：只在叶子节点产生 chunk，中间节点仅作祖先链。

    叶子节点 = 不包含更深层级子标题的 section。
    每个 chunk 携带完整祖先链（如 "## 二、操作系统\n### 2.1 进程与线程"），
    保证检索「操作系统有哪些内容」时能拉回所有子节的代表性 chunk。
    当叶子节点的正文超过 chunk_size 时，内部使用递归切割+滑动窗口重叠。
    """
    heading_matches = list(_HEADING_RE.finditer(text))

    if not heading_matches:
        return _split_recursive(text, DEFAULT_SEPARATORS, chunk_size)

    # ── 解析所有 sections ──
    sections = []
    for i, match in enumerate(heading_matches):
        level = len(match.group(1))
        heading_text = match.group(0).strip()
        content_start = match.end()
        content_end = heading_matches[i + 1].start() if i + 1 < len(heading_matches) else len(text)
        body = text[content_start:content_end].strip()
        sections.append((level, heading_text, body))

    # ── 处理序言（第一个标题之前的文本）──
    first_h_start = heading_matches[0].start()
    preface = text[:first_h_start].strip()
    result: List[str] = []
    if preface:
        result.extend(_split_recursive(preface, DEFAULT_SEPARATORS, chunk_size))

    # ── 递归处理：只输出叶子节点 ──
    ancestry: List[tuple] = []  # [(level, heading_text), ...]

    def _has_deeper_headings(body_text: str, current_level: int) -> bool:
        for m in _HEADING_RE.finditer(body_text):
            if len(m.group(1)) > current_level:
                return True
        return False

    def _extract_preamble(body_text: str) -> str:
        """提取 body 中第一个更深层标题之前的文本。"""
        deeper_matches = [m for m in _HEADING_RE.finditer(body_text) if len(m.group(1)) > ancestry[-1][0]]
        if deeper_matches:
            return body_text[:deeper_matches[0].start()].strip()
        return body_text

    def _emit_leaf(body_text: str):
        """将当前 ancestry + body 输出为一个或多个 chunk。"""
        chain_parts = [h for _, h in ancestry]
        chain_text = "\n".join(chain_parts)
        section_text = f"{chain_text}\n{body_text}" if body_text else chain_text

        if len(section_text) <= chunk_size:
            result.append(section_text)
        else:
            sub_chunks = _split_recursive(body_text, DEFAULT_SEPARATORS, chunk_size)
            # 对长段落的子块应用滑动窗口重叠，保证上下文连贯
            if chunk_overlap > 0 and len(sub_chunks) > 1:
                sub_chunks = _apply_sliding_window(sub_chunks, chunk_overlap)
            for sc in sub_chunks:
                result.append(f"{chain_text}\n{sc}")

    def _process(level: int, heading_text: str, body_text: str):
        # 更新祖先链
        while ancestry and ancestry[-1][0] >= level:
            ancestry.pop()
        ancestry.append((level, heading_text))

        if not body_text:
            return

        if not _has_deeper_headings(body_text, level):
            # 叶子节点：直接输出
            _emit_leaf(body_text)
        else:
            # 中间节点：先输出 preamble，再递归处理子 section
            preamble = _extract_preamble(body_text)
            if preamble:
                _emit_leaf(preamble)

            sub_matches = list(_HEADING_RE.finditer(body_text))
            for i, sm in enumerate(sub_matches):
                sl = len(sm.group(1))
                sh = sm.group(0).strip()
                ss = sm.end()
                se = sub_matches[i + 1].start() if i + 1 < len(sub_matches) else len(body_text)
                sb = body_text[ss:se].strip()
                _process(sl, sh, sb)

            # 恢复祖先链
            ancestry.pop()

    for lv, hd, bd in sections:
        _process(lv, hd, bd)

    return result


def _strip_markdown_inline(text: str) -> str:
    result = text
    for pattern, replacement in _INLINE_CLEANERS:
        result = pattern.sub(replacement, result)
    return result


def _split_recursive(text: str, separators: List[str], chunk_size: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0] if separators else None
    if sep is None:
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    splits = text.split(sep)
    result = []
    buffer = ""

    for part in splits:
        candidate = buffer + (sep if buffer else "") + part
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            if buffer:
                result.append(buffer)
            if len(part) > chunk_size:
                if len(separators) > 1:
                    sub_chunks = _split_recursive(part, separators[1:], chunk_size)
                    result.extend(sub_chunks[:-1])
                    buffer = sub_chunks[-1]
                else:
                    hard_chunks = [part[i:i+chunk_size] for i in range(0, len(part), chunk_size)]
                    result.extend(hard_chunks[:-1])
                    buffer = hard_chunks[-1]
            else:
                buffer = part

    if buffer.strip():
        result.append(buffer)

    return result


def _apply_sliding_window(chunks: List[str], overlap: int) -> List[str]:
    if len(chunks) <= 1:
        return chunks

    overlapped = []
    for i, chunk in enumerate(chunks):
        if i < len(chunks) - 1:
            overlap_text = chunks[i + 1][:overlap]
            overlapped.append(chunk + "\n" + overlap_text)
        else:
            overlapped.append(chunk)

    return overlapped
