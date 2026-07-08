"""文档解析器。

支持 PDF（PyPDF2 → PaddleOCR 回退）、DOCX（python-docx）、Markdown、TXT 四种格式。
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}

OCR_FALLBACK_THRESHOLD = 50


def parse_document(file_path: str) -> tuple:
    """解析文档，返回 (text, is_markdown) 元组。

    - PDF：PyMuPDF 字体分析 → PyPDF2 → PaddleOCR 回退链
    - DOCX：python-docx 读取段落样式，Heading → Markdown 标题
    - MD：直接读取（is_markdown=True）
    - TXT：直接读取（is_markdown=False）

    Args:
        file_path: 文件路径

    Returns:
        (text, is_markdown): 文本内容和是否包含 Markdown 标题结构
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式: {ext}，支持: {SUPPORTED_EXTENSIONS}")

    logger.info(f"解析文档: {path.name} ({ext})")

    if ext == ".pdf":
        return _parse_pdf(file_path)
    elif ext == ".docx":
        return _parse_docx(file_path)
    elif ext == ".md":
        return _parse_markdown(file_path), True
    elif ext == ".txt":
        return _parse_text(file_path), False
    else:
        raise ValueError(f"未实现的解析器: {ext}")


def _parse_pdf(file_path: str) -> tuple:
    """解析 PDF 文件，尝试通过字体大小检测标题层级。

    优先使用 PyMuPDF (fitz) 获取文本块的字体信息，根据字体大小
    推断标题层级并转为 Markdown 格式。若 PyMuPDF 不可用或提取
    文本不足，回退到 PyPDF2 → PaddleOCR。

    返回 (text, is_markdown) 元组。
    """
    # ── 优先尝试 PyMuPDF 结构化解析 ──
    try:
        text, has_headings = _parse_pdf_with_fitz(file_path)
        if len(text.strip()) >= OCR_FALLBACK_THRESHOLD:
            return text, has_headings
        logger.info("PyMuPDF 提取文本不足，回退到 PyPDF2")
    except ImportError:
        logger.warning("PyMuPDF 未安装，跳过字体分析。安装: pip install PyMuPDF")
    except Exception as e:
        logger.warning(f"PyMuPDF 解析失败: {e}，回退到 PyPDF2")

    # ── 回退到 PyPDF2 ──
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise RuntimeError("PyPDF2 未安装，请执行: pip install PyPDF2")

    reader = PdfReader(file_path)
    texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            texts.append(text)

    extracted = "\n\n".join(texts) if texts else ""

    if len(extracted.strip()) >= OCR_FALLBACK_THRESHOLD:
        return extracted, False

    logger.info(f"PyPDF2 提取文本不足 {OCR_FALLBACK_THRESHOLD} 字符，回退到 PaddleOCR")
    ocr_text = _ocr_pdf(file_path)
    if ocr_text:
        return ocr_text, False

    return extracted, False


def _parse_pdf_with_fitz(file_path: str) -> tuple:
    """使用 PyMuPDF 解析 PDF，基于字体大小推断标题层级。

    策略：
      1. 遍历每页的 text blocks，获取 span 级别的字体大小
      2. 统计所有字体大小的分布，确定正文基准大小（出现频率最高）
      3. 字体大小 > 基准 * 1.3 且加粗 → H1
         字体大小 > 基准 * 1.15 且加粗 → H2
         字体大小 > 基准 * 1.05 且加粗 → H3
      4. 将标题转为 Markdown 格式（#/##/###）

    返回 (markdown_text, has_headings)。
    """
    import fitz  # PyMuPDF

    doc = fitz.open(file_path)

    # ── 第一遍：收集所有文本块和字体信息 ──
    all_blocks = []  # [(page_num, text, font_size, is_bold), ...]
    font_sizes = []  # 所有非空文本的字体大小

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
        for block in blocks:
            if block.get("type") != 0:  # 0 = text block
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue
                    font_size = round(span.get("size", 12), 1)
                    font_flags = span.get("flags", 0)
                    # flag bit 4 (16) = bold
                    is_bold = bool(font_flags & 16)
                    font_name = span.get("font", "")
                    # 字体名包含 Bold/bold 也视为加粗
                    if "bold" in font_name.lower() or "black" in font_name.lower():
                        is_bold = True
                    all_blocks.append((page_num, text, font_size, is_bold))
                    font_sizes.append(font_size)

    doc.close()

    if not all_blocks:
        return "", False

    # ── 确定正文基准字体大小 ──
    from collections import Counter
    size_counter = Counter(font_sizes)
    # 取出现频率最高的字体大小作为正文基准
    base_size = size_counter.most_common(1)[0][0]

    # ── 标题检测阈值 ──
    h1_threshold = base_size * 1.3
    h2_threshold = base_size * 1.15
    h3_threshold = base_size * 1.05

    # ── 第二遍：构建 Markdown 文本 ──
    lines = []
    has_headings = False
    current_paragraph = []

    for page_num, text, font_size, is_bold in all_blocks:
        if is_bold and font_size >= h1_threshold:
            # H1
            if current_paragraph:
                lines.append("".join(current_paragraph))
                current_paragraph = []
            lines.append(f"\n# {text}\n")
            has_headings = True
        elif is_bold and font_size >= h2_threshold:
            # H2
            if current_paragraph:
                lines.append("".join(current_paragraph))
                current_paragraph = []
            lines.append(f"\n## {text}\n")
            has_headings = True
        elif is_bold and font_size >= h3_threshold:
            # H3
            if current_paragraph:
                lines.append("".join(current_paragraph))
                current_paragraph = []
            lines.append(f"\n### {text}\n")
            has_headings = True
        else:
            # 正文：累积到当前段落
            current_paragraph.append(text)
            # 句末标点视为段落分隔
            if text.endswith(("。", "！", "？", ".", "!", "?", "；", ";")):
                lines.append("".join(current_paragraph))
                current_paragraph = []

    if current_paragraph:
        lines.append("".join(current_paragraph))

    result = "\n\n".join(line for line in lines if line.strip())
    logger.info(
        f"PyMuPDF 解析完成: {len(result)} 字符, "
        f"基准字体={base_size}, 标题检测={'是' if has_headings else '否'}"
    )
    return result, has_headings


def _ocr_pdf(file_path: str) -> str:
    try:
        import fitz  # pymupdf
    except ImportError:
        logger.warning("PyMuPDF 未安装，无法渲染 PDF 页面，跳过 OCR。安装: pip install PyMuPDF")
        return ""

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.warning("PaddleOCR 未安装，跳过 OCR。安装: pip install paddlepaddle paddleocr")
        return ""

    doc = fitz.open(file_path)
    ocr = PaddleOCR(lang="ch")  # V3 API: 默认 PP-OCRv6，无需额外参数
    page_texts = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=200)
        img_bytes = pix.tobytes("png")

        # PaddleOCR V3: 使用 predict() 替代已弃用的 ocr()
        result = ocr.predict(img_bytes)
        if result and result[0]:
            # V3 返回格式略有不同，兼容处理
            lines = []
            for item in result[0]:
                if isinstance(item, dict) and "rec_text" in item:
                    lines.append(item["rec_text"])
                elif isinstance(item, list) and len(item) > 1 and isinstance(item[1], str):
                    lines.append(item[1])
                elif isinstance(item, list) and len(item) > 1:
                    lines.append(item[1][0] if isinstance(item[1], list) else str(item))
            if lines:
                page_texts.append("\n".join(lines))
        else:
            logger.debug(f"OCR 未识别到文字: 第 {i + 1} 页")

    doc.close()
    logger.info(f"PaddleOCR 完成 {len(page_texts)}/{len(doc)} 页识别")

    return "\n\n".join(page_texts)


def _parse_docx(file_path: str) -> tuple:
    """解析 DOCX 文件，保留标题层级结构。

    使用 python-docx 读取段落样式，将 Heading 1/2/3/4/5/6 转换为
    Markdown 标题（#/##/###/####/#####/######），正文段落保持原样。
    返回 (text, is_markdown) 元组：
      - is_markdown=True 表示文本包含 Markdown 标题，可走标题感知分块
      - is_markdown=False 表示无标题样式，走递归字符分割
    """
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx 未安装，请执行: pip install python-docx")

    doc = Document(file_path)
    lines = []
    has_headings = False

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            lines.append("")  # 保留空行作为段落分隔
            continue

        style_name = ""
        if para.style:
            style_name = para.style.name or ""

        # 检测 Word 内置标题样式：Heading 1 ~ Heading 6
        # 也兼容 "标题 1" 等中文样式名
        heading_level = _detect_docx_heading_level(style_name)

        if heading_level > 0:
            lines.append(f"{'#' * heading_level} {text}")
            has_headings = True
        else:
            lines.append(text)

    result = "\n\n".join(line for line in lines if line)
    logger.info(f"DOCX 解析完成: {len(result)} 字符, 标题检测={'是' if has_headings else '否'}")
    return result, has_headings


def _detect_docx_heading_level(style_name: str) -> int:
    """从段落样式名推断标题层级。

    支持：
      - "Heading 1" ~ "Heading 6"（英文）
      - "标题 1" ~ "标题 6"（中文）
      - "Title"（映射为 H1）
    返回 0 表示非标题。
    """
    if not style_name:
        return 0

    name = style_name.strip()

    # 英文 Heading N
    for i in range(1, 7):
        if name.lower() == f"heading {i}":
            return i

    # 中文 标题 N
    for i in range(1, 7):
        if name == f"标题 {i}" or name == f"标题{i}":
            return i

    # Title 样式映射为 H1
    if name.lower() == "title" or name == "标题":
        return 1

    return 0


def _parse_markdown(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
