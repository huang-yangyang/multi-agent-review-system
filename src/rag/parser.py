"""文档解析器。

支持 PDF（PyPDF2 → PaddleOCR 回退）、DOCX（python-docx）、Markdown、TXT 四种格式。
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}

OCR_FALLBACK_THRESHOLD = 50


def parse_document(file_path: str) -> str:
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
        return _parse_markdown(file_path)
    elif ext == ".txt":
        return _parse_text(file_path)
    else:
        raise ValueError(f"未实现的解析器: {ext}")


def _parse_pdf(file_path: str) -> str:
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
        return extracted

    logger.info(f"PyPDF2 提取文本不足 {OCR_FALLBACK_THRESHOLD} 字符，回退到 PaddleOCR")
    ocr_text = _ocr_pdf(file_path)
    if ocr_text:
        return ocr_text

    return extracted


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


def _parse_docx(file_path: str) -> str:
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx 未安装，请执行: pip install python-docx")

    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def _parse_markdown(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _parse_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
