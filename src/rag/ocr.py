"""PDF OCR 独立模块。

封装 PyPDF2 → PaddleOCR 回退链，对外提供单一 OCR 接口。
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

OCR_FALLBACK_THRESHOLD = 50


def ocr_pdf(file_path: str, dpi: int = 200) -> str:
    """对 PDF 执行 OCR，返回识别文本。

    策略:
    1. 先用 PyPDF2 提取文本
    2. 如果文本不足 50 字符，回退到 PaddleOCR（通过 PyMuPDF 渲染页面）

    Args:
        file_path: PDF 文件绝对路径
        dpi: 页面渲染 DPI（影响 OCR 精度与速度）

    Returns:
        识别的文本内容（可能为空字符串）
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {file_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"非 PDF 文件: {file_path}")

    # 尝试 PyPDF2 直接提取
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        logger.warning("PyPDF2 未安装")
        return _paddle_ocr_fallback(file_path, dpi)

    try:
        reader = PdfReader(file_path)
        texts = [page.extract_text() or "" for page in reader.pages]
        extracted = "\n\n".join(texts).strip()
    except Exception as e:
        logger.warning(f"PyPDF2 解析失败: {e}")
        return _paddle_ocr_fallback(file_path, dpi)

    if len(extracted) >= OCR_FALLBACK_THRESHOLD:
        return extracted

    logger.info(f"PyPDF2 仅提取 {len(extracted)} 字符，回退到 PaddleOCR")

    ocr_text = _paddle_ocr_fallback(file_path, dpi)
    return ocr_text if ocr_text else extracted


def _paddle_ocr_fallback(file_path: str, dpi: int = 200) -> str:
    """PaddleOCR 回退链路：PyMuPDF 渲染 + PaddleOCR 识别。"""
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF 未安装，跳过 OCR")
        return ""

    try:
        from paddleocr import PaddleOCR
    except ImportError:
        logger.warning("PaddleOCR 未安装，跳过 OCR")
        return ""

    doc = fitz.open(file_path)
    ocr = PaddleOCR(lang="ch", ocr_version="PP-OCRv4", use_angle_cls=True)
    page_texts = []

    for i, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        result = ocr.ocr(img_bytes, cls=True)
        if result and result[0]:
            lines = [line[1][0] for line in result[0]]
            page_texts.append("\n".join(lines))
        else:
            logger.debug(f"OCR 页面 {i+1} 无文字")

    doc.close()
    logger.info(f"PaddleOCR 完成 {len(page_texts)}/{len(doc)} 页")
    return "\n\n".join(page_texts)
