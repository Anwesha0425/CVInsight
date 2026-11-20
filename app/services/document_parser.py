"""Document parsing service — extracts plain text from PDF, DOCX, and TXT files."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    text = ""
    try:
        import PyPDF2
        with open(file_path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception as exc:
        logger.warning("PDF extraction failed for %s: %s", file_path, exc)
    return text


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file using docx2txt."""
    try:
        import docx2txt
        return docx2txt.process(file_path) or ""
    except Exception as exc:
        logger.warning("DOCX extraction failed for %s: %s", file_path, exc)
        return ""


def extract_text_from_txt(file_path: str) -> str:
    """Read plain text from a .txt file."""
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception as exc:
        logger.warning("TXT extraction failed for %s: %s", file_path, exc)
        return ""


def extract_text(file_path: str) -> str:
    """Dispatch to the appropriate extractor based on file extension."""
    lower = file_path.lower()
    if lower.endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    if lower.endswith(".docx"):
        return extract_text_from_docx(file_path)
    if lower.endswith(".txt"):
        return extract_text_from_txt(file_path)
    logger.warning("Unsupported file type: %s", file_path)
    return ""
