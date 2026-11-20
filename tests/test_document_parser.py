"""Unit tests for the document_parser service."""
import os
import tempfile
import pytest
from app.services.document_parser import extract_text_from_txt, extract_text


def test_extract_text_from_txt():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as fh:
        fh.write("Hello, World!")
        path = fh.name
    try:
        assert extract_text_from_txt(path) == "Hello, World!"
        assert extract_text(path) == "Hello, World!"
    finally:
        os.unlink(path)


def test_extract_text_unsupported_extension():
    assert extract_text("some_file.xyz") == ""


def test_extract_text_nonexistent_pdf():
    result = extract_text_from_txt("/nonexistent/file.txt")
    assert result == ""
