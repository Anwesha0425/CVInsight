"""Unit tests for the skills_service."""
import pytest
from app.services.skills_service import extract_skills_from_text, SKILLS_TAXONOMY


def test_skills_taxonomy_is_set():
    assert isinstance(SKILLS_TAXONOMY, set)
    assert "python" in SKILLS_TAXONOMY
    assert "docker" in SKILLS_TAXONOMY


def test_extract_skills_multiword():
    text = "Experienced in machine learning and deep learning frameworks."
    skills = extract_skills_from_text(text)
    assert "machine learning" in skills
    assert "deep learning" in skills


def test_extract_skills_empty_text():
    skills = extract_skills_from_text("")
    assert isinstance(skills, set)
    assert len(skills) == 0
