"""LangChain tool definitions for the CVInsight Recruiter Agent."""
from __future__ import annotations

import json
import logging
import os
from typing import List

from langchain_core.tools import tool
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@tool
def search_candidates_vector_db(query: str, top_k: int = 3) -> str:
    """
    Search candidate resumes in the FAISS vector database using semantic similarity.
    Use for high-level qualifications, domain expertise, project experience, or complex JDs.

    Args:
        query: The semantic search query or job description snippet.
        top_k: Number of top matches to retrieve (default 3).
    """
    from app.database.repository import get_all_candidates
    from app.services.embedding_service import get_embeddings
    from langchain_community.vectorstores import FAISS

    candidates = get_all_candidates()
    if not candidates:
        return "No candidates found in the database. Please upload resumes first."

    try:
        documents = [
            Document(
                page_content=f"Candidate: {c['filename']}\nSkills: {c.get('extracted_skills', '[]')}\n\n{c['content']}",
                metadata={"filename": c["filename"]},
            )
            for c in candidates
        ]
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(documents, embeddings)
        docs = vectorstore.as_retriever(search_kwargs={"k": min(int(top_k), len(candidates))}).invoke(query)
        return "\n".join(f"--- Result {i+1} ---\n{doc.page_content[:600]}...\n" for i, doc in enumerate(docs))
    except Exception as exc:
        logger.error("Vector search error: %s", exc)
        return f"Error executing vector search: {exc}"


@tool
def parse_candidate_document(file_path: str) -> str:
    """
    Parse and extract raw text from a candidate resume (PDF, DOCX, or TXT).
    Use when the recruiter provides a local file path.

    Args:
        file_path: Absolute or relative path to the resume file.
    """
    from app.services.document_parser import extract_text

    if not os.path.exists(file_path):
        return f"File not found: {file_path}"
    try:
        text = extract_text(file_path)
        if not text:
            return f"No text extracted from {file_path}."
        return f"Parsed {file_path} ({len(text)} chars):\n\n{text[:1200]}..."
    except Exception as exc:
        logger.error("Document parse error: %s", exc)
        return f"Error parsing {file_path}: {exc}"


@tool
def filter_candidates_by_skill(skill_name: str) -> str:
    """
    Filter candidates by exact skill metadata tag stored in SQLite.
    Use for specific skill lookups: 'python', 'docker', 'kubernetes', 'aws', etc.

    Args:
        skill_name: The skill to filter by.
    """
    from app.database.repository import get_all_candidates

    candidates = get_all_candidates()
    if not candidates:
        return "No candidates found in the database."

    target = skill_name.strip().lower()
    matched = []
    for c in candidates:
        raw = c.get("extracted_skills", "[]")
        try:
            skills = [s.lower() for s in (json.loads(raw) if isinstance(raw, str) else raw)]
        except Exception:
            skills = []
        if target in skills:
            matched.append({"filename": c["filename"], "skills": skills[:12]})

    if not matched:
        return f"No candidates with skill '{skill_name}'."
    lines = [f"Found {len(matched)} candidate(s) with skill '{skill_name}':"]
    lines += [f"- {m['filename']} (skills: {', '.join(m['skills'])})" for m in matched]
    return "\n".join(lines)


@tool
def ask_recruiter_clarification(question: str) -> str:
    """
    Ask the recruiter a clarifying question when the request is too vague or missing constraints.

    Args:
        question: The specific clarifying question to ask.
    """
    return f"CLARIFICATION_NEEDED: {question}"


TOOLS: List = [
    search_candidates_vector_db,
    filter_candidates_by_skill,
    parse_candidate_document,
    ask_recruiter_clarification,
]
