"""LLM service — Gemini chat model, Pydantic structured output, skill-gap analysis."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from app.observability.tracing import observe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic schema for structured LLM output
# ---------------------------------------------------------------------------
class SkillGapAnalysis(BaseModel):
    """Recruiter evaluation of a candidate against a job description."""

    match_score: int = Field(
        ..., ge=0, le=100,
        description="Overall match score 0–100."
    )
    matched_skills: List[str] = Field(
        default_factory=list,
        description="Skills from the resume that match the JD."
    )
    missing_skills: List[str] = Field(
        default_factory=list,
        description="Required JD skills absent from the resume."
    )
    reasoning: str = Field(
        ...,
        description="2-3 sentence recruiter rationale."
    )


def get_llm() -> ChatGoogleGenerativeAI:
    """Return a configured Gemini chat model instance."""
    from app.config import settings
    api_key = settings.gemini_api_key or settings.google_api_key
    return ChatGoogleGenerativeAI(
        model=settings.model_name,
        temperature=0.3,
        google_api_key=api_key,
    )


@observe(name="gemini_skill_gap_analysis", as_type="generation")
def generate_resume_insights(job_description: str, resume_text: str) -> Dict[str, Any]:
    """
    Use Gemini with Pydantic .with_structured_output() to produce typed skill-gap analysis.
    Returns a dict with match_score, matched_skills, missing_skills, reasoning plus
    backward-compatible aliases for existing templates.
    """
    try:
        llm = get_llm()
        structured_llm = llm.with_structured_output(SkillGapAnalysis)

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an expert technical recruiter. "
                "Analyse the candidate's resume strictly against the job description.\n"
                "1. Assign an integer match_score 0–100.\n"
                "2. Extract matched_skills: skills in the resume relevant to the JD.\n"
                "3. Identify missing_skills: key JD requirements absent from the resume.\n"
                "4. Write a 2–3 sentence reasoning summary."
            )),
            ("human", "Job Description:\n{job_description}\n\nCandidate Resume:\n{resume_text}"),
        ])

        result: SkillGapAnalysis = (prompt | structured_llm).invoke(
            {"job_description": job_description, "resume_text": resume_text}
        )

        return {
            "match_score": result.match_score,
            "matched_skills": result.matched_skills,
            "missing_skills": result.missing_skills,
            "reasoning": result.reasoning,
            # backward-compat aliases
            "extracted_skills": result.matched_skills,
            "match_reasoning": result.reasoning,
        }

    except Exception as exc:
        logger.error("Structured LLM analysis failed: %s", exc)
        return {
            "match_score": 0,
            "matched_skills": [],
            "missing_skills": [],
            "reasoning": f"LLM error: {exc}",
            "extracted_skills": [],
            "match_reasoning": f"LLM error: {exc}",
            "error": str(exc),
        }
