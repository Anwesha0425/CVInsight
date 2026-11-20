"""Resume matching routes — /  and /matcher."""
from __future__ import annotations

import os
import numpy as np
from flask import Blueprint, request, render_template

from werkzeug.utils import secure_filename

from app.config import settings
from app.database.repository import add_candidate, get_all_candidates
from app.services.document_parser import extract_text
from app.services.skills_service import extract_skills_from_text
from app.services.similarity_service import get_similarity_scores
from app.services.llm_service import generate_resume_insights

matcher_bp = Blueprint("matcher", __name__)


@matcher_bp.route("/")
def index():
    return render_template("index.html")


@matcher_bp.route("/matcher", methods=["POST"])
def matcher():
    job_description = request.form.get("job_description", "")
    resume_files = request.files.getlist("resumes")

    # Save newly uploaded resumes → DB
    for file in resume_files:
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(settings.upload_folder, filename)
            file.save(filepath)
            text = extract_text(filepath)
            skills = extract_skills_from_text(text)
            add_candidate(filename, text, skills)
            try:
                os.remove(filepath)
            except OSError:
                pass

    all_candidates = get_all_candidates()
    if not all_candidates:
        return render_template("index.html", message="No candidates in database and no new files uploaded.")

    resumes_text = [c["content"] for c in all_candidates]
    filenames = [c["filename"] for c in all_candidates]

    similarity_scores = get_similarity_scores(job_description, resumes_text)
    similarities = np.array(similarity_scores)
    top_n = settings.top_candidates
    top_indices = similarities.argsort()[-top_n:][::-1]

    top_resumes = [filenames[i] for i in top_indices]
    top_resumes_text = [resumes_text[i] for i in top_indices]
    similarity_percentages = [
        round(min(100.0, max(0.0, float(similarities[i]) * 100)), 1) for i in top_indices
    ]

    ai_insights = []
    for i, resume_text in enumerate(top_resumes_text):
        if i < settings.llm_analyze_top_n:
            insight = generate_resume_insights(job_description, resume_text)
        else:
            insight = {"error": "Skipped LLM analysis (only top 3 analysed)"}
        ai_insights.append(insight)

    return render_template(
        "index.html",
        message="Top Matching Candidates",
        top_resumes=top_resumes,
        similarity_scores=similarity_percentages,
        ai_insights=ai_insights,
    )
