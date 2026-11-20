"""Semantic similarity scoring service using local SentenceTransformer embeddings."""
from __future__ import annotations

import os
import logging
from typing import List

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from app.observability.tracing import observe, TraceSpan

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load SentenceTransformer to avoid blocking startup
# ---------------------------------------------------------------------------
_st_model = None


def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        from app.config import settings
        logger.info("Loading SentenceTransformer model '%s'...", settings.st_model_name)
        _st_model = SentenceTransformer(settings.st_model_name)
        logger.info("SentenceTransformer model loaded.")
    return _st_model


@observe(name="stage1_semantic_retrieval", as_type="retrieval")
def get_similarity_scores(job_description: str, resumes_text: List[str]) -> List[float]:
    """
    Stage 1 retrieval: computes cosine similarity between job description and each resume.
    Returns a list of float scores in [0, 1].
    """
    if not job_description or not resumes_text:
        return []

    from sentence_transformers import util

    model = _get_st_model()
    with TraceSpan("sentence_transformer_encode", extra={"num_resumes": len(resumes_text)}):
        job_emb = model.encode(job_description, convert_to_tensor=True)
        resume_embs = model.encode(resumes_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(job_emb, resume_embs)[0]

    return cosine_scores.tolist()
