"""Embeddings factory — Google Generative AI with local SentenceTransformer fallback."""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import logging
from typing import List

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class LocalSentenceTransformerEmbeddings(Embeddings):
    """LangChain-compatible embeddings wrapper around SentenceTransformer for offline environments."""

    def __init__(self, st_model):
        self.st_model = st_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.st_model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.st_model.encode([text], convert_to_numpy=True)[0].tolist()


# Cache the local SentenceTransformer instance to avoid re-loading on each call
_local_st_embeddings = None


def get_embeddings() -> Embeddings:
    """
    Returns the best available embeddings instance:
    1. GoogleGenerativeAIEmbeddings if GEMINI_API_KEY is set
    2. LocalSentenceTransformerEmbeddings as fallback
    """
    from app.config import settings

    api_key = settings.gemini_api_key or settings.google_api_key
    if api_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            return GoogleGenerativeAIEmbeddings(
                model=settings.embedding_model,
                google_api_key=api_key,
            )
        except Exception as exc:
            logger.warning("Google embeddings unavailable, falling back to local: %s", exc)

    # Local fallback with caching
    global _local_st_embeddings
    if _local_st_embeddings is None:
        from sentence_transformers import SentenceTransformer
        from app.config import settings as s
        st_model = SentenceTransformer(s.st_model_name)
        _local_st_embeddings = LocalSentenceTransformerEmbeddings(st_model)
    return _local_st_embeddings
