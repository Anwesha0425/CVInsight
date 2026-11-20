"""Two-stage RAG retrieval service using FAISS + Gemini LLM."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from app.services.embedding_service import get_embeddings
from app.services.llm_service import get_llm
from app.observability.tracing import observe, TraceSpan

logger = logging.getLogger(__name__)


@observe(name="two_stage_rag_candidate_qa", as_type="chain")
def ask_candidates_question(query: str, candidates: List[Dict[str, Any]]) -> str:
    """
    Stage 2 RAG: build an in-memory FAISS index from candidates and answer recruiter questions.
    """
    if not candidates:
        return "No candidates available in the database to search."

    try:
        documents = [
            Document(
                page_content=f"Candidate: {c['filename']}\n\n{c['content']}",
                metadata={"filename": c["filename"], "id": c.get("id")},
            )
            for c in candidates
        ]

        with TraceSpan("faiss_vector_retrieval", extra={"num_candidates": len(candidates), "top_k": 3}) as span:
            embeddings = get_embeddings()
            vectorstore = FAISS.from_documents(documents, embeddings)
            retriever = vectorstore.as_retriever(search_kwargs={"k": min(3, len(candidates))})
            relevant_docs = retriever.invoke(query)
            span.record_count(len(relevant_docs))

        context = "\n\n---\n\n".join(doc.page_content for doc in relevant_docs)

        prompt = PromptTemplate.from_template(
            "You are an HR assistant helping a recruiter find the best candidates.\n"
            "Use the following resume contexts to answer the question. Always mention candidate names.\n"
            "If the answer is not in the context, say so.\n\n"
            "Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
        )

        llm = get_llm()
        with TraceSpan("rag_llm_synthesis"):
            response = (prompt | llm).invoke({"context": context, "query": query})

        return response.content if hasattr(response, "content") else str(response)

    except Exception as exc:
        logger.error("RAG pipeline error: %s", exc)
        return f"Sorry, an error occurred: {exc}"
