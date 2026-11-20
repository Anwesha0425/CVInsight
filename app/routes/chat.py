"""Chat and agentic routing endpoints — /ask and /agent."""
from __future__ import annotations

from flask import Blueprint, request, jsonify

from app.database.repository import get_all_candidates
from app.services.retrieval_service import ask_candidates_question
from app.agent.recruiter_agent import run_recruiter_agent

chat_bp = Blueprint("chat", __name__)


@chat_bp.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    mode = data.get("mode", "rag")

    if not query:
        return jsonify({"answer": "Please ask a valid question."}), 400

    if mode == "agent":
        result = run_recruiter_agent(query)
        return jsonify({"answer": result.get("answer", "No response generated."), "tool_calls": result.get("tool_calls", []), "mode": "agent"})

    all_candidates = get_all_candidates()
    answer = ask_candidates_question(query, all_candidates)
    return jsonify({"answer": answer, "mode": "rag"})


@chat_bp.route("/agent", methods=["POST"])
def agent_endpoint():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"answer": "Please provide a query for the recruiter agent."}), 400
    return jsonify(run_recruiter_agent(query))
