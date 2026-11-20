"""CVInsight Recruiter Agent — LangGraph ReAct agent runner."""
from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from app.agent.tools import TOOLS
from app.agent.prompts import SYSTEM_PROMPT
from app.services.llm_service import get_llm
from app.observability.tracing import observe, TraceSpan

logger = logging.getLogger(__name__)


def get_recruiter_agent():
    """Build and return the compiled LangGraph ReAct agent."""
    llm = get_llm()
    try:
        return create_react_agent(model=llm, tools=TOOLS, prompt=SYSTEM_PROMPT)
    except TypeError:
        return create_react_agent(model=llm, tools=TOOLS, state_modifier=SYSTEM_PROMPT)


@observe(name="agentic_recruiter_routing", as_type="agent")
def run_recruiter_agent(query: str) -> Dict[str, Any]:
    """
    Execute the agentic workflow on a recruiter query.
    Returns answer, tool_calls list, and total_messages count.
    """
    agent = get_recruiter_agent()
    inputs = {"messages": [HumanMessage(content=query)]}

    try:
        with TraceSpan("agent_execution_graph", extra={"query": query[:60]}):
            result = agent.invoke(inputs)

        messages = result.get("messages", [])
        final_answer = messages[-1].content if messages else "No response generated."

        tool_calls = [
            {"tool": tc.get("name"), "args": tc.get("args")}
            for msg in messages
            if hasattr(msg, "tool_calls") and msg.tool_calls
            for tc in msg.tool_calls
        ]

        return {
            "query": query,
            "answer": final_answer,
            "tool_calls": tool_calls,
            "total_messages": len(messages),
        }

    except Exception as exc:
        logger.error("Agent execution error: %s", exc)
        return {"query": query, "answer": f"Agent error: {exc}", "tool_calls": [], "error": str(exc)}
