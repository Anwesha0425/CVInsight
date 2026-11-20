"""
CVInsight Tracing & Observability Module
----------------------------------------
Supports both LangSmith (LANGCHAIN_TRACING_V2=true) and Langfuse (@observe)
to trace the two-stage retrieval pipeline, logging:
- Embedding retrieval latency (ms)
- LLM response latency (ms)
- Token usage (prompt, completion, total)
- Per-query estimated cost
"""

import os
import time
import logging
import functools
from typing import Any, Callable, Optional, Dict

# Configure logger
logger = logging.getLogger("CVInsight.Tracing")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Pricing Constants (Google Gemini 1.5 Flash reference rates)
# Input: $0.075 per 1M tokens ($0.000000075 / token)
# Output: $0.30 per 1M tokens ($0.00000030 / token)
# ---------------------------------------------------------------------------
COST_PER_INPUT_TOKEN = 0.075 / 1_000_000
COST_PER_OUTPUT_TOKEN = 0.30 / 1_000_000


def calculate_token_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculates estimated LLM query cost in USD based on Gemini 1.5 Flash pricing."""
    return (input_tokens * COST_PER_INPUT_TOKEN) + (output_tokens * COST_PER_OUTPUT_TOKEN)


# ---------------------------------------------------------------------------
# LangSmith Setup
# ---------------------------------------------------------------------------
LANGSMITH_AVAILABLE = False
try:
    from langsmith import traceable
    LANGSMITH_AVAILABLE = True
except ImportError:
    def traceable(name: Optional[str] = None, **kwargs):
        def decorator(func: Callable):
            return func
        return decorator

# Check if LangSmith tracing is active via environment
def is_langsmith_enabled() -> bool:
    return os.getenv("LANGCHAIN_TRACING_V2", "false").lower() in ("true", "1", "yes")


# ---------------------------------------------------------------------------
# Langfuse Setup / Fallback @observe Decorator
# ---------------------------------------------------------------------------
LANGFUSE_AVAILABLE = False
_langfuse_observe = None

try:
    from langfuse.decorators import observe as lf_observe
    _langfuse_observe = lf_observe
    LANGFUSE_AVAILABLE = True
except ImportError:
    pass


def observe(name: Optional[str] = None, as_type: Optional[str] = None):
    """
    Decorator for tracing pipeline functions.
    - If Langfuse is installed and configured, delegates to Langfuse @observe.
    - Also applies LangSmith @traceable if LangSmith is enabled.
    - Always records local latency, token metrics, and per-query cost.
    """
    def decorator(func: Callable):
        span_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            error_occurred = None
            result = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error_occurred = e
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                # Extract token usage if available on result or langchain response
                input_tokens = 0
                output_tokens = 0
                total_tokens = 0

                if result is not None:
                    # Check for LangChain response usage_metadata
                    usage_meta = getattr(result, "usage_metadata", None)
                    if isinstance(usage_meta, dict):
                        input_tokens = usage_meta.get("input_tokens", 0)
                        output_tokens = usage_meta.get("output_tokens", 0)
                        total_tokens = usage_meta.get("total_tokens", input_tokens + output_tokens)
                    elif isinstance(result, dict) and "usage_metadata" in result:
                        u = result["usage_metadata"]
                        input_tokens = u.get("input_tokens", 0)
                        output_tokens = u.get("output_tokens", 0)
                        total_tokens = u.get("total_tokens", input_tokens + output_tokens)

                cost = calculate_token_cost(input_tokens, output_tokens)

                # Format log output
                metrics_info = f"latency={elapsed_ms:.2f}ms"
                if total_tokens > 0:
                    metrics_info += (
                        f" | tokens=[in:{input_tokens}, out:{output_tokens}, total:{total_tokens}]"
                        f" | est_cost=${cost:.6f}"
                    )

                status = "FAILED" if error_occurred else "SUCCESS"
                logger.info(f"[{span_name}] status={status} | {metrics_info}")

        # If Langfuse is available, wrap with it
        if LANGFUSE_AVAILABLE and _langfuse_observe:
            try:
                wrapper = _langfuse_observe(name=span_name, as_type=as_type)(wrapper)
            except Exception:
                pass

        # If LangSmith is available and enabled, wrap with traceable
        if LANGSMITH_AVAILABLE and is_langsmith_enabled():
            try:
                wrapper = traceable(name=span_name)(wrapper)
            except Exception:
                pass

        return wrapper
    return decorator


class TraceSpan:
    """
    Context manager for measuring and logging latency and metrics of code blocks.
    Usage:
        with TraceSpan("faiss_embedding_retrieval", extra={"k": 3}) as span:
            docs = retriever.invoke(query)
            span.record_count(len(docs))
    """
    def __init__(self, name: str, extra: Optional[Dict[str, Any]] = None):
        self.name = name
        self.extra = extra or {}
        self.count: Optional[int] = None
        self.start_time: float = 0.0
        self.elapsed_ms: float = 0.0

    def record_count(self, count: int):
        self.count = count

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed_ms = (time.perf_counter() - self.start_time) * 1000
        status = "FAILED" if exc_type else "SUCCESS"
        extra_str = f" | count={self.count}" if self.count is not None else ""
        if self.extra:
            extra_str += " | " + " ".join(f"{k}={v}" for k, v in self.extra.items())
        logger.info(f"[TraceSpan:{self.name}] status={status} | latency={self.elapsed_ms:.2f}ms{extra_str}")
