# CVInsight 🎯

> **AI-powered resume matching & agentic recruiter assistant** — built with Gemini, LangGraph, FAISS, and Pydantic structured outputs.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-agentic-green.svg)](https://github.com/langchain-ai/langgraph)
[![Pydantic v2](https://img.shields.io/badge/Pydantic-v2-red.svg)](https://docs.pydantic.dev/)

---

## ✨ What it does

CVInsight is a full-stack AI recruiter that:

1. **Semantic Resume Matching** — SentenceTransformer cosine similarity ranks all uploaded resumes against a job description in real time (Stage 1 retrieval).
2. **Pydantic Structured Skill-Gap Analysis** — Gemini LLM returns `SkillGapAnalysis(match_score, matched_skills, missing_skills, reasoning)` via `.with_structured_output()` — never breaks on malformed JSON.
3. **Two-Stage RAG Chat** — FAISS vector store + Gemini LLM answers freeform recruiter questions grounded in real candidate data.
4. **Agentic Tool Router** — LangGraph ReAct agent dynamically routes to: vector DB search, metadata skill-filter, resume document parser, or recruiter clarification.
5. **LangSmith + Langfuse Tracing** — `@observe` decorator logs latency, token usage, and per-query cost across the entire pipeline.

---

## 🏗️ Project Structure

```
cvinsight/
├── app/                        # Main application package
│   ├── __init__.py             # Flask app factory (create_app)
│   ├── config.py               # Pydantic BaseSettings — env-driven config
│   ├── routes/
│   │   ├── matcher.py          # / and /matcher endpoints
│   │   └── chat.py             # /ask and /agent endpoints
│   ├── services/
│   │   ├── document_parser.py  # PDF / DOCX / TXT text extraction
│   │   ├── embedding_service.py# Google + local ST embeddings factory
│   │   ├── llm_service.py      # Gemini LLM + SkillGapAnalysis schema
│   │   ├── retrieval_service.py# FAISS two-stage RAG pipeline
│   │   ├── similarity_service.py# SentenceTransformer Stage-1 scoring
│   │   └── skills_service.py   # Taxonomy + NLP skills extraction
│   ├── agent/
│   │   ├── tools.py            # @tool definitions (4 tools)
│   │   ├── prompts.py          # Agent system prompt
│   │   └── recruiter_agent.py  # LangGraph ReAct runner
│   ├── database/
│   │   └── repository.py       # SQLite CRUD (init, add, get)
│   └── observability/
│       └── tracing.py          # @observe + TraceSpan (dual LangSmith/Langfuse)
├── templates/
│   └── index.html              # Dark glass-morphism UI
├── static/                     # CSS / JS / images (future)
├── examples/                   # Sample resumes for local testing
│   └── sample_resumes/
│       └── sample_resume.pdf
├── scripts/                    # Operational & deployment utilities
│   └── push_hf.py              # Automated deploy to Hugging Face Spaces
├── uploads/                    # Runtime uploads folder (gitignored, .gitkeep)
├── evals/
│   ├── benchmark_dataset.json  # 18-sample ground-truth dataset
│   ├── benchmark_dataset.csv
│   └── run_evals.py            # Precision@K, Recall@K, MRR, LLM-as-a-Judge
├── tests/
│   ├── conftest.py
│   ├── test_document_parser.py
│   ├── test_skills_service.py
│   └── test_similarity_service.py
├── config/
│   └── logging.yaml            # Rotating file + console logging
├── run.py                      # Dev server entry point
├── wsgi.py                     # Production WSGI (gunicorn)
├── pyproject.toml              # Black + Ruff config, project metadata
├── Makefile                    # make dev / test / eval / lint / format
├── Dockerfile                  # Gunicorn production image
├── docker-compose.yml          # Local dev with volume mounts
├── requirements.txt
└── requirements-dev.txt        # pytest, black, ruff, mypy
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/CVInsight.git
cd CVInsight
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env — add your GEMINI_API_KEY at minimum
```

### 3. Run

```bash
python run.py          # Dev server on http://localhost:5000
# or
make dev
```

---

## ⚙️ Configuration (`.env`)

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | ✅ | Google Gemini API key |
| `LANGCHAIN_TRACING_V2` | Optional | `true` to enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | Optional | LangSmith API key |
| `LANGCHAIN_PROJECT` | Optional | LangSmith project name (default: `CVInsight`) |
| `LANGFUSE_PUBLIC_KEY` | Optional | Langfuse public key |
| `LANGFUSE_SECRET_KEY` | Optional | Langfuse secret key |

---

## 🛠️ Make Targets

```bash
make dev          # Start Flask dev server
make test         # Run pytest suite
make eval         # Run golden set benchmark (18 candidates)
make eval-kaggle  # Run 50-sample Kaggle benchmark (10 categories)
make eval-full    # Full benchmark with LLM-as-a-Judge
make lint         # ruff + mypy
make format       # black + ruff --fix
make deploy       # Automated upload to Hugging Face Spaces
make clean        # Remove __pycache__, *.pyc, candidates.db
```

---

## 📊 Benchmark Results

CVInsight is rigorously benchmarked across both a curated engineering golden set and a 50-candidate Kaggle multi-domain dataset (covering Data Science, Java, DevOps, Python, Web Designing, HR, Database, Security, Blockchain, and QA):

| Metric | Kaggle Benchmark (50 Candidates) | Golden Set (18 Candidates) |
|---|---|---|
| **Mean Reciprocal Rank (MRR)** | **1.0000** | **0.9722** |
| **Precision@1** | **100.0%** | **94.44%** |
| **Precision@3** | **90.00%** | **74.07%** |
| **Recall@5** | **84.00%** | **86.11%** |
| **LLM Faithfulness Score** | **100.0%** | **100.0%** |
| **Skill Hallucination Rate** | **0.00%** | **0.00%** |

```bash
# Run Kaggle benchmark (50 samples across 10 categories)
make eval-kaggle
# or: python evals/run_evals.py --dataset evals/kaggle_benchmark_dataset.json --output evals/kaggle_eval_results.json

# Ingest any custom Kaggle CSV
python scripts/ingest_kaggle_resumes.py path/to/UpdatedResumeDataSet.csv
```

---

## 🤖 Architecture

```
Upload Resume (PDF/DOCX/TXT)
         │
         ▼
 DocumentParser ──► SQLite DB (candidates)
                          │
              ┌───────────┴────────────┐
              │  Stage 1: ST Cosine    │  Stage 2: FAISS RAG
              │  Similarity Ranking    │  + Gemini LLM Answer
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │  Gemini .with_         │
              │  structured_output()   │  ← Pydantic SkillGapAnalysis
              │  → SkillGapAnalysis    │
              └───────────┬────────────┘
                          │
              ┌───────────▼────────────┐
              │  LangGraph ReAct Agent │
              │  ┌──────────────────┐  │
              │  │ search_vector_db │  │
              │  │ filter_by_skill  │  │
              │  │ parse_document   │  │
              │  │ ask_clarification│  │
              │  └──────────────────┘  │
              └────────────────────────┘
                          │
              @observe + TraceSpan ──► LangSmith / Langfuse
```

---

## 🐳 Docker

```bash
# Build & run
docker compose up --build

# Or plain Docker
docker build -t cvinsight .
docker run -p 5000:5000 -e GEMINI_API_KEY=your_key cvinsight
```

---

## 🧪 Testing

```bash
make test
# or
python -m pytest tests/ -v --tb=short
```

---

## 📁 Key Design Decisions

| Decision | Rationale |
|---|---|
| **Pydantic `.with_structured_output()`** | Eliminates JSON parse errors; enforces type safety on LLM responses |
| **Two-stage retrieval** | ST cosine similarity (fast, O(n)) → FAISS semantic search (deep, top-k) |
| **Flask Blueprints** | Clean separation of matcher vs. chat/agent routes |
| **App factory pattern** | Enables testing with different configs without global state |
| **Service layer** | Each concern (parsing, embedding, LLM, retrieval) is independently testable |
| **Dual tracing** | Works with either LangSmith or Langfuse — no vendor lock-in |

---

## 📄 License

MIT License — feel free to fork and build on it.
