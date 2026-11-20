"""
CVInsight Evaluation & Ground-Truth Benchmark Runner
---------------------------------------------------
Computes:
1. FAISS Retrieval Precision@K, Recall@K, and MRR across the candidate pool.
2. LLM-as-a-Judge Faithfulness Check: verifies that the skill-gap output
   strictly grounds extracted skills in the candidate's resume without hallucinations.
3. Missing Skills Recall & Precision against ground-truth expectations.
"""

import os
import sys
import json
import argparse
from typing import List, Dict, Any, Set

# Ensure project root is on PYTHONPATH so app package is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# New package imports (industry-standard structure)
from app.services.embedding_service import get_embeddings
from app.services.llm_service import get_llm, generate_resume_insights, SkillGapAnalysis
from app.services.skills_service import extract_skills_from_text
from app.observability.tracing import logger


def load_benchmark(dataset_path: str = "evals/benchmark_dataset.json") -> List[Dict[str, Any]]:
    """Loads the benchmark dataset."""
    full_path = os.path.join(PROJECT_ROOT, dataset_path) if not os.path.isabs(dataset_path) else dataset_path
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Benchmark dataset not found at: {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_faiss_benchmark_index(samples: List[Dict[str, Any]]):
    """Indexes all candidate resumes in the benchmark into FAISS."""
    documents = []
    for s in samples:
        content = f"Candidate ID: {s['candidate_id']}\nName: {s['candidate_name']}\n\nResume:\n{s['resume']}"
        metadata = {
            "candidate_id": s["candidate_id"],
            "candidate_name": s["candidate_name"]
        }
        documents.append(Document(page_content=content, metadata=metadata))

    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


def evaluate_retrieval(vectorstore, samples: List[Dict[str, Any]], ks: List[int] = [1, 3, 5]) -> Dict[str, Any]:
    """
    Evaluates FAISS retrieval for Precision@K, Recall@K, and Mean Reciprocal Rank (MRR).
    """
    results_by_k = {k: {"precision": [], "recall": []} for k in ks}
    reciprocal_ranks = []

    print("\n" + "=" * 65)
    print(" 🔍 RUNNING FAISS RETRIEVAL EVALUATION")
    print("=" * 65)

    max_k = max(ks)
    retriever = vectorstore.as_retriever(search_kwargs={"k": max_k})

    for sample in samples:
        query = sample["job_description"]
        expected = set(sample["expected_top_matches"])

        retrieved_docs = retriever.invoke(query)
        retrieved_ids = [doc.metadata.get("candidate_id") for doc in retrieved_docs]

        # Calculate Reciprocal Rank (MRR)
        rr = 0.0
        for rank, cand_id in enumerate(retrieved_ids, 1):
            if cand_id in expected:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        # Calculate Precision@K and Recall@K
        for k in ks:
            top_k_ids = set(retrieved_ids[:k])
            hits = len(top_k_ids.intersection(expected))
            precision_k = hits / float(k)
            recall_k = hits / float(len(expected)) if expected else 0.0

            results_by_k[k]["precision"].append(precision_k)
            results_by_k[k]["recall"].append(recall_k)

    summary = {
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
        "metrics_by_k": {}
    }

    for k in ks:
        p_avg = sum(results_by_k[k]["precision"]) / len(results_by_k[k]["precision"])
        r_avg = sum(results_by_k[k]["recall"]) / len(results_by_k[k]["recall"])
        summary["metrics_by_k"][f"K={k}"] = {
            f"Precision@{k}": round(p_avg, 4),
            f"Recall@{k}": round(r_avg, 4)
        }

    return summary


def heuristic_faithfulness_check(resume_text: str, matched_skills: List[str]) -> Dict[str, Any]:
    """
    Heuristic ground-truth verification checking whether extracted skills are present in the resume.
    Ensures zero hallucinations even in offline environments without active LLM API keys.
    """
    resume_lower = resume_text.lower()
    hallucinated = []
    verified = []

    for skill in matched_skills:
        skill_clean = skill.strip().lower()
        if not skill_clean:
            continue
        # Check direct substring or word boundary
        if skill_clean in resume_lower:
            verified.append(skill)
        else:
            # Check individual tokens for compound skills
            tokens = skill_clean.split()
            if len(tokens) > 1 and all(t in resume_lower for t in tokens if len(t) > 2):
                verified.append(skill)
            else:
                hallucinated.append(skill)

    total = len(matched_skills)
    score = (len(verified) / float(total)) if total > 0 else 1.0

    return {
        "faithful": len(hallucinated) == 0,
        "hallucinated_skills": hallucinated,
        "verified_skills": verified,
        "faithfulness_score": round(score, 4),
        "judge_type": "heuristic_grounding"
    }


def llm_judge_faithfulness(resume_text: str, matched_skills: List[str], reasoning: str) -> Dict[str, Any]:
    """
    LLM-as-a-Judge: Uses Gemini to audit extracted skill-gap reasoning for hallucinations.
    Falls back gracefully to heuristic grounding if API is unavailable.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return heuristic_faithfulness_check(resume_text, matched_skills)

    try:
        from langchain_core.prompts import ChatPromptTemplate
        llm = get_llm()

        judge_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an impartial Judge auditing recruiter skill extraction for hallucination. "
                "Evaluate whether the Extracted Matched Skills are strictly supported by the Candidate Resume. "
                "Any skill listed that the candidate does NOT possess in their resume is a hallucination. "
                "Respond ONLY with valid JSON with keys: "
                "'faithful' (bool), 'hallucinated_skills' (list of strings), 'faithfulness_score' (float 0.0-1.0), 'reason' (string)."
            )),
            ("human", (
                "Candidate Resume:\n{resume}\n\n"
                "Extracted Matched Skills:\n{matched_skills}\n\n"
                "Recruiter Reasoning:\n{reasoning}\n\n"
                "Evaluate faithfulness strictly:"
            ))
        ])

        chain = judge_prompt | llm
        response = chain.invoke({
            "resume": resume_text,
            "matched_skills": json.dumps(matched_skills),
            "reasoning": reasoning
        })

        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        parsed = json.loads(content)
        parsed["judge_type"] = "llm_as_a_judge"
        return parsed

    except Exception as e:
        logger.warning(f"LLM Judge call failed ({e}), falling back to heuristic grounding.")
        return heuristic_faithfulness_check(resume_text, matched_skills)


def evaluate_faithfulness_and_skill_gap(samples: List[Dict[str, Any]], max_eval: int = 5, use_llm: bool = True) -> Dict[str, Any]:
    """
    Evaluates Skill Gap Extraction and Faithfulness / Anti-Hallucination.
    """
    print("\n" + "=" * 65)
    print(" ⚖️  RUNNING SKILL-GAP & FAITHFULNESS (LLM-AS-A-JUDGE) EVALUATION")
    print("=" * 65)

    eval_subset = samples[:max_eval]
    faithfulness_scores = []
    hallucination_counts = 0
    total_extracted_skills = 0

    missing_skill_precisions = []
    missing_skill_recalls = []

    detailed_records = []

    for idx, sample in enumerate(eval_subset, 1):
        resume = sample["resume"]
        jd = sample["job_description"]
        expected_missing = set(s.lower() for s in sample["expected_missing_skills"])

        # Run structured skill gap extraction
        if use_llm and (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            insight = generate_resume_insights(jd, resume)
        else:
            # Deterministic extraction fallback for offline evaluation
            cand_skills = list(extract_skills_from_text(resume))
            jd_skills = list(extract_skills_from_text(jd))
            matched = [s for s in cand_skills if s in jd_skills]
            missing = [s for s in jd_skills if s not in cand_skills]
            insight = {
                "match_score": 85 if matched else 40,
                "matched_skills": matched,
                "missing_skills": missing,
                "reasoning": f"Candidate possesses {len(matched)} matching competencies and lacks {len(missing)} requirements."
            }

        matched_skills = insight.get("matched_skills", [])
        missing_skills = insight.get("missing_skills", [])
        reasoning = insight.get("reasoning", "")

        # Run faithfulness judge
        if use_llm:
            judge_res = llm_judge_faithfulness(resume, matched_skills, reasoning)
        else:
            judge_res = heuristic_faithfulness_check(resume, matched_skills)

        f_score = judge_res.get("faithfulness_score", 1.0)
        faithfulness_scores.append(f_score)

        hallucinations = judge_res.get("hallucinated_skills", [])
        hallucination_counts += len(hallucinations)
        total_extracted_skills += len(matched_skills)

        # Missing skills overlap
        pred_missing = set(s.lower() for s in missing_skills)
        overlap = len(pred_missing.intersection(expected_missing))
        p_miss = (overlap / float(len(pred_missing))) if pred_missing else 1.0
        r_miss = (overlap / float(len(expected_missing))) if expected_missing else 1.0

        missing_skill_precisions.append(p_miss)
        missing_skill_recalls.append(r_miss)

        status_icon = "✅ PASS" if f_score >= 0.85 else "⚠️ WARN"
        print(f"[{idx}/{len(eval_subset)}] {sample['candidate_name'][:38]:<38} | Faithfulness: {f_score:.2f} {status_icon}")
        if hallucinations:
            print(f"    ↳ Hallucinated Skills Detected: {hallucinations}")

        detailed_records.append({
            "id": sample["id"],
            "candidate_name": sample["candidate_name"],
            "faithfulness_score": f_score,
            "hallucinated_skills": hallucinations,
            "matched_skills_count": len(matched_skills),
            "missing_skills_overlap": overlap,
            "judge_result": judge_res
        })

    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 1.0
    hallucination_rate = (hallucination_counts / float(total_extracted_skills)) if total_extracted_skills else 0.0

    return {
        "num_evaluated": len(eval_subset),
        "mean_faithfulness_score": round(avg_faithfulness, 4),
        "total_extracted_skills": total_extracted_skills,
        "hallucinated_skills_count": hallucination_counts,
        "hallucination_rate": round(hallucination_rate, 4),
        "mean_missing_skills_recall": round(sum(missing_skill_recalls) / len(missing_skill_recalls), 4) if missing_skill_recalls else 0.0,
        "detailed_evaluations": detailed_records
    }


def print_summary_table(retrieval_summary: Dict[str, Any], faithfulness_summary: Dict[str, Any], pool_size: int = 18):
    """Prints a structured ASCII report of evaluation metrics."""
    print("\n" + "=" * 65)
    print("                CVINSIGHT BENCHMARK EVALUATION REPORT")
    print("=" * 65)

    print(f"\n📊 1. FAISS RETRIEVAL METRICS (Across {pool_size} Candidate Pool):")
    print("-" * 55)
    print(f"  • Mean Reciprocal Rank (MRR):  {retrieval_summary['mrr']:.4f}")
    for k_label, metrics in retrieval_summary["metrics_by_k"].items():
        p_val = metrics[f"Precision@{k_label.split('=')[1]}"]
        r_val = metrics[f"Recall@{k_label.split('=')[1]}"]
        print(f"  • {k_label:<6} | Precision: {p_val:.4f} | Recall: {r_val:.4f}")

    print("\n🛡️  2. SKILL-GAP & LLM-AS-A-JUDGE FAITHFULNESS METRICS:")
    print("-" * 55)
    print(f"  • Samples Evaluated:            {faithfulness_summary['num_evaluated']}")
    print(f"  • Mean Faithfulness Score:      {faithfulness_summary['mean_faithfulness_score'] * 100:.2f}%")
    print(f"  • Total Extracted Skills:       {faithfulness_summary['total_extracted_skills']}")
    print(f"  • Hallucinated Skills Detected: {faithfulness_summary['hallucinated_skills_count']}")
    print(f"  • Skill Hallucination Rate:     {faithfulness_summary['hallucination_rate'] * 100:.2f}%")
    print(f"  • Missing Skills Recall:        {faithfulness_summary['mean_missing_skills_recall'] * 100:.2f}%")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run CVInsight Evaluation Benchmark")
    parser.add_argument("--dataset", default="evals/benchmark_dataset.json", help="Path to benchmark JSON")
    parser.add_argument("--max-eval", type=int, default=5, help="Number of samples to evaluate for LLM-as-a-Judge")
    parser.add_argument("--no-llm", action="store_true", help="Force heuristic offline judge (no LLM API calls)")
    parser.add_argument("--output", default="evals/eval_results.json", help="Output path for eval report")
    args = parser.parse_args()

    print("Loading benchmark dataset...")
    samples = load_benchmark(args.dataset)
    print(f"Loaded {len(samples)} benchmark samples from {args.dataset}.")

    # 1. Index candidates in FAISS
    print("Building FAISS index for candidate pool...")
    vectorstore = build_faiss_benchmark_index(samples)

    # 2. Evaluate retrieval (Precision@K / Recall@K)
    retrieval_results = evaluate_retrieval(vectorstore, samples, ks=[1, 3, 5])

    # 3. Evaluate skill-gap faithfulness & hallucination
    faithfulness_results = evaluate_faithfulness_and_skill_gap(
        samples,
        max_eval=args.max_eval,
        use_llm=not args.no_llm
    )

    # 4. Print Summary
    print_summary_table(retrieval_results, faithfulness_results, pool_size=len(samples))

    # 5. Save report to disk
    full_output = {
        "retrieval_metrics": retrieval_results,
        "faithfulness_metrics": faithfulness_results
    }
    out_path = os.path.join(PROJECT_ROOT, args.output) if not os.path.isabs(args.output) else args.output
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"Detailed evaluation metrics saved to: {out_path}\n")


if __name__ == "__main__":
    main()
