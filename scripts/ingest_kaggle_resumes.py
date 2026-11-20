"""
Kaggle Resume Dataset Ingestion & Benchmark Formatter
---------------------------------------------------
Converts standard Kaggle resume datasets (e.g., 'UpdatedResumeDataSet.csv'
or 'Resume.csv' with columns like Category/Resume) into the CVInsight
evaluation benchmark schema with realistic target job descriptions
and ground-truth match annotations.
"""

import os
import sys
import json
import csv
import re
from typing import List, Dict, Any

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.services.skills_service import extract_skills_from_text


def clean_resume_text(text: str) -> str:
    """Cleans raw scraped resume text from Kaggle datasets."""
    # Remove URLs
    text = re.sub(r'http\S+\s*', ' ', text)
    # Remove RT and cc
    text = re.sub(r'\bRT\b|\bcc\b', ' ', text)
    # Remove hashtags and mentions
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', ' ', text)
    # Remove punctuation sequences but preserve technical symbols
    text = re.sub(r'[^\w\s\+\#\.\-]', ' ', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Realistic Job Descriptions tailored to common Kaggle resume categories
CATEGORY_JOB_DESCRIPTIONS: Dict[str, Dict[str, Any]] = {
    "Data Science": {
        "jd": "Seeking a Senior Data Scientist proficient in Python, machine learning, Pandas, Scikit-learn, and statistics. Experience with deep learning (PyTorch/TensorFlow) and SQL databases required. Nice to have: MLOps with Docker and cloud deployment on AWS.",
        "required_skills": ["python", "machine learning", "pandas", "scikit-learn", "sql"],
        "differentiating_skills": ["aws", "docker", "pytorch"]
    },
    "Java Developer": {
        "jd": "Hiring a Java Developer with 4+ years building enterprise microservices using Java, Spring Boot, Hibernate, and REST APIs. Strong database skills in MySQL or PostgreSQL and build tools like Maven/Gradle required. Cloud experience with Docker is a plus.",
        "required_skills": ["java", "spring", "spring boot", "hibernate", "sql"],
        "differentiating_skills": ["docker", "kubernetes", "microservices"]
    },
    "DevOps Engineer": {
        "jd": "Looking for a DevOps Engineer to manage cloud infrastructure and CI/CD pipelines. Core skills: Linux, Docker, Kubernetes, Terraform, AWS, Jenkins, and Git. Experience with Prometheus, Grafana, and Bash automation strongly preferred.",
        "required_skills": ["docker", "kubernetes", "linux", "aws", "terraform", "ci/cd"],
        "differentiating_skills": ["prometheus", "grafana", "ansible"]
    },
    "Python Developer": {
        "jd": "We are seeking a Python Developer to build backend services and APIs. Must be experienced in Python, Django or FastAPI, PostgreSQL, Redis, and asynchronous programming with Celery. Familiarity with Docker and Git version control is mandatory.",
        "required_skills": ["python", "django", "fastapi", "postgresql", "redis"],
        "differentiating_skills": ["celery", "docker", "graphql"]
    },
    "Web Designing": {
        "jd": "Looking for a Web Designer and Frontend Developer skilled in HTML5, CSS3, JavaScript, Bootstrap, Tailwind, and Figma. Experience creating responsive layouts and translating UI/UX wireframes into functional web pages. React experience is a bonus.",
        "required_skills": ["html", "css", "javascript", "bootstrap", "figma"],
        "differentiating_skills": ["react", "tailwind", "responsive design"]
    },
    "HR": {
        "jd": "Seeking an experienced HR Specialist to oversee full-cycle recruiting, talent acquisition, onboarding, and employee relations. Must have excellent communication, interview scheduling, ATS management, and performance appraisal skills.",
        "required_skills": ["recruiting", "talent acquisition", "communication", "human resources", "management"],
        "differentiating_skills": ["hris", "payroll", "employee engagement"]
    },
    "Database": {
        "jd": "Hiring a Database Administrator (DBA) proficient in Oracle, SQL Server, MySQL, and PostgreSQL. Responsible for schema design, query optimization, high availability, clustering, and disaster recovery. Experience with cloud databases on AWS RDS is preferred.",
        "required_skills": ["oracle", "mysql", "sql server", "postgresql", "database"],
        "differentiating_skills": ["aws", "replication", "performance tuning"]
    },
    "Network Security Engineer": {
        "jd": "We need a Network Security Engineer with deep expertise in firewalls, VPNs, IDS/IPS, TCP/IP, Linux, and penetration testing. Relevant certifications (CISSP, CEH) or experience with Wireshark and vulnerability assessments required.",
        "required_skills": ["network security", "linux", "firewalls", "tcp/ip", "security"],
        "differentiating_skills": ["wireshark", "penetration testing", "cissp"]
    },
    "Blockchain": {
        "jd": "Seeking a Blockchain Engineer to develop decentralized smart contracts. Strong proficiency in Ethereum, Solidity, Web3.js, smart contract auditing, and cryptography required. Experience with Go, Rust, or Python is a plus.",
        "required_skills": ["blockchain", "smart contracts", "cryptography", "web3", "solidity"],
        "differentiating_skills": ["rust", "go", "ethereum"]
    },
    "Testing": {
        "jd": "Looking for a QA Automation Engineer experienced in Selenium, PyTest or JUnit, TestNG, and CI/CD testing pipelines. Strong skills in manual testing, automated regression suites, bug tracking with Jira, and API testing with Postman.",
        "required_skills": ["testing", "selenium", "jira", "pytest", "api testing"],
        "differentiating_skills": ["cicd", "performance testing", "postman"]
    }
}


def convert_kaggle_records_to_benchmark(records: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Converts raw Kaggle records into benchmark format."""
    benchmark_dataset = []

    for idx, rec in enumerate(records, 1):
        category = rec.get("Category", rec.get("category", "General")).strip()
        raw_text = rec.get("Resume", rec.get("resume", rec.get("Resume_str", "")))
        cleaned = clean_resume_text(raw_text)

        # Get matching JD template or fallback
        jd_info = CATEGORY_JOB_DESCRIPTIONS.get(category, {
            "jd": f"Seeking a skilled professional in {category} with demonstrated industry experience and strong technical proficiency.",
            "required_skills": [category.lower()],
            "differentiating_skills": ["leadership", "agile"]
        })

        cand_id = f"cand_k{idx:03d}"
        extracted_skills = extract_skills_from_text(cleaned)
        missing_skills = [s for s in jd_info["differentiating_skills"] if s not in extracted_skills]

        benchmark_dataset.append({
            "id": f"eval_k{idx:03d}",
            "candidate_id": cand_id,
            "candidate_name": f"Kaggle Candidate {idx:02d} ({category})",
            "category": category,
            "resume": cleaned[:1500],  # clean 1500 character resume excerpt
            "job_description": jd_info["jd"],
            "expected_top_matches": [cand_id],
            "expected_missing_skills": missing_skills
        })

    return benchmark_dataset


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not csv_path or not os.path.exists(csv_path):
        print("Usage: python scripts/ingest_kaggle_resumes.py <path_to_kaggle_csv>")
        sys.exit(1)

    print(f"Reading Kaggle dataset from: {csv_path}")
    records = []
    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)

    print(f"Loaded {len(records)} records from CSV.")
    benchmark = convert_kaggle_records_to_benchmark(records)

    out_json = os.path.join(PROJECT_ROOT, "evals", "kaggle_benchmark_dataset.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)

    print(f"Successfully converted and saved {len(benchmark)} samples to {out_json}")


if __name__ == "__main__":
    main()
