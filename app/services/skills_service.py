"""Skills taxonomy and NLP-based extraction service."""
from __future__ import annotations

import logging
from typing import Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Comprehensive skills taxonomy for exact matching
# ---------------------------------------------------------------------------
SKILLS_TAXONOMY: Set[str] = {
    # Programming Languages
    "python", "java", "c++", "c#", "javascript", "typescript", "ruby", "php",
    "swift", "kotlin", "go", "rust", "scala", "r", "matlab", "sql", "html",
    "css", "bash", "shell", "perl", "dart", "assembly",
    # Frameworks & Libraries
    "react", "angular", "vue", "node.js", "express", "django", "flask",
    "fastapi", "spring", "spring boot", "asp.net", "laravel", "ruby on rails",
    "next.js", "nuxt.js", "svelte", "bootstrap", "tailwind", "jquery",
    "pytorch", "tensorflow", "keras", "scikit-learn", "pandas", "numpy",
    "matplotlib", "seaborn", "opencv", "nltk", "spacy", "huggingface",
    "transformers", "graphql", "apollo", "prisma", "hibernate", "entity framework",
    # Databases
    "mysql", "postgresql", "sqlite", "oracle", "sql server", "mongodb", "redis",
    "cassandra", "dynamodb", "neo4j", "couchbase", "mariadb", "firebase",
    "supabase", "elasticsearch", "cockroachdb",
    # Cloud & DevOps
    "aws", "amazon web services", "azure", "gcp", "google cloud", "docker",
    "kubernetes", "terraform", "ansible", "jenkins", "gitlab ci", "github actions",
    "circleci", "travis ci", "nginx", "apache", "linux", "unix", "ubuntu",
    "centos", "prometheus", "grafana", "splunk", "datadog", "new relic", "kafka",
    "rabbitmq", "activemq", "celery", "airflow", "snowflake", "redshift", "bigquery",
    # Methodologies & Concepts
    "agile", "scrum", "kanban", "waterfall", "tdd", "test driven development",
    "bdd", "ci/cd", "continuous integration", "continuous deployment", "oop",
    "object oriented programming", "functional programming", "microservices",
    "serverless", "rest", "restful api", "soap", "mvc", "mvvm", "machine learning",
    "deep learning", "artificial intelligence", "data science", "data engineering",
    "nlp", "natural language processing", "computer vision", "data mining",
    "big data", "blockchain", "cryptography", "web3", "smart contracts",
    # Tools & Platforms
    "git", "github", "gitlab", "bitbucket", "jira", "confluence", "trello",
    "asana", "slack", "microsoft teams", "zoom", "figma", "sketch", "adobe xd",
    "photoshop", "illustrator", "salesforce", "hubspot", "tableau", "power bi",
    "looker", "excel",
    # Soft Skills
    "leadership", "management", "communication", "teamwork", "problem solving",
    "critical thinking", "project management", "agile methodologies",
}

# Lazy-load spaCy model
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Skill extraction will be limited.")
            _nlp = False  # sentinel so we don't retry
    return _nlp if _nlp else None


def extract_skills_from_text(text: str) -> Set[str]:
    """
    Extract skills from resume text using NLP token matching against SKILLS_TAXONOMY.
    Falls back to multi-word substring matching when spaCy is unavailable.
    """
    skills: Set[str] = set()
    text_lower = text.lower()

    nlp = _get_nlp()
    if nlp:
        doc = nlp(text_lower)
        for token in doc:
            if token.pos_ in {"PROPN", "NOUN", "X"} and len(token.text) > 1:
                if token.text in SKILLS_TAXONOMY:
                    skills.add(token.text)

    # Always check multi-word skills via substring
    for skill in SKILLS_TAXONOMY:
        if " " in skill and skill in text_lower:
            skills.add(skill)

    return skills
