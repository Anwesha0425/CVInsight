"""Deploy CVInsight application to Hugging Face Spaces."""
import os
from huggingface_hub import HfApi

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print(f"Starting upload to Hugging Face from: {REPO_ROOT}")
api = HfApi()

api.create_repo(
    repo_id="Friday425/resume-matching-system",
    repo_type="space",
    space_sdk="docker",
    exist_ok=True
)

api.upload_folder(
    folder_path=REPO_ROOT,
    repo_id="Friday425/resume-matching-system",
    repo_type="space",
    ignore_patterns=[
        ".git/*",
        ".pytest_cache/*",
        "__pycache__/*",
        "*/__pycache__/*",
        "candidates.db",
        "uploads/*",
        ".env",
        "scripts/*",
        "tests/*",
        "evals/eval_results.json",
        "*.pyc"
    ]
)
print("Upload complete!")
