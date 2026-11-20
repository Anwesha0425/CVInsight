.PHONY: dev test eval eval-full lint format clean deploy

dev:
	python run.py

test:
	python -m pytest tests/ -v

eval:
	python evals/run_evals.py --max-eval 5 --no-llm

eval-kaggle:
	python evals/run_evals.py --dataset evals/kaggle_benchmark_dataset.json --output evals/kaggle_eval_results.json --max-eval 5 --no-llm

eval-full:
	python evals/run_evals.py

lint:
	python -m ruff check app/ tests/ evals/ scripts/
	python -m mypy app/ --ignore-missing-imports

format:
	python -m black app/ tests/ evals/ scripts/ run.py wsgi.py
	python -m ruff check --fix app/ tests/

deploy:
	python scripts/push_hf.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f candidates.db
