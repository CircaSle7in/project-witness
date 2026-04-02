.PHONY: all eval report transforms cockpit test lint format clean setup

all: lint test eval report

eval:
	python -m src.eval.harness

transforms:
	python scripts/run_transforms.py

report: transforms
	python -m src.eval.reporter

cockpit:
	python -m src.cockpit.app

test:
	python -m pytest tests/ -v

lint:
	python -m ruff check src/ tests/

format:
	python -m ruff format src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .ruff_cache
	rm -f data/results/*.duckdb

setup:
	python3.12 -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	@echo "Run 'source .venv/bin/activate' to activate"
