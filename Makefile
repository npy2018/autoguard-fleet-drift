.PHONY: install test lint demo api
install:
	pip install -e '.[dev]'
test:
	pytest
lint:
	ruff check .
demo:
	python scripts/run_demo.py
api:
	uvicorn app:app --reload
