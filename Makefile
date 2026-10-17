.PHONY: install test lint typecheck format check report train serve

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

format:
	ruff format .
	ruff check --fix .

# The same gate as CI
check: lint typecheck
	pytest --cov-fail-under=80

# Whole synthetic study: docs/results.md and docs/img/*.png (about 30 seconds on a laptop)
report:
	recsys report

train:
	recsys train

serve: train
	recsys serve
