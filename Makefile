SHELL := /bin/bash

init:
	poetry install
	poetry env info
	@echo "Created virtual environment"
test:
	poetry run pytest --cov=ekf_slam/ --cov-report=term-missing --no-cov-on-fail

format:
	ruff format
	ruff check --fix
	poetry run mypy ekf_slam/ tests/ config/ examples/ --ignore-missing-imports

clean:
	rm -rf .venv
	rm -rf .mypy_cache
	rm -rf .pytest_cache
	rm -rf build/
	rm -rf dist/
	rm -rf juninit-pytest.xml
	find . -name ".coverage*" -delete
	find . -name --pycache__ -exec rm -r {} +
