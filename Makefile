PYTHON := uv run python

.PHONY: help install run demo test report clean docker-build docker-run docker-demo

help:
	@echo "PDE-Agents-lite"
	@echo ""
	@echo "  make install       uv sync — create .venv from uv.lock (exact pinned env)"
	@echo "  make run           Start the interactive CLI (reads .env)"
	@echo "  make demo          Run a quick two-request offline demo"
	@echo "  make test          Run the full test suite"
	@echo "  make report        Regenerate FULL_REPORT.md (scope-compliance audit)"
	@echo "  make clean         Remove caches, .pyc files, and runs.db"
	@echo ""
	@echo "  make docker-build  Build the Docker image"
	@echo "  make docker-run    Run the interactive CLI in Docker"
	@echo "  make docker-demo   Run the offline demo in Docker"

install:
	uv sync

run:
	$(PYTHON) main.py

demo:
	$(PYTHON) main.py "Simulate a copper plate at 373K left, 273K right"
	$(PYTHON) main.py "What is the recent run history?"

test:
	$(PYTHON) tests/test_core.py

report:
	$(PYTHON) generate_report.py

clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -f runs.db

docker-build:
	docker compose build

docker-run:
	docker compose run --rm pde-agents

docker-demo:
	docker compose run --rm pde-agents python main.py \
		"Simulate a copper plate at 373K left, 273K right"
