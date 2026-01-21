# GBxCuLE Learning Lab - Makefile
# All commands use uv for reproducible environments

.PHONY: help setup fmt lint test roms bench smoke verify check hooks clean

# Variables
PY := uv run python
RUFF := uv run ruff
PYTEST := uv run pytest
PYRIGHT := uv run pyright

# Directories to lint/format (exclude third_party and generated)
SRC_DIRS := src bench tools tests

# Output directories
ROM_OUT := bench/roms/out
RUNS_OUT := bench/runs

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "GBxCuLE Learning Lab - Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Quick start:"
	@echo "  make setup    # Install dependencies"
	@echo "  make hooks    # Install git hooks"
	@echo "  make check    # Run all checks (used by pre-commit)"

setup: ## Install dependencies via uv
	uv sync
	@mkdir -p $(ROM_OUT) $(RUNS_OUT)
	@echo "Setup complete. Run 'make hooks' to install git hooks."

fmt: ## Format code and apply safe lint fixes
	$(RUFF) format $(SRC_DIRS)
	$(RUFF) check --fix $(SRC_DIRS)

lint: ## Check formatting and lint (no modifications)
	$(RUFF) format --check $(SRC_DIRS)
	$(RUFF) check $(SRC_DIRS)

test: ## Run unit tests
	$(PYTEST) -q

typecheck: ## Run type checking with pyright
	$(PYRIGHT)

roms: ## Generate micro-ROMs
	@mkdir -p $(ROM_OUT)
	$(PY) bench/roms/build_micro_rom.py

smoke: roms ## Run minimal sanity check (fast, for commit hook)
	@echo "Running smoke tests..."
	$(PYTEST) -q
	@echo "Smoke tests passed."

bench: roms ## Run baseline benchmarks
	@echo "Benchmark harness not yet implemented (Epic 7)"
	@echo "For now, run: make smoke"

verify: ## Run verification mode (scaffold)
	@echo "Verification mode not yet implemented (Epic 8)"
	@echo "For now, run: make test"

check: lint test smoke ## Run all checks (commit hook gate)
	@echo ""
	@echo "All checks passed!"

hooks: ## Install git hooks
	git config core.hooksPath .githooks
	@echo "Git hooks installed. Pre-commit will run 'make check'."

clean: ## Remove generated files
	rm -rf $(ROM_OUT)/* $(RUNS_OUT)/*
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "Cleaned generated files."
