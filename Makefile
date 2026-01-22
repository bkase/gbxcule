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
	@uv sync
	@mkdir -p $(ROM_OUT) $(RUNS_OUT)

fmt: ## Format code and apply safe lint fixes
	@$(RUFF) format $(SRC_DIRS)
	@$(RUFF) check --fix $(SRC_DIRS) || true

lint: ## Check formatting and lint (no modifications)
	@$(RUFF) format --check $(SRC_DIRS) > /dev/null
	@$(RUFF) check $(SRC_DIRS) > /dev/null

test: ## Run unit tests
	@$(PYTEST) -q --tb=short

typecheck: ## Run type checking with pyright
	@$(PYRIGHT) > /dev/null

roms: ## Generate micro-ROMs
	@mkdir -p $(ROM_OUT)
	@$(PY) bench/roms/build_micro_rom.py > /dev/null

smoke: roms ## Run minimal sanity check (fast, for commit hook)
	@# Smoke just ensures ROMs build; tests run separately in check

bench: roms ## Run baseline benchmarks
	@$(PY) bench/harness.py --backend pyboy_single --rom $(ROM_OUT)/ALU_LOOP.gb --steps 100 --warmup-steps 10
	@$(PY) bench/harness.py --backend pyboy_vec_mp --rom $(ROM_OUT)/ALU_LOOP.gb --steps 100 --warmup-steps 10 --num-envs 360 --num-workers 20

verify: roms ## Run verification mode (expected to fail in M0 due to DUT stub)
	@$(PY) bench/harness.py --verify --ref-backend pyboy_single --dut-backend warp_vec --rom $(ROM_OUT)/ALU_LOOP.gb --verify-steps 4 --compare-every 1 || echo "Mismatch expected (DUT is a stub). Check bench/runs/mismatch/ for repro bundle."

check: lint test smoke ## Run all checks (commit hook gate)

hooks: ## Install git hooks
	@git config core.hooksPath .githooks

clean: ## Remove generated files
	@rm -rf $(ROM_OUT)/* $(RUNS_OUT)/*
	@rm -rf .pytest_cache .ruff_cache
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
