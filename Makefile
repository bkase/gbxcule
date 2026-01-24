# GBxCuLE Learning Lab - Makefile
# All commands use uv for reproducible environments

.PHONY: help setup setup-puffer ensure-puffer fmt lint test roms build-warp bench bench-cpu-puffer bench-e4-cpu bench-e4-gpu smoke verify verify-smoke verify-mismatch verify-gpu bench-gpu check-gpu check hooks clean

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

# M3 GPU gate defaults (override via make VAR=... on DGX)
M3_VERIFY_STEPS ?= 1024
M3_COMPARE_EVERY ?= 1
M3_FRAMES_PER_STEP ?= 1
M3_MEM_REGION ?= C000:C100
M3_ENV_COUNTS ?= 1,8,64,512,2048,8192
M3_BENCH_STEPS ?= 100
M3_BENCH_WARMUP_STEPS ?= 10
M3_BENCH_SYNC_EVERY ?= 64
M3_RELEASE_AFTER_FRAMES ?= 1
GPU_SMOKE_VERIFY_STEPS ?= 64

# E4 scaling defaults
E4_BASELINE_ENV_COUNTS ?= 1,8,64,128
E4_DUT_ENV_COUNTS ?= 1,8,64,512,2048,8192,16384
E4_STEPS ?= 200
E4_WARMUP_STEPS ?= 10
E4_SYNC_EVERY ?= 64
E4_FRAMES_PER_STEP ?= 24
E4_RELEASE_AFTER_FRAMES ?= 8
E4_STAGE ?= full_step
E4_BASELINE_BACKEND ?= pyboy_puffer_vec
E4_DUT_CPU_BACKEND ?= warp_vec_cpu
E4_DUT_GPU_BACKEND ?= warp_vec_cuda
E4_PUFFER_VEC_BACKEND ?= puffer_mp_sync
E4_ACTION_GEN ?= seeded_random
E4_ACTIONS_SEED ?= 1234
E4_ACTION_CODEC ?= pokemonred_puffer_v0
E4_SUITE ?= bench/roms/suite.yaml
WARP_BUILD_TIMEOUT ?= 900
DEV_WARP_MODE ?= debug

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

setup-puffer: ## Install pufferlib optional dependencies via uv
	@uv sync --group puffer
	@mkdir -p $(ROM_OUT) $(RUNS_OUT)

ensure-puffer: ## Ensure pufferlib + gymnasium deps are installed
	@$(PY) -c "import pufferlib, gymnasium" >/dev/null 2>&1 || { echo \"Installing puffer deps...\"; uv sync --group puffer; }

fmt: ## Format code and apply safe lint fixes
	@$(RUFF) format $(SRC_DIRS)
	@$(RUFF) check --fix $(SRC_DIRS) || true

lint: ## Check formatting and lint (no modifications)
	@$(RUFF) format --check $(SRC_DIRS) > /dev/null
	@$(RUFF) check $(SRC_DIRS) > /dev/null

test: ## Run unit tests
	@GBXCULE_SKIP_CUDA=1 GBXCULE_WARP_MODE=$(DEV_WARP_MODE) $(PYTEST) -q --tb=short

typecheck: ## Run type checking with pyright
	@$(PYRIGHT) > /dev/null

roms: ## Generate micro-ROMs
	@mkdir -p $(ROM_OUT)
	@$(PY) bench/roms/build_micro_rom.py > /dev/null

build-warp: fmt ## Compile Warp CPU kernels (warm cache for tests)
	@if command -v timeout >/dev/null 2>&1; then \
		GBXCULE_WARP_MODE=$(DEV_WARP_MODE) timeout $(WARP_BUILD_TIMEOUT) $(PY) -c 'from gbxcule.kernels.cpu_step import warmup_warp_cpu; [warmup_warp_cpu(stage=s, obs_dim=32) for s in ("emulate_only","full_step","reward_only","obs_only")]'; \
	else \
		GBXCULE_WARP_MODE=$(DEV_WARP_MODE) $(PY) -c 'from gbxcule.kernels.cpu_step import warmup_warp_cpu; [warmup_warp_cpu(stage=s, obs_dim=32) for s in ("emulate_only","full_step","reward_only","obs_only")]'; \
	fi

smoke: roms ## Run minimal sanity check (fast, for commit hook)
	@# Smoke just ensures ROMs build; tests run separately in check

bench: roms ## Run baseline benchmarks
	@$(PY) bench/harness.py --backend pyboy_single --rom $(ROM_OUT)/ALU_LOOP.gb --steps 100 --warmup-steps 10
	@$(PY) bench/harness.py --backend pyboy_vec_mp --rom $(ROM_OUT)/ALU_LOOP.gb --steps 100 --warmup-steps 10 --num-envs 360 --num-workers 20
	@$(PY) bench/harness.py --backend warp_vec_cpu --rom $(ROM_OUT)/ALU_LOOP.gb --steps 100 --warmup-steps 10

bench-cpu-puffer: roms ensure-puffer ## Run pufferlib CPU scaling sweep
	@$(PY) bench/harness.py --backend pyboy_puffer_vec --rom $(ROM_OUT)/ALU_LOOP.gb --env-counts $(M3_ENV_COUNTS) --steps $(M3_BENCH_STEPS) --warmup-steps $(M3_BENCH_WARMUP_STEPS) --frames-per-step $(M3_FRAMES_PER_STEP) --release-after-frames $(M3_RELEASE_AFTER_FRAMES) --puffer-vec-backend puffer_mp_sync

verify: roms ## Run verification (pyboy_single vs warp_vec_cpu; should pass)
	@$(PY) bench/harness.py --verify --ref-backend pyboy_single --dut-backend warp_vec_cpu --rom $(ROM_OUT)/ALU_LOOP.gb --verify-steps 1024 --compare-every 1 --frames-per-step 1
	@$(PY) bench/harness.py --verify --ref-backend pyboy_single --dut-backend warp_vec_cpu --rom $(ROM_OUT)/MEM_RWB.gb --verify-steps 1024 --compare-every 1 --frames-per-step 1 --mem-region C000:C100

verify-smoke: roms ## Quick verification smoke (frames_per_step=24)
	@$(PY) bench/harness.py --verify --ref-backend pyboy_single --dut-backend warp_vec_cpu --rom $(ROM_OUT)/ALU_LOOP.gb --verify-steps 16 --compare-every 1 --frames-per-step 24
	@$(PY) bench/harness.py --verify --ref-backend pyboy_single --dut-backend warp_vec_cpu --rom $(ROM_OUT)/MEM_RWB.gb --verify-steps 16 --compare-every 1 --frames-per-step 24 --mem-region C000:C100

verify-mismatch: roms ## Exercise mismatch bundle path (expected fail)
	@set -eu; \
	out="$$(mktemp)"; \
	if $(PY) bench/harness.py --verify --ref-backend pyboy_single --dut-backend stub_bad --rom $(ROM_OUT)/ALU_LOOP.gb --verify-steps 1 --compare-every 1 --frames-per-step 1 > "$$out" 2>&1; then \
		echo "Error: expected mismatch but verify passed" >&2; \
		cat "$$out" >&2; \
		rm "$$out"; \
		exit 1; \
	fi; \
	bundle="$$(grep '^Bundle: ' "$$out" | sed 's/^Bundle: //')"; \
	if [ -z "$$bundle" ]; then \
		echo "Error: missing bundle path in verify output" >&2; \
		cat "$$out" >&2; \
		rm "$$out"; \
		exit 1; \
	fi; \
	test -f "$$bundle/metadata.json"; \
	test -f "$$bundle/repro.sh"; \
	test -f "$$bundle/actions.jsonl"; \
	test -f "$$bundle/rom.gb"; \
	rm "$$out"

check: fmt lint typecheck roms build-warp test ## Run all checks (commit hook gate)

verify-gpu: roms ## M3 must-pass verify (DGX/CUDA)
	@command -v nvidia-smi >/dev/null 2>&1 || { echo "Error: CUDA GPU required (nvidia-smi not found)"; exit 1; }
	@$(PY) bench/harness.py --verify --ref-backend pyboy_single --dut-backend warp_vec_cuda --rom $(ROM_OUT)/ALU_LOOP.gb --verify-steps $(M3_VERIFY_STEPS) --compare-every $(M3_COMPARE_EVERY) --frames-per-step $(M3_FRAMES_PER_STEP)
	@$(PY) bench/harness.py --verify --ref-backend pyboy_single --dut-backend warp_vec_cuda --rom $(ROM_OUT)/MEM_RWB.gb --verify-steps $(M3_VERIFY_STEPS) --compare-every $(M3_COMPARE_EVERY) --frames-per-step $(M3_FRAMES_PER_STEP) --mem-region $(M3_MEM_REGION)

bench-gpu: roms ## M3 scaling sweep (DGX/CUDA)
	@command -v nvidia-smi >/dev/null 2>&1 || { echo "Error: CUDA GPU required (nvidia-smi not found)"; exit 1; }
	@set -eu; \
	report_dir="$(RUNS_OUT)/reports/$$(date -u +%Y%m%d_%H%M%S)"; \
	mkdir -p "$$report_dir"; \
	echo "Report dir: $$report_dir"; \
	$(PY) bench/harness.py --backend pyboy_vec_mp --rom $(ROM_OUT)/ALU_LOOP.gb --env-counts $(M3_ENV_COUNTS) --steps $(M3_BENCH_STEPS) --warmup-steps $(M3_BENCH_WARMUP_STEPS) --frames-per-step $(M3_FRAMES_PER_STEP) --release-after-frames $(M3_RELEASE_AFTER_FRAMES) --sync-every $(M3_BENCH_SYNC_EVERY) --output-dir "$$report_dir"; \
	$(PY) bench/harness.py --backend warp_vec_cuda --rom $(ROM_OUT)/ALU_LOOP.gb --env-counts $(M3_ENV_COUNTS) --steps $(M3_BENCH_STEPS) --warmup-steps $(M3_BENCH_WARMUP_STEPS) --frames-per-step $(M3_FRAMES_PER_STEP) --release-after-frames $(M3_RELEASE_AFTER_FRAMES) --sync-every $(M3_BENCH_SYNC_EVERY) --output-dir "$$report_dir"; \
	$(PY) bench/analysis/summarize.py --report-dir "$$report_dir" --out "$$report_dir/summary.md"; \
	$(PY) bench/analysis/plot_scaling.py --report-dir "$$report_dir" --out "$$report_dir/scaling.png"; \
	echo "Report: $$report_dir"

bench-e4-cpu: roms ensure-puffer ## E4 scaling sweep (CPU baseline vs warp_vec_cpu)
	@set -eu; \
	report_dir="$(RUNS_OUT)/reports/$$(date -u +%Y%m%d_%H%M%S)_e4_cpu"; \
	mkdir -p "$$report_dir"; \
	echo "Report dir: $$report_dir"; \
	$(PY) bench/harness.py --backend $(E4_BASELINE_BACKEND) --suite $(E4_SUITE) --env-counts $(E4_BASELINE_ENV_COUNTS) --steps $(E4_STEPS) --warmup-steps $(E4_WARMUP_STEPS) --frames-per-step $(E4_FRAMES_PER_STEP) --release-after-frames $(E4_RELEASE_AFTER_FRAMES) --stage $(E4_STAGE) --sync-every $(E4_SYNC_EVERY) --output-dir "$$report_dir" --action-gen $(E4_ACTION_GEN) --actions-seed $(E4_ACTIONS_SEED) --action-codec $(E4_ACTION_CODEC) --puffer-vec-backend $(E4_PUFFER_VEC_BACKEND); \
	$(PY) bench/harness.py --backend $(E4_DUT_CPU_BACKEND) --suite $(E4_SUITE) --env-counts $(E4_BASELINE_ENV_COUNTS) --steps $(E4_STEPS) --warmup-steps $(E4_WARMUP_STEPS) --frames-per-step $(E4_FRAMES_PER_STEP) --release-after-frames $(E4_RELEASE_AFTER_FRAMES) --stage $(E4_STAGE) --sync-every $(E4_SYNC_EVERY) --output-dir "$$report_dir" --action-gen $(E4_ACTION_GEN) --actions-seed $(E4_ACTIONS_SEED) --action-codec $(E4_ACTION_CODEC); \
	rom_ids="$$(SUITE_PATH="$(E4_SUITE)" $(PY) -c 'import os,yaml; from pathlib import Path; suite=yaml.safe_load(Path(os.environ["SUITE_PATH"]).read_text()); roms=suite.get("roms", []) if isinstance(suite, dict) else []; ids=[rom.get("id") or Path(rom["path"]).stem for rom in roms]; print(" ".join(ids))')"; \
	for rom_id in $$rom_ids; do \
		$(PY) bench/analysis/summarize.py --report-dir "$$report_dir/$$rom_id" --out "$$report_dir/$$rom_id/summary.md" --strict; \
		$(PY) bench/analysis/plot_scaling.py --report-dir "$$report_dir/$$rom_id" --out "$$report_dir/$$rom_id/scaling.png"; \
	done; \
	echo "Report: $$report_dir"

bench-e4-gpu: roms ensure-puffer ## E4 scaling sweep (DGX/CUDA; puffer baseline vs warp_vec_cuda)
	@command -v nvidia-smi >/dev/null 2>&1 || { echo "Error: CUDA GPU required (nvidia-smi not found)"; exit 1; }
	@set -eu; \
	report_dir="$(RUNS_OUT)/reports/$$(date -u +%Y%m%d_%H%M%S)_e4_gpu"; \
	mkdir -p "$$report_dir"; \
	echo "Report dir: $$report_dir"; \
	$(PY) bench/harness.py --backend $(E4_BASELINE_BACKEND) --suite $(E4_SUITE) --env-counts $(E4_BASELINE_ENV_COUNTS) --steps $(E4_STEPS) --warmup-steps $(E4_WARMUP_STEPS) --frames-per-step $(E4_FRAMES_PER_STEP) --release-after-frames $(E4_RELEASE_AFTER_FRAMES) --stage $(E4_STAGE) --sync-every $(E4_SYNC_EVERY) --output-dir "$$report_dir" --action-gen $(E4_ACTION_GEN) --actions-seed $(E4_ACTIONS_SEED) --action-codec $(E4_ACTION_CODEC) --puffer-vec-backend $(E4_PUFFER_VEC_BACKEND); \
	$(PY) bench/harness.py --backend $(E4_DUT_GPU_BACKEND) --suite $(E4_SUITE) --env-counts $(E4_DUT_ENV_COUNTS) --steps $(E4_STEPS) --warmup-steps $(E4_WARMUP_STEPS) --frames-per-step $(E4_FRAMES_PER_STEP) --release-after-frames $(E4_RELEASE_AFTER_FRAMES) --stage $(E4_STAGE) --sync-every $(E4_SYNC_EVERY) --output-dir "$$report_dir" --action-gen $(E4_ACTION_GEN) --actions-seed $(E4_ACTIONS_SEED) --action-codec $(E4_ACTION_CODEC); \
	rom_ids="$$(SUITE_PATH="$(E4_SUITE)" $(PY) -c 'import os,yaml; from pathlib import Path; suite=yaml.safe_load(Path(os.environ["SUITE_PATH"]).read_text()); roms=suite.get("roms", []) if isinstance(suite, dict) else []; ids=[rom.get("id") or Path(rom["path"]).stem for rom in roms]; print(" ".join(ids))')"; \
	for rom_id in $$rom_ids; do \
		$(PY) bench/analysis/summarize.py --report-dir "$$report_dir/$$rom_id" --out "$$report_dir/$$rom_id/summary.md" --strict; \
		$(PY) bench/analysis/plot_scaling.py --report-dir "$$report_dir/$$rom_id" --out "$$report_dir/$$rom_id/scaling.png"; \
	done; \
	echo "Report: $$report_dir"

check-gpu: roms ## Fast-ish DGX gate (CUDA smoke)
	@command -v nvidia-smi >/dev/null 2>&1 || { echo "Error: CUDA GPU required (nvidia-smi not found)"; exit 1; }
	@$(PY) bench/harness.py --verify --ref-backend pyboy_single --dut-backend warp_vec_cuda --rom $(ROM_OUT)/ALU_LOOP.gb --verify-steps $(GPU_SMOKE_VERIFY_STEPS) --compare-every $(M3_COMPARE_EVERY) --frames-per-step $(M3_FRAMES_PER_STEP)
	@$(PY) bench/harness.py --verify --ref-backend pyboy_single --dut-backend warp_vec_cuda --rom $(ROM_OUT)/MEM_RWB.gb --verify-steps $(GPU_SMOKE_VERIFY_STEPS) --compare-every $(M3_COMPARE_EVERY) --frames-per-step $(M3_FRAMES_PER_STEP) --mem-region $(M3_MEM_REGION)

hooks: ## Install git hooks
	@git config core.hooksPath .githooks

clean: ## Remove generated files
	@rm -rf $(ROM_OUT)/* $(RUNS_OUT)/*
	@rm -rf .pytest_cache .ruff_cache
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
