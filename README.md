# GBxCuLE Learning Lab

GPU-native many-env Game Boy runtime (Warp/CUDA) + benchmark/verification harness.

## What This Is

A research project exploring whether GPU-native emulation can accelerate reinforcement learning training loops. The reference oracle is [PyBoy](https://github.com/Baekalfen/PyBoy), a mature Python Game Boy emulator.

The system includes:
- **CPU baselines** (`pyboy_single`, `pyboy_vec_mp`) for honest comparison
- **Benchmark harness** measuring steps-per-second with proper warmup
- **Verification scaffold** comparing GPU vs CPU register states step-by-step
- **Micro-ROM test suite** exercising CPU instructions deterministically

## The Hypothesis

> A GPU-native multi-environment Game Boy runtime can achieve **meaningful steady-state throughput speedups** relative to CPU multiprocessing baselines on a moderately powerful NVIDIA GPU for emulator stepping workloads representative of RL training loops.

### What Counts as Success

| Metric | Target |
|--------|--------|
| Emulate-only throughput | ≥1.5× vs CPU baseline at scale |
| With reward extraction | ≥1.2× vs CPU baseline |
| Correctness | Zero register mismatches vs PyBoy oracle |

## Install (uv-only)

This project uses [uv](https://docs.astral.sh/uv/) as the single way to install and run everything.

```bash
# Clone the repo
git clone https://github.com/<your-fork>/gbxcule.git
cd gbxcule

# Install dependencies
make setup

# Install git hooks (runs checks before each commit)
make hooks
```

## Generate Micro-ROMs

The benchmark uses small deterministic ROMs that exercise specific CPU instructions:

```bash
make roms
```

Output: `bench/roms/out/*.gb`

## Run Baselines

```bash
make bench
```

Example output:
```
Backend: pyboy_single
ROM: ALU_LOOP.gb
Envs: 1
Steps: 100 (warmup: 10)
Time: 0.112s
Total SPS: 892.5
Per-env SPS: 892.5
Frames/sec: 21420.3
Artifact: bench/runs/20260121_153157_pyboy_single_ALU_LOOP.json

Backend: pyboy_vec_mp
ROM: ALU_LOOP.gb
Envs: 4
Steps: 100 (warmup: 10)
Time: 0.276s
Total SPS: 1447.9
Per-env SPS: 362.0
Frames/sec: 34749.7
Artifact: bench/runs/20260121_153158_pyboy_vec_mp_ALU_LOOP.json
```

### Direct CLI Usage

```bash
# Single environment
uv run python bench/harness.py \
  --backend pyboy_single \
  --rom bench/roms/out/ALU_LOOP.gb \
  --steps 1000 --warmup-steps 100

# Multiprocessing (4 envs across 2 workers)
uv run python bench/harness.py \
  --backend pyboy_vec_mp \
  --rom bench/roms/out/ALU_LOOP.gb \
  --steps 1000 --warmup-steps 100 \
  --num-envs 4 --num-workers 2

# Scaling sweep
uv run python bench/harness.py \
  --backend pyboy_vec_mp \
  --rom bench/roms/out/ALU_LOOP.gb \
  --steps 500 --warmup-steps 50 \
  --env-counts 1,2,4,8
```

## Run Verification (Expected to Fail in M0)

The verification mode compares a reference backend (PyBoy) against a device-under-test:

```bash
make verify
```

In M0, the DUT (`warp_vec`) is a stub that returns obviously wrong state, so verification always fails. This is intentional - the scaffold exists to catch real bugs once the GPU backend is implemented.

Example output:
```
Verification mode: ref=pyboy_single vs dut=warp_vec
ROM: ALU_LOOP.gb
Steps: 4, compare every 1

MISMATCH at step 0
First differing fields: ['pc', 'a', 'f', 'c', 'e']
Bundle: bench/runs/mismatch/20260121_153204_ALU_LOOP_pyboy_single_vs_warp_vec
Repro: bench/runs/mismatch/20260121_153204_ALU_LOOP_pyboy_single_vs_warp_vec/repro.sh
```

### Mismatch Bundles

When verification fails, a repro bundle is written containing:

| File | Contents |
|------|----------|
| `metadata.json` | ROM SHA, backends, seeds, git commit |
| `ref_state.json` | Reference CPU registers at mismatch |
| `dut_state.json` | DUT CPU registers at mismatch |
| `diff.json` | Field-by-field differences |
| `actions.jsonl` | Complete action trace for replay |
| `repro.sh` | One-command reproduction script |

To reproduce a mismatch:
```bash
bash bench/runs/mismatch/<bundle>/repro.sh
```

## Artifacts

All outputs are structured JSON with stable schemas:

```
bench/runs/
├── <timestamp>_<backend>_<rom>.json      # Benchmark results
├── <timestamp>__scaling.json              # Scaling sweep results
└── mismatch/
    └── <timestamp>_<rom>_<ref>_vs_<dut>/  # Verification failures
        ├── metadata.json
        ├── ref_state.json
        ├── dut_state.json
        ├── diff.json
        ├── actions.jsonl
        └── repro.sh
```

## Interpreting Results

### Steps Per Second (SPS) vs Frames Per Second

- **SPS**: How many `step()` calls per second (each step = multiple frames)
- **Frames/sec**: `SPS × frames_per_step` (default: 24 frames/step)
- **Total SPS**: `steps × num_envs / seconds` (aggregate throughput)
- **Per-env SPS**: `steps / seconds` (single-env equivalent)

For RL training, **Total SPS** is what matters - it's how many environment transitions you get per wall-clock second.

### Warmup vs Measured

- Warmup steps are excluded from timing (JIT compilation, cache warming)
- Only "measured_steps" count toward SPS calculations
- Check `results.warmup_steps` and `results.measured_steps` in artifacts

## Common Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install dependencies via uv |
| `make hooks` | Install git pre-commit hooks |
| `make fmt` | Format code and apply safe lint fixes |
| `make lint` | Check formatting and lint |
| `make test` | Run unit tests |
| `make roms` | Generate micro-ROMs |
| `make bench` | Run baseline benchmarks |
| `make verify` | Run verification (expected fail in M0) |
| `make check` | Run all checks (commit hook gate) |

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Engineering architecture and tech stack
- [CONSTITUTION.md](CONSTITUTION.md) - Project principles and constraints
- [history/prd.md](history/prd.md) - Product requirements document
- [history/m0-epics.md](history/m0-epics.md) - M0 milestone epics and stories

---

## Devlog: M0 Shipped

**TL;DR**: Built the measurement infrastructure first. CPU baselines + micro-ROMs + mismatch automation are live. No GPU code yet - that's the point.

### Why this order?

Most emulator projects start with the fun part (GPU kernels) and bolt on testing later. We're inverting that:

1. **Micro-ROMs**: Tiny deterministic ROMs that exercise specific instructions
2. **CPU baselines**: Honest comparison targets (single + multiprocessing)
3. **Verification scaffold**: Step-by-step register comparison with repro bundles

The hypothesis is that GPU-native Game Boy emulation can beat CPU multiprocessing at scale. But "beat" needs a definition, and "at scale" needs measurement. M0 gives us both.

### First numbers

```
pyboy_single:  892 SPS (1 env)
pyboy_vec_mp: 1448 SPS (4 envs, 2 workers)
```

These are our targets to beat. The GPU backend needs to exceed ~1.5× these numbers at scale (many envs) to validate the hypothesis.

### What's next

M1: Implement actual Warp/CUDA backend. The verification scaffold will catch every register mismatch. When `make verify` passes, we'll know the GPU is correct.

### Try it

```bash
git clone https://github.com/<fork>/gbxcule && cd gbxcule
make setup && make bench
```

All results land in `bench/runs/*.json` with schema version 1.
