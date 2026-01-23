# GBxCuLE Learning Lab

GPU-native many-env Game Boy runtime (Warp/CUDA) + benchmark/verification harness.

## What This Is

A research project exploring whether GPU-native emulation can accelerate reinforcement learning training loops. The reference oracle is [PyBoy](https://github.com/Baekalfen/PyBoy), a mature Python Game Boy emulator.

The system includes:

- **CPU baselines** (`pyboy_single`, `pyboy_vec_mp`) for honest comparison
- **Benchmark harness** measuring steps-per-second with proper warmup
- **Verification harness** comparing `warp_vec_cpu` vs `pyboy_single` step-by-step
- **Micro-ROM test suite** exercising CPU instructions deterministically

## The Hypothesis

> A GPU-native multi-environment Game Boy runtime can achieve **meaningful steady-state throughput speedups** relative to CPU multiprocessing baselines on a moderately powerful NVIDIA GPU for emulator stepping workloads representative of RL training loops.

### What Counts as Success

| Metric                  | Target                                   |
| ----------------------- | ---------------------------------------- |
| Emulate-only throughput | ≥1.5× vs CPU baseline at scale           |
| With reward extraction  | ≥1.2× vs CPU baseline                    |
| Correctness             | Zero register mismatches vs PyBoy oracle |

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
Envs: 360
Steps: 100 (warmup: 10)
Time: 3.42s
Total SPS: 10522.9
Per-env SPS: 29.2
Frames/sec: 252548.7
Artifact: bench/runs/20260122_001005_pyboy_vec_mp_ALU_LOOP.json
```

## Run Verification (Expected to Pass)

Verification compares a reference backend (PyBoy) against a device-under-test:

```bash
make verify
```

The default `make verify` profile runs `pyboy_single` vs `warp_vec_cpu` on both micro-ROMs and should pass.

For a quick sanity pass (RL-ish step quantum):

```bash
make verify-smoke
```

To exercise the failure path and confirm mismatch bundles are being written:

```bash
make verify-mismatch
```

You can also run the harness directly. Example with memory hashing enabled for MEM_RWB:

```bash
uv run python bench/harness.py \
  --verify \
  --ref-backend pyboy_single \
  --dut-backend warp_vec_cpu \
  --rom bench/roms/out/MEM_RWB.gb \
  --verify-steps 1024 \
  --compare-every 1 \
  --frames-per-step 1 \
  --mem-region C000:C100
```

Example mismatch output:

```
Verification mode: ref=pyboy_single vs dut=warp_vec_cpu
ROM: MEM_RWB.gb
Steps: 1024, compare every 1
Memory regions: C000:C100

MISMATCH at step 17
First differing fields: ['pc', 'a', 'f', 'b', 'h']
Bundle: bench/runs/mismatch/<bundle>/
Repro: bench/runs/mismatch/<bundle>/repro.sh
```

## M3 (DGX) Gates

These targets are the **M3 contract** for DGX/CUDA runs and use the explicit
`warp_vec_cuda` backend:

```bash
make verify-gpu
make check-gpu
make bench-gpu
```

Defaults are defined in `Makefile` (override via variable prefixes, e.g.
`M3_VERIFY_STEPS=2048 make verify-gpu`). The scaling gate uses
`M3_ENV_COUNTS=1,8,64,512,2048,8192` and writes artifacts + report outputs to
`bench/runs/reports/<timestamp>/`.

## E4 Scaling (full_step)

E4 runs compute minimal reward/obs on device and use the suite ROMs.
They are **not** part of the commit gate; run them when you want scaling data.

```bash
make bench-e4-cpu
make bench-e4-gpu
```

Reports are written per ROM under:

```
bench/runs/reports/<timestamp>_e4_cpu/<ROM_ID>/
bench/runs/reports/<timestamp>_e4_gpu/<ROM_ID>/
```

Key overrides (all optional):

```bash
E4_ENV_COUNTS=1,8,64,512,2048,8192 \
E4_STEPS=200 \
E4_WARMUP_STEPS=10 \
E4_SYNC_EVERY=64 \
E4_FRAMES_PER_STEP=24 \
E4_RELEASE_AFTER_FRAMES=8 \
E4_STAGE=full_step \
E4_BASELINE_BACKEND=pyboy_puffer_vec \
E4_PUFFER_VEC_BACKEND=puffer_mp_sync \
E4_ACTION_GEN=seeded_random \
E4_ACTIONS_SEED=1234 \
E4_ACTION_CODEC=pokemonred_puffer_v0 \
make bench-e4-gpu
```

### Mismatch Bundles

When verification fails, a repro bundle is written containing:

| File             | Contents                             |
| ---------------- | ------------------------------------ |
| `metadata.json`  | ROM SHA, backends, seeds, git commit, GPU/driver, Warp version+provenance |
| `ref_state.json` | Reference CPU registers at mismatch  |
| `dut_state.json` | DUT CPU registers at mismatch        |
| `diff.json`      | Field-by-field differences           |
| `actions.jsonl`  | Complete action trace for replay     |
| `repro.sh`       | One-command reproduction script      |
| `rom.gb`         | Embedded ROM bytes (hermetic repro)  |
| `mem_ref_*.bin`  | Optional memory dumps (small regions) |
| `mem_dut_*.bin`  | Optional memory dumps (small regions) |

To reproduce a mismatch:

```bash
bash bench/runs/mismatch/<bundle>/repro.sh
```

## Artifacts

All outputs are structured JSON with stable schemas:

```
bench/runs/
├── reports/
│   └── <timestamp>/
│       ├── <timestamp>_pyboy_vec_mp_ALU_LOOP__scaling.json
│       ├── <timestamp>_warp_vec_cuda_ALU_LOOP__scaling.json
│       ├── summary.md
│       └── scaling.png
├── <timestamp>_<backend>_<rom>.json      # Benchmark results
├── <timestamp>__scaling.json              # Scaling sweep results
└── mismatch/
    └── <timestamp>_<rom>_<ref>_vs_<dut>/  # Verification failures
        ├── metadata.json
        ├── ref_state.json
        ├── dut_state.json
        ├── diff.json
        ├── actions.jsonl
        ├── rom.gb
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

| Command       | Description                            |
| ------------- | -------------------------------------- |
| `make setup`  | Install dependencies via uv            |
| `make hooks`  | Install git pre-commit hooks           |
| `make fmt`    | Format code and apply safe lint fixes  |
| `make lint`   | Check formatting and lint              |
| `make test`   | Run unit tests                         |
| `make roms`   | Generate micro-ROMs                    |
| `make bench`  | Run baseline benchmarks                |
| `make verify` | Run verification (expected pass)       |
| `make verify-smoke` | Quick verification smoke          |
| `make verify-mismatch` | Exercise mismatch bundle path |
| `make verify-gpu` | M3 must-pass verify (DGX/CUDA)      |
| `make check-gpu` | Fast-ish DGX gate (CUDA smoke)       |
| `make bench-gpu` | M3 scaling sweep (DGX/CUDA)          |
| `make check`  | Run all checks (commit hook gate)      |

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
pyboy_single:   890 SPS (1 env)
pyboy_vec_mp: 10523 SPS (360 envs, 20 workers)
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
