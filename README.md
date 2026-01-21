# GBxCuLE Learning Lab

GPU-native many-env Game Boy runtime (Warp→CUDA) + benchmark/verification harness.

## Hypothesis

> A GPU-native multi-environment Game Boy runtime can achieve **meaningful steady-state throughput speedups** relative to CPU multiprocessing baselines on a moderately powerful NVIDIA GPU for emulator stepping workloads representative of RL training loops.

## Developer Quickstart

This project uses [uv](https://docs.astral.sh/uv/) as the single way to install and run everything.

```bash
# Install dependencies and create virtual environment
uv sync

# Run tests
uv run pytest

# Run the package entry point
uv run python -m gbxcule

# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type check
uv run pyright
```

## Lock Discipline

- `uv sync` is how you set up / update your environment
- `uv.lock` is committed and is the single source of truth
- In CI/hooks, use `uv run --locked` or `--frozen` to fail rather than silently rewrite lock

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Engineering architecture and tech stack
- [CONSTITUTION.md](CONSTITUTION.md) - Project principles and constraints
- [history/prd.md](history/prd.md) - Product requirements document
- [history/m0-epics.md](history/m0-epics.md) - M0 milestone epics and stories

## Artifacts

Benchmark results are written to `bench/runs/` (gitignored).
Generated micro-ROMs are written to `bench/roms/out/` (gitignored).
