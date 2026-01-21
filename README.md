# GBxCuLE Learning Lab

GPU-native many-env Game Boy runtime (Warp→CUDA) + benchmark/verification harness.

## Hypothesis

> A GPU-native multi-environment Game Boy runtime can achieve **meaningful steady-state throughput speedups** relative to CPU multiprocessing baselines on a moderately powerful NVIDIA GPU for emulator stepping workloads representative of RL training loops.

## Developer Quickstart

This project uses [uv](https://docs.astral.sh/uv/) as the single way to install and run everything.

```bash
# Install dependencies
make setup

# Install git hooks (runs checks before each commit)
make hooks

# Run all checks (what the pre-commit hook runs)
make check

# See all available commands
make help
```

### Common Commands

| Command | Description |
|---------|-------------|
| `make setup` | Install dependencies via uv |
| `make hooks` | Install git pre-commit hooks |
| `make fmt` | Format code and apply safe lint fixes |
| `make lint` | Check formatting and lint (no modifications) |
| `make test` | Run unit tests |
| `make roms` | Generate micro-ROMs |
| `make smoke` | Run minimal sanity check |
| `make bench` | Run baseline benchmarks |
| `make check` | Run all checks (commit hook gate) |

### Direct uv Commands

You can also run commands directly with uv:

```bash
uv sync              # Install dependencies
uv run pytest        # Run tests
uv run python -m gbxcule  # Run package entry point
uv run ruff format . # Format code
uv run ruff check .  # Lint code
uv run pyright       # Type check
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
