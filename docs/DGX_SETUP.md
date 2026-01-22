# DGX Spark Setup (linux/aarch64)

This project uses `uv` for all installs. DGX Spark is **linux/aarch64**, and WS1 pins the Warp **CUDA 13** wheel via a direct URL.

## Install

```bash
uv sync
```

## Verify Warp + CUDA

```bash
uv run python - <<'PY'
import warp as wp
wp.init()
print("warp version:", wp.__version__)
print("devices:", [d.alias for d in wp.get_devices()])
PY
```

## Prove cu13 provenance (PEP 610)

```bash
uv run python - <<'PY'
import importlib.metadata as md
dist = md.distribution("warp-lang")
print(dist.read_text("direct_url.json"))
PY
```

The `direct_url.json` output should include the pinned cu13 wheel URL. If it is missing, the install likely came from PyPI instead of the direct URL.
