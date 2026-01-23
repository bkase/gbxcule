# DGX Spark PufferLib Setup

This repo supports **real PufferLib** baselines for PyBoy via the optional
dependency group `puffer`.

## 1) System packages (Ubuntu/DGX)

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  python3-dev \
  libgl1 \
  libglib2.0-0 \
  libsdl2-2.0-0
```

If OpenCV wheels fail to import, add:

```bash
sudo apt-get install -y libsm6 libxext6 libxrender1
```

## 2) Install Python deps

```bash
uv sync --group puffer
```

### Note on NumPy

`pufferlib==1.0.1` requires **numpy==1.23.3**. Running `uv sync --group puffer`
will downgrade NumPy in the current environment. If you need to keep NumPy 2.x
for Warp/CUDA work, use a dedicated venv for puffer baselines:

```bash
uv venv .venv-puffer
source .venv-puffer/bin/activate
uv sync --group puffer
```

## 3) Smoke test (Serial backend)

```bash
make roms
uv run python bench/harness.py \
  --backend pyboy_puffer_vec \
  --puffer-vec-backend puffer_serial \
  --rom bench/roms/out/ALU_LOOP.gb \
  --num-envs 1 \
  --steps 2 \
  --warmup-steps 1 \
  --frames-per-step 1
```

## 4) Scaling sweep (Multiprocessing backend)

```bash
make bench-cpu-puffer
```
