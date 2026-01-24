# M4 Workstream 5 Plan — E4 Scaling Reports

This document is **Workstream 5 as referenced in** `history/m4-architecture.md` (“Workstream / PR slicing”, item 5: **E4 scaling reports**). It is **not** the older WS5 numbering from `history/m3-workstreams.md`.

This is the **spec phase** plan (per `CONSTITUTION.md`): do all research now, then implement with verifiable gates.

---

## 0) Goal + outputs (Definition of Done)

- One command produces a **timestamped report folder** under `bench/runs/reports/...` containing, **for each E4 ROM**, the two scaling artifacts (baseline + DUT), a summary table, and a plot (plus an optional top-level index).
- Reports are **honest**: configs match (ROM hash, action schedule, stage, action generator, sync policy, etc.), and artifacts include enough metadata to prevent “apples-to-oranges” comparisons.

---

## 1) Benchmark matrix to lock down (spec)

Use the E4 protocol described in `history/m4-architecture.md` §9 (“Benchmark protocol for E4”) and the “receipt” requirement in §9.3 (“Divergence receipt”).

### 1.1 ROMs (suite-driven)

- Run on the micro-ROM suite (`bench/roms/suite.yaml`), which should include:
  - `ALU_LOOP` (already there)
  - `MEM_RWB` (already there)
  - `JOY_DIVERGE_PERSIST` (added in WS3 per `history/m4-architecture.md`)

### 1.2 Backends

- **Baseline (CPU):** `pyboy_puffer_vec` when available (WS4), with a documented fallback to `pyboy_vec_mp` for non-DGX environments.
- **DUT (GPU):** `warp_vec_cuda`
- Optional secondary comparisons (nice-to-have, not required for M4 WS5):
  - `warp_vec_cpu` vs baseline (helps debug stage overhead without GPU)

### 1.3 Stage + schedule

- Stage: `full_step` (this is the “E4” point of the ladder).
- Schedule defaults (RL-ish): `frames_per_step=24`, `release_after_frames=8` (matches `bench/roms/suite.yaml` defaults).
- Sync policy: keep explicit (`sync_every`) and record it in artifacts (harness already records `sync_every` in scaling artifacts).

### 1.4 Actions (must force divergence when needed)

- For ALU/MEM: `seeded_random` with fixed seed for reproducibility.
- For `JOY_DIVERGE_PERSIST`: add/standardize a `striped` generator (described in `history/m4-architecture.md`) so envs are deterministically split across buttons and divergence is sustained.
- Artifacts must record generator name/version/seed/pattern (current schema only has `{name, version, seed}`).

### 1.5 Divergence “receipt” (integrity signal)

- For divergence ROMs, record a cheap end-of-window receipt: `unique_pc_count` (and/or ratio) computed once per env-count run **after timing** (per `history/m4-architecture.md`) so measurement isn’t polluted.

---

## 2) Harness wiring required for WS5 (minimal changes, high leverage)

### 2.1 Make `--suite` real for scaling sweeps

Today `--suite` only uses the first ROM. For WS5:

- Scaling mode with `--suite` iterates **all** suite ROMs, honoring per-ROM overrides (`frames_per_step`, `release_after_frames`), and writes artifacts into per-ROM subdirs.
- Suggested output layout:
  - `bench/runs/reports/<ts>_e4/<ROM_ID>/...__scaling.json`
  - This prevents the analysis scripts from accidentally pairing artifacts from different ROMs.

### 2.2 Stage must actually affect execution

- The CLI accepts `--stage` and records it in scaling artifacts, but the benchmark loop never passes stage into the backend stepping loop (it just calls `backend.step(actions)`).
- WS5 should include a **sanity assertion mechanism** so “full_step” reports aren’t silently “emulate_only”:
  - simplest: require backends expose `backend.stage` and record it (and/or record a `kernel_name` / `kernel_variant` string for Warp).

### 2.3 Add `striped` action generator + metadata

- Implement `striped` in harness action generation (today only `noop` and `seeded_random` exist).
- Metadata should include the pattern and codec name/version once WS1 exists.

### 2.4 Add divergence receipt hook (post-timing)

- Add an optional per-run “receipt” extraction hook (e.g., `get_pc_snapshot()` on Warp backends) that is called **after** measurement and stored in results entries.

---

## 3) Report generation (Makefile + analysis scripts)

### 3.1 New Make targets (don’t break M3)

- Keep existing M3 gates unchanged (`make verify-gpu`, `make bench-gpu`, etc.).
- Add:
  - `make bench-e4-cpu` (runs baseline vs `warp_vec_cpu` across suite, generates reports)
  - `make bench-e4-gpu` (DGX/CUDA-only; runs baseline vs `warp_vec_cuda`, generates reports)
- Define E4-tunable knobs (mirroring existing M3 style variables):
  - `E4_ENV_COUNTS`, `E4_STEPS`, `E4_WARMUP_STEPS`, `E4_FRAMES_PER_STEP`, `E4_RELEASE_AFTER_FRAMES`, `E4_SYNC_EVERY`, `E4_STAGE=full_step`
  - `E4_BASELINE_BACKEND` defaulting to `pyboy_puffer_vec` with a documented fallback path.

### 3.2 Per-ROM summaries/plots (reuse existing scripts; tighten mismatch checks)

Current summarizer/plotter assume one ROM per dir and hardcode backend names.

- Update `BASELINE_BACKENDS` to include `pyboy_puffer_vec` (and optionally keep `pyboy_vec_mp`).
- Expand config mismatch checks to include `release_after_frames` (and later `action_codec`, `action_schedule`, `vec_backend`).
- Run summary generation with `--strict` in the Make target so misconfigured comparisons fail fast.

### 3.3 Optional top-level index

- Generate `bench/runs/reports/<ts>_e4/index.md` that links each ROM’s `summary.md` and embeds key headline numbers (max speedup, best env-count, and whether E4 target ≥1.2× is met).

---

## 4) WS5 Acceptance checklist

- `make bench-e4-gpu` produces, in one run:
  - per-ROM report folders (suite-driven)
  - 2 scaling artifacts per ROM (baseline + DUT)
  - `summary.md` + `scaling.png` per ROM
  - (optional) `index.md` at report root
- Summary generation fails if configs differ (strict “artifact honesty”).
- Divergence ROM report includes a receipt (e.g., `unique_pc_count`) to prove it really diverged.
- No impact to the ≤2-minute CPU gate: E4 targets are **not** part of `make check`.

---

## 5) Suggested beads issue breakdown (optional)

- WS5.1 “Suite scaling sweep support” (make `--suite` iterate ROMs end-to-end)
- WS5.2 “E4 make targets + env knobs”
- WS5.3 “Striped generator + divergence receipt recording”
- WS5.4 “Summarize/plot support for puffer baseline + stricter config checks”
- WS5.5 “README: M4/E4 bench instructions”

