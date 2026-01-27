# Simple RL — Milestone 1 Spec + Plan (Multi-env downsampled pixels on GPU)

Date: 2026-01-25

Source context:
- `history/simple-rl.md` → “Milestone 1 — Multi-env downsampled pixel buffer on GPU”
- `CONSTITUTION.md` → **this is the spec** (correctness-by-construction, functional core/imperative shell, verifiable rewards, latency budgets, snapshot verification)

This document is the **spec** for Milestone 1 and the execution plan. If behavior changes, update this document and the tests together.

---

## M1 Goal (from `history/simple-rl.md`)

Produce a downsampled pixel buffer on GPU:
- `pix: uint8[N, 72, 80]` (shades 0..3)
- updated every `step()` for all envs
- **no host copies in the hot path**

This pixel buffer is the “camera” for RL (policy sees pixels only; no RAM features).

---

## Non-negotiables (source constraints)

From `history/simple-rl.md`:
1. **Everything on GPU**: env stepping (Warp/CUDA) + policy/value forward/backward (PyTorch CUDA).
2. **No RAM reads for the policy** (pixels only).
3. **24 frames per step** (action frequency).
4. **Minimal + incrementally testable**: every milestone has a fast, deterministic gate.

Additional constraint (explicit):
- **Do not add a NOOP action.** Keep the action set/codecs exactly as-is.

---

## Definition of Done (DoD)

- **API/contract exists:** a single canonical “pixel buffer contract” defines shape, dtype, value range, and layout.
- **Renderer exists:** a Warp kernel produces `pix[N,72,80]` shades 0..3 for **all envs** directly from each env’s `mem`.
- **Backend wired:** `WarpVecCudaBackend.step()` launches the renderer kernel (no CPU readback; no forced global sync).
- **Torch interop works:** `wp.to_torch(pix_wp)` yields a stable, zero-copy Torch view for downstream RL code.
- **Verifiable reward:** automated tests provide exit-0/exit-1 gates:
  - determinism (same start + same actions → same pixel hashes),
  - multi-env correctness (per-env base addressing; no “env0 only” bug),
  - sanity vs scanline-accurate env0 renderer (downsampled comparison).
- **Fast by default:** CPU tests are fast; CUDA tests are optional/skipped when CUDA is unavailable.

---

## The Spec (CONSTITUTION) → M1 Requirements

Below are the Constitution clauses (spec) and how they constrain M1’s design/implementation.

### I. The Doctrine of Correctness

- **Correctness by Construction**
  - Make the pixel buffer contract explicit and hard to misuse:
    - dtype `uint8`, range `[0,3]`, shape `[N,72,80]`, memory layout env-major contiguous.
    - backend accessors raise if pixels were not enabled/allocated.
  - Make “env indexing” unambiguous: every PPU read uses `base = env_idx * MEM_SIZE`.

- **Functional Core, Imperative Shell**
  - Renderer kernel is a pure function of inputs: `mem + constants → out_pixels`.
  - Backend `step()` is orchestration only: run cpu-step kernel, then renderer kernel.

- **Unidirectional Data Flow**
  - Enforce the direction: `actions → cpu_step → pixels(out)`.
  - (Later milestones) `pixels → policy → actions` happens outside the backend, in a torch-only wrapper.

- **Verifiable Rewards**
  - M1 acceptance is driven by tests that produce a simple pass/fail signal.
  - Prefer hash/snapshot verification for pixel buffers.

- **The AI Judge**
  - Keep new modules ruff/pyright clean; follow repo conventions for Warp DSL files (exclude only if necessary).

### II. The Velocity of Tooling

- **Latency is Technical Debt**
  - Ensure CPU-default tests stay fast; CUDA-only verification is optional.

- **Native Speed / Density & Concision**
  - Keep the renderer minimal and self-contained; avoid over-engineering (no per-scanline latch arrays for all envs in M1).

- **Code is Cheap, Specs are Precious**
  - Treat this doc as the prompt: tests + code must line up with the contract defined here.

### III. The Shifting Left

- **Inverted Test Pyramid**
  - Gate correctness at the cheapest layers first:
    - unit-ish kernel invariants (hash determinism),
    - integration parity for a few deterministic micro ROMs,
    - optional CUDA parity.

- **Golden & Snapshot Verification**
  - For pixels, freeze hashes/snapshots rather than writing verbose per-pixel assertions.

### IV. Immutable Runtime / Deps

- **Supply Chain Minimalism**
  - No new dependencies for M1; torch interop is optional (import torch only where needed).

### V. Observability & Self-Healing

- **Structured mismatch artifacts**
  - On test failure, print enough structured info to repro: ROM name, step index, env index, hash, and (optionally) write a tiny image artifact behind a flag.

### VI. Knowledge Graph (Documentation)

- Keep this plan updated as code lands; treat it as living documentation.

---

## Design Decisions (M1)

### D1 — Pixel buffer contract

- Output representation:
  - `shade ∈ {0,1,2,3}` (DMG 4-shade)
  - `pix: uint8[N,72,80]`
  - Layout: `pix_flat[env_idx*out_h*out_w + oy*out_w + ox]`

- Downsample mapping:
  - Start with a deterministic nearest-neighbor rule from `160×144` → `80×72`:
    - `x = ox*2`, `y = oy*2`
  - If needed later, upgrade to 2×2 pooling (still deterministic) but do not do that in M1 unless tests show instability.

### D2 — Snapshot renderer (not scanline-accurate)

M1 explicitly does **not** attempt to reproduce scanline latch accuracy for all envs (too much memory/complexity).

Instead:
- read PPU regs “at end of step” directly from each env’s memory (`0xFFxx` IO region),
- render a single snapshot frame (BG + window + sprites) deterministically.

This renderer is the RL “camera”: it must be self-consistent and deterministic; pixel-perfect match to PyBoy is not required for all scenes, but we do need sanity checks.

### D3 — Action space remains unchanged

- No changes to action codecs.
- No NOOP added.
- RL must work with the existing action set.

---

## Implementation Plan

### M1.0 — Add kernel: `ppu_render_shades_downsampled_all_envs`

Add `src/gbxcule/kernels/ppu_render_downsampled.py`:
- `@wp.kernel` with `tid` over `num_envs*out_h*out_w`.
- Compute `(env_idx, oy, ox)` from `tid`.
- Compute `(x,y)` in 160×144 using the downsample rule.
- Read per-env PPU regs from `mem[base + 0xFFxx]`:
  - `LCDC, SCY, SCX, BGP, OBP0, OBP1, WY, WX` (and any others required for window/sprites).
- Render shade 0..3 using the same logic patterns as `src/gbxcule/kernels/ppu_step.py` but generalized for env indexing.
- Write to `out_pixels` (uint8).

Key invariants to enforce:
- every `mem[...]` read uses `base = env_idx * MEM_SIZE`,
- output is always in `[0,3]`,
- no global state, no host IO.

### M1.1 — Wire into backend (CUDA-first)

Update `src/gbxcule/backends/warp_vec.py`:
- Add an opt-in flag (recommend: `render_pixels: bool = False`) to avoid changing existing `render_bg` semantics.
- Allocate `self._pix` as `wp.zeros(num_envs*out_h*out_w, dtype=wp.uint8, device=device)` when enabled.
- In `step()`, after the main cpu-step kernel, launch the new renderer kernel when enabled.
- Expose accessors:
  - `pixels_wp()` → Warp array
  - `pixels_torch()` → `wp.to_torch(self._pix).view(N,72,80)` (optional; raise cleanly if torch absent)

Concurrency rule:
- do not add a forced global `wp.synchronize()` to the CUDA path (keep stream-ordered async behavior).

### M1.2 — Tests (verifiable rewards)

Add `tests/test_ppu_downsampled_pixels.py` with CPU-fast gates:

1) **Determinism**
- For deterministic micro ROM(s), fixed seed + fixed action trace yields identical `pix` hash sequence.

2) **Multi-env correctness**
- Run `num_envs >= 2` with divergent env behavior and assert pixel hashes diverge accordingly.
- This catches missing `base = env_idx * MEM_SIZE` bugs.

3) **Sanity vs env0 scanline-accurate renderer**
- For env0 only, compare:
  - `downsample(frame_bg_shade_env0)` vs `pix[0]` produced by the snapshot renderer
  - allow exact match for stable ROMs where latches don’t matter; otherwise enforce a tight mismatch threshold with clear failure output.

Optional CUDA parity (skipped when CUDA unavailable):
- For one micro ROM, assert CPU and CUDA snapshot renderers match exactly (same mem/state/actions).

Failure artifacts:
- On mismatch, print: ROM, step_idx, env_idx, hash values; optionally dump PNGs behind a flag.

### M1.3 — Minimal debug tooling (non-gating)

Add a tiny script under `tools/` or `bench/` (non-test) to:
- run a few steps,
- save `pix[0]` to a PNG (palette-mapped),
- help visually confirm “reasonable frames”.

This is optional and should not be required for DoD.

### M1.4 — Update docs/spec pointers

- Update `history/simple-rl.md` to point at the actual kernel/backend APIs once implemented (file names + how to enable pixels + which tests prove it).

---

## Risks & Mitigations

- **R1: Snapshot renderer differs from scanline-accurate env0 renderer**
  - Mitigation: treat snapshot renderer as the RL camera; validate self-consistency and a small set of stable scenes; keep env0 accurate path for debugging only.

- **R2: False positives / overly strict parity checks**
  - Mitigation: for sanity comparisons, use a mismatch threshold and report mismatch stats; only require exact matches on stable micro ROMs.

- **R3: Performance regressions (kernel too heavy)**
  - Mitigation: keep downsample dimensions small (80×72), avoid extra passes, avoid syncs, and prefer simple deterministic sampling in M1.

