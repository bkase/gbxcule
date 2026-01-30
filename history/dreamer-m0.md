# Dreamer v3 M0 Plan: Scaffolding + Golden Bridge Skeleton

This plan derives from:

- `history/dreamer-plan.md` (official Dreamer v3 milestone sequence)
- `history/dreamer-gotchas.md` (implementation constraints discovered in analysis)
- repo conventions in `src/gbxcule/rl/` (lazy torch imports, small helpers, strict validation)
- sheeprl reference implementation in `third_party/sheeprl/`:
  - `sheeprl/algos/dreamer_v3/agent.py` (RSSM + init weights + model wiring)
  - `sheeprl/algos/dreamer_v3/loss.py` (KL weighting + continue loss)
  - `sheeprl/algos/dreamer_v3/utils.py` (Moments/ReturnEMA + init)
  - `sheeprl/configs/algo/dreamer_v3.yaml` (default hyperparameters)

M0 is about building the **verification scaffolding and contracts** _before_ heavy model code. It must be fast, deterministic, and CPU-only.

---

## M0 Objective

Create the Dreamer v3 scaffolding that enforces **precision policy**, **RNG determinism**, and **shape/dtype contracts**, plus a **Golden Bridge skeleton** for parity fixtures. This should compile cleanly, run deterministic tests, and avoid pulling in any training dependencies at import time.

---

## Non-Negotiables (M0)

- **Precision policy is explicit**: replay inputs are `uint8` (packed2), model internals are `float32`, and RNN/GRU math must be defined as FP32 in policy.
- **No global RNG** in core Dreamer components. All randomness must accept an explicit `torch.Generator`.
- **Time‑major tensors** are the internal contract; helper functions should enforce or convert explicitly.
- **No dependency pollution**: the production package must not import `hydra`, `lightning`, `fabric`, etc. Fixture generation lives in `tools/`.
- **CPU‑only tests** and deterministic algorithms for M0 gates.
- **Reference parity must track sheeprl**: config names and defaults should align with `third_party/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml` to avoid future migration.

---

## Deliverables

### 1) Dreamer module scaffolding
Create `src/gbxcule/rl/dreamer_v3/` with:

- `__init__.py`: minimal exports only.
- `config.py`: dataclasses + validation (see below).
- `schema.py`: dtype/shape constants + assertion helpers.
- `rng.py`: generator utilities; forbids global RNG usage.
- `engine_cpu.py` / `engine_cuda.py`: stub engine classes with explicit TODOs.

### 2) Golden Bridge skeleton

- `tests/fixtures/dreamer_v3/` directory (empty but committed).
- `tests/fixtures/dreamer_v3/manifest.json` (empty list or placeholder schema).
- `tools/dreamer_v3/golden_bridge.py` (CLI skeleton):
  - accepts `--out tests/fixtures/dreamer_v3/` and `--subset` flags.
  - dynamically imports reference implementation **inside** `main()`.
  - writes fixtures + manifest, never imported by production code.

### 3) Test harness (`tests/rl_dreamer/`)

- `conftest.py`: enables deterministic algorithms when possible.
- `test_config.py`: config validation (happy/invalid cases).
- `test_rng.py`: generator required; determinism across runs.
- `test_schema.py`: asserts for time‑major, dtype, packed2 shape.
- `test_fixtures.py`: loader handles empty/missing fixtures gracefully (skip with message).

---

## Config Plan (M0)

Create a minimal-but-forward-compatible config set with explicit validation. Use sheeprl's naming where possible so later parity work is a data-mapping exercise, not a refactor.

- `PrecisionPolicy`:
  - `model_dtype` (default `float32`)
  - `rnn_dtype` (default `float32`, matches gotcha)
  - `autocast` (bool, default `False`)
  - `allow_tf32` (bool, default `False`)

- `ReplayConfig` (still needed for our engine path):
  - `capacity` (int)
  - `seq_len` (int)
  - `batch_size` (int)
  - `commit_stride` (int)

- `AlgoConfig` (sheeprl-aligned keys; include now even if not wired yet):
  - `gamma` (default `0.996996996996997`)
  - `lmbda` (default `0.95`)
  - `horizon` (default `15`)
  - `replay_ratio` (default `1`)
  - `learning_starts` (default `1024`)
  - `per_rank_batch_size` (int)
  - `per_rank_sequence_length` (int)
  - `unimix` (default `0.01`)
  - `hafner_initialization` (default `True`)
  - `cnn_keys.encoder/decoder` and `mlp_keys.encoder/decoder` (lists, default empty)

- `WorldModelConfig` (sheeprl-aligned scalars now, structured subconfigs later):
  - `discrete_size` (default `32`)
  - `stochastic_size` (default `32`)
  - `kl_dynamic` (default `0.5`)
  - `kl_representation` (default `0.1`)
  - `kl_free_nats` (default `1.0`)
  - `kl_regularizer` (default `1.0`)
  - `continue_scale_factor` (default `1.0`)
  - `decoupled_rssm` (default `False`)
  - `learnable_initial_recurrent_state` (default `True`)
  - `reward_model.bins` (default `255`)

- `ActorConfig` / `MomentsConfig` (future ReturnEMA alignment):
  - `moments.decay` (default `0.99`)
  - `moments.max` (default `1.0`)
  - `moments.percentile.low/high` (defaults `0.05/0.95`)

- `CriticConfig` (bins default `255`) and other fields can be placeholders for now.

- `DreamerV3Config`:
  - `precision: PrecisionPolicy`
  - `replay: ReplayConfig`
  - `algo: AlgoConfig`
  - `world_model: WorldModelConfig`
  - `actor: ActorConfig` (moments only for M0)
  - `critic: CriticConfig` (bins only for M0)
  - `seed` (int)
  - `distribution` (placeholder dict or small dataclass, per sheeprl `distribution.type`)

Validation rules (M0 tests must enforce):

- `bins >= 2`
- `kl_free_nats >= 0`
- `kl_dynamic/kl_representation/kl_regularizer >= 0`
- `percentile.low < percentile.high` and both in `(0, 1)`
- `commit_stride >= 1`
- `seq_len >= 2`
- `batch_size >= 1`
- `discrete_size >= 2`, `stochastic_size >= 1`
- dtype values are members of an allowed set (`torch.float32` only for now)

---

## Schema Plan (M0)

`schema.py` should codify the canonical tensor contracts and allow strict checks without importing models:

- Constants:
  - `OBS_PACKED2_DTYPE = uint8`
  - `OBS_PACKED2_SHAPE = (1, 72, 20)`
  - `ACTION_DTYPE = int32` (int64 ok for inputs)
  - `REWARD_DTYPE = float32`
  - `CONTINUE_DTYPE = float32`
  - `EPISODE_ID_DTYPE = int32`

- Helper assertions (pure utility functions):
  - `assert_time_major(t, name)` (expects `[T, B, ...]`)
  - `assert_packed2_obs(t, name)` (expects `[T, B, 1, 72, 20]` and `uint8`)
  - `assert_float32(t, name)` / `assert_int(t, name)`

Keep these as **functional core** helpers so they can be reused by ReplayRing and RSSM later.

---

## RNG Plan (M0)

`rng.py` should provide the only allowed RNG interface for Dreamer core code:

- `require_generator(gen)` → raises if `None`
- `make_generator(seed, device=None)` → returns `torch.Generator`
- `fork_generator(gen)` → deterministic child generator
- `randn`, `rand`, `randint` wrappers that **require** a generator

Tests must prove that two generators with the same seed produce identical results and that passing `None` raises.

---

## Golden Bridge Plan (M0 Skeleton)

The Golden Bridge exists to generate tiny parity fixtures from a trusted Dreamer v3 implementation without contaminating production imports.

M0 requirements:

- A stub CLI that documents required env vars or paths for the reference code.
- A loader that can read a manifest + tensor files (but **does not require** fixtures to exist yet).
- Fixture format: pick something simple and stable now (suggested: `torch.save` `.pt` files + `manifest.json`).
- Manifest fields: `name`, `dtype`, `shape`, `file`, `notes`.
- The CLI should assume the local reference implementation lives at `third_party/sheeprl/` and add it to `sys.path` inside `main()`.
- When fixture generation begins (M1+), mirror sheeprl preprocessing quirks:
  - `data["is_first"][0] = 1`
  - actions are shifted with an explicit zero action at time 0
- For two-hot bins, record the exact bin tensor into fixtures to avoid `linspace` drift (see `sheeprl/utils/utils.py`).

M0 does **not** require actual fixtures yet; only the skeleton and loader behavior.

---

## Implementation Steps (Detailed)

1) **Track the work (beads_rust)**
   - `br create --title="dreamer v3 m0: scaffolding + golden bridge skeleton" --type=task --priority=2`
   - `br update <id> --status=in_progress`

2) **Extract sheeprl-aligned defaults**
   - Read `third_party/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml` and `dreamer_v3_*` overrides.
   - Record defaults in `config.py` and tests so naming matches sheeprl.

3) **Add Dreamer module scaffolding**
   - Create `src/gbxcule/rl/dreamer_v3/` and empty `__init__.py`.
   - Implement `config.py`, `schema.py`, `rng.py` per above.
   - Add `engine_cpu.py` / `engine_cuda.py` stubs (class + docstring + `NotImplementedError`).
   - Use the repo pattern of lazy `torch` import (`_require_torch`).

4) **Golden Bridge skeleton**
   - Add `tools/dreamer_v3/golden_bridge.py` with CLI help, no hard dependencies.
   - Add `tests/fixtures/dreamer_v3/manifest.json` with empty placeholder list.
   - Add `tests/rl_dreamer/fixtures.py` loader to read manifest + files.

5) **Test harness**
   - `tests/rl_dreamer/conftest.py` to force deterministic algorithms where safe.
   - `tests/rl_dreamer/test_config.py` for validation.
   - `tests/rl_dreamer/test_rng.py` for generator requirements.
   - `tests/rl_dreamer/test_schema.py` for contract checks.
   - `tests/rl_dreamer/test_fixtures.py` to assert graceful skip or error messages.

6) **Housekeeping**
   - Ensure `__all__` exports and docstrings are present for new modules.
   - Keep all files ASCII; no extra deps.

---

## Test Plan / Gates

- M0 CPU gate (required):
  - `uv run pytest -q tests/rl_dreamer`

- Optional sanity (full suite, not required for M0):
  - `uv run pytest -q tests/test_rl_models.py` (ensures no import regressions)

No GPU tests in M0.

---

## Acceptance Criteria

M0 is complete when:

- The Dreamer v3 scaffolding modules exist and import cleanly.
- Config validation and RNG contract tests are deterministic and pass on CPU.
- The Golden Bridge skeleton can run (even if it emits no fixtures yet) and the loader handles empty fixtures gracefully.
- No new runtime dependencies or global RNG usage are introduced.

---

## Non‑Goals (Explicitly Out of Scope)

- ReplayRing implementation (M1).
- Distributions / symlog-twohot math (M2).
- RSSM, world model, or behavior learning (M3–M5).
- CUDA ingestion or async engine work (M6–M7).

---

## Open Questions / Decisions to Confirm

- **Fixture format choice:** `.pt` vs `.npz` (recommend `.pt` to preserve dtypes precisely).
- **Reference implementation location:** local checkout path or pinned commit hash?
- **Config defaults:** pick conservative defaults now or defer some fields until M1/M2?

---

## Session Close‑Out (Repo Protocol)

- `git status`
- `git add <files>`
- `br sync --flush-only`
- `git commit -m "dreamer v3 m0: scaffolding + golden bridge skeleton"`
- `git push`
- `br close <id> --reason="Completed"`
