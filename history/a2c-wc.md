# Workstream C Plan — Streaming A2C Trainer (TD(0), no rollout buffer)

This is the detailed plan for **Workstream C** (“Streaming A2C trainer”) derived from:

- `history/a2c-training.md` (note: `history/a2c-train.md` does not exist in this repo)
- `CONSTITUTION.md` (**this is the spec**; mapped below into concrete requirements)

Target date context: **2026-01-27**.

---

## Workstream C Goal (What We Ship)

Deliver `tools/rl_m5_train_a2c.py`: a **streaming A2C / TD(0)** trainer (no rollout buffer) that:

- Scales to **N = 8,192–16,384 envs** (GPU), with **T=1** streaming updates and optional **grad accumulation**.
- Works with the existing pixels-only env (`PokeredPixelsGoalEnv`) and model (`PixelActorCriticCNN`).
- Writes **structured JSONL logs** (one record per optimizer step) and **atomic checkpoints** (resume-able).
- Includes a **small-N smoke / self-test mode** that is fast and verifiable (exit 0/1).
- Minimizes per-step overhead; avoids logging giant tensors.

Non-goals (for Workstream C scope):

- Adding the NOOP action codec v1 (Workstream A).
- Env perf changes like `stack_k==1` fast path (Workstream B).
- Evaluation script correctness improvements (Workstream D).
- Goal template generation pipeline (Workstream E).

---

## Constitution (Spec) → Workstream C Requirements

The constitution is the spec. This section translates every relevant clause into “what we must do” for Workstream C.

### I. The Doctrine of Correctness

**Correctness by Construction: make invalid states unrepresentable.**

- Define a single `TrainConfig` dataclass for the trainer with explicit types + validation:
  - Enforce invariants early: `num_envs>=1`, `stack_k>=1`, `update_every>=1`, `total_env_steps>=num_envs`, `max_steps>=1`.
  - Enforce sensible ranges: `gamma in (0, 1]`, `lr>0`, `entropy_coef>=0`, `value_coef>=0`, `grad_clip>0`.
- Enforce strict tensor contracts at boundaries (fail fast):
  - Env expects `actions`: CUDA `torch.int32`, shape `[N]` (no implicit cast).
  - Model expects `obs`: `torch.uint8`, shape `[N, K, H, W]`.
  - Loss path expects:
    - `logits`: `float32 [N, A]`
    - `values`: `float32 [N]`
    - `rewards`: `float32 [N]`
    - `done/trunc`: `bool [N]`
    - `v_next`: `float32 [N]` under `no_grad`.

**Functional Core, Imperative Shell.**

- Put TD(0) math + loss computation in a pure-ish module (CPU-testable), e.g. `src/gbxcule/rl/a2c.py`.
- Keep `tools/rl_m5_train_a2c.py` as orchestration only:
  - args/config, env/model/optim setup, training loop, logging/checkpointing, clean shutdown.

**Unidirectional Data Flow (Reducer-style where possible).**

- Treat each optimizer step as: `(state, observed stats) -> (new state, JSON record)`.
- Centralize stat aggregation; emit one JSON record per optimizer step (not per env-step).

**Verifiable Rewards (automated exit 0/1).**

- Add a CPU-only unit test that performs an A2C update end-to-end on a toy env and asserts finiteness + parameter update.
- Add `--self-test` (no ROM/GPU required) and/or `--smoke` mode that exits `0` on success and `1` on invariant violation.

**The AI Judge (LLM review before human review).**

- Encode correctness via checks/tests so failures are machine-detectable (don’t rely on manual review).

### II. The Velocity of Tooling

**Latency is Technical Debt: full test suite ≤ 2 minutes.**

- Keep tests tiny and CPU-only; avoid ROM/GPU requirements for core correctness.
- Lazy import heavy deps (`torch`, `warp`) using `_require_torch()` pattern.

**Native Speed.**

- No Python per-env loops in training; use batched torch ops.
- Avoid per-step large allocations and large logs.

**Density & Concision.**

- Reuse existing helpers and patterns:
  - `PixelActorCriticCNN`, `logprob_from_logits`
  - `_atomic_save` pattern from `tools/rl_m5_train.py`
  - CUDA availability gating pattern.

**Code is Cheap, Specs are Precious.**

- Freeze the trainer CLI contract and JSONL schema in the script docstring.
- Ensure checkpoints contain everything needed to resume and reproduce.

### III. The Shifting Left

**Test pyramid inverted (cheapest checks first).**

1. Types/pyright: keep core math typed; avoid `Any` in the math path.
2. Unit tests: CPU-only TD(0) A2C update test.
3. Integration: optional/manual GPU smoke run (requires ROM/state/goal).
4. Golden/snapshot: optionally snapshot the JSONL schema/keys (prevent silent drift).

**Spec-driven development.**

- Write the exact TD(0) equations + detach rules (below) before coding.

### IV. The Immutable Runtime (Infrastructure & Deps)

**Easy and Hermetic-ish.**

- Use existing toolchain: `uv` + pinned RL deps (`torch`) already in `pyproject.toml`.
- No new dependencies for Workstream C.

**Supply Chain Minimalism.**

- Structured logging via stdlib `json` (JSONL), no extra libs.

**Reproducible Builds / Runs.**

- Checkpoints include: model, optimizer, config, counters, RNG states.
- Meta log includes versions (torch, warp if available).

### V. Observability & Self-Healing

**Structured Logs Only.**

- JSONL only; first line is `{"meta": ...}`; then step records.

**Crash-only Software.**

- Atomic checkpoint writes; safe resume via `--resume`.
- Handle `KeyboardInterrupt` by saving a final checkpoint and closing the env.

**Minimize tokens; track the “why”.**

- Silent on success except JSONL.
- On failure, emit a single JSON error record with last counters and a run id.

### VI. The Knowledge Graph (Documentation)

**Single Source of Truth.**

- Trainer usage and schema live in:
  - `tools/rl_m5_train_a2c.py` docstring (canonical)
  - `history/a2c-training.md` (updated to reference the real script name and usage)

**Living Documentation.**

- Add an import/availability test for the new core module to prevent drift.

---

## Detailed Execution Plan (Workstream C)

### 0) Lock Decisions (Before Coding)

- **Recurrence**: start with `stack_k=1` and *no recurrence* (per `history/a2c-training.md`); leave GRU as follow-on work.
- **Action codec default**: default to `pokemonred_puffer_v0` until Workstream A lands v1; still accept `--action-codec` override.
- **Update cadence**: implement `--update-every` grad accumulation; default `4`.
- **Counting**: use `--total-env-steps` (global transitions) rather than “updates”.

### 1) Implement the Functional Core (Pure A2C TD(0) math)

Add `src/gbxcule/rl/a2c.py`:

- `a2c_td0_losses(logits, actions, values, rewards, done, trunc, v_next, *, gamma, value_coef, entropy_coef) -> dict[str, Tensor]`
- Exact math (must match the spec in `history/a2c-training.md`):
  - `not_done = ~(done | trunc)`
  - `target = rewards + gamma * not_done.float() * v_next`  (v_next computed under `no_grad`)
  - `adv = target - values`
  - `loss_policy = -(logp(actions|logits) * adv.detach()).mean()`
  - `loss_value = value_coef * (target.detach() - values).pow(2).mean()`
  - `entropy = categorical_entropy(logits).mean()`
  - `loss_entropy = -entropy_coef * entropy`
  - `loss_total = loss_policy + loss_value + loss_entropy`
- Enforce shape/dtype validation similar to `src/gbxcule/rl/ppo.py`.
- Keep CPU-compatible so tests can run without CUDA/ROM.

### 2) Build the Trainer Script (`tools/rl_m5_train_a2c.py`)

Structure it like `tools/rl_m5_train.py`:

- `_require_torch()`, `_cuda_available()`, `_atomic_save()`
- `TrainConfig` dataclass
- `_parse_args()` and `main()`

**CLI knobs (minimum viable set):**

- Env:
  - `--rom`, `--state`, `--goal-dir`
  - `--num-envs`
  - `--frames-per-step` (default 24)
  - `--release-after-frames` (default 8)
  - `--stack-k` (default 1 for scale)
  - `--action-codec` (default v0 for now)
  - `--max-steps`
- Reward shaping (passed through to env):
  - `--step-cost`, `--alpha`, `--goal-bonus`, `--tau`, `--k-consecutive`
- Trainer:
  - `--lr`, `--gamma`, `--value-coef`, `--entropy-coef`, `--grad-clip`
  - `--update-every` (grad accumulation window)
  - `--total-env-steps`
  - `--checkpoint-every-opt-steps`
- Housekeeping:
  - `--seed`, `--output-dir`, `--resume`
  - `--smoke` and/or `--self-test`

**Training loop (streaming TD(0), no rollout buffer):**

- `obs = env.reset(seed=seed)`
- Maintain counters:
  - `env_steps` (increments by `num_envs` each env-step)
  - `opt_steps` (increments per optimizer step)
- Repeat until `env_steps >= total_env_steps`:
  1. forward: `logits, v = model(obs)`
  2. sample actions: `a ~ Categorical(softmax(logits))` (sample in int64, then cast to `int32`)
  3. env step: `next_obs, r, done, trunc, info = env.step(a_int32)`
  4. bootstrap: `_, v_next = model(next_obs)` under `no_grad`
  5. losses: `a2c_td0_losses(...)`
  6. grad accumulation:
     - Scale loss by `1/update_every` so effective gradient is stable.
     - Backward every env-step; call `optimizer.step()` every `update_every` env-steps.
  7. stats aggregation (accumulate within each optimizer-step window):
     - `reward_mean`, `done_rate`, `trunc_rate`, `reset_rate=(done|trunc).mean()`
     - optionally `dist_{p10,p50,p90}` if `info["dist"]` exists, but never log full vectors.
  8. On optimizer step:
     - `clip_grad_norm_` to `--grad-clip`
     - write one JSON record to `train_log.jsonl`
     - save atomic checkpoint on schedule
  9. `obs = next_obs`

**Resume / crash-only requirements:**

- Checkpoint contains:
  - `model` state, `optimizer` state
  - `config` (serialized)
  - `env_steps`, `opt_steps`
  - RNG states (`torch` CPU + CUDA) to support determinism in debugging
- `--resume` loads checkpoint and continues (append JSONL, don’t rewrite).
- On `KeyboardInterrupt`, write a final checkpoint and exit 0 (unless invariants violated).

**Output layout:**

- Default `output_dir`: `bench/runs/rl_m5_a2c/<timestamp>/`
  - `train_log.jsonl`
  - `checkpoint.pt`

### 3) Verifiable Rewards: Tests + Self-Test Mode

Add `tests/test_rl_a2c_smoke.py`:

- CPU-only toy env similar to `tests/test_rl_m5_smoke.py`
- Runs:
  - forward -> sample -> step -> bootstrap -> compute TD(0) target -> losses -> optimizer step
- Assertions:
  - `loss_total` is finite
  - at least one model parameter tensor changes after the optimizer step (strong signal trainer is wired correctly)

Add a script self-test:

- `uv run python tools/rl_m5_train_a2c.py --self-test`
  - Does not require ROM/state/goal-dir
  - Runs ~2 optimizer steps and prints one JSON summary
  - Exits `0` on pass, `1` on fail

### 4) Documentation Updates (Knowledge Graph)

- Update `history/a2c-training.md` “Workstream C” section:
  - confirm script name (`tools/rl_m5_train_a2c.py`)
  - include a canonical example command + mention `--total-env-steps` and `--update-every`
- In `tools/rl_m5_train_a2c.py` docstring:
  - freeze the JSONL schema (field names + meaning + types)

### 5) Acceptance Criteria (Definition of Done)

Workstream C is done when:

- `tools/rl_m5_train_a2c.py --help` works and does not eagerly import CUDA-heavy deps.
- `pytest -q` passes in CPU mode (`GBXCULE_SKIP_CUDA=1`), including the new A2C smoke test.
- `--self-test` exits `0` and prints a JSON summary.
- On a GPU box with user-provided ROM/state/goal-dir:
  - `train_log.jsonl` contains:
    - one `{"meta": ...}` record
    - N optimizer-step records with stable keys
  - `checkpoint.pt` is written atomically and `--resume` continues with monotonically increasing counters.

