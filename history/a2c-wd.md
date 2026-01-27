# A2C Workstream D Plan — Eval Correctness

This is the detailed plan for **Workstream D (Eval correctness)**, derived from:

- `CONSTITUTION.md` (project spec; constraints below)
- `history/a2c-training.md` (Workstream D definition)

---

## Spec Constraints (from `CONSTITUTION.md`)

### I. The Doctrine of Correctness

- **Correctness by Construction:** Make invalid states unrepresentable. For eval, that means metric definitions are explicit and the CLI/API prevents invalid argument combinations (e.g., trajectory dumping with multi-env unless the format is defined).
- **Functional Core, Imperative Shell:** Put evaluation bookkeeping/aggregation in a pure core; keep IO/CLI wiring in the shell (`tools/`).
- **Unidirectional Data Flow:** Model evaluation bookkeeping as a reducer: `StepResult -> EvalState -> EvalState'`. Avoid “break on first done” control flow.
- **Verifiable Rewards:** Add automated checks that fail (exit 1 / test failure) if eval becomes misleading (premature termination, wrong denominators, mixing done/trunc semantics).
- **The AI Judge:** Treat eval output schema + tests as the gate; humans review intent, not correctness.

### II. The Velocity of Tooling

- **Latency is Technical Debt:** Keep eval correctness tests CPU-only and tiny so the full suite remains fast.
- **Native Speed:** Avoid adding heavy dependencies just for eval; reuse existing patterns.
- **Density & Concision:** One canonical eval implementation shared by `tools/rl_m5_eval.py` and training-time eval.
- **Code is Cheap, Specs are Precious:** Define metric semantics (spec) before implementation.

### III. The Shifting Left

- **Inverted Test Pyramid:** Validate via input checks/types, then unit tests for the eval reducer/core, then schema/snapshot tests, then (optional) CUDA smoke.
- **Spec-Driven Development:** “Eval semantics spec” is the prompt; code follows.

### IV. The Immutable Runtime (Infrastructure & Deps)

- **Easy and Hermetic-ish:** Keep torch as an optional import (dynamic import like existing RL modules).
- **Supply Chain Minimalism:** Don’t introduce new deps unless strictly necessary.
- **Reproducible Builds:** Deterministic eval given `(checkpoint, goal_dir, seed)`; stable JSON output.

### V. Observability & Self-Healing

- **Structured Logs Only:** Emit JSON summary to stdout; optional JSONL for per-episode/trajectory dumps.
- **Crash-Only Software:** Errors include enough config context to repro; clean exits.
- **Minimize Tokens, Track the “Why”:** Silent on success except the JSON; loud structured errors on failure.

### VI. The Knowledge Graph (Documentation)

- **Single Source of Truth:** Eval metric definitions live in-repo and are referenced by tools/tests.
- **Living Documentation:** Add tests that prevent doc/code drift.

---

## Workstream D Scope (from `history/a2c-training.md`)

**Goal:** correct evaluation metrics, ideally with `num_envs=1` eval.

Tasks:

1. Create/modify eval script to run `num_envs=1`
2. Report:
   - `success_rate`
   - `median_steps_to_goal`
   - `mean_return`
3. Optionally dump a trajectory (`actions + dist`) for a single run for debugging

---

## Detailed Plan

### D0 — Lock eval semantics (spec-first)

- Write a short “Eval Semantics” spec (docstring + optionally a `docs/` note) defining:
  - **Episode end:** `terminated = done | trunc` from env step (env is autoreset).
  - **Success:** `done == True` at termination step.
  - **Failure:** `trunc == True` with `done == False`.
  - **`success_rate`:** `successes / episodes_requested`.
  - **`median_steps_to_goal`:** median steps over **successful** episodes only; define behavior when there are zero successes (`0` vs `null`) and stick to it.
  - **`mean_return`:** mean episodic return over **all** evaluated episodes.
- Decide and document the median convention for even counts (e.g., lower median vs average).

### D1 — Audit and name the current failure modes

- Confirm and capture the two core correctness hazards in current code:
  - `tools/rl_m5_eval.py` (and the copied `_eval_greedy` in training) **breaks on `any(done|trunc)`**, biasing metrics toward “time-to-first-finish”.
  - `tools/rl_m5_train.py` currently evaluates using the **same env instance** as training; `env.reset()` during eval can desynchronize `obs` used for training.
- Turn these into regression tests/acceptance checks (see D5).

**D1 Findings (current state):**

- **Premature termination in eval:** `_eval_greedy` breaks when *any* env finishes, so metrics reflect “first env to finish,” not full episode accounting.
- **Training desync risk:** `tools/rl_m5_train.py` reuses the training env for eval, and calls `env.reset()` inside eval, which invalidates the training `obs` buffer.
- **Acceptance checks to add (D5):**
  - Eval collects exactly `episodes_requested`, independent of individual env termination timing.
  - Success rate denominator is `episodes_requested`.
  - Training eval does not mutate training env or `obs`.

### D2 — Implement a shared eval core (reducer-style)

- Create a reusable module (e.g., `src/gbxcule/rl/eval.py`) that:
  - Accepts `(env, policy_fn, episodes, *, max_steps_override?, trajectory_dump?)`.
  - Tracks per-env accumulators (`ep_return`, `ep_steps`) and a global `episodes_collected`.
  - On each step:
    - compute actions via `policy_fn(obs)`
    - call `env.step(actions)`
    - update accumulators
    - for each env with `terminated == True`, finalize one episode record (success, steps, return, optional dist/action trace), then reset that env’s accumulators (env already autoresets).
  - Stops exactly when `episodes_collected == episodes_requested`.
- Keep the module import torch-free at import time (dynamic import only inside code paths that require torch).

### D3 — Fix `tools/rl_m5_eval.py` to be correct by default

- Make correctness the default:
  - Default `--num-envs=1` (recommended for now).
  - Add `--seed` and pass to `env.reset(seed=...)`.
  - Add `--action-codec` passthrough (important once goal templates encode codec metadata).
  - Run model under `torch.inference_mode()` and `model.eval()`.
- Output:
  - Print one JSON summary line with the spec’d fields.
  - Optional `--dump-trajectory <path>` writing JSONL of `{t, action, reward, done, trunc, dist}`; either restrict to `num_envs==1` or define a multi-env format and validate args.

### D4 — Make training-time eval correct and non-invasive (`tools/rl_m5_train.py`)

- Replace the inline `_eval_greedy(env, model, ...)` with:
  - A **separate eval env instance** (`num_envs=1`, same config) so training env + `obs` are never reset by eval.
  - The shared eval core from D2.
- Ensure the train loop restores `model.train()` after eval and does not mutate training `obs` via eval.

### D5 — Add fast tests that catch “lying eval”

- Add CPU-only tests using a toy autoreset env to assert:
  - With `num_envs > 1`, evaluation collects exactly `episodes_requested` and does not stop on the first termination.
  - `success_rate` denominator is `episodes_requested`.
  - `median_steps_to_goal` excludes trunc-only episodes (per D0 spec).
  - Trajectory dump validation rejects invalid arg combos (if restricted).
- Optional: a tiny snapshot/schema test that asserts the JSON summary keys to prevent drift.

### D6 — Observability polish (still within D)

- Add *cheap* debug-friendly summary fields (no giant tensors), e.g.:
  - `steps_p50_success`, `return_mean_success`, `return_mean_fail`, `dist_at_end_p50`
- Ensure CUDA-unavailable paths remain structured JSON (e.g., `{"skipped": "..."}`) and exit 0 (current behavior).

---

## Acceptance Criteria

- `tools/rl_m5_eval.py` produces correct metrics for autoreset envs and does not depend on “first env to finish”.
- Enabling eval in `tools/rl_m5_train.py` no longer corrupts training state (training `obs` stays in sync with the training env).
- Metric definitions are explicit (D0) and enforced by fast tests (D5).
- Tests remain CPU-only and small enough to preserve the constitution’s fast-suite constraint.
