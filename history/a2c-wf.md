# A2C Workstream F Plan — Observability (metrics + debugging artifacts)

## Scope (from `history/a2c-training.md`)

**Goal:** metrics that scale to 16K envs without drowning you.

**Requested tasks:**

1. Extend training log records with:
   - `done_rate`, `trunc_rate`, `reset_rate`
   - `dist_{p10,p50,p90}` (computed from `info["dist"]`)
   - action histogram (8 bins)
   - entropy + value stats
2. Add a lightweight plotter:
   - reward_mean, done_rate, dist_median, entropy vs step
3. Add a single-env GIF dump tool (optional but extremely helpful)

---

## Constitution-as-spec (`CONSTITUTION.md`) → concrete requirements for Workstream F

Treat each Constitution line as a *non-optional constraint* on observability. This section translates it into implementation requirements and acceptance criteria.

### I. The Doctrine of Correctness

- **Correctness by Construction**
  - Requirement: make “invalid logs” hard to produce.
  - Plan: define a single canonical log schema (typed dataclass / TypedDict) for `train_log.jsonl` records; add a validator tool that fails fast on missing keys, wrong types, NaNs/Infs, and non-monotonic counters.

- **Functional Core, Imperative Shell**
  - Requirement: metric computation must be pure/testable, not smeared across scripts.
  - Plan: put metric math in a pure module (e.g. `src/gbxcule/rl/metrics.py`) with functions like `compute_step_metrics(...) -> dict[str, float|list[int]]`. Training scripts do I/O only (write JSONL, save ckpt).

- **Unidirectional Data Flow**
  - Requirement: avoid ad-hoc globals; make metrics aggregation explicit.
  - Plan: implement a `MetricsAccumulator` (“reducer”) that consumes per-step signals (`reward/done/trunc/dist/actions/logits/values`) and emits a compact record at a chosen cadence (`log_every_steps` or per-optimizer-step).

- **Verifiable Rewards (exit 0/1)**
  - Requirement: every critical feedback loop has an automated check.
  - Plan: add CLI tools that return **0/1** and print JSON (machine-readable):
    - `validate_train_log.py` (schema + numeric sanity + monotonicity)
    - `plot_train_log.py` (generates plots; nonzero if no usable data)
    - `dump_policy_gif.py` (nonzero if it can’t load ckpt/env or produce frames)

- **The AI Judge**
  - Requirement: logs must be structured enough for an agent to debug without humans.
  - Plan: ensure every run has a `run_id`, every record has a `trace_id`, and failures emit receipts with enough state to reproduce (config + seed + ROM/state hashes + last frames).

### II. The Velocity of Tooling

- **Latency is Technical Debt (≤ 2 minutes full suite)**
  - Requirement: tests added for F must be tiny and fast.
  - Plan: unit tests only (pure metric math, JSON schema validation). Keep integration checks optional and skip when CUDA is unavailable.

- **Native Speed / Density & Concision**
  - Requirement: observability must not tank SPS at 8K–16K envs.
  - Plan: compute metrics on GPU and sync once per record; avoid per-env CPU transfers; avoid giant JSON payloads.

- **Code is Cheap, Specs are Precious**
  - Requirement: schema/spec first.
  - Plan: write the schema + validator before wiring producers.

### III. The Shifting Left

- **Inverted test pyramid / snapshot verification**
  - Requirement: validate cheapest layers first.
  - Plan:
    1) unit tests for metric functions (percentiles, histograms, rate math)
    2) golden/snapshot sample of one JSONL record (schema-stable)
    3) optional smoke run that writes a handful of records (only if CUDA available)

- **Spec-driven development**
  - Requirement: the log schema doc is the contract/prompt.
  - Plan: treat the schema doc as binding; breaking changes require updating doc + snapshots together.

### IV. The Immutable Runtime (Infrastructure & deps)

- **Easy & Hermetic-ish**
  - Requirement: don’t add runtime complexity.
  - Plan: use deps already present: `torch`, `PIL` (already used), `matplotlib` (already in repo deps). Avoid external encoders/ffmpeg.

- **Supply Chain Minimalism**
  - Requirement: avoid new logging/plotting frameworks.
  - Plan: JSONL + a minimal matplotlib plotter; reuse existing palette/image utilities.

- **Reproducible Builds**
  - Requirement: a run must be reproducible from artifacts.
  - Plan: meta record includes git SHA, ROM/state/goal template hashes, all CLI args, versions (`torch`, `warp`, CUDA device name), seed, action codec id, `stack_k`, `frames_per_step`.

### V. Observability & Self-Healing

- **Structured Logs Only**
  - Requirement: raw logs are JSON; humans get derived views.
  - Plan: `train_log.jsonl` is the single source of truth; human views are generated from it (`plot_train_log.py`).

- **Crash-Only Software**
  - Requirement: safe to restart/resume without corrupting artifacts.
  - Plan: atomic checkpoint writes; on resume, emit an explicit resume record to keep plots interpretable.

- **Minimize Tokens, Track the “Why”**
  - Requirement: silence on success; loud, structured failure receipts.
  - Plan:
    - normal mode: compact scalar metrics only (no per-env vectors)
    - failure mode: write a “receipt” dir keyed by `trace_id` containing last-frame PNG / last-frames NPY / last N actions + dist summary + config snapshot.

### VI. The Knowledge Graph (Documentation)

- **Single Source of Truth**
  - Requirement: schema/tools live in-repo.
  - Plan: add `docs/rl_observability.md` defining schema + usage for plot/gif/validator tools.

- **Living Documentation**
  - Requirement: docs stay correct.
  - Plan: validate a checked-in example record against the validator in tests.

---

## Deliverables (artifacts)

1. **Log schema v1** (`train_log.jsonl`)
   - line 1: `{"meta": {...}}`
   - subsequent: `{"trace_id": "...", "env_steps": ..., "opt_steps": ..., ...metrics...}`
   - optional errors: `{"trace_id": "...", "error": {...}}`

2. **Metrics computation + aggregation**
   - pure metrics module + reducer/accumulator that emits compact records.

3. **Plotter tool**
   - reads JSONL and writes `train.png` (multi-panel) with reward_mean, done_rate, dist_p50, entropy vs step.

4. **Single-env GIF dump tool**
   - loads a ckpt, runs 1 env, saves `rollout.gif` + `rollout.jsonl` (actions/reward/dist/done/trunc).

5. **Validator tool + tests**
   - `validate_train_log.py` + fast unit tests.

---

## Detailed implementation plan (step-by-step)

### F1) Define the logging contract (schema + field definitions)

**Goal:** you can’t “accidentally” log garbage.

- Decide canonical counters:
  - `env_steps`: total environment steps processed (N envs * steps)
  - `opt_steps`: number of optimizer steps taken
  - (PPO-only) keep `update` but also compute `env_steps` for comparability
- Define required fields per record (v1):
  - identifiers: `run_id`, `trace_id`
  - time/perf: `wall_time_s`, `sps` (env-steps/sec), optional `gpu_name`
  - env outcomes: `done_rate`, `trunc_rate`, `reset_rate`, `success_rate` (= done / reset, guarded)
  - reward stats: `reward_mean` (p10/p50/p90 optional later)
  - dist stats: `dist_p10`, `dist_p50`, `dist_p90`
  - policy stats: `entropy_mean`
  - value stats: `value_mean`, `value_std` (optional min/max)
  - action stats: `action_hist` (length = num_actions, counts or probs)
- Write validator rules:
  - required keys, allowed extras
  - monotonicity (`env_steps`, `opt_steps`)
  - numeric sanity (no NaN/Inf)

**Acceptance criteria**
- A sample record round-trips through the validator.
- Validator fails with a clear JSON error on schema violations.

### F2) Implement metric computation as a pure module

**Goal:** metrics are correct, fast, and testable.

- Inputs already available in training loops:
  - `actions` (int32 tensor)
  - `reward` (float tensor)
  - `done`, `trunc` (bool tensors)
  - `info["dist"]` and `info["reset_mask"]` from `PokeredPixelsGoalEnv.step()`
  - `logits`, `values` from the model
- Compute on GPU; sync once per emitted record:
  - rates: `mean(done)`, `mean(trunc)`, `mean(reset_mask)`
  - dist percentiles: p10/p50/p90 via `kthvalue`/`median` (no full vector logging)
  - action histogram: `torch.bincount(actions, minlength=num_actions)`
  - entropy: from logits (mean entropy)
  - value stats: mean/std

**Performance constraints**
- No `.cpu().tolist()` on length-16K tensors.
- One host sync per record (bundle scalar reads).

**Acceptance criteria**
- Unit tests cover percentiles/hist/rates on synthetic tensors.
- No measurable regression at small N in a quick smoke check.

### F3) Wire metrics into training scripts (producer side)

**Goal:** every run produces consistent, comparable observability.

- Extend `tools/rl_m5_train.py` (PPO-lite) record:
  - aggregate across the rollout rather than per env-step
  - `done/trunc/reset` counts accumulated across rollout steps
  - action histogram aggregated over rollout
  - entropy/value averaged over rollout (or from batch forward pass)
  - dist percentiles from last step (or a rolling statistic)

- For streaming A2C (Workstream C), log per optimizer step (or per grad-accum window):
  - reuse `MetricsAccumulator` to control cadence.

**Crash-only + resume requirements**
- On resume: emit a `{"resume": {...}}` record (or embed in meta) so plots don’t silently splice time-series.
- Keep atomic ckpt writes; link ckpt ↔ record with a `trace_id`.

**Acceptance criteria**
- `train_log.jsonl` contains meta + >0 records with the new fields for each trainer.

### F4) Add a lightweight plotter (human view generated from JSONL)

**Goal:** see reward_mean/done_rate/dist/entropy in seconds.

- Add a tool patterned after `bench/analysis/plot_scaling.py`:
  - inputs: `--log train_log.jsonl`, `--out train.png`, optional `--smoothing`
  - output: multi-panel plot:
    1) reward_mean
    2) done_rate + trunc_rate
    3) dist_p50 (and p10/p90 shaded if present)
    4) entropy_mean (and value_mean optional)
  - handle missing keys gracefully (skip subplot or warn via JSON).

**Acceptance criteria**
- Plotter succeeds on an existing `tools/rl_m5_train.py` run directory.

### F5) Add single-env GIF dump tool (debugging artifact)

**Goal:** when metrics lie, you can see agent behavior.

- Tool behavior:
  - load model checkpoint
  - run `PokeredPixelsGoalEnv` with `num_envs=1`
  - choose policy: `--greedy` or `--sample`
  - run `--steps N` or until done
- Outputs:
  - `rollout.gif` (palette frames)
  - `rollout.jsonl` with `step, action, reward, dist, done, trunc`
  - `meta.json` with config + hashes

**Acceptance criteria**
- Produces a GIF without external encoders (PIL-only).

### F6) Add verifiable checks (validator + tests + doc contract)

**Goal:** automated “this run is inspectable” gating.

- `validate_train_log.py`:
  - exit 0 on success; 1 on failure
  - prints JSON summary: record count, first/last steps, anomalies
- Tests:
  - unit tests for metrics functions
  - unit test for validator using a fixture log
- Docs:
  - `docs/rl_observability.md` defines schema + usage
  - include a tiny example JSONL snippet as a fixture and validate it in tests

**Acceptance criteria**
- `uv run pytest` stays comfortably under the 2-minute budget.

---

## Risks / pitfalls (guardrails)

- **Logging giant tensors:** never log full `dist` vectors; log only percentiles/mean.
- **Entropy collapse:** log entropy every record and plot it; detect collapse early.
- **False-positive goal detection:** use dist percentiles + done_rate + GIF tool + `replay_goal_template.py` receipts to triangulate.

---

## Optional beads breakdown (if you want issues)

- F1: Define `train_log.jsonl` schema + validator
- F2: Implement GPU metrics module + tests
- F3: Integrate metrics into PPO + A2C trainers
- F4: Add `plot_train_log` tool
- F5: Add single-env GIF dump tool
- F6: Docs + living-doc test

