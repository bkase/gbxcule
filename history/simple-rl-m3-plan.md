# Simple-RL Milestone 3 Plan — Goal Template Capture + Done Detection (Pixels-only)

This plan is for **Simple-RL Milestone 3** as described in `history/simple-rl.md` (“Goal template capture + done detection”).
It is **not** the repo’s separate “M3 (DGX) gates” milestone for CUDA correctness/scaling (see `history/m3-architecture.md` / `history/m3-workstreams.md`).

This is a **spec-first** plan per `CONSTITUTION.md`: lock the contract now, then implement with automated, deterministic gates.

---

## 0) Non-negotiables (locked)

From `history/simple-rl.md`:

1. **Everything on GPU** in the hot path: stepping + pixel extraction + goal distance + done detection.
2. **No RAM reads for the policy** (goal matching must be pixel-only).
3. **24 frames per step** (action frequency stays 24).
4. **Minimal + incrementally testable**: every slice has a fast deterministic gate.

User constraint (explicit):

- **Do not add a NOOP action.** Keep the current action space/codec **as-is** (`pokemonred_puffer_v0`, 7 actions).

Implications:

- “Waiting” in traces must be expressed using existing actions (e.g., a direction into a wall, or a button that is inert on that screen), not a new NOOP.

---

## 1) Spec: `CONSTITUTION.md` (verbatim, then applied)

### I. The Doctrine of Correctness

- **Correctness by Construction:** Make invalid states unrepresentable. If the types align, the logic should be sound.
- **Functional Core, Imperative Shell:** Business logic is pure and side-effect-free. IO and state mutations live at the edges.
- **Unidirectional Data Flow:** State management follows a strict Reducer pattern wherever possible. One way data flows: Action → Reducer → New State.
- **Verifiable Rewards:** In the era of AI, manual review is insufficient. Design feedback loops where agents receive automated, verifiable rewards (exit 0 or exit 1). If an agent writes code, a formal check (type, test, or proof) must verify it immediately.
- **The AI Judge:** LLM review before human review for concision, correctness, performance, and security.

### II. The Velocity of Tooling

- **Latency is Technical Debt:** Max tolerance: **2 minutes** for the full test suite.
- **Native Speed:** Prefer Rust over Node.js for infrastructure (e.g., ox over eslint). Build bespoke utilities when standard tools are too slow.
- **Density & Concision:** Minimize tokens. Prefer dense, expressive code. Embed DSLs when they reduce complexity. One canonical way to do each thing.
- **Code is Cheap, Specs are Precious:** Generate boilerplate; focus energy on specs and verification. Agents fill in implementations. If we generate 100x faster, we can throw away 100x as much.

### III. The Shifting Left

- **The Test Pyramid Inverted:** Push validation to the cheapest, fastest layer.

1. **Compiler/Types:** Catch it here first. (0ms cost)
2. **Unit/Property Tests:** Catch it here second. (ms cost)
3. **Integration Tests:** Keep these fast. (hundreds ms cost)
4. **Golden & Snapshot Verification:** Do not write assertions for complex outputs; approve snapshots. For UI, CLI output, or JSON structures, freeze the expected output. Agents detect diffs; humans only review intentional changes. (ms cost)
5. **Agentic E2E:** Agents simulate user behavior to verify the "shell." (seconds cost)
6. **Human Review:** Reserved for architectural intent. "Does it work?" is automated; humans review "should it work this way?"

- **Spec-Driven Development:** Write the spec before the code; it's the prompt. Ambiguous specs produce failing agents; this is the feedback loop.

### IV. The Immutable Runtime (Infrastructure & Deps)

- **Easy and Hermetic-ish:** Optimize for standard Codex/Claude cloud environments. Prefer **Rust stable**, **uv** (Python), and **bun** (Node). Pin versions to ensure reproducibility. Fall back to Nix/Docker only when standard tooling fails.
- **Supply Chain Minimalism:** Prefer copying a 50-line utility over adding a 50MB dependency. Dependencies are liabilities. (Don't reinvent good wheels.)
- **Reproducible Builds:** The artifact produced by CI must be bit-for-bit identical to the artifact produced locally.

### V. Observability & Self-Healing

- **Structured Logs Only:** Human-readable logs are generated from tools; raw logs are machine-readable (JSON). Agents need structured data to diagnose failures.
- **Crash-Only Software:** Systems are idempotent and can be restarted at any time. Fear corrupt state, not crashes.
- **Minimize Tokens, Track the "Why":** Silence on success. On failure: loud errors with Trace ID and state snapshot, enough for an agent to auto-generate a repro.

### VI. The Knowledge Graph (Documentation)

- **Single Source of Truth:** If it's not in the repo, it doesn't exist. Markdown, JSONL, beads_rust for in-repo task tracking.
- **Living Documentation:** Documentation is code. It is linted. It is tested. If the code changes and the docs don't, the build fails (e.g., doc-tests in Rust).

---

## 2) How this plan implements the spec (applied constraints)

### Correctness by construction

- Define a **versioned goal artifact** format with explicit shapes/dtypes and strict load-time validation.
- Make mismatches non-silent (fail fast if meta doesn’t match runtime config).

### Functional core, imperative shell

- “Core”: pure tensor functions for `dist` and the `done` reducer.
- “Shell”: capture/replay CLIs do file IO + backend stepping and call into the core.

### Unidirectional data flow (Reducer)

- Track goal matching state as explicit tensors:
  - `consec_match[N]` (int32)
  - (optional) `prev_dist[N]` for debugging/calibration
- Update rule is a reducer: `(prev_state, obs) -> (new_state, dist, done)`.

### Verifiable rewards (exit 0/1)

- Add a **single replay gate**: run a fixed start state + fixed `actions.jsonl` and assert `done=True` within the trace; exit 0 on pass, 1 on fail.

### 2-minute suite discipline

- Unit tests are CPU-only and fast.
- End-to-end replay tests are `pytest.skip` unless ROM/state/GPU are available; the “real” gate is a make/CLI command run on the right machine.

### Snapshot verification

- The `goal_template` file(s) are the snapshot.
- Changing a goal snapshot requires an explicit `--force` (no accidental overwrites).

### Observability

- Capture/replay emit machine-readable JSONL per step when `--log-jsonl` is enabled.
- On failure, write a minimal “receipt” bundle (dist curve + last few frames).

---

## 3) M3 Definition of Done (locked contract)

M3 is done when all of the following exist and pass:

1. A **goal template artifact** (pixels-only, downsampled 4-shade space), including strict metadata.
2. A **deterministic action trace** (`actions.jsonl`) that reaches the goal from a saved start `.state`.
3. A **GPU-only matcher** that computes `dist` and a `done` mask from pixels only.
4. A **replay gate** (one command) that replays the trace and returns exit 0 when `done` triggers reliably.

---

## 4) Interfaces + artifacts (make it impossible to be ambiguous)

### 4.1 Pixels + stack (policy-facing)

- Shade space: `uint8` values `0..3`
- Downsample: `H=72, W=80` (exact /2 from 160×144)
- Stack depth: `K=4`
- Tensor shapes:
  - frame: `uint8[N, 72, 80]`
  - stack: `uint8[N, 4, 72, 80]` (latest in a fixed position, documented)

### 4.2 Goal template file format

Canonical on disk:

- `goal_template.npy` (recommended: `uint8[72,80]` single-frame for M3)
- `goal_template.meta.json` (required)
- Optional debug:
  - `goal_template.png` (for humans only)

Meta must include (minimum):

- `schema_version` (start at 1)
- `created_at` (UTC ISO8601)
- `rom_path` (string) + `rom_sha256`
- `start_state_path` + `start_state_sha256`
- `actions_path` + `actions_sha256`
- `action_codec_id` (must be `pokemonred_puffer_v0`)
- `frames_per_step=24`, `release_after_frames=8`
- `downsample_h=72`, `downsample_w=80`, `stack_k=4`, `shade_levels=4`
- matcher config:
  - `dist_metric="l1_mean_norm"`
  - `tau` (float)
  - `k_consecutive` (int)
- `pipeline_version` (bumps if renderer/stack semantics change)

### 4.3 Action trace format (do not invent a new one)

Use the harness-compatible format (`bench/harness.py::load_actions_trace`):

- JSONL file
- each line is a **JSON list of ints**, one per env: e.g. for `N=1` each line is `[3]`.

---

## 5) Implementation slices (each with a deterministic gate)

### Slice A — Core matcher + reducer (pure, test-first)

**Goal:** given `frame_uint8` and `goal_uint8`, compute `dist` and update `done`.

Deliverables:

- `dist = mean(abs(frame - goal)) / 3.0` (float32/float16 on GPU)
- Reducer:
  - `consec = consec+1 if dist < tau else 0`
  - `done = consec >= k_consecutive`

Gates:

- Unit tests for:
  - shape/dtype validation
  - reducer behavior (threshold crossing, reset, K-consecutive semantics)

### Slice B — Goal artifact IO (strict validation)

**Goal:** load/save goal templates with metadata, atomically.

Deliverables:

- Save: write to temp dir + rename.
- Load: validate:
  - correct dtype/shape
  - meta matches runtime config (codec id, downsample dims, etc.)

Gates:

- Unit tests: invalid meta/shape fails fast with clear error.

### Slice C — Capture CLI (imperative shell)

**Goal:** produce `goal_template.npy` + meta by replaying a fixed trace from a fixed start state.

Deliverables:

- CLI that:
  1. creates backend (GPU path)
  2. loads ROM + start `.state`
  3. replays `actions.jsonl` for `T` steps
  4. captures final downsampled shade frame (the same pipeline used by training)
  5. writes artifacts

Gates:

- “Capture smoke” run produces a template and a human-viewable PNG.

### Slice D — Replay gate (exit 0/1, structured logs)

**Goal:** deterministic green/red check for M3.

Deliverables:

- Replay script:
  - loads goal + meta
  - replays the same start state + actions trace
  - computes `dist` + reducer entirely on GPU each step
  - returns:
    - exit 0 if any env hits done within the trace
    - exit 1 otherwise
  - optional `--log-jsonl` per-step logs
  - on failure, writes a small receipt bundle (dist curve + last frames)

Gates:

- Running the replay on the recorded trace passes repeatedly (no flakiness).

---

## 6) Calibration protocol (how we pick τ without vibes)

Add a calibration mode (can be part of replay):

- On a goal-reaching trace, record `dist_t` curve.
- Set defaults:
  - `k_consecutive = 2` (fade/transition robustness)
  - `tau = min(dist_t over last ~10 steps) + margin`
- Store chosen `tau`/`k` in `goal_template.meta.json`.

---

## 7) “No NOOP” strategy (explicit, because it changes trace design)

Because the action space is fixed (no NOOP), we standardize how traces represent waiting:

- Prefer a direction pressed **into a wall** when in a stable map state.
- If a screen is known to ignore a button (often **B**), use that as the “wait” action.
- Avoid using `A`/`START` as waits (they can open menus and create non-local effects).

M3 acceptance requires the trace to be robust (replay succeeds repeatedly), not “works once”.

---

## 8) Test plan (shift-left, keep suite fast)

### Always-on unit tests (CPU)

- Matcher math on toy tensors (no GPU).
- Reducer correctness for K-consecutive logic.
- Goal artifact validation failures (meta mismatch, wrong shapes).

### Optional integration test (skipped unless assets + CUDA exist)

- End-to-end replay of:
  - a local ROM path
  - a local start `.state`
  - a local `actions.jsonl`
  - a local `goal_template.npy`
- Skip cleanly if any prerequisite is missing.

Primary enforcement is the **replay gate command** on the machine that has the assets.

---

## 9) Risks + mitigations

### Risk A: renderer mismatch between capture and training

Mitigation:

- Capture must use the **same pixel pipeline** that training/replay uses (downsampled shade camera).
- Avoid mixing in PyBoy RGBA quantization for the goal snapshot unless training uses it too.

### Risk B: goal snapshot false positives

Mitigation:

- `k_consecutive > 1`
- Optionally allow multiple goal templates and require `min(dist_i) < tau` (still pixel-only).

### Risk C: “waiting” without NOOP introduces trace fragility

Mitigation:

- Choose start and route where inert inputs are well-defined (wall-hold or inert button).
- Validate trace stability by repeating replay multiple times and hashing frames/dist curves.

---

## 10) Suggested beads issue breakdown (optional)

If tracking in beads_rust (`br`):

1. M3-A: Goal matcher + reducer (pure) + unit tests
2. M3-B: Goal artifact schema + strict IO (meta + hashing + atomic writes)
3. M3-C: Capture CLI (replay trace → write goal snapshot)
4. M3-D: Replay gate CLI (exit 0/1 + optional receipt bundle)
5. M3-E: Calibration mode + default τ/K policy
6. M3-F: Minimal docs: “How to capture goal + how to replay gate”

