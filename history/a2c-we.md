# A2C Workstream E — Stage Goals for “It’s time to go” (Pokémon Red)

## End Goal (Updated)

Train a pixels-only agent to:

1. Return to the player’s house
2. Go upstairs
3. Press **A** on the video game system
4. Reach the final chat box where the screen says **“It’s time to go”** (must match visually / exactly).

This workstream defines the **goal-template pipeline** and **verification gates** needed to support that end goal at scale (8K–16K envs) with streaming A2C.

---

## Spec: `CONSTITUTION.md` (Requirements Applied to Workstream E)

### I. The Doctrine of Correctness

- **Correctness by Construction:** Make goal artifacts hard to misuse.
  - Enforce strict load-time validation via `GoalTemplateMeta` (`src/gbxcule/rl/goal_template.py`).
  - Introduce a single stage manifest (single source of truth) so training/eval never rely on “magic paths”.
- **Functional Core, Imperative Shell:**
  - Core logic: pure transforms/validation (meta checks, distance checks, manifest parsing).
  - Shell: scripts that load ROM/state, render, save artifacts, run replay gates.
- **Unidirectional Data Flow:**
  - `StageSpec` → `StartState` (+ optional `ActionsTrace`) → `GoalTemplate + Meta` → `Verification Report`.
- **Verifiable Rewards (exit 0/1):**
  - Each stage must have an automated verification that returns `0` on pass and `1` on failure.
- **AI Judge:**
  - Prefer agent/LLM review of diffs/metadata consistency before human review.

### II. The Velocity of Tooling

- **Full test suite ≤ 2 minutes:**
  - Do **not** put GPU/PyBoy-heavy steps under `make test`.
  - Stage verification runs as a separate command/target (GPU machine) and remains quick (seconds).
- **Supply chain minimalism / density:**
  - Reuse existing modules/CLIs; avoid new heavy deps.
  - One canonical way to create goal templates and verify them.

### III. The Shifting Left

- **Validate early and cheaply:**
  - CPU-fast checks for meta/schema/manifest + file presence.
- **Snapshot verification:**
  - `goal_template.npy` is the snapshot; changes require explicit `--force` and human intent review via generated PNGs.
- **Agentic E2E:**
  - Replay traces through the actual env and ensure `done` triggers reliably.

### IV. The Immutable Runtime

- Use existing repo tooling (`uv`, pinned deps).
- Reproducible artifacts:
  - meta includes SHA256 of ROM/state/actions (already supported).

### V. Observability & Self-Healing

- **Structured logs:** JSON summaries per stage; JSONL for any per-step curves.
- **Failure receipts:** dump last frame + dist curve + repro command on failure.

### VI. The Knowledge Graph

- Everything is committed in-repo:
  - stage directories, `rl_stages.json` manifest, docs, traces, images.

---

## Inputs You Already Have (Saved States)

You have (names/paths to be provided):

- `S1`: start state (in Professor’s lab after Pokémon is chosen)
- `S2`: state for exiting Professor Oak’s place
- `S3`: state right before the player enters his house again
- `S4`: state right after he enters the house
- `S5`: state when the screen says “it’s time to go”

These are sufficient to define **stable stage boundaries** and to extract **exact goal screens** from states.

---

## Recommended Stage Decomposition (5 stages)

Rationale: isolate visually and behaviorally distinct sub-goals so distance shaping stays well-behaved and the final interaction is not brittle.

### Stage 1 — Exit Oak’s lab

- **Start:** `S1` (lab after Pokémon chosen)
- **Goal:** “outside / exiting Oak’s place” (from `S2`)
- **Purpose:** teach leaving the lab environment.

### Stage 2 — Return home (outside)

- **Start:** `S2`
- **Goal:** “right before entering house” (from `S3`)
- **Purpose:** teach navigation back to home door.

### Stage 3 — Enter house

- **Start:** `S3`
- **Goal:** “just inside house” (from `S4`)
- **Purpose:** isolate door transition and interior start pose.

### Stage 4 — Upstairs aligned at the video game system (derived state)

- **Start:** `S4`
- **Goal:** **NEW derived state**: “upstairs, standing at the video game system in interact position”
- **Purpose:** make the final chat-box stage short and stable.

How to get this derived goal state:
- Load `S4` and take a short deterministic action script to go upstairs and align at the system.
- Save that end state as `S4b` (committed).

### Stage 5 — Final chat box (“It’s time to go”) (exact)

- **Start:** `S4b` (upstairs aligned at system)
- **Goal:** `S5` (exact “it’s time to go” screen)
- **Purpose:** teach the “press A + advance text” micro-sequence.

---

## Optional Minimum Alternative (3 stages)

If you want fewer stages, the minimum that usually still works:

1. `S1` → `S2` (exit Oak)
2. `S2` → `S4` (get home and enter house)
3. `S4` → `S5` (do upstairs + system + text)

This is higher risk: Stage 3 becomes long/brittle and shaping gets noisier.

---

## Frozen “Stage ABI” (must match training)

All captured goal templates and verification must agree on:

- `action_codec_id`: `pokemonred_puffer_v1` (includes NOOP; required for stable “wait”)
- `frames_per_step`: `24`
- `release_after_frames`: `8`
- `stack_k`: `1` (scale mode)
- `downsample_h/w`: `72/80`
- `shade_levels`: `4`
- dist metric: normalized mean L1 (`compute_dist_l1(...)/3`)

Any mismatch should fail fast during template load (`load_goal_template(...)`) and during manifest validation.

---

## Artifact Layout (one directory per stage)

Proposed canonical layout:

- `states/rl_stage1_exit_oak/`
- `states/rl_stage2_return_home/`
- `states/rl_stage3_enter_house/`
- `states/rl_stage4_at_snes/`
- `states/rl_stage5_time_to_go/`

Each stage dir contains:

- `start.state` — start snapshot for that stage
- `goal_source.state` — the saved state that defines the *exact* goal screen (for provenance)
- `goal_template.npy` — `uint8[72,80]` (or `uint8[1,72,80]` if you standardize to 3D; prefer 2D for `stack_k=1`)
- `goal_template.meta.json` — strict metadata (includes SHA256s)
- `start_scaled.png` — visual render of start state (for intent review)
- `goal_template_scaled.png` — visual render of goal template (for intent review)
- (optional) `positive_actions.jsonl` — a deterministic trace that reaches goal from `start.state`
- (optional) `negative_actions.jsonl` — a trace that should *not* reach goal (false-positive guard)
- `README.md` — what the stage is + exact repro commands

---

## Stage Manifest (Single Source of Truth)

Add `states/rl_stages.json` with:

- global ABI block (codec/fps/release/stack/downsample/shades)
- an ordered list of stages:
  - `id`, `name`, `desc`
  - `start_state`, `goal_dir` (stage dir), `goal_source_state`
  - recommended `max_steps` for training (per-stage)

All verification tooling reads this manifest.

---

## Extraction Pipeline (State → Template + Visuals)

Goal: extract the target screen **exactly** from a saved state (no replay required just to capture pixels).

Canonical capture step per stage:

1. Load ROM + `goal_source.state` in a Warp backend (CPU is fine for capture).
2. Render downsampled pixels (`uint8[72,80]`).
3. Save:
   - `goal_template.npy`
   - `goal_template.meta.json` (ROM/state SHA256, ABI fields; `tau/k` initially placeholder)
   - `goal_template_scaled.png` (palette-mapped for humans)
4. Also render `start.state` → `start_scaled.png` for visual context.

Notes:
- Prefer capturing **2D** templates (`[72,80]`) when `stack_k=1`. The goal matcher already supports comparing a stacked frame to a 2D goal via “last frame”.

---

## Verification (Must-Have Gates)

All gates should be `exit 0` on pass, `exit 1` on fail, and produce machine-readable output (JSON/JSONL).

### Gate A — Manifest + meta consistency (CPU-fast)

For each stage:
- required files exist
- `goal_template.meta.json` matches the manifest ABI (codec/fps/release/stack/downsample/shades)
- template dtype/shape correct

### Gate B — Goal matches itself (template sanity)

For each stage:
- Load `goal_source.state`, render frame, compute `dist(frame, goal_template)`:
  - must be ~0 (or below an extremely small epsilon).

### Gate C — Start does not match goal (false-positive baseline)

For each stage:
- Load `start.state`, render frame, compute dist:
  - must be **well above** `tau`.

### Gate D — Behavioral replay (recommended)

If `positive_actions.jsonl` exists for a stage:
- Replay via GPU env stepping and verify goal triggers (`exit 0`).
- Replay a negative trace (or random actions) and verify goal does **not** trigger (`exit 1`).

Use existing verifier logic (`src/gbxcule/rl/replay_goal_template.py`) as the canonical exit-code gate.

---

## Calibrating `tau` and `k_consecutive`

Per stage:

1. Run a positive replay and record the distance curve near the end.
2. Choose:
   - `tau = tail_min + margin` (explicit margin, e.g. `+0.01`)
   - `k_consecutive = 4` as a starting default (reduce only if distance flickers)
3. Store `tau` and `k_consecutive` in `goal_template.meta.json`.
4. Re-run Gate B/C/D and require pass.

---

## Visual Review Requirement (Human Intent Check)

Before considering a stage “done”, visually confirm:

- `start_scaled.png` corresponds to the intended start screen.
- `goal_template_scaled.png` corresponds to the intended goal screen.
- For Stage 5, confirm the chat box is **exactly** the desired “It’s time to go” screen.

This is the required human review step; everything else is automated.

---

## Open Inputs Needed (to finalize this workstream)

Please provide the exact file paths (or filenames under `states/`) for:

- `S1` (Oak’s lab after Pokémon chosen)
- `S2` (exiting Oak’s place)
- `S3` (before entering house)
- `S4` (after entering house)
- `S5` (“it’s time to go”)

Once those are known, the stage directories + manifest can be concretely authored and the capture/verify commands can be written verbatim in each stage `README.md`.

