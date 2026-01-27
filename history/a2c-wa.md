# Workstream A Plan (Breaking): Canonical NOOP Action Space

This plan is derived from:
- `history/a2c-training.md` (note: there is no `history/a2c-train.md` in this repo)
- `CONSTITUTION.md` (this is the spec; its requirements are embedded below)

## Spec (CONSTITUTION.md) Embedded Requirements (How This Plan Applies)

### I. The Doctrine of Correctness
- **Correctness by Construction:** Action semantics must be explicit and total. NOOP must be unambiguous in both PyBoy and Warp.
- **Functional Core, Imperative Shell:** Keep the action codec definitions pure (`src/gbxcule/core/action_codec.py`). Backends/tools own side effects (press/release scheduling, IO).
- **Unidirectional Data Flow:** `action_codec_id` is the single identifier that flows config -> backend -> env meta -> goal templates -> train/eval. No hidden remaps.
- **Verifiable Rewards:** We still keep a small automated check that fails if NOOP or indices are wrong (but we avoid adding big new test ROMs).

### II. The Velocity of Tooling
- **Latency is Technical Debt:** Validate via fast checks (unit tests and lightweight codegen assertions), not long training runs.
- **Density & Concision:** One canonical mapping; remove magic constants like “7 actions” from call sites.

### III. The Shifting Left
- Put the guardrails at the boundary (codec + warp joypad mapping) so downstream RL isn’t debugging input semantics.

### VI. The Knowledge Graph (Documentation)
- Breaking changes are allowed here, but they must be loud: update in-repo docs/configs so users don’t unknowingly run with old assumptions.

## Workstream A Goal (modified per request)
Make **NOOP part of the canonical action space** and **delete the old 7-action codec** (no dual support). Everything moves to **8 actions**.

Breaking-change consequences (accepted):
- Any goal templates captured with the previous 7-action mapping are invalid and must be recaptured.
- Old action traces/checkpoints are semantically invalid (indices changed); treat them as incompatible.

## Detailed Plan

### Phase 0 — Declare the New Canonical Contract (Spec-First)
1. **Single canonical codec id**
   - Keep exactly one registered codec id going forward: `pokemonred_puffer_v1`.
   - Delete `pokemonred_puffer_v0` entirely (no backwards compatibility).
2. **Canonical action order (8 actions)**
   - `("NOOP","A","B","START","UP","DOWN","LEFT","RIGHT")` with indices `[0..7]`.
3. **Kernel-facing id**
   - With only one codec, simplify kernel ids to: `KERNEL_CODEC_IDS["pokemonred_puffer_v1"] == 0`.
4. **Invariants to enforce**
   - PyBoy: `to_pyboy_button(0) == None`
   - Warp: NOOP yields `(dpad_mask, button_mask) == (0, 0)`

Deliverable: a short “contract comment” near the codec definition plus a repo-wide migration checklist.

### Phase 1 — Delete the Old Codec and Replace Defaults
Files:
- `src/gbxcule/core/action_codec.py`
- `src/gbxcule/backends/common.py` (defaults)

Steps:
1. Remove v0:
   - delete `POKERED_PUFFER_V0_ID`, `_POKERED_PUFFER_V0`, and the `_REGISTRY` entry.
2. Make v1 the only codec:
   - define `POKERED_PUFFER_V1_ID = "pokemonred_puffer_v1"`
   - register only v1 in `_REGISTRY`
   - set `DEFAULT_ACTION_CODEC_ID = POKERED_PUFFER_V1_ID`
3. Define the canonical mapping:
   - `_pyboy_buttons = (None,"a","b","start","up","down","left","right")`
   - `_dpad_masks = (0,0,0,0,DPAD_UP,DPAD_DOWN,DPAD_LEFT,DPAD_RIGHT)`
   - `_button_masks = (0,BUTTON_A,BUTTON_B,BUTTON_START,0,0,0,0)`
4. Update kernel ids:
   - `KERNEL_CODEC_IDS = {POKERED_PUFFER_V1_ID: 0}`

### Phase 2 — Update Warp Joypad Mapping to the New Indices
Risk: Warp’s mapping in `src/gbxcule/kernels/cpu_step_builder.py` is currently hard-coded for the 7-action indices. If not updated, NOOP (0) will behave like A.

Files:
- `src/gbxcule/kernels/cpu_step_builder.py`

Steps:
1. Update `action_dpad_mask` to match v1 indices:
   - `UP` == 4, `DOWN` == 5, `LEFT` == 6, `RIGHT` == 7
2. Update `action_button_mask` to match v1 indices:
   - `A` == 1, `B` == 2, `START` == 3
3. Keep the signature stable, but simplify logic:
   - keep the `codec_id` parameter if it avoids broader churn, but treat it as unused (single canonical mapping).

### Phase 3 — Repo-Wide Migration (Break It On Purpose)
Goal: remove all references and assumptions of:
- `pokemonred_puffer_v0`
- “7 actions” / `[0..6]` checks

Mechanics:
1. Replace codec id strings everywhere:
   - `rg -n \"pokemonred_puffer_v0\" -S .` then update to `pokemonred_puffer_v1`
2. Update action-range checks/constants:
   - `ACTION_MAX = 6` -> `ACTION_MAX = 7`
   - remove any `num_actions == 7` hard-fails
3. Update tests and toy models that assume 7 actions:
   - RL model/logit shape tests (`tests/test_rl_models.py`)
   - PPO helper tests (`tests/test_rl_ppo.py`)
   - RL smoke tests (`tests/test_rl_m5_smoke.py`)
   - goal template tests that assert codec id (`tests/test_rl_goal_template.py`)
4. Update docs/config defaults so the break is loud, not silent:
   - `README.md`, `Makefile` (e.g. `E4_ACTION_CODEC`), `configs/*.json`, and any tool docs.

### Phase 4 — Minimal Verification (No New Micro-ROM)
Per request: avoid adding new micro-ROMs. Keep checks small but meaningful.

1. **Codec unit tests (must-have)**
   - Update `tests/test_action_codec.py` to assert:
     - codec exists and `num_actions == 8`
     - NOOP maps to `None` and `(0, 0)`
     - all other actions map to expected masks
2. **Warp mapping guard (simple)**
   - Add a test that generates the cpu_step module into a temp cache dir (pattern already exists in `tests/test_cpu_step_builder.py`).
   - Read the generated `cpu_step_*.py` file and assert the mapping reflects v1 indices (textual checks):
     - `action == 1` -> `BUTTON_A`, `action == 2` -> `BUTTON_B`, `action == 3` -> `BUTTON_START`
     - `action == 4/5/6/7` -> DPAD masks

Verifiable reward:
- `uv run pytest -q`

### Phase 5 — Update RL Tooling to Use 8 Actions
Files:
- `tools/rl_m5_train.py`
- any RL scripts that assume 7 actions or clamp `[0..6]`

Steps:
1. Remove the “expected 7 actions” guard.
2. Ensure models are created with `num_actions=env.backend.num_actions` (now 8).
3. If any scripts expose `--action-codec`, either remove the flag entirely or restrict it to the single canonical id.

## Definition of Done (Workstream A)
- Only one action codec exists: `pokemonred_puffer_v1` (8 actions with NOOP at index 0).
- Repo contains no `pokemonred_puffer_v0` references and no 7-action assumptions.
- Warp joypad mapping matches the new indices (NOOP does not press A).
- A small automated check fails if the warp mapping is still the old one.

