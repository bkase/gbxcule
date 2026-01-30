# Dreamer v3 - Milestone M3 Plan (RSSM + Scan Loop + AMP Stability)

Date: 2026-01-30

Source context:
- `history/dreamer-plan.md` (Master Plan v2: M3 scope, invariants, gates)
- `history/dreamer-gotchas.md` (learnable init, Hafner init, is_first reset, AMP stability)
- `history/dreamer-m2.md` (math/distribution conventions that RSSM depends on)
- `third_party/sheeprl/` (reference Dreamer v3 implementation; CPU-focused but authoritative)
- Repo reality check: Dreamer v3 scaffolding exists under `src/gbxcule/rl/dreamer_v3/`, but RSSM is not implemented yet.

This document is the **spec + execution plan** for Dreamer v3 **Milestone M3**:
**RSSM core + time-major scan loop with is_first reset masking + GRU FP32 internals**.
If behavior changes, update this plan and tests together.

---

## Objective (M3)

Deliver a correct, stable, and testable **RSSM** implementation that:
- matches sheeprl semantics for reset, unimix, and initial state
- supports **time-major** `[T, B, ...]` scan with mid-sequence resets
- runs stably under mixed precision by forcing **FP32 internal GRU math**
- provides **fixtures + tests** that lock behavior against the reference

This milestone is CPU-first and deterministic; no CUDA-only dependencies.

### Non-goals (explicitly out of scope)

- World model losses or reconstruction heads (M4)
- Behavior learning / imagination rollouts (M5)
- Async engine integration (M7)
- GPU ingestion / zero-copy replay (M6)
- Full decoupled RSSM training path (we can stub config but not implement unless needed)

---

## Spec: RSSM contracts and behavior

### 1) Latent state layout (time-major)

- Deterministic state: `h_t` (recurrent state), shape `[T, B, H]`
- Stochastic state: `z_t`, categorical one-hot with shape `[T, B, S, D]`
  - `S = stochastic_size`, `D = discrete_size`
- Flattened stochastic for concatenation: `z_flat = z.reshape(..., S * D)`
- Feature vector for heads: `feat = concat(z_flat, h_t)`

All internal sequences are **time-major** `[T, B, ...]`.
Any `[B, T, ...]` conversions must be explicit and tested.

### 2) Reset semantics (`is_first`)

At each time step `t`, if `is_first[t] == 1`:
- **Mask action**: `action = (1 - is_first) * action`
- **Reset recurrent state** to initial: `h = (1 - is_first) * h + is_first * h0`
- **Reset posterior seed** to initial posterior: `z = (1 - is_first) * z + is_first * z0`

This mirrors sheeprl behavior in `RSSM.dynamic`.

**Training-time shift:** sheeprl forces `is_first[0] = 1` for every sampled sequence.
We keep true `is_first` in replay; the scan function should optionally enforce this
(e.g., `force_first_reset=True`) for training parity.

### 3) Action alignment

Replay stores actions aligned with `obs[t]`. Dreamer dynamics expect the **previous action**.
For sequences, use:

```
shifted_actions = concat(zero_action, actions[:-1])
```

This is done outside RSSM, but RSSM utilities should expose a helper to avoid drift.

### 4) Learnable initial state (gotcha)

Sheeprl uses a **learnable initial recurrent state**:
- Parameter `initial_recurrent_state` shape `[H]`
- Use `tanh` transform before expanding to batch: `h0 = tanh(param)`
- Initial posterior `z0` is computed via **transition** from `h0` with `sample_state=False`.

If config disables learnable init, use zeros (still passed through tanh for parity).

### 5) Unimix (categorical smoothing)

Before sampling:
- `probs = softmax(logits)`
- `probs = (1 - unimix) * probs + unimix * uniform`
- `logits = probs_to_logits(probs)`

Apply unimix for both **transition** (prior) and **representation** (posterior) logits.

### 6) Stochastic state sampling

- Use `OneHotCategoricalStraightThrough` (ST) for sampling.
- `sample=False` returns `dist.mode` (deterministic; required for fixtures).
- `sample=True` uses `rsample()` for the straight-through gradient.
- For tests needing determinism, use `sample=False` or a controlled generator.

### 7) GRU precision stability (mandatory)

Even under `torch.autocast`, **all GRU internal math must be FP32**:
- Cast input and hidden state to float32 for gate computation
- Optionally keep recurrent state stored as float32 in the RSSM
- Cast outputs back to model dtype at the boundary

This prevents AMP instabilities observed in sheeprl/dreamerv3.

### 8) Hafner initialization (gotcha)

Implement sheeprl/DreamerV3 initialization:
- `init_weights` for Linear/Conv: truncated normal with scale based on fan-in/out
- `uniform_init_weights(scale)` for specific final layers
- Initialize LayerNorm weights to 1, bias to 0

Apply `init_weights` to recurrent, transition, and representation models.
If `hafner_initialization` is true, apply `uniform_init_weights(1.0)` to the
final layers of transition and representation models (per sheeprl).

---

## Module layout (M3 additions)

Add under `src/gbxcule/rl/dreamer_v3/`:

- `rssm.py`
  - `LayerNormGRUCell`
  - `RecurrentModel`
  - `TransitionModel` / `RepresentationModel`
  - `RSSM` core class (`dynamic`, `imagination`, `scan`, `get_initial_states`)
- `mlp.py` (or `layers.py`)
  - minimal MLP builder matching sheeprl order: Linear -> LayerNorm -> activation
- `init.py` (or `utils.py`)
  - `init_weights`, `uniform_init_weights`
  - `probs_to_logits`, `onehot_st` helper

Update `__init__.py` exports as needed for tests.

---

## Config additions (dreamer_v3/config.py)

Add and validate RSSM-specific configs to match sheeprl defaults:

- `world_model.recurrent_model.recurrent_state_size` (default 4096)
- `world_model.recurrent_model.dense_units` (default 1024)
- `world_model.recurrent_model.layer_norm` (`LayerNorm`, eps=1e-3)
- `world_model.transition_model.hidden_size` (default 1024)
- `world_model.representation_model.hidden_size` (default 1024)
- `algo.dense_units`, `algo.mlp_layers`, `algo.dense_act`
- `algo.hafner_initialization` already exists; ensure RSSM honors it
- `precision.rnn_dtype` already exists; RSSM must obey it

Validation rules:
- sizes must be positive
- `unimix` in [0, 1]
- `rnn_dtype` must be float32

---

## Reference alignment (sheeprl specifics to mirror)

From `third_party/sheeprl/sheeprl/algos/dreamer_v3/agent.py`:
- `LayerNormGRUCell` uses concatenated `[h, x]` -> Linear(3H) -> LayerNorm -> gates
- Gate update uses `update = sigmoid(update - 1)` (Hafner trick)
- `action = (1 - is_first) * action`
- `h = (1 - is_first) * h + is_first * h0`
- `z = (1 - is_first) * z + is_first * z0`
- `get_initial_states()` uses `tanh(initial_state)` and transition with `sample_state=False`
- Unimix uses probability mixing then `probs_to_logits`

We should match these exactly for parity fixtures.

---

## Fixtures (Golden Bridge)

Extend `tools/dreamer_v3/golden_bridge.py` to emit small RSSM fixtures using
`third_party/sheeprl` as the reference.

### Fixture set (minimal but sufficient)

1) **Single-step dynamic fixture**
   - Inputs: `h_prev`, `z_prev`, `action`, `embedded_obs`, `is_first`
   - Outputs: `h`, `prior_logits`, `posterior_logits`, `z_prior`, `z_post`

2) **Short scan with mid-sequence reset**
   - `T=3`, `B=2`, `is_first = [1, 0, 1]` on one env
   - Ensure reset happens in the middle of a window

3) **Initial state fixture**
   - `initial_recurrent_state` param values
   - Expected `h0` and `z0` (mode from transition)

Store fixtures under `tests/fixtures/dreamer_v3/` (JSON or NPY) with float32.

---

## Test plan (CPU, deterministic)

Create `tests/rl_dreamer/test_rssm.py` and cover:

1) **Shape + dtype**
   - `scan` outputs `[T, B, ...]` with correct dtypes

2) **is_first reset masking**
   - When `is_first=1`, `action` zeroed, `h` and `z` reset to initial

3) **Learnable initial state**
   - `get_initial_states()` uses `tanh(param)` and matches fixture

4) **Unimix behavior**
   - Verify logits change when `unimix > 0`; probabilities still sum to 1

5) **AMP stability**
   - Run RSSM under `torch.autocast` (CPU or CUDA if available)
   - Assert outputs finite and close to FP32 reference within tolerance

6) **Fixture parity**
   - Compare dynamic step and scan outputs to Golden Bridge fixtures

7) **Deterministic sampling mode**
   - `sample=False` yields deterministic outputs for fixtures

Gate: `pytest -q tests/rl_dreamer/test_rssm.py` passes on CPU.

---

## Step-by-step execution plan

1) **Scaffold M3 modules**
   - Add `rssm.py`, `mlp.py`, `init.py` under `src/gbxcule/rl/dreamer_v3/`.

2) **Implement LayerNormGRUCell (FP32 internals)**
   - Mirror sheeprl logic and gate equations.
   - Cast internal math to float32 and cast outputs back to model dtype.

3) **Implement Recurrent/Transition/Representation models**
   - Use MLP builder consistent with sheeprl (LayerNorm after projection).
   - Apply Hafner initialization (init + uniform init for final layers).

4) **Implement RSSM core**
   - `get_initial_states()`
   - `dynamic()` one-step with reset masking
   - `imagination()` prior-only step
   - `scan()` time-major loop with optional `force_first_reset` and action shift helper.

5) **Golden Bridge fixtures**
   - Extend `tools/dreamer_v3/golden_bridge.py` to dump RSSM fixtures from sheeprl.

6) **Tests**
   - Add deterministic unit tests and parity tests using fixtures.

7) **Gate**
   - Run `pytest -q tests/rl_dreamer/test_rssm.py` (CPU) and ensure stability.

---

## Definition of Done (M3 gate)

- RSSM core implemented with time-major scan and is_first reset masking.
- GRU internal math stable under autocast (FP32 policy enforced).
- Hafner initialization applied to RSSM models.
- Golden Bridge fixtures for RSSM present and consumed by tests.
- All M3 tests pass on CPU deterministically.
