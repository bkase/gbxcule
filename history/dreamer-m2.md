# Dreamer v3 - Milestone M2 Plan (Math + Distributions + ReturnEMA)

Date: 2026-01-30

Source context:
- `history/dreamer-plan.md` (Master Plan v2; M2 scope and gates)
- `history/dreamer-gotchas.md` (TwoHot bin alignment + symlog usage constraints)
- Repo reality check: no Dreamer v3 module exists yet under `src/gbxcule/rl/`

This document is the **spec + execution plan** for Dreamer v3 **Milestone M2**:
**Math & distributions primitives (symlog-twohot) + ReturnEMA percentile normalizer**.
If behavior changes, update this document and the tests together.

---

## Objective (M2)

Deliver the math primitives and distribution building blocks required by Dreamer v3:

- `symlog` / `symexp` primitives
- two-hot bucketization in **symlog space**
- `SymlogTwoHot` distribution with correct `log_prob`, `mean`, and `mode`
- a robust **ReturnEMA** percentile-based normalizer (p05/p95 EMA)
- Golden Bridge fixtures + tests that lock the semantics

This milestone must be fully CPU deterministic and independent of CUDA.

### Non-goals (explicitly out of scope for M2)

- RSSM, world model, or behavior learning logic
- Any replay or engine integration
- CUDA-specific code paths or performance optimization

---

## Spec: Contracts and correctness rules

### 1) Symlog / Symexp

- Definitions (must be exact):
  - `symlog(x) = sign(x) * log(1 + |x|)`
  - `symexp(y) = sign(y) * (exp(|y|) - 1)`
- These are monotone, inverse-ish transforms with bounded roundtrip error.
- Implementation must be vectorized, stable at `x=0`, and **float32** by default.

### 2) TwoHot in symlog space

- TwoHot operates on **symlog values**, not raw values:
  - `y = symlog(x)`
  - `bins` are **linearly spaced in y-space**
- Given `y` and ordered `bins`, TwoHot assigns weight to the adjacent pair:
  - lower index `i` and upper index `i+1`
  - `w = (y - bins[i]) / (bins[i+1]-bins[i])`
  - weights: `(1-w)` at `i`, `w` at `i+1`
  - clamp to ends if `y` is outside bin range

**Critical gotcha (from `history/dreamer-gotchas.md`):**
Do **not** rely on dynamic `torch.linspace` at test time for bin generation.
We must serialize the exact reference `bins` in fixtures and compare against those.
(If runtime bin generation is required, it must match fixture values bitwise.)

### 3) SymlogTwoHot Distribution

- Input: logits over `num_bins` (shape `[..., K]`)
- `log_prob(x)`:
  - compute `y = symlog(x)`
  - compute twohot weights for `y`
  - log-prob is dot(twohot, log_softmax(logits))
- `mean`/`mode`:
  - expectation in symlog space: `y_mean = sum(probs * bins)`
  - convert back via `symexp(y_mean)`
  - `mode` uses argmax bins and `symexp`

### 4) ReturnEMA (percentile normalizer)

- Tracks EMA of low/high percentiles (p05/p95) of returns/lambda targets.
- Outputs `(offset, invscale)` used as:
  - `x_norm = (x - offset) * invscale`
- Default percentiles: 5% and 95% (fixed unless explicitly changed).
- Scale logic:
  - `range = high - low`
  - `scale = max(max_range, range)`
  - `invscale = 1 / scale` (never divide by 0)
- Should handle:
  - first call initialization
  - constant-valued inputs
  - outliers without numeric blowups

---

## Deliverables (code + tests + fixtures)

### A) Code modules (new Dreamer v3 package)

Create `src/gbxcule/rl/dreamer_v3/` (if missing) with:

1) `math.py`
   - `symlog(x)` and `symexp(x)`
   - `twohot(y, bins)` returning weights
   - helper `twohot_to_value(weights, bins)` for mean reconstruction
   - all functions accept tensors with arbitrary leading dims

2) `dists.py`
   - `SymlogTwoHot(logits, bins)` distribution wrapper
   - `MSEDistribution` (simple regression distribution for deterministic heads)
   - `BernoulliSafeMode` (optional guard; clamp logits if needed for stability)

3) `return_ema.py`
   - `ReturnEMA(decay, percentiles=(0.05, 0.95), max_range=1.0, eps=1e-8)`
   - `update(values)` returns `(offset, invscale)`
   - `normalize(values)` convenience method

4) `__init__.py` exports for tests / future modules

### B) Golden Bridge fixtures

Add fixtures under `tests/fixtures/dreamer_v3/`:

- `bins.npy` or `bins.json` (exact float32 bin values from reference)
- `symlog_cases.json` (input -> symlog/symexp expected)
- `twohot_cases.json` (input -> twohot weights expected)
- `symlog_twohot_dist.json`:
  - logits, target values, expected log_prob + mean
- `return_ema_cases.json`:
  - sequence of batches, expected EMA low/high + offset/invscale

These fixtures must be generated from the reference Dreamer v3 implementation
(Golden Bridge). Keep dimensions tiny (e.g., K=7 or K=255 with small batch sizes).

---

## Test plan (CPU-only, deterministic)

Add a new test package: `tests/rl_dreamer/`.

### 1) math primitives

- `test_symlog_symexp_roundtrip`:
  - random and edge cases (`0`, small, large, negative)
  - `symexp(symlog(x))` within tolerance
- `test_twohot_mass_conservation`:
  - weights sum to 1 and are non-negative
- `test_twohot_edges`:
  - values below min and above max clamp correctly

### 2) distributions

- `test_symlog_twohot_log_prob_parity` (fixtures)
- `test_symlog_twohot_mean_parity` (fixtures)
- `test_mode_matches_argmax_bin` (simple sanity)

### 3) ReturnEMA

- `test_return_ema_monotone_update`:
  - EMA low/high move toward new percentiles
- `test_return_ema_outlier_stability`:
  - extreme values do not cause inf/NaN
- `test_return_ema_constant_input`:
  - invscale finite; normalize stable

### 4) Determinism

- If using torch randomness in tests, use local `torch.Generator`.
- Keep CPU-only; skip CUDA.

**Gate:** `pytest -q tests/rl_dreamer` passes on CPU with deterministic results.

---

## Step-by-step execution plan

1) **Scaffold Dreamer v3 package**
   - Create `src/gbxcule/rl/dreamer_v3/` with `__init__.py`.
   - Ensure import does not hard-depend on CUDA.

2) **Implement math primitives (`math.py`)**
   - Add `symlog`, `symexp`, `twohot` helpers.
   - Use float32 for internal computation; preserve device.
   - Add unit tests for roundtrip + twohot properties.

3) **Implement distributions (`dists.py`)**
   - Implement `SymlogTwoHot` with `log_prob`, `mean`, `mode`.
   - Ensure it accepts bins as a tensor (from fixture or config).
   - Add simple `MSEDistribution` and a safe Bernoulli helper as needed.

4) **Implement ReturnEMA (`return_ema.py`)**
   - Support initialization-on-first-call semantics.
   - Use `torch.quantile` (or `kthvalue`) across flattened tensor.
   - Apply EMA update and scale clamp.
   - Add tests for monotone update + stability.

5) **Golden Bridge fixtures**
   - Extend fixture generator to export:
     - exact `bins` tensor
     - symlog/twohot cases
     - distribution log_prob + mean
     - ReturnEMA updates for small sequences
   - Store as JSON/NPY under `tests/fixtures/dreamer_v3/`.

6) **Parity tests**
   - Use fixtures to validate all math/distribution outputs.
   - Set tight tolerances (float32) and deterministic seeds.

7) **Gate + cleanup**
   - Run `pytest -q tests/rl_dreamer` on CPU.
   - Ensure no CUDA or global RNG dependency leaks in M2 code.

---

## Definition of Done (M2 gate)

- All M2 tests pass on CPU:
  - symlog/symexp roundtrip
  - twohot mass conservation & edge behavior
  - SymlogTwoHot distribution parity (log_prob/mean)
  - ReturnEMA stability and parity
- Golden Bridge fixtures exist and are referenced by tests.
- Bins are **exactly** fixture-matched (no silent re-generation drift).

---

## Risks & mitigations

1) **TwoHot bin drift** (torch.linspace rounding)
   - Mitigation: use fixture-provided bin values; validate parity tests.

2) **ReturnEMA scale blowups**
   - Mitigation: clamp with `max_range` + `eps`; test constant/outlier cases.

3) **Float64 leaks** (unexpected dtype)
   - Mitigation: cast to float32 in math primitives; assert dtypes in tests.

---

## Open questions / decisions to confirm

- Exact bin range for symlog buckets (`symlog_low/high` or `value_range`).
- Whether ReturnEMA should use p05/p95 or p10/p90 in the reference.
- Whether `MSEDistribution` and `BernoulliSafeMode` are needed in M2 or deferred.

Resolve these in Golden Bridge fixtures and lock the behavior with tests.
