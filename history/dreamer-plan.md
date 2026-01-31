# Master Plan v2: GBxCuLE → Dreamer v3 (zero-copy GPU + deterministic CPU + Golden Bridge)

## 0) Revised execution order (optimized)

1. **M0:** Scaffolding + Golden Bridge skeleton
2. **M1:** ReplayRing (CPU first) with **mandatory `episode_id`**, **non-strict sampling**, and **`continue` semantics**
3. **M2:** Math & distributions (**symlog-twohot**) + **ReturnEMA percentile normalizer**
4. **M3:** RSSM + scan loop with **`is_first` reset masking**, and **GRU FP32-internals stability**
5. **M4:** World model + reconstruction loss (**stop-gradient KL + free bits**, no KL balancing) + packed2 unpack micro-benchmark
6. **M5:** Behavior learning (imagination) using **ReturnEMA normalization** + `continue` for lambda-returns
7. **M6:** CUDA direct-write replay ingestion + zero-copy gates (+ packed2 unpack contingency kernel if needed)
8. **M7:** Async Dreamer engine (**commit_stride configurable & decoupled**) + replay-ratio scheduler
9. **M8:** Full system validation (Standing Still + Exit Oak) + regression gates

---

## 1) Architectural North Star

We build **two runtime engines** sharing the same functional core (models/math/losses), differing only in execution and IO:

### A) Production Engine: `DreamerV3Engine` (CUDA, zero-copy, async)

- **Data path:** Warp → **packed2 uint8 in VRAM** → replay ring in VRAM → sampling in VRAM → training in VRAM
- **Concurrency:** **Actor stream** (env stepping + replay writes) and **Learner stream** (sampling + training)
- **Synchronization:** **commit markers + CUDA events** fence “data becomes readable” at segment boundaries (not per slot)
- **Invariants:** No host transfers in the hot path; pointer stability; no implicit syncs; bounded overwrite safety

### B) Reference Engine: `DreamerV3EngineCPU` (CPU, deterministic, test harness)

- **Data path:** env (toy or Warp CPU) → CPU replay → training on CPU
- **Determinism:** strict; **global RNG forbidden** in core components; sampling requires `torch.Generator`
- **Role:** fast correctness loop (<2 min), reproducible bug isolation

---

## 2) Core decisions & contracts (lock these now)

### 2.1 Golden Bridge (truth without runtime dependency pollution)

- A single tool generates compact fixtures from a known-correct Dreamer v3 implementation and writes them into `tests/fixtures/dreamer_v3/`
- Production code **never** imports `hydra`, `lightning`, `fabric`, etc.
- Fixtures are **input/output pairs** for:
  - math primitives
  - symlog-twohot distribution behavior (log_prob/mean)
  - RSSM dynamic scan (priors/posteriors logits + selected states)
  - reconstruction loss components (including stop-gradient KL + free bits)
  - imagination/returns normalization on tiny cases

**Fixture policy**

- Keep fixtures **tiny** (small dims) and **few** (cover semantics, not every layer).
- Avoid sampling brittleness:
  - prefer storing logits and deterministic transforms;
  - when sampling is needed, use an explicit generator and store sampled indices.

---

### 2.2 Canonical tensor contracts (and who transposes)

We standardize on **time-major internally**:

- **Replay storage:** `[Tcap, N, ...]` (time, env)
- **Sampled sequences:** `[T, B, ...]` (time, batch)
- If any module wants `[B, T, ...]`, it must call an explicit helper (centralized + tested), used sparingly.

---

### 2.3 Replay schema (transition-aligned, direct-write friendly; boundary-safe; truncation-correct)

Each time index `t` stores the transition out of `obs[t]`:

- `obs[t]` : observation at time t (**packed2 uint8**: `[N, 1, 72, 20]`)
- `action[t]` : int32 `[N]` (policy action id; onehot derived on-the-fly for RSSM input)
- `reward[t]` : float32 `[N]`
- `is_first[t]` : bool `[N]` (whether `obs[t]` starts an episode)
- **`continue[t]` : float32 `[N]` (value bootstrapping semantics)**
  - `continue = 0.0` for true terminal (`terminated`)
  - `continue = 1.0` for timeout (`truncated`) and for normal steps
  - resets may still occur on truncation; `continue` encodes bootstrapping, not reset

- **`episode_id[t]` : int32 `[N]` (mandatory)**
  - strictly increasing per env on every reset; used to validate temporal continuity

Optional (debug/logging; not required for algorithm correctness):

- `terminated[t]` bool `[N]`
- `truncated[t]` bool `[N]`

**Is_first contract**

- If env resets between `t` and `t+1`, then:
  - `is_first[t+1]=1`
  - `episode_id[t+1]=episode_id[t]+1`

- Crucially: `is_first` may appear _inside_ sampled sequences; Dreamer handles this by resetting latent state in the scan loop.

**Sampling contract (non-strict)**

- Sampling **does not reject** sequences containing `is_first` mid-window.
- Instead, the RSSM scan must reset state where `is_first[t] == 1`.

---

### 2.4 Packed2 contract

- Replay stores **packed2**.
- Encoder supports packed2 **natively** (unpack + normalize inside model).
- Unpacked frames are only materialized if explicitly requested for debugging or benchmarking.

---

## 3) Execution milestones (incremental, verifiable, automated)

### Milestone M0 — Scaffolding + contracts + Golden Bridge skeleton

**Goal:** Establish verification infrastructure before heavy implementation.

**Deliverables**

- `src/gbxcule/rl/dreamer_v3/`
  - `config.py` (dataclasses + validation; includes: bins, free_bits, precision policy, commit_stride, etc.)
  - `schema.py` (shape/dtype constants + helper checks)
  - `rng.py` (explicit generator utilities; forbid global RNG in core)
  - empty `engine_cpu.py`, `engine_cuda.py`

- Golden fixture generator scaffold
- `tests/rl_dreamer/` harness
  - CPU determinism fixture (`torch.use_deterministic_algorithms(True)` where safe)

**Tests**

- config validation
- RNG contract tests (core sampling must accept `Generator`)

**Gate**

- CPU `pytest -q` green, no CUDA required.

---

### Milestone M1 — ReplayRing (device-agnostic) + continuity invariants (episode_id) + continue semantics

**Goal:** Build the memory spine correctly (where most silent bugs live).

**Deliverables**

- `ReplayRing` with time-major tensors:
  - `obs: uint8[Tcap, N, 1, 72, 20]` (packed2)
  - `action: int32[Tcap, N]`
  - `reward: float32[Tcap, N]`
  - `is_first: bool[Tcap, N]`
  - `continue: float32[Tcap, N]`
  - `episode_id: int32[Tcap, N]` (mandatory)

- `push_step(t, ...)` and `sample_sequences(B, T, gen)`:
  - sampling allows boundaries (`is_first` anywhere)
  - deterministic indices via generator

- Replay invariant checker (tests + optional debug mode):
  - `episode_id[t+1] == episode_id[t] OR is_first[t+1] == 1`
  - `is_first[t+1] == 1 => episode_id[t+1] == episode_id[t] + 1`
  - `continue[t] in {0.0, 1.0}` (or tight float check)

**Tests (CPU)**

- wraparound correctness (write past capacity)
- temporal continuity checks using `episode_id` (including around wrap)
- deterministic sampling (same seed/gen → same indices)
- truncation bootstrapping correctness:
  - truncated step writes `continue=1.0` even though reset occurs

**Gate**

- CPU tests green and deterministic.

---

### Milestone M2 — Math & distributions primitives (symlog-twohot) + ReturnEMA percentile normalizer

**Goal:** Nail the “atoms” Dreamer v3 depends on for stability.

**Deliverables**

- `math.py`: `symlog`, `symexp`, two-hot bucketization helpers
- **Symlog-spaced TwoHot distribution**
  - Implement `SymlogTwoHot` (or implement TwoHot with internal symlog):
    - for `log_prob(x)`: compute `y = symlog(x)` then two-hot in **linear bins over y**
    - for `mean/mode`: expectation in symlog space then `symexp`

  - Bins: typically **255** (configurable), uniformly spaced over symlog-range (configurable)

- `dists.py`: `SymlogDistribution`, `MSEDistribution`, `BernoulliSafeMode` as needed
- **ReturnEMA class (percentile-based normalization, mandatory for v3 behavior learning)**
  - Tracks EMA of **p05** and **p95** of returns (or lambda targets) across batches
  - Produces `(offset, invscale)` where `invscale = max(1/max_range, high-low)` and offset=low
  - Decay configurable; percentiles fixed (5%/95%) unless explicitly changed

**Tests**

- parity fixtures for symlog/twohot and distribution log_prob/mean
- property tests: twohot mass conservation; symlog/symexp roundtrip bounds
- ReturnEMA behavior tests:
  - monotone EMA response, stable under outliers, no divide-by-zero

**Gate**

- CPU parity passes within tight tolerance; ReturnEMA passes stability tests.

---

### Milestone M3 — RSSM core with is_first reset masking + GRU FP32 internals (AMP stability)

**Goal:** Implement Dreamer v3 RSSM that can train across episode boundaries and not explode under AMP.

**Deliverables**

- `rssm.py`:
  - recurrent model (LayerNormGRUCell-based)
  - transition model (prior logits)
  - representation model (posterior logits)
  - unimix
  - learnable initial recurrent state option

- **Dynamic scan consumes `is_first` and resets state inside the loop**
  - `is_first` is time-major `[T, B]` (or adapted) and applied at each step:
    - reset deterministic state: `h = h * (1 - is_first[t]) + h0 * is_first[t]`
    - reset posterior seed similarly to an initial posterior

- **GRU precision stability policy (mandatory)**
  - Force FP32 for internal GRU math even under autocast:
    - compute gates/state update in float32
    - optionally keep recurrent state stored in float32
    - cast outputs back to model dtype as needed

- Sampling utilities:
  - ST onehot categorical sampling; deterministic mode for tests

**Tests**

- shape tests (time-major `[T, B, ...]`)
- multi-boundary sequences: verify correct resets where `is_first[t]==1`
- AMP/autocast stability test:
  - run RSSM forward under `torch.autocast` (cuda and/or cpu where applicable)
  - assert outputs finite and no NaNs; compare to FP32 baseline within tolerance

- parity fixtures for priors/posteriors logits on tiny deterministic cases

**Gate**

- CPU deterministic RSSM tests + parity green; AMP stability test passes.

---

### Milestone M4 — World model forward + reconstruction loss (no KL balancing; stop-grad KL + free bits) + unpack micro-benchmark

**Goal:** Get Dreamer v3 reconstruction objective correct and stable, and confirm packed2 unpack performance.

**Deliverables**

- `world_model.py`
  - packed2 encoder (unpack + normalize inside)
  - decoder
  - reward head (symlog-twohot bins)
  - continue head (Bernoulli)

- **Reconstruction loss: stop-gradient KL + free bits (no KL balancing)**
  - Compute KL in two stop-grad paths:
    - **Dynamics KL:** `KL(stop_grad(posterior) || prior)` (updates dynamics/transition)
    - **Representation KL:** `KL(posterior || stop_grad(prior))` (updates representation)

  - Apply **free bits / free nats** threshold (>= 1.0 default) before averaging
  - Total KL = dynamics_kl + representation_kl (no balancing coefficient)

- `train_world_model.py`: one WM train step function
- **Packed2 unpack micro-benchmark**
  - benchmark encoder forward (and optionally full WM forward) on representative batch sizes
  - record ms/step and throughput; define acceptable ceiling

- **Contingency hook**
  - `unpack_impl = "lut" | "triton" | "warp"` (default “lut”)
  - stable interface so M6 can swap without refactoring

**Tests**

- reconstruction loss parity fixture (including stop-grad KL behavior + free bits)
- backward: grads finite and non-zero
- tiny “overfit one batch”: recon loss decreases
- free-bits test: KL term never drops below threshold in loss aggregation (per design)

**Gate**

- CPU world model train step stable; loss parity green; unpack benchmark exists with fallback path.

---

### Milestone M5 — Behavior learning (imagination) with ReturnEMA normalization + continue-correct lambda returns

**Goal:** Learning “inside the dream” without scale collapse.

**Deliverables**

- `imagination.py`: latent rollout for horizon H
- Lambda-return computation uses:
  - discounts = `gamma * continue[t]`

- **Return normalization is mandatory**
  - Use ReturnEMA to normalize:
    - lambda targets (returns) and baseline values into a stable range

  - actor objective uses normalized advantage:
    - `adv = norm(lambda) - norm(value_baseline)`

- actor loss + critic loss + target critic updates

**Tests**

- imagination shape tests
- gradients flow (actor + critic + rssm)
- ReturnEMA integration test:
  - scaling returns up/down does not destabilize loss magnitudes

- truncation correctness:
  - `continue=1` timeouts still bootstrap; `continue=0` true terminal does not

**Gate**

- CPU “single train step” reduces loss on toy data; no NaNs; normalization and continue semantics validated.

---

### Milestone M6 — GPU ingestion: Warp → replay.obs direct-write + zero-copy gates + optional unpack kernel integration

**Goal:** Make packed2 direct-write real and enforce zero-copy invariants.

**Deliverables**

- `ReplayRingCUDA` (same API as ReplayRing, allocated on CUDA)
- `obs_slot(t)` returns `[N, 1, 72, 20]` contiguous view for direct write
- Commit scheme:
  - actor writes slots continuously
  - every `commit_stride` steps (config), actor records a CUDA event and updates `committed_t`
  - learner samples only from `<= committed_t - safety_margin`

- If benchmark demands it, integrate `unpack_impl` alternative kernel (Triton/Warp)

**CUDA tests (skippable)**

- pointer stability: `data_ptr()` stable across steps
- stream ordering: learner sees committed data only after fence
- profiler memcpy gate: **no HtoD/DtoH memcpy in hot loop**
- unpack perf regression test (optional DGX gate): ensure chosen unpack_impl meets throughput target

**Gate**

- DGX GPU gate: memcpy gate + pointer stability + ingest smoke (+ unpack perf if enabled).

---

### Milestone M7 — Async Dreamer engine integration (commit_stride decoupled; learner starvation-proof)

**Goal:** Replace PPO engine architecture with Dreamer’s two-loop training without starving the learner.

**Deliverables**

- `async_dreamer_v3_engine.py`
  - actor stream: env step + render → replay write
  - learner stream: sample sequences → WM update → behavior update
  - periodic weight sync learner→actor (device-to-device copy) + event
  - metrics: timing (actor_ms/learner_ms/overlap_eff), loss scalars, replay ratio
  - failfast bundle integration (non-finite detection, device/shape asserts in debug)

- **Config decoupling to avoid starvation**
  - `steps_per_rollout` (actor-side convenience) independent from:
  - `commit_stride` (visibility cadence)
  - `seq_len` (learner training requirement)
  - Engine ensures learner can assemble full sequences by:
    - committing frequently enough (good default: `commit_stride <= seq_len/2`)
    - `min_ready_steps` guard before training begins
    - sampling windows from stable committed region

**Tests**

- CPU engine toy smoke (deterministic)
- CUDA tiny smoke (skippable)
- CUDA memcpy gate for engine path (skippable)
- starvation test (CPU + CUDA optional):
  - with small `steps_per_rollout`, ensure learner still trains when `commit_stride` is small enough

**Gate**

- CPU gate green always; GPU gate green on DGX for main.

---

### Milestone M8 — Full system validation & regression gates

**Goal:** Prove learning on real tasks + lock quality bars.

**Deliverables**

- Unified CLI: `tools/rl_train_gpu.py --algo dreamer_v3`
- Eval tool: `tools/rl_eval.py --algo dreamer_v3`
- Bench tool: `tools/rl_gpu_bench_dreamer.py`
- Validation runbook: `docs/dreamer_v3_validation.md`
- Two validation scenarios:
  1. **Standing Still / Reconstruction**: recon quality improves fast
  2. **Exit Oak**: success rate rises; stable training (no collapse)

**Gates**

- CPU daily gate: M0–M5 tests + CPU toy engine smoke
- DGX “main” gate: CUDA smoke + memcpy gate + pointer stability + unpack perf (if enabled) + basic throughput sanity

---

## 4) Parallel workstreams (safe concurrency)

1. ReplayRing + continuity invariants + continue semantics (M1)
2. Golden Bridge + fixtures infrastructure (M0/M2/M4 parity fixtures)
3. RSSM scan + is_first masking + AMP stability (M3)
4. World model loss (stop-grad KL + free bits) + unpack benchmark (M4)
5. Warp direct-write + commit scheme + profiler gate (+ unpack kernel if needed) (M6)

---

## 5) Risk register (and baked-in mitigations)

1. **Episode boundary poisoning (silent)**
   → non-strict sampling + explicit `is_first` masking inside RSSM + mandatory `episode_id` invariants.

2. **Incorrect bootstrapping at time limits**
   → store `continue` (float32) and compute lambda-returns with `gamma * continue`.

3. **Scale collapse in behavior learning**
   → mandatory ReturnEMA percentile normalization (EMA of p05/p95) for targets/baselines/advantages.

4. **RNN/GRU instability under AMP**
   → FP32 internal GRU math policy + explicit autocast stability tests.

5. **Hidden host transfers**
   → profiler memcpy gate + runtime asserts + no `.cpu()` in engine loops.

6. **Replay sampling performance / learner starvation**
   → commit markers + **configurable commit_stride** decoupled from rollout length + `min_ready_steps` guard.

7. **Packed2 unpack becomes training bottleneck**
   → encoder micro-benchmark + contingency switch to Triton/Warp unpack kernel.

8. **Shape drift over time**
   → canonical time-major contract + explicit conversion helpers + tests.

---

## 6) Recommended starting point

Start with **M0 scaffolding** and **M1 ReplayRing + tests** in parallel. This locks the data contract (including `continue` + mandatory `episode_id` + boundary-allowed sampling) and verification framework before the model/engine complexity arrives.
