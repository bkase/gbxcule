# Dreamer v3 M5 Plan: Behavior Learning (Imagination + ReturnEMA)

This document is the **spec + execution plan** for Milestone **M5** in `history/dreamer-plan.md`. It is scoped to **behavior learning** (imagination, returns, actor/critic, normalization, and target updates). The reference implementation for semantics is **sheeprl DreamerV3** in `third_party/sheeprl/`. If behavior changes, update this file and the tests together.

---

## 0) Dependencies (must already exist and be correct)

M5 assumes M0–M4 are implemented and validated:

- **Replay + sampling (M1)**: time‑major sequences with `continue` and `episode_id` invariants.
- **Math + dists (M2)**: `symlog/symexp`, symlog‑twohot dist, `ReturnEMA`.
- **RSSM (M3)**: `obs_step` / `img_step` with `is_first` masking and FP32 GRU internals.
- **World model (M4)**: encoder/decoder/reward/continue heads, KL loss with correct weighting/free‑bits, packed2 support.
- **Golden Bridge**: fixtures for imagination + return normalization (tiny, deterministic cases).
- **Replay CUDA ingestion (M6)**: complete; M5 must run on CPU and should remain CUDA‑compatible but must not depend on CUDA.

If any of the above are missing or wrong, **fix them first**; M5 will amplify those errors.

---

## 1) Goal (what “M5 done” means)

Implement **behavior learning** for Dreamer v3:

- **Imagination rollout** from posterior states using the RSSM and the actor policy (discrete or continuous).
- **Lambda returns** computed with **`continue` semantics** (`discount = gamma * continue`).
- **ReturnEMA normalization** applied to lambda values and baselines for the actor advantage (not critic targets).
- **Actor + critic losses**, including the **critic EMA regularization term** (target mean matching).
- **Target critic update** (slow‑moving EMA).

The result must be deterministic on CPU and verifiable with tiny fixtures, while matching the **sheeprl** behavior-learning semantics (see §2, §4–§7).

---

## 2) Non‑negotiables (gotchas to encode in code + tests)

1) **`continue` semantics**
   - `continue = 0.0` for true terminal, `1.0` for truncation/timeouts.
   - Lambda returns use `continues * gamma` (see §5).

2) **First imagined continue comes from real data**
   - Match sheeprl: `true_continue = 1 - terminated` for the **first imagined step**.
   - For our replay contract, this equals the stored `continue` (since truncation keeps `continue=1.0`).

3) **Predicted continues must be hard 0/1**
   - Use `BernoulliSafeMode(...).mode` (not samples, not probs).

4) **Discount weights must match sheeprl**
   - `discount = cumprod(continues * gamma, dim=0) / gamma`
   - Use `discount[:-1]` to weight actor/critic losses.

5) **ReturnEMA normalization is mandatory (actor only)**
   - Normalize **lambda values** and **value baselines** for advantage only.
   - **Do not** normalize critic targets (sheeprl trains critic on raw lambda values).
   - `ReturnEMA` must use **percentiles** (p05/p95) and **EMA**.

6) **Critic regularization (target mean matching)**
   - Critic loss is **not just** `-log_prob(lambda)`.
   - Add `-log_prob(target_critic_mean)` (coef = 1 in sheeprl).

7) **Discrete vs continuous actor objectives differ**
   - **Discrete**: REINFORCE with ST‑onehot actions: `log_prob(action) * advantage.detach()`.
   - **Continuous**: reparameterized objective: `advantage` (no log_prob term).

8) **No global RNG**
   - Action sampling must accept a `torch.Generator` for determinism.

---

## 3) Module layout (new files)

Create a Dreamer v3 behavior package:

- `src/gbxcule/rl/dreamer_v3/imagination.py`
  - `imagine_rollout(...)` (time‑major outputs, **H+1** length like sheeprl)
  - `compute_discounts(...)` (sheeprl formula)
- `src/gbxcule/rl/dreamer_v3/returns.py`
  - `lambda_returns(rewards, values, continues_gamma, lmbda)`
- `src/gbxcule/rl/dreamer_v3/behavior.py`
  - actor/critic loss helpers
  - `behavior_step(...)` (actor + critic update)
- `src/gbxcule/rl/dreamer_v3/targets.py`
  - target critic EMA utilities

Reuse existing utilities:
- `return_ema.ReturnEMA` for Moments/normalization.
- `dists.TwoHotEncodingDistribution` and `dists.BernoulliSafeMode`.
- `rssm.RSSM.imagination` for latent transitions.
- `rng.py` for deterministic generators.

(Exact filenames are flexible, but keep **behavior logic isolated** from engine/IO.)

### 3.1 Config additions (M5)

Extend `DreamerV3Config` to include:
- `actor.ent_coef` (entropy bonus; default 0 or small).
- `actor.clip_gradients` (optional).
- `critic.tau` (EMA factor).
- `critic.target_update_freq` (update cadence).
- `critic.clip_gradients` (optional).

Use existing `algo.{gamma,lmbda,horizon}` and `actor.moments` for ReturnEMA.

---

## 4) Imagination rollout (core algorithm)

### 4.1 Inputs

- **Start states**: posterior RSSM states from real data.
  - Use time‑major `[T, B, ...]`; flatten to `[T*B, ...]` for imagination.
- **Actor**: discrete (ST one‑hot via `OneHotCategoricalStraightThrough` + unimix) or continuous (reparam).
- **World model**: `img_step` for latent transitions; reward + continue heads.
- **Real continue**: `continue_real` for the first imagined step (from data).

### 4.2 Outputs (time‑major)

For horizon `H` and batch `B' = T*B` (**sheeprl uses H+1 length**):

- `features: [H+1, B', F]` (z0..zH)
- `actions: [H+1, B', A]` (one‑hot, ST for discrete; last action is ignored in loss)
- `rewards: [H+1, B']` (predicted; we will use `rewards[1:]`)
- `values: [H+1, B']` (critic mean; we use `[1:]` for lambda targets, `[:-1]` for baseline)
- `continues: [H+1, B']` (first step real, remaining predicted via BernoulliSafeMode **mode**)
- `discounts: [H+1, B']` (weights from `continues`, see §5.3)

### 4.3 Gradient control (match sheeprl)

- **Do not detach** imagined trajectories for reward/value prediction (needed for reparam actor).
- **Detach imagined trajectories** only when feeding actor/critic networks for their own parameters.
- When generating actions inside the imagination loop, feed **detached latent states** into the actor (`actor(latent.detach())`) to match sheeprl.
- Discrete actor uses **detached advantage** in the REINFORCE term.
- Continuous actor uses **reparameterized advantage** (no log_prob).
- World model parameters are **not** updated in behavior step (keep them out of actor/critic optimizers).

### 4.4 Determinism

- Action sampling accepts `torch.Generator`.
- Provide `mode="sample" | "mode"` for evaluation.

---

## 5) Lambda returns (correct bootstrapping + continue)

### 5.1 Inputs (sheeprl indexing)

- `rewards = predicted_rewards[1:]` (length H)
- `values = predicted_values[1:]` (length H)
- `continues_gamma = continues[1:] * gamma` (length H, already scaled)
- `baseline = predicted_values[:-1]` (length H, used for advantage)
- `gamma`, `lambda` from config.

### 5.2 Definition (matches `sheeprl.algos.dreamer_v3.utils.compute_lambda_values`)

Given `continues_gamma` already includes `gamma`:

```
vals = [values[-1:]]
interm = rewards + continues_gamma * values * (1 - lambda)
for t in reversed(range(len(continues_gamma))):
    vals.append(interm[t] + continues_gamma[t] * lambda * vals[-1])
lambda_values = reverse(vals)[:-1]
```

### 5.3 Discount weights (for loss weighting; sheeprl)

- `continues_raw` is 0/1 (first step from real data, rest from model).
- `discount = cumprod(continues_raw * gamma, dim=0) / gamma`
- Use `discount[:-1]` to weight actor/critic losses.

---

## 6) ReturnEMA normalization (mandatory)

### 6.1 Update (sheeprl Moments)

- Update with **lambda values** (detached) and gather across ranks if distributed.
- Compute p05/p95, then EMA: `low = decay*low + (1-decay)*p05`, `high = decay*high + (1-decay)*p95`.
- `invscale = max(1/actor.moments.max, high - low)` (note: used as denominator).

### 6.2 Normalize

- `norm(x) = (x - offset) / invscale`.

### 6.3 Usage

- Advantage: `adv = norm(lambda_values) - norm(values_baseline)`.
- Critic loss uses **raw** lambda values (no normalization).

---

## 7) Actor + critic losses (match sheeprl)

### 7.1 Actor loss

- **Baseline**: `baseline = predicted_values[:-1]` (critic mean).
- `adv = norm(lambda_values) - norm(baseline)`.
- **Discrete** (ST one‑hot):  
  `objective = sum(log_prob(action_detached)) * adv.detach()`
- **Continuous** (reparam):  
  `objective = adv`
- Entropy term (if available):  
  `entropy = ent_coef * sum(entropy(dist))`
- Final loss (align to H steps):  
  `policy_loss = -mean(discount[:-1].detach() * (objective + entropy)[:-1])`

### 7.2 Critic loss (with EMA regularization)

- Use **TwoHotEncodingDistribution** (sheeprl uses two‑hot bins).
- `qv = critic(imagined_trajectories[:-1])`
- `target_mean = target_critic(imagined_trajectories[:-1]).mean`
- Loss (no extra coef in sheeprl):
  - `L_value = -qv.log_prob(lambda_values.detach())`
  - `L_reg = -qv.log_prob(target_mean.detach())`
  - `L_critic = mean((L_value + L_reg) * discount[:-1].squeeze(-1))`

### 7.3 Target critic update (sheeprl)

- Update **periodically**, not necessarily every step:
  - `per_rank_target_network_update_freq` controls cadence.
  - At the first update: `tau = 1.0` (hard sync).
  - Otherwise: `target = tau * critic + (1 - tau) * target`.

---

## 8) Metrics (make debugging cheap)

Log (per update):

- `actor_loss`, `critic_loss`, `value_reg_loss`, `entropy`
- `lambda_mean`, `lambda_std`, `lambda_p05`, `lambda_p95`
- `ema_offset`, `ema_invscale`
- `discount_mean`, `discount_min`
- `adv_mean`, `adv_std`
- `target_critic_mean` stats

---

## 9) Tests & fixtures (CPU‑first, deterministic)

### 9.1 Unit tests

1) **Lambda returns correctness**
   - Hand‑checked tiny tensors.
   - Validate `continue=0` stops bootstrapping; `continue=1` bootstraps.

2) **Discount weights**
   - Verify `discount = cumprod(continue * gamma) / gamma`.
   - Ensure first step uses **real continue** (`1 - terminated`).

3) **ReturnEMA normalization**
   - Stable under scaling; no divide‑by‑zero; monotone response to outliers.

4) **Critic regularization**
   - Ensure loss includes both terms with **coef=1** (sheeprl).
   - Verify gradients flow to critic params.

### 9.2 Integration tests

1) **Imagination shape test**
   - `imagine_rollout` outputs **`[H+1, B', ...]`** and matches dtype/device.

2) **Gradient flow test**
   - Actor + critic parameters receive non‑zero gradients.
   - World model parameters are not stepped by behavior optimizers.

3) **ReturnEMA integration**
   - Scaling rewards changes critic loss scale (expected), but **normalized advantage** remains stable.

### 9.3 Golden fixtures

- Tiny fixtures for:
  - imagination rollout (one step)
  - lambda returns + normalization
  - critic regularization term

---

## 10) Acceptance gates (M5 done)

1) **All M5 tests green on CPU** (deterministic).
2) **Behavior step smoke** on toy data:
   - loss finite, gradients finite, no NaNs.
3) **ReturnEMA + continue semantics validated**
   - explicit tests for `continue=0/1`, truncation vs terminal.

---

## 11) Suggested execution order

1) Implement `returns.py` + tests (lambda returns + discount weights).
2) Implement `imagination.py` + shape tests.
3) Implement `behavior.py` losses + target update + tests.
4) Add ReturnEMA integration + critic regularization + fixtures.
5) Run full M5 test suite + CPU smoke.

---

## 12) Out of scope for M5

- Async engine integration (M7).
- Full system training / validation (M8).

---

## 13) Notes for future integration (M7/M8)

- Keep all behavior functions **pure** (no IO, no global RNG).
- Expose config flags for `ent_coef`, `tau`, `horizon`, `lambda` (value‑reg default = 1 unless explicitly added).
- Keep time‑major contract everywhere (`[T, B, ...]`).

