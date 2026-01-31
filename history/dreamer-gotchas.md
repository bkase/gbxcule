This response consolidates the critical findings from both the plan review and the `sheeprl` codebase analysis. These are the specific technical constraints ("gotchas") you must track to ensure the implementation works.

### **M0: Scaffolding & Configs**

- **Gotcha: Precision Policy (Mixed Precision).**
- **The Issue:** `sheeprl` uses `float32` for model internals (especially the Recurrent State/GRU) but inputs are often `uint8`. Relying on implicit PyTorch autocast usually breaks the recurrent state update.
- **The Fix:** Define a strict policy in `config.py`: Inputs/Replay=`uint8`, Model Internals=`float32`, Output Distributions=`float32`. Explicitly cast GRU internals to `float32`.

### **M1: ReplayRing & Sampling**

- **Gotcha: The "Is_First" Shift.**
- **The Issue:** `sheeprl` resets the RNN state at the start of _every_ training batch (index 0) regardless of the episode boundary. It essentially treats the first step of any sampled sequence as a "cut".
- **The Fix:** Your `ReplayRing` must store true `is_first` flags. When sampling for training (M7), the learner loop must handle this alignment—either by forcing a reset at index 0 or (better) bootstrapping from a stored state.

- **Gotcha: Continuity Invariants.**
- **The Issue:** Incorrect handling of `done` vs `truncated` poisons lambda returns.
- **The Fix:** Rely on the `episode_id` invariant in `ReplayRing` to validate continuity. Store `continue` as a float (0.0 for terminal, 1.0 for others) for direct use in lambda calculations.

### **M2: Math & Distributions**

- **Gotcha: TwoHot Bucket Alignment.**
- **The Issue:** `torch.linspace` rounding differences can shift bucket indices by 1, exploding KL loss.
- **The Fix:** Do not generate bins dynamically. Serialize the _exact_ bucket values (`bins` tensor) from the reference into your Golden Bridge fixtures.

- **Gotcha: Symlog Input Scope.**
- **The Issue:** `MLPEncoder` (vectors) applies `symlog` by default. `CNNEncoder` (pixels) does _not_ (it uses `x / 255.0 - 0.5`). Applying `symlog` to pixels destroys texture.
- **The Fix:** `world_model.py` must distinguish between `obs_type="rgb"` (linear scale) and `obs_type="vector"` (symlog).

### **M3: RSSM & Scan Loop**

- **Gotcha: Learnable Initial State.**
- **The Issue:** `sheeprl`'s RSSM uses a learnable parameter for the initial state, not zeros.
- **The Fix:** In the scan loop, when `is_first[t] == 1`, blend the state: `h[t] = (1 - is_first) * h_prev + is_first * h_init_param`.

- **Gotcha: Hafner Initialization.**
- **The Issue:** Standard initialization causes value explosion in mixed precision.
- **The Fix:** Implement the specific `init_weights` (Hafner init) logic from `sheeprl` for the GRU and dense layers.

### **M4: World Model & Loss (The "No Balancing" Myth)**

- **Gotcha: KL Loss Scaling.**
- **The Issue:** "No balancing" in V3 refers to _dynamic_ tuning. The code _does_ use fixed weights: typically **0.5** for dynamics loss (`L_dyn`) and **0.1** for representation loss (`L_rep`), both clamped with `free_nats` (1.0).
- **The Fix:** Do not assume 1.0 weights. Expose `beta_dyn=0.5` and `beta_rep=0.1` in `config.py` and implement the weighted sum: `L_KL = beta_dyn * max(KL_dyn, free) + beta_rep * max(KL_rep, free)`.

- **Gotcha: Bernoulli Continue Target.**
- **The Fix:** The `continue` head loss must use strictly 0.0/1.0 targets (derived from `1 - terminated`), separate from the gamma discounting logic.

### **M5: Behavior Learning (Critic Regularization)**

- **Gotcha: Target Mean Matching.**
- **The Issue:** The critic loss is not just MSE. It minimizes `-log_prob(return) - log_prob(target_critic_mean)`. This regularizes the critic's distribution toward the slow-moving target's mean.
- **The Fix:** Implement this **EMA Regularization Loss** term in the critic update (M5).

- **Gotcha: Discount Vector.**
- **The Fix:** `sheeprl` computes discount as `cumprod(continues * gamma)`. Ensure `continues` includes the real data (`1 - terminated`) for the first imagination step.

### **M6: GPU Ingestion**

- **Gotcha: Alignment.**
- **The Fix:** Ensure `ReplayRingCUDA` allocations use `torch.empty(..., device='cuda')` to guarantee 256-byte alignment, which matters for vectorized loads in kernels.

### **M7: Async Engine**

- **Gotcha: Stale Weights.**
- **The Fix:** Explicitly sync weights (`learner -> actor`) _before_ recording the `policy_ready_event` to prevent the Actor from waking up and using old weights (race condition).

### **M8: Validation**

- **Gotcha: Latent Drift.**
- **The Fix:** Monitor `prior_entropy` vs. `posterior_entropy` during the "Standing Still" test. Significant divergence implies broken `unimix` or KL regularization.

**Immediate Action:**
Update **M4** requirements to include `beta_dyn`/`beta_rep` scaling and **M5** to include the critic regularization term. Generate a fixture for a **single RSSM step** first.
