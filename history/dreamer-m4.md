# Dreamer v3 M4 Plan: World Model, Reconstruction Loss, Packed2 Bench
# (Aligned with sheeprl DreamerV3 in third_party/sheeprl)

## 0) Scope and success criteria

In scope:
- World model forward pass for packed2 pixel observations (and vector obs path if supported).
- Reconstruction losses for obs, reward, and continue.
- Stop-gradient KL with free-nats clamp and fixed weights (beta_dyn, beta_rep) and KL regularizer.
- One-step training function for the world model with metrics.
- Packed2 unpack micro-benchmark with a stable interface for future kernel swaps.
- Fixtures + tests that lock the above behavior.

Out of scope:
- Behavior learning (M5), async engine (M7), GPU direct-write replay (M6), dynamic KL balancing.

Success criteria (M4 gate):
- CPU tests pass: loss parity fixtures, stop-grad behavior, free-nats clamp, one-batch overfit.
- World model forward/backward produces finite grads and stable loss curves.
- Packed2 benchmark exists, runs, and records throughput/latency.

---

## 0.1) Sheeprl reference alignment (what this plan mirrors)

From third_party/sheeprl:
- Obs reconstruction: MSEDistribution for images, SymlogDistribution for vectors.
- Reward: TwoHotEncodingDistribution with symlog/symexp; bins are torch.linspace(low=-20, high=20, bins).
- Continue: BernoulliSafeMode(logits=continue_model(...)), targets are strict 0/1 via (1 - terminated).
- KL: stop-grad both ways with free-nats clamp, weighted by kl_dynamic=0.5 and kl_representation=0.1, then scaled by kl_regularizer (default 1.0).
- Pixel normalization: /255 - 0.5 for image inputs (match via packed2 scaling).

---

## 1) Inputs, outputs, and contracts

Inputs (time-major):
- obs: uint8 packed2, shape [T, B, 1, H, W_bytes]
- action: int32, [T, B]
- reward: float32, [T, B]
- is_first: bool, [T, B]
- continue: float32, [T, B] (strict 0.0 or 1.0, 1 - terminated)

World model outputs:
- RSSM posterior + prior states for each time step.
- Features for decoder/reward/continue heads.
- Distributions for obs, reward, continue.
- Metrics: recon loss terms, KL dyn/rep, entropy, etc.

Contracts to preserve:
- Packed2 unpack is internal to the encoder. No materialized full frames outside the model by default.
- Pixel path does NOT apply symlog to inputs. Vector path DOES apply symlog (per gotchas).
- Pixel normalization matches sheeprl: float inputs in [-0.5, 0.5].
- Continue head targets must be strict 0/1 (use replay continue or 1 - terminated).
- KL is stop-grad both ways with free-nats clamp and fixed weights; add kl_regularizer scale.

---

## 2) Config additions (dreamer_v3/config.py)

Add and validate:
- obs_type: "rgb" or "vector"
- obs_format: "packed2" or "u8" (packed2 primary)
- unpack_impl: "lut" | "triton" | "warp" (default "lut")
- free_nats: float (default 1.0)
- beta_dyn: float (default 0.5, sheeprl kl_dynamic)
- beta_rep: float (default 0.1, sheeprl kl_representation)
- kl_regularizer: float (default 1.0, sheeprl kl_regularizer)
- reward_bins: int (default 255, sheeprl reward_model.bins)
- reward_twohot_low: float (default -20.0)
- reward_twohot_high: float (default 20.0)
- continue_scale_factor: float (default 1.0, sheeprl continue_scale_factor)
- obs_norm: "zero_centered" (pixels scaled to [-0.5, 0.5])
- precision policy fields (match M0/M3, ensure GRU internals FP32)

Validation rules:
- free_nats >= 0
- beta_dyn > 0, beta_rep > 0
- kl_regularizer > 0
- reward_twohot_low < reward_twohot_high
- obs_type and obs_format consistent
- reward_bins matches fixture bins length

---

## 3) Module layout and responsibilities

Create under src/gbxcule/rl/dreamer_v3/ (or reuse existing scaffolding):
- world_model.py
  - WorldModel class
  - forward() to compute prior/posterior + head outputs
  - loss() to compute recon and KL
- encoders.py
  - Packed2PixelEncoder (unpack + normalize inside)
  - VectorEncoder (symlog on inputs)
- decoders.py
  - PixelDecoder (paired with MSEDistribution)
  - VectorDecoder (paired with SymlogDistribution)
- heads.py
  - RewardHead (TwoHotEncodingDistribution; symlog/symexp; bins from config)
  - ContinueHead (BernoulliSafeMode)
- dists.py (or reuse existing)
  - MSEDistribution, SymlogDistribution, TwoHotEncodingDistribution, BernoulliSafeMode
- train_world_model.py
  - wm_train_step(batch, model, optim, cfg, gen) -> loss, metrics

Notes:
- Reuse gbxcule.rl.packed_pixels.get_unpack_lut for LUT-based unpack.
- Keep time-major tensors throughout; explicitly document any [B, T] conversions.
- Prefer dict observations to mirror sheeprl (cnn_keys + mlp_keys); single-key case still supported.

---

## 4) World model forward pass design

Data flow (time-major):
1) Encoder:
   - If obs_type="rgb": unpack packed2 to uint8, then scale to float in [-0.5, 0.5].
     - For packed2: pixel in {0,1,2,3}. Use x = (x / 3.0) - 0.5 to mirror sheeprl's (x/255 - 0.5).
   - If obs_type="vector": cast to float32, apply symlog.
2) RSSM:
   - Use M3 RSSM scan with is_first reset masking.
   - Produce posterior and prior at each step, plus features.
3) Heads:
   - Decoder distributions:
     - cnn_keys -> MSEDistribution(recon, dims=len(spatial_dims))
     - mlp_keys -> SymlogDistribution(recon, dims=len(vector_dims))
   - Reward distribution: TwoHotEncodingDistribution(logits, dims=1, low/high from config).
   - Continue distribution: Independent(BernoulliSafeMode(logits), 1).

Outputs should include:
- posterior/prior distributions and samples (for loss).
- features used by each head.
- per-head distribution objects for log_prob.

---

## 5) Reconstruction and KL loss (core of M4)

Reconstruction losses:
- observation_loss = -sum_k po[k].log_prob(obs[k]) where:
  - cnn keys use MSEDistribution (sum over spatial dims)
  - mlp keys use SymlogDistribution (sum over feature dims)
- reward_loss = -TwoHotEncodingDistribution.log_prob(reward) using symlog.
- continue_loss = continue_scale_factor * -BernoulliSafeMode.log_prob(continue_target).

KL losses (stop-grad both ways):
- kl_dyn = KL(stop_grad(posterior) || prior)
- kl_rep = KL(posterior || stop_grad(prior))

Free-nats clamp:
- kl_dyn = max(kl_dyn, free_nats)
- kl_rep = max(kl_rep, free_nats)

Fixed weights (per gotchas):
- kl_total = beta_dyn * kl_dyn + beta_rep * kl_rep

Total loss:
- total = mean(kl_regularizer * kl_total + observation_loss + reward_loss + continue_loss)

Implementation details:
- compute KL per time step and batch, then reduce (mean) after free-nats clamp.
- ensure stop-grad is applied at the distribution level (detach stats or samples)
- metrics should expose unclamped and clamped KLs for debugging.
- use OneHotCategoricalStraightThrough inside Independent(..., 1) for KL (sheeprl).

---

## 6) Training step (train_world_model.py)

Provide a small, testable training wrapper:
- Accept a batch dict (obs, action, reward, is_first, continue).
- Run model.forward + model.loss.
- Backprop, step optimizer, return scalar loss and metrics.
- Make RNG explicit (torch.Generator) where sampling is needed.
- Ensure continue targets use (1 - terminated) and remain strict 0/1.

Add a minimal overfit utility:
- run N steps on a fixed batch and confirm recon loss decreases.

---

## 7) Golden Bridge fixtures (parity locks)

Fixtures needed for M4:
- Encoder + decoder reconstruction on tiny inputs (pixel and/or vector path).
- Reward head log_prob/mean vs reference.
- Continue head log_prob on 0/1 targets.
- KL dyn/rep with stop-grad and free-nats clamp (exact values).

Fixture policy:
- Use reference bins (capture torch.linspace(-20, 20, bins) from sheeprl).
- Keep shapes tiny but cover edge cases:
  - is_first inside the sequence
  - continue targets with both 0 and 1
  - KL below and above free_nats

---

## 8) Tests

Add tests under tests/rl_dreamer/:
- test_world_model_shapes.py
  - time-major input/output shapes and dtype checks
- test_world_model_loss.py
  - stop-grad behavior (dyn vs rep) via gradient checks
  - free-nats clamp enforced
  - beta_dyn/beta_rep scaling applied
  - kl_regularizer applied
- test_continue_head.py
  - only 0/1 targets accepted; loss behaves as expected
  - continue_scale_factor applied
- test_pixel_encoder.py
  - packed2 unpack correctness; no symlog for pixels; normalize to [-0.5, 0.5]
- test_overfit_one_batch.py
  - recon loss decreases on tiny fixed batch
- fixture parity tests
  - compare against Golden Bridge outputs
  - reward twohot bins/low/high parity with sheeprl

---

## 9) Packed2 unpack micro-benchmark

Goal:
- quantify the cost of LUT unpack in encoder and set a baseline.

Implementation:
- new script (suggested path): tools/rl_dreamer_packed2_bench.py
- options:
  - device: cpu/cuda
  - batch size, sequence length, H/W
  - unpack_impl (lut by default)
- timing:
  - CPU: time.perf_counter
  - CUDA: torch.cuda.Event for accurate timings
- output:
  - ms per step, pixels/s, and effective GB/s
  - optional JSON dump under bench/runs/

Acceptance threshold:
- Set a soft threshold based on the first run and assert no regression > 20%.

---

## 10) Risks and mitigations

- KL scaling mismatch (common Dreamer v3 pitfall)
  - Mitigation: explicit beta_dyn/beta_rep config + fixture parity.
- Wrong continue target semantics
  - Mitigation: test continue head with 0/1 only; document that continue != gamma.
- Symlog applied to pixels
  - Mitigation: encoder tests and fixture parity.
- Reward twohot bins mismatch (low/high or bin count)
  - Mitigation: capture bins from sheeprl and fixture parity.
- Packed2 decode bottleneck
  - Mitigation: micro-benchmark + unpack_impl hook for future kernels.

---

## 11) Done checklist

- [ ] New world model modules compiled and imported cleanly.
- [ ] Loss function implements stop-grad KL, free-nats clamp, fixed weights, and kl_regularizer.
- [ ] Continue head uses strict 0/1 targets.
- [ ] Pixel normalization is [-0.5, 0.5] for packed2 inputs.
- [ ] Fixtures generated and parity tests pass.
- [ ] Overfit-one-batch test shows decreasing recon loss.
- [ ] Packed2 micro-benchmark script exists and runs on CPU (CUDA optional).
