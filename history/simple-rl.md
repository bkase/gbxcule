## GBxCuLE → GPU-only RL from Pixels (Pokémon Red, 24 frames/step, PyTorch)

# 0) Non-negotiable constraints & design stance

### Hard constraints

1. **Everything on GPU**: env stepping (Warp/CUDA) + policy/value forward/backward (PyTorch CUDA).
2. **No RAM reads for the policy** (and no handcrafted RAM features in the observation).
3. **24 frames per step** (action frequency).
4. **Minimal + incrementally testable**: every milestone has a fast, deterministic gate.

### Practical interpretation of “no RAM”

- The **policy never sees RAM**.
- The emulator/renderer _obviously uses VRAM/OAM/IO registers internally_ to produce the screen (that’s still “eyes,” not privileged features).

---

# 1) Observation: “eyes” pipeline (pixels-on-GPU)

## 1.1 What pixels do we use?

Use a **4-shade** screen representation (0..3) consistent with your existing `read_frame_bg_shade_env0` / harness quantization logic.

- Source resolution: **160×144**
- Training resolution: **84×84** (classic Atari-style) or **80×72** (exact /2 in each axis is 80×72).
  - I recommend **80×72** first because it’s trivial downsample and maps cleanly from 160×144.

Observation tensor layout for the policy:

- `obs_pixels: uint8[N, H, W]` on GPU
- Convert to `float32/float16` in PyTorch: `x = obs_pixels.float() / 3.0`

## 1.2 Frame stacking (still “eyes,” but makes it learnable)

Single frames can be partially observable (motion, direction, transitions). Minimal fix:

- Keep a **stack of the last 4 frames**: `uint8[N, 4, H, W]`

This is easy and avoids needing an LSTM.

## 1.3 Where do these pixels come from in your codebase?

Right now you have:

- A working env0 renderer path:
  - scanline latch capture in the mega-kernel
  - `ppu_render_bg_env0` kernel that writes `frame_bg_shade_env0`
  - backend methods to read env0 frames (CPU and CUDA)

But training requires **N-env frames**, not only env0.

### New kernel you will add: `ppu_render_shades_downsampled_all_envs`

- Runs on GPU.
- Produces **downsampled shade images for all envs**:
  - `out: uint8[num_envs * out_h * out_w]`

**Key design choice:** do _not_ try to reuse env0 scanline latch arrays for all envs initially. That would require `num_envs × SCREEN_H` latch buffers for many regs and will balloon memory + complexity.

Instead, do the minimal viable renderer:

- Read PPU regs from `mem` at end-of-step (`LCDC/SCX/SCY/BGP/WX/WY/OBP0/OBP1`, etc.)
- Render a single “snapshot” frame (BG + window + sprites) using the same tile logic you already have in `ppu_render_bg_env0`, but generalized to env `i`.
- It won’t be perfectly scanline-accurate, but it’s consistent and “eye-like,” and you can validate it against PyBoy for the handful of screens you care about.

---

# 2) Goal definition & reward from pixels only

## 2.1 Goal: “this screen looks like the target house interior”

Workflow:

1. You (once) navigate to the target house interior (human/scripted).
2. Capture the downsampled shade frame stack (or just last frame).
3. Save as `goal_template.npy` (uint8[H,W]) or `uint8[4,H,W]`.

During training:

- Compute a similarity metric between current frame (or stack) and goal template.
- Trigger `done=True` when similarity crosses a threshold (optionally for K consecutive steps).

## 2.2 Similarity metric (GPU friendly)

Let `f_t` be current `uint8[H,W]`, `g` the goal template.

Define distance:

- `dist = mean(|f_t - g|) / 3.0` (normalized L1)
  or
- `dist = mean((f_t - g)^2) / 9.0` (normalized MSE)

Both are simple elementwise ops in torch and stay on GPU.

**Goal condition:**

- `done = (dist < τ)` for `K` consecutive steps
  - start with `τ ≈ 0.03–0.06`, `K=2`

This handles fade/transition frames.

## 2.3 Reward (dense enough to learn, still respects “speed”)

We want “fewest frames,” so the reward should have:

- a step cost
- a dense progress term from similarity improvement
- a terminal bonus when the goal matches

Proposed per-step reward:

- `r = step_cost + α * (dist_{t-1} - dist_t)`
- If `done`: `r += goal_bonus`

Defaults (reasonable first pass):

- `step_cost = -0.01` (per _step_, i.e., per 24 frames)
- `α = 1.0`
- `goal_bonus = +5.0` to `+20.0`

Because you care about speed, you can also scale step cost by frames:

- `step_cost = -0.01 * (frames_per_step / 24)` (keeps semantics stable if you change the quantum)

All of this is computed from pixels only, on GPU, in torch.

---

# 3) Minimal policy network architecture (pixels) — Question 1

### Input

- `x: float16/float32 [N, 4, 72, 80]` (stack of 4 shade frames)

### Minimal CNN (Atari-ish, but slightly smaller)

**Trunk**

- Conv2d(4 → 32, kernel=8, stride=4), ReLU
- Conv2d(32 → 64, kernel=4, stride=2), ReLU
- Conv2d(64 → 64, kernel=3, stride=1), ReLU
- Flatten
- Linear(… → 512), ReLU

**Heads**

- Policy logits: Linear(512 → num_actions)
- Value: Linear(512 → 1)

That’s the smallest “reliably works” pixel-policy template.

---

# 4) RL algorithm choice — Question 4

### Recommendation: PPO-lite (still minimal, much more stable than vanilla A2C)

Why:

- Clipped objective is robust to reward scaling and helps with sparse-ish goals.
- Implementation is still small.

**Batching**

- `N` parallel envs on GPU (start 256–2048; scale up once stable)
- rollout length `T=128` steps
- one PPO update per rollout, `epochs=2`, minibatch optional (or do full-batch to start)

---

# 5) Warp ↔ PyTorch bridge (zero-copy, stream-correct) — Question 3

Warp already supports:

- Passing external arrays (PyTorch tensors) directly to kernels via CUDA array interfaces. ([NVIDIA GitHub][1])
- `wp.to_torch()` / `wp.from_torch()` zero-copy conversions. ([NVIDIA GitHub][1])
- Stream interop via `warp.stream_from_torch()` / `warp.stream_to_torch()`. ([NVIDIA GitHub][1])
- Warp launches on CUDA are asynchronous; ordering is stream-based. ([NVIDIA GitHub][2])

### Concrete bridge design

Add to `WarpVecCudaBackend` (or a thin RL wrapper around it):

1. Allocate a device buffer for pixels:

- `self._pix = wp.zeros(num_envs*out_h*out_w, dtype=wp.uint8, device=cuda)`

1. Expose torch views:

- `pix_t = wp.to_torch(self._pix).view(num_envs, out_h, out_w)` (uint8)
- Maintain a torch-side frame stack ring buffer:
  - `stack_t: uint8[num_envs, 4, out_h, out_w]`
  - each step: shift/roll + write latest frame

1. Step function uses the **torch current stream**:

- `with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):`
  - launch the mega-kernel (step)
  - launch the renderer kernel (fill `self._pix`)

- no global synchronize; everything stays ordered on that stream. ([NVIDIA GitHub][2])

1. Actions remain on GPU:

- sample actions in torch → `actions_t: int64/int32 [N]`
- pass `actions_t` directly into `wp.launch(...)` (no host copy). ([NVIDIA GitHub][1])

---

# 6) Environment reset: start state + per-episode resets (still pixel-only policy)

Even with pixel-only observations, we still need to start from the same saved Pokémon state.

### ResetCache (GPU)

- Load `.state` once (env0), snapshot all relevant arrays (mem, regs, timers, cart state, etc.)
- Store snapshot in device buffers (or pinned host + device copy)
- Provide `reset_mask(done_mask)`:
  - Warp kernel copies snapshot into envs where done/trunc=1

This is necessary for vectorized PPO to train efficiently.

---

# 7) Concrete implementation plan (milestones + workstreams) — Question 5

## Workstreams

- **WS-A: Pixels & renderer (GPU)**
- **WS-B: Torch bridge & stream correctness**
- **WS-C: ResetCache & episode management**
- **WS-D: Goal template + reward/done from pixels**
- **WS-E: PPO-lite training loop + eval**

---

## Milestone 0 — “Single-env pixel smoke”

**Goal:** prove we can generate a meaningful frame buffer for env0 every step.

- Reuse existing env0 path:
  - `render_bg=True` in `warp_vec_cuda` already triggers env0 latch capture + env0 render kernel.

- Add a tiny script:
  - reset from your `.state`
  - take random actions
  - read `read_frame_bg_shade_env0()` every step (debug-only)

**Acceptance**

- You can visually save PNGs (from shades) and they look like Oak’s lab / transitions.

---

## Milestone 1 — “Multi-env downsampled pixel buffer on GPU”

**Goal:** produce `pix[N,72,80]` on GPU, no host copies in the hot path.

- Implemented kernel: `src/gbxcule/kernels/ppu_render_downsampled.py` (`ppu_render_shades_downsampled_all_envs`):
  - `@wp.kernel` with `tid` over `num_envs*out_h*out_w`
  - compute `(env, oy, ox)` from tid
  - map to `(x,y)` in 160×144
  - read PPU regs from `mem[base + 0xFFxx]`
  - render shade (BG/window/sprites) → write to `out`

- Backend wiring (CPU + CUDA): `render_pixels=True` allocates `self._pix` and launches the renderer after the step kernel.
  - Accessors: `pixels_wp()` (Warp array) and `pixels_torch()` (stable torch view).
  - Dims are centralized in `gbxcule.core.abi`: `DOWNSAMPLE_H=72`, `DOWNSAMPLE_W=80`.

**Acceptance**

- `pix_t = wp.to_torch(self._pix)` updates every step without calling `.cpu()` (see `pixels_torch()`).
- For env0, downsampled renderer roughly matches env0 accurate renderer (sanity gate in `tests/test_ppu_downsampled_pixels.py`).

---

## Milestone 2 — “Torch-only env wrapper (no reward in Warp yet)”

**Goal:** end-to-end stepping loop: pixels → policy → actions → pixels.

- Implement `gbxcule/rl/pokered_pixels_env.py`:
  - wraps `WarpVecCudaBackend`
  - manages:
    - frame stack buffer `stack_t`
    - per-env `prev_dist`, `consec_match`, `episode_step`
    - `done_mask` + `reset_mask`

**Acceptance**

- Deterministic smoke: with fixed seed and fixed action trace, the same pixel hashes repeat.

---

## Milestone 3 — “Goal template capture + done detection”

**Goal:** define success purely by pixels.

- Add `gbxcule/rl/capture_goal_template.py`:
  - run `pyboy_single` (or warp env0) with a scripted/human-recorded action trace to enter the target house
  - capture the downsampled goal frame
  - save `goal_template.npy`

- Implement goal detection in torch:
  - compute `dist`
  - `done = dist < τ` for K consecutive steps

**Acceptance**

- Replay the recorded goal-reaching trace in the GPU env; `done` triggers reliably near the end.

---

## Milestone 4 — “Reward shaping from pixels + reset_mask”

**Goal:** produce stable learning signal without RAM.

- Add reward:
  - `r = -0.01 + α*(prev_dist - dist) + goal_bonus*done`

- Reset:
  - `reset_mask(done | trunc)` copies ResetCache into those envs

- Add truncation:
  - `episode_step >= max_steps` → `trunc=1`

**Acceptance**

- In a short run, you see non-zero reward variation and episodes recycle correctly.

---

## Milestone 5 — “PPO-lite training on GPU”

**Goal:** agent learns to reach the target screen faster (more often).

- Implement PPO-lite in `gbxcule/rl/ppo_pixels_torch.py`
  - store rollout tensors on GPU:
    - obs (stack), actions, rewards, dones, values, logprobs

  - compute GAE advantages
  - PPO update

- Add evaluation mode:
  - run `E` eval episodes with greedy policy
  - report success rate + median steps to goal

**Acceptance**

- Success rate increases above random baseline.
- Median steps-to-goal decreases over training.

---

# 8) Test plan (gated per milestone) — Question 6

## Tests for Milestone 1 (renderer)

1. **Renderer determinism**
   - Same start state + same action trace → identical pixel hash sequence.

2. **Renderer sanity vs env0 “accurate”**
   - Compare downsampled env0 from the multi-env renderer vs env0 latch-based renderer on a few fixed frames (not perfect, but close enough threshold).

## Tests for Milestone 2 (torch bridge + streams)

1. **Zero-copy proof**
   - `pix_t.data_ptr()` stable across steps, but contents change (no re-wrap).

2. **Stream correctness**
   - step → immediately run a torch op using `pix_t` on same stream; ensure no stale reads / race (no global sync). ([NVIDIA GitHub][2])

## Tests for Milestone 3 (goal detection)

1. **Goal replay triggers done**
   - Run known “reach goal” trace; done triggers at expected window.

## Tests for Milestone 4 (episode mechanics)

1. **reset_mask correctness**
   - Mark half envs done; after reset_mask those envs’ first-frame hashes equal the start state; others unchanged.

## Tests for Milestone 5 (RL smoke)

1. **Gradient smoke**
   - One PPO update: finite loss, finite grads, params change.

2. **Mini learning regression (short)**
   - fixed seed, small N, few updates → success rate > random baseline.

---

# 9) Minimal defaults (first working config)

- `frames_per_step = 24`
- `release_after_frames = 8`
- Downsample: `H=72, W=80`
- Stack: `K=4`
- PPO:
  - `N = 1024` (start), scale later
  - `T = 128`
  - `lr = 2.5e-4`
  - `gamma=0.99`, `gae_lambda=0.95`
  - `clip=0.1`
  - `entropy_coef=0.01`
  - `value_coef=0.5`

- Reward:
  - `step_cost=-0.01`
  - `α=1.0`
  - `goal_bonus=10.0`
  - `τ=0.05`, `K=2 consecutive matches`

---

# 10) Key risks & mitigations

### Risk A: multi-env renderer not visually faithful enough

Mitigation:

- Treat the downsampled renderer as the “camera” and ensure it is **self-consistent**, deterministic, and correlates with the important screens.
- Use env0 latch-based renderer only for **debugging/validation**, not training.

### Risk B: goal template overfits / false positives

Mitigation:

- Use consecutive match K>1
- Use a small set of goal templates (e.g., 3 frames after entering house) and require min distance across them.

---

If you want the plan to be _even more “eyes-only interesting”_ without making it much bigger, the cleanest next step after PPO works is to add an intrinsic reward like **RND** (still pixels-only, still GPU), but I kept it out of the core plan to keep the first version minimal.

[1]: https://nvidia.github.io/warp/user_guide/interoperability.html "Interoperability — Warp 1.11.0"
[2]: https://nvidia.github.io/warp/deep_dive/concurrency.html "Concurrency — Warp 1.11.0"
