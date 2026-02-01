# T7 Plan: Model supports senses reconstruction (MLP decoder + wiring)

## Findings (current code)
- Decoders live in `src/gbxcule/rl/dreamer_v3/decoders.py`:
  - `CNNDecoder` (image recon), `MLPDecoder` (vector recon), and `MultiDecoder` (merges outputs).
- `MultiDecoder` is already implemented in `src/gbxcule/rl/dreamer_v3/decoders.py` and merges dict outputs from CNN/MLP decoders.
- `WorldModel` is in `src/gbxcule/rl/dreamer_v3/world_model.py` and already accepts `cnn_keys` and `mlp_keys`.
  - `prepare_obs()` only transforms `cnn_keys` (packed2 -> float); non-CNN keys pass through unchanged.
  - `build_distributions()` uses `MSEDistribution` for `cnn_keys` and `SymlogDistribution` for `mlp_keys`.
- `SymlogDistribution` is defined in `src/gbxcule/rl/dreamer_v3/dists.py` and is only used in `WorldModel.build_distributions()` for `mlp_keys`.
- Current call sites (e.g. `tools/rl_train_gpu.py`, `tools/rl_dreamer_visualize.py`) instantiate `MultiDecoder(decoder, None)` and pass `mlp_keys=[]`.

## Implementation plan

### 1) MLPDecoder availability
- **Status:** `MLPDecoder` already exists.
- **Action:** no new decoder implementation needed; use the existing `MLPDecoder` for `keys=["senses"]` with `output_dims=[senses_dim]`.
- **Note:** the decoder handles `[T, B, latent]` by flattening time/batch and restoring shapes. It emits `[T, B, senses_dim]`, which matches `SymlogDistribution(dims=len(recon.shape[2:]))`.

### 2) MultiDecoder integration (CNN + MLP)
- Instantiate both decoders and pass to `MultiDecoder(cnn_decoder, mlp_decoder)`.
- Ensure keys are disjoint: `cnn_decoder.keys=["pixels"]`, `mlp_decoder.keys=["senses"]`.
- Confirm `MultiDecoder.forward()` returns both keys and raises if neither decoder is present.

### 3) WorldModel wiring for `mlp_keys=["senses"]`
- Wire `WorldModel(cnn_keys=["pixels"], mlp_keys=["senses"])`.
- Ensure `prepare_obs()` is called with `obs_format="packed2"` so pixels are unpacked; `senses` stays float32 and unscaled.
- Verify `build_distributions()` creates `SymlogDistribution` for `senses` and uses its `log_prob()` in reconstruction loss.
- No `WorldModel` code changes expected unless `senses` arrives as non-float (if so, add float casting for `mlp_keys` in `prepare_obs()`).

### 4) Tests with dummy dict obs sequences
Add a new model test or extend existing ones to cover dual-modality reconstruction:

**Test setup (dummy data):**
- `obs = {"pixels": uint8 packed2 [T,B,1,H,W_bytes], "senses": float32 [T,B,senses_dim]}`.
- `actions`: one-hot or float [T,B,action_dim].
- `is_first`: float [T,B,1].

**Model setup:**
- Encoder: `MultiEncoder(Packed2PixelEncoder(keys=["pixels"]), MLPEncoder(keys=["senses"], input_dims=[senses_dim]))`.
- Decoder: `MultiDecoder(CNNDecoder(...keys=["pixels"]) , MLPDecoder(...keys=["senses"], output_dims=[senses_dim]))`.
- `WorldModel(..., cnn_keys=["pixels"], mlp_keys=["senses"])`.

**Assertions:**
- `outputs.reconstructions["pixels"].shape == (T,B,1,H,W)`
- `outputs.reconstructions["senses"].shape == (T,B,senses_dim)`
- `model.build_distributions(outputs)` returns a `SymlogDistribution` for `senses`.
- `model.loss(...)` returns finite values.

**Suggested placement:**
- New test file: `tests/rl_dreamer/test_world_model_senses.py`, or extend `tests/rl_dreamer/test_world_model_shapes.py` with a second test for `senses`.

**Test command:**
- `uv run pytest tests/rl_dreamer/test_world_model_shapes.py -k senses` (or the new test file).

## Done criteria
- `MLPDecoder` is used for `senses` and returns correct shapes for `[T,B,latent]` inputs.
- `MultiDecoder` returns both `pixels` and `senses` reconstructions when both decoders are provided.
- `WorldModel` is instantiated with `mlp_keys=["senses"]` and produces `SymlogDistribution` for `senses` in `build_distributions()`.
- A unit-style test with dummy dict obs sequences passes and verifies dual-modality reconstruction + finite loss.
