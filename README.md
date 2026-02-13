# GBxCuLE

A GPU-accelerated vectorized Game Boy emulator for reinforcement learning. Emulation cores run fully on the GPU.

Adapting [CuLE: GPU-Accelerated Atari Emulation for Reinforcement Learning](https://arxiv.org/abs/1907.08467) (Dalton et al, NVLabs) for Game Boy instead of Atari.

**24 million FPS. 400,000x realtime. >3x faster than SOTA. On a home GPU.**

Pokemon Red complete. Built in 5 days with a cybernetic forge (LLM-assisted development with strong automated verification). Read more in [the blog post](https://bkase.dev/posts/cybernetic-forge).

## Performance

On an NVIDIA DGX Spark [20-core ARM, GB10 GPU], initializing from a Tetris savestate:

1. Advances 24 frames per step (frame skip = 24)
2. Extracts the observation tensor
3. Samples a button-press action (random policy)
4. Applies the action

No cheating: the GPU executes divergent instruction streams across warp lanes via distinct joypad inputs.

| Backend           | Best SPS   | Env Count |
| ----------------- | ---------- | --------- |
| **CUDA Warp**     | **17,200** | 16,384    |
| PyBoy + PufferLib | 5,366      | 64        |

Untuned, on a dev box. Datacenter GPUs will be faster.

## How It Works

The system compiles a Game Boy emulator into a single monolithic [NVIDIA Warp](https://nvidia.github.io/warp/) kernel. Each GPU thread simulates one complete Game Boy. At 16,384 environments, that's 16,384 Game Boys running in parallel on a single GPU.

The architecture follows a **functional core / imperative shell** pattern:

```
src/gbxcule/
├── core/           # Pure logic: ABI layouts, ISA definitions, action codecs
├── kernels/        # Warp GPU/CPU kernels: CPU step, PPU render
│   └── cpu_templates/  # Instruction templates (ALU, loads, jumps, etc.)
├── backends/       # Environment implementations (PyBoy oracle, Warp CPU/CUDA)
└── rl/             # RL algorithms and Pokemon Red environments
```

### The Warp Template System

This is the core innovation that makes the project tractable. Game Boy CPU emulation requires implementing ~500 opcodes. Rather than hand-writing CUDA, we use a **template-based code generation pipeline**:

**1. ISA definitions** (`core/isa_sm83.py`) declare each opcode with a template key and placeholder mappings:

```python
# Opcode 0x3E: LD A,d8
OpcodeSpec(opcode=0x3E, template_key="ld_r8_d8", replacements={"REG_i": "a_i"})

# Opcode 0x06: LD B,d8  -- same template, different register
OpcodeSpec(opcode=0x06, template_key="ld_r8_d8", replacements={"REG_i": "b_i"})
```

**2. Templates** (`kernels/cpu_templates/`) are plain Python functions using Warp primitives. Placeholders like `REG_i` get substituted at build time:

```python
def template_inc_r8(pc_i: int, f_i: int, REG_i: int) -> None:
    old = REG_i
    REG_i = (REG_i + 1) & 0xFF
    z = wp.where(REG_i == 0, 1, 0)
    hflag = wp.where((old & 0x0F) == 0x0F, 1, 0)
    cflag = (f_i >> 4) & 0x1
    f_i = make_flags(z, 0, hflag, cflag)
    pc_i = (pc_i + 1) & 0xFFFF
    cycles = 4
```

Note `wp.where()` instead of `if/else` for branchless GPU-friendly code.

**3. LibCST code generation** (`kernels/cpu_step_builder.py`) performs AST-level transformations:

- Specializes templates by renaming placeholders (e.g. `REG_i` -> `a_i`)
- Builds a two-level bucketed dispatch tree (bucket by high nibble, then linear scan)
- Injects the dispatch tree into the kernel skeleton
- SHA256-hashes the result and caches the generated module

The final output is a single `@wp.kernel` function containing all ~500 opcodes, which Warp JIT-compiles to CUDA PTX. The generated kernel is cached at `~/.cache/gbxcule/warp_kernels/`.

### Micro-ROM Test Harness

Correctness is verified against [PyBoy](https://github.com/Baekalfen/PyBoy) as a trusted oracle. The system generates 23 license-safe micro-ROMs that exercise specific emulator features:

| Category     | ROMs                                               | What they test                    |
| ------------ | -------------------------------------------------- | --------------------------------- |
| CPU/ALU      | ALU_LOOP, ALU_FLAGS, ALU16_SP, CB_BITOPS           | Arithmetic, flags, bit operations |
| Memory       | MEM_RWB, LOADS_BASIC                               | Read/write, load/store variants   |
| Control flow | FLOW_STACK                                         | CALL/RET/JR/JP, PUSH/POP          |
| Interrupts   | TIMER_DIV_BASIC, TIMER_IRQ_HALT, EI_DELAY          | Timer, HALT wake, EI delay        |
| MBC          | MBC1_SWITCH, MBC3_SWITCH, MBC1_RAM, MBC3_RAM       | Bank switching, cartridge RAM     |
| PPU          | BG_STATIC, BG_SCROLL_ANIM, PPU_WINDOW, PPU_SPRITES | Tiles, scrolling, window, sprites |
| Divergence   | JOY_DIVERGE_PERSIST                                | Joypad-driven per-env divergence  |

The verification harness (`bench/harness.py`) runs the oracle and device-under-test (DUT) in lockstep, comparing CPU state at every step. On mismatch, it writes a hermetic **repro bundle** containing the ROM, action trace, both states, a diff, and a one-command `repro.sh` script.

LLM agents used these micro-ROMs during development: Codex one-shotted test ROMs in raw hex, then ran them against both the reference emulator and the DUT to find its own bugs.

### Parallel Environment Management

Each environment gets a slice of flat, strided GPU buffers:

- `mem[env_idx * 65536 .. (env_idx+1) * 65536]` — full 64KB address space per Game Boy
- Per-env CPU registers, PPU state, cartridge state as parallel arrays
- Observations and rewards computed inside the kernel (zero CPU-GPU transfer)

Episode resets use a CuLE-style **snapshot cache**: a golden state is captured once, then a Warp kernel performs masked memcpy to restore only terminated environments. This avoids disk I/O entirely during training.

Torch integration is zero-copy via `wp.to_torch()`, so the RL training loop never leaves the GPU.

## Quick Start

Requires [uv](https://docs.astral.sh/uv/) and a Game Boy ROM (not included).

```bash
git clone https://github.com/bkase/gbxcule.git && cd gbxcule
make setup    # Install dependencies
make hooks    # Install pre-commit hooks (runs make check)
make roms     # Generate micro-ROMs
```

### Verify Correctness (CPU)

```bash
make verify        # Step-exact comparison: PyBoy oracle vs Warp CPU
make verify-smoke  # Quick smoke test (frames_per_step=24)
```

### Run Benchmarks

```bash
# CPU baselines
make bench

# Tetris benchmark (requires tetris.gb ROM and CUDA GPU)
make bench-tetris-gpu

# Full scaling sweep (CUDA)
make bench-gpu
```

### GPU Verification

```bash
make verify-gpu    # PyBoy oracle vs Warp CUDA
make check-gpu     # Fast CUDA smoke test
```

### Run All Checks

```bash
make check         # Format, lint, typecheck, build ROMs, compile kernels, test
```

This is what the pre-commit hook runs. The CPU-only gate completes in under 2 minutes.

## RL Infrastructure

The emulator was built to enable GPU-native RL training on Pokemon Red. While I explored several approaches before setting the project aside, the infrastructure is functional and may be useful to others.

### What's Here

**Three RL algorithms**, all wired up to the GPU emulator:

- **PPO** (sync and async) — most developed, scales to 16k parallel environments
- **A2C** — simpler streaming variant
- **DreamerV3** — full world-model implementation (~5k lines) with RSSM, imagination rollouts, and CUDA replay buffers

**Pokemon Red environments** with quest-aware observations:

- `pokered_packed_parcel_env.py` — multi-modal observations (packed pixels + game senses + event flags), hash-based exploration bonuses, quest-specific rewards (parcel pickup/delivery), curiosity reset on key events
- `pokered_packed_goal_env.py` — goal-template matching via L1 pixel distance
- `pokered_pixels_env.py` — basic pixel observations with frame stacking

**Neural network architectures:**

- NatureCNN (Atari-style, 3 conv layers)
- IMPALA ResNet (deeper, residual blocks)
- Dual-lobe model (CNN for pixels + MLP for game senses + MLP for event flags, fused)

**A five-stage curriculum** with pre-computed savestates:

| Stage | Task                   | Max Steps |
| ----- | ---------------------- | --------- |
| 1     | Exit Oak's lab         | 128       |
| 2     | Return home (outside)  | 128       |
| 3     | Enter house            | 96        |
| 4     | Upstairs at video game | 96        |
| 5     | It's time to go        | 64        |

**Training scripts** in `tools/`:

```bash
# Sync PPO for Oak's Parcel quest (most recent, recommended starting point)
uv run python tools/rl_train_ppo_parcel.py

# Async PPO variant
uv run python tools/rl_train_async_parcel.py

# DreamerV3
uv run python tools/rl_train_gpu.py
```

### What Didn't Work (and Why)

We tried PPO, A2C, and DreamerV3 on the Oak's Parcel delivery quest. DreamerV3 could learn a world model but was sample-inefficient for this shaped problem. PPO with exploration bonuses showed promise on short navigation stages but hit walls on the full quest. Pokemon Red RL remains hard due to long credit assignment and sparse rewards.

The emulator and training infrastructure work. The RL problem is unsolved. If you're excited about Pokemon RL, this is a solid foundation — start with Stage 1 (Exit Oak's lab) and the sync PPO setup.

## Project Structure

```
gbxcule/
├── src/gbxcule/
│   ├── core/               # Pure logic (ABI, ISA, action codec, signatures)
│   ├── kernels/            # Warp kernels + template system
│   │   └── cpu_templates/  # Instruction templates (alu, loads, jumps, stack, bitops)
│   ├── backends/           # PyBoy oracle, Warp CPU/CUDA, PufferLib integrations
│   └── rl/                 # PPO, A2C, DreamerV3, Pokemon Red envs
│       └── dreamer_v3/     # Full DreamerV3 implementation
├── bench/
│   ├── harness.py          # Benchmark + verification harness
│   ├── tetris_bench.py     # Tetris-specific benchmark
│   ├── roms/               # Micro-ROM generator + suite.yaml
│   └── analysis/           # Scaling plots, summary reports
├── tests/                  # 90 test files (CPU instructions, PPU, RL algorithms)
├── tools/                  # Training scripts, profiling, capture utilities
├── states/                 # Savestates for RL stages
├── configs/                # YAML/JSON configurations
├── history/                # Planning documents (PRD, architecture, milestone plans)
├── ARCHITECTURE.md         # Engineering architecture (detailed)
└── CONSTITUTION.md         # Project principles
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — full engineering architecture and tech stack decisions
- [history/prd.md](history/prd.md) — product requirements document
- [history/](history/) — milestone plans, epic breakdowns, development history

## Acknowledgments

- [CuLE](https://arxiv.org/abs/1907.08467) (Dalton et al, NVLabs) — the original GPU-accelerated Atari emulation paper that inspired this project
- [PyBoy](https://github.com/Baekalfen/PyBoy) — trusted reference emulator used as the oracle
- [NVIDIA Warp](https://nvidia.github.io/warp/) — Python framework for JIT-compiling to CUDA kernels
- [Peter Whidden's Pokemon RL](https://www.youtube.com/watch?v=DcYLT37ImBY) and [drubinstein's Pokemon RL Experiment](https://drubinstein.github.io/pokerl/) — inspiration for the Pokemon Red RL goal
- [PufferLib](https://github.com/PufferAI/PufferLib) — vectorized environment framework used for CPU baselines
