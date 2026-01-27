"""Streaming A2C (TD(0)) trainer for pixels-only Pokémon Red (CUDA).

JSONL schema (train_log.jsonl):
  - First line: {"meta": <TrainConfig as dict>, "torch_version": str,
                 "warp_version": str|None}
  - Per-optimizer-step record:
      {
        "opt_step": int,
        "env_steps": int,
        "loss_total": float,
        "loss_policy": float,
        "loss_value": float,
        "loss_entropy": float,
        "entropy": float,
        "reward_mean": float,
        "done_rate": float,
        "trunc_rate": float,
        "reset_rate": float,
        "sps": int,
        "accum_steps": int
      }
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _require_torch():
    import importlib

    return importlib.import_module("torch")


def _cuda_available() -> bool:
    if os.environ.get("GBXCULE_SKIP_CUDA") == "1":
        return False
    try:
        import warp as wp

        wp.init()
        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


def _atomic_save(path: Path, payload: dict[str, Any]) -> None:
    torch = _require_torch()
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


@dataclass(frozen=True)
class TrainConfig:
    rom: str
    state: str
    goal_dir: str
    frames_per_step: int
    release_after_frames: int
    num_envs: int
    stack_k: int
    action_codec: str
    max_steps: int
    step_cost: float
    alpha: float
    goal_bonus: float
    tau: float | None
    k_consecutive: int | None
    info_mode: str
    lr: float
    gamma: float
    value_coef: float
    entropy_coef: float
    grad_clip: float
    update_every: int
    total_env_steps: int
    checkpoint_every_opt_steps: int
    seed: int
    output_dir: str

    def validate(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if self.stack_k < 1:
            raise ValueError("stack_k must be >= 1")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.update_every < 1:
            raise ValueError("update_every must be >= 1")
        if self.total_env_steps < self.num_envs:
            raise ValueError("total_env_steps must be >= num_envs")
        if not (0.0 < self.gamma <= 1.0):
            raise ValueError("gamma must be in (0, 1]")
        if self.lr <= 0.0:
            raise ValueError("lr must be > 0")
        if self.value_coef < 0.0:
            raise ValueError("value_coef must be >= 0")
        if self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be >= 0")
        if self.grad_clip <= 0.0:
            raise ValueError("grad_clip must be > 0")
        if self.info_mode not in {"full", "stats", "none"}:
            raise ValueError("info_mode must be full, stats, or none")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--state", required=True, help="Path to .state file")
    parser.add_argument("--goal-dir", required=True, help="Goal template directory")
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--stack-k", type=int, default=1)
    parser.add_argument(
        "--action-codec", default="pokemonred_puffer_v1", help="Action codec id"
    )
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--step-cost", type=float, default=-0.01)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--goal-bonus", type=float, default=10.0)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--k-consecutive", type=int, default=None)
    parser.add_argument(
        "--info-mode", default="stats", choices=("full", "stats", "none")
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--update-every", type=int, default=4)
    parser.add_argument("--total-env-steps", type=int, default=1_000_000)
    parser.add_argument("--checkpoint-every-opt-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for logs/checkpoints (default: bench/runs/rl_m5_a2c/<ts>)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def _torch_versions() -> tuple[str, str | None]:
    torch = _require_torch()
    warp_version = None
    try:
        import warp as wp

        warp_version = getattr(wp, "__version__", None)
    except Exception:
        warp_version = None
    return str(torch.__version__), warp_version


def main() -> int:
    args = _parse_args()
    if args.self_test:
        return _run_self_test()
    if not _cuda_available():
        print(json.dumps({"skipped": "CUDA not available"}))
        return 0
    torch = _require_torch()
    if not torch.cuda.is_available():
        print(json.dumps({"skipped": "torch CUDA not available"}))
        return 0

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("bench/runs/rl_m5_a2c") / time.strftime("%Y%m%d_%H%M%S")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    ckpt_path = output_dir / "checkpoint.pt"

    cfg = TrainConfig(
        rom=str(Path(args.rom)),
        state=str(Path(args.state)),
        goal_dir=str(Path(args.goal_dir)),
        frames_per_step=int(args.frames_per_step),
        release_after_frames=int(args.release_after_frames),
        num_envs=int(args.num_envs),
        stack_k=int(args.stack_k),
        action_codec=str(args.action_codec),
        max_steps=int(args.max_steps),
        step_cost=float(args.step_cost),
        alpha=float(args.alpha),
        goal_bonus=float(args.goal_bonus),
        tau=float(args.tau) if args.tau is not None else None,
        k_consecutive=int(args.k_consecutive)
        if args.k_consecutive is not None
        else None,
        info_mode=str(args.info_mode),
        lr=float(args.lr),
        gamma=float(args.gamma),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        grad_clip=float(args.grad_clip),
        update_every=int(args.update_every),
        total_env_steps=int(args.total_env_steps),
        checkpoint_every_opt_steps=int(args.checkpoint_every_opt_steps),
        seed=int(args.seed),
        output_dir=str(output_dir),
    )
    cfg.validate()

    from gbxcule.rl.a2c import a2c_td0_losses
    from gbxcule.rl.models import PixelActorCriticCNN
    from gbxcule.rl.pokered_pixels_goal_env import PokeredPixelsGoalEnv

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    env = PokeredPixelsGoalEnv(
        cfg.rom,
        state_path=cfg.state,
        goal_dir=cfg.goal_dir,
        num_envs=cfg.num_envs,
        frames_per_step=cfg.frames_per_step,
        release_after_frames=cfg.release_after_frames,
        stack_k=cfg.stack_k,
        action_codec=cfg.action_codec,
        max_steps=cfg.max_steps,
        step_cost=cfg.step_cost,
        alpha=cfg.alpha,
        goal_bonus=cfg.goal_bonus,
        tau=cfg.tau,
        k_consecutive=cfg.k_consecutive,
        info_mode=cfg.info_mode,
    )

    model = PixelActorCriticCNN(
        num_actions=env.backend.num_actions, in_frames=env.stack_k
    )
    model.to(device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    start_env_steps = 0
    start_opt_steps = 0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_env_steps = int(ckpt.get("env_steps", 0))
        start_opt_steps = int(ckpt.get("opt_steps", 0))
        rng_state = ckpt.get("rng_state")
        cuda_rng_state = ckpt.get("cuda_rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    obs = env.reset(seed=cfg.seed)
    env_steps = start_env_steps
    opt_steps = start_opt_steps

    torch_version, warp_version = _torch_versions()
    with log_path.open("a", encoding="utf-8") as log_f:
        if opt_steps == 0 and env_steps == 0:
            log_f.write(
                json.dumps(
                    {
                        "meta": asdict(cfg),
                        "torch_version": torch_version,
                        "warp_version": warp_version,
                    }
                )
                + "\n"
            )

        optimizer.zero_grad(set_to_none=True)
        window_start = time.time()
        accum_steps = 0
        reward_sum = 0.0
        done_sum = 0.0
        trunc_sum = 0.0
        reset_sum = 0.0
        info_dist_mean = None

        while env_steps < cfg.total_env_steps:
            logits, values = model(obs)
            actions_i64 = torch.multinomial(
                torch.softmax(logits, dim=-1), num_samples=1
            ).squeeze(1)
            actions = actions_i64.to(torch.int32)
            next_obs, reward, done, trunc, info = env.step(actions)

            with torch.no_grad():
                _, v_next = model(next_obs)

            losses = a2c_td0_losses(
                logits,
                actions_i64,
                values,
                reward,
                done,
                trunc,
                v_next,
                gamma=cfg.gamma,
                value_coef=cfg.value_coef,
                entropy_coef=cfg.entropy_coef,
            )

            (losses["loss_total"] / float(cfg.update_every)).backward()
            accum_steps += 1
            env_steps += cfg.num_envs

            reward_sum += float(reward.mean().item())
            done_sum += float(done.to(torch.float32).mean().item())
            trunc_sum += float(trunc.to(torch.float32).mean().item())
            reset_sum += float((done | trunc).to(torch.float32).mean().item())
            if isinstance(info, dict) and "dist_mean" in info:
                info_dist_mean = float(info["dist_mean"])

            if accum_steps >= cfg.update_every:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=cfg.grad_clip
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_steps += 1

                elapsed = max(1e-6, time.time() - window_start)
                sps = int(cfg.num_envs * accum_steps / elapsed)
                record = {
                    "opt_step": opt_steps,
                    "env_steps": env_steps,
                    "loss_total": float(losses["loss_total"].item()),
                    "loss_policy": float(losses["loss_policy"].item()),
                    "loss_value": float(losses["loss_value"].item()),
                    "loss_entropy": float(losses["loss_entropy"].item()),
                    "entropy": float(losses["entropy"].item()),
                    "reward_mean": reward_sum / float(accum_steps),
                    "done_rate": done_sum / float(accum_steps),
                    "trunc_rate": trunc_sum / float(accum_steps),
                    "reset_rate": reset_sum / float(accum_steps),
                    "sps": sps,
                    "accum_steps": accum_steps,
                }
                if info_dist_mean is not None:
                    record["dist_mean"] = info_dist_mean
                log_f.write(json.dumps(record) + "\n")
                log_f.flush()

                if cfg.checkpoint_every_opt_steps > 0 and (
                    opt_steps % cfg.checkpoint_every_opt_steps == 0
                ):
                    _atomic_save(
                        ckpt_path,
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "config": asdict(cfg),
                            "env_steps": env_steps,
                            "opt_steps": opt_steps,
                            "rng_state": torch.get_rng_state(),
                            "cuda_rng_state": torch.cuda.get_rng_state_all(),
                        },
                    )

                window_start = time.time()
                accum_steps = 0
                reward_sum = 0.0
                done_sum = 0.0
                trunc_sum = 0.0
                reset_sum = 0.0
                info_dist_mean = None

            obs = next_obs

    _atomic_save(
        ckpt_path,
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": asdict(cfg),
            "env_steps": env_steps,
            "opt_steps": opt_steps,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        },
    )
    env.close()
    return 0


def _run_self_test() -> int:
    torch = _require_torch()
    from gbxcule.core.abi import DOWNSAMPLE_H, DOWNSAMPLE_W
    from gbxcule.rl.a2c import a2c_td0_losses
    from gbxcule.rl.models import PixelActorCriticCNN

    class _ToyPixelsEnv:
        def __init__(self, num_envs: int, stack_k: int) -> None:
            self.num_envs = num_envs
            self.stack_k = stack_k
            self._step = torch.zeros((num_envs,), dtype=torch.int32)
            self._obs = torch.zeros(
                (num_envs, stack_k, DOWNSAMPLE_H, DOWNSAMPLE_W), dtype=torch.uint8
            )

        def reset(self):  # type: ignore[no-untyped-def]
            self._step.zero_()
            self._obs.zero_()
            return self._obs

        def step(self, actions):  # type: ignore[no-untyped-def]
            self._step.add_(1)
            self._obs[:, :-1].copy_(self._obs[:, 1:])
            fill = (actions % 4).to(torch.uint8).view(self.num_envs, 1, 1)
            self._obs[:, -1].copy_(fill.expand(-1, DOWNSAMPLE_H, DOWNSAMPLE_W))
            reward = self._obs[:, -1].to(torch.float32).mean(dim=(1, 2)) / 3.0
            done = self._step >= 2
            trunc = self._step >= 3
            return self._obs, reward, done, trunc, {}

    try:
        torch.manual_seed(0)
        env = _ToyPixelsEnv(num_envs=4, stack_k=2)
        model = PixelActorCriticCNN(num_actions=8, in_frames=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        obs = env.reset()
        optimizer.zero_grad(set_to_none=True)
        losses = None
        for _ in range(2):
            logits, values = model(obs)
            actions_i64 = torch.multinomial(
                torch.softmax(logits, dim=-1), num_samples=1
            ).squeeze(1)
            next_obs, reward, done, trunc, _ = env.step(actions_i64.to(torch.int32))
            with torch.no_grad():
                _, v_next = model(next_obs)
            losses = a2c_td0_losses(
                logits,
                actions_i64,
                values,
                reward,
                done,
                trunc,
                v_next,
                gamma=0.99,
                value_coef=0.5,
                entropy_coef=0.01,
            )
            (losses["loss_total"] / 2.0).backward()
            obs = next_obs

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        if losses is None:
            raise RuntimeError("Self-test failed to produce losses")
        print(
            json.dumps(
                {
                    "self_test": "ok",
                    "loss_total": float(losses["loss_total"].item()),
                    "entropy": float(losses["entropy"].item()),
                }
            )
        )
        return 0
    except Exception as exc:
        print(json.dumps({"self_test": "failed", "error": str(exc)}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
