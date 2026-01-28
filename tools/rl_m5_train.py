"""PPO-lite training loop for pixels-only Pokémon Red (CUDA).

Usage:
  uv run python tools/rl_m5_train.py --rom <rom> --state <state> --goal-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4


def _require_torch():
    import importlib

    return importlib.import_module("torch")


@dataclass(frozen=True)
class TrainConfig:
    rom: str
    state: str
    goal_dir: str
    frames_per_step: int
    release_after_frames: int
    num_envs: int
    steps_per_rollout: int
    updates: int
    lr: float
    gamma: float
    gae_lambda: float
    clip: float
    value_coef: float
    entropy_coef: float
    seed: int
    eval_every: int
    eval_episodes: int
    output_dir: str


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


def _torch_versions() -> tuple[str, str | None]:
    torch = _require_torch()
    warp_version = None
    try:
        import warp as wp

        warp_version = getattr(wp, "__version__", None)
    except Exception:
        warp_version = None
    return str(torch.__version__), warp_version


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rom", required=True, help="Path to ROM")
    parser.add_argument("--state", required=True, help="Path to .state file")
    parser.add_argument("--goal-dir", required=True, help="Goal template directory")
    parser.add_argument("--frames-per-step", type=int, default=24)
    parser.add_argument("--release-after-frames", type=int, default=8)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps-per-rollout", type=int, default=128)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for logs/checkpoints (default: bench/runs/rl_m5/<ts>)",
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
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
        else Path("bench/runs/rl_m5") / time.strftime("%Y%m%d_%H%M%S")
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
        num_envs=args.num_envs,
        steps_per_rollout=args.steps_per_rollout,
        updates=args.updates,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip=args.clip,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        output_dir=str(output_dir),
    )

    from gbxcule.rl.eval import run_greedy_eval
    from gbxcule.rl.goal_template import compute_sha256
    from gbxcule.rl.metrics import MetricsAccumulator
    from gbxcule.rl.models import PixelActorCriticCNN
    from gbxcule.rl.pokered_pixels_goal_env import PokeredPixelsGoalEnv
    from gbxcule.rl.ppo import compute_gae, logprob_from_logits, ppo_losses
    from gbxcule.rl.rollout import RolloutBuffer
    from gbxcule.rl.train_log_schema import SCHEMA_VERSION

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    env = PokeredPixelsGoalEnv(
        cfg.rom,
        state_path=cfg.state,
        goal_dir=cfg.goal_dir,
        num_envs=cfg.num_envs,
        frames_per_step=cfg.frames_per_step,
        release_after_frames=cfg.release_after_frames,
    )

    model = PixelActorCriticCNN(
        num_actions=env.backend.num_actions, in_frames=env.stack_k
    )
    model.to(device="cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    rollout = RolloutBuffer(
        steps=cfg.steps_per_rollout,
        num_envs=cfg.num_envs,
        stack_k=env.stack_k,
        device="cuda",
    )

    start_update = 0
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cuda")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_update = int(ckpt.get("update", 0))

    obs = env.reset(seed=cfg.seed)
    run_id = output_dir.name or uuid4().hex[:8]
    train_start = time.time()
    torch_version, warp_version = _torch_versions()

    goal_template_path = Path(cfg.goal_dir) / "goal_template.npy"
    if not goal_template_path.exists():
        raise FileNotFoundError(f"Goal template not found: {goal_template_path}")

    metrics = MetricsAccumulator(
        num_envs=cfg.num_envs,
        num_actions=env.backend.num_actions,
        device="cuda",
    )

    eval_env = None
    if cfg.eval_every > 0:
        eval_env = PokeredPixelsGoalEnv(
            cfg.rom,
            state_path=cfg.state,
            goal_dir=cfg.goal_dir,
            num_envs=1,
            frames_per_step=cfg.frames_per_step,
            release_after_frames=cfg.release_after_frames,
        )

    with log_path.open("a", encoding="utf-8") as log_f:
        if start_update == 0:
            log_f.write(
                json.dumps(
                    {
                        "meta": {
                            "schema_version": SCHEMA_VERSION,
                            "run_id": run_id,
                            "rom_path": cfg.rom,
                            "rom_sha256": compute_sha256(Path(cfg.rom)),
                            "state_path": cfg.state,
                            "state_sha256": compute_sha256(Path(cfg.state)),
                            "goal_dir": cfg.goal_dir,
                            "goal_sha256": compute_sha256(goal_template_path),
                            "action_codec_id": env.backend.action_codec.id,
                            "frames_per_step": cfg.frames_per_step,
                            "release_after_frames": cfg.release_after_frames,
                            "stack_k": env.stack_k,
                            "num_envs": cfg.num_envs,
                            "num_actions": env.backend.num_actions,
                            "seed": cfg.seed,
                        },
                        "config": asdict(cfg),
                        "torch_version": torch_version,
                        "warp_version": warp_version,
                    }
                )
                + "\n"
            )
        for update_idx in range(start_update, cfg.updates):
            rollout.reset()
            update_start = time.time()
            for _ in range(cfg.steps_per_rollout):
                logits, values = model(obs)
                actions_i64 = torch.multinomial(
                    torch.softmax(logits, dim=-1), num_samples=1
                ).squeeze(1)
                logprobs = logprob_from_logits(logits, actions_i64)
                actions = actions_i64.to(torch.int32)
                next_obs, reward, done, trunc, info = env.step(actions)
                rollout.add(
                    obs,
                    actions,
                    reward,
                    done | trunc,
                    values.detach(),
                    logprobs.detach(),
                )
                if isinstance(info, dict) and "dist" in info:
                    reset_mask = info.get("reset_mask", done | trunc)
                    metrics.update(
                        reward=reward,
                        done=done,
                        trunc=trunc,
                        reset_mask=reset_mask,
                        dist=info["dist"],
                        actions=actions_i64,
                        logits=logits,
                        values=values,
                    )
                obs = next_obs

            with torch.no_grad():
                _, last_value = model(obs)
            advantages, returns = compute_gae(
                rollout.rewards,
                rollout.values,
                rollout.dones,
                last_value,
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )

            batch = rollout.as_batch(flatten_obs=True)
            logits, values = model(batch["obs_u8"])
            losses = ppo_losses(
                logits,
                batch["actions"],
                batch["logprobs"],
                returns.reshape(-1),
                advantages.reshape(-1),
                values,
                clip=cfg.clip,
                value_coef=cfg.value_coef,
                entropy_coef=cfg.entropy_coef,
            )

            optimizer.zero_grad()
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            eval_metrics = {}
            if eval_env is not None and (update_idx + 1) % cfg.eval_every == 0:
                summary = run_greedy_eval(
                    eval_env,
                    model,
                    episodes=cfg.eval_episodes,
                    seed=cfg.seed,
                )
                model.train()
                eval_metrics = {
                    "eval_success_rate": summary.success_rate,
                    "eval_median_steps": summary.median_steps_to_goal,
                    "eval_mean_return": summary.mean_return,
                    "eval_steps_p50": summary.steps_p50_success,
                    "eval_return_mean_success": summary.return_mean_success,
                    "eval_return_mean_fail": summary.return_mean_fail,
                    "eval_dist_at_end_p50": summary.dist_at_end_p50,
                }

            metrics_record = metrics.as_record()
            env_steps = (update_idx + 1) * cfg.num_envs * cfg.steps_per_rollout
            record = {
                "run_id": run_id,
                "trace_id": f"{run_id}:{update_idx + 1}",
                "update": update_idx,
                "opt_steps": update_idx + 1,
                "env_steps": env_steps,
                "wall_time_s": time.time() - train_start,
                "loss_total": float(losses["loss_total"].item()),
                "loss_policy": float(losses["loss_policy"].item()),
                "loss_value": float(losses["loss_value"].item()),
                "loss_entropy": float(losses["loss_entropy"].item()),
                "entropy": float(losses["entropy"].item()),
                "approx_kl": float(losses["approx_kl"].item()),
                "clipfrac": float(losses["clipfrac"].item()),
                "sps": int(
                    cfg.num_envs
                    * cfg.steps_per_rollout
                    / max(1e-6, time.time() - update_start)
                ),
                **metrics_record,
                **eval_metrics,
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()
            metrics.reset()

            _atomic_save(
                ckpt_path,
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "config": asdict(cfg),
                    "update": update_idx + 1,
                },
            )

    env.close()
    if eval_env is not None:
        eval_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
