"""Dreamer v3 configuration (M0) and async engine config (M7)."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception:
        return None


@dataclass
class PrecisionPolicy:
    model_dtype: Any = "float32"


@dataclass
class PercentileConfig:
    low: float = 0.05
    high: float = 0.95


@dataclass
class MomentsConfig:
    decay: float = 0.99
    max: float = 1.0
    percentile: PercentileConfig = field(default_factory=PercentileConfig)


@dataclass
class ActorConfig:
    ent_coef: float = 0.0
    moments: MomentsConfig = field(default_factory=MomentsConfig)


@dataclass
class RewardModelConfig:
    bins: int = 255
    low: float = -20.0
    high: float = 20.0


@dataclass
class WorldModelConfig:
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    kl_dynamic: float = 0.5
    kl_representation: float = 0.1
    kl_free_nats: float = 1.0
    kl_regularizer: float = 1.0


@dataclass
class CriticConfig:
    bins: int = 255
    tau: float = 0.005
    target_update_freq: int = 1


@dataclass
class DreamerV3Config:
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)


def _is_float32(value: Any) -> bool:
    torch = _require_torch()
    if torch is not None and isinstance(value, torch.dtype):
        return value is torch.float32
    if isinstance(value, str):
        return value == "float32"
    return False


def validate_config(cfg: DreamerV3Config) -> list[str]:
    errors: list[str] = []
    if cfg.world_model.reward_model.bins < 2:
        errors.append("world_model.reward_model.bins must be >= 2")
    if cfg.critic.bins < 2:
        errors.append("critic.bins must be >= 2")
    if cfg.world_model.reward_model.low >= cfg.world_model.reward_model.high:
        errors.append("world_model.reward_model.low must be < high")

    low = cfg.actor.moments.percentile.low
    high = cfg.actor.moments.percentile.high
    if not (0.0 < low < high < 1.0):
        errors.append("actor.moments.percentile.low invalid")
        errors.append("actor.moments.percentile.high invalid")

    if cfg.actor.ent_coef < 0.0:
        errors.append("actor.ent_coef must be >= 0")

    if cfg.critic.tau <= 0.0:
        errors.append("critic.tau must be > 0")
    if cfg.critic.target_update_freq <= 0:
        errors.append("critic.target_update_freq must be >= 1")

    if not _is_float32(cfg.precision.model_dtype):
        errors.append("precision.model_dtype must be float32")

    return errors


@dataclass(frozen=True)
class DreamerEngineConfig:
    num_envs: int = 1
    obs_shape: tuple[int, ...] = (1, 72, 20)
    replay_capacity: int = 1024
    seq_len: int = 16
    batch_size: int = 8
    steps_per_rollout: int = 32
    commit_stride: int = 8
    replay_ratio: float = 1.0
    learning_starts: int = 0
    pretrain_steps: int = 0
    min_ready_steps: int = 0
    safety_margin: int = 0
    max_learner_steps_per_tick: int | None = None
    device: str = "cpu"
    seed: int = 0
    debug: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "num_envs", int(self.num_envs))
        object.__setattr__(self, "replay_capacity", int(self.replay_capacity))
        object.__setattr__(self, "seq_len", int(self.seq_len))
        object.__setattr__(self, "batch_size", int(self.batch_size))
        object.__setattr__(self, "steps_per_rollout", int(self.steps_per_rollout))
        object.__setattr__(self, "commit_stride", int(self.commit_stride))
        object.__setattr__(self, "learning_starts", int(self.learning_starts))
        object.__setattr__(self, "pretrain_steps", int(self.pretrain_steps))
        if self.min_ready_steps <= 0:
            object.__setattr__(self, "min_ready_steps", int(self.seq_len))
        if self.safety_margin <= 0:
            object.__setattr__(self, "safety_margin", int(self.seq_len))
        self._validate()

    def _validate(self) -> None:
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if self.replay_capacity < 2:
            raise ValueError("replay_capacity must be >= 2")
        if self.seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.steps_per_rollout < 1:
            raise ValueError("steps_per_rollout must be >= 1")
        if self.commit_stride < 1:
            raise ValueError("commit_stride must be >= 1")
        if self.replay_ratio < 0:
            raise ValueError("replay_ratio must be >= 0")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must be >= 0")
        if self.pretrain_steps < 0:
            raise ValueError("pretrain_steps must be >= 0")
        if self.min_ready_steps < self.seq_len:
            raise ValueError("min_ready_steps must be >= seq_len")
        if self.safety_margin < self.seq_len:
            raise ValueError("safety_margin must be >= seq_len")
        if self.replay_capacity < self.seq_len + 1:
            raise ValueError("replay_capacity must be >= seq_len + 1")
        if self.commit_stride > max(1, self.seq_len // 2):
            warnings.warn(
                "commit_stride is larger than seq_len/2; learner may starve",
                stacklevel=2,
            )
        if (
            self.max_learner_steps_per_tick is not None
            and self.max_learner_steps_per_tick < 1
        ):
            raise ValueError("max_learner_steps_per_tick must be >= 1")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("device must be 'cpu' or 'cuda'")
        if not self.obs_shape:
            raise ValueError("obs_shape must be non-empty")


__all__ = [
    "ActorConfig",
    "CriticConfig",
    "DreamerEngineConfig",
    "DreamerV3Config",
    "MomentsConfig",
    "PercentileConfig",
    "PrecisionPolicy",
    "RewardModelConfig",
    "WorldModelConfig",
    "validate_config",
]
