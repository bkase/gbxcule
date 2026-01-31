"""Dreamer v3 configuration scaffolding (M0)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

_ALLOWED_FLOAT_DTYPES = {"float32", "torch.float32"}


def _require_torch() -> Any:
    import importlib

    try:
        return importlib.import_module("torch")
    except Exception as exc:
        raise RuntimeError(
            "Torch is required for gbxcule.rl.dreamer_v3. Install with `uv sync`."
        ) from exc


def _dtype_name(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        torch = _require_torch()
    except Exception:
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    return str(value)


def _validate_positive(name: str, value: int | float) -> list[str]:
    if value <= 0:
        return [f"{name} must be > 0"]
    return []


def _validate_non_negative(name: str, value: int | float) -> list[str]:
    if value < 0:
        return [f"{name} must be >= 0"]
    return []


def _validate_range(name: str, value: float, *, low: float, high: float) -> list[str]:
    if not (low < value <= high):
        return [f"{name} must be in ({low}, {high}]"]
    return []


def _validate_list(name: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        return [f"{name} must be a list"]
    return []


@dataclass
class PrecisionPolicy:
    model_dtype: Any = "float32"
    rnn_dtype: Any = "float32"
    autocast: bool = False
    allow_tf32: bool = False

    def validate(self) -> list[str]:
        errors: list[str] = []
        model_dtype = _dtype_name(self.model_dtype)
        rnn_dtype = _dtype_name(self.rnn_dtype)
        if model_dtype not in _ALLOWED_FLOAT_DTYPES:
            errors.append(f"model_dtype must be one of {sorted(_ALLOWED_FLOAT_DTYPES)}")
        if rnn_dtype not in _ALLOWED_FLOAT_DTYPES:
            errors.append(f"rnn_dtype must be one of {sorted(_ALLOWED_FLOAT_DTYPES)}")
        if not isinstance(self.autocast, bool):
            errors.append("autocast must be bool")
        if not isinstance(self.allow_tf32, bool):
            errors.append("allow_tf32 must be bool")
        return errors


@dataclass
class ReplayConfig:
    capacity: int = 1_000_000
    seq_len: int = 64
    batch_size: int = 16
    commit_stride: int = 8

    def validate(self) -> list[str]:
        errors: list[str] = []
        errors += _validate_positive("capacity", self.capacity)
        if self.seq_len < 2:
            errors.append("seq_len must be >= 2")
        if self.batch_size < 1:
            errors.append("batch_size must be >= 1")
        if self.commit_stride < 1:
            errors.append("commit_stride must be >= 1")
        return errors


@dataclass
class KeyGroup:
    encoder: list[str] = field(default_factory=list)
    decoder: list[str] = field(default_factory=list)

    def validate(self, prefix: str) -> list[str]:
        errors: list[str] = []
        errors += _validate_list(f"{prefix}.encoder", self.encoder)
        errors += _validate_list(f"{prefix}.decoder", self.decoder)
        return errors


@dataclass
class AlgoConfig:
    gamma: float = 0.996996996996997
    lmbda: float = 0.95
    horizon: int = 15
    replay_ratio: float = 1.0
    learning_starts: int = 1024
    per_rank_batch_size: int = 16
    per_rank_sequence_length: int = 64
    dense_units: int = 1024
    mlp_layers: int = 5
    dense_act: str = "torch.nn.SiLU"
    unimix: float = 0.01
    hafner_initialization: bool = True
    cnn_keys: KeyGroup = field(default_factory=KeyGroup)
    mlp_keys: KeyGroup = field(default_factory=KeyGroup)

    def validate(self) -> list[str]:
        errors: list[str] = []
        errors += _validate_range("gamma", self.gamma, low=0.0, high=1.0)
        errors += _validate_range("lmbda", self.lmbda, low=0.0, high=1.0)
        errors += _validate_positive("horizon", self.horizon)
        errors += _validate_non_negative("replay_ratio", self.replay_ratio)
        errors += _validate_non_negative("learning_starts", self.learning_starts)
        if self.per_rank_batch_size < 1:
            errors.append("per_rank_batch_size must be >= 1")
        if self.per_rank_sequence_length < 2:
            errors.append("per_rank_sequence_length must be >= 2")
        if self.dense_units < 1:
            errors.append("dense_units must be >= 1")
        if self.mlp_layers < 1:
            errors.append("mlp_layers must be >= 1")
        if not isinstance(self.dense_act, str) or not self.dense_act:
            errors.append("dense_act must be non-empty string")
        if not 0.0 <= self.unimix <= 1.0:
            errors.append("unimix must be in [0, 1]")
        if not isinstance(self.hafner_initialization, bool):
            errors.append("hafner_initialization must be bool")
        errors += self.cnn_keys.validate("cnn_keys")
        errors += self.mlp_keys.validate("mlp_keys")
        return errors


@dataclass
class RewardModelConfig:
    bins: int = 255
    low: float = -20.0
    high: float = 20.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.bins < 2:
            errors.append("reward_model.bins must be >= 2")
        if self.low >= self.high:
            errors.append("reward_model.low must be < reward_model.high")
        return errors


@dataclass
class LayerNormConfig:
    eps: float = 1e-3

    def validate(self) -> list[str]:
        errors: list[str] = []
        errors += _validate_positive("layer_norm.eps", self.eps)
        return errors


@dataclass
class RecurrentModelConfig:
    recurrent_state_size: int = 4096
    dense_units: int = 1024
    layer_norm: LayerNormConfig = field(default_factory=LayerNormConfig)

    def validate(self) -> list[str]:
        errors: list[str] = []
        errors += _validate_positive(
            "recurrent_model.recurrent_state_size", self.recurrent_state_size
        )
        errors += _validate_positive("recurrent_model.dense_units", self.dense_units)
        errors += self.layer_norm.validate()
        return errors


@dataclass
class TransitionModelConfig:
    hidden_size: int = 1024
    layer_norm: LayerNormConfig = field(default_factory=LayerNormConfig)

    def validate(self) -> list[str]:
        errors: list[str] = []
        errors += _validate_positive("transition_model.hidden_size", self.hidden_size)
        errors += self.layer_norm.validate()
        return errors


@dataclass
class RepresentationModelConfig:
    hidden_size: int = 1024
    layer_norm: LayerNormConfig = field(default_factory=LayerNormConfig)

    def validate(self) -> list[str]:
        errors: list[str] = []
        errors += _validate_positive(
            "representation_model.hidden_size", self.hidden_size
        )
        errors += self.layer_norm.validate()
        return errors


@dataclass
class WorldModelConfig:
    discrete_size: int = 32
    stochastic_size: int = 32
    kl_dynamic: float = 0.5
    kl_representation: float = 0.1
    kl_free_nats: float = 1.0
    kl_regularizer: float = 1.0
    continue_scale_factor: float = 1.0
    decoupled_rssm: bool = False
    learnable_initial_recurrent_state: bool = True
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    recurrent_model: RecurrentModelConfig = field(default_factory=RecurrentModelConfig)
    transition_model: TransitionModelConfig = field(
        default_factory=TransitionModelConfig
    )
    representation_model: RepresentationModelConfig = field(
        default_factory=RepresentationModelConfig
    )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.discrete_size < 2:
            errors.append("discrete_size must be >= 2")
        if self.stochastic_size < 1:
            errors.append("stochastic_size must be >= 1")
        errors += _validate_non_negative("kl_dynamic", self.kl_dynamic)
        errors += _validate_non_negative("kl_representation", self.kl_representation)
        errors += _validate_non_negative("kl_free_nats", self.kl_free_nats)
        errors += _validate_non_negative("kl_regularizer", self.kl_regularizer)
        errors += _validate_non_negative(
            "continue_scale_factor", self.continue_scale_factor
        )
        if not isinstance(self.decoupled_rssm, bool):
            errors.append("decoupled_rssm must be bool")
        if not isinstance(self.learnable_initial_recurrent_state, bool):
            errors.append("learnable_initial_recurrent_state must be bool")
        errors += self.reward_model.validate()
        errors += self.recurrent_model.validate()
        errors += self.transition_model.validate()
        errors += self.representation_model.validate()
        return errors


@dataclass
class PercentileConfig:
    low: float = 0.05
    high: float = 0.95

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not 0.0 < self.low < 1.0:
            errors.append("percentile.low must be in (0, 1)")
        if not 0.0 < self.high < 1.0:
            errors.append("percentile.high must be in (0, 1)")
        if self.low >= self.high:
            errors.append("percentile.low must be < percentile.high")
        return errors


@dataclass
class MomentsConfig:
    decay: float = 0.99
    max: float = 1.0
    percentile: PercentileConfig = field(default_factory=PercentileConfig)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not 0.0 < self.decay <= 1.0:
            errors.append("moments.decay must be in (0, 1]")
        errors += _validate_positive("moments.max", self.max)
        errors += self.percentile.validate()
        return errors


@dataclass
class ActorConfig:
    ent_coef: float = 0.0
    clip_gradients: float | None = None
    moments: MomentsConfig = field(default_factory=MomentsConfig)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.ent_coef < 0:
            errors.append("actor.ent_coef must be >= 0")
        if self.clip_gradients is not None and self.clip_gradients <= 0:
            errors.append("actor.clip_gradients must be > 0 when set")
        errors += self.moments.validate()
        return errors


@dataclass
class CriticConfig:
    bins: int = 255
    tau: float = 0.005
    target_update_freq: int = 1
    clip_gradients: float | None = None

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.bins < 2:
            errors.append("critic.bins must be >= 2")
        if not 0.0 < self.tau <= 1.0:
            errors.append("critic.tau must be in (0, 1]")
        if self.target_update_freq < 1:
            errors.append("critic.target_update_freq must be >= 1")
        if self.clip_gradients is not None and self.clip_gradients <= 0:
            errors.append("critic.clip_gradients must be > 0 when set")
        return errors


@dataclass
class DistributionConfig:
    type: str = "auto"

    def validate(self) -> list[str]:
        if not isinstance(self.type, str) or not self.type:
            return ["distribution.type must be non-empty string"]
        return []


@dataclass
class ObservationConfig:
    obs_type: str = "rgb"
    obs_format: str = "packed2"
    obs_norm: str = "zero_centered"
    unpack_impl: str = "lut"

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.obs_type not in ("rgb", "vector"):
            errors.append("observation.obs_type must be 'rgb' or 'vector'")
        if self.obs_format not in ("packed2", "u8"):
            errors.append("observation.obs_format must be 'packed2' or 'u8'")
        if self.obs_norm not in ("zero_centered",):
            errors.append("observation.obs_norm must be 'zero_centered'")
        if self.unpack_impl not in ("lut", "triton", "warp"):
            errors.append("observation.unpack_impl must be 'lut', 'triton', or 'warp'")
        return errors


@dataclass
class DreamerV3Config:
    precision: PrecisionPolicy = field(default_factory=PrecisionPolicy)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    algo: AlgoConfig = field(default_factory=AlgoConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    critic: CriticConfig = field(default_factory=CriticConfig)
    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    seed: int = 0

    def validate(self) -> list[str]:
        errors: list[str] = []
        errors += self.precision.validate()
        errors += self.replay.validate()
        errors += self.algo.validate()
        errors += self.world_model.validate()
        errors += self.actor.validate()
        errors += self.critic.validate()
        errors += self.distribution.validate()
        errors += self.observation.validate()
        if not isinstance(self.seed, int):
            errors.append("seed must be int")
        return errors


def validate_config(config: DreamerV3Config) -> list[str]:
    return config.validate()


def validate_config_or_raise(config: DreamerV3Config) -> None:
    errors = validate_config(config)
    if errors:
        joined = "\n".join(errors)
        raise ValueError(f"DreamerV3Config validation failed:\n{joined}")


def flatten_config_errors(errors: Iterable[str]) -> str:
    return "\n".join(errors)
