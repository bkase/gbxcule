from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.dreamer_v3.decoders import CNNDecoder, MultiDecoder  # noqa: E402
from gbxcule.rl.dreamer_v3.encoders import (  # noqa: E402
    MultiEncoder,
    Packed2PixelEncoder,
)
from gbxcule.rl.dreamer_v3.heads import ContinueHead, RewardHead  # noqa: E402
from gbxcule.rl.dreamer_v3.rssm import RSSM, RecurrentModel  # noqa: E402
from gbxcule.rl.dreamer_v3.train_world_model import wm_train_step  # noqa: E402
from gbxcule.rl.dreamer_v3.world_model import WorldModel  # noqa: E402


def _make_mlp(input_dim: int, output_dim: int):  # type: ignore[no-untyped-def]
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, output_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(output_dim, output_dim),
    )


def _build_model(h: int, w: int, action_dim: int) -> WorldModel:
    stochastic_size = 2
    discrete = 3
    stoch_state_size = stochastic_size * discrete
    recurrent_state_size = 8

    cnn_encoder = Packed2PixelEncoder(
        keys=["pixels"],
        image_size=(h, w),
        channels_multiplier=2,
        stages=2,
    )
    encoder = MultiEncoder(cnn_encoder, None)
    embedded_dim = encoder.cnn_output_dim

    recurrent_model = RecurrentModel(
        input_size=stoch_state_size + action_dim,
        recurrent_state_size=recurrent_state_size,
        dense_units=8,
    )
    representation_model = _make_mlp(
        recurrent_state_size + embedded_dim, stoch_state_size
    )
    transition_model = _make_mlp(recurrent_state_size, stoch_state_size)
    rssm = RSSM(
        recurrent_model=recurrent_model,
        representation_model=representation_model,
        transition_model=transition_model,
        discrete=discrete,
        unimix=0.0,
    )

    decoder = CNNDecoder(
        keys=["pixels"],
        output_channels=[1],
        channels_multiplier=2,
        latent_state_size=stoch_state_size + recurrent_state_size,
        encoder_output_shape=cnn_encoder.output_shape,
        stages=2,
    )
    observation_model = MultiDecoder(decoder, None)
    reward_model = RewardHead(
        input_dim=stoch_state_size + recurrent_state_size,
        bins=7,
        mlp_layers=2,
        dense_units=16,
    )
    continue_model = ContinueHead(
        input_dim=stoch_state_size + recurrent_state_size,
        mlp_layers=2,
        dense_units=16,
    )

    return WorldModel(
        encoder=encoder,
        rssm=rssm,
        observation_model=observation_model,
        reward_model=reward_model,
        continue_model=continue_model,
        cnn_keys=["pixels"],
        mlp_keys=[],
        reward_low=-20.0,
        reward_high=20.0,
        continue_scale_factor=1.0,
    )


def test_overfit_one_batch() -> None:
    torch.manual_seed(0)
    t, b = 6, 2
    h, w = 4, 4
    w_bytes = w // 4
    action_dim = 4

    model = _build_model(h, w, action_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    batch = {
        "obs": {
            "pixels": torch.randint(0, 4, (t, b, 1, h, w_bytes), dtype=torch.uint8)
        },
        "actions": torch.nn.functional.one_hot(
            torch.randint(0, action_dim, (t, b)), action_dim
        ).float(),
        "rewards": torch.zeros((t, b, 1), dtype=torch.float32),
        "is_first": torch.zeros((t, b, 1), dtype=torch.float32),
        "terminated": torch.zeros((t, b, 1), dtype=torch.float32),
    }
    batch["is_first"][0] = 1.0

    losses = []
    for _ in range(8):
        metrics = wm_train_step(
            batch,
            model,
            optimizer,
            obs_format="packed2",
            kl_dynamic=0.5,
            kl_representation=0.1,
            kl_free_nats=1.0,
            kl_regularizer=1.0,
            action_dim=action_dim,
        )
        losses.append(metrics.loss.item())
    assert losses[-1] < losses[0]
