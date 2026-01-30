from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gbxcule.rl.models import PixelActorCriticCNN  # noqa: E402
from gbxcule.rl.packed_pixels import pack_2bpp_u8  # noqa: E402
from gbxcule.rl.ppo import logprob_from_logits, ppo_update_minibatch  # noqa: E402


def test_packed_model_unpack_forward() -> None:
    torch.manual_seed(0)
    obs = torch.randint(0, 4, (4, 1, 72, 80), dtype=torch.uint8)
    packed = pack_2bpp_u8(obs)
    model = PixelActorCriticCNN(num_actions=8, in_frames=1, input_format="packed2")
    logits, values = model(packed)
    assert logits.shape == (4, 8)
    assert values.shape == (4,)


def test_ppo_minibatch_update_smoke() -> None:
    torch.manual_seed(1)
    batch = 32
    model = PixelActorCriticCNN(num_actions=6, in_frames=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    obs = torch.randint(0, 4, (batch, 1, 72, 80), dtype=torch.uint8)
    with torch.no_grad():
        logits, values = model(obs)
        actions = torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(1)
        old_logprobs = logprob_from_logits(logits, actions)
    returns = torch.randn((batch,), dtype=torch.float32)
    advantages = torch.randn((batch,), dtype=torch.float32)

    before = sum(p.detach().abs().sum().item() for p in model.parameters())
    stats = ppo_update_minibatch(
        model=model,
        optimizer=optimizer,
        obs=obs,
        actions=actions,
        old_logprobs=old_logprobs,
        returns=returns,
        advantages=advantages,
        clip=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        ppo_epochs=2,
        minibatch_size=8,
        grad_clip=0.5,
    )
    after = sum(p.detach().abs().sum().item() for p in model.parameters())
    assert stats["updates"] > 0
    assert before != after
