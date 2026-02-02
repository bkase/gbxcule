"""Tests for DualLobeActorCritic model."""

import pytest
import torch

from gbxcule.rl.dual_lobe_model import DualLobeActorCritic
from gbxcule.rl.pokered_packed_parcel_env import EVENTS_LENGTH, SENSES_DIM


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestDualLobeActorCritic:
    """Test DualLobeActorCritic model."""

    def test_init(self) -> None:
        """Test model initialization."""
        model = DualLobeActorCritic(
            num_actions=18,
            senses_dim=SENSES_DIM,
            events_dim=EVENTS_LENGTH,
        )
        assert model.num_actions == 18
        assert model.senses_dim == SENSES_DIM
        assert model.events_dim == EVENTS_LENGTH

    def test_forward_shapes(self) -> None:
        """Test forward pass produces correct output shapes."""
        model = DualLobeActorCritic(
            num_actions=18,
            senses_dim=SENSES_DIM,
            events_dim=EVENTS_LENGTH,
        ).to("cuda")

        batch_size = 32
        pixels = torch.randint(
            0, 256, (batch_size, 1, 72, 20), dtype=torch.uint8, device="cuda"
        )
        senses = torch.randn(batch_size, SENSES_DIM, dtype=torch.float32, device="cuda")
        events = torch.randint(
            0, 256, (batch_size, EVENTS_LENGTH), dtype=torch.uint8, device="cuda"
        )

        logits, values = model(pixels, senses, events)

        assert logits.shape == (batch_size, 18)
        assert values.shape == (batch_size,)
        assert logits.dtype == torch.float32
        assert values.dtype == torch.float32

    def test_forward_large_batch(self) -> None:
        """Test forward pass with large batch (simulating many envs)."""
        model = DualLobeActorCritic(
            num_actions=18,
            senses_dim=SENSES_DIM,
            events_dim=EVENTS_LENGTH,
        ).to("cuda")

        batch_size = 4096
        pixels = torch.randint(
            0, 256, (batch_size, 1, 72, 20), dtype=torch.uint8, device="cuda"
        )
        senses = torch.randn(batch_size, SENSES_DIM, dtype=torch.float32, device="cuda")
        events = torch.randint(
            0, 256, (batch_size, EVENTS_LENGTH), dtype=torch.uint8, device="cuda"
        )

        logits, values = model(pixels, senses, events)

        assert logits.shape == (batch_size, 18)
        assert values.shape == (batch_size,)

    def test_gradient_flow(self) -> None:
        """Test gradients flow through all model components."""
        model = DualLobeActorCritic(
            num_actions=18,
            senses_dim=SENSES_DIM,
            events_dim=EVENTS_LENGTH,
        ).to("cuda")

        batch_size = 16
        pixels = torch.randint(
            0, 256, (batch_size, 1, 72, 20), dtype=torch.uint8, device="cuda"
        )
        senses = torch.randn(batch_size, SENSES_DIM, dtype=torch.float32, device="cuda")
        events = torch.randint(
            0, 256, (batch_size, EVENTS_LENGTH), dtype=torch.uint8, device="cuda"
        )

        logits, values = model(pixels, senses, events)
        loss = logits.mean() + values.mean()
        loss.backward()

        # Check gradients exist for all parameters
        for name, param in [
            ("cnn", model.cnn),
            ("senses_mlp", model.senses_mlp),
            ("events_mlp", model.events_mlp),
            ("fusion", model.fusion),
            ("policy", model.policy),
            ("value", model.value),
        ]:
            for p in param.parameters():
                assert p.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(p.grad).any(), f"NaN gradient for {name}"

    def test_state_dict_roundtrip(self) -> None:
        """Test save/load state dict."""
        model1 = DualLobeActorCritic(
            num_actions=18,
            senses_dim=SENSES_DIM,
            events_dim=EVENTS_LENGTH,
        ).to("cuda")

        model2 = DualLobeActorCritic(
            num_actions=18,
            senses_dim=SENSES_DIM,
            events_dim=EVENTS_LENGTH,
        ).to("cuda")

        # Models should produce different outputs initially
        batch_size = 4
        pixels = torch.randint(
            0, 256, (batch_size, 1, 72, 20), dtype=torch.uint8, device="cuda"
        )
        senses = torch.randn(batch_size, SENSES_DIM, dtype=torch.float32, device="cuda")
        events = torch.randint(
            0, 256, (batch_size, EVENTS_LENGTH), dtype=torch.uint8, device="cuda"
        )

        out1a, _ = model1(pixels, senses, events)
        out2a, _ = model2(pixels, senses, events)

        # Load state dict
        model2.load_state_dict(model1.state_dict())

        out1b, _ = model1(pixels, senses, events)
        out2b, _ = model2(pixels, senses, events)

        # After loading, outputs should be identical
        assert torch.allclose(out1b, out2b)

    def test_train_eval_mode(self) -> None:
        """Test train/eval mode switching."""
        model = DualLobeActorCritic(
            num_actions=18,
            senses_dim=SENSES_DIM,
            events_dim=EVENTS_LENGTH,
        ).to("cuda")

        model.train()
        for module in model._all_modules:
            assert module.training

        model.eval()
        for module in model._all_modules:
            assert not module.training

    def test_invalid_inputs(self) -> None:
        """Test error handling for invalid inputs."""
        model = DualLobeActorCritic(
            num_actions=18,
            senses_dim=SENSES_DIM,
            events_dim=EVENTS_LENGTH,
        ).to("cuda")

        batch_size = 4
        pixels = torch.randint(
            0, 256, (batch_size, 1, 72, 20), dtype=torch.uint8, device="cuda"
        )
        senses = torch.randn(batch_size, SENSES_DIM, dtype=torch.float32, device="cuda")
        events = torch.randint(
            0, 256, (batch_size, EVENTS_LENGTH), dtype=torch.uint8, device="cuda"
        )

        # Wrong pixel shape
        with pytest.raises(ValueError, match="pixels must be 4D"):
            model(pixels.squeeze(1), senses, events)

        # Wrong senses dim
        with pytest.raises(ValueError, match="senses dim mismatch"):
            model(pixels, senses[:, :2], events)

        # Wrong events dim
        with pytest.raises(ValueError, match="events dim mismatch"):
            model(pixels, senses, events[:, :100])

        # Batch size mismatch
        with pytest.raises(ValueError, match="batch size mismatch"):
            model(pixels, senses[:2], events)

    def test_custom_architecture(self) -> None:
        """Test custom CNN/MLP architecture."""
        model = DualLobeActorCritic(
            num_actions=10,
            senses_dim=4,
            events_dim=100,
            cnn_channels=(16, 32, 32),
            mlp_hidden=64,
            fusion_hidden=256,
        ).to("cuda")

        batch_size = 8
        pixels = torch.randint(
            0, 256, (batch_size, 1, 72, 20), dtype=torch.uint8, device="cuda"
        )
        senses = torch.randn(batch_size, 4, dtype=torch.float32, device="cuda")
        events = torch.randint(
            0, 256, (batch_size, 100), dtype=torch.uint8, device="cuda"
        )

        logits, values = model(pixels, senses, events)
        assert logits.shape == (batch_size, 10)
        assert values.shape == (batch_size,)
