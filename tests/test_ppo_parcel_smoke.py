"""Smoke tests for PPO parcel training components."""

import pytest
import torch

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


class TestPPOParcelSmoke:
    """Smoke tests for PPO parcel training."""

    def test_import_dual_lobe_model(self) -> None:
        """Test that DualLobeActorCritic can be imported."""
        from gbxcule.rl.dual_lobe_model import DualLobeActorCritic

        assert DualLobeActorCritic is not None

    def test_import_parcel_env(self) -> None:
        """Test that PokeredPackedParcelEnv can be imported."""
        from gbxcule.rl.pokered_packed_parcel_env import (
            EVENTS_LENGTH,
            SENSES_DIM,
            PokeredPackedParcelEnv,
        )

        assert PokeredPackedParcelEnv is not None
        assert SENSES_DIM == 4
        assert EVENTS_LENGTH == 320

    def test_import_ppo_utils(self) -> None:
        """Test that PPO utilities can be imported."""
        from gbxcule.rl.ppo import compute_gae, logprob_from_logits, ppo_losses

        assert compute_gae is not None
        assert logprob_from_logits is not None
        assert ppo_losses is not None

    def test_model_env_integration(self) -> None:
        """Test that model and env work together."""
        from gbxcule.rl.dual_lobe_model import DualLobeActorCritic
        from gbxcule.rl.pokered_packed_parcel_env import (
            EVENTS_LENGTH,
            SENSES_DIM,
            PokeredPackedParcelEnv,
        )

        # Create small env
        env = PokeredPackedParcelEnv(
            rom_path="red.gb",
            state_path="states/pokemonred_bulbasaur_roundtrip2.state",
            num_envs=2,
            max_steps=10,
        )

        try:
            # Create model
            model = DualLobeActorCritic(
                num_actions=env.num_actions,
                senses_dim=SENSES_DIM,
                events_dim=EVENTS_LENGTH,
            ).to("cuda")

            # Reset env
            obs = env.reset()

            # Forward pass
            logits, values = model(
                obs["pixels"],
                obs["senses"],
                obs["events"],
            )

            assert logits.shape == (2, env.num_actions)
            assert values.shape == (2,)

            # Step env
            probs = torch.softmax(logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1).to(torch.int32)
            next_obs, reward, terminated, truncated, info = env.step(actions)

            assert next_obs["pixels"].shape == (2, 1, 72, 20)
            assert next_obs["senses"].shape == (2, SENSES_DIM)
            assert next_obs["events"].shape == (2, EVENTS_LENGTH)
            assert reward.shape == (2,)

        finally:
            env.close()

    def test_gae_computation(self) -> None:
        """Test GAE computation with long-horizon parameters."""
        from gbxcule.rl.ppo import compute_gae

        steps = 128
        num_envs = 4
        gamma = 0.999  # Long horizon
        gae_lambda = 0.98  # High lambda

        rewards = torch.randn(steps, num_envs, dtype=torch.float32, device="cuda")
        values = torch.randn(steps, num_envs, dtype=torch.float32, device="cuda")
        dones = torch.zeros(steps, num_envs, dtype=torch.bool, device="cuda")
        last_value = torch.randn(num_envs, dtype=torch.float32, device="cuda")

        advantages, returns = compute_gae(
            rewards, values, dones, last_value, gamma=gamma, gae_lambda=gae_lambda
        )

        assert advantages.shape == (steps, num_envs)
        assert returns.shape == (steps, num_envs)
        assert not torch.isnan(advantages).any()
        assert not torch.isnan(returns).any()

    def test_ppo_losses(self) -> None:
        """Test PPO loss computation."""
        from gbxcule.rl.ppo import ppo_losses

        batch = 64
        num_actions = 18

        logits = torch.randn(batch, num_actions, dtype=torch.float32, device="cuda")
        actions = torch.randint(
            0, num_actions, (batch,), dtype=torch.int64, device="cuda"
        )
        old_logprobs = torch.randn(batch, dtype=torch.float32, device="cuda")
        returns = torch.randn(batch, dtype=torch.float32, device="cuda")
        advantages = torch.randn(batch, dtype=torch.float32, device="cuda")
        values = torch.randn(batch, dtype=torch.float32, device="cuda")

        losses = ppo_losses(
            logits,
            actions,
            old_logprobs,
            returns,
            advantages,
            values,
            clip=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
        )

        assert "loss_total" in losses
        assert "loss_policy" in losses
        assert "loss_value" in losses
        assert "entropy" in losses
        assert not torch.isnan(losses["loss_total"])


class TestPPOParcelTrainingLoop:
    """Test a minimal training loop."""

    @pytest.fixture
    def env(self, request):
        """Create small env for training tests."""
        from gbxcule.rl.pokered_packed_parcel_env import PokeredPackedParcelEnv

        env = PokeredPackedParcelEnv(
            rom_path="red.gb",
            state_path="states/pokemonred_bulbasaur_roundtrip2.state",
            num_envs=4,
            max_steps=50,
            curiosity_reset_on_parcel=True,
        )
        env.reset()
        yield env
        env.close()

    def test_mini_training_loop(self, env) -> None:
        """Test a minimal training loop runs without errors."""
        from gbxcule.rl.dual_lobe_model import DualLobeActorCritic
        from gbxcule.rl.pokered_packed_parcel_env import EVENTS_LENGTH, SENSES_DIM
        from gbxcule.rl.ppo import compute_gae, logprob_from_logits, ppo_losses

        model = DualLobeActorCritic(
            num_actions=env.num_actions,
            senses_dim=SENSES_DIM,
            events_dim=EVENTS_LENGTH,
        ).to("cuda")

        optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

        # Collect small rollout
        steps = 8
        num_envs = 4

        rollout_pixels = torch.empty(
            (steps, num_envs, 1, 72, 20), dtype=torch.uint8, device="cuda"
        )
        rollout_senses = torch.empty(
            (steps, num_envs, SENSES_DIM), dtype=torch.float32, device="cuda"
        )
        rollout_events = torch.empty(
            (steps, num_envs, EVENTS_LENGTH), dtype=torch.uint8, device="cuda"
        )
        rollout_actions = torch.empty(
            (steps, num_envs), dtype=torch.int32, device="cuda"
        )
        rollout_rewards = torch.empty(
            (steps, num_envs), dtype=torch.float32, device="cuda"
        )
        rollout_dones = torch.empty((steps, num_envs), dtype=torch.bool, device="cuda")
        rollout_values = torch.empty(
            (steps, num_envs), dtype=torch.float32, device="cuda"
        )
        rollout_logprobs = torch.empty(
            (steps, num_envs), dtype=torch.float32, device="cuda"
        )

        obs = env.reset()
        pixels = obs["pixels"]
        senses = obs["senses"]
        events = obs["events"]

        model.eval()
        with torch.no_grad():
            for t in range(steps):
                logits, values = model(pixels, senses, events)
                probs = torch.softmax(logits, dim=-1)
                actions_i64 = torch.multinomial(probs, num_samples=1).squeeze(1)
                logprobs = logprob_from_logits(logits, actions_i64)
                actions = actions_i64.to(torch.int32)

                rollout_pixels[t].copy_(pixels)
                rollout_senses[t].copy_(senses)
                rollout_events[t].copy_(events)
                rollout_actions[t].copy_(actions)
                rollout_values[t].copy_(values)
                rollout_logprobs[t].copy_(logprobs)

                next_obs, reward, terminated, truncated, _ = env.step(actions)
                done = terminated | truncated

                rollout_rewards[t].copy_(reward)
                rollout_dones[t].copy_(done)

                if done.any():
                    env.reset_mask(done)

                pixels = next_obs["pixels"]
                senses = next_obs["senses"]
                events = next_obs["events"]

            _, last_value = model(pixels, senses, events)

        # Compute advantages
        advantages, returns = compute_gae(
            rollout_rewards,
            rollout_values,
            rollout_dones,
            last_value,
            gamma=0.999,
            gae_lambda=0.98,
        )

        # PPO update
        model.train()
        batch_size = steps * num_envs
        flat_pixels = rollout_pixels.reshape(batch_size, 1, 72, 20)
        flat_senses = rollout_senses.reshape(batch_size, SENSES_DIM)
        flat_events = rollout_events.reshape(batch_size, EVENTS_LENGTH)
        flat_actions = rollout_actions.reshape(batch_size)
        flat_old_logprobs = rollout_logprobs.reshape(batch_size)
        flat_returns = returns.reshape(batch_size)
        flat_advantages = advantages.reshape(batch_size)

        # Normalize advantages
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-8
        )

        logits, values = model(flat_pixels, flat_senses, flat_events)
        losses = ppo_losses(
            logits,
            flat_actions,
            flat_old_logprobs,
            flat_returns,
            flat_advantages,
            values,
            clip=0.2,
            value_coef=0.5,
            entropy_coef=0.01,
            normalize_adv=False,
        )

        optimizer.zero_grad()
        losses["loss_total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        # Verify no NaN in parameters
        for param in model.parameters():
            assert not torch.isnan(param).any()
