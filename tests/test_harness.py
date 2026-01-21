"""Tests for harness action generation and utilities.

Tests cover:
- Deterministic action generation
- noop and seeded_random generators
- Action generator metadata
"""

from __future__ import annotations

# Import from bench directory
import sys
from pathlib import Path

import numpy as np
import pytest

# Add bench directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "bench"))

from harness import (  # noqa: E402
    ACTION_GEN_VERSION,
    generate_actions,
    get_action_gen_metadata,
)


class TestGenerateActions:
    """Tests for generate_actions pure function."""

    def test_noop_returns_zeros(self) -> None:
        """noop generator returns all zeros."""
        actions = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=None,
            gen_name="noop",
        )
        assert actions.shape == (4,)
        assert actions.dtype == np.int32
        assert np.all(actions == 0)

    def test_noop_ignores_seed(self) -> None:
        """noop generator works the same regardless of seed."""
        actions1 = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=42,
            gen_name="noop",
        )
        actions2 = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=99,
            gen_name="noop",
        )
        np.testing.assert_array_equal(actions1, actions2)

    def test_seeded_random_deterministic(self) -> None:
        """seeded_random produces identical results for same inputs."""
        actions1 = generate_actions(
            step_idx=5,
            num_envs=8,
            seed=42,
            gen_name="seeded_random",
        )
        actions2 = generate_actions(
            step_idx=5,
            num_envs=8,
            seed=42,
            gen_name="seeded_random",
        )
        np.testing.assert_array_equal(actions1, actions2)

    def test_seeded_random_varies_by_step_idx(self) -> None:
        """seeded_random produces different actions for different step indices."""
        actions1 = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=42,
            gen_name="seeded_random",
        )
        actions2 = generate_actions(
            step_idx=1,
            num_envs=4,
            seed=42,
            gen_name="seeded_random",
        )
        # Should be different (extremely high probability)
        assert not np.array_equal(actions1, actions2)

    def test_seeded_random_varies_by_seed(self) -> None:
        """seeded_random produces different actions for different seeds."""
        actions1 = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=42,
            gen_name="seeded_random",
        )
        actions2 = generate_actions(
            step_idx=0,
            num_envs=4,
            seed=99,
            gen_name="seeded_random",
        )
        # Should be different (extremely high probability)
        assert not np.array_equal(actions1, actions2)

    def test_seeded_random_requires_seed(self) -> None:
        """seeded_random raises ValueError if seed is None."""
        with pytest.raises(ValueError, match="requires a seed"):
            generate_actions(
                step_idx=0,
                num_envs=4,
                seed=None,
                gen_name="seeded_random",
            )

    def test_seeded_random_action_range(self) -> None:
        """seeded_random produces actions in valid range [0, 8]."""
        actions = generate_actions(
            step_idx=0,
            num_envs=100,
            seed=42,
            gen_name="seeded_random",
        )
        assert np.all(actions >= 0)
        assert np.all(actions < 9)

    def test_unknown_generator_raises(self) -> None:
        """Unknown generator name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown action generator"):
            generate_actions(
                step_idx=0,
                num_envs=4,
                seed=42,
                gen_name="unknown_gen",
            )

    def test_single_env_batch_semantics(self) -> None:
        """Single env still returns array with correct shape."""
        actions = generate_actions(
            step_idx=0,
            num_envs=1,
            seed=42,
            gen_name="noop",
        )
        assert actions.shape == (1,)
        assert actions.dtype == np.int32


class TestGetActionGenMetadata:
    """Tests for get_action_gen_metadata function."""

    def test_metadata_structure(self) -> None:
        """Metadata contains name, version, seed."""
        meta = get_action_gen_metadata("noop", None)
        assert meta["name"] == "noop"
        assert meta["version"] == ACTION_GEN_VERSION
        assert meta["seed"] is None

    def test_metadata_with_seed(self) -> None:
        """Metadata records seed correctly."""
        meta = get_action_gen_metadata("seeded_random", 42)
        assert meta["name"] == "seeded_random"
        assert meta["version"] == ACTION_GEN_VERSION
        assert meta["seed"] == 42

    def test_version_is_stable(self) -> None:
        """Version string is stable for reproducibility."""
        assert ACTION_GEN_VERSION == "1.0"
