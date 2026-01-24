"""Tests for action codec definitions."""

from __future__ import annotations

import pytest

from gbxcule.core.action_codec import (
    BUTTON_A,
    BUTTON_B,
    BUTTON_START,
    DPAD_DOWN,
    DPAD_LEFT,
    DPAD_RIGHT,
    DPAD_UP,
    POKERED_PUFFER_V0_ID,
    get_action_codec,
    list_action_codecs,
)


def test_list_action_codecs_contains_known_ids() -> None:
    """Registry exposes the expected codec ids."""
    ids = list_action_codecs()
    assert POKERED_PUFFER_V0_ID in ids


def test_pokered_puffer_v0_metadata_and_lengths() -> None:
    """PokemonRedPuffer codec metadata and lengths are consistent."""
    codec = get_action_codec(POKERED_PUFFER_V0_ID)
    assert codec.name == "pokemonred_puffer"
    assert codec.version == "v0"
    assert codec.num_actions == len(codec.action_names)
    assert codec.num_actions == 7


def test_pokered_puffer_v0_pyboy_mapping() -> None:
    """PokemonRedPuffer codec maps to PyBoy button names correctly."""
    codec = get_action_codec(POKERED_PUFFER_V0_ID)
    expected = {
        "A": "a",
        "B": "b",
        "START": "start",
        "UP": "up",
        "DOWN": "down",
        "LEFT": "left",
        "RIGHT": "right",
    }
    for idx, name in enumerate(codec.action_names):
        assert codec.to_pyboy_button(idx) == expected[name]


def test_pokered_puffer_v0_joypad_masks() -> None:
    """PokemonRedPuffer codec returns correct JOYP masks."""
    codec = get_action_codec(POKERED_PUFFER_V0_ID)
    expected = {
        "A": (0, BUTTON_A),
        "B": (0, BUTTON_B),
        "START": (0, BUTTON_START),
        "UP": (DPAD_UP, 0),
        "DOWN": (DPAD_DOWN, 0),
        "LEFT": (DPAD_LEFT, 0),
        "RIGHT": (DPAD_RIGHT, 0),
    }
    for idx, name in enumerate(codec.action_names):
        assert codec.to_joypad_mask(idx) == expected[name]


def test_invalid_action_raises() -> None:
    """Out-of-range actions raise ValueError."""
    codec = get_action_codec(POKERED_PUFFER_V0_ID)
    with pytest.raises(ValueError, match="out of range"):
        codec.to_pyboy_button(-1)
    with pytest.raises(ValueError, match="out of range"):
        codec.to_joypad_mask(codec.num_actions)
