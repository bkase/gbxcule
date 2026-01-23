"""Action codec definitions for mapping discrete actions to inputs.

This module is pure and has no heavy runtime dependencies. It provides a
versioned registry of action codecs so action semantics are explicit and
auditable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ActionCodec(Protocol):
    """Protocol for action codecs.

    Action codecs define a stable mapping from action indices to:
    - PyBoy button names
    - JOYP bitmasks (for Warp JOYP emulation)
    """

    name: str
    version: str
    action_names: tuple[str, ...]

    @property
    def num_actions(self) -> int: ...

    def validate_action(self, action: int) -> None: ...

    def to_pyboy_button(self, action: int) -> str | None: ...

    def to_joypad_mask(self, action: int) -> tuple[int, int]: ...


# JOYP bit conventions (low nibble, 1 bit per button)
# D-pad bits: Right, Left, Up, Down (0..3)
# Button bits: A, B, Select, Start (0..3)
DPAD_RIGHT = 1 << 0
DPAD_LEFT = 1 << 1
DPAD_UP = 1 << 2
DPAD_DOWN = 1 << 3

BUTTON_A = 1 << 0
BUTTON_B = 1 << 1
BUTTON_SELECT = 1 << 2
BUTTON_START = 1 << 3


@dataclass(frozen=True)
class ActionCodecDef:
    """Concrete action codec definition."""

    name: str
    version: str
    action_names: tuple[str, ...]
    _pyboy_buttons: tuple[str | None, ...]
    _dpad_masks: tuple[int, ...]
    _button_masks: tuple[int, ...]

    @property
    def num_actions(self) -> int:
        return len(self.action_names)

    def validate_action(self, action: int) -> None:
        if action < 0 or action >= self.num_actions:
            raise ValueError(f"Action {action} out of range [0, {self.num_actions})")

    def to_pyboy_button(self, action: int) -> str | None:
        self.validate_action(action)
        return self._pyboy_buttons[action]

    def to_joypad_mask(self, action: int) -> tuple[int, int]:
        self.validate_action(action)
        return (self._dpad_masks[action], self._button_masks[action])


LEGACY_V0_ID = "legacy_v0"
POKERED_PUFFER_V0_ID = "pokemonred_puffer_v0"


_LEGACY_V0 = ActionCodecDef(
    name="legacy",
    version="v0",
    action_names=(
        "NOOP",
        "UP",
        "DOWN",
        "LEFT",
        "RIGHT",
        "A",
        "B",
        "START",
        "SELECT",
    ),
    _pyboy_buttons=(
        None,
        "up",
        "down",
        "left",
        "right",
        "a",
        "b",
        "start",
        "select",
    ),
    _dpad_masks=(
        0,
        DPAD_UP,
        DPAD_DOWN,
        DPAD_LEFT,
        DPAD_RIGHT,
        0,
        0,
        0,
        0,
    ),
    _button_masks=(
        0,
        0,
        0,
        0,
        0,
        BUTTON_A,
        BUTTON_B,
        BUTTON_START,
        BUTTON_SELECT,
    ),
)

_POKERED_PUFFER_V0 = ActionCodecDef(
    name="pokemonred_puffer",
    version="v0",
    action_names=(
        "A",
        "B",
        "START",
        "UP",
        "DOWN",
        "LEFT",
        "RIGHT",
    ),
    _pyboy_buttons=(
        "a",
        "b",
        "start",
        "up",
        "down",
        "left",
        "right",
    ),
    _dpad_masks=(
        0,
        0,
        0,
        DPAD_UP,
        DPAD_DOWN,
        DPAD_LEFT,
        DPAD_RIGHT,
    ),
    _button_masks=(
        BUTTON_A,
        BUTTON_B,
        BUTTON_START,
        0,
        0,
        0,
        0,
    ),
)


_REGISTRY: dict[str, ActionCodecDef] = {
    LEGACY_V0_ID: _LEGACY_V0,
    POKERED_PUFFER_V0_ID: _POKERED_PUFFER_V0,
}


def get_action_codec(codec_id: str) -> ActionCodecDef:
    """Resolve an action codec by id."""
    codec = _REGISTRY.get(codec_id)
    if codec is None:
        raise ValueError(f"Unknown action codec: {codec_id}")
    return codec


def list_action_codecs() -> tuple[str, ...]:
    """Return the list of registered codec ids."""
    return tuple(_REGISTRY.keys())
