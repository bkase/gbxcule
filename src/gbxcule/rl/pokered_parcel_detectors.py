"""Pokemon Red parcel detectors (RAM-based)."""

from __future__ import annotations

from typing import Final, cast

try:
    import torch
except ImportError as exc:  # pragma: no cover - torch is a hard requirement for RL
    raise RuntimeError(
        "Torch is required for gbxcule.rl.pokered_parcel_detectors. "
        "Install with `uv sync`."
    ) from exc

EVENTS_START: Final[int] = 0xD747
EVENTS_LENGTH: Final[int] = 320

WNUM_BAG_ITEMS_ADDR: Final[int] = 0xD31D
WBAG_ITEMS_ADDR: Final[int] = 0xD31E
BAG_CAPACITY: Final[int] = 20
BAG_BYTES: Final[int] = 2 * BAG_CAPACITY

OAKS_PARCEL_ITEM_ID: Final[int] = 0x46

EVENT_OAK_GOT_PARCEL_ADDR: Final[int] = 0xD74E
EVENT_OAK_GOT_PARCEL_BIT: Final[int] = 0
EVENT_GOT_OAKS_PARCEL_BIT: Final[int] = 1

EVENT_OAK_GOT_PARCEL_INDEX: Final[int] = EVENT_OAK_GOT_PARCEL_ADDR - EVENTS_START

_MEM_MIN_LEN: Final[int] = max(WNUM_BAG_ITEMS_ADDR + 1, WBAG_ITEMS_ADDR + BAG_BYTES)
_EVENTS_MIN_LEN: Final[int] = EVENT_OAK_GOT_PARCEL_INDEX + 1


def _normalize_u8_2d(t: torch.Tensor, *, name: str) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if t.dtype is not torch.uint8:
        raise ValueError(f"{name} must be torch.uint8, got {t.dtype}")
    if t.ndim == 1:
        t = t.unsqueeze(0)
    elif t.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D, got shape {tuple(t.shape)}")
    return t


def _validate_inputs(
    mem_u8: torch.Tensor, events_u8: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mem_u8 = _normalize_u8_2d(mem_u8, name="mem_u8")
    events_u8 = _normalize_u8_2d(events_u8, name="events_u8")
    if mem_u8.shape[0] != events_u8.shape[0]:
        raise ValueError(
            "mem_u8 and events_u8 must have matching batch size, "
            f"got {mem_u8.shape[0]} vs {events_u8.shape[0]}"
        )
    if mem_u8.shape[1] < _MEM_MIN_LEN:
        raise ValueError(
            f"mem_u8 must have at least {_MEM_MIN_LEN} bytes, got {mem_u8.shape[1]}"
        )
    if events_u8.shape[1] < _EVENTS_MIN_LEN:
        raise ValueError(
            f"events_u8 must have at least {_EVENTS_MIN_LEN} bytes, "
            f"got {events_u8.shape[1]}"
        )
    return mem_u8, events_u8


def _event_bit(events_u8: torch.Tensor, *, index: int, bit: int) -> torch.BoolTensor:
    mask = 1 << bit
    return cast(torch.BoolTensor, (events_u8[:, index] & mask) != 0)


def has_parcel(mem_u8: torch.Tensor, events_u8: torch.Tensor) -> torch.BoolTensor:
    """Return True when Oak's Parcel is present in the bag."""
    mem_u8, _ = _validate_inputs(mem_u8, events_u8)
    bag = mem_u8[:, WBAG_ITEMS_ADDR : WBAG_ITEMS_ADDR + BAG_BYTES]
    item_ids = bag[:, 0::2]
    num_items = mem_u8[:, WNUM_BAG_ITEMS_ADDR].to(torch.int64)
    idx = torch.arange(BAG_CAPACITY, device=mem_u8.device)
    mask = idx < num_items[:, None]
    return cast(torch.BoolTensor, ((item_ids == OAKS_PARCEL_ITEM_ID) & mask).any(dim=1))


def delivered_parcel(mem_u8: torch.Tensor, events_u8: torch.Tensor) -> torch.BoolTensor:
    """Return True when Oak has received the parcel."""
    _, events_u8 = _validate_inputs(mem_u8, events_u8)
    return _event_bit(
        events_u8, index=EVENT_OAK_GOT_PARCEL_INDEX, bit=EVENT_OAK_GOT_PARCEL_BIT
    )
