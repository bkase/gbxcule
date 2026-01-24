"""SM83/LR35902 opcode specification tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class OpcodeSpec:
    opcode: int
    mnemonic: str
    length: int
    cycles: tuple[int, ...]
    template_key: str | None
    replacements: dict[str, str]
    group: str
    implemented: bool


def _make_unimplemented(opcode: int, prefix: str) -> OpcodeSpec:
    return OpcodeSpec(
        opcode=opcode,
        mnemonic=f"UNIMPL_{prefix}{opcode:02X}",
        length=1,
        cycles=(0,),
        template_key=None,
        replacements={},
        group="unimplemented",
        implemented=False,
    )


def _build_table(prefix: str, overrides: Mapping[int, OpcodeSpec]) -> tuple[OpcodeSpec, ...]:
    table = [_make_unimplemented(opcode, prefix) for opcode in range(256)]
    for opcode, spec in overrides.items():
        if spec.opcode != opcode:
            raise ValueError(f"Opcode mismatch for {prefix}{opcode:02X}")
        table[opcode] = spec
    return tuple(table)


def _spec(
    *,
    opcode: int,
    mnemonic: str,
    length: int,
    cycles: tuple[int, ...],
    template_key: str,
    replacements: dict[str, str] | None = None,
    group: str,
) -> OpcodeSpec:
    return OpcodeSpec(
        opcode=opcode,
        mnemonic=mnemonic,
        length=length,
        cycles=cycles,
        template_key=template_key,
        replacements=replacements or {},
        group=group,
        implemented=True,
    )


_UNPREFIXED_OVERRIDES: dict[int, OpcodeSpec] = {
    0x00: _spec(
        opcode=0x00,
        mnemonic="NOP",
        length=1,
        cycles=(4,),
        template_key="nop",
        group="misc",
    ),
    0xC3: _spec(
        opcode=0xC3,
        mnemonic="JP a16",
        length=3,
        cycles=(16,),
        template_key="jp_a16",
        group="jumps",
    ),
    0x18: _spec(
        opcode=0x18,
        mnemonic="JR r8",
        length=2,
        cycles=(12,),
        template_key="jr_r8",
        group="jumps",
    ),
    0x20: _spec(
        opcode=0x20,
        mnemonic="JR NZ,r8",
        length=2,
        cycles=(12, 8),
        template_key="jr_nz_r8",
        group="jumps",
    ),
    0x28: _spec(
        opcode=0x28,
        mnemonic="JR Z,r8",
        length=2,
        cycles=(12, 8),
        template_key="jr_z_r8",
        group="jumps",
    ),
    0x3E: _spec(
        opcode=0x3E,
        mnemonic="LD A,d8",
        length=2,
        cycles=(8,),
        template_key="ld_r8_d8",
        replacements={"REG_i": "a_i"},
        group="loads",
    ),
    0x06: _spec(
        opcode=0x06,
        mnemonic="LD B,d8",
        length=2,
        cycles=(8,),
        template_key="ld_r8_d8",
        replacements={"REG_i": "b_i"},
        group="loads",
    ),
    0x21: _spec(
        opcode=0x21,
        mnemonic="LD HL,d16",
        length=3,
        cycles=(12,),
        template_key="ld_r16_d16",
        replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
        group="loads",
    ),
    0x3C: _spec(
        opcode=0x3C,
        mnemonic="INC A",
        length=1,
        cycles=(4,),
        template_key="inc_r8",
        replacements={"REG_i": "a_i"},
        group="alu",
    ),
    0x04: _spec(
        opcode=0x04,
        mnemonic="INC B",
        length=1,
        cycles=(4,),
        template_key="inc_r8",
        replacements={"REG_i": "b_i"},
        group="alu",
    ),
    0x05: _spec(
        opcode=0x05,
        mnemonic="DEC B",
        length=1,
        cycles=(4,),
        template_key="dec_r8",
        replacements={"REG_i": "b_i"},
        group="alu",
    ),
    0x23: _spec(
        opcode=0x23,
        mnemonic="INC HL",
        length=1,
        cycles=(8,),
        template_key="inc_r16",
        replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
        group="alu",
    ),
    0x80: _spec(
        opcode=0x80,
        mnemonic="ADD A,B",
        length=1,
        cycles=(4,),
        template_key="add_a_r8",
        replacements={"REG_i": "b_i"},
        group="alu",
    ),
    0xE6: _spec(
        opcode=0xE6,
        mnemonic="AND d8",
        length=2,
        cycles=(8,),
        template_key="and_a_d8",
        group="alu",
    ),
    0x77: _spec(
        opcode=0x77,
        mnemonic="LD (HL),A",
        length=1,
        cycles=(8,),
        template_key="ld_hl_r8",
        replacements={"HREG_i": "h_i", "LREG_i": "l_i", "SRC_i": "a_i"},
        group="loads",
    ),
    0x46: _spec(
        opcode=0x46,
        mnemonic="LD B,(HL)",
        length=1,
        cycles=(8,),
        template_key="ld_r8_hl",
        replacements={"HREG_i": "h_i", "LREG_i": "l_i", "DST_i": "b_i"},
        group="loads",
    ),
    0x7E: _spec(
        opcode=0x7E,
        mnemonic="LD A,(HL)",
        length=1,
        cycles=(8,),
        template_key="ld_r8_hl",
        replacements={"HREG_i": "h_i", "LREG_i": "l_i", "DST_i": "a_i"},
        group="loads",
    ),
}


UNPREFIXED_SPECS = _build_table("OP", _UNPREFIXED_OVERRIDES)
CB_SPECS = _build_table("CB", {})


def iter_unprefixed() -> tuple[OpcodeSpec, ...]:
    return UNPREFIXED_SPECS


def iter_cb() -> tuple[OpcodeSpec, ...]:
    return CB_SPECS


def implemented_specs(prefix: str = "unprefixed") -> list[OpcodeSpec]:
    table = UNPREFIXED_SPECS if prefix == "unprefixed" else CB_SPECS
    return [spec for spec in table if spec.implemented]


def implemented_count(prefix: str = "unprefixed") -> int:
    return len(implemented_specs(prefix))
