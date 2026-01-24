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

_REG_ORDER = ("B", "C", "D", "E", "H", "L", "(HL)", "A")
_REG_VAR = {
    "A": "a_i",
    "B": "b_i",
    "C": "c_i",
    "D": "d_i",
    "E": "e_i",
    "H": "h_i",
    "L": "l_i",
}


def _add_spec(spec: OpcodeSpec) -> None:
    _UNPREFIXED_OVERRIDES[spec.opcode] = spec


def _add_load_families() -> None:
    # LD r8, r8 (0x40..0x7F), excluding HALT (0x76)
    for dst_idx, dst in enumerate(_REG_ORDER):
        for src_idx, src in enumerate(_REG_ORDER):
            opcode = 0x40 | (dst_idx << 3) | src_idx
            if opcode == 0x76:
                continue
            if dst == "(HL)" and src == "(HL)":
                continue
            if dst == "(HL)":
                _add_spec(
                    _spec(
                        opcode=opcode,
                        mnemonic=f"LD (HL),{src}",
                        length=1,
                        cycles=(8,),
                        template_key="ld_hl_r8",
                        replacements={
                            "HREG_i": "h_i",
                            "LREG_i": "l_i",
                            "SRC_i": _REG_VAR[src],
                        },
                        group="loads",
                    )
                )
            elif src == "(HL)":
                _add_spec(
                    _spec(
                        opcode=opcode,
                        mnemonic=f"LD {dst},(HL)",
                        length=1,
                        cycles=(8,),
                        template_key="ld_r8_hl",
                        replacements={
                            "HREG_i": "h_i",
                            "LREG_i": "l_i",
                            "DST_i": _REG_VAR[dst],
                        },
                        group="loads",
                    )
                )
            else:
                _add_spec(
                    _spec(
                        opcode=opcode,
                        mnemonic=f"LD {dst},{src}",
                        length=1,
                        cycles=(4,),
                        template_key="ld_r8_r8",
                        replacements={
                            "DST_i": _REG_VAR[dst],
                            "SRC_i": _REG_VAR[src],
                        },
                        group="loads",
                    )
                )

    # LD r8, d8 (including (HL))
    r8_d8 = [
        (0x06, "B"),
        (0x0E, "C"),
        (0x16, "D"),
        (0x1E, "E"),
        (0x26, "H"),
        (0x2E, "L"),
        (0x36, "(HL)"),
        (0x3E, "A"),
    ]
    for opcode, reg in r8_d8:
        if reg == "(HL)":
            _add_spec(
                _spec(
                    opcode=opcode,
                    mnemonic="LD (HL),d8",
                    length=2,
                    cycles=(12,),
                    template_key="ld_hl_d8",
                    replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
                    group="loads",
                )
            )
        else:
            _add_spec(
                _spec(
                    opcode=opcode,
                    mnemonic=f"LD {reg},d8",
                    length=2,
                    cycles=(8,),
                    template_key="ld_r8_d8",
                    replacements={"REG_i": _REG_VAR[reg]},
                    group="loads",
                )
            )

    # LD rr, d16
    rr_d16 = [
        (0x01, "BC"),
        (0x11, "DE"),
        (0x21, "HL"),
        (0x31, "SP"),
    ]
    rr_regs = {
        "BC": ("b_i", "c_i"),
        "DE": ("d_i", "e_i"),
        "HL": ("h_i", "l_i"),
    }
    for opcode, reg in rr_d16:
        if reg == "SP":
            _add_spec(
                _spec(
                    opcode=opcode,
                    mnemonic="LD SP,d16",
                    length=3,
                    cycles=(12,),
                    template_key="ld_sp_d16",
                    group="loads",
                )
            )
        else:
            hi, lo = rr_regs[reg]
            _add_spec(
                _spec(
                    opcode=opcode,
                    mnemonic=f"LD {reg},d16",
                    length=3,
                    cycles=(12,),
                    template_key="ld_r16_d16",
                    replacements={"HREG_i": hi, "LREG_i": lo},
                    group="loads",
                )
            )

    # LD (a16),SP
    _add_spec(
        _spec(
            opcode=0x08,
            mnemonic="LD (a16),SP",
            length=3,
            cycles=(20,),
            template_key="ld_a16_sp",
            group="loads",
        )
    )

    # LD (BC),A / LD (DE),A / LD A,(BC) / LD A,(DE)
    _add_spec(
        _spec(
            opcode=0x02,
            mnemonic="LD (BC),A",
            length=1,
            cycles=(8,),
            template_key="ld_bc_a",
            group="loads",
        )
    )
    _add_spec(
        _spec(
            opcode=0x12,
            mnemonic="LD (DE),A",
            length=1,
            cycles=(8,),
            template_key="ld_de_a",
            group="loads",
        )
    )
    _add_spec(
        _spec(
            opcode=0x0A,
            mnemonic="LD A,(BC)",
            length=1,
            cycles=(8,),
            template_key="ld_a_bc",
            group="loads",
        )
    )
    _add_spec(
        _spec(
            opcode=0x1A,
            mnemonic="LD A,(DE)",
            length=1,
            cycles=(8,),
            template_key="ld_a_de",
            group="loads",
        )
    )

    # LDI/LDD A <-> (HL)
    _add_spec(
        _spec(
            opcode=0x22,
            mnemonic="LDI (HL),A",
            length=1,
            cycles=(8,),
            template_key="ld_hl_inc_a",
            replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
            group="loads",
        )
    )
    _add_spec(
        _spec(
            opcode=0x32,
            mnemonic="LDD (HL),A",
            length=1,
            cycles=(8,),
            template_key="ld_hl_dec_a",
            replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
            group="loads",
        )
    )
    _add_spec(
        _spec(
            opcode=0x2A,
            mnemonic="LDI A,(HL)",
            length=1,
            cycles=(8,),
            template_key="ld_a_hl_inc",
            replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
            group="loads",
        )
    )
    _add_spec(
        _spec(
            opcode=0x3A,
            mnemonic="LDD A,(HL)",
            length=1,
            cycles=(8,),
            template_key="ld_a_hl_dec",
            replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
            group="loads",
        )
    )

    # LD (a16),A / LD A,(a16)
    _add_spec(
        _spec(
            opcode=0xEA,
            mnemonic="LD (a16),A",
            length=3,
            cycles=(16,),
            template_key="ld_a16_a",
            group="loads",
        )
    )
    _add_spec(
        _spec(
            opcode=0xFA,
            mnemonic="LD A,(a16)",
            length=3,
            cycles=(16,),
            template_key="ld_a_a16",
            group="loads",
        )
    )

    # LDH (a8),A / LDH A,(a8)
    _add_spec(
        _spec(
            opcode=0xE0,
            mnemonic="LDH (a8),A",
            length=2,
            cycles=(12,),
            template_key="ldh_a8_a",
            group="loads",
        )
    )
    _add_spec(
        _spec(
            opcode=0xF0,
            mnemonic="LDH A,(a8)",
            length=2,
            cycles=(12,),
            template_key="ldh_a_a8",
            group="loads",
        )
    )

    # LD (C),A / LD A,(C)
    _add_spec(
        _spec(
            opcode=0xE2,
            mnemonic="LD (C),A",
            length=1,
            cycles=(8,),
            template_key="ldh_c_a",
            group="loads",
        )
    )
    _add_spec(
        _spec(
            opcode=0xF2,
            mnemonic="LD A,(C)",
            length=1,
            cycles=(8,),
            template_key="ldh_a_c",
            group="loads",
        )
    )

    # LD SP,HL
    _add_spec(
        _spec(
            opcode=0xF9,
            mnemonic="LD SP,HL",
            length=1,
            cycles=(8,),
            template_key="ld_sp_hl",
            replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
            group="loads",
        )
    )


_add_load_families()


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
