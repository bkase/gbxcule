"""SM83/LR35902 opcode specification tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


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


def _build_table(
    prefix: str, overrides: Mapping[int, OpcodeSpec]
) -> tuple[OpcodeSpec, ...]:
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

_CB_OVERRIDES: dict[int, OpcodeSpec] = {}

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


def _add_cb_spec(spec: OpcodeSpec) -> None:
    _CB_OVERRIDES[spec.opcode] = spec


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


def _add_alu_families() -> None:
    alu_ops = [
        ("ADD A,", 0x80, "add_a_r8", "add_a_hl", "add_a_d8", 0xC6),
        ("ADC A,", 0x88, "adc_a_r8", "adc_a_hl", "adc_a_d8", 0xCE),
        ("SUB ", 0x90, "sub_a_r8", "sub_a_hl", "sub_a_d8", 0xD6),
        ("SBC A,", 0x98, "sbc_a_r8", "sbc_a_hl", "sbc_a_d8", 0xDE),
        ("AND ", 0xA0, "and_a_r8", "and_a_hl", "and_a_d8", 0xE6),
        ("XOR ", 0xA8, "xor_a_r8", "xor_a_hl", "xor_a_d8", 0xEE),
        ("OR ", 0xB0, "or_a_r8", "or_a_hl", "or_a_d8", 0xF6),
        ("CP ", 0xB8, "cp_a_r8", "cp_a_hl", "cp_a_d8", 0xFE),
    ]

    for prefix, base, tmpl_r, tmpl_hl, tmpl_d8, imm_opcode in alu_ops:
        for idx, reg in enumerate(_REG_ORDER):
            opcode = base + idx
            if reg == "(HL)":
                _add_spec(
                    _spec(
                        opcode=opcode,
                        mnemonic=f"{prefix}(HL)",
                        length=1,
                        cycles=(8,),
                        template_key=tmpl_hl,
                        replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
                        group="alu",
                    )
                )
            else:
                _add_spec(
                    _spec(
                        opcode=opcode,
                        mnemonic=f"{prefix}{reg}",
                        length=1,
                        cycles=(4,),
                        template_key=tmpl_r,
                        replacements={"REG_i": _REG_VAR[reg]},
                        group="alu",
                    )
                )
        _add_spec(
            _spec(
                opcode=imm_opcode,
                mnemonic=f"{prefix}d8",
                length=2,
                cycles=(8,),
                template_key=tmpl_d8,
                group="alu",
            )
        )

    # INC/DEC r8 (including (HL))
    inc_codes = [0x04, 0x0C, 0x14, 0x1C, 0x24, 0x2C, 0x34, 0x3C]
    dec_codes = [0x05, 0x0D, 0x15, 0x1D, 0x25, 0x2D, 0x35, 0x3D]
    for idx, reg in enumerate(_REG_ORDER):
        inc_opcode = inc_codes[idx]
        dec_opcode = dec_codes[idx]
        if reg == "(HL)":
            _add_spec(
                _spec(
                    opcode=inc_opcode,
                    mnemonic="INC (HL)",
                    length=1,
                    cycles=(12,),
                    template_key="inc_hl",
                    replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
                    group="alu",
                )
            )
            _add_spec(
                _spec(
                    opcode=dec_opcode,
                    mnemonic="DEC (HL)",
                    length=1,
                    cycles=(12,),
                    template_key="dec_hl",
                    replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
                    group="alu",
                )
            )
        else:
            _add_spec(
                _spec(
                    opcode=inc_opcode,
                    mnemonic=f"INC {reg}",
                    length=1,
                    cycles=(4,),
                    template_key="inc_r8",
                    replacements={"REG_i": _REG_VAR[reg]},
                    group="alu",
                )
            )
            _add_spec(
                _spec(
                    opcode=dec_opcode,
                    mnemonic=f"DEC {reg}",
                    length=1,
                    cycles=(4,),
                    template_key="dec_r8",
                    replacements={"REG_i": _REG_VAR[reg]},
                    group="alu",
                )
            )

    # Misc ALU ops
    _add_spec(
        _spec(
            opcode=0x27,
            mnemonic="DAA",
            length=1,
            cycles=(4,),
            template_key="daa",
            group="alu",
        )
    )
    _add_spec(
        _spec(
            opcode=0x2F,
            mnemonic="CPL",
            length=1,
            cycles=(4,),
            template_key="cpl",
            group="alu",
        )
    )
    _add_spec(
        _spec(
            opcode=0x37,
            mnemonic="SCF",
            length=1,
            cycles=(4,),
            template_key="scf",
            group="alu",
        )
    )
    _add_spec(
        _spec(
            opcode=0x3F,
            mnemonic="CCF",
            length=1,
            cycles=(4,),
            template_key="ccf",
            group="alu",
        )
    )


_add_alu_families()


def _add_bitops_families() -> None:
    rotates = [
        (0x07, "RLCA", "rlca"),
        (0x0F, "RRCA", "rrca"),
        (0x17, "RLA", "rla"),
        (0x1F, "RRA", "rra"),
    ]
    for opcode, mnemonic, template_key in rotates:
        _add_spec(
            _spec(
                opcode=opcode,
                mnemonic=mnemonic,
                length=1,
                cycles=(4,),
                template_key=template_key,
                group="bitops",
            )
        )


_add_bitops_families()


def _add_alu16_families() -> None:
    # ADD HL, rr
    add_hl = [
        (0x09, "BC", ("b_i", "c_i"), "add_hl_r16"),
        (0x19, "DE", ("d_i", "e_i"), "add_hl_r16"),
        (0x29, "HL", ("h_i", "l_i"), "add_hl_r16"),
        (0x39, "SP", None, "add_hl_sp"),
    ]
    for opcode, reg, regs, template_key in add_hl:
        replacements = {"HREG_i": "h_i", "LREG_i": "l_i"}
        if regs:
            replacements.update({"SRC_HREG_i": regs[0], "SRC_LREG_i": regs[1]})
        _add_spec(
            _spec(
                opcode=opcode,
                mnemonic=f"ADD HL,{reg}",
                length=1,
                cycles=(8,),
                template_key=template_key,
                replacements=replacements,
                group="alu16",
            )
        )

    # INC rr
    inc_rr = [
        (0x03, "BC", ("b_i", "c_i"), "inc_r16"),
        (0x13, "DE", ("d_i", "e_i"), "inc_r16"),
        (0x23, "HL", ("h_i", "l_i"), "inc_r16"),
        (0x33, "SP", None, "inc_sp"),
    ]
    for opcode, reg, regs, template_key in inc_rr:
        replacements = {}
        if regs:
            replacements = {"HREG_i": regs[0], "LREG_i": regs[1]}
        _add_spec(
            _spec(
                opcode=opcode,
                mnemonic=f"INC {reg}",
                length=1,
                cycles=(8,),
                template_key=template_key,
                replacements=replacements,
                group="alu16",
            )
        )

    # DEC rr
    dec_rr = [
        (0x0B, "BC", ("b_i", "c_i"), "dec_r16"),
        (0x1B, "DE", ("d_i", "e_i"), "dec_r16"),
        (0x2B, "HL", ("h_i", "l_i"), "dec_r16"),
        (0x3B, "SP", None, "dec_sp"),
    ]
    for opcode, reg, regs, template_key in dec_rr:
        replacements = {}
        if regs:
            replacements = {"HREG_i": regs[0], "LREG_i": regs[1]}
        _add_spec(
            _spec(
                opcode=opcode,
                mnemonic=f"DEC {reg}",
                length=1,
                cycles=(8,),
                template_key=template_key,
                replacements=replacements,
                group="alu16",
            )
        )

    # ADD SP, e8
    _add_spec(
        _spec(
            opcode=0xE8,
            mnemonic="ADD SP,e8",
            length=2,
            cycles=(16,),
            template_key="add_sp_e8",
            group="alu16",
        )
    )

    # LD HL, SP+e8
    _add_spec(
        _spec(
            opcode=0xF8,
            mnemonic="LD HL,SP+e8",
            length=2,
            cycles=(12,),
            template_key="ld_hl_sp_e8",
            replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
            group="alu16",
        )
    )


_add_alu16_families()


def _add_control_flow_families() -> None:
    # JR NC/C
    _add_spec(
        _spec(
            opcode=0x30,
            mnemonic="JR NC,r8",
            length=2,
            cycles=(12, 8),
            template_key="jr_nc_r8",
            group="jumps",
        )
    )
    _add_spec(
        _spec(
            opcode=0x38,
            mnemonic="JR C,r8",
            length=2,
            cycles=(12, 8),
            template_key="jr_c_r8",
            group="jumps",
        )
    )

    # JP cc, a16
    jp_cc = [
        (0xC2, "NZ", "jp_nz_a16"),
        (0xCA, "Z", "jp_z_a16"),
        (0xD2, "NC", "jp_nc_a16"),
        (0xDA, "C", "jp_c_a16"),
    ]
    for opcode, cond, template_key in jp_cc:
        _add_spec(
            _spec(
                opcode=opcode,
                mnemonic=f"JP {cond},a16",
                length=3,
                cycles=(16, 12),
                template_key=template_key,
                group="jumps",
            )
        )

    # JP (HL)
    _add_spec(
        _spec(
            opcode=0xE9,
            mnemonic="JP (HL)",
            length=1,
            cycles=(4,),
            template_key="jp_hl",
            replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
            group="jumps",
        )
    )

    # CALL a16 + CALL cc
    _add_spec(
        _spec(
            opcode=0xCD,
            mnemonic="CALL a16",
            length=3,
            cycles=(24,),
            template_key="call_a16",
            group="stack",
        )
    )
    call_cc = [
        (0xC4, "NZ", "call_nz_a16"),
        (0xCC, "Z", "call_z_a16"),
        (0xD4, "NC", "call_nc_a16"),
        (0xDC, "C", "call_c_a16"),
    ]
    for opcode, cond, template_key in call_cc:
        _add_spec(
            _spec(
                opcode=opcode,
                mnemonic=f"CALL {cond},a16",
                length=3,
                cycles=(24, 12),
                template_key=template_key,
                group="stack",
            )
        )

    # RET + RETI + RET cc
    _add_spec(
        _spec(
            opcode=0xC9,
            mnemonic="RET",
            length=1,
            cycles=(16,),
            template_key="ret",
            group="stack",
        )
    )
    _add_spec(
        _spec(
            opcode=0xD9,
            mnemonic="RETI",
            length=1,
            cycles=(16,),
            template_key="reti",
            group="stack",
        )
    )
    ret_cc = [
        (0xC0, "NZ", "ret_nz"),
        (0xC8, "Z", "ret_z"),
        (0xD0, "NC", "ret_nc"),
        (0xD8, "C", "ret_c"),
    ]
    for opcode, cond, template_key in ret_cc:
        _add_spec(
            _spec(
                opcode=opcode,
                mnemonic=f"RET {cond}",
                length=1,
                cycles=(20, 8),
                template_key=template_key,
                group="stack",
            )
        )

    # RST n
    rst_ops = [
        (0xC7, "00H", "rst_00"),
        (0xCF, "08H", "rst_08"),
        (0xD7, "10H", "rst_10"),
        (0xDF, "18H", "rst_18"),
        (0xE7, "20H", "rst_20"),
        (0xEF, "28H", "rst_28"),
        (0xF7, "30H", "rst_30"),
        (0xFF, "38H", "rst_38"),
    ]
    for opcode, vec, template_key in rst_ops:
        _add_spec(
            _spec(
                opcode=opcode,
                mnemonic=f"RST {vec}",
                length=1,
                cycles=(16,),
                template_key=template_key,
                group="stack",
            )
        )

    # PUSH rr + POP rr
    push_rr = [
        (0xC5, "BC", ("b_i", "c_i"), "push_r16"),
        (0xD5, "DE", ("d_i", "e_i"), "push_r16"),
        (0xE5, "HL", ("h_i", "l_i"), "push_r16"),
    ]
    for opcode, reg, regs, template_key in push_rr:
        _add_spec(
            _spec(
                opcode=opcode,
                mnemonic=f"PUSH {reg}",
                length=1,
                cycles=(16,),
                template_key=template_key,
                replacements={"HREG_i": regs[0], "LREG_i": regs[1]},
                group="stack",
            )
        )
    _add_spec(
        _spec(
            opcode=0xF5,
            mnemonic="PUSH AF",
            length=1,
            cycles=(16,),
            template_key="push_af",
            group="stack",
        )
    )

    pop_rr = [
        (0xC1, "BC", ("b_i", "c_i"), "pop_r16"),
        (0xD1, "DE", ("d_i", "e_i"), "pop_r16"),
        (0xE1, "HL", ("h_i", "l_i"), "pop_r16"),
    ]
    for opcode, reg, regs, template_key in pop_rr:
        _add_spec(
            _spec(
                opcode=opcode,
                mnemonic=f"POP {reg}",
                length=1,
                cycles=(12,),
                template_key=template_key,
                replacements={"HREG_i": regs[0], "LREG_i": regs[1]},
                group="stack",
            )
        )
    _add_spec(
        _spec(
            opcode=0xF1,
            mnemonic="POP AF",
            length=1,
            cycles=(12,),
            template_key="pop_af",
            group="stack",
        )
    )


def _add_cb_families() -> None:
    cb_rotates = [
        ("RLC", 0x00, "cb_rlc_r8", "cb_rlc_hl"),
        ("RRC", 0x08, "cb_rrc_r8", "cb_rrc_hl"),
        ("RL", 0x10, "cb_rl_r8", "cb_rl_hl"),
        ("RR", 0x18, "cb_rr_r8", "cb_rr_hl"),
        ("SLA", 0x20, "cb_sla_r8", "cb_sla_hl"),
        ("SRA", 0x28, "cb_sra_r8", "cb_sra_hl"),
        ("SWAP", 0x30, "cb_swap_r8", "cb_swap_hl"),
        ("SRL", 0x38, "cb_srl_r8", "cb_srl_hl"),
    ]
    for mnemonic, base, tmpl_r8, tmpl_hl in cb_rotates:
        for idx, reg in enumerate(_REG_ORDER):
            opcode = base + idx
            if reg == "(HL)":
                _add_cb_spec(
                    _spec(
                        opcode=opcode,
                        mnemonic=f"{mnemonic} (HL)",
                        length=2,
                        cycles=(16,),
                        template_key=tmpl_hl,
                        replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
                        group="bitops",
                    )
                )
            else:
                _add_cb_spec(
                    _spec(
                        opcode=opcode,
                        mnemonic=f"{mnemonic} {reg}",
                        length=2,
                        cycles=(8,),
                        template_key=tmpl_r8,
                        replacements={"REG_i": _REG_VAR[reg]},
                        group="bitops",
                    )
                )

    for bit in range(8):
        bit_base = 0x40 + (bit * 8)
        for idx, reg in enumerate(_REG_ORDER):
            opcode = bit_base + idx
            if reg == "(HL)":
                _add_cb_spec(
                    _spec(
                        opcode=opcode,
                        mnemonic=f"BIT {bit},(HL)",
                        length=2,
                        cycles=(16,),
                        template_key="cb_bit_hl",
                        replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
                        group="bitops",
                    )
                )
            else:
                _add_cb_spec(
                    _spec(
                        opcode=opcode,
                        mnemonic=f"BIT {bit},{reg}",
                        length=2,
                        cycles=(8,),
                        template_key="cb_bit_r8",
                        replacements={"REG_i": _REG_VAR[reg]},
                        group="bitops",
                    )
                )

    for bit in range(8):
        res_base = 0x80 + (bit * 8)
        for idx, reg in enumerate(_REG_ORDER):
            opcode = res_base + idx
            if reg == "(HL)":
                _add_cb_spec(
                    _spec(
                        opcode=opcode,
                        mnemonic=f"RES {bit},(HL)",
                        length=2,
                        cycles=(16,),
                        template_key="cb_res_hl",
                        replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
                        group="bitops",
                    )
                )
            else:
                _add_cb_spec(
                    _spec(
                        opcode=opcode,
                        mnemonic=f"RES {bit},{reg}",
                        length=2,
                        cycles=(8,),
                        template_key="cb_res_r8",
                        replacements={"REG_i": _REG_VAR[reg]},
                        group="bitops",
                    )
                )

    for bit in range(8):
        set_base = 0xC0 + (bit * 8)
        for idx, reg in enumerate(_REG_ORDER):
            opcode = set_base + idx
            if reg == "(HL)":
                _add_cb_spec(
                    _spec(
                        opcode=opcode,
                        mnemonic=f"SET {bit},(HL)",
                        length=2,
                        cycles=(16,),
                        template_key="cb_set_hl",
                        replacements={"HREG_i": "h_i", "LREG_i": "l_i"},
                        group="bitops",
                    )
                )
            else:
                _add_cb_spec(
                    _spec(
                        opcode=opcode,
                        mnemonic=f"SET {bit},{reg}",
                        length=2,
                        cycles=(8,),
                        template_key="cb_set_r8",
                        replacements={"REG_i": _REG_VAR[reg]},
                        group="bitops",
                    )
                )


_add_control_flow_families()
_add_cb_families()


UNPREFIXED_SPECS = _build_table("OP", _UNPREFIXED_OVERRIDES)
CB_SPECS = _build_table("CB", _CB_OVERRIDES)


def iter_unprefixed() -> tuple[OpcodeSpec, ...]:
    return UNPREFIXED_SPECS


def iter_cb() -> tuple[OpcodeSpec, ...]:
    return CB_SPECS


def implemented_specs(prefix: str = "unprefixed") -> list[OpcodeSpec]:
    table = UNPREFIXED_SPECS if prefix == "unprefixed" else CB_SPECS
    return [spec for spec in table if spec.implemented]


def implemented_count(prefix: str = "unprefixed") -> int:
    return len(implemented_specs(prefix))
