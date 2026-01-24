"""ISA table coverage tests for SM83 opcode specs."""

from __future__ import annotations

from gbxcule.core import isa_sm83


def _assert_table_complete(table: tuple[isa_sm83.OpcodeSpec, ...]) -> None:
    assert len(table) == 256
    opcodes = [spec.opcode for spec in table]
    assert len(set(opcodes)) == 256
    assert set(opcodes) == set(range(256))
    for spec in table:
        assert isinstance(spec.mnemonic, str) and spec.mnemonic
        assert isinstance(spec.length, int) and spec.length >= 1
        assert isinstance(spec.cycles, tuple) and len(spec.cycles) >= 1
        assert spec.implemented == (spec.template_key is not None)


def test_unprefixed_table_complete() -> None:
    _assert_table_complete(isa_sm83.UNPREFIXED_SPECS)


def test_cb_table_complete() -> None:
    _assert_table_complete(isa_sm83.CB_SPECS)
