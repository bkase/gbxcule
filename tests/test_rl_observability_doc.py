from __future__ import annotations

from pathlib import Path

from gbxcule.rl.train_log_schema import iter_jsonl, validate_meta, validate_record


def test_observability_fixture_schema() -> None:
    fixture = Path(__file__).parent / "fixtures" / "train_log_example.jsonl"
    payloads = list(iter_jsonl(fixture))
    assert payloads, "fixture must contain records"
    meta_wrapper = payloads[0]
    assert "meta" in meta_wrapper
    meta = meta_wrapper["meta"]
    errors = validate_meta(meta)
    assert not errors, f"meta validation failed: {errors}"

    num_actions = meta.get("num_actions")
    record = payloads[1]
    errors = validate_record(record, num_actions=num_actions)
    assert not errors, f"record validation failed: {errors}"
