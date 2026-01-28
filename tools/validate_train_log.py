#!/usr/bin/env python3
"""Validate train_log.jsonl schema and numeric sanity (exit 0/1)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gbxcule.rl.train_log_schema import iter_jsonl, validate_meta, validate_record


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="Path to train_log.jsonl")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    log_path = Path(args.log)
    if not log_path.exists():
        raise FileNotFoundError(f"log not found: {log_path}")

    errors: list[str] = []
    record_count = 0
    meta = None
    last_env_steps = None
    last_opt_steps = None

    for payload in iter_jsonl(log_path):
        if "meta" in payload:
            if meta is not None:
                errors.append("multiple meta records found")
            meta = payload["meta"]
            if not isinstance(meta, dict):
                errors.append("meta record must be an object")
                continue
            errors.extend(validate_meta(meta))
            continue

        record_count += 1
        num_actions = None
        if isinstance(meta, dict):
            num_actions = meta.get("num_actions")
        rec_errors = validate_record(payload, num_actions=num_actions)
        errors.extend(rec_errors)
        if "env_steps" in payload and isinstance(payload["env_steps"], int):
            if last_env_steps is not None and payload["env_steps"] < last_env_steps:
                errors.append("env_steps not monotonic")
            last_env_steps = payload["env_steps"]
        if "opt_steps" in payload and isinstance(payload["opt_steps"], int):
            if last_opt_steps is not None and payload["opt_steps"] < last_opt_steps:
                errors.append("opt_steps not monotonic")
            last_opt_steps = payload["opt_steps"]

    summary = {
        "ok": not errors,
        "records": record_count,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
