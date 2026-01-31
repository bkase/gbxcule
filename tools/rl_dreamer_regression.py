#!/usr/bin/env python3
"""Capture and compare Dreamer v3 regression baselines (JSON/JSONL)."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeGuard

DEFAULT_KEYS = [
    "Loss/world_model_loss",
    "Loss/observation_loss",
    "Loss/reward_loss",
    "Loss/state_loss",
    "Loss/continue_loss",
    "State/kl",
    "State/post_entropy",
    "State/prior_entropy",
    "Loss/policy_loss",
    "Loss/value_loss",
    "Grads/world_model",
    "Grads/actor",
    "Grads/critic",
    "action_entropy",
    "value_mean",
    "value_std",
    "adv_mean",
    "adv_std",
    "ret_p05",
    "ret_p95",
    "ret_scale",
    "replay_ratio",
    "replay_size",
    "ready_steps",
    "sps",
    "train_sps",
]


@dataclass
class InputPayload:
    records: list[dict[str, Any]]
    source: Path
    kind: str


@dataclass
class Baseline:
    schema_version: int
    created_at: str
    source: str
    input_kind: str
    env_steps_min: int | None
    env_steps_max: int | None
    count: int
    keys: list[str]
    metrics: dict[str, float]
    missing_keys: list[str]
    threshold_pct: float
    abs_tol: float


@dataclass
class CompareResult:
    ok: bool
    compared: int
    failures: int
    missing: int
    threshold_pct: float
    abs_tol: float
    details: list[dict[str, Any]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture", help="Capture baseline JSON")
    capture.add_argument("--input", required=True, help="Path to JSON/JSONL")
    capture.add_argument("--output", required=True, help="Baseline output JSON path")
    capture.add_argument(
        "--keys",
        default=None,
        help="Comma-separated metric keys (default: Dreamer v3 core)",
    )
    capture.add_argument(
        "--env-steps-min",
        type=int,
        default=None,
        help="Optional env_steps lower bound for JSONL",
    )
    capture.add_argument(
        "--env-steps-max",
        type=int,
        default=None,
        help="Optional env_steps upper bound for JSONL",
    )
    capture.add_argument(
        "--last",
        type=int,
        default=None,
        help="Use the last N records (after env_steps filtering)",
    )
    capture.add_argument(
        "--threshold-pct",
        type=float,
        default=0.15,
        help="Default percent threshold stored in baseline",
    )
    capture.add_argument(
        "--abs-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for near-zero baselines",
    )

    compare = subparsers.add_parser("compare", help="Compare against baseline")
    compare.add_argument("--baseline", required=True, help="Baseline JSON path")
    compare.add_argument("--input", required=True, help="Path to JSON/JSONL")
    compare.add_argument(
        "--threshold-pct",
        type=float,
        default=None,
        help="Override baseline threshold percentage",
    )
    compare.add_argument(
        "--abs-tol",
        type=float,
        default=None,
        help="Override baseline absolute tolerance",
    )
    compare.add_argument(
        "--last",
        type=int,
        default=None,
        help="Use the last N records (after env_steps filtering)",
    )
    return parser.parse_args()


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _median(values: Iterable[float]) -> float | None:
    data = sorted(values)
    if not data:
        return None
    idx = (len(data) - 1) // 2
    return float(data[idx])


def _load_records(path: Path) -> InputPayload:
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    if path.suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
        return InputPayload(records=records, source=path, kind="jsonl")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("JSON input must be an object")
    return InputPayload(records=[record], source=path, kind="json")


def _parse_keys(raw: str | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_KEYS)
    keys = [item.strip() for item in raw.split(",") if item.strip()]
    if not keys:
        raise ValueError("--keys must list at least one key")
    return keys


def _filter_records(
    records: list[dict[str, Any]],
    *,
    env_steps_min: int | None,
    env_steps_max: int | None,
    last: int | None,
) -> list[dict[str, Any]]:
    filtered = records
    if env_steps_min is not None or env_steps_max is not None:
        windowed: list[dict[str, Any]] = []
        for rec in filtered:
            steps = rec.get("env_steps")
            if not _is_number(steps):
                continue
            steps_val = float(steps)
            if env_steps_min is not None and steps_val < env_steps_min:
                continue
            if env_steps_max is not None and steps_val > env_steps_max:
                continue
            windowed.append(rec)
        filtered = windowed
    if last is not None:
        if last < 1:
            raise ValueError("--last must be >= 1")
        filtered = filtered[-last:]
    return filtered


def _compute_metrics(
    records: list[dict[str, Any]], keys: list[str]
) -> tuple[dict[str, float], list[str]]:
    metrics: dict[str, float] = {}
    missing: list[str] = []
    for key in keys:
        values: list[float] = []
        for rec in records:
            value = rec.get(key)
            if _is_number(value):
                values.append(float(value))
        median = _median(values)
        if median is None:
            missing.append(key)
            continue
        metrics[key] = median
    return metrics, missing


def _capture(args: argparse.Namespace) -> Baseline:
    payload = _load_records(Path(args.input))
    keys = _parse_keys(args.keys)
    if payload.kind == "jsonl":
        records = _filter_records(
            payload.records,
            env_steps_min=args.env_steps_min,
            env_steps_max=args.env_steps_max,
            last=args.last,
        )
        if not records:
            raise ValueError("No records matched the filter")
        env_steps = [
            float(rec["env_steps"])
            for rec in records
            if _is_number(rec.get("env_steps"))
        ]
        env_min = int(min(env_steps)) if env_steps else None
        env_max = int(max(env_steps)) if env_steps else None
    else:
        records = payload.records
        env_min = None
        env_max = None

    metrics, missing = _compute_metrics(records, keys)

    baseline = Baseline(
        schema_version=1,
        created_at=datetime.now(UTC).isoformat(),
        source=str(payload.source),
        input_kind=payload.kind,
        env_steps_min=env_min,
        env_steps_max=env_max,
        count=len(records),
        keys=sorted(metrics.keys()),
        metrics=metrics,
        missing_keys=missing,
        threshold_pct=float(args.threshold_pct),
        abs_tol=float(args.abs_tol),
    )
    return baseline


def _compare(args: argparse.Namespace, baseline: Baseline) -> CompareResult:  # type: ignore[no-untyped-def]
    payload = _load_records(Path(args.input))
    threshold_pct = (
        float(args.threshold_pct)
        if args.threshold_pct is not None
        else float(baseline.threshold_pct)
    )
    abs_tol = (
        float(args.abs_tol) if args.abs_tol is not None else float(baseline.abs_tol)
    )

    records = payload.records
    if baseline.input_kind == "jsonl":
        records = _filter_records(
            records,
            env_steps_min=baseline.env_steps_min,
            env_steps_max=baseline.env_steps_max,
            last=args.last,
        )
    elif args.last is not None:
        records = records[-args.last :]

    if not records:
        raise ValueError("No records matched baseline window")

    metrics, missing = _compute_metrics(records, baseline.keys)

    details: list[dict[str, Any]] = []
    failures = 0
    for key in baseline.keys:
        base_val = baseline.metrics.get(key)
        cur_val = metrics.get(key)
        if base_val is None or cur_val is None:
            failures += 1
            details.append(
                {
                    "key": key,
                    "baseline": base_val,
                    "current": cur_val,
                    "ok": False,
                    "reason": "missing",
                }
            )
            continue
        delta = abs(cur_val - base_val)
        denom = max(abs(base_val), abs_tol)
        pct = delta / denom if denom > 0 else float("inf")
        ok = pct <= threshold_pct
        if not ok:
            failures += 1
        details.append(
            {
                "key": key,
                "baseline": base_val,
                "current": cur_val,
                "delta": delta,
                "pct": pct,
                "ok": ok,
            }
        )

    result = CompareResult(
        ok=failures == 0,
        compared=len(baseline.keys),
        failures=failures,
        missing=len(missing),
        threshold_pct=threshold_pct,
        abs_tol=abs_tol,
        details=details,
    )
    return result


def main() -> int:
    args = _parse_args()

    if args.command == "capture":
        baseline = _capture(args)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(baseline.__dict__, indent=2) + "\n")
        print(json.dumps({"ok": True, "baseline": str(output_path)}))
        if baseline.missing_keys:
            print(
                json.dumps(
                    {
                        "missing_keys": baseline.missing_keys,
                        "count": len(baseline.missing_keys),
                    }
                )
            )
        return 0

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        raise FileNotFoundError(f"baseline not found: {baseline_path}")
    baseline = Baseline(**json.loads(baseline_path.read_text(encoding="utf-8")))
    result = _compare(args, baseline)
    payload = {
        "ok": result.ok,
        "compared": result.compared,
        "failures": result.failures,
        "missing": result.missing,
        "threshold_pct": result.threshold_pct,
        "abs_tol": result.abs_tol,
        "details": result.details,
    }
    print(json.dumps(payload, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
