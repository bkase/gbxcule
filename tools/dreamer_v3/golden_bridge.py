"""Golden Bridge fixture generator scaffold for Dreamer v3.

This tool should only be used from CLI. It may import sheeprl or other
reference implementations inside `main()` to avoid polluting runtime deps.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Dreamer v3 parity fixtures (skeleton)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/dreamer_v3"),
        help="Output directory for fixtures",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="all",
        help="Fixture subset selector (placeholder)",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=7,
        help="Number of bins for TwoHot fixtures (default: 7)",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=-20.0,
        help="Low bound for symlog bins",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=20.0,
        help="High bound for symlog bins",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest",
    )
    return parser.parse_args(argv)


def _ensure_reference_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    sheeprl_path = repo_root / "third_party" / "sheeprl"
    sys.path.insert(0, str(sheeprl_path))


def _write_manifest(
    out_dir: Path, entries: list[dict[str, Any]], *, force: bool
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _require_torch():  # type: ignore[no-untyped-def]
    try:
        import torch
    except Exception as exc:  # pragma: no cover - CLI only
        raise RuntimeError("torch is required to generate Dreamer v3 fixtures") from exc
    return torch


def _twohot_weights(y, bins, *, torch):  # type: ignore[no-untyped-def]
    import torch.nn.functional as F

    below = (bins <= y).to(torch.int32).sum(dim=-1, keepdim=True) - 1
    above = below + 1
    max_idx = torch.full_like(above, bins.numel() - 1)
    above = torch.minimum(above, max_idx)
    below = torch.maximum(below, torch.zeros_like(below))
    equal = below == above
    dist_to_below = torch.where(equal, torch.ones_like(y), torch.abs(bins[below] - y))
    dist_to_above = torch.where(equal, torch.ones_like(y), torch.abs(bins[above] - y))
    total = dist_to_below + dist_to_above
    weight_below = dist_to_above / total
    weight_above = dist_to_below / total
    below_oh = F.one_hot(below.to(torch.int64), bins.numel())
    above_oh = F.one_hot(above.to(torch.int64), bins.numel())
    return (
        below_oh * weight_below[..., None] + above_oh * weight_above[..., None]
    ).squeeze(-2)


def _quantile(values, q):  # type: ignore[no-untyped-def]
    values = sorted(values)
    n = len(values)
    if n == 1:
        return values[0]
    idx = q * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    if lo == hi:
        return values[lo]
    w = idx - lo
    return values[lo] + w * (values[hi] - values[lo])


def _moments_updates(
    batches: Iterable[Iterable[float]],
    *,
    decay: float,
    max_value: float,
    p_low: float,
    p_high: float,
) -> list[dict[str, Any]]:
    low_state = 0.0
    high_state = 0.0
    updates: list[dict[str, Any]] = []
    for batch in batches:
        batch_list = list(batch)
        low_q = _quantile(batch_list, p_low)
        high_q = _quantile(batch_list, p_high)
        low_state = decay * low_state + (1 - decay) * low_q
        high_state = decay * high_state + (1 - decay) * high_q
        invscale = max(1.0 / max_value, high_state - low_state)
        updates.append(
            {
                "values": batch_list,
                "low": low_state,
                "high": high_state,
                "invscale": invscale,
            }
        )
    return updates


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = _parse_args(argv)
    _ensure_reference_on_path()
    torch = _require_torch()
    from sheeprl.utils.distribution import TwoHotEncodingDistribution  # type: ignore
    from sheeprl.utils.utils import symexp, symlog  # type: ignore

    args.out.mkdir(parents=True, exist_ok=True)

    bins = torch.linspace(
        float(args.low),
        float(args.high),
        int(args.bins),
        device="cpu",
        dtype=torch.float32,
    )
    _write_json(
        args.out / "bins.json",
        {"bins": bins.tolist(), "low": args.low, "high": args.high},
    )

    symlog_inputs = torch.tensor(
        [-1000.0, -10.0, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0, 1000.0],
        dtype=torch.float32,
    )
    symlog_values = symlog(symlog_inputs)
    symexp_values = symexp(symlog_values)
    symlog_cases = [
        {"x": float(x), "symlog": float(y), "symexp": float(z)}
        for x, y, z in zip(
            symlog_inputs.tolist(),
            symlog_values.tolist(),
            symexp_values.tolist(),
            strict=True,
        )
    ]
    _write_json(args.out / "symlog_cases.json", {"cases": symlog_cases})

    twohot_inputs = torch.tensor([-1e9, -10.0, 0.0, 10.0, 1e9], dtype=torch.float32)
    twohot_cases = []
    for x in twohot_inputs:
        y = symlog(x)
        weights = _twohot_weights(y, bins, torch=torch)
        twohot_cases.append(
            {"x": float(x), "y": float(y), "weights": weights.squeeze(0).tolist()}
        )
    _write_json(args.out / "twohot_cases.json", {"cases": twohot_cases})

    logits = torch.tensor(
        [
            [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
            [1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([-2.0, 3.0], dtype=torch.float32).unsqueeze(-1)
    dist = TwoHotEncodingDistribution(
        logits, dims=1, low=float(args.low), high=float(args.high)
    )
    log_prob = dist.log_prob(targets).tolist()
    mean = dist.mean.squeeze(-1).tolist()
    _write_json(
        args.out / "symlog_twohot_dist.json",
        {
            "logits": logits.tolist(),
            "targets": targets.squeeze(-1).tolist(),
            "log_prob": log_prob,
            "mean": mean,
        },
    )

    decay = 0.99
    max_value = 1.0
    p_low = 0.05
    p_high = 0.95
    batches = [
        [i * 100.0 for i in range(11)],
        [100.0 + i * 100.0 for i in range(11)],
    ]
    return_ema_cases = {
        "config": {
            "decay": decay,
            "max_value": max_value,
            "percentiles": [p_low, p_high],
        },
        "updates": _moments_updates(
            batches,
            decay=decay,
            max_value=max_value,
            p_low=p_low,
            p_high=p_high,
        ),
    }
    _write_json(args.out / "return_ema_cases.json", return_ema_cases)

    entries = [
        {
            "name": "bins",
            "dtype": "float32",
            "shape": [int(args.bins)],
            "file": "bins.json",
            "notes": "linspace in symlog space",
        },
        {
            "name": "symlog_cases",
            "dtype": "json",
            "shape": [len(symlog_cases)],
            "file": "symlog_cases.json",
            "notes": "symlog/symexp roundtrip cases",
        },
        {
            "name": "twohot_cases",
            "dtype": "json",
            "shape": [len(twohot_cases)],
            "file": "twohot_cases.json",
            "notes": "twohot weights for selected values",
        },
        {
            "name": "symlog_twohot_dist",
            "dtype": "json",
            "shape": [],
            "file": "symlog_twohot_dist.json",
            "notes": "log_prob and mean for SymlogTwoHot distribution",
        },
        {
            "name": "return_ema_cases",
            "dtype": "json",
            "shape": [],
            "file": "return_ema_cases.json",
            "notes": "Moments/ReturnEMA updates",
        },
    ]
    _write_manifest(args.out, entries, force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
