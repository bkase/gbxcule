"""Summarize benchmark results."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASELINE_BACKENDS = {"pyboy_puffer_vec", "pyboy_vec_mp"}
DUT_BACKENDS = {"warp_vec_cuda"}


@dataclass(frozen=True)
class ScalingArtifact:
    path: Path
    data: dict[str, Any]

    @property
    def backend(self) -> str:
        return str(self.data.get("sweep_config", {}).get("backend"))

    @property
    def run_id(self) -> str:
        return str(self.data.get("run_id", self.path.stem))

    @property
    def timestamp(self) -> str:
        return str(self.data.get("timestamp_utc", ""))


def load_artifact(path: Path) -> ScalingArtifact:
    with path.open() as f:
        data = json.load(f)
    return ScalingArtifact(path=path, data=data)


def find_scaling_artifacts(report_dir: Path) -> list[ScalingArtifact]:
    paths = sorted(report_dir.glob("*__scaling.json"))
    return [load_artifact(path) for path in paths]


def pick_artifact(
    artifacts: list[ScalingArtifact], backend_names: set[str]
) -> ScalingArtifact | None:
    matches = [a for a in artifacts if a.backend in backend_names]
    if not matches:
        return None
    matches.sort(key=lambda a: a.timestamp)
    return matches[-1]


def _results_map(artifact: ScalingArtifact) -> dict[int, dict[str, Any]]:
    results = artifact.data.get("results", [])
    out: dict[int, dict[str, Any]] = {}
    for entry in results:
        num_envs = entry.get("num_envs")
        if isinstance(num_envs, int):
            out[num_envs] = entry
    return out


def _format_num(value: float) -> str:
    return f"{value:.1f}"


def _format_speedup(value: float) -> str:
    return f"{value:.2f}x"


def _compare_configs(
    baseline: ScalingArtifact, dut: ScalingArtifact
) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    strict_warnings: list[str] = []
    base_cfg = baseline.data.get("sweep_config", {})
    dut_cfg = dut.data.get("sweep_config", {})

    keys = [
        "rom_sha256",
        "frames_per_step",
        "release_after_frames",
        "steps",
        "warmup_steps",
        "stage",
        "action_generator",
        "action_codec",
        "action_schedule",
        "vec_backend",
        "sync_every",
    ]
    for key in keys:
        base_val = base_cfg.get(key)
        dut_val = dut_cfg.get(key)
        if key == "vec_backend" and (base_val is None or dut_val is None):
            continue
        if base_val != dut_val:
            message = f"Config mismatch for '{key}': baseline={base_val} dut={dut_val}"
            warnings.append(message)
            strict_warnings.append(message)

    base_envs = base_cfg.get("env_counts")
    dut_envs = dut_cfg.get("env_counts")
    if base_envs != dut_envs:
        message = f"env_counts differ: baseline={base_envs} dut={dut_envs}"
        warnings.append(message)
        base_set = set(base_envs or [])
        dut_set = set(dut_envs or [])
        if not (base_set & dut_set):
            strict_warnings.append(message)
    return warnings, strict_warnings


def _build_table(
    baseline_map: dict[int, dict[str, Any]],
    dut_map: dict[int, dict[str, Any]],
    *,
    fmt: str,
) -> str:
    envs = sorted(set(baseline_map) & set(dut_map))
    headers = [
        "envs",
        "baseline_sps",
        "dut_sps",
        "speedup",
        "baseline_per_env",
        "dut_per_env",
    ]

    rows = []
    for env in envs:
        base = baseline_map[env]
        dut = dut_map[env]
        base_sps = float(base.get("total_sps", 0.0))
        dut_sps = float(dut.get("total_sps", 0.0))
        speedup = dut_sps / base_sps if base_sps > 0 else 0.0
        rows.append(
            [
                str(env),
                _format_num(base_sps),
                _format_num(dut_sps),
                _format_speedup(speedup),
                _format_num(float(base.get("per_env_sps", 0.0))),
                _format_num(float(dut.get("per_env_sps", 0.0))),
            ]
        )

    if fmt == "markdown":
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        lines.extend("| " + " | ".join(row) + " |" for row in rows)
        return "\n".join(lines)

    col_widths = [max(len(h), 6) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(value))

    def _fmt_row(values: list[str]) -> str:
        return "  ".join(
            value.rjust(col_widths[idx]) for idx, value in enumerate(values)
        )

    lines = [_fmt_row(headers), _fmt_row(["-" * w for w in col_widths])]
    lines.extend(_fmt_row(row) for row in rows)
    return "\n".join(lines)


def _render_summary(
    baseline: ScalingArtifact,
    dut: ScalingArtifact,
    *,
    fmt: str,
    strict: bool,
) -> str:
    warnings, strict_warnings = _compare_configs(baseline, dut)

    base_cfg = baseline.data.get("sweep_config", {})
    dut_cfg = dut.data.get("sweep_config", {})

    baseline_map = _results_map(baseline)
    dut_map = _results_map(dut)
    missing_base = sorted(set(dut_map) - set(baseline_map))
    missing_dut = sorted(set(baseline_map) - set(dut_map))
    if missing_base:
        warnings.append(f"Missing baseline results for envs: {missing_base}")
    if missing_dut:
        warnings.append(f"Missing DUT results for envs: {missing_dut}")

    if strict and strict_warnings:
        raise SystemExit("Config mismatch detected; rerun without --strict.")

    lines = []
    lines.append("Scaling Summary")
    lines.append(f"Baseline: {baseline.backend} ({baseline.run_id})")
    lines.append(f"DUT: {dut.backend} ({dut.run_id})")
    lines.append(f"ROM: {Path(base_cfg.get('rom_path', '')).name}")
    lines.append(
        "Frames/step: "
        f"{base_cfg.get('frames_per_step')} | Steps: {base_cfg.get('steps')} "
        f"(warmup {base_cfg.get('warmup_steps')}) | Stage: {base_cfg.get('stage')}"
    )
    lines.append(
        "Sync every: "
        f"baseline={base_cfg.get('sync_every')} dut={dut_cfg.get('sync_every')}"
    )
    lines.append("")
    lines.append(_build_table(baseline_map, dut_map, fmt=fmt))

    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in warnings)

    return "\n".join(lines).rstrip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize scaling artifacts.")
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory containing *__scaling.json artifacts.",
    )
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--dut", type=str, default=None)
    parser.add_argument(
        "--format",
        choices=["markdown", "text"],
        default="markdown",
        help="Output format for the summary table.",
    )
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if configs or env counts differ.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    baseline: ScalingArtifact | None = None
    dut: ScalingArtifact | None = None

    if args.report_dir:
        artifacts = find_scaling_artifacts(Path(args.report_dir))
        baseline = pick_artifact(artifacts, BASELINE_BACKENDS)
        dut = pick_artifact(artifacts, DUT_BACKENDS)
        if (baseline is None or dut is None) and len(artifacts) >= 2:
            baseline = artifacts[0]
            dut = artifacts[1]
    else:
        if args.baseline and args.dut:
            baseline = load_artifact(Path(args.baseline))
            dut = load_artifact(Path(args.dut))

    if baseline is None or dut is None:
        raise SystemExit(
            "Provide --report-dir with scaling artifacts or --baseline/--dut paths."
        )

    summary = _render_summary(baseline, dut, fmt=args.format, strict=args.strict)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(summary)
    else:
        print(summary, end="")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
