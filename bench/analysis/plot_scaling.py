"""Plot scaling curves from benchmark results."""

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


def extract_series(
    artifact: ScalingArtifact, metric: str
) -> tuple[list[int], list[float]]:
    results = artifact.data.get("results", [])
    series = []
    for entry in results:
        num_envs = entry.get("num_envs")
        if isinstance(num_envs, int):
            series.append((num_envs, float(entry.get(metric, 0.0))))
    series.sort(key=lambda x: x[0])
    envs = [env for env, _ in series]
    values = [value for _, value in series]
    return envs, values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot scaling artifacts.")
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory containing *__scaling.json artifacts.",
    )
    parser.add_argument("--baseline", type=str, default=None)
    parser.add_argument("--dut", type=str, default=None)
    parser.add_argument(
        "--metric",
        choices=["total_sps", "per_env_sps", "frames_per_sec"],
        default="total_sps",
        help="Metric to plot.",
    )
    parser.add_argument(
        "--xscale",
        choices=["linear", "log2"],
        default="log2",
        help="X-axis scale for env counts.",
    )
    parser.add_argument(
        "--with-speedup",
        action="store_true",
        help="Include speedup curve (DUT / baseline).",
    )
    parser.add_argument("--out", type=str, default="scaling.png")
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

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise SystemExit(
            "matplotlib is required for plotting. Install it and retry."
        ) from exc

    baseline_envs, baseline_values = extract_series(baseline, args.metric)
    dut_envs, dut_values = extract_series(dut, args.metric)

    if args.with_speedup:
        fig, (ax_main, ax_speed) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    else:
        fig, ax_main = plt.subplots(figsize=(7, 4))
        ax_speed = None

    ax_main.plot(
        baseline_envs,
        baseline_values,
        marker="o",
        label=f"{baseline.backend}",
    )
    ax_main.plot(
        dut_envs,
        dut_values,
        marker="o",
        label=f"{dut.backend}",
    )

    ax_main.set_ylabel(args.metric)
    ax_main.legend()
    ax_main.grid(True, which="both", linestyle="--", alpha=0.4)

    if args.xscale == "log2":
        ax_main.set_xscale("log", base=2)
    ax_main.set_xticks(sorted(set(baseline_envs + dut_envs)))
    ax_main.set_xlabel("env_count")

    if ax_speed is not None:
        common_envs = sorted(set(baseline_envs) & set(dut_envs))
        base_map = dict(zip(baseline_envs, baseline_values, strict=True))
        dut_map = dict(zip(dut_envs, dut_values, strict=True))
        speedups = [
            dut_map[env] / base_map[env] if base_map[env] > 0 else 0.0
            for env in common_envs
        ]
        ax_speed.plot(common_envs, speedups, marker="o", color="black")
        ax_speed.set_ylabel("speedup")
        ax_speed.grid(True, which="both", linestyle="--", alpha=0.4)
        if args.xscale == "log2":
            ax_speed.set_xscale("log", base=2)
        ax_speed.set_xticks(common_envs)
        ax_speed.set_xlabel("env_count")

    title_rom = Path(baseline.data.get("sweep_config", {}).get("rom_path", "")).name
    fig.suptitle(f"Scaling ({title_rom})")
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)

    print(f"Wrote plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
