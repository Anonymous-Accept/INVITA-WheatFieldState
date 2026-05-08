"""Summarize neural representation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.paths import default_data_root, default_output_root, default_split_root  # noqa: E402

NEURAL_EXPERIMENTS = {
    "Tabular Transformer": ("tabular_transformer", "Tabular Transformer", "full_split"),
    "Observation-set Transformer": ("observation_set_transformer", "Observation-set Transformer", "full_split"),
    "Sensor-sequence TCN": ("sensor_sequence_tcn", "Sensor-sequence TCN", "modality_subset"),
    "Field-camera image-set model": ("field_camera_image_set_model", "Field-camera image-set model", "modality_subset"),
    "Gated stacker": ("gated_stacker", "Gated stacker", "same_row_fusion"),
}

REFERENCE_BASELINES = {
    "Source-date prior": Path("source_date_prior/plot_disjoint"),
    "Crop-date prior": Path("crop_date_prior/plot_disjoint"),
    "Tabular metadata model": Path("tabular_metadata_model/plot_disjoint"),
    "Observation-availability model": Path("observation_availability_model/plot_disjoint"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=default_output_root())
    parser.add_argument("--neural-results-root", type=Path, default=default_output_root() / "neural_baselines")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "neural_baselines" / "summary_plot_disjoint",
    )
    parser.add_argument("--run-id", default="plot_disjoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    statuses = _statuses(args.neural_results_root, args.run_id)

    full = _reference_rows(args.results_root)
    full.extend(_neural_metric_rows(args.neural_results_root, args.run_id, surface="full_split"))
    pd.DataFrame(full).to_csv(args.output_dir / "table_full_split_baselines.csv", index=False)

    subset = _neural_metric_rows(args.neural_results_root, args.run_id, surface="modality_subset")
    pd.DataFrame(subset).to_csv(args.output_dir / "table_modality_subset_results.csv", index=False)

    fusion = _same_row_fusion_rows(args.neural_results_root, args.run_id)
    pd.DataFrame(fusion).to_csv(args.output_dir / "table_same_row_fusion.csv", index=False)

    slices = _slice_rows(args.neural_results_root, args.run_id)
    pd.DataFrame(slices).to_csv(args.output_dir / "table_lai_fcover_target_slices.csv", index=False)

    coverage = _coverage_rows(args.neural_results_root, args.run_id)
    pd.DataFrame(coverage).to_csv(args.output_dir / "table_coverage.csv", index=False)

    report = _report(statuses, full, subset, fusion, coverage, args)
    (args.output_dir / "NEURAL_BASELINE_REPORT.md").write_text(report + "\n")
    print(f"Wrote neural representation summary to {args.output_dir}")


def _statuses(root: Path, run_id: str) -> list[dict[str, Any]]:
    rows = []
    for internal_id, (dirname, method, surface) in NEURAL_EXPERIMENTS.items():
        run_dir = root / dirname / run_id
        metrics = run_dir / "metrics.csv"
        coverage = run_dir / "coverage.csv"
        if metrics.exists():
            status = "ran"
        elif run_dir.exists():
            status = "partially_ran"
        else:
            status = "not_run"
        rows.append(
            {
                "id": internal_id,
                "method": method,
                "surface": surface,
                "status": status,
                "run_dir": str(run_dir),
                "has_metrics": metrics.exists(),
                "has_coverage": coverage.exists(),
            }
        )
    return rows


def _reference_rows(results_root: Path) -> list[dict[str, Any]]:
    rows = []
    for method, rel_path in REFERENCE_BASELINES.items():
        path = results_root / rel_path / "metrics.csv"
        if not path.exists():
            continue
        metrics = pd.read_csv(path)
        for record in metrics.to_dict("records"):
            if record.get("split") != "test":
                continue
            rows.append(
                {
                    "target": record.get("task"),
                    "method": method,
                    "test_examples": record.get("n"),
                    "mae": record.get("mae"),
                    "rmse": record.get("rmse"),
                    "r2": record.get("r2"),
                    "notes": "reference_baseline",
                }
            )
    return rows


def _neural_metric_rows(root: Path, run_id: str, *, surface: str) -> list[dict[str, Any]]:
    rows = []
    for _internal_id, (dirname, method, exp_surface) in NEURAL_EXPERIMENTS.items():
        if exp_surface != surface:
            continue
        path = root / dirname / run_id / "metrics.csv"
        if not path.exists():
            continue
        metrics = pd.read_csv(path)
        for record in metrics.to_dict("records"):
            if record.get("split") != "test":
                continue
            rows.append(
                {
                    "target": record.get("task"),
                    "method": method,
                    "test_examples": record.get("n"),
                    "mae": record.get("mae"),
                    "rmse": record.get("rmse"),
                    "r2": record.get("r2"),
                    "notes": record.get("routes", ""),
                }
            )
    return rows


def _same_row_fusion_rows(root: Path, run_id: str) -> list[dict[str, Any]]:
    path = root / "gated_stacker" / run_id / "same_row_fusion.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).to_dict("records")


def _slice_rows(root: Path, run_id: str) -> list[dict[str, Any]]:
    rows = []
    for _internal_id, (dirname, method, _surface) in NEURAL_EXPERIMENTS.items():
        path = root / dirname / run_id / "target_name_slice_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(2, "method", method)
        rows.extend(frame.to_dict("records"))
    return rows


def _coverage_rows(root: Path, run_id: str) -> list[dict[str, Any]]:
    rows = []
    for _internal_id, (dirname, method, _surface) in NEURAL_EXPERIMENTS.items():
        path = root / dirname / run_id / "coverage.csv"
        if not path.exists():
            rows.append({"method": method, "status": "not_run", "path": str(path)})
            continue
        frame = pd.read_csv(path)
        frame.insert(1, "summary_method", method)
        rows.extend(frame.to_dict("records"))
    return rows


def _report(
    statuses: list[dict[str, Any]],
    full: list[dict[str, Any]],
    subset: list[dict[str, Any]],
    fusion: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Neural Baseline Report",
        "",
        f"- run_id: `{args.run_id}`",
        f"- neural_results_root: `{args.neural_results_root}`",
        "",
        "## Completed Experiments",
        "",
        "| ID | Method | Status | Run Dir |",
        "| --- | --- | --- | --- |",
    ]
    for row in statuses:
        lines.append(
            f"| {row['id']} | {row['method']} | {row['status']} | {row['run_dir']} |"
        )
    lines.extend(
        [
            "",
            "## Main Results",
            "",
            f"- full_split_rows: {len(full)}",
            f"- modality_subset_rows: {len(subset)}",
            f"- same_row_fusion_rows: {len(fusion)}",
            f"- coverage_rows: {len(coverage)}",
            "",
            "## Blocked Or Not Run",
            "",
        ]
    )
    for row in statuses:
        if row["status"] != "ran":
            lines.append(f"- {row['id']} {row['method']}: {row['status']}")
    lines.extend(
        [
            "",
            "## Paper Integration Recommendation",
            "",
            "Use `table_full_split_baselines.csv` for full split methods, "
            "`table_modality_subset_results.csv` for subset methods, and "
            "`table_same_row_fusion.csv` for fusion claims. Do not compare subset "
            "methods against full-task methods without same-row controls.",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
