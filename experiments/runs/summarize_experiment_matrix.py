"""Summarize completed baseline artifacts into one experiment matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.paths import default_data_root, default_output_root, default_split_root  # noqa: E402

PREDICTION_RUNS = {
    "Source-date prior": default_output_root() / "source_date_prior" / "plot_disjoint",
    "Crop-date prior": default_output_root() / "crop_date_prior" / "plot_disjoint",
    "Tabular metadata model": default_output_root() / "tabular_metadata_model" / "plot_disjoint",
    "Observation-availability model": default_output_root() / "observation_availability_model" / "plot_disjoint",
    "Sensor-summary model": default_output_root() / "sensor_summary_model" / "plot_disjoint",
    "Frozen image-feature model": default_output_root() / "frozen_image_feature_model" / "plot_disjoint",
    "Sensor-sequence Transformer": default_output_root()
    / "sensor_sequence_transformer"
    / "plot_disjoint_sensor_sequence_transformer",
    "Linear stacker": default_output_root()
    / "linear_stacker"
    / "plot_disjoint_linear_stacker",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "summary_matrix",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = _prediction_rows()
    matrix = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(args.output_dir / "experiment_matrix.csv", index=False)
    print(f"Wrote {len(matrix)} rows to {args.output_dir}")


def _prediction_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline, run_dir in PREDICTION_RUNS.items():
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            rows.append(
                {
                    "baseline": baseline,
                    "result_type": "missing_prediction_metrics",
                    "run_dir": str(run_dir),
                }
            )
            continue
        metrics = pd.read_csv(metrics_path)
        for record in metrics.to_dict("records"):
            rows.append(
                {
                    "baseline": record.get("baseline", baseline),
                    "result_type": "prediction_metrics",
                    "run_dir": str(run_dir),
                    "task": record.get("task"),
                    "split": record.get("split"),
                    "n": _maybe_int(record.get("n")),
                    "mae": record.get("mae"),
                    "rmse": record.get("rmse"),
                    "r2": record.get("r2"),
                    "routes": record.get("routes", ""),
                }
            )
    return rows


def _maybe_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No experiment artifacts found._"
    columns = [
        "baseline",
        "result_type",
        "task",
        "split",
        "n",
        "mae",
        "rmse",
        "r2",
        "accuracy",
        "n_scored",
        "parse_ok_rate",
        "answer_parse_rate",
        "model_profile",
        "model_id",
        "run_dir",
    ]
    display = frame.reindex(columns=columns).fillna("")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for record in display.to_dict("records"):
        values = [_format_cell(record.get(column, "")) for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


if __name__ == "__main__":
    main()
