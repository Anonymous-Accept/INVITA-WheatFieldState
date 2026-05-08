"""Artifact helpers for neural representation runs."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.benchmarks.baselines._shared import (
    attach_errors,
    metrics_by_target_name,
    regression_metrics,
    write_json,
)


def dataclass_payload(value: Any) -> Any:
    """Return a JSON-safe payload for dataclass configs."""

    return asdict(value) if is_dataclass(value) else value


def limit_frame(frame: pd.DataFrame, limit: int | None, *, seed: int) -> pd.DataFrame:
    """Deterministically cap a frame for smoke tests."""

    if limit is None or len(frame) <= limit:
        return frame
    return frame.sample(n=limit, random_state=seed).reset_index(drop=True)


def write_model_config(config: Any, path: Path) -> None:
    """Write a model config JSON artifact."""

    write_json(dataclass_payload(config), path)


def evaluate_and_write(
    *,
    model: Any,
    task_data: Any,
    baseline_name: str,
    eval_splits: list[str],
    run_dir: Path,
    task_name: str,
) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Evaluate a fitted model and write predictions under predictions/{task}."""

    metric_rows: list[dict[str, Any]] = []
    target_metric_frames: list[pd.DataFrame] = []
    prediction_dir = run_dir / "predictions" / task_name
    prediction_dir.mkdir(parents=True, exist_ok=True)

    for split in eval_splits:
        frame = task_data.split(split)
        if frame.empty:
            continue
        prediction_frame = attach_errors(model.predict_frame(frame))
        prediction_frame.to_parquet(prediction_dir / f"{split}.parquet", index=False)
        metrics = regression_metrics(
            prediction_frame["target_value_num"], prediction_frame["prediction"]
        )
        metric_rows.append(
            {
                "task": task_name,
                "split": split,
                "baseline": baseline_name,
                **metrics,
            }
        )
        by_target = metrics_by_target_name(prediction_frame)
        by_target.insert(0, "baseline", baseline_name)
        by_target.insert(0, "split", split)
        by_target.insert(0, "task", task_name)
        target_metric_frames.append(by_target)

    return metric_rows, target_metric_frames


def write_aggregate_outputs(
    *,
    run_dir: Path,
    metrics_rows: list[dict[str, Any]],
    target_metric_frames: list[pd.DataFrame],
    coverage_rows: list[dict[str, Any]],
) -> None:
    """Write aggregate metrics, coverage, and JSON summaries."""

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_rows, indent=2, sort_keys=True) + "\n"
    )
    if target_metric_frames:
        pd.concat(target_metric_frames, ignore_index=True).to_csv(
            run_dir / "target_name_slice_metrics.csv", index=False
        )
    pd.DataFrame(coverage_rows).to_csv(run_dir / "coverage.csv", index=False)


def training_summary_row(
    *,
    task_name: str,
    baseline_name: str,
    model: Any,
    train_n: int,
    val_n: int,
    test_n: int,
) -> dict[str, Any]:
    """Build one task-level training summary row from a fitted model."""

    summary = getattr(model, "training_summary", {}) or {}
    return {
        "task": task_name,
        "baseline": baseline_name,
        "train": train_n,
        "val": val_n,
        "test": test_n,
        **summary,
    }


def write_training_summary(*, run_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write task-level neural training diagnostics."""

    if rows:
        pd.DataFrame(rows).to_csv(run_dir / "training_summary.csv", index=False)


def write_readme(
    *,
    path: Path,
    title: str,
    body: str,
) -> None:
    """Write a short run README."""

    path.write_text(f"# {title}\n\n{body.strip()}\n")
