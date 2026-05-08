"""Utilities shared by baseline run scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmarks.baselines._shared import (
    attach_errors,
    metrics_by_target_name,
    regression_metrics,
)


def evaluate_task_model(
    *,
    model,
    task_data,
    baseline_name: str,
    eval_splits: list[str],
    task_dir: Path,
) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    """Evaluate a fitted task model and write per-task prediction artifacts."""

    task_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    target_metric_frames: list[pd.DataFrame] = []
    fallback_frames: list[pd.DataFrame] = []

    for split in eval_splits:
        eval_frame = task_data.split(split)
        if eval_frame.empty:
            continue
        prediction_frame = attach_errors(model.predict_frame(eval_frame))
        prediction_frame.to_parquet(task_dir / f"predictions_{split}.parquet", index=False)

        split_metrics = regression_metrics(
            prediction_frame["target_value_num"], prediction_frame["prediction"]
        )
        metric_rows.append(
            {
                "task": task_data.task_name,
                "split": split,
                "baseline": baseline_name,
                **split_metrics,
            }
        )

        by_target = metrics_by_target_name(prediction_frame)
        by_target.insert(0, "baseline", baseline_name)
        by_target.insert(0, "split", split)
        by_target.insert(0, "task", task_data.task_name)
        target_metric_frames.append(by_target)

        if hasattr(model, "fallback_usage"):
            fallback = model.fallback_usage(prediction_frame)
            fallback.insert(0, "baseline", baseline_name)
            fallback.insert(0, "split", split)
            fallback.insert(0, "task", task_data.task_name)
            fallback_frames.append(fallback)

    if hasattr(model, "feature_importance"):
        feature_importance = model.feature_importance()
        if not feature_importance.empty:
            feature_importance.to_csv(task_dir / "feature_importance.csv", index=False)

    return metric_rows, target_metric_frames, fallback_frames


def write_aggregate_csvs(
    *,
    run_dir: Path,
    metrics_rows: list[dict],
    target_metric_frames: list[pd.DataFrame],
    fallback_frames: list[pd.DataFrame],
) -> None:
    """Write aggregate metrics for a baseline run."""

    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
    if target_metric_frames:
        pd.concat(target_metric_frames, ignore_index=True).to_csv(
            run_dir / "metrics_by_target_name.csv", index=False
        )
    if fallback_frames:
        pd.concat(fallback_frames, ignore_index=True).to_csv(
            run_dir / "fallback_usage.csv", index=False
        )
