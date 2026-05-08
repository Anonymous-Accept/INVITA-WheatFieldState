"""Run the Source-date prior experiment on official INVITA splits."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.paths import default_data_root, default_output_root, default_split_root  # noqa: E402
from src.benchmarks.baselines.source_date_prior.dataloader import (  # noqa: E402
    TASKS,
    SourceDatePriorDataConfig,
    SourceDatePriorDataLoader,
    write_validation_report,
)
from src.benchmarks.baselines.source_date_prior.predictor import (  # noqa: E402
    SourceDatePrior,
    SourceDatePriorConfig,
    attach_errors,
    metrics_by_target_name,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Root directory of the INVITA dataset build.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=default_split_root(),
        help="Split root with {task}/{train,val,test}.csv files.",
    )
    parser.add_argument(
        "--enforce-trial-group-exclusivity",
        action="store_true",
        help="Fail if a (trial_year, trial_code) group appears in multiple splits.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "source_date_prior",
        help="Directory where experiment artifacts will be written.",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS),
        choices=list(TASKS),
        help="Tasks to evaluate.",
    )
    parser.add_argument(
        "--eval-splits",
        nargs="+",
        default=["val", "test"],
        choices=["val", "test"],
        help="Official non-training splits to evaluate.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run identifier. Defaults to a UTC timestamp.",
    )
    parser.add_argument(
        "--doy-bin-size",
        type=int,
        default=14,
        help="Day-of-year bin size used by the climatology lookup.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="Minimum training samples required for a lookup group.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "median"],
        default="mean",
        help="Training aggregation used within climatology groups.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    run_config = {
        "baseline": "Source-date prior",
        "run_id": run_id,
        "data_root": str(args.data_root),
        "tasks": args.tasks,
        "split_root": str(args.split_root) if args.split_root else None,
        "enforce_trial_group_exclusivity": args.enforce_trial_group_exclusivity,
        "eval_splits": args.eval_splits,
        "doy_bin_size": args.doy_bin_size,
        "min_samples": args.min_samples,
        "aggregation": args.aggregation,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "allowed_inputs": ["target provenance", "site metadata", "crop metadata", "calendar"],
        "disallowed_inputs": ["payload assets", "sensor values", "raster values", "field-camera images", "weather records"],
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n"
    )

    loader = SourceDatePriorDataLoader(
        SourceDatePriorDataConfig(
            data_root=args.data_root,
            split_root=args.split_root,
            enforce_trial_group_exclusivity=args.enforce_trial_group_exclusivity,
        )
    )
    model_config = SourceDatePriorConfig(
        aggregation=args.aggregation,
        doy_bin_size=args.doy_bin_size,
        min_samples=args.min_samples,
    )

    metrics_rows = []
    target_metric_frames = []
    fallback_frames = []
    validation_report = {}

    for task_name in args.tasks:
        logger.info("Loading %s", task_name)
        task_data = loader.load_task(task_name)
        validation_report[task_name] = task_data.validation_report

        task_dir = run_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        write_validation_report(
            task_data.validation_report, task_dir / "split_validation.json"
        )

        model = SourceDatePrior(model_config)
        model.fit(task_data.train)

        for split in args.eval_splits:
            eval_frame = task_data.split(split)
            prediction_frame = attach_errors(model.predict_frame(eval_frame))
            prediction_path = task_dir / f"predictions_{split}.parquet"
            prediction_frame.to_parquet(prediction_path, index=False)

            split_metrics = model.evaluate(eval_frame)
            metrics_rows.append(
                {
                    "task": task_name,
                    "split": split,
                    "baseline": "Source-date prior",
                    **split_metrics,
                }
            )

            by_target = metrics_by_target_name(prediction_frame)
            by_target.insert(0, "baseline", "Source-date prior")
            by_target.insert(0, "split", split)
            by_target.insert(0, "task", task_name)
            target_metric_frames.append(by_target)

            fallback = model.fallback_usage(prediction_frame)
            fallback.insert(0, "baseline", "Source-date prior")
            fallback.insert(0, "split", split)
            fallback.insert(0, "task", task_name)
            fallback_frames.append(fallback)

            logger.info(
                "%s %s: n=%d mae=%.6f rmse=%.6f r2=%.6f",
                task_name,
                split,
                split_metrics["n"],
                split_metrics["mae"],
                split_metrics["rmse"],
                split_metrics["r2"],
            )

    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(run_dir / "metrics.csv", index=False)

    if target_metric_frames:
        pd.concat(target_metric_frames, ignore_index=True).to_csv(
            run_dir / "metrics_by_target_name.csv", index=False
        )
    if fallback_frames:
        pd.concat(fallback_frames, ignore_index=True).to_csv(
            run_dir / "fallback_usage.csv", index=False
        )

    (run_dir / "split_validation.json").write_text(
        json.dumps(validation_report, indent=2, sort_keys=True) + "\n"
    )
    logger.info("Source-date prior run complete: %s", run_dir)
    return run_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
