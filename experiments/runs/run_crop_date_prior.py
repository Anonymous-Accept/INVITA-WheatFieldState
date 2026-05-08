"""Run the Crop-date prior experiment."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.paths import default_data_root, default_output_root, default_split_root  # noqa: E402
from _run_utils import evaluate_task_model, write_aggregate_csvs  # noqa: E402

from src.benchmarks.baselines._shared import (  # noqa: E402
    TASKS,
    BaselineDataConfig,
    write_json,
)
from src.benchmarks.baselines.crop_date_prior import (  # noqa: E402
    CropDatePriorConfig,
    CropDatePriorDataLoader,
    CropDatePrior,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument(
        "--split-root",
        type=Path,
        default=default_split_root(),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "crop_date_prior",
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--aggregation", choices=["mean", "median"], default="mean")
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    run_config = {
        "baseline": "Crop-date prior",
        "run_id": run_id,
        "data_root": str(args.data_root),
        "split_root": str(args.split_root),
        "tasks": args.tasks,
        "eval_splits": args.eval_splits,
        "aggregation": args.aggregation,
        "min_samples": args.min_samples,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "allowed_inputs": ["target metadata", "plot agronomic metadata", "calendar"],
        "disallowed_inputs": [
            "payload assets",
            "sensor values",
            "raster values",
            "field-camera images",
            "weather records",
            "raw target identifiers as model features",
        ],
    }
    write_json(run_config, run_dir / "run_config.json")

    loader = CropDatePriorDataLoader(
        BaselineDataConfig(data_root=args.data_root, split_root=args.split_root)
    )
    model_config = CropDatePriorConfig(
        aggregation=args.aggregation,
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
        write_json(task_data.validation_report, run_dir / task_name / "split_validation.json")

        model = CropDatePrior(model_config)
        model.fit(task_data.train)
        task_metrics, task_target_metrics, task_fallback = evaluate_task_model(
            model=model,
            task_data=task_data,
            baseline_name="Crop-date prior",
            eval_splits=args.eval_splits,
            task_dir=run_dir / task_name,
        )
        metrics_rows.extend(task_metrics)
        target_metric_frames.extend(task_target_metrics)
        fallback_frames.extend(task_fallback)

        for row in task_metrics:
            logger.info(
                "%s %s: n=%d mae=%.6f rmse=%.6f r2=%.6f",
                row["task"],
                row["split"],
                row["n"],
                row["mae"],
                row["rmse"],
                row["r2"],
            )

    write_aggregate_csvs(
        run_dir=run_dir,
        metrics_rows=metrics_rows,
        target_metric_frames=target_metric_frames,
        fallback_frames=fallback_frames,
    )
    (run_dir / "split_validation.json").write_text(
        json.dumps(validation_report, indent=2, sort_keys=True) + "\n"
    )
    logger.info("Crop-date prior run complete: %s", run_dir)
    return run_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
