"""Run the Frozen image-feature model experiment."""

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
from src.benchmarks.baselines.frozen_image_feature_model import (  # noqa: E402
    FrozenImageFeatureConfig,
    FrozenImageFeatureDataLoader,
    FrozenImageFeatureModel,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root", type=Path, default=default_data_root()
    )
    parser.add_argument(
        "--split-root", type=Path, default=default_split_root()
    )
    parser.add_argument(
        "--output-dir", type=Path, default=default_output_root() / "frozen_image_feature_model"
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument(
        "--eval-splits", nargs="+", default=["val", "test"], choices=["val", "test"]
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_config = FrozenImageFeatureConfig(
        random_state=args.random_state,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
    )
    write_json(
        {
            "baseline": "Frozen image-feature model",
            "run_id": run_id,
            "data_root": str(args.data_root),
            "split_root": str(args.split_root),
            "tasks": args.tasks,
            "eval_splits": args.eval_splits,
            "model": {
                "n_estimators": args.n_estimators,
                "learning_rate": args.learning_rate,
                "max_depth": args.max_depth,
                "random_state": args.random_state,
            },
            "implementation_assumption": "Uses true field-camera payload images from legal pre-target inputs_index assets, encoded by frozen ImageNet SqueezeNet features and aggregated per target. Whole-trial UAV orthomosaics are excluded because current plot geometry is unresolved.",
            "required_modality_policy": "Targets without at least one successfully encoded legal pre-target field-camera image are excluded from Frozen image-feature model.",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        run_dir / "run_config.json",
    )

    loader = FrozenImageFeatureDataLoader(
        BaselineDataConfig(data_root=args.data_root, split_root=args.split_root)
    )
    metrics_rows = []
    target_metric_frames = []
    fallback_frames = []
    validation_report = {}

    for task_name in args.tasks:
        logger.info("Loading %s", task_name)
        task_data = loader.load_task(task_name)
        validation_report[task_name] = task_data.validation_report
        write_json(
            task_data.validation_report, run_dir / task_name / "split_validation.json"
        )
        if task_data.train.empty:
            logger.warning(
                "%s skipped: no Frozen image-feature model training rows after modality filtering", task_name
            )
            continue
        model = FrozenImageFeatureModel(model_config).fit(task_data.train)
        task_metrics, task_target_metrics, task_fallback = evaluate_task_model(
            model=model,
            task_data=task_data,
            baseline_name="Frozen image-feature model",
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
    logger.info("Frozen image-feature model run complete: %s", run_dir)
    return run_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(parse_args())


if __name__ == "__main__":
    main()
