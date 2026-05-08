"""Run the Sensor-sequence Transformer experiment."""

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
from src.benchmarks.baselines.sensor_sequence_transformer import (  # noqa: E402
    SensorSequenceTransformerConfig,
    SensorSequenceDataLoader,
    SensorSequenceTransformer,
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
        "--output-dir",
        type=Path,
        default=default_output_root() / "sensor_sequence_transformer",
    )
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument(
        "--eval-splits", nargs="+", default=["val", "test"], choices=["val", "test"]
    )
    parser.add_argument("--run-id", default="plot_disjoint_sensor_sequence_transformer")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--min-feature-std", type=float, default=1.0)
    parser.add_argument("--feature-clip", type=float, default=10.0)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to auto: cuda when available, otherwise cpu.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    run_id = args.run_id
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_config = SensorSequenceTransformerConfig(
        random_state=args.random_state,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        min_feature_std=args.min_feature_std,
        feature_clip=args.feature_clip,
        device=args.device,
        num_workers=args.num_workers,
    )
    resolved_device = str(SensorSequenceTransformer(model_config).device)
    write_json(
        {
            "baseline": "Sensor-sequence Transformer",
            "run_id": run_id,
            "data_root": str(args.data_root),
            "split_root": str(args.split_root),
            "tasks": args.tasks,
            "eval_splits": args.eval_splits,
            "model": {
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "batch_size": args.batch_size,
                "n_epochs": args.n_epochs,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "dropout": args.dropout,
                "min_feature_std": args.min_feature_std,
                "feature_clip": args.feature_clip,
                "requested_device": args.device or "auto",
                "resolved_device": resolved_device,
                "num_workers": args.num_workers,
                "random_state": args.random_state,
            },
            "implementation_assumption": "Uses masked fixed-bin pre-target sequence features from the Sensor-summary model legal-input history builder with true payload-backed GreenSeeker and UAV-MS numeric observations. Missing time bins are masked in the temporal encoder; metric streams are fused with an attention gate.",
            "required_modality_policy": "Targets without at least one legal pre-target payload-backed sensor observation are excluded from Sensor-sequence Transformer.",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        run_dir / "run_config.json",
    )

    loader = SensorSequenceDataLoader(
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
                "%s skipped: no Sensor-sequence Transformer training rows after modality filtering", task_name
            )
            continue
        model = SensorSequenceTransformer(model_config).fit(task_data.train)
        task_metrics, task_target_metrics, task_fallback = evaluate_task_model(
            model=model,
            task_data=task_data,
            baseline_name="Sensor-sequence Transformer",
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
    logger.info("Sensor-sequence Transformer run complete: %s", run_dir)
    return run_dir


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(parse_args())


if __name__ == "__main__":
    main()
