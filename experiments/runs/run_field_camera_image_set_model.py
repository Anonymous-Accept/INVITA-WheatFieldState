"""Run the Field-camera image-set model neural representation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.paths import default_data_root, default_output_root, default_split_root  # noqa: E402
from src.benchmarks.baselines._shared import (  # noqa: E402
    TASKS,
    BaselineDataConfig,
    BaselineTaskData,
    write_json,
)
from src.benchmarks.baselines.frozen_image_feature_model import FrozenImageFeatureDataLoader  # noqa: E402
from src.benchmarks.neural_baselines.common.artifacts import (  # noqa: E402
    evaluate_and_write,
    limit_frame,
    training_summary_row,
    write_aggregate_outputs,
    write_model_config,
    write_readme,
    write_training_summary,
)
from src.benchmarks.neural_baselines.common.torch_tabular import (  # noqa: E402
    TabularTransformerConfig,
    TabularTransformerRegressor,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--split-root", type=Path, default=default_split_root())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "neural_baselines" / "field_camera_image_set_model",
    )
    parser.add_argument("--run-id", default="plot_disjoint")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--eval-splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    parser.add_argument("--encoder", default="frozen_image_feature_squeezenet_fallback")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    model_config = TabularTransformerConfig(
        categorical_features=["target_name"],
        numeric_features=[],
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
    )
    write_model_config(model_config, run_dir / "model_config.json")
    write_json(
        {
            "experiment_id": "field_camera_image_set_model",
            "paper_method": "Field-camera image-set model",
            "run_id": args.run_id,
            "data_root": str(args.data_root),
            "split_root": str(args.split_root),
            "tasks": args.tasks,
            "eval_splits": args.eval_splits,
            "encoder": args.encoder,
            "seed": args.seed,
            "limits": {
                "limit_train": args.limit_train,
                "limit_val": args.limit_val,
                "limit_test": args.limit_test,
            },
            "implementation_note": "Uses Frozen image-feature model legal pre-target field-camera image-set aggregate embedding features as a reproducible fallback neural head.",
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        run_dir / "run_config.json",
    )
    write_readme(
        path=run_dir / "README.md",
        title="Field-camera image-set model",
        body=(
            "Reproducible Field-camera image-set model. It reuses the "
            "Frozen image-feature model leakage-controlled field-camera image-set aggregate embeddings "
            "and trains a neural regressor. This is not a DINO/CLIP modern-encoder "
            "variant; that remains a future stronger Field-camera image-set model variant."
        ),
    )

    loader = FrozenImageFeatureDataLoader(
        BaselineDataConfig(data_root=args.data_root, split_root=args.split_root)
    )
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    target_metric_frames = []
    coverage_rows = []
    training_rows = []
    validation_report = {}
    for task_name in args.tasks:
        logger.info("Loading %s", task_name)
        task_data = _limited_task_data(loader.load_task(task_name), args)
        validation_report[task_name] = task_data.validation_report
        write_json(task_data.validation_report, run_dir / task_name / "split_validation.json")
        rgb_features = _rgb_features(task_data.train)
        coverage_rows.append(_coverage_row(task_name, task_data, len(rgb_features)))
        if task_data.train.empty or not rgb_features:
            coverage_rows[-1]["status"] = "not_available_no_legal_field_camera"
            continue
        task_model_config = TabularTransformerConfig(
            **{**model_config.__dict__, "numeric_features": rgb_features}
        )
        model = TabularTransformerRegressor(task_model_config).fit(task_data.train, task_data.val)
        write_json(model.training_summary, run_dir / task_name / "training_summary.json")
        training_rows.append(
            training_summary_row(
                task_name=task_name,
                baseline_name="field_camera_image_set_model",
                model=model,
                train_n=len(task_data.train),
                val_n=len(task_data.val),
                test_n=len(task_data.test),
            )
        )
        if model.model is not None:
            torch.save(
                {
                    "state_dict": model.model.state_dict(),
                    "category_maps": model.category_maps,
                    "numeric_mean": model.numeric_mean,
                    "numeric_std": model.numeric_std,
                    "target_mean": model.target_mean,
                    "target_std": model.target_std,
                    "rgb_features": rgb_features,
                },
                checkpoint_dir / f"{task_name}.pt",
            )
        task_metrics, task_target_metrics = evaluate_and_write(
            model=model,
            task_data=task_data,
            baseline_name="field_camera_image_set_model",
            eval_splits=args.eval_splits,
            run_dir=run_dir,
            task_name=task_name,
        )
        metrics_rows.extend(task_metrics)
        target_metric_frames.extend(task_target_metrics)
        for row in task_metrics:
            logger.info(
                "%s %s n=%d mae=%.6f rmse=%.6f r2=%.6f",
                row["task"],
                row["split"],
                row["n"],
                row["mae"],
                row["rmse"],
                row["r2"],
            )
    write_aggregate_outputs(
        run_dir=run_dir,
        metrics_rows=metrics_rows,
        target_metric_frames=target_metric_frames,
        coverage_rows=coverage_rows,
    )
    write_training_summary(run_dir=run_dir, rows=training_rows)
    (run_dir / "split_validation.json").write_text(
        json.dumps(validation_report, indent=2, sort_keys=True) + "\n"
    )
    logger.info("Field-camera image-set model run complete: %s", run_dir)
    return run_dir


def _limited_task_data(task_data: BaselineTaskData, args: argparse.Namespace) -> BaselineTaskData:
    return BaselineTaskData(
        task_name=task_data.task_name,
        train=limit_frame(task_data.train, args.limit_train, seed=args.seed),
        val=limit_frame(task_data.val, args.limit_val, seed=args.seed),
        test=limit_frame(task_data.test, args.limit_test, seed=args.seed),
        validation_report=task_data.validation_report,
    )


def _rgb_features(frame) -> list[str]:
    return sorted(col for col in frame.columns if col.startswith("rgb_"))


def _coverage_row(task_name: str, task_data: BaselineTaskData, n_features: int) -> dict:
    return {
        "task": task_name,
        "method": "Field-camera image-set model",
        "required_observations": "legal pre-target field-camera image",
        "train": len(task_data.train),
        "val": len(task_data.val),
        "test": len(task_data.test),
        "coverage_percent": None,
        "rgb_feature_count": n_features,
        "status": "ran" if len(task_data.train) and n_features else "not_available_no_legal_field_camera",
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
