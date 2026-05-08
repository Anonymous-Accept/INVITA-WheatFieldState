"""Run Gated stacker prediction-level fusion."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.paths import default_data_root, default_output_root, default_split_root  # noqa: E402
from src.benchmarks.baselines._shared import (  # noqa: E402
    PREDICTION_COL,
    TARGET_COL,
    TASKS,
    attach_errors,
    metrics_by_target_name,
    regression_metrics,
    write_json,
)
from src.benchmarks.neural_baselines.common.artifacts import write_readme  # noqa: E402
from src.benchmarks.neural_baselines.common.torch_tabular import set_torch_seed  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_RUN_ID = "plot_disjoint"
DEFAULT_SENSOR_SEQUENCE_TRANSFORMER_RUN_ID = "plot_disjoint_sensor_sequence_transformer"


class GatedStackerNet(nn.Module):
    """Gated stacker route-prediction model."""

    def __init__(self, n_routes: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(n_routes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_routes),
        )
        self.correction = nn.Sequential(
            nn.Linear(n_routes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.softmax(self.gate(x), dim=1)
        weighted = (gates * x).sum(dim=1)
        return weighted + 0.1 * self.correction(x).squeeze(-1)

    def gate_weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.gate(x), dim=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=default_output_root())
    parser.add_argument("--neural-results-root", type=Path, default=default_output_root() / "neural_baselines")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_root() / "neural_baselines" / "gated_stacker",
    )
    parser.add_argument("--run-id", default="plot_disjoint")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument(
        "--routes",
        default="tabular_metadata,observation_availability,sensor_summary,frozen_image_feature,sensor_sequence_transformer,tabular_transformer,observation_set_transformer,sensor_sequence_tcn,field_camera_image_set_model",
        help="Comma-separated candidate route names.",
    )
    parser.add_argument("--route-root", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--stacking-policy", choices=["validation_only"], default="validation_only")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    run_dir = args.output_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    route_roots = _route_roots(
        results_root=args.results_root,
        neural_results_root=args.neural_results_root,
        run_id=args.run_id,
        overrides=args.route_root,
    )
    requested_routes = [route.strip() for route in args.routes.split(",") if route.strip()]
    write_json(
        {
            "experiment_id": "gated_stacker",
            "paper_method": "Gated stacker",
            "run_id": args.run_id,
            "tasks": args.tasks,
            "requested_routes": requested_routes,
            "route_roots": {key: str(value) for key, value in route_roots.items()},
            "stacking_policy": "validation_only",
            "seed": args.seed,
            "model": {
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "max_epochs": args.max_epochs,
                "patience": args.patience,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "device": args.device or "auto",
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        run_dir / "run_config.json",
    )
    write_json(
        {
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        run_dir / "model_config.json",
    )
    write_readme(
        path=run_dir / "README.md",
        title="Gated stacker",
        body=(
            "Gated stacker prediction-level diagnostic. The model is trained on "
            "same-row validation predictions and evaluated on same-row test "
            "predictions. Fusion claims must be read only against same-row route metrics."
        ),
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_torch_seed(args.seed)
    metrics_rows = []
    target_metric_frames = []
    coverage_rows = []
    same_row_rows = []
    gate_rows = []
    training_rows = []
    prediction_root = run_dir / "predictions"
    prediction_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for task_name in args.tasks:
        usable_routes = _usable_routes(task_name, requested_routes, route_roots)
        if len(usable_routes) < 2:
            coverage_rows.append(
                {
                    "task": task_name,
                    "method": "Gated stacker",
                    "status": "skipped_fewer_than_two_routes",
                    "routes": ",".join(usable_routes),
                }
            )
            continue
        val_frame = _aligned_frame(task_name, "val", usable_routes, route_roots)
        test_frame = _aligned_frame(task_name, "test", usable_routes, route_roots)
        if val_frame.empty or test_frame.empty:
            coverage_rows.append(
                {
                    "task": task_name,
                    "method": "Gated stacker",
                    "status": "skipped_empty_same_row_intersection",
                    "routes": ",".join(usable_routes),
                    "n_val_same_row": len(val_frame),
                    "n_test_same_row": len(test_frame),
                }
            )
            continue
        model, norm, training_summary = _fit_gated_model(
            val_frame,
            usable_routes,
            args=args,
            device=device,
        )
        pred = _predict_frame(model, norm, test_frame, usable_routes, device=device)
        task_pred_dir = prediction_root / task_name
        task_pred_dir.mkdir(parents=True, exist_ok=True)
        task_dir = run_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        pred.to_parquet(task_pred_dir / "test.parquet", index=False)
        val_frame.to_parquet(task_dir / "stacking_train_val.parquet", index=False)
        test_frame.to_parquet(task_dir / "same_row_test_manifest.parquet", index=False)
        write_json(training_summary, task_dir / "training_summary.json")
        training_rows.append(
            {
                "task": task_name,
                "baseline": "gated_stacker",
                "train": len(val_frame),
                "val": len(val_frame),
                "test": len(test_frame),
                "routes": ",".join(usable_routes),
                **training_summary,
            }
        )
        torch.save({"state_dict": model.state_dict(), "routes": usable_routes, **norm}, checkpoint_dir / f"{task_name}.pt")

        metrics = regression_metrics(pred[TARGET_COL], pred[PREDICTION_COL])
        metrics_rows.append(
            {
                "task": task_name,
                "split": "test",
                "baseline": "gated_stacker",
                "routes": ",".join(usable_routes),
                "n_stacking_train_val": len(val_frame),
                **metrics,
            }
        )
        by_target = metrics_by_target_name(pred)
        by_target.insert(0, "routes", ",".join(usable_routes))
        by_target.insert(0, "baseline", "gated_stacker")
        by_target.insert(0, "split", "test")
        by_target.insert(0, "task", task_name)
        target_metric_frames.append(by_target)

        route_metrics = _route_metrics(test_frame, usable_routes)
        best = route_metrics.sort_values("mae", ascending=True).iloc[0]
        same_row_rows.append(
            {
                "task": task_name,
                "routes": ",".join(usable_routes),
                "n_test": len(test_frame),
                "best_same_row_single_route": best["route"],
                "best_single_mae": best["mae"],
                "fusion_mae": metrics["mae"],
                "delta_mae": metrics["mae"] - best["mae"],
                "fusion_wins": bool(metrics["mae"] < best["mae"]),
            }
        )
        gate_rows.extend(_gate_summary(model, norm, test_frame, usable_routes, device=device, task_name=task_name))
        coverage_rows.append(
            {
                "task": task_name,
                "method": "Gated stacker",
                "status": "ran",
                "routes": ",".join(usable_routes),
                "train": len(val_frame),
                "val": len(val_frame),
                "test": len(test_frame),
                "coverage_percent": None,
            }
        )
        logger.info(
            "%s Gated stacker routes=%s n=%d mae=%.6f best=%s %.6f",
            task_name,
            ",".join(usable_routes),
            metrics["n"],
            metrics["mae"],
            best["route"],
            best["mae"],
        )

    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps(metrics_rows, indent=2, sort_keys=True) + "\n")
    if target_metric_frames:
        pd.concat(target_metric_frames, ignore_index=True).to_csv(run_dir / "target_name_slice_metrics.csv", index=False)
    pd.DataFrame(coverage_rows).to_csv(run_dir / "coverage.csv", index=False)
    pd.DataFrame(same_row_rows).to_csv(run_dir / "same_row_fusion.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(run_dir / "gate_weights.csv", index=False)
    pd.DataFrame(training_rows).to_csv(run_dir / "training_summary.csv", index=False)
    logger.info("Gated stacker run complete: %s", run_dir)
    return run_dir


def _route_roots(
    *,
    results_root: Path,
    neural_results_root: Path,
    run_id: str,
    overrides: list[str],
) -> dict[str, Path]:
    roots = {
        "tabular_metadata": results_root / "tabular_metadata_model" / DEFAULT_CONTEXT_RUN_ID,
        "observation_availability": results_root / "observation_availability_model" / DEFAULT_CONTEXT_RUN_ID,
        "sensor_summary": results_root / "sensor_summary_model" / DEFAULT_CONTEXT_RUN_ID,
        "frozen_image_feature": results_root / "frozen_image_feature_model" / DEFAULT_CONTEXT_RUN_ID,
        "sensor_sequence_transformer": results_root / "sensor_sequence_transformer" / DEFAULT_SENSOR_SEQUENCE_TRANSFORMER_RUN_ID,
        "tabular_transformer": neural_results_root / "tabular_transformer" / run_id,
        "observation_set_transformer": neural_results_root / "observation_set_transformer" / run_id,
        "sensor_sequence_tcn": neural_results_root / "sensor_sequence_tcn" / run_id,
        "field_camera_image_set_model": neural_results_root / "field_camera_image_set_model" / run_id,
    }
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --route-root, expected NAME=PATH: {item}")
        name, path = item.split("=", 1)
        roots[name.strip()] = Path(path)
    return roots


def _prediction_path(root: Path, task_name: str, split: str) -> Path | None:
    neural = root / "predictions" / task_name / f"{split}.parquet"
    if neural.exists():
        return neural
    baseline = root / task_name / f"predictions_{split}.parquet"
    if baseline.exists():
        return baseline
    return None


def _usable_routes(task_name: str, requested_routes: list[str], route_roots: dict[str, Path]) -> list[str]:
    usable = []
    for route in requested_routes:
        root = route_roots.get(route)
        if root is None:
            continue
        if _prediction_path(root, task_name, "val") and _prediction_path(root, task_name, "test"):
            usable.append(route)
    return usable


def _aligned_frame(task_name: str, split: str, routes: list[str], route_roots: dict[str, Path]) -> pd.DataFrame:
    base = None
    for route in routes:
        path = _prediction_path(route_roots[route], task_name, split)
        frame = pd.read_parquet(path)
        keep = ["target_uid", TARGET_COL, "target_name", PREDICTION_COL]
        if base is None:
            base = frame[keep].rename(columns={PREDICTION_COL: f"pred_{route}"})
        else:
            base = base.merge(
                frame[["target_uid", PREDICTION_COL]].rename(columns={PREDICTION_COL: f"pred_{route}"}),
                on="target_uid",
                how="inner",
                validate="one_to_one",
            )
    return base if base is not None else pd.DataFrame()


def _fit_gated_model(frame: pd.DataFrame, routes: list[str], *, args: argparse.Namespace, device: torch.device):
    x = frame[[f"pred_{route}" for route in routes]].to_numpy(dtype=np.float32)
    y = frame[TARGET_COL].to_numpy(dtype=np.float32)
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0) + 1e-6
    y_mean = float(y.mean())
    y_std = float(y.std() + 1e-8)
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
    model = GatedStackerNet(len(routes), args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_norm.astype(np.float32)), torch.from_numpy(y_norm.astype(np.float32))),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    criterion = nn.MSELoss()
    best_state = None
    best_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    start_time = perf_counter()
    epochs_trained = 0
    stopped_early = False
    for epoch in range(args.max_epochs):
        epochs_trained = epoch + 1
        model.train()
        total = 0.0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu()) * len(batch_x)
            count += len(batch_x)
        epoch_loss = total / max(count, 1)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                stopped_early = True
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return (
        model,
        {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std},
        {
            "epochs_trained": int(epochs_trained),
            "best_epoch": int(best_epoch),
            "best_train_loss": float(best_loss),
            "stopped_early": bool(stopped_early),
            "train_rows": int(len(frame)),
            "batch_size": int(args.batch_size),
            "max_epochs": int(args.max_epochs),
            "patience": int(args.patience),
            "train_seconds": float(perf_counter() - start_time),
            "device": str(device),
        },
    )


def _predict_frame(model, norm: dict, frame: pd.DataFrame, routes: list[str], *, device: torch.device) -> pd.DataFrame:
    x = frame[[f"pred_{route}" for route in routes]].to_numpy(dtype=np.float32)
    x_norm = (x - norm["x_mean"]) / norm["x_std"]
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x_norm.astype(np.float32)).to(device)).cpu().numpy()
    output = frame.reset_index(drop=True).copy()
    output[PREDICTION_COL] = pred_norm * norm["y_std"] + norm["y_mean"]
    return attach_errors(output)


def _route_metrics(frame: pd.DataFrame, routes: list[str]) -> pd.DataFrame:
    rows = []
    for route in routes:
        metrics = regression_metrics(frame[TARGET_COL], frame[f"pred_{route}"])
        rows.append({"route": route, **metrics})
    return pd.DataFrame(rows)


def _gate_summary(model, norm: dict, frame: pd.DataFrame, routes: list[str], *, device: torch.device, task_name: str) -> list[dict]:
    x = frame[[f"pred_{route}" for route in routes]].to_numpy(dtype=np.float32)
    x_norm = (x - norm["x_mean"]) / norm["x_std"]
    model.eval()
    with torch.no_grad():
        weights = model.gate_weights(torch.from_numpy(x_norm.astype(np.float32)).to(device)).cpu().numpy()
    mean_weights = weights.mean(axis=0)
    return [
        {"task": task_name, "route": route, "mean_gate_weight": float(weight)}
        for route, weight in zip(routes, mean_weights, strict=True)
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
