"""Run Linear stacker prediction-level fusion from same-row validated route predictions."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

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
from src.benchmarks.baselines.linear_stacker import LinearStacker  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_ROUTE_ROOTS = {
    "tabular_metadata": default_output_root() / "tabular_metadata_model" / "plot_disjoint",
    "sensor_summary": default_output_root() / "sensor_summary_model" / "plot_disjoint",
    "frozen_image_feature": default_output_root() / "frozen_image_feature_model" / "plot_disjoint",
    "sensor_sequence_transformer": default_output_root()
    / "sensor_sequence_transformer"
    / "plot_disjoint_sensor_sequence_transformer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=default_output_root() / "linear_stacker"
    )
    parser.add_argument("--run-id", default="plot_disjoint_linear_stacker")
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument(
        "--routes",
        nargs="+",
        default=["tabular_metadata", "sensor_summary", "frozen_image_feature", "sensor_sequence_transformer"],
        help="Route names to consider. Default: tabular_metadata sensor_summary frozen_image_feature sensor_sequence_transformer.",
    )
    parser.add_argument(
        "--route-root",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override or add a route prediction root.",
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def run(args: argparse.Namespace) -> Path:
    requested_routes = list(dict.fromkeys(args.routes))
    run_id = args.run_id
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    route_roots = _resolve_route_roots(args.route_root)
    write_json(
        {
            "baseline": "Linear stacker",
            "run_id": run_id,
            "tasks": args.tasks,
            "requested_routes": requested_routes,
            "route_roots": {
                name: str(route_roots[name])
                for name in requested_routes
                if name in route_roots
            },
            "stacking_policy": "Fit Ridge on same-row validation predictions; evaluate on same-row test predictions.",
            "route_policy": "Use requested routes only when both validation and test prediction artifacts exist for the task. Raster-geometry route is excluded until real raster zonal extraction exists.",
            "model": {
                "alpha": args.alpha,
                "random_state": args.random_state,
            },
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
        run_dir / "run_config.json",
    )

    metrics_rows: list[dict] = []
    target_metric_frames: list[pd.DataFrame] = []
    route_metric_frames: list[pd.DataFrame] = []
    coverage_rows: list[dict] = []
    weight_rows: list[dict] = []

    for task_name in args.tasks:
        logger.info("Preparing %s", task_name)
        usable_routes = _usable_routes(
            task_name=task_name,
            requested_routes=requested_routes,
            route_roots=route_roots,
        )
        if len(usable_routes) < 2:
            logger.warning(
                "%s skipped: need at least two routes with val/test predictions; got %s",
                task_name,
                usable_routes,
            )
            coverage_rows.append(
                {
                    "task": task_name,
                    "status": "skipped",
                    "reason": "fewer_than_two_usable_routes",
                    "routes": ",".join(usable_routes),
                }
            )
            continue

        train_frame = _aligned_route_frame(
            task_name=task_name,
            split="val",
            routes=usable_routes,
            route_roots=route_roots,
        )
        test_frame = _aligned_route_frame(
            task_name=task_name,
            split="test",
            routes=usable_routes,
            route_roots=route_roots,
        )
        if train_frame.empty or test_frame.empty:
            logger.warning("%s skipped: empty same-row fusion frame", task_name)
            coverage_rows.append(
                {
                    "task": task_name,
                    "status": "skipped",
                    "reason": "empty_same_row_intersection",
                    "routes": ",".join(usable_routes),
                    "n_val_same_row": len(train_frame),
                    "n_test_same_row": len(test_frame),
                }
            )
            continue

        model = LinearStacker(alpha=args.alpha, random_state=args.random_state).fit(
            _route_prediction_arrays(train_frame, usable_routes),
            train_frame[TARGET_COL].to_numpy(dtype=float),
            is_categorical=False,
        )
        prediction_frame = _predict_fusion_frame(
            model=model,
            frame=test_frame,
            routes=usable_routes,
        )
        task_dir = run_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        prediction_frame.to_parquet(task_dir / "predictions_test.parquet", index=False)
        train_frame.to_parquet(task_dir / "stacking_train_val.parquet", index=False)
        test_frame.to_parquet(task_dir / "same_row_test_manifest.parquet", index=False)

        split_metrics = regression_metrics(
            prediction_frame[TARGET_COL], prediction_frame[PREDICTION_COL]
        )
        metrics_rows.append(
            {
                "task": task_name,
                "split": "test",
                "baseline": "Linear stacker",
                "routes": ",".join(usable_routes),
                "n_stacking_train_val": len(train_frame),
                **split_metrics,
            }
        )

        by_target = metrics_by_target_name(prediction_frame)
        by_target.insert(0, "routes", ",".join(usable_routes))
        by_target.insert(0, "baseline", "Linear stacker")
        by_target.insert(0, "split", "test")
        by_target.insert(0, "task", task_name)
        target_metric_frames.append(by_target)

        route_metrics = _route_metrics_on_frame(test_frame, usable_routes)
        route_metrics.insert(0, "task", task_name)
        route_metrics.insert(1, "split", "test")
        route_metric_frames.append(route_metrics)
        route_metrics.to_csv(task_dir / "route_metrics_on_fusion_rows.csv", index=False)

        coverage = _coverage_rows(
            task_name=task_name,
            routes=usable_routes,
            route_roots=route_roots,
            n_val_same_row=len(train_frame),
            n_test_same_row=len(test_frame),
        )
        coverage_rows.extend(coverage)
        pd.DataFrame(coverage).to_csv(task_dir / "route_coverage.csv", index=False)

        weights = _fusion_weight_rows(task_name, usable_routes, model)
        weight_rows.extend(weights)
        pd.DataFrame(weights).to_csv(task_dir / "fusion_weights.csv", index=False)

        best_route = route_metrics.sort_values("mae", ascending=True).iloc[0]
        logger.info(
            "%s Linear stacker routes=%s n=%d mae=%.6f best_route=%s mae=%.6f",
            task_name,
            ",".join(usable_routes),
            split_metrics["n"],
            split_metrics["mae"],
            best_route["baseline"],
            best_route["mae"],
        )

    _write_aggregate_outputs(
        run_dir=run_dir,
        metrics_rows=metrics_rows,
        target_metric_frames=target_metric_frames,
        route_metric_frames=route_metric_frames,
        coverage_rows=coverage_rows,
        weight_rows=weight_rows,
    )
    logger.info("Linear stacker run complete: %s", run_dir)
    return run_dir


def _resolve_route_roots(overrides: list[str]) -> dict[str, Path]:
    route_roots = dict(DEFAULT_ROUTE_ROOTS)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid --route-root value, expected NAME=PATH: {item}")
        name, path = item.split("=", 1)
        route_roots[name.strip()] = Path(path)
    return route_roots


def _usable_routes(
    *,
    task_name: str,
    requested_routes: list[str],
    route_roots: dict[str, Path],
) -> list[str]:
    usable = []
    for route in requested_routes:
        root = route_roots.get(route)
        if root is None:
            logger.warning("Unknown Linear stacker route requested: %s", route)
            continue
        if _prediction_path(root, task_name, "val").exists() and _prediction_path(
            root, task_name, "test"
        ).exists():
            usable.append(route)
    return usable


def _aligned_route_frame(
    *,
    task_name: str,
    split: str,
    routes: list[str],
    route_roots: dict[str, Path],
) -> pd.DataFrame:
    frames = [
        _load_route_predictions(
            route=route,
            path=_prediction_path(route_roots[route], task_name, split),
        )
        for route in routes
    ]
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on="target_uid", how="inner", validate="one_to_one")
    if merged.empty:
        return merged

    first_route = routes[0]
    output = pd.DataFrame(
        {
            "target_uid": merged["target_uid"],
            "target_name": merged[f"target_name_{first_route}"],
            TARGET_COL: merged[f"target_value_num_{first_route}"],
        }
    )
    for route in routes:
        route_target = merged[f"target_value_num_{route}"]
        if not np.allclose(output[TARGET_COL], route_target, equal_nan=False):
            raise ValueError(f"{task_name} {split}: target values differ for route {route}")
        route_name = merged[f"target_name_{route}"].astype(str)
        if not route_name.eq(output["target_name"].astype(str)).all():
            raise ValueError(f"{task_name} {split}: target_name differs for route {route}")
        output[f"prediction_{route}"] = merged[f"prediction_{route}"].astype(float)
    return output.sort_values("target_uid").reset_index(drop=True)


def _load_route_predictions(*, route: str, path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(
        path,
        columns=["target_uid", "target_name", TARGET_COL, PREDICTION_COL],
    )
    if frame["target_uid"].duplicated().any():
        raise ValueError(f"Duplicate target_uid values in {path}")
    return frame.rename(
        columns={
            "target_name": f"target_name_{route}",
            TARGET_COL: f"target_value_num_{route}",
            PREDICTION_COL: f"prediction_{route}",
        }
    )


def _prediction_path(root: Path, task_name: str, split: str) -> Path:
    return root / task_name / f"predictions_{split}.parquet"


def _route_prediction_arrays(
    frame: pd.DataFrame, routes: list[str]
) -> dict[str, np.ndarray]:
    return {
        route: frame[f"prediction_{route}"].to_numpy(dtype=float) for route in routes
    }


def _predict_fusion_frame(
    *,
    model: LinearStacker,
    frame: pd.DataFrame,
    routes: list[str],
) -> pd.DataFrame:
    output = frame.copy()
    output[PREDICTION_COL] = model.predict(_route_prediction_arrays(frame, routes))
    return attach_errors(output)


def _route_metrics_on_frame(frame: pd.DataFrame, routes: list[str]) -> pd.DataFrame:
    rows = []
    for route in routes:
        metrics = regression_metrics(
            frame[TARGET_COL], frame[f"prediction_{route}"].to_numpy(dtype=float)
        )
        rows.append({"baseline": route.upper(), **metrics})
    return pd.DataFrame(rows)


def _coverage_rows(
    *,
    task_name: str,
    routes: list[str],
    route_roots: dict[str, Path],
    n_val_same_row: int,
    n_test_same_row: int,
) -> list[dict]:
    rows: list[dict] = []
    for route in routes:
        for split in ("val", "test"):
            path = _prediction_path(route_roots[route], task_name, split)
            n_rows = len(pd.read_parquet(path, columns=["target_uid"]))
            rows.append(
                {
                    "task": task_name,
                    "status": "used",
                    "route": route,
                    "split": split,
                    "n_route_rows": n_rows,
                    "n_val_same_row": n_val_same_row,
                    "n_test_same_row": n_test_same_row,
                    "routes": ",".join(routes),
                }
            )
    return rows


def _fusion_weight_rows(
    task_name: str, routes: list[str], model: LinearStacker
) -> list[dict]:
    fitted = model.model
    if fitted is None or not hasattr(fitted, "coef_"):
        return []
    coef = np.asarray(fitted.coef_).reshape(-1)
    intercept = float(np.asarray(getattr(fitted, "intercept_", [0.0])).reshape(-1)[0])
    rows = [
        {
            "task": task_name,
            "route": route,
            "weight": float(weight),
            "intercept": intercept,
        }
        for route, weight in zip(model.route_names, coef, strict=True)
    ]
    return rows


def _write_aggregate_outputs(
    *,
    run_dir: Path,
    metrics_rows: list[dict],
    target_metric_frames: list[pd.DataFrame],
    route_metric_frames: list[pd.DataFrame],
    coverage_rows: list[dict],
    weight_rows: list[dict],
) -> None:
    pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
    if target_metric_frames:
        pd.concat(target_metric_frames, ignore_index=True).to_csv(
            run_dir / "metrics_by_target_name.csv", index=False
        )
    if route_metric_frames:
        pd.concat(route_metric_frames, ignore_index=True).to_csv(
            run_dir / "route_metrics_on_fusion_rows.csv", index=False
        )
    pd.DataFrame(coverage_rows).to_csv(run_dir / "route_coverage.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(run_dir / "fusion_weights.csv", index=False)
    (run_dir / "fusion_summary.json").write_text(
        json.dumps(
            {
                "n_tasks_with_metrics": len({row["task"] for row in metrics_rows}),
                "tasks": sorted({row["task"] for row in metrics_rows}),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(parse_args())


if __name__ == "__main__":
    main()
