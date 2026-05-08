"""Shared utilities for rigorous INVITA baseline experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

try:  # pragma: no cover - exercised only when LightGBM is installed.
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None

from src.benchmarks.paths import default_data_root


TASKS = (
    "NDVI",
    "LAI",
    "FCover",
    "Zadoks",
)

SPLITS = ("train", "val", "test")

TARGET_COL = "target_value_num"
PREDICTION_COL = "prediction"


@dataclass(frozen=True)
class BaselineDataConfig:
    """Filesystem configuration shared by baseline-specific dataloaders."""

    data_root: Path = field(default_factory=default_data_root)
    split_root: Path | None = None
    enforce_plot_exclusivity: bool = True


@dataclass
class BaselineTaskData:
    """Train/validation/test frames for one baseline and one task."""

    task_name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    validation_report: dict

    def split(self, name: str) -> pd.DataFrame:
        """Return one named split."""

        if name not in SPLITS:
            raise ValueError(f"Unknown split: {name}")
        return getattr(self, name)


class PredictsFrame(Protocol):
    """Protocol for baseline predictors used by generic runners."""

    def fit(self, data: pd.DataFrame) -> object:
        """Fit from a training frame."""

    def predict_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a frame containing predictions."""


def split_dir(config: BaselineDataConfig, task_name: str) -> Path:
    """Resolve the split directory for one task."""

    if config.split_root is not None:
        return Path(config.split_root) / task_name
    return Path(config.data_root) / "tasks" / task_name / "splits"


def validate_task_name(task_name: str) -> None:
    """Raise if a task name is unknown."""

    if task_name not in TASKS:
        raise ValueError(f"Unknown task: {task_name}")


def read_split_frame(
    split_path: Path,
    targets: pd.DataFrame,
    *,
    required_columns: tuple[str, ...],
    drop_unknown_targets: bool = False,
) -> pd.DataFrame:
    """Load one split CSV and join it to a prepared target table."""

    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    split_df = pd.read_csv(split_path, usecols=["target_uid"])
    if split_df["target_uid"].duplicated().any():
        duplicates = split_df.loc[
            split_df["target_uid"].duplicated(), "target_uid"
        ].head(5)
        raise ValueError(f"Duplicate target_uid values in {split_path}: {duplicates}")

    if drop_unknown_targets:
        target_ids = set(targets["target_uid"].tolist())
        split_df = split_df.loc[split_df["target_uid"].isin(target_ids)].copy()

    frame = split_df.merge(targets, on="target_uid", how="left", validate="one_to_one")
    if frame["plot_uid"].isna().any():
        missing = frame.loc[frame["plot_uid"].isna(), "target_uid"].head(5).tolist()
        raise ValueError(f"Split references unknown target_uid values: {missing}")

    missing_columns = set(required_columns) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Split frame missing required columns: {sorted(missing_columns)}"
        )
    return frame


def validate_split_integrity(
    task_name: str,
    targets: pd.DataFrame,
    split_frames: dict[str, pd.DataFrame],
    *,
    enforce_plot_exclusivity: bool = True,
) -> dict:
    """Validate target coverage, target leakage, and plot leakage."""

    split_ids = {
        split: set(frame["target_uid"].tolist()) for split, frame in split_frames.items()
    }
    target_ids = set(targets["target_uid"].tolist())
    union_ids = set().union(*split_ids.values())

    missing_from_splits = target_ids - union_ids
    unknown_in_splits = union_ids - target_ids
    if missing_from_splits:
        raise ValueError(
            f"{task_name} has targets missing from splits: "
            f"{sorted(missing_from_splits)[:5]}"
        )
    if unknown_in_splits:
        raise ValueError(
            f"{task_name} splits include unknown targets: "
            f"{sorted(unknown_in_splits)[:5]}"
        )

    uid_overlaps: dict[str, int] = {}
    plot_overlaps: dict[str, int] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        uid_overlap = split_ids[left] & split_ids[right]
        if uid_overlap:
            raise ValueError(
                f"{task_name} split target leakage {left}-{right}: "
                f"{sorted(uid_overlap)[:5]}"
            )
        uid_overlaps[f"{left}_{right}"] = 0

        left_plots = set(split_frames[left]["plot_uid"].astype(str))
        right_plots = set(split_frames[right]["plot_uid"].astype(str))
        plot_overlap = left_plots & right_plots
        if plot_overlap and enforce_plot_exclusivity:
            raise ValueError(
                f"{task_name} split plot leakage {left}-{right}: "
                f"{sorted(plot_overlap)[:5]}"
            )
        plot_overlaps[f"{left}_{right}"] = len(plot_overlap)

    return {
        "task_name": task_name,
        "n_targets": int(len(targets)),
        "split_rows": {split: int(len(frame)) for split, frame in split_frames.items()},
        "target_uid_overlap": uid_overlaps,
        "plot_uid_overlap": plot_overlaps,
    }


def write_json(payload: dict, path: Path) -> None:
    """Write a JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def normalize_categorical(series: pd.Series) -> pd.Series:
    """Normalize categorical metadata while preserving missingness as unknown."""

    output = series.fillna("unknown").astype(str).str.strip()
    output.loc[output == ""] = "unknown"
    output.loc[output.str.lower().isin({"none", "nan", "<na>"})] = "unknown"
    return output


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach date-derived features used by metadata baselines."""

    output = frame.copy()
    output["target_date"] = pd.to_datetime(output["target_date"], errors="raise")
    output["target_doy"] = output["target_date"].dt.dayofyear.astype(int)
    output["target_month"] = output["target_date"].dt.month.astype(int)
    output["target_week"] = output["target_date"].dt.isocalendar().week.astype(int)
    output["target_doy_bin_14"] = ((output["target_doy"] - 1) // 14).astype(int)

    if "sowing_date" in output.columns:
        sowing = pd.to_datetime(output["sowing_date"], errors="coerce")
        output["sowing_doy"] = sowing.dt.dayofyear
        output["days_since_sowing"] = (output["target_date"] - sowing).dt.days
    else:
        output["sowing_doy"] = np.nan
        output["days_since_sowing"] = np.nan
    return output


def regression_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> dict[str, float | int]:
    """Compute regression metrics used for all four numeric target tasks."""

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    return {
        "n": int(len(y_true_array)),
        "mae": float(mean_absolute_error(y_true_array, y_pred_array)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_array, y_pred_array))),
        "r2": float(r2_score(y_true_array, y_pred_array))
        if len(np.unique(y_true_array)) > 1
        else float("nan"),
    }


def attach_errors(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """Attach error columns to a prediction frame."""

    output = prediction_frame.copy()
    output["error"] = output[PREDICTION_COL] - output[TARGET_COL]
    output["absolute_error"] = output["error"].abs()
    output["squared_error"] = output["error"] ** 2
    return output


def metrics_by_target_name(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics per target_name."""

    rows: list[dict[str, float | int | str]] = []
    for target_name, group in prediction_frame.groupby("target_name", dropna=False):
        metrics = regression_metrics(group[TARGET_COL], group[PREDICTION_COL])
        rows.append({"target_name": str(target_name), **metrics})
    return pd.DataFrame(rows).sort_values("target_name").reset_index(drop=True)


def build_regression_pipeline(
    *,
    categorical_features: list[str],
    numeric_features: list[str],
    random_state: int,
    n_estimators: int,
    learning_rate: float = 0.05,
    max_depth: int = 6,
) -> Pipeline:
    """Build a deterministic regression pipeline for tabular baselines."""

    categorical_transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_transformer, categorical_features),
            ("numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    if LGBMRegressor is not None:
        model = LGBMRegressor(
            objective="regression",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    else:  # pragma: no cover
        model = HistGradientBoostingRegressor(
            max_iter=n_estimators,
            learning_rate=learning_rate,
            max_leaf_nodes=31,
            random_state=random_state,
        )

    return Pipeline([("preprocess", preprocessor), ("model", model)])
