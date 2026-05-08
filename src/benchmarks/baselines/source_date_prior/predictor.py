"""
Source-date prior baseline.

This module contains only the statistical predictor. It does not read files,
create splits, or inspect payload assets. Those responsibilities live in the
baseline-specific dataloader and runner.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


TARGET_COL = "target_value_num"
DATE_COL = "target_date"
PREDICTION_COL = "prediction"
FALLBACK_COL = "fallback_level"


@dataclass(frozen=True)
class GroupLevel:
    """A named climatology lookup level."""

    name: str
    columns: tuple[str, ...]
    min_samples: int = 1


@dataclass
class SourceDatePriorConfig:
    """Configuration for Source-date prior."""

    aggregation: str = "mean"
    doy_bin_size: int = 14
    min_samples: int = 3
    group_levels: list[GroupLevel] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.aggregation not in {"mean", "median"}:
            raise ValueError("aggregation must be 'mean' or 'median'")
        if self.doy_bin_size <= 0:
            raise ValueError("doy_bin_size must be positive")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if not self.group_levels:
            self.group_levels = default_group_levels(self.min_samples)


def default_group_levels(min_samples: int = 3) -> list[GroupLevel]:
    """Return the canonical Source-date prior fallback ladder."""

    return [
        GroupLevel(
            "site_source_year_calendar",
            (
                "target_name",
                "source_dataset",
                "instrument",
                "trial_year",
                "state",
                "region_name",
                "site_id",
                "crop_type",
                "doy_bin",
            ),
            min_samples,
        ),
        GroupLevel(
            "site_year_calendar",
            (
                "target_name",
                "instrument",
                "trial_year",
                "state",
                "region_name",
                "site_id",
                "crop_type",
                "doy_bin",
            ),
            min_samples,
        ),
        GroupLevel(
            "region_year_calendar",
            (
                "target_name",
                "instrument",
                "trial_year",
                "state",
                "region_name",
                "crop_type",
                "doy_bin",
            ),
            min_samples,
        ),
        GroupLevel(
            "state_year_calendar",
            ("target_name", "trial_year", "state", "crop_type", "doy_bin"),
            min_samples,
        ),
        GroupLevel(
            "crop_year_calendar",
            ("target_name", "trial_year", "crop_type", "doy_bin"),
            min_samples,
        ),
        GroupLevel(
            "target_calendar",
            ("target_name", "doy_bin"),
            min_samples,
        ),
        GroupLevel("target_global", ("target_name",), 1),
    ]


class SourceDatePrior:
    """
    Source-aware climatology with an explicit fallback ladder.

    Source-date prior tests how much can be predicted from target provenance, site/crop
    structure, season, and calendar position alone. It deliberately does not use
    sensor, raster, RGB, payload, or weather measurements.
    """

    def __init__(self, config: SourceDatePriorConfig | None = None) -> None:
        self.config = config or SourceDatePriorConfig()
        self.tables: dict[str, pd.DataFrame] = {}
        self.global_fallback: float | None = None
        self.fitted_columns: set[str] = set()

    def fit(self, data: pd.DataFrame) -> SourceDatePrior:
        """Fit climatology lookup tables from the training split only."""

        frame = self._prepare_frame(data, require_target=True)
        self.fitted_columns = set(frame.columns)
        self.global_fallback = float(frame[TARGET_COL].mean())
        self.tables = {}

        for level in self.config.group_levels:
            missing = set(level.columns) - set(frame.columns)
            if missing:
                raise ValueError(f"Missing columns for {level.name}: {sorted(missing)}")

            grouped = (
                frame.groupby(list(level.columns), dropna=False)[TARGET_COL]
                .agg([self.config.aggregation, "count"])
                .reset_index()
                .rename(
                    columns={
                        self.config.aggregation: PREDICTION_COL,
                        "count": "n_samples",
                    }
                )
            )
            grouped = grouped[grouped["n_samples"] >= level.min_samples].copy()
            self.tables[level.name] = grouped
            logger.info(
                "Source-date prior fitted %s with %d groups", level.name, len(grouped)
            )

        logger.info("Source-date prior global fallback: %.6f", self.global_fallback)
        return self

    def predict_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict and return per-row prediction metadata.

        The returned frame preserves the input row order and contains
        `prediction` plus `fallback_level` columns.
        """

        if self.global_fallback is None or not self.tables:
            raise ValueError("Model is not fitted. Call fit() first.")

        frame = self._prepare_frame(data, require_target=False).reset_index(drop=True)
        frame["_row_id"] = np.arange(len(frame))
        result = pd.DataFrame({"_row_id": frame["_row_id"]})
        result[PREDICTION_COL] = np.nan
        result[FALLBACK_COL] = pd.NA

        for level in self.config.group_levels:
            table = self.tables[level.name]
            if table.empty:
                continue

            unresolved_ids = result.loc[result[PREDICTION_COL].isna(), "_row_id"]
            if unresolved_ids.empty:
                break

            candidates = frame.loc[frame["_row_id"].isin(unresolved_ids)].copy()
            merged = candidates.merge(
                table[list(level.columns) + [PREDICTION_COL]],
                on=list(level.columns),
                how="left",
                validate="many_to_one",
            )
            hits = merged[merged[PREDICTION_COL].notna()][["_row_id", PREDICTION_COL]]
            if hits.empty:
                continue

            hit_index = result["_row_id"].isin(hits["_row_id"])
            hit_values = hits.set_index("_row_id")[PREDICTION_COL]
            result.loc[hit_index, PREDICTION_COL] = result.loc[
                hit_index, "_row_id"
            ].map(hit_values)
            result.loc[hit_index, FALLBACK_COL] = level.name

        unresolved = result[PREDICTION_COL].isna()
        if unresolved.any():
            result.loc[unresolved, PREDICTION_COL] = self.global_fallback
            result.loc[unresolved, FALLBACK_COL] = "global"

        output = data.reset_index(drop=True).copy()
        output[PREDICTION_COL] = result[PREDICTION_COL].astype(float).to_numpy()
        output[FALLBACK_COL] = result[FALLBACK_COL].astype(str).to_numpy()
        return output

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return predictions as a NumPy array."""

        return self.predict_frame(data)[PREDICTION_COL].to_numpy()

    def evaluate(self, data: pd.DataFrame) -> dict[str, float | int]:
        """Evaluate predictions for a labeled split."""

        predictions = self.predict(data)
        return regression_metrics(data[TARGET_COL].to_numpy(), predictions)

    def fallback_usage(self, prediction_frame: pd.DataFrame) -> pd.DataFrame:
        """Summarize fallback ladder usage for an evaluated split."""

        counts = prediction_frame[FALLBACK_COL].value_counts(dropna=False).rename_axis(
            FALLBACK_COL
        )
        usage = counts.reset_index(name="n_rows")
        usage["fraction"] = usage["n_rows"] / len(prediction_frame)
        return usage.sort_values(["n_rows", FALLBACK_COL], ascending=[False, True])

    def _prepare_frame(self, data: pd.DataFrame, require_target: bool) -> pd.DataFrame:
        required = self.required_columns(require_target=require_target)
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"Missing required Source-date prior columns: {sorted(missing)}")

        frame = data.copy()
        frame[DATE_COL] = pd.to_datetime(frame[DATE_COL], errors="raise")
        frame["doy_bin"] = compute_doy_bin(frame[DATE_COL], self.config.doy_bin_size)

        categorical_cols = set().union(
            *(set(level.columns) for level in self.config.group_levels)
        ) - {"doy_bin", "trial_year"}
        for col in categorical_cols:
            if col in frame.columns:
                frame[col] = frame[col].fillna("unknown").astype(str)
                frame.loc[frame[col].str.len() == 0, col] = "unknown"

        if "trial_year" in frame.columns:
            frame["trial_year"] = pd.to_numeric(frame["trial_year"], errors="raise").astype(
                int
            )

        if require_target:
            frame[TARGET_COL] = pd.to_numeric(frame[TARGET_COL], errors="raise")
            if frame[TARGET_COL].isna().any():
                raise ValueError("target_value_num contains missing values")

        return frame

    def required_columns(self, require_target: bool = True) -> set[str]:
        columns = {DATE_COL}
        for level in self.config.group_levels:
            columns.update(level.columns)
        columns.discard("doy_bin")
        if require_target:
            columns.add(TARGET_COL)
        return columns


def compute_doy_bin(dates: pd.Series, doy_bin_size: int) -> pd.Series:
    """Compute zero-based day-of-year bins."""

    parsed = pd.to_datetime(dates, errors="raise")
    return ((parsed.dt.dayofyear - 1) // doy_bin_size).astype(int)


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float | int]:
    """Compute the numeric metrics used by Source-date prior for all four tasks."""

    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true_array, y_pred_array)
    rmse = float(np.sqrt(mean_squared_error(y_true_array, y_pred_array)))
    metrics: dict[str, float | int] = {
        "n": int(len(y_true_array)),
        "mae": float(mae),
        "rmse": rmse,
        "r2": float(r2_score(y_true_array, y_pred_array))
        if len(np.unique(y_true_array)) > 1
        else float("nan"),
    }
    return metrics


def metrics_by_target_name(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics separately for each target_name."""

    rows: list[dict[str, float | int | str]] = []
    for target_name, group in prediction_frame.groupby("target_name", dropna=False):
        metrics = regression_metrics(group[TARGET_COL], group[PREDICTION_COL])
        rows.append({"target_name": str(target_name), **metrics})
    return pd.DataFrame(rows).sort_values("target_name").reset_index(drop=True)


def attach_errors(prediction_frame: pd.DataFrame) -> pd.DataFrame:
    """Attach absolute and squared error columns to a prediction manifest."""

    output = prediction_frame.copy()
    output["error"] = output[PREDICTION_COL] - output[TARGET_COL]
    output["absolute_error"] = output["error"].abs()
    output["squared_error"] = output["error"] ** 2
    return output
