"""Crop-date prior baseline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.benchmarks.baselines._shared import (
    PREDICTION_COL,
    TARGET_COL,
    normalize_categorical,
    regression_metrics,
)

logger = logging.getLogger(__name__)

FALLBACK_COL = "fallback_level"


@dataclass(frozen=True)
class HierarchyLevel:
    """One lookup level in the Crop-date prior lookup ladder."""

    name: str
    columns: tuple[str, ...]
    min_samples: int = 1


@dataclass
class CropDatePriorConfig:
    """Configuration for Crop-date prior."""

    aggregation: str = "mean"
    min_samples: int = 3
    hierarchy_levels: list[HierarchyLevel] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.aggregation not in {"mean", "median"}:
            raise ValueError("aggregation must be 'mean' or 'median'")
        if self.min_samples <= 0:
            raise ValueError("min_samples must be positive")
        if not self.hierarchy_levels:
            self.hierarchy_levels = default_hierarchy_levels(self.min_samples)


def default_hierarchy_levels(min_samples: int = 3) -> list[HierarchyLevel]:
    """Return the canonical Crop-date prior lookup ladder."""

    return [
        HierarchyLevel(
            "target_crop_cultivar_year_calendar",
            ("target_name", "crop_type", "cultivar_key", "trial_year", "target_doy_bin_14"),
            min_samples,
        ),
        HierarchyLevel(
            "target_crop_cultivar_calendar",
            ("target_name", "crop_type", "cultivar_key", "target_doy_bin_14"),
            min_samples,
        ),
        HierarchyLevel(
            "target_crop_year_calendar",
            ("target_name", "crop_type", "trial_year", "target_doy_bin_14"),
            min_samples,
        ),
        HierarchyLevel(
            "target_crop_calendar",
            ("target_name", "crop_type", "target_doy_bin_14"),
            min_samples,
        ),
        HierarchyLevel(
            "target_calendar",
            ("target_name", "target_doy_bin_14"),
            min_samples,
        ),
        HierarchyLevel("target_global", ("target_name",), 1),
    ]


class CropDatePrior:
    """
    Hierarchical agronomic mean predictor.

    Crop-date prior tests whether cultivar and crop hierarchy explains target variation
    beyond the Source-date prior baseline. It never uses payload values
    or post-target observations.
    """

    def __init__(self, config: CropDatePriorConfig | None = None) -> None:
        self.config = config or CropDatePriorConfig()
        self.tables: dict[str, pd.DataFrame] = {}
        self.global_fallback: float | None = None

    def fit(self, data: pd.DataFrame) -> CropDatePrior:
        """Fit lookup tables from the training split only."""

        frame = self._prepare_frame(data, require_target=True)
        self.global_fallback = float(frame[TARGET_COL].mean())
        self.tables = {}

        for level in self.config.hierarchy_levels:
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
            logger.info("Crop-date prior fitted %s with %d groups", level.name, len(grouped))

        return self

    def predict_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict values and return prediction metadata."""

        if self.global_fallback is None or not self.tables:
            raise ValueError("Model is not fitted. Call fit() first.")

        frame = self._prepare_frame(data, require_target=False).reset_index(drop=True)
        frame["_row_id"] = np.arange(len(frame))
        result = pd.DataFrame({"_row_id": frame["_row_id"]})
        result[PREDICTION_COL] = np.nan
        result[FALLBACK_COL] = pd.NA

        for level in self.config.hierarchy_levels:
            table = self.tables[level.name]
            if table.empty:
                continue

            unresolved = result.loc[result[PREDICTION_COL].isna(), "_row_id"]
            if unresolved.empty:
                break

            candidates = frame.loc[frame["_row_id"].isin(unresolved)].copy()
            merged = candidates.merge(
                table[list(level.columns) + [PREDICTION_COL]],
                on=list(level.columns),
                how="left",
                validate="many_to_one",
            )
            hits = merged[merged[PREDICTION_COL].notna()][["_row_id", PREDICTION_COL]]
            if hits.empty:
                continue

            hit_values = hits.set_index("_row_id")[PREDICTION_COL]
            hit_rows = result["_row_id"].isin(hits["_row_id"])
            result.loc[hit_rows, PREDICTION_COL] = result.loc[
                hit_rows, "_row_id"
            ].map(hit_values)
            result.loc[hit_rows, FALLBACK_COL] = level.name

        unresolved_rows = result[PREDICTION_COL].isna()
        if unresolved_rows.any():
            result.loc[unresolved_rows, PREDICTION_COL] = self.global_fallback
            result.loc[unresolved_rows, FALLBACK_COL] = "global"

        output = data.reset_index(drop=True).copy()
        output[PREDICTION_COL] = result[PREDICTION_COL].astype(float).to_numpy()
        output[FALLBACK_COL] = result[FALLBACK_COL].astype(str).to_numpy()
        return output

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return predictions as a NumPy array."""

        return self.predict_frame(data)[PREDICTION_COL].to_numpy()

    def evaluate(self, data: pd.DataFrame) -> dict[str, float | int]:
        """Evaluate one labeled split."""

        return regression_metrics(data[TARGET_COL], self.predict(data))

    def fallback_usage(self, prediction_frame: pd.DataFrame) -> pd.DataFrame:
        """Summarize fallback usage for an evaluated split."""

        counts = prediction_frame[FALLBACK_COL].value_counts(dropna=False).rename_axis(
            FALLBACK_COL
        )
        usage = counts.reset_index(name="n_rows")
        usage["fraction"] = usage["n_rows"] / len(prediction_frame)
        return usage.sort_values(["n_rows", FALLBACK_COL], ascending=[False, True])

    def _prepare_frame(self, data: pd.DataFrame, require_target: bool) -> pd.DataFrame:
        columns = self.required_columns(require_target=require_target)
        missing = columns - set(data.columns)
        if missing:
            raise ValueError(f"Missing required Crop-date prior columns: {sorted(missing)}")

        frame = data.copy()
        for col in ("target_name", "crop_type", "cultivar_key"):
            frame[col] = normalize_categorical(frame[col])
        frame["trial_year"] = pd.to_numeric(frame["trial_year"], errors="raise").astype(
            int
        )
        frame["target_doy_bin_14"] = pd.to_numeric(
            frame["target_doy_bin_14"], errors="raise"
        ).astype(int)
        if require_target:
            frame[TARGET_COL] = pd.to_numeric(frame[TARGET_COL], errors="raise")
            if frame[TARGET_COL].isna().any():
                raise ValueError("target_value_num contains missing values")
        return frame

    def required_columns(self, require_target: bool = True) -> set[str]:
        """Return required predictor columns."""

        columns: set[str] = set()
        for level in self.config.hierarchy_levels:
            columns.update(level.columns)
        if require_target:
            columns.add(TARGET_COL)
        return columns
