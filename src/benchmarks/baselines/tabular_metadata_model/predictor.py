"""Tabular metadata model baseline."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.benchmarks.baselines._shared import (
    PREDICTION_COL,
    TARGET_COL,
    build_regression_pipeline,
    normalize_categorical,
    regression_metrics,
)


@dataclass
class TabularMetadataConfig:
    """Configuration for the Tabular metadata model."""

    random_state: int = 42
    n_estimators: int = 300
    learning_rate: float = 0.05
    max_depth: int = 6
    categorical_features: list[str] = field(default_factory=list)
    numeric_features: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.categorical_features:
            self.categorical_features = default_categorical_features()
        if not self.numeric_features:
            self.numeric_features = default_numeric_features()


def default_categorical_features() -> list[str]:
    """Return the canonical Tabular metadata model categorical metadata features."""

    return [
        "target_name",
        "source_dataset",
        "instrument",
        "subset",
        "state",
        "region_name",
        "site_id",
        "site_name",
        "crop_type",
        "cultivar_key",
        "replicate",
        "block",
    ]


def default_numeric_features() -> list[str]:
    """Return the canonical Tabular metadata model numeric metadata features."""

    return [
        "trial_year",
        "target_doy",
        "target_month",
        "target_week",
        "target_doy_bin_14",
        "sowing_doy",
        "days_since_sowing",
        "area_m2",
    ]


class TabularMetadataModel:
    """
    Tabular metadata model gradient boosting predictor.

    Tabular metadata model excludes raw identifiers, payload values, weather values, sensor values,
    raster data, and RGB data. Cultivar/genotype metadata is used only when it is
    available in plot metadata.
    """

    def __init__(self, config: TabularMetadataConfig | None = None) -> None:
        self.config = config or TabularMetadataConfig()
        self.model = build_regression_pipeline(
            categorical_features=self.config.categorical_features,
            numeric_features=self.config.numeric_features,
            random_state=self.config.random_state,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
        )

    @property
    def feature_columns(self) -> list[str]:
        """Return the ordered model feature list."""

        return self.config.categorical_features + self.config.numeric_features

    def fit(self, data: pd.DataFrame) -> TabularMetadataModel:
        """Fit the model from the training split only."""

        frame = self._prepare_frame(data, require_target=True)
        self.model.fit(frame[self.feature_columns], frame[TARGET_COL])
        return self

    def predict_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict values and return a prediction frame."""

        frame = self._prepare_frame(data, require_target=False)
        predictions = self.model.predict(frame[self.feature_columns])
        output = data.reset_index(drop=True).copy()
        output[PREDICTION_COL] = np.asarray(predictions, dtype=float)
        return output

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Return predictions as a NumPy array."""

        return self.predict_frame(data)[PREDICTION_COL].to_numpy()

    def evaluate(self, data: pd.DataFrame) -> dict[str, float | int]:
        """Evaluate one labeled split."""

        return regression_metrics(data[TARGET_COL], self.predict(data))

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importance when the underlying model exposes it."""

        fitted_model = self.model.named_steps["model"]
        if not hasattr(fitted_model, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])

        return pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": fitted_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    def _prepare_frame(self, data: pd.DataFrame, require_target: bool) -> pd.DataFrame:
        missing = set(self.feature_columns) - set(data.columns)
        if require_target:
            missing |= {TARGET_COL} - set(data.columns)
        if missing:
            raise ValueError(f"Missing required Tabular metadata model columns: {sorted(missing)}")

        frame = data.copy()
        for col in self.config.categorical_features:
            frame[col] = normalize_categorical(frame[col])
        for col in self.config.numeric_features:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        if require_target:
            frame[TARGET_COL] = pd.to_numeric(frame[TARGET_COL], errors="raise")
            if frame[TARGET_COL].isna().any():
                raise ValueError("target_value_num contains missing values")
        return frame
