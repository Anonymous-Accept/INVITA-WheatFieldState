"""Observation-availability model baseline."""

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
class ObservationAvailabilityConfig:
    """Configuration for Observation-availability model."""

    random_state: int = 42
    n_estimators: int = 250
    learning_rate: float = 0.05
    max_depth: int = 5
    categorical_features: list[str] = field(default_factory=lambda: ["target_name"])
    numeric_features: list[str] = field(default_factory=list)


class ObservationAvailabilityModel:
    """
    Observation-availability model gradient boosting predictor.

    Observation-availability model tests whether the pattern of available modalities and pre-target
    observation timing encodes crop-state signal. It does not consume payload
    values.
    """

    def __init__(self, config: ObservationAvailabilityConfig | None = None) -> None:
        self.config = config or ObservationAvailabilityConfig()
        self.model = None
        self._numeric_features: list[str] = []

    @property
    def feature_columns(self) -> list[str]:
        """Return the ordered model feature list."""

        return self.config.categorical_features + self._numeric_features

    def fit(self, data: pd.DataFrame) -> ObservationAvailabilityModel:
        """Fit the model from the training split only."""

        frame = self._prepare_frame(data, require_target=True, fit=True)
        self.model = build_regression_pipeline(
            categorical_features=self.config.categorical_features,
            numeric_features=self._numeric_features,
            random_state=self.config.random_state,
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            max_depth=self.config.max_depth,
        )
        self.model.fit(frame[self.feature_columns], frame[TARGET_COL])
        return self

    def predict_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict values and return a prediction frame."""

        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        frame = self._prepare_frame(data, require_target=False, fit=False)
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

        if self.model is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        fitted_model = self.model.named_steps["model"]
        if not hasattr(fitted_model, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])
        return pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": fitted_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

    def _prepare_frame(
        self, data: pd.DataFrame, *, require_target: bool, fit: bool
    ) -> pd.DataFrame:
        frame = data.copy()
        for col in self.config.categorical_features:
            if col not in frame.columns:
                raise ValueError(f"Missing required Observation-availability model categorical column: {col}")
            frame[col] = normalize_categorical(frame[col])

        if fit:
            self._numeric_features = self.config.numeric_features or [
                col
                for col in frame.columns
                if col.startswith(("n_", "has_", "days_since_last_"))
            ]
            self._numeric_features = sorted(set(self._numeric_features))

        missing = set(self._numeric_features) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required Observation-availability model numeric columns: {sorted(missing)}")
        for col in self._numeric_features:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)

        if require_target:
            frame[TARGET_COL] = pd.to_numeric(frame[TARGET_COL], errors="raise")
            if frame[TARGET_COL].isna().any():
                raise ValueError("target_value_num contains missing values")
        return frame
