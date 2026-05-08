"""
Linear stacker: Prediction-level fusion

Stacks predictions from validated single-route baselines using Ridge regression.
Raster-geometry route is not required by this interface; it can be added later only after real
raster zonal extraction is available.

ML-for-Agriculture Question:
    Are metadata, temporal, raster, and field-camera image routes complementary?

Required For:
    Same-row fusion diagnostics

Broad multimodal fusion claims are allowed only if Linear stacker beats the best
single-route baseline on identical rows.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class LinearStacker:
    """Linear stacker: Prediction-level fusion of validated baseline predictions."""

    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        self.alpha = alpha
        self.random_state = random_state
        self.model = None
        self.is_categorical = None
        self.label_encoder = None
        self.route_names: list[str] = []

    def fit(
        self,
        route_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
        is_categorical: bool = False,
    ) -> LinearStacker:
        """Fit fusion model on same-row predictions from validated routes."""

        logger.info("Fitting Linear stacker prediction-level fusion model")
        self.is_categorical = is_categorical
        self.route_names = sorted(route_predictions)
        x = _stack_routes(route_predictions, self.route_names)
        if x.shape[1] < 2:
            raise ValueError("Linear stacker requires at least two validated route predictions")
        if len(y_true) != x.shape[0]:
            raise ValueError("y_true length does not match route prediction rows")

        logger.info("Fusion input shape: %s", x.shape)
        if self.is_categorical:
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y_true)
            self.model = RidgeClassifier(alpha=self.alpha, random_state=self.random_state)
        else:
            y = y_true
            self.model = Ridge(alpha=self.alpha, random_state=self.random_state)

        self.model.fit(x, y)
        if hasattr(self.model, "coef_"):
            logger.info("Fusion weights: %s", self.model.coef_)
        return self

    def predict(self, route_predictions: dict[str, np.ndarray]) -> np.ndarray:
        """Predict using the fitted same-row fusion model."""

        if self.model is None:
            raise ValueError("Model not fitted.")
        x = _stack_routes(route_predictions, self.route_names)
        predictions = self.model.predict(x)
        if self.is_categorical and self.label_encoder:
            predictions = self.label_encoder.inverse_transform(predictions.astype(int))
        return predictions

    def evaluate(
        self,
        route_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> dict[str, float]:
        """Predict and evaluate fusion on same-row route predictions."""

        predictions = self.predict(route_predictions)
        metrics: dict[str, float] = {}
        if self.is_categorical:
            metrics["accuracy"] = accuracy_score(y_true, predictions)
            metrics["f1_macro"] = f1_score(
                y_true, predictions, average="macro", zero_division=0
            )
        else:
            metrics["mae"] = mean_absolute_error(y_true, predictions)
            metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, predictions)))
            metrics["r2"] = r2_score(y_true, predictions)
        return metrics


def _stack_routes(
    route_predictions: dict[str, np.ndarray], route_names: list[str]
) -> np.ndarray:
    missing = set(route_names) - set(route_predictions)
    if missing:
        raise ValueError(f"Missing route predictions: {sorted(missing)}")
    arrays = [np.asarray(route_predictions[name]) for name in route_names]
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError("Route predictions must have identical row counts")
    return np.column_stack(arrays)
