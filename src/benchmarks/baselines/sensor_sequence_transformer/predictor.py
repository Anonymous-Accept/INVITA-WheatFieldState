"""Sensor-sequence Transformer masked temporal Transformer baseline for sensor histories."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.benchmarks.baselines._shared import (
    PREDICTION_COL,
    TARGET_COL,
    regression_metrics,
)

SEQ_RE = re.compile(
    r"^hist_(?P<metric>.+)_seq_(?P<bin>\d+)_(?P<kind>mean|count)$"
)


@dataclass
class SensorSequenceTransformerConfig:
    """Configuration for the Sensor-sequence Transformer masked temporal model."""

    random_state: int = 42
    hidden_dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    batch_size: int = 2048
    n_epochs: int = 120
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    min_feature_std: float = 1.0
    feature_clip: float = 10.0
    device: str | None = None
    num_workers: int = 0
    sequence_features: list[str] = field(default_factory=list)


@dataclass
class SequenceLayout:
    """Fitted layout and normalizers for fixed-bin history sequences."""

    metrics: list[str]
    bins: list[int]
    value_mean: np.ndarray
    value_std: np.ndarray
    count_mean: np.ndarray
    count_std: np.ndarray
    total_count_mean: np.ndarray
    total_count_std: np.ndarray
    days_since_mean: np.ndarray
    days_since_std: np.ndarray


class MaskedTemporalTransformer(nn.Module):
    """Masked temporal encoder with attention-based adaptive metric fusion."""

    def __init__(
        self,
        *,
        n_metrics: int,
        n_bins: int,
        input_dim: int,
        hidden_dim: int,
        n_layers: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError("hidden_dim must be divisible by n_heads")
        self.n_metrics = n_metrics
        self.n_bins = n_bins
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.metric_embedding = nn.Embedding(n_metrics, hidden_dim)
        self.time_embedding = nn.Embedding(n_bins, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.time_attention = nn.Linear(hidden_dim, 1)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        metric_ids = torch.arange(n_metrics).repeat_interleave(n_bins)
        time_ids = torch.arange(n_bins).repeat(n_metrics)
        self.register_buffer("metric_ids", metric_ids, persistent=False)
        self.register_buffer("time_ids", time_ids, persistent=False)

    def forward(
        self,
        values: torch.Tensor,
        observed_mask: torch.Tensor,
        metric_stats: torch.Tensor,
    ) -> torch.Tensor:
        n_rows, n_metrics, n_bins, n_features = values.shape
        token_values = values.reshape(n_rows, n_metrics * n_bins, n_features)
        token_mask = observed_mask.reshape(n_rows, n_metrics * n_bins)

        tokens = self.input_projection(token_values)
        tokens = tokens + self.metric_embedding(self.metric_ids).unsqueeze(0)
        tokens = tokens + self.time_embedding(self.time_ids).unsqueeze(0)

        encoded = self.encoder(tokens, src_key_padding_mask=~token_mask)
        encoded = encoded.reshape(n_rows, n_metrics, n_bins, -1)

        time_scores = self.time_attention(encoded).squeeze(-1)
        time_weights = _masked_softmax(time_scores, observed_mask, dim=2)
        metric_embeddings = (encoded * time_weights.unsqueeze(-1)).sum(dim=2)

        metric_present = observed_mask.any(dim=2)
        fusion_input = torch.cat([metric_embeddings, metric_stats], dim=-1)
        fusion_scores = self.fusion_gate(fusion_input).squeeze(-1)
        fusion_weights = _masked_softmax(fusion_scores, metric_present, dim=1)
        pooled = (metric_embeddings * fusion_weights.unsqueeze(-1)).sum(dim=1)
        return self.regressor(pooled).squeeze(-1)


class SensorSequenceTransformer:
    """
    Learned masked temporal sensor-history baseline.

    Sensor-sequence Transformer consumes only leakage-controlled fixed-bin sequence columns from the Sensor-summary model
    legal-input history builder. Missing bins are represented by explicit masks,
    and metric streams are fused with an attention gate.
    """

    def __init__(self, config: SensorSequenceTransformerConfig | None = None) -> None:
        self.config = config or SensorSequenceTransformerConfig()
        if self.config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        self.model: MaskedTemporalTransformer | None = None
        self.layout: SequenceLayout | None = None
        self.target_mean: float | None = None
        self.target_std: float | None = None
        torch.manual_seed(self.config.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.random_state)

    def fit(self, data: pd.DataFrame) -> SensorSequenceTransformer:
        """Fit the masked temporal model from the training split only."""

        values, observed_mask, metric_stats = self._prepare_tensors(data, fit=True)
        y = pd.to_numeric(data[TARGET_COL], errors="raise").to_numpy(dtype=np.float32)
        self.target_mean = float(y.mean())
        self.target_std = float(y.std() + 1e-8)
        y_norm = (y - self.target_mean) / self.target_std

        if self.layout is None:
            raise ValueError("Sensor-sequence Transformer sequence layout was not fitted")
        self.model = MaskedTemporalTransformer(
            n_metrics=len(self.layout.metrics),
            n_bins=len(self.layout.bins),
            input_dim=values.shape[-1],
            hidden_dim=self.config.hidden_dim,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            dropout=self.config.dropout,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.MSELoss()
        dataset = TensorDataset(
            torch.from_numpy(values.astype(np.float32)),
            torch.from_numpy(observed_mask.astype(bool)),
            torch.from_numpy(metric_stats.astype(np.float32)),
            torch.from_numpy(y_norm.astype(np.float32)),
        )
        generator = torch.Generator().manual_seed(self.config.random_state)
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )

        self.model.train()
        for _ in range(self.config.n_epochs):
            for batch_values, batch_mask, batch_stats, batch_y in loader:
                batch_values = batch_values.to(self.device, non_blocking=True)
                batch_mask = batch_mask.to(self.device, non_blocking=True)
                batch_stats = batch_stats.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(
                    self.model(batch_values, batch_mask, batch_stats), batch_y
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                optimizer.step()
        return self

    def predict_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict values and return a prediction frame."""

        if self.model is None or self.target_mean is None or self.target_std is None:
            raise ValueError("Model is not fitted. Call fit() first.")
        values, observed_mask, metric_stats = self._prepare_tensors(data, fit=False)
        self.model.eval()
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(values), self.config.batch_size):
                batch_values = torch.from_numpy(
                    values[start : start + self.config.batch_size].astype(np.float32)
                ).to(self.device)
                batch_mask = torch.from_numpy(
                    observed_mask[start : start + self.config.batch_size].astype(bool)
                ).to(self.device)
                batch_stats = torch.from_numpy(
                    metric_stats[start : start + self.config.batch_size].astype(
                        np.float32
                    )
                ).to(self.device)
                pred = self.model(batch_values, batch_mask, batch_stats).cpu().numpy()
                predictions.append(pred)
        y_norm = (
            np.concatenate(predictions) if predictions else np.array([], dtype=float)
        )
        output = data.reset_index(drop=True).copy()
        output[PREDICTION_COL] = y_norm * self.target_std + self.target_mean
        return output

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        return self.predict_frame(data)[PREDICTION_COL].to_numpy()

    def evaluate(self, data: pd.DataFrame) -> dict[str, float | int]:
        return regression_metrics(data[TARGET_COL], self.predict(data))

    def _prepare_tensors(
        self, data: pd.DataFrame, *, fit: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if fit:
            self.layout = self._fit_layout(data)
        if self.layout is None:
            raise ValueError("Sensor-sequence Transformer sequence layout is not fitted")

        layout = self.layout
        n_rows = len(data)
        n_metrics = len(layout.metrics)
        n_bins = len(layout.bins)
        values = np.zeros((n_rows, n_metrics, n_bins, 3), dtype=np.float32)
        observed_mask = np.zeros((n_rows, n_metrics, n_bins), dtype=bool)
        metric_stats = np.zeros((n_rows, n_metrics, 3), dtype=np.float32)

        for metric_idx, metric in enumerate(layout.metrics):
            metric_counts = np.zeros((n_rows, n_bins), dtype=np.float32)
            metric_means = np.zeros((n_rows, n_bins), dtype=np.float32)
            metric_observed = np.zeros((n_rows, n_bins), dtype=bool)

            for bin_pos, bin_idx in enumerate(layout.bins):
                mean_col = _seq_col(metric, bin_idx, "mean")
                count_col = _seq_col(metric, bin_idx, "count")
                means = _numeric_column(data, mean_col, n_rows, fill_value=np.nan)
                counts = _numeric_column(data, count_col, n_rows, fill_value=0.0)
                observed = (counts > 0) & np.isfinite(means)
                metric_means[:, bin_pos] = np.nan_to_num(means, nan=0.0)
                metric_counts[:, bin_pos] = np.nan_to_num(counts, nan=0.0)
                metric_observed[:, bin_pos] = observed

            value_norm = (
                metric_means - layout.value_mean[metric_idx]
            ) / layout.value_std[metric_idx]
            count_log = np.log1p(metric_counts)
            count_norm = (
                count_log - layout.count_mean[metric_idx]
            ) / layout.count_std[metric_idx]
            time_pos = (
                np.linspace(0.0, 1.0, n_bins, dtype=np.float32)
                if n_bins > 1
                else np.zeros(n_bins, dtype=np.float32)
            )

            values[:, metric_idx, :, 0] = np.where(metric_observed, value_norm, 0.0)
            values[:, metric_idx, :, 1] = np.where(metric_observed, count_norm, 0.0)
            values[:, metric_idx, :, 2] = time_pos[None, :]
            observed_mask[:, metric_idx, :] = metric_observed

            total_counts = metric_counts.sum(axis=1)
            present = total_counts > 0
            total_count_norm = (
                np.log1p(total_counts) - layout.total_count_mean[metric_idx]
            ) / layout.total_count_std[metric_idx]
            days_col = f"hist_{metric}_days_since_last"
            days = _numeric_column(data, days_col, n_rows, fill_value=999.0)
            days = np.where(present, days, 999.0)
            days_norm = (
                np.log1p(np.clip(days, 0.0, 999.0))
                - layout.days_since_mean[metric_idx]
            ) / layout.days_since_std[metric_idx]
            metric_stats[:, metric_idx, 0] = present.astype(np.float32)
            metric_stats[:, metric_idx, 1] = total_count_norm
            metric_stats[:, metric_idx, 2] = days_norm

        values = np.clip(values, -self.config.feature_clip, self.config.feature_clip)
        metric_stats[:, :, 1:] = np.clip(
            metric_stats[:, :, 1:],
            -self.config.feature_clip,
            self.config.feature_clip,
        )
        return values, observed_mask, metric_stats

    def _fit_layout(self, data: pd.DataFrame) -> SequenceLayout:
        metrics, bins = _discover_sequence_layout(data, self.config.sequence_features)
        if not metrics or not bins:
            raise ValueError("No Sensor-sequence Transformer sequence mean/count features found")

        value_mean = []
        value_std = []
        count_mean = []
        count_std = []
        total_count_mean = []
        total_count_std = []
        days_since_mean = []
        days_since_std = []

        n_rows = len(data)
        for metric in metrics:
            observed_values: list[np.ndarray] = []
            observed_counts: list[np.ndarray] = []
            total_counts = np.zeros(n_rows, dtype=np.float32)
            for bin_idx in bins:
                means = _numeric_column(
                    data, _seq_col(metric, bin_idx, "mean"), n_rows, fill_value=np.nan
                )
                counts = _numeric_column(
                    data, _seq_col(metric, bin_idx, "count"), n_rows, fill_value=0.0
                )
                observed = (counts > 0) & np.isfinite(means)
                if observed.any():
                    observed_values.append(means[observed].astype(np.float32))
                    observed_counts.append(np.log1p(counts[observed]).astype(np.float32))
                total_counts += np.nan_to_num(counts, nan=0.0).astype(np.float32)

            all_values = (
                np.concatenate(observed_values)
                if observed_values
                else np.array([0.0], dtype=np.float32)
            )
            all_counts = (
                np.concatenate(observed_counts)
                if observed_counts
                else np.array([0.0], dtype=np.float32)
            )
            value_mean.append(float(all_values.mean()))
            value_std.append(_stable_std(all_values, self.config.min_feature_std))
            count_mean.append(float(all_counts.mean()))
            count_std.append(_stable_std(all_counts, self.config.min_feature_std))

            total_count_log = np.log1p(total_counts)
            total_count_mean.append(float(total_count_log.mean()))
            total_count_std.append(
                _stable_std(total_count_log, self.config.min_feature_std)
            )

            days = _numeric_column(
                data, f"hist_{metric}_days_since_last", n_rows, fill_value=999.0
            )
            days = np.where(total_counts > 0, days, 999.0)
            days_log = np.log1p(np.clip(days, 0.0, 999.0)).astype(np.float32)
            days_since_mean.append(float(days_log.mean()))
            days_since_std.append(_stable_std(days_log, self.config.min_feature_std))

        return SequenceLayout(
            metrics=metrics,
            bins=bins,
            value_mean=np.asarray(value_mean, dtype=np.float32),
            value_std=np.asarray(value_std, dtype=np.float32),
            count_mean=np.asarray(count_mean, dtype=np.float32),
            count_std=np.asarray(count_std, dtype=np.float32),
            total_count_mean=np.asarray(total_count_mean, dtype=np.float32),
            total_count_std=np.asarray(total_count_std, dtype=np.float32),
            days_since_mean=np.asarray(days_since_mean, dtype=np.float32),
            days_since_std=np.asarray(days_since_std, dtype=np.float32),
        )


def _masked_softmax(
    scores: torch.Tensor, mask: torch.Tensor, *, dim: int
) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool)
    masked_scores = scores.masked_fill(~mask, -1e9)
    weights = torch.softmax(masked_scores, dim=dim)
    weights = weights * mask.to(dtype=weights.dtype)
    denom = weights.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return weights / denom


def _discover_sequence_layout(
    data: pd.DataFrame, configured_features: list[str]
) -> tuple[list[str], list[int]]:
    columns = configured_features or list(data.columns)
    means: set[tuple[str, int]] = set()
    counts: set[tuple[str, int]] = set()
    for column in columns:
        match = SEQ_RE.match(str(column))
        if not match:
            continue
        key = (match.group("metric"), int(match.group("bin")))
        if match.group("kind") == "mean":
            means.add(key)
        else:
            counts.add(key)
    pairs = means & counts
    metrics = sorted({metric for metric, _ in pairs})
    bins = sorted({bin_idx for _, bin_idx in pairs})
    return metrics, bins


def _seq_col(metric: str, bin_idx: int, kind: str) -> str:
    return f"hist_{metric}_seq_{bin_idx:02d}_{kind}"


def _numeric_column(
    data: pd.DataFrame, column: str, n_rows: int, *, fill_value: float
) -> np.ndarray:
    if column not in data.columns:
        return np.full(n_rows, fill_value, dtype=np.float32)
    values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(fill_value):
        return values
    return np.nan_to_num(values, nan=fill_value).astype(np.float32)


def _stable_std(values: np.ndarray, minimum: float) -> float:
    std = float(np.std(values))
    return std if std >= minimum else float(minimum)
