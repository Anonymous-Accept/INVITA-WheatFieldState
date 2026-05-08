"""PyTorch temporal convolution regressor for sensor-history features."""

from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.benchmarks.baselines._shared import PREDICTION_COL, TARGET_COL
from src.benchmarks.neural_baselines.common.torch_tabular import set_torch_seed

SEQ_RE = re.compile(r"^hist_(?P<metric>.+)_seq_(?P<bin>\d+)_(?P<kind>mean|count)$")


@dataclass
class SensorSequenceTCNConfig:
    """Configuration for the Sensor-sequence TCN baseline."""

    hidden_channels: int = 64
    n_layers: int = 3
    kernel_size: int = 3
    dropout: float = 0.1
    batch_size: int = 1024
    max_epochs: int = 80
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_feature_std: float = 1.0
    feature_clip: float = 10.0
    seed: int = 42
    device: str | None = None


class TemporalConvBlock(nn.Module):
    """Residual 1D temporal convolution block."""

    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GELU(),
        )
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class SensorSequenceTCNNet(nn.Module):
    """TCN over fixed sensor-history time bins."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_channels: int,
        n_layers: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
        ]
        layers.extend(
            TemporalConvBlock(hidden_channels, kernel_size, dropout)
            for _ in range(n_layers)
        )
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor, time_mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        weights = time_mask.float().unsqueeze(1)
        pooled = (encoded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        return self.head(pooled).squeeze(-1)


class SensorSequenceTCNRegressor:
    """Fit/predict wrapper for Sensor-sequence TCN."""

    def __init__(self, config: SensorSequenceTCNConfig) -> None:
        self.config = config
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.metrics: list[str] = []
        self.bins: list[int] = []
        self.value_mean: np.ndarray | None = None
        self.value_std: np.ndarray | None = None
        self.count_mean: np.ndarray | None = None
        self.count_std: np.ndarray | None = None
        self.target_mean: float | None = None
        self.target_std: float | None = None
        self.model: SensorSequenceTCNNet | None = None
        self.training_summary: dict[str, object] = {}
        set_torch_seed(config.seed)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "SensorSequenceTCNRegressor":
        x_train, mask_train = self._fit_transform(train)
        y = pd.to_numeric(train[TARGET_COL], errors="raise").to_numpy(dtype=np.float32)
        self.target_mean = float(y.mean())
        self.target_std = float(y.std() + 1e-8)
        y_norm = ((y - self.target_mean) / self.target_std).astype(np.float32)
        self.model = SensorSequenceTCNNet(
            in_channels=x_train.shape[1],
            hidden_channels=self.config.hidden_channels,
            n_layers=self.config.n_layers,
            kernel_size=self.config.kernel_size,
            dropout=self.config.dropout,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loader = _loader(x_train, mask_train, y_norm, self.config.batch_size, seed=self.config.seed)
        val_payload = None
        if val is not None and not val.empty:
            x_val, mask_val = self._transform(val)
            y_val = pd.to_numeric(val[TARGET_COL], errors="raise").to_numpy(dtype=np.float32)
            val_payload = (x_val, mask_val, y_val)
        best_state = None
        best_mae = float("inf")
        best_epoch = -1
        bad_epochs = 0
        criterion = nn.MSELoss()
        start_time = perf_counter()
        epochs_trained = 0
        stopped_early = False
        for epoch in range(self.config.max_epochs):
            epochs_trained = epoch + 1
            self.model.train()
            for batch_x, batch_mask, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_mask = batch_mask.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(batch_x, batch_mask), batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                optimizer.step()
            mae = self._validation_mae(val_payload)
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }
                bad_epochs = 0
            else:
                bad_epochs += 1
                if val_payload is not None and bad_epochs >= self.config.patience:
                    stopped_early = True
                    break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.training_summary = {
            "epochs_trained": int(epochs_trained),
            "best_epoch": int(best_epoch),
            "best_val_mae": float(best_mae),
            "stopped_early": bool(stopped_early),
            "train_rows": int(len(train)),
            "val_rows": int(len(val)) if val is not None else 0,
            "batch_size": int(self.config.batch_size),
            "max_epochs": int(self.config.max_epochs),
            "patience": int(self.config.patience),
            "train_seconds": float(perf_counter() - start_time),
            "device": str(self.device),
        }
        return self

    def predict_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.target_mean is None or self.target_std is None:
            raise ValueError("Model is not fitted")
        x, mask = self._transform(data)
        preds = self._predict_arrays(x, mask)
        output = data.reset_index(drop=True).copy()
        output[PREDICTION_COL] = preds * self.target_std + self.target_mean
        return output

    def _fit_transform(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        self.metrics, self.bins = _discover_layout(data)
        if not self.metrics or not self.bins:
            raise ValueError("No hist_*_seq_* mean/count features found")
        raw_values, raw_counts, observed = _raw_arrays(data, self.metrics, self.bins)
        self.value_mean, self.value_std = _metric_stats(raw_values, observed, self.config.min_feature_std)
        log_counts = np.log1p(raw_counts)
        self.count_mean, self.count_std = _metric_stats(log_counts, observed, self.config.min_feature_std)
        return self._arrays_to_channels(raw_values, raw_counts, observed)

    def _transform(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raw_values, raw_counts, observed = _raw_arrays(data, self.metrics, self.bins)
        return self._arrays_to_channels(raw_values, raw_counts, observed)

    def _arrays_to_channels(
        self,
        raw_values: np.ndarray,
        raw_counts: np.ndarray,
        observed: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        value_norm = (raw_values - self.value_mean[:, None]) / self.value_std[:, None]
        count_norm = (np.log1p(raw_counts) - self.count_mean[:, None]) / self.count_std[:, None]
        value_norm = np.where(observed, value_norm, 0.0)
        count_norm = np.where(observed, count_norm, 0.0)
        channels = np.concatenate(
            [
                value_norm,
                count_norm,
                observed.astype(np.float32),
            ],
            axis=1,
        )
        channels = np.clip(channels, -self.config.feature_clip, self.config.feature_clip)
        time_mask = observed.any(axis=1)
        return channels.astype(np.float32), time_mask.astype(bool)

    def _validation_mae(self, payload: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> float:
        if payload is None:
            return 0.0
        x, mask, y = payload
        preds = self._predict_arrays(x, mask)
        return float(np.mean(np.abs(preds * self.target_std + self.target_mean - y)))

    def _predict_arrays(self, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted")
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(x), self.config.batch_size):
                batch_x = torch.from_numpy(x[start : start + self.config.batch_size]).to(self.device)
                batch_mask = torch.from_numpy(mask[start : start + self.config.batch_size]).to(self.device)
                preds.append(self.model(batch_x, batch_mask).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([], dtype=float)


def _discover_layout(data: pd.DataFrame) -> tuple[list[str], list[int]]:
    means: set[tuple[str, int]] = set()
    counts: set[tuple[str, int]] = set()
    for column in data.columns:
        match = SEQ_RE.match(str(column))
        if not match:
            continue
        key = (match.group("metric"), int(match.group("bin")))
        if match.group("kind") == "mean":
            means.add(key)
        else:
            counts.add(key)
    pairs = means & counts
    return sorted({metric for metric, _ in pairs}), sorted({bin_idx for _, bin_idx in pairs})


def _raw_arrays(
    data: pd.DataFrame, metrics: list[str], bins: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.zeros((len(data), len(metrics), len(bins)), dtype=np.float32)
    counts = np.zeros_like(values)
    observed = np.zeros_like(values, dtype=bool)
    for metric_idx, metric in enumerate(metrics):
        for bin_pos, bin_idx in enumerate(bins):
            mean_col = f"hist_{metric}_seq_{bin_idx:02d}_mean"
            count_col = f"hist_{metric}_seq_{bin_idx:02d}_count"
            means = _numeric_column(data, mean_col, fill_value=np.nan)
            cnts = _numeric_column(data, count_col, fill_value=0.0)
            obs = (cnts > 0) & np.isfinite(means)
            values[:, metric_idx, bin_pos] = np.nan_to_num(means, nan=0.0)
            counts[:, metric_idx, bin_pos] = np.nan_to_num(cnts, nan=0.0)
            observed[:, metric_idx, bin_pos] = obs
    return values, counts, observed


def _numeric_column(data: pd.DataFrame, column: str, *, fill_value: float) -> np.ndarray:
    if column not in data:
        return np.full(len(data), fill_value, dtype=np.float32)
    values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=np.float32)
    if np.isnan(fill_value):
        return values
    return np.nan_to_num(values, nan=fill_value).astype(np.float32)


def _metric_stats(values: np.ndarray, observed: np.ndarray, minimum_std: float) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for metric_idx in range(values.shape[1]):
        vals = values[:, metric_idx, :][observed[:, metric_idx, :]]
        if len(vals) == 0:
            vals = np.array([0.0], dtype=np.float32)
        means.append(float(vals.mean()))
        stds.append(max(float(vals.std()), minimum_std))
    return np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32)


def _loader(x: np.ndarray, mask: np.ndarray, y: np.ndarray, batch_size: int, *, seed: int) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(mask.astype(bool)),
            torch.from_numpy(y.astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
