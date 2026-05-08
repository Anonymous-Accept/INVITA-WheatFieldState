"""Small PyTorch tabular transformer regressor."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.benchmarks.baselines._shared import PREDICTION_COL, TARGET_COL, normalize_categorical


@dataclass
class TabularTransformerConfig:
    """Configuration for compact tabular transformer regressors."""

    categorical_features: list[str] = field(default_factory=list)
    numeric_features: list[str] = field(default_factory=list)
    embedding_dim: int = 64
    n_layers: int = 3
    n_heads: int = 4
    dropout: float = 0.1
    mlp_hidden: int = 128
    batch_size: int = 2048
    max_epochs: int = 100
    patience: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str | None = None


class TabularTransformerNet(nn.Module):
    """Transformer over categorical tokens plus one numeric token."""

    def __init__(
        self,
        *,
        cardinalities: list[int],
        n_numeric: int,
        config: TabularTransformerConfig,
    ) -> None:
        super().__init__()
        if config.embedding_dim % config.n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.cat_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, config.embedding_dim)
            for cardinality in cardinalities
        )
        self.numeric_projection = nn.Sequential(
            nn.Linear(max(n_numeric, 1), config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.n_heads,
            dim_feedforward=config.embedding_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(config.embedding_dim),
            nn.Linear(config.embedding_dim, config.mlp_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden, 1),
        )

    def forward(self, cat_x: torch.Tensor, num_x: torch.Tensor) -> torch.Tensor:
        tokens = [self.cls.expand(cat_x.shape[0], -1, -1)]
        for idx, embedding in enumerate(self.cat_embeddings):
            tokens.append(embedding(cat_x[:, idx]).unsqueeze(1))
        tokens.append(self.numeric_projection(num_x).unsqueeze(1))
        encoded = self.encoder(torch.cat(tokens, dim=1))
        return self.head(encoded[:, 0]).squeeze(-1)


class TabularTransformerRegressor:
    """Fit/predict wrapper with train-only preprocessing and early stopping."""

    def __init__(self, config: TabularTransformerConfig) -> None:
        self.config = config
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.category_maps: dict[str, dict[str, int]] = {}
        self.numeric_mean: np.ndarray | None = None
        self.numeric_std: np.ndarray | None = None
        self.target_mean: float | None = None
        self.target_std: float | None = None
        self.model: TabularTransformerNet | None = None
        self.training_summary: dict[str, Any] = {}
        set_torch_seed(config.seed)

    @property
    def feature_columns(self) -> list[str]:
        return self.config.categorical_features + self.config.numeric_features

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "TabularTransformerRegressor":
        cat_train, num_train = self._fit_transform_x(train)
        y = pd.to_numeric(train[TARGET_COL], errors="raise").to_numpy(dtype=np.float32)
        self.target_mean = float(y.mean())
        self.target_std = float(y.std() + 1e-8)
        y_train = ((y - self.target_mean) / self.target_std).astype(np.float32)

        cardinalities = [
            max(mapping.values()) + 1 for mapping in self.category_maps.values()
        ]
        self.model = TabularTransformerNet(
            cardinalities=cardinalities,
            n_numeric=max(len(self.config.numeric_features), 1),
            config=self.config,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        train_loader = _loader(
            cat_train, num_train, y_train, self.config.batch_size, shuffle=True, seed=self.config.seed
        )

        val_payload = None
        if val is not None and not val.empty:
            cat_val, num_val = self._transform_x(val)
            y_val = pd.to_numeric(val[TARGET_COL], errors="raise").to_numpy(dtype=np.float32)
            val_payload = (cat_val, num_val, y_val)

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
            for batch_cat, batch_num, batch_y in train_loader:
                batch_cat = batch_cat.to(self.device)
                batch_num = batch_num.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(batch_cat, batch_num), batch_y)
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
        cat_x, num_x = self._transform_x(data)
        self.model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(data), self.config.batch_size):
                cat = torch.from_numpy(cat_x[start : start + self.config.batch_size]).to(
                    self.device
                )
                num = torch.from_numpy(num_x[start : start + self.config.batch_size]).to(
                    self.device
                )
                preds.append(self.model(cat, num).cpu().numpy())
        y_norm = np.concatenate(preds) if preds else np.array([], dtype=float)
        output = data.reset_index(drop=True).copy()
        output[PREDICTION_COL] = y_norm * self.target_std + self.target_mean
        return output

    def _fit_transform_x(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        for feature in self.config.categorical_features:
            values = normalize_categorical(data[feature]).tolist()
            uniques = sorted(set(values))
            self.category_maps[feature] = {"__UNK__": 0, **{value: idx + 1 for idx, value in enumerate(uniques)}}
        numeric = _numeric_matrix(data, self.config.numeric_features)
        if numeric.shape[1]:
            valid = np.isfinite(numeric)
            counts = valid.sum(axis=0)
            sums = np.where(valid, numeric, 0.0).sum(axis=0)
            self.numeric_mean = np.divide(
                sums,
                counts,
                out=np.zeros(numeric.shape[1], dtype=np.float32),
                where=counts > 0,
            ).astype(np.float32)
            centered = np.where(valid, numeric - self.numeric_mean, 0.0)
            variances = np.divide(
                (centered**2).sum(axis=0),
                counts,
                out=np.ones(numeric.shape[1], dtype=np.float32),
                where=counts > 0,
            )
            self.numeric_std = np.sqrt(variances).astype(np.float32)
            self.numeric_std[self.numeric_std < 1e-6] = 1.0
        else:
            self.numeric_mean = np.zeros(1, dtype=np.float32)
            self.numeric_std = np.ones(1, dtype=np.float32)
        return self._transform_x(data)

    def _transform_x(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        missing = set(self.feature_columns) - set(data.columns)
        if missing:
            raise ValueError(f"Missing tabular features: {sorted(missing)}")
        cats = []
        for feature in self.config.categorical_features:
            mapping = self.category_maps[feature]
            values = normalize_categorical(data[feature]).map(mapping).fillna(0).astype(int)
            cats.append(values.to_numpy(dtype=np.int64))
        cat_x = (
            np.column_stack(cats).astype(np.int64)
            if cats
            else np.zeros((len(data), 0), dtype=np.int64)
        )
        numeric = _numeric_matrix(data, self.config.numeric_features)
        if numeric.shape[1] == 0:
            numeric = np.zeros((len(data), 1), dtype=np.float32)
        else:
            numeric = np.where(np.isfinite(numeric), numeric, self.numeric_mean)
            numeric = (numeric - self.numeric_mean) / self.numeric_std
        return cat_x, numeric.astype(np.float32)

    def _validation_mae(self, val_payload: tuple[np.ndarray, np.ndarray, np.ndarray] | None) -> float:
        if val_payload is None:
            return 0.0
        if self.model is None or self.target_mean is None or self.target_std is None:
            raise ValueError("Model is not fitted")
        cat_x, num_x, y = val_payload
        preds = self._predict_arrays(cat_x, num_x)
        pred_y = preds * self.target_std + self.target_mean
        return float(np.mean(np.abs(pred_y - y)))

    def _predict_arrays(self, cat_x: np.ndarray, num_x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted")
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(cat_x), self.config.batch_size):
                cat = torch.from_numpy(cat_x[start : start + self.config.batch_size]).to(self.device)
                num = torch.from_numpy(num_x[start : start + self.config.batch_size]).to(self.device)
                preds.append(self.model(cat, num).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([], dtype=float)


def set_torch_seed(seed: int) -> None:
    """Set deterministic seeds where practical."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _numeric_matrix(data: pd.DataFrame, features: list[str]) -> np.ndarray:
    if not features:
        return np.zeros((len(data), 0), dtype=np.float32)
    return data[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)


def _loader(
    cat_x: np.ndarray,
    num_x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(cat_x.astype(np.int64)),
        torch.from_numpy(num_x.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )
