"""PyTorch Observation-set Transformer regressor."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.benchmarks.baselines._shared import PREDICTION_COL, TARGET_COL, normalize_categorical
from src.benchmarks.neural_baselines.common.torch_tabular import set_torch_seed


@dataclass
class ObservationSetTransformerConfig:
    """Configuration for the Observation-set Transformer."""

    query_categorical_features: list[str] = field(default_factory=list)
    query_numeric_features: list[str] = field(default_factory=list)
    max_tokens: int = 64
    embedding_dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    mlp_hidden: int = 128
    batch_size: int = 1024
    max_epochs: int = 60
    patience: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    device: str | None = None


TOKEN_CAT_FIELDS = ("modality", "relationship", "time_window")
TOKEN_NUM_FIELDS = (
    "count",
    "unique_assets",
    "unique_dates",
    "min_days_before",
    "mean_days_before",
    "max_days_before",
    "has_dated_observation",
)


class ObservationSetTransformerNet(nn.Module):
    """Transformer over query token and available-observation tokens."""

    def __init__(
        self,
        *,
        query_cardinalities: list[int],
        token_cardinalities: list[int],
        n_query_numeric: int,
        n_token_numeric: int,
        config: ObservationSetTransformerConfig,
    ) -> None:
        super().__init__()
        if config.embedding_dim % config.n_heads != 0:
            raise ValueError("embedding_dim must be divisible by n_heads")
        self.query_cat_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, config.embedding_dim)
            for cardinality in query_cardinalities
        )
        self.token_cat_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, config.embedding_dim)
            for cardinality in token_cardinalities
        )
        self.query_numeric_projection = nn.Sequential(
            nn.Linear(max(n_query_numeric, 1), config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
        )
        self.token_numeric_projection = nn.Sequential(
            nn.Linear(n_token_numeric, config.embedding_dim),
            nn.LayerNorm(config.embedding_dim),
            nn.GELU(),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        self.query_type = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        self.obs_type = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
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

    def forward(
        self,
        query_cat: torch.Tensor,
        query_num: torch.Tensor,
        token_cat: torch.Tensor,
        token_num: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = query_num.shape[0]
        query = self.query_numeric_projection(query_num)
        for idx, embedding in enumerate(self.query_cat_embeddings):
            query = query + embedding(query_cat[:, idx])
        query = query.unsqueeze(1) + self.query_type

        obs = self.token_numeric_projection(token_num)
        for idx, embedding in enumerate(self.token_cat_embeddings):
            obs = obs + embedding(token_cat[:, :, idx])
        obs = obs + self.obs_type

        cls = self.cls.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, query, obs], dim=1)
        cls_query_mask = torch.ones((batch_size, 2), dtype=torch.bool, device=token_mask.device)
        attention_mask = torch.cat([cls_query_mask, token_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=~attention_mask)
        return self.head(encoded[:, 0]).squeeze(-1)


class ObservationSetTransformerRegressor:
    """Fit/predict wrapper for the Observation-set Transformer."""

    def __init__(self, config: ObservationSetTransformerConfig) -> None:
        self.config = config
        self.device = torch.device(
            config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.query_maps: dict[str, dict[str, int]] = {}
        self.token_maps: dict[str, dict[str, int]] = {}
        self.query_mean: np.ndarray | None = None
        self.query_std: np.ndarray | None = None
        self.token_mean: np.ndarray | None = None
        self.token_std: np.ndarray | None = None
        self.target_mean: float | None = None
        self.target_std: float | None = None
        self.model: ObservationSetTransformerNet | None = None
        self.training_summary: dict[str, Any] = {}
        set_torch_seed(config.seed)

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None) -> "ObservationSetTransformerRegressor":
        arrays = self._fit_transform(train)
        y = pd.to_numeric(train[TARGET_COL], errors="raise").to_numpy(dtype=np.float32)
        self.target_mean = float(y.mean())
        self.target_std = float(y.std() + 1e-8)
        y_norm = ((y - self.target_mean) / self.target_std).astype(np.float32)
        self.model = ObservationSetTransformerNet(
            query_cardinalities=[max(m.values()) + 1 for m in self.query_maps.values()],
            token_cardinalities=[max(m.values()) + 1 for m in self.token_maps.values()],
            n_query_numeric=max(len(self.config.query_numeric_features), 1),
            n_token_numeric=len(TOKEN_NUM_FIELDS),
            config=self.config,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loader = _obs_loader(*arrays, y_norm, batch_size=self.config.batch_size, shuffle=True, seed=self.config.seed)
        val_payload = None
        if val is not None and not val.empty:
            val_payload = (*self._transform(val), pd.to_numeric(val[TARGET_COL], errors="raise").to_numpy(dtype=np.float32))
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
            for batch in loader:
                q_cat, q_num, t_cat, t_num, t_mask, batch_y = [item.to(self.device) for item in batch]
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model(q_cat, q_num, t_cat, t_num, t_mask.bool()), batch_y)
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
        arrays = self._transform(data)
        preds = self._predict_arrays(arrays)
        output = data.reset_index(drop=True).copy()
        output[PREDICTION_COL] = preds * self.target_std + self.target_mean
        return output

    def _fit_transform(self, data: pd.DataFrame):
        for feature in self.config.query_categorical_features:
            values = normalize_categorical(data[feature]).tolist()
            self.query_maps[feature] = {"__UNK__": 0, **{v: i + 1 for i, v in enumerate(sorted(set(values)))}}
        token_values = _all_tokens(data)
        for field in TOKEN_CAT_FIELDS:
            values = sorted({str(token.get(field, "unknown")) for token in token_values})
            self.token_maps[field] = {"__UNK__": 0, **{v: i + 1 for i, v in enumerate(values)}}
        query_numeric = _numeric_matrix(data, self.config.query_numeric_features)
        self.query_mean, self.query_std = _fit_normalizer(query_numeric)
        token_numeric = np.array(
            [[float(token.get(field, 0.0)) for field in TOKEN_NUM_FIELDS] for token in token_values],
            dtype=np.float32,
        )
        self.token_mean, self.token_std = _fit_normalizer(token_numeric)
        return self._transform(data)

    def _transform(self, data: pd.DataFrame):
        q_cats = []
        for feature in self.config.query_categorical_features:
            mapping = self.query_maps[feature]
            q_cats.append(normalize_categorical(data[feature]).map(mapping).fillna(0).astype(int).to_numpy())
        query_cat = np.column_stack(q_cats).astype(np.int64) if q_cats else np.zeros((len(data), 0), dtype=np.int64)
        query_num = _numeric_matrix(data, self.config.query_numeric_features)
        if query_num.shape[1] == 0:
            query_num = np.zeros((len(data), 1), dtype=np.float32)
        else:
            query_num = _normalize(query_num, self.query_mean, self.query_std)

        token_cat = np.zeros((len(data), self.config.max_tokens, len(TOKEN_CAT_FIELDS)), dtype=np.int64)
        token_num = np.zeros((len(data), self.config.max_tokens, len(TOKEN_NUM_FIELDS)), dtype=np.float32)
        token_mask = np.zeros((len(data), self.config.max_tokens), dtype=bool)
        for row_idx, payload in enumerate(data["obs_tokens_json"].tolist()):
            tokens = _parse_tokens(payload)[: self.config.max_tokens]
            for token_idx, token in enumerate(tokens):
                token_mask[row_idx, token_idx] = True
                for field_idx, field in enumerate(TOKEN_CAT_FIELDS):
                    token_cat[row_idx, token_idx, field_idx] = self.token_maps[field].get(str(token.get(field, "unknown")), 0)
                token_num[row_idx, token_idx, :] = [
                    float(token.get(field, 0.0)) for field in TOKEN_NUM_FIELDS
                ]
        token_num = _normalize(token_num, self.token_mean, self.token_std)
        token_num[~token_mask] = 0.0
        return (
            query_cat,
            query_num.astype(np.float32),
            token_cat,
            token_num.astype(np.float32),
            token_mask,
        )

    def _validation_mae(self, payload: tuple | None) -> float:
        if payload is None:
            return 0.0
        *arrays, y = payload
        preds = self._predict_arrays(tuple(arrays))
        return float(np.mean(np.abs(preds * self.target_std + self.target_mean - y)))

    def _predict_arrays(self, arrays: tuple[np.ndarray, ...]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not fitted")
        q_cat, q_num, t_cat, t_num, t_mask = arrays
        self.model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, len(q_num), self.config.batch_size):
                batch = [
                    torch.from_numpy(array[start : start + self.config.batch_size]).to(self.device)
                    for array in arrays
                ]
                batch[4] = batch[4].bool()
                preds.append(self.model(*batch).cpu().numpy())
        return np.concatenate(preds) if preds else np.array([], dtype=float)


def _all_tokens(data: pd.DataFrame) -> list[dict[str, Any]]:
    tokens: list[dict[str, Any]] = []
    for payload in data["obs_tokens_json"].tolist():
        tokens.extend(_parse_tokens(payload))
    return tokens or [{"modality": "none", "relationship": "none", "time_window": "none"}]


def _parse_tokens(payload: str) -> list[dict[str, Any]]:
    try:
        tokens = json.loads(payload)
    except json.JSONDecodeError:
        return []
    return tokens if isinstance(tokens, list) else []


def _numeric_matrix(data: pd.DataFrame, features: list[str]) -> np.ndarray:
    if not features:
        return np.zeros((len(data), 0), dtype=np.float32)
    return data[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)


def _fit_normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if values.ndim == 3:
        flat = values.reshape(-1, values.shape[-1])
    else:
        flat = values
    if flat.shape[1] == 0:
        return np.zeros(1, dtype=np.float32), np.ones(1, dtype=np.float32)
    valid = np.isfinite(flat)
    counts = valid.sum(axis=0)
    sums = np.where(valid, flat, 0.0).sum(axis=0)
    mean = np.divide(sums, counts, out=np.zeros(flat.shape[1], dtype=np.float32), where=counts > 0).astype(np.float32)
    centered = np.where(valid, flat - mean, 0.0)
    var = np.divide((centered**2).sum(axis=0), counts, out=np.ones(flat.shape[1], dtype=np.float32), where=counts > 0)
    std = np.sqrt(var).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _normalize(values: np.ndarray, mean: np.ndarray | None, std: np.ndarray | None) -> np.ndarray:
    if mean is None or std is None:
        raise ValueError("Normalizer is not fitted")
    values = np.where(np.isfinite(values), values, mean)
    return (values - mean) / std


def _obs_loader(*arrays, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    tensors = []
    for array in arrays:
        if array.dtype == np.int64:
            tensors.append(torch.from_numpy(array.astype(np.int64)))
        elif array.dtype == bool:
            tensors.append(torch.from_numpy(array.astype(bool)))
        else:
            tensors.append(torch.from_numpy(array.astype(np.float32)))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )
