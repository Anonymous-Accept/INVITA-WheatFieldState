"""Dataloader for Observation-set Transformer available-observation token baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarks.baselines._shared import BaselineDataConfig, BaselineTaskData, normalize_categorical
from src.benchmarks.baselines.tabular_metadata_model import TabularMetadataDataLoader

OBSERVATION_SET_INPUT_COLUMNS = (
    "target_uid",
    "asset_uid",
    "modality_verified",
    "acquisition_date",
    "relationship",
)

OBSERVATION_SET_TARGET_COLUMNS = (
    "target_uid",
    "target_date",
    "source_asset_uid",
)

TIME_WINDOWS = (
    "no_date",
    "0_7",
    "8_14",
    "15_30",
    "31_60",
    "61_plus",
)


class ObservationSetDataLoader:
    """Load full-task metadata query rows with available-observation tokens."""

    def __init__(self, config: BaselineDataConfig | None = None) -> None:
        self.config = config or BaselineDataConfig()
        self.data_root = Path(self.config.data_root)
        self._tabular_metadata_loader = TabularMetadataDataLoader(config)

    def load_task(self, task_name: str) -> BaselineTaskData:
        """Load Observation-set Transformer train/validation/test frames for one task."""

        base = self._tabular_metadata_loader.load_task(task_name)
        tokens = self._build_tokens(task_name)
        return BaselineTaskData(
            task_name=task_name,
            train=self._attach_tokens(base.train, tokens),
            val=self._attach_tokens(base.val, tokens),
            test=self._attach_tokens(base.test, tokens),
            validation_report={
                **base.validation_report,
                "observation_set_token_policy": {
                    "tokenization": "inputs_index rows grouped by modality, relationship, and pre-target time window",
                    "time_windows": list(TIME_WINDOWS),
                    "post_target_policy": "dated observations on or after target_date are excluded",
                    "same_source_policy": "input asset matching source_asset_uid is excluded",
                    "payload_policy": "raw payload values are not loaded",
                },
            },
        )

    def _build_tokens(self, task_name: str) -> pd.DataFrame:
        task_dir = self.data_root / "tasks" / task_name
        inputs = pd.read_parquet(task_dir / "inputs_index.parquet", columns=list(OBSERVATION_SET_INPUT_COLUMNS))
        targets = pd.read_parquet(task_dir / "targets.parquet", columns=list(OBSERVATION_SET_TARGET_COLUMNS))
        inputs = inputs.copy()
        targets = targets.copy()
        inputs["asset_uid"] = normalize_categorical(inputs["asset_uid"])
        inputs["modality_verified"] = normalize_categorical(inputs["modality_verified"])
        inputs["relationship"] = normalize_categorical(inputs["relationship"])
        inputs["acquisition_date"] = pd.to_datetime(inputs["acquisition_date"], errors="coerce")
        targets["source_asset_uid"] = normalize_categorical(targets["source_asset_uid"])
        targets["target_date"] = pd.to_datetime(targets["target_date"], errors="raise")

        merged = inputs.merge(targets, on="target_uid", how="inner", validate="many_to_one")
        merged = merged.loc[merged["asset_uid"].ne(merged["source_asset_uid"])].copy()
        dated = merged["acquisition_date"].notna()
        merged = merged.loc[(~dated) | merged["acquisition_date"].lt(merged["target_date"])].copy()
        merged["days_before_target"] = np.nan
        dated = merged["acquisition_date"].notna()
        merged.loc[dated, "days_before_target"] = (
            merged.loc[dated, "target_date"] - merged.loc[dated, "acquisition_date"]
        ).dt.days.astype(float)
        merged["time_window"] = merged["days_before_target"].map(_time_window)
        merged["has_dated_observation"] = merged["acquisition_date"].notna().astype(float)

        grouped = (
            merged.groupby(
                ["target_uid", "modality_verified", "relationship", "time_window"],
                dropna=False,
            )
            .agg(
                count=("asset_uid", "size"),
                unique_assets=("asset_uid", "nunique"),
                unique_dates=("acquisition_date", "nunique"),
                min_days_before=("days_before_target", "min"),
                mean_days_before=("days_before_target", "mean"),
                max_days_before=("days_before_target", "max"),
                has_dated_observation=("has_dated_observation", "max"),
            )
            .reset_index()
        )
        for col in ("min_days_before", "mean_days_before", "max_days_before"):
            grouped[col] = grouped[col].fillna(999.0)
        grouped["token"] = grouped.apply(_token_payload, axis=1)
        tokens = grouped.groupby("target_uid")["token"].apply(list).reset_index()
        tokens["obs_tokens_json"] = tokens["token"].map(json.dumps)
        tokens["obs_token_count"] = tokens["token"].map(len).astype(int)
        return tokens[["target_uid", "obs_tokens_json", "obs_token_count"]]

    def _attach_tokens(self, frame: pd.DataFrame, tokens: pd.DataFrame) -> pd.DataFrame:
        output = frame.merge(tokens, on="target_uid", how="left", validate="one_to_one")
        output["obs_tokens_json"] = output["obs_tokens_json"].fillna("[]")
        output["obs_token_count"] = output["obs_token_count"].fillna(0).astype(int)
        return output


def _time_window(days_before: float) -> str:
    if pd.isna(days_before):
        return "no_date"
    if days_before <= 7:
        return "0_7"
    if days_before <= 14:
        return "8_14"
    if days_before <= 30:
        return "15_30"
    if days_before <= 60:
        return "31_60"
    return "61_plus"


def _token_payload(row: pd.Series) -> dict:
    return {
        "modality": str(row["modality_verified"]),
        "relationship": str(row["relationship"]),
        "time_window": str(row["time_window"]),
        "count": float(row["count"]),
        "unique_assets": float(row["unique_assets"]),
        "unique_dates": float(row["unique_dates"]),
        "min_days_before": float(row["min_days_before"]),
        "mean_days_before": float(row["mean_days_before"]),
        "max_days_before": float(row["max_days_before"]),
        "has_dated_observation": float(row["has_dated_observation"]),
    }
