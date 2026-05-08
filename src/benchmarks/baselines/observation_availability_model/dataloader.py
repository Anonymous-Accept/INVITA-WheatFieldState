"""Independent dataloader for the Observation-availability model."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmarks.baselines._shared import (
    SPLITS,
    BaselineDataConfig,
    BaselineTaskData,
    add_calendar_features,
    normalize_categorical,
    read_split_frame,
    split_dir,
    validate_split_integrity,
    validate_task_name,
)

OBSERVATION_AVAILABILITY_TARGET_COLUMNS = (
    "target_uid",
    "plot_uid",
    "target_date",
    "target_name",
    "target_value_num",
    "source_asset_uid",
)

OBSERVATION_AVAILABILITY_INPUT_COLUMNS = (
    "target_uid",
    "asset_uid",
    "modality_verified",
    "acquisition_date",
    "temporal_offset_days",
    "relationship",
    "leakage_filter_notes",
)

STATIC_MODALITIES = frozenset({"tabular", "weather_station_distances"})

DYNAMIC_MODALITIES = (
    "greenseeker_handheld",
    "uav_multispectral_l2_inversion",
    "field_camera",
    "satellite_scene",
    "uav_orthomosaic",
    "uav_orthomosaic_plot_clip",
    "weather_trial_silo",
    "weather_master_silo",
)

OBSERVATION_AVAILABILITY_REQUIRED_COLUMNS = (
    "target_uid",
    "plot_uid",
    "target_date",
    "target_name",
    "target_value_num",
    "n_legal_assets",
    "n_modalities_present",
)


class ObservationAvailabilityDataLoader:
    """
    Load Observation-availability model features for Observation-availability model.

    Observation-availability model reads target metadata and `inputs_index.parquet` only. It never reads
    payload values. Dynamic inputs are counted only when their acquisition date
    is strictly before the target date.
    """

    def __init__(self, config: BaselineDataConfig | None = None) -> None:
        self.config = config or BaselineDataConfig()
        self.data_root = Path(self.config.data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

    def load_task(self, task_name: str) -> BaselineTaskData:
        """Load Observation-availability model train/validation/test frames for one task."""

        validate_task_name(task_name)
        task_dir = self.data_root / "tasks" / task_name
        targets_path = task_dir / "targets.parquet"
        inputs_path = task_dir / "inputs_index.parquet"
        if not targets_path.exists():
            raise FileNotFoundError(f"Missing targets file: {targets_path}")
        if not inputs_path.exists():
            raise FileNotFoundError(f"Missing inputs index file: {inputs_path}")

        targets = pd.read_parquet(targets_path, columns=list(OBSERVATION_AVAILABILITY_TARGET_COLUMNS))
        inputs = pd.read_parquet(inputs_path, columns=list(OBSERVATION_AVAILABILITY_INPUT_COLUMNS))
        targets = self._prepare_targets(task_name, targets, inputs)

        task_split_dir = split_dir(self.config, task_name)
        split_frames = {
            split: read_split_frame(
                task_split_dir / f"{split}.csv",
                targets,
                required_columns=OBSERVATION_AVAILABILITY_REQUIRED_COLUMNS,
            )
            for split in SPLITS
        }
        report = validate_split_integrity(
            task_name,
            targets,
            split_frames,
            enforce_plot_exclusivity=self.config.enforce_plot_exclusivity,
        )
        report["availability_audit"] = self._availability_audit(targets)
        return BaselineTaskData(task_name=task_name, validation_report=report, **split_frames)

    def _prepare_targets(
        self, task_name: str, targets: pd.DataFrame, inputs: pd.DataFrame
    ) -> pd.DataFrame:
        missing = set(OBSERVATION_AVAILABILITY_TARGET_COLUMNS) - set(targets.columns)
        if missing:
            raise ValueError(f"{task_name} targets missing columns: {sorted(missing)}")
        if targets["target_uid"].duplicated().any():
            raise ValueError(f"{task_name} targets contain duplicate target_uid values")

        frame = add_calendar_features(targets)
        frame["target_name"] = normalize_categorical(frame["target_name"])
        frame["target_value_num"] = pd.to_numeric(
            frame["target_value_num"], errors="raise"
        )
        if frame["target_value_num"].isna().any():
            raise ValueError(f"{task_name} contains missing target_value_num values")

        features = self._build_availability_features(frame, inputs)
        frame = frame.merge(features, on="target_uid", how="left", validate="one_to_one")
        feature_cols = [col for col in features.columns if col != "target_uid"]
        frame[feature_cols] = frame[feature_cols].fillna(0.0)
        return frame

    def _build_availability_features(
        self, targets: pd.DataFrame, inputs: pd.DataFrame
    ) -> pd.DataFrame:
        input_frame = inputs.merge(
            targets[["target_uid", "target_date", "source_asset_uid"]],
            on="target_uid",
            how="inner",
            validate="many_to_one",
        )
        input_frame["modality_verified"] = normalize_categorical(
            input_frame["modality_verified"]
        )
        input_frame["acquisition_date"] = pd.to_datetime(
            input_frame["acquisition_date"], errors="coerce"
        )
        input_frame["target_date"] = pd.to_datetime(
            input_frame["target_date"], errors="raise"
        )
        input_frame["asset_uid"] = normalize_categorical(input_frame["asset_uid"])
        input_frame["source_asset_uid"] = normalize_categorical(
            input_frame["source_asset_uid"]
        )

        same_source = input_frame["asset_uid"].eq(input_frame["source_asset_uid"])
        is_static = input_frame["modality_verified"].isin(STATIC_MODALITIES)
        is_dynamic_legal = (
            input_frame["acquisition_date"].notna()
            & input_frame["acquisition_date"].lt(input_frame["target_date"])
        )
        legal = input_frame.loc[~same_source & (is_static | is_dynamic_legal)].copy()

        output = pd.DataFrame({"target_uid": targets["target_uid"]})
        modality_counts = (
            legal.groupby(["target_uid", "modality_verified"], observed=True)
            .size()
            .unstack(fill_value=0)
        )
        modality_counts = modality_counts.rename(
            columns={modality: f"n_{_safe_name(modality)}" for modality in modality_counts.columns}
        )
        output = output.merge(
            modality_counts.reset_index(), on="target_uid", how="left"
        )

        count_cols = [col for col in output.columns if col.startswith("n_")]
        output[count_cols] = output[count_cols].fillna(0.0)
        for modality in (*STATIC_MODALITIES, *DYNAMIC_MODALITIES):
            count_col = f"n_{_safe_name(modality)}"
            if count_col not in output.columns:
                output[count_col] = 0.0
            output[f"has_{_safe_name(modality)}"] = (output[count_col] > 0).astype(float)

        count_cols = [col for col in output.columns if col.startswith("n_")]
        output["n_legal_assets"] = output[count_cols].sum(axis=1)
        output["n_modalities_present"] = (
            output[[col for col in output.columns if col.startswith("has_")]].sum(axis=1)
        )

        temporal = legal.loc[legal["acquisition_date"].notna()].copy()
        output = output.merge(
            self._days_since_features(targets, temporal), on="target_uid", how="left"
        )
        days_cols = [col for col in output.columns if col.startswith("days_since_last_")]
        output[days_cols] = output[days_cols].fillna(999.0)
        return output

    def _days_since_features(
        self, targets: pd.DataFrame, temporal: pd.DataFrame
    ) -> pd.DataFrame:
        target_dates = targets[["target_uid", "target_date"]].copy()
        target_dates["target_date"] = pd.to_datetime(
            target_dates["target_date"], errors="raise"
        )
        output = pd.DataFrame({"target_uid": targets["target_uid"]})

        if temporal.empty:
            output["days_since_last_any"] = np.nan
            for modality in DYNAMIC_MODALITIES:
                output[f"days_since_last_{_safe_name(modality)}"] = np.nan
            return output

        last_any = (
            temporal.groupby("target_uid", observed=True)["acquisition_date"]
            .max()
            .rename("last_any_date")
            .reset_index()
        )
        output = output.merge(last_any, on="target_uid", how="left")

        last_by_modality = (
            temporal.groupby(["target_uid", "modality_verified"], observed=True)[
                "acquisition_date"
            ]
            .max()
            .reset_index()
        )
        last_by_modality["modality_verified"] = last_by_modality[
            "modality_verified"
        ].map(_safe_name)
        pivot = last_by_modality.pivot(
            index="target_uid",
            columns="modality_verified",
            values="acquisition_date",
        )
        pivot = pivot.rename(
            columns={col: f"last_{col}_date" for col in pivot.columns}
        ).reset_index()
        output = output.merge(pivot, on="target_uid", how="left")
        output = output.merge(target_dates, on="target_uid", how="left")

        output["days_since_last_any"] = (
            output["target_date"] - output["last_any_date"]
        ).dt.days
        for modality in DYNAMIC_MODALITIES:
            safe = _safe_name(modality)
            date_col = f"last_{safe}_date"
            out_col = f"days_since_last_{safe}"
            if date_col in output.columns:
                output[out_col] = (output["target_date"] - output[date_col]).dt.days
            else:
                output[out_col] = np.nan

        keep_cols = ["target_uid", "days_since_last_any"] + [
            f"days_since_last_{_safe_name(modality)}" for modality in DYNAMIC_MODALITIES
        ]
        return output[keep_cols]

    def _availability_audit(self, targets: pd.DataFrame) -> dict:
        count_cols = [col for col in targets.columns if col.startswith("n_")]
        has_cols = [col for col in targets.columns if col.startswith("has_")]
        return {
            "feature_columns": sorted(count_cols + has_cols),
            "rows_with_any_legal_asset": int((targets["n_legal_assets"] > 0).sum()),
            "rows_total": int(len(targets)),
        }


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
