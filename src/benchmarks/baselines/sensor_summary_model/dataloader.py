"""Independent dataloader for the Sensor-summary model safe sensor-history baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmarks.baselines._history import (
    DEFAULT_HISTORY_WINDOWS,
    build_input_history_features,
    history_feature_columns,
)
from src.benchmarks.baselines._payload_numeric import (
    SENSOR_NUMERIC_MODALITIES,
    load_payload_numeric_observations,
)
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

SENSOR_SUMMARY_TARGET_COLUMNS = (
    "target_uid",
    "plot_uid",
    "target_date",
    "target_name",
    "target_value_num",
    "source_asset_uid",
    "trial_year",
)

SENSOR_SUMMARY_INPUT_COLUMNS = (
    "target_uid",
    "asset_uid",
    "modality_verified",
    "acquisition_date",
)

SENSOR_SUMMARY_MODALITY_PATTERN = r"greenseeker|uav_multispectral"

SENSOR_SUMMARY_REQUIRED_COLUMNS = (
    "target_uid",
    "plot_uid",
    "target_date",
    "target_name",
    "target_value_num",
    "source_asset_uid",
    "hist_has_history",
)


class SensorSummaryDataLoader:
    """
    Load leakage-controlled historical sensor features for Sensor-summary model.

    Sensor-summary model uses true payload-backed GreenSeeker and UAV-MS numeric observations
    from legal pre-target input assets and excludes the current target source
    asset.
    """

    def __init__(self, config: BaselineDataConfig | None = None) -> None:
        self.config = config or BaselineDataConfig()
        self.data_root = Path(self.config.data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")
        self._observations_by_task: dict[str, pd.DataFrame] = {}

    def load_task(self, task_name: str) -> BaselineTaskData:
        """Load Sensor-summary model train/validation/test frames for one task."""

        validate_task_name(task_name)
        targets_path = self.data_root / "tasks" / task_name / "targets.parquet"
        inputs_path = self.data_root / "tasks" / task_name / "inputs_index.parquet"
        if not targets_path.exists():
            raise FileNotFoundError(f"Missing targets file: {targets_path}")
        if not inputs_path.exists():
            raise FileNotFoundError(f"Missing inputs index file: {inputs_path}")

        targets = pd.read_parquet(targets_path, columns=list(SENSOR_SUMMARY_TARGET_COLUMNS))
        inputs = pd.read_parquet(inputs_path, columns=list(SENSOR_SUMMARY_INPUT_COLUMNS))
        targets = self._prepare_targets(task_name, targets)
        history = build_input_history_features(
            targets,
            inputs,
            self._load_observations(task_name, inputs),
            prefix="hist",
            windows=DEFAULT_HISTORY_WINDOWS,
            modality_pattern=SENSOR_SUMMARY_MODALITY_PATTERN,
        )
        targets = targets.merge(
            history, on="target_uid", how="left", validate="one_to_one"
        )
        feature_cols = history_feature_columns(targets, "hist")
        targets[feature_cols] = targets[feature_cols].fillna(0.0)
        rows_before_subset = len(targets)
        targets = targets.loc[targets["hist_has_history"] > 0].copy()

        task_split_dir = split_dir(self.config, task_name)
        split_frames = {
            split: read_split_frame(
                task_split_dir / f"{split}.csv",
                targets,
                required_columns=SENSOR_SUMMARY_REQUIRED_COLUMNS,
                drop_unknown_targets=True,
            )
            for split in SPLITS
        }
        report = validate_split_integrity(
            task_name,
            targets,
            split_frames,
            enforce_plot_exclusivity=self.config.enforce_plot_exclusivity,
        )
        report["history_audit"] = self._history_audit(
            targets, feature_cols, rows_before_subset
        )
        return BaselineTaskData(
            task_name=task_name, validation_report=report, **split_frames
        )

    def _prepare_targets(self, task_name: str, targets: pd.DataFrame) -> pd.DataFrame:
        missing = set(SENSOR_SUMMARY_TARGET_COLUMNS) - set(targets.columns)
        if missing:
            raise ValueError(f"{task_name} targets missing columns: {sorted(missing)}")
        if targets["target_uid"].duplicated().any():
            raise ValueError(f"{task_name} targets contain duplicate target_uid values")

        frame = add_calendar_features(targets)
        frame["target_name"] = normalize_categorical(frame["target_name"])
        frame["source_asset_uid"] = normalize_categorical(frame["source_asset_uid"])
        frame["target_value_num"] = pd.to_numeric(
            frame["target_value_num"], errors="raise"
        )
        if frame["target_value_num"].isna().any():
            raise ValueError(f"{task_name} contains missing target_value_num values")
        return frame

    def _load_observations(self, task_name: str, inputs: pd.DataFrame) -> pd.DataFrame:
        if task_name not in self._observations_by_task:
            sensor_inputs = inputs.loc[
                inputs["modality_verified"].isin(SENSOR_NUMERIC_MODALITIES),
                "asset_uid",
            ]
            self._observations_by_task[task_name] = load_payload_numeric_observations(
                self.data_root,
                asset_uids=sensor_inputs,
                modalities=SENSOR_NUMERIC_MODALITIES,
            )
        return self._observations_by_task[task_name]

    def _history_audit(
        self,
        targets: pd.DataFrame,
        feature_cols: list[str],
        rows_before_subset: int,
    ) -> dict:
        return {
            "feature_count": len(feature_cols),
            "rows_with_history": int((targets["hist_has_history"] > 0).sum()),
            "rows_after_required_modality_subset": int(len(targets)),
            "rows_before_required_modality_subset": int(rows_before_subset),
            "windows_days": list(DEFAULT_HISTORY_WINDOWS),
            "input_asset_policy": "history values are limited to legal pre-target inputs_index assets for each target",
            "required_modality_policy": "targets without at least one legal pre-target payload-backed sensor observation are excluded",
            "observation_source": "true payload CSV values from GreenSeeker and UAV-MS assets",
        }
