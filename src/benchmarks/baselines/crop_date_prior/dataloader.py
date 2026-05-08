"""Independent dataloader for the Crop-date prior baseline."""

from __future__ import annotations

from pathlib import Path

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

CROP_DATE_PRIOR_TARGET_COLUMNS = (
    "target_uid",
    "plot_uid",
    "target_date",
    "target_name",
    "target_value_num",
    "trial_year",
    "trial_code",
    "state",
    "region_name",
    "site_id",
    "crop_type",
    "sowing_date",
)

CROP_DATE_PRIOR_PLOT_COLUMNS = (
    "plot_uid",
    "cultivar_id",
    "cultivar_name",
    "area_m2",
    "sowing_date",
    "state",
    "region_name",
    "site_id",
    "crop_type",
)

CROP_DATE_PRIOR_REQUIRED_COLUMNS = (
    "target_uid",
    "plot_uid",
    "target_date",
    "target_name",
    "target_value_num",
    "trial_year",
    "crop_type",
    "cultivar_key",
    "target_doy_bin_14",
)


class CropDatePriorDataLoader:
    """
    Load the data allowed for Crop-date prior.

    Crop-date prior uses target metadata and plot-level agronomic hierarchy fields. It does
    not read payload assets, weather records, sensor values, raster data, or RGB
    images.
    """

    def __init__(self, config: BaselineDataConfig | None = None) -> None:
        self.config = config or BaselineDataConfig()
        self.data_root = Path(self.config.data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

    def load_task(self, task_name: str) -> BaselineTaskData:
        """Load Crop-date prior train/validation/test frames for one task."""

        validate_task_name(task_name)
        task_dir = self.data_root / "tasks" / task_name
        targets_path = task_dir / "targets.parquet"
        plots_path = self.data_root / "shared" / "plots.parquet"
        if not targets_path.exists():
            raise FileNotFoundError(f"Missing targets file: {targets_path}")
        if not plots_path.exists():
            raise FileNotFoundError(f"Missing plots file: {plots_path}")

        targets = pd.read_parquet(targets_path, columns=list(CROP_DATE_PRIOR_TARGET_COLUMNS))
        plots = pd.read_parquet(plots_path, columns=list(CROP_DATE_PRIOR_PLOT_COLUMNS))
        targets = self._prepare_targets(task_name, targets, plots)

        task_split_dir = split_dir(self.config, task_name)
        split_frames = {
            split: read_split_frame(
                task_split_dir / f"{split}.csv",
                targets,
                required_columns=CROP_DATE_PRIOR_REQUIRED_COLUMNS,
            )
            for split in SPLITS
        }
        report = validate_split_integrity(
            task_name,
            targets,
            split_frames,
            enforce_plot_exclusivity=self.config.enforce_plot_exclusivity,
        )
        return BaselineTaskData(task_name=task_name, validation_report=report, **split_frames)

    def _prepare_targets(
        self, task_name: str, targets: pd.DataFrame, plots: pd.DataFrame
    ) -> pd.DataFrame:
        missing = set(CROP_DATE_PRIOR_TARGET_COLUMNS) - set(targets.columns)
        if missing:
            raise ValueError(f"{task_name} targets missing columns: {sorted(missing)}")
        if targets["target_uid"].duplicated().any():
            raise ValueError(f"{task_name} targets contain duplicate target_uid values")

        frame = targets.merge(plots, on="plot_uid", how="left", suffixes=("", "_plot"))

        for col in ("state", "region_name", "site_id", "crop_type", "sowing_date"):
            plot_col = f"{col}_plot"
            if plot_col in frame.columns:
                frame[col] = frame[col].combine_first(frame[plot_col])
                frame = frame.drop(columns=[plot_col])

        frame = add_calendar_features(frame)
        frame["target_value_num"] = pd.to_numeric(
            frame["target_value_num"], errors="raise"
        )
        if frame["target_value_num"].isna().any():
            raise ValueError(f"{task_name} contains missing target_value_num values")

        for col in (
            "target_name",
            "trial_code",
            "state",
            "region_name",
            "site_id",
            "crop_type",
            "cultivar_id",
            "cultivar_name",
        ):
            frame[col] = normalize_categorical(frame[col])

        frame["cultivar_key"] = frame["cultivar_name"]
        missing_cultivar = frame["cultivar_key"].eq("unknown")
        frame.loc[missing_cultivar, "cultivar_key"] = frame.loc[
            missing_cultivar, "cultivar_id"
        ]
        frame.loc[frame["cultivar_key"].eq("unknown"), "cultivar_key"] = (
            "unknown_cultivar"
        )

        frame["trial_year"] = pd.to_numeric(frame["trial_year"], errors="raise").astype(
            int
        )
        frame["area_m2"] = pd.to_numeric(frame.get("area_m2"), errors="coerce")
        return frame
