"""Independent dataloader for the Source-date prior climatology experiment."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.benchmarks.paths import default_data_root

TASKS = (
    "NDVI",
    "LAI",
    "FCover",
    "Zadoks",
)

SPLITS = ("train", "val", "test")

SOURCE_DATE_PRIOR_COLUMNS = (
    "target_uid",
    "plot_uid",
    "target_date",
    "target_name",
    "target_value_num",
    "source_dataset",
    "instrument",
    "trial_code",
    "trial_year",
    "state",
    "region_name",
    "site_id",
    "crop_type",
)


@dataclass(frozen=True)
class SourceDatePriorDataConfig:
    """Filesystem configuration for Source-date prior data loading."""

    data_root: Path = field(default_factory=default_data_root)
    split_root: Path | None = None
    enforce_plot_exclusivity: bool = True
    enforce_trial_group_exclusivity: bool = False


@dataclass
class SourceDatePriorTaskData:
    """Official Source-date prior data splits for one task."""

    task_name: str
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    validation_report: dict

    def split(self, name: str) -> pd.DataFrame:
        if name not in SPLITS:
            raise ValueError(f"Unknown split: {name}")
        return getattr(self, name)


class SourceDatePriorDataLoader:
    """
    Load only the data allowed for Source-date prior.

    Source-date prior deliberately ignores payload assets and weather files. It uses target
    provenance, site/crop metadata, season, and calendar position from
    `targets.parquet`, joined to official deterministic split files.
    """

    def __init__(self, config: SourceDatePriorDataConfig | None = None) -> None:
        self.config = config or SourceDatePriorDataConfig()
        self.data_root = Path(self.config.data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

    def load_task(self, task_name: str) -> SourceDatePriorTaskData:
        """Load and validate official splits for a task."""

        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}")

        task_dir = self.data_root / "tasks" / task_name
        split_dir = self._split_dir(task_name)
        targets_path = task_dir / "targets.parquet"
        if not targets_path.exists():
            raise FileNotFoundError(f"Missing targets file: {targets_path}")

        targets = pd.read_parquet(targets_path, columns=list(SOURCE_DATE_PRIOR_COLUMNS))
        self._validate_targets(task_name, targets)

        split_frames = {
            split: self._load_split(split_dir, split, targets) for split in SPLITS
        }
        report = self._validate_splits(task_name, targets, split_frames)
        return SourceDatePriorTaskData(task_name=task_name, validation_report=report, **split_frames)

    def load_many(self, task_names: Iterable[str]) -> dict[str, SourceDatePriorTaskData]:
        """Load several tasks."""

        return {task_name: self.load_task(task_name) for task_name in task_names}

    def _split_dir(self, task_name: str) -> Path:
        if self.config.split_root is not None:
            return Path(self.config.split_root) / task_name
        return self.data_root / "tasks" / task_name / "splits"

    def _load_split(
        self, split_dir: Path, split: str, targets: pd.DataFrame
    ) -> pd.DataFrame:
        split_path = split_dir / f"{split}.csv"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        split_df = pd.read_csv(split_path, usecols=["target_uid"])
        if split_df["target_uid"].duplicated().any():
            duplicates = split_df.loc[
                split_df["target_uid"].duplicated(), "target_uid"
            ].head(5)
            raise ValueError(f"Duplicate target_uid values in {split_path}: {duplicates}")

        frame = split_df.merge(targets, on="target_uid", how="left", validate="one_to_one")
        if frame["plot_uid"].isna().any():
            missing = frame.loc[frame["plot_uid"].isna(), "target_uid"].head(5).tolist()
            raise ValueError(f"Split references unknown target_uid values: {missing}")

        frame["target_date"] = pd.to_datetime(frame["target_date"], errors="raise")
        frame["target_value_num"] = pd.to_numeric(
            frame["target_value_num"], errors="raise"
        )
        return frame

    def _validate_targets(self, task_name: str, targets: pd.DataFrame) -> None:
        missing_columns = set(SOURCE_DATE_PRIOR_COLUMNS) - set(targets.columns)
        if missing_columns:
            raise ValueError(
                f"{task_name} targets missing required columns: {sorted(missing_columns)}"
            )
        if targets["target_uid"].duplicated().any():
            raise ValueError(f"{task_name} targets contain duplicate target_uid values")
        if targets["target_value_num"].isna().any():
            raise ValueError(f"{task_name} contains missing target_value_num values")
        pd.to_datetime(targets["target_date"], errors="raise")

    def _validate_splits(
        self,
        task_name: str,
        targets: pd.DataFrame,
        split_frames: dict[str, pd.DataFrame],
    ) -> dict:
        split_ids = {
            split: set(frame["target_uid"].tolist())
            for split, frame in split_frames.items()
        }
        target_ids = set(targets["target_uid"].tolist())
        union_ids = set().union(*split_ids.values())

        missing_from_splits = target_ids - union_ids
        unknown_in_splits = union_ids - target_ids
        if missing_from_splits:
            raise ValueError(
                f"{task_name} has targets missing from official splits: "
                f"{sorted(missing_from_splits)[:5]}"
            )
        if unknown_in_splits:
            raise ValueError(
                f"{task_name} splits include unknown targets: "
                f"{sorted(unknown_in_splits)[:5]}"
            )

        uid_overlaps = {}
        plot_overlaps = {}
        group_overlaps = {}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            uid_overlap = split_ids[left] & split_ids[right]
            if uid_overlap:
                raise ValueError(
                    f"{task_name} split target leakage {left}-{right}: "
                    f"{sorted(uid_overlap)[:5]}"
                )
            uid_overlaps[f"{left}_{right}"] = 0

            left_plots = set(split_frames[left]["plot_uid"].astype(str))
            right_plots = set(split_frames[right]["plot_uid"].astype(str))
            plot_overlap = left_plots & right_plots
            if plot_overlap and self.config.enforce_plot_exclusivity:
                raise ValueError(
                    f"{task_name} split plot leakage {left}-{right}: "
                    f"{sorted(plot_overlap)[:5]}"
                )
            plot_overlaps[f"{left}_{right}"] = len(plot_overlap)

            left_groups = set(
                zip(
                    split_frames[left]["trial_year"].astype(int),
                    split_frames[left]["trial_code"].astype(str), strict=False,
                )
            )
            right_groups = set(
                zip(
                    split_frames[right]["trial_year"].astype(int),
                    split_frames[right]["trial_code"].astype(str), strict=False,
                )
            )
            group_overlap = left_groups & right_groups
            if group_overlap and self.config.enforce_trial_group_exclusivity:
                raise ValueError(
                    f"{task_name} split group leakage {left}-{right}: "
                    f"{sorted(group_overlap)[:5]}"
                )
            group_overlaps[f"{left}_{right}"] = len(group_overlap)

        return {
            "task_name": task_name,
            "n_targets": int(len(targets)),
            "split_rows": {
                split: int(len(frame)) for split, frame in split_frames.items()
            },
            "split_trial_groups": {
                split: int(
                    frame[["trial_year", "trial_code"]].drop_duplicates().shape[0]
                )
                for split, frame in split_frames.items()
            },
            "target_uid_overlap": uid_overlaps,
            "plot_uid_overlap": plot_overlaps,
            "trial_group_overlap": group_overlaps,
        }


def write_validation_report(report: dict, path: Path) -> None:
    """Write a split validation report as JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
