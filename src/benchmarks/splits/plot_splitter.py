"""Plot-group split generation and auditing for INVITA benchmark tasks."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

TASKS = (
    "NDVI",
    "LAI",
    "FCover",
    "Zadoks",
)
SPLITS = ("train", "val", "test")
DEFAULT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for deterministic plot-level split generation."""

    data_root: Path
    output_root: Path
    ratios: dict[str, float]
    seed: int = 42

    def __post_init__(self) -> None:
        if set(self.ratios) != set(SPLITS):
            raise ValueError(f"ratios must contain exactly {SPLITS}")
        total = sum(self.ratios.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"ratios must sum to 1.0, got {total}")


class PlotSplitGenerator:
    """
    Generate deterministic 7:1:2 splits with plot-level exclusivity.

    The optimizer assigns whole `plot_uid` groups to splits. It tries to match
    row ratios and target-name distributions. Cultivar/genotype distribution is
    audited but not forced when rare genotypes make exact coverage impossible.
    """

    def __init__(self, config: SplitConfig) -> None:
        self.config = config
        self.data_root = Path(config.data_root)
        self.output_root = Path(config.output_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")

    def generate_all(self, tasks: tuple[str, ...] = TASKS) -> dict[str, dict]:
        """Generate splits for several tasks and return audit reports."""

        reports = {}
        for task_name in tasks:
            reports[task_name] = self.generate_task(task_name)
        manifest = {
            "split_name": self.output_root.name,
            "ratios": self.config.ratios,
            "seed": self.config.seed,
            "group_key": "plot_uid",
            "tasks": reports,
        }
        self.output_root.mkdir(parents=True, exist_ok=True)
        (self.output_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        return reports

    def generate_task(self, task_name: str) -> dict:
        """Generate plot-level splits for one task."""

        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}")

        targets = self._load_targets(task_name)
        groups = self._build_groups(targets)
        assignments = self._assign_groups(groups)
        split_frames = self._materialize_splits(targets, assignments)
        report = audit_split_frames(targets, split_frames)

        task_dir = self.output_root / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        for split_name, frame in split_frames.items():
            columns = [
                "target_uid",
                "plot_uid",
                "trial_year",
                "trial_code",
                "target_name",
                "cultivar_id",
            ]
            optional_columns = ["cultivar_name"]
            save_columns = columns + [c for c in optional_columns if c in frame.columns]
            frame[save_columns].to_csv(task_dir / f"{split_name}.csv", index=False)
        (task_dir / "audit.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        return report

    def _load_targets(self, task_name: str) -> pd.DataFrame:
        targets = pd.read_parquet(
            self.data_root / "tasks" / task_name / "targets.parquet",
            columns=[
                "target_uid",
                "plot_uid",
                "target_name",
                "trial_year",
                "trial_code",
                "state",
                "region_name",
            ],
        )
        plots = pd.read_parquet(
            self.data_root / "shared" / "plots.parquet",
            columns=["plot_uid", "cultivar_id", "cultivar_name"],
        )
        merged = targets.merge(plots, on="plot_uid", how="left", validate="many_to_one")
        merged["cultivar_id"] = merged["cultivar_id"].fillna("unknown").astype(str)
        merged["cultivar_name"] = merged["cultivar_name"].fillna("unknown").astype(str)
        merged.loc[merged["cultivar_name"].str.len() == 0, "cultivar_name"] = "unknown"
        return merged

    def _build_groups(self, targets: pd.DataFrame) -> pd.DataFrame:
        target_names = sorted(targets["target_name"].dropna().unique())
        target_counts = (
            targets.pivot_table(
                index="plot_uid",
                columns="target_name",
                values="target_uid",
                aggfunc="count",
                fill_value=0,
            )
            .reindex(columns=target_names, fill_value=0)
            .reset_index()
        )
        meta = (
            targets.groupby("plot_uid", as_index=False)
            .agg(
                n_rows=("target_uid", "size"),
                trial_year=("trial_year", "first"),
                trial_code=("trial_code", "first"),
                state=("state", "first"),
                region_name=("region_name", "first"),
                cultivar_id=("cultivar_id", "first"),
                cultivar_name=("cultivar_name", "first"),
            )
            .merge(target_counts, on="plot_uid", how="left", validate="one_to_one")
        )
        meta["target_combo"] = meta[target_names].gt(0).apply(
            lambda row: ",".join([name for name, present in row.items() if present]), axis=1
        )
        return meta

    def _assign_groups(self, groups: pd.DataFrame) -> dict[str, str]:
        rng = random.Random(self.config.seed)
        assignments: dict[str, str] = {}

        for _, stratum in groups.groupby("target_combo", sort=True):
            ordered = stratum.copy()
            ordered["_rand"] = [rng.random() for _ in range(len(ordered))]
            ordered = ordered.sort_values(["n_rows", "_rand"], ascending=[False, True])

            total_rows = int(ordered["n_rows"].sum())
            desired_rows = {
                split: total_rows * self.config.ratios[split] for split in SPLITS
            }
            current_rows = dict.fromkeys(SPLITS, 0)

            for _, group in ordered.iterrows():
                best_split = min(
                    SPLITS,
                    key=lambda split: self._row_balance_score(
                        split, group, current_rows, desired_rows
                    ),
                )
                assignments[str(group["plot_uid"])] = best_split
                current_rows[best_split] += int(group["n_rows"])

        return assignments

    def _row_balance_score(
        self,
        candidate_split: str,
        group: pd.Series,
        current_rows: dict[str, int],
        desired_rows: dict[str, float],
    ) -> float:
        score = 0.0
        for split in SPLITS:
            rows = current_rows[split]
            if split == candidate_split:
                rows += int(group["n_rows"])
            scale = max(desired_rows[split], 1.0)
            error = (rows - desired_rows[split]) / scale
            score += error * error
        return score

    def _materialize_splits(
        self, targets: pd.DataFrame, assignments: dict[str, str]
    ) -> dict[str, pd.DataFrame]:
        frame = targets.copy()
        frame["split"] = frame["plot_uid"].map(assignments)
        if frame["split"].isna().any():
            raise ValueError("Some targets did not receive a split assignment")
        return {
            split: frame[frame["split"] == split].drop(columns=["split"]).copy()
            for split in SPLITS
        }


def audit_split_frames(
    targets: pd.DataFrame, split_frames: dict[str, pd.DataFrame]
) -> dict:
    """Audit split ratios, plot leakage, and cultivar coverage."""

    total_rows = sum(len(frame) for frame in split_frames.values())
    target_ids = set(targets["target_uid"])
    split_ids = {
        split: set(frame["target_uid"]) for split, frame in split_frames.items()
    }
    split_plots = {split: set(frame["plot_uid"]) for split, frame in split_frames.items()}
    split_groups = {
        split: set(zip(frame["trial_year"], frame["trial_code"], strict=False))
        for split, frame in split_frames.items()
    }
    split_cultivars = {
        split: set(frame["cultivar_id"].dropna().astype(str))
        for split, frame in split_frames.items()
    }

    pair_reports = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        pair_reports[f"{left}_{right}"] = {
            "target_uid_overlap": len(split_ids[left] & split_ids[right]),
            "plot_uid_overlap": len(split_plots[left] & split_plots[right]),
            "trial_group_overlap": len(split_groups[left] & split_groups[right]),
        }

    target_name_counts = {
        split: frame["target_name"].value_counts().sort_index().to_dict()
        for split, frame in split_frames.items()
    }
    cultivar_counts = {
        split: int(frame["cultivar_id"].nunique(dropna=True))
        for split, frame in split_frames.items()
    }

    return {
        "rows": {split: int(len(frame)) for split, frame in split_frames.items()},
        "ratios": {
            split: float(len(frame) / total_rows) for split, frame in split_frames.items()
        },
        "n_targets": int(len(targets)),
        "coverage_complete": set().union(*split_ids.values()) == target_ids,
        "pairwise_overlap": pair_reports,
        "target_name_counts": target_name_counts,
        "plot_counts": {
            split: int(frame["plot_uid"].nunique()) for split, frame in split_frames.items()
        },
        "trial_group_counts": {
            split: int(frame[["trial_year", "trial_code"]].drop_duplicates().shape[0])
            for split, frame in split_frames.items()
        },
        "cultivar_counts": cultivar_counts,
        "cultivars_missing_from_train": {
            "val": len(split_cultivars["val"] - split_cultivars["train"]),
            "test": len(split_cultivars["test"] - split_cultivars["train"]),
        },
        "cultivars_in_all_splits": len(
            split_cultivars["train"]
            & split_cultivars["val"]
            & split_cultivars["test"]
        ),
    }
