"""Tests for the Source-date prior baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.benchmarks.baselines.source_date_prior.dataloader import (
    SourceDatePriorDataConfig,
    SourceDatePriorDataLoader,
)
from src.benchmarks.baselines.source_date_prior.predictor import (
    PREDICTION_COL,
    TARGET_COL,
    SourceDatePrior,
)


def _rows() -> list[dict]:
    return [
        {
            "target_uid": "t1",
            "plot_uid": "p1",
            "target_date": "2021-06-01",
            "target_name": "ndvi",
            TARGET_COL: 0.40,
            "source_dataset": "obs.csv",
            "instrument": "sensor_a",
            "trial_code": "TRIAL_A",
            "trial_year": 2021,
            "state": "WA",
            "region_name": "R1",
            "site_id": "S1",
            "crop_type": "Wheat",
        },
        {
            "target_uid": "t2",
            "plot_uid": "p2",
            "target_date": "2021-06-02",
            "target_name": "ndvi",
            TARGET_COL: 0.50,
            "source_dataset": "obs.csv",
            "instrument": "sensor_a",
            "trial_code": "TRIAL_A",
            "trial_year": 2021,
            "state": "WA",
            "region_name": "R1",
            "site_id": "S1",
            "crop_type": "Wheat",
        },
        {
            "target_uid": "t3",
            "plot_uid": "p3",
            "target_date": "2021-06-03",
            "target_name": "ndvi",
            TARGET_COL: 0.60,
            "source_dataset": "obs.csv",
            "instrument": "sensor_a",
            "trial_code": "TRIAL_B",
            "trial_year": 2021,
            "state": "WA",
            "region_name": "R1",
            "site_id": "S2",
            "crop_type": "Wheat",
        },
        {
            "target_uid": "t4",
            "plot_uid": "p4",
            "target_date": "2021-06-04",
            "target_name": "ndvi",
            TARGET_COL: 0.70,
            "source_dataset": "obs.csv",
            "instrument": "sensor_a",
            "trial_code": "TRIAL_C",
            "trial_year": 2021,
            "state": "WA",
            "region_name": "R1",
            "site_id": "S3",
            "crop_type": "Wheat",
        },
    ]


def test_predictor_uses_fallback_ladder() -> None:
    train = pd.DataFrame(_rows())
    test = train.iloc[[0]].copy()
    test.loc[:, "target_uid"] = "new"
    test.loc[:, "site_id"] = "unseen_site"

    model = SourceDatePrior()
    model.fit(train)
    predictions = model.predict_frame(test)

    assert predictions[PREDICTION_COL].notna().all()
    assert predictions.loc[0, "fallback_level"] != "global"
    assert np.isfinite(predictions.loc[0, PREDICTION_COL])


def test_predictor_reports_metrics() -> None:
    frame = pd.DataFrame(_rows())
    model = SourceDatePrior()
    model.fit(frame.iloc[:3])

    metrics = model.evaluate(frame.iloc[3:])

    assert metrics["n"] == 1
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_loader_uses_official_split_files(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks" / "NDVI"
    (task_dir / "splits").mkdir(parents=True)
    targets = pd.DataFrame(_rows())
    targets.to_parquet(task_dir / "targets.parquet", index=False)

    targets.iloc[[0, 1]][["target_uid", "plot_uid", "trial_year", "trial_code"]].to_csv(
        task_dir / "splits" / "train.csv", index=False
    )
    targets.iloc[[2]][["target_uid", "plot_uid", "trial_year", "trial_code"]].to_csv(
        task_dir / "splits" / "val.csv", index=False
    )
    targets.iloc[[3]][["target_uid", "plot_uid", "trial_year", "trial_code"]].to_csv(
        task_dir / "splits" / "test.csv", index=False
    )

    data = SourceDatePriorDataLoader(SourceDatePriorDataConfig(data_root=tmp_path))
    loaded = data.load_task("NDVI")

    assert len(loaded.train) == 2
    assert len(loaded.val) == 1
    assert len(loaded.test) == 1
    assert loaded.validation_report["target_uid_overlap"]["train_test"] == 0


def test_loader_rejects_group_leakage(tmp_path: Path) -> None:
    task_dir = tmp_path / "tasks" / "NDVI"
    (task_dir / "splits").mkdir(parents=True)
    targets = pd.DataFrame(_rows())
    targets.to_parquet(task_dir / "targets.parquet", index=False)

    targets.iloc[[0, 1]][["target_uid", "plot_uid", "trial_year", "trial_code"]].to_csv(
        task_dir / "splits" / "train.csv", index=False
    )
    targets.iloc[[2]][["target_uid", "plot_uid", "trial_year", "trial_code"]].to_csv(
        task_dir / "splits" / "val.csv", index=False
    )
    leaked = targets.iloc[[1]].copy()
    leaked.loc[:, "target_uid"] = "leaked_new_uid"
    leaked.loc[:, "plot_uid"] = "p_leaked"
    pd.concat([targets.iloc[[3]], leaked], ignore_index=True)[["target_uid", "plot_uid", "trial_year", "trial_code"]].to_csv(
        task_dir / "splits" / "test.csv", index=False
    )
    targets = pd.concat([targets, leaked], ignore_index=True)
    targets.to_parquet(task_dir / "targets.parquet", index=False)

    loader = SourceDatePriorDataLoader(
        SourceDatePriorDataConfig(data_root=tmp_path, enforce_trial_group_exclusivity=True)
    )
    with pytest.raises(ValueError, match="split group leakage"):
        loader.load_task("NDVI")
