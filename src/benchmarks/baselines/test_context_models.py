"""Focused tests for Crop-date prior and Observation-availability model behavior."""

from __future__ import annotations

import pandas as pd

from src.benchmarks.baselines.crop_date_prior import CropDatePriorConfig, CropDatePrior
from src.benchmarks.baselines.tabular_metadata_model import TabularMetadataConfig, TabularMetadataModel
from src.benchmarks.baselines.observation_availability_model.dataloader import ObservationAvailabilityDataLoader
from src.benchmarks.baselines.observation_availability_model.predictor import ObservationAvailabilityModel, ObservationAvailabilityConfig


def test_crop_date_prior_uses_target_aware_hierarchy() -> None:
    train = pd.DataFrame(
        {
            "target_name": ["lai", "lai", "laicab", "laicab"],
            "crop_type": ["Wheat", "Wheat", "Wheat", "Wheat"],
            "cultivar_key": ["A", "A", "A", "A"],
            "trial_year": [2021, 2021, 2021, 2021],
            "target_doy_bin_14": [10, 10, 10, 10],
            "target_value_num": [1.0, 2.0, 50.0, 70.0],
        }
    )
    test = train.drop(columns=["target_value_num"]).copy()
    model = CropDatePrior(CropDatePriorConfig(min_samples=1)).fit(train)

    pred = model.predict_frame(test)

    assert pred.loc[pred["target_name"] == "lai", "prediction"].iloc[0] == 1.5
    assert pred.loc[pred["target_name"] == "laicab", "prediction"].iloc[0] == 60.0


def test_tabular_metadata_model_predicts_with_unknown_categories() -> None:
    train = pd.DataFrame(
        {
            "target_name": ["ndvi", "ndvi", "ndvi", "ndvi"],
            "source_dataset": ["a", "a", "b", "b"],
            "instrument": ["gs", "gs", "gs", "gs"],
            "subset": ["main", "main", "main", "main"],
            "state": ["WA", "WA", "NSW", "NSW"],
            "region_name": ["R1", "R1", "R2", "R2"],
            "site_id": ["S1", "S1", "S2", "S2"],
            "site_name": ["Site1", "Site1", "Site2", "Site2"],
            "crop_type": ["Wheat", "Wheat", "Wheat", "Wheat"],
            "cultivar_key": ["A", "B", "A", "B"],
            "replicate": ["1", "1", "2", "2"],
            "block": ["x", "x", "y", "y"],
            "trial_year": [2020, 2020, 2021, 2021],
            "target_doy": [100, 110, 120, 130],
            "target_month": [4, 4, 4, 5],
            "target_week": [14, 15, 16, 17],
            "target_doy_bin_14": [7, 7, 8, 9],
            "sowing_doy": [60, 60, 61, 61],
            "days_since_sowing": [40, 50, 59, 69],
            "area_m2": [10.0, 10.0, 12.0, 12.0],
            "target_value_num": [0.2, 0.3, 0.7, 0.8],
        }
    )
    test = train.iloc[[0]].copy()
    test.loc[:, "cultivar_key"] = "unseen"
    model = TabularMetadataModel(TabularMetadataConfig(n_estimators=5, max_depth=2)).fit(train)

    pred = model.predict_frame(test)

    assert pred["prediction"].notna().all()


def test_observation_availability_model_filters_future_same_day_and_source_assets() -> None:
    targets = pd.DataFrame(
        {
            "target_uid": ["t1"],
            "plot_uid": ["p1"],
            "target_date": ["2021-08-10"],
            "target_name": ["ndvi"],
            "target_value_num": [0.5],
            "source_asset_uid": ["source"],
        }
    )
    inputs = pd.DataFrame(
        {
            "target_uid": ["t1", "t1", "t1", "t1", "t1"],
            "asset_uid": ["past", "same_day", "future", "source", "tab"],
            "modality_verified": [
                "greenseeker_handheld",
                "greenseeker_handheld",
                "greenseeker_handheld",
                "greenseeker_handheld",
                "tabular",
            ],
            "acquisition_date": [
                "2021-08-01",
                "2021-08-10",
                "2021-08-11",
                "2021-08-01",
                "",
            ],
            "temporal_offset_days": [-9, 0, 1, -9, pd.NA],
            "relationship": ["plot_asset"] * 5,
            "leakage_filter_notes": [""] * 5,
        }
    )
    loader = ObservationAvailabilityDataLoader.__new__(ObservationAvailabilityDataLoader)

    prepared = loader._prepare_targets("NDVI", targets, inputs)

    assert prepared["n_greenseeker_handheld"].iloc[0] == 1
    assert prepared["n_tabular"].iloc[0] == 1
    assert prepared["days_since_last_greenseeker_handheld"].iloc[0] == 9


def test_observation_availability_model_predicts_from_availability_features() -> None:
    train = pd.DataFrame(
        {
            "target_name": ["ndvi", "ndvi", "ndvi", "ndvi"],
            "n_legal_assets": [1, 2, 10, 11],
            "n_modalities_present": [1, 1, 2, 2],
            "has_greenseeker_handheld": [0, 0, 1, 1],
            "days_since_last_any": [999, 999, 3, 4],
            "target_value_num": [0.1, 0.2, 0.8, 0.9],
        }
    )
    model = ObservationAvailabilityModel(ObservationAvailabilityConfig(n_estimators=5, max_depth=2)).fit(train)
    pred = model.predict_frame(train)

    assert pred["prediction"].notna().all()
