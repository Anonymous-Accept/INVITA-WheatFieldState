"""Focused leakage and interface tests for Sensor-summary model and Sensor-sequence Transformer behavior."""

from __future__ import annotations

import pandas as pd

from src.benchmarks.baselines._history import (
    build_history_features,
    build_input_history_features,
)
from src.benchmarks.baselines._payload_images import IMAGE_EMBEDDING_COLUMNS
from src.benchmarks.baselines._payload_numeric import _plot_token_from_plot_uid
from src.benchmarks.baselines.frozen_image_feature_model.dataloader import FrozenImageFeatureDataLoader
from src.benchmarks.baselines.sensor_sequence_transformer import SensorSequenceTransformerConfig, SensorSequenceTransformer


def test_history_features_exclude_same_day_future_and_source_asset() -> None:
    targets = pd.DataFrame(
        {
            "target_uid": ["t1"],
            "plot_uid": ["p1"],
            "target_date": ["2021-08-10"],
            "source_asset_uid": ["source"],
        }
    )
    observations = pd.DataFrame(
        {
            "plot_uid": ["p1", "p1", "p1", "p1"],
            "observation_date": [
                "2021-08-01",
                "2021-08-10",
                "2021-08-11",
                "2021-08-02",
            ],
            "metric": ["ndvi", "ndvi", "ndvi", "ndvi"],
            "value": [0.2, 0.9, 0.8, 0.7],
            "asset_uid": ["past", "same_day", "future", "source"],
        }
    )

    features = build_history_features(
        targets,
        observations,
        prefix="hist",
        windows=(14,),
        sequence_bins=2,
        sequence_horizon_days=14,
    )

    assert features["hist_ndvi_14d_count"].iloc[0] == 1
    assert features["hist_ndvi_14d_mean"].iloc[0] == 0.2
    assert features["hist_ndvi_14d_last"].iloc[0] == 0.2
    assert features["hist_ndvi_days_since_last"].iloc[0] == 9
    assert features["hist_has_history"].iloc[0] == 1.0


def test_input_history_features_require_legal_target_input_asset() -> None:
    targets = pd.DataFrame(
        {
            "target_uid": ["t1"],
            "plot_uid": ["p1"],
            "target_date": ["2021-08-10"],
            "source_asset_uid": ["source"],
        }
    )
    inputs = pd.DataFrame(
        {
            "target_uid": ["t1", "t1", "t1", "t1"],
            "asset_uid": ["legal", "unlisted_value", "same_day", "source"],
            "modality_verified": [
                "greenseeker_handheld",
                "greenseeker_handheld",
                "greenseeker_handheld",
                "greenseeker_handheld",
            ],
            "acquisition_date": [
                "2021-08-01",
                "2021-08-01",
                "2021-08-10",
                "2021-08-01",
            ],
        }
    )
    observations = pd.DataFrame(
        {
            "plot_uid": ["p1", "p1", "p1", "p1", "p2"],
            "observation_date": [
                "2021-08-01",
                "2021-08-01",
                "2021-08-09",
                "2021-08-01",
                "2021-08-01",
            ],
            "metric": ["ndvi", "ndvi", "ndvi", "ndvi", "ndvi"],
            "value": [0.2, 0.9, 0.7, 0.6, 0.8],
            "asset_uid": ["legal", "not_in_inputs", "same_day", "source", "legal"],
        }
    )

    features = build_input_history_features(
        targets,
        inputs,
        observations,
        prefix="hist",
        windows=(14,),
        sequence_bins=2,
        sequence_horizon_days=14,
        modality_pattern="greenseeker",
    )

    assert features["hist_ndvi_14d_count"].iloc[0] == 1
    assert features["hist_ndvi_14d_mean"].iloc[0] == 0.2
    assert features["hist_ndvi_days_since_last"].iloc[0] == 9


def test_uav_payload_plot_token_uses_uav_xy_order() -> None:
    assert _plot_token_from_plot_uid("SYNTHETIC_TRIAL::SYNTHETIC_PLOT_1_10") == "x10_y1"


def test_frozen_image_feature_model_filters_and_embeds_legal_rgb_assets() -> None:
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
            "asset_uid": ["past", "same_day", "future", "source", "sensor"],
            "modality_verified": [
                "field_camera",
                "field_camera",
                "field_camera",
                "field_camera",
                "greenseeker_handheld",
            ],
            "acquisition_date": [
                "2021-08-01",
                "2021-08-10",
                "2021-08-11",
                "2021-08-01",
                "2021-08-01",
            ],
            "relationship": ["plot_asset"] * 5,
        }
    )
    loader = FrozenImageFeatureDataLoader.__new__(FrozenImageFeatureDataLoader)
    embeddings = pd.DataFrame(
        {
            "asset_uid": ["past"],
            "image_embedding_available": [1.0],
            **{column: [0.0] for column in IMAGE_EMBEDDING_COLUMNS},
        }
    )
    embeddings.loc[0, "image_emb_000"] = 0.5

    prepared = loader._prepare_targets(
        "NDVI", targets, inputs, image_embeddings=embeddings
    )

    assert prepared["rgb_n_pre_target_images"].iloc[0] == 1
    assert prepared["rgb_n_encoded_images"].iloc[0] == 1
    assert prepared["rgb_days_since_last_image"].iloc[0] == 9
    assert prepared["rgb_has_pre_target_image"].iloc[0] == 1.0
    assert prepared["rgb_embedding_mean_000"].iloc[0] == 0.5


def test_sensor_sequence_transformer_predicts_from_fixed_sequence_features() -> None:
    frame = pd.DataFrame(
        {
            "hist_ndvi_seq_00_mean": [0.1, 0.2, 0.8, 0.9],
            "hist_ndvi_seq_00_count": [1.0, 1.0, 2.0, 2.0],
            "hist_ndvi_seq_01_mean": [0.2, 0.3, 0.9, 1.0],
            "hist_ndvi_seq_01_count": [1.0, 1.0, 2.0, 2.0],
            "target_value_num": [0.15, 0.25, 0.85, 0.95],
        }
    )
    model = SensorSequenceTransformer(
        SensorSequenceTransformerConfig(hidden_dim=8, batch_size=2, n_epochs=1, device="cpu")
    ).fit(frame)

    pred = model.predict_frame(frame)

    assert pred["prediction"].notna().all()
    assert len(pred) == len(frame)
