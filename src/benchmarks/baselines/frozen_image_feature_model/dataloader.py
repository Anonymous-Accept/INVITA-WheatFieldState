"""Independent dataloader for the Frozen image-feature model baseline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.benchmarks.baselines._payload_images import (
    IMAGE_EMBEDDING_COLUMNS,
    IMAGE_EMBEDDING_DIM,
    IMAGE_EMBEDDING_MODEL,
    RGB_IMAGE_MODALITIES,
    empty_image_embeddings,
    load_payload_image_embeddings,
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

FROZEN_IMAGE_FEATURE_TARGET_COLUMNS = (
    "target_uid",
    "plot_uid",
    "target_date",
    "target_name",
    "target_value_num",
    "source_asset_uid",
)

FROZEN_IMAGE_FEATURE_INPUT_COLUMNS = (
    "target_uid",
    "asset_uid",
    "modality_verified",
    "acquisition_date",
    "relationship",
)

RGB_MODALITIES = frozenset(RGB_IMAGE_MODALITIES)

RGB_EMBEDDING_MEAN_COLUMNS = tuple(
    f"rgb_embedding_mean_{idx:03d}" for idx in range(IMAGE_EMBEDDING_DIM)
)
RGB_EMBEDDING_STD_COLUMNS = tuple(
    f"rgb_embedding_std_{idx:03d}" for idx in range(IMAGE_EMBEDDING_DIM)
)
RGB_SUMMARY_COLUMNS = (
    "rgb_n_pre_target_images",
    "rgb_n_encoded_images",
    "rgb_days_since_last_image",
    "rgb_has_pre_target_image",
)

FROZEN_IMAGE_FEATURE_REQUIRED_COLUMNS = (
    "target_uid",
    "plot_uid",
    "target_date",
    "target_name",
    "target_value_num",
    "rgb_has_pre_target_image",
)


class FrozenImageFeatureDataLoader:
    """
    Load pre-target field-camera image features for Frozen image-feature model.

    Frozen image-feature model decodes true field-camera payload images, extracts frozen ImageNet
    SqueezeNet features, and aggregates the most recent legal pre-target image
    embeddings for each target. It does not use target manifests as image
    features.
    """

    def __init__(self, config: BaselineDataConfig | None = None) -> None:
        self.config = config or BaselineDataConfig()
        self.data_root = Path(self.config.data_root)
        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root does not exist: {self.data_root}")
        self._image_embedding_cache = empty_image_embeddings()

    def load_task(self, task_name: str) -> BaselineTaskData:
        """Load Frozen image-feature model train/validation/test frames for one task."""

        validate_task_name(task_name)
        task_dir = self.data_root / "tasks" / task_name
        targets_path = task_dir / "targets.parquet"
        inputs_path = task_dir / "inputs_index.parquet"
        if not targets_path.exists():
            raise FileNotFoundError(f"Missing targets file: {targets_path}")
        if not inputs_path.exists():
            raise FileNotFoundError(f"Missing inputs index file: {inputs_path}")

        targets = pd.read_parquet(targets_path, columns=list(FROZEN_IMAGE_FEATURE_TARGET_COLUMNS))
        inputs = pd.read_parquet(inputs_path, columns=list(FROZEN_IMAGE_FEATURE_INPUT_COLUMNS))
        targets = self._prepare_targets(task_name, targets, inputs)
        rows_before_subset = len(targets)
        targets = targets.loc[targets["rgb_n_encoded_images"] > 0].copy()

        task_split_dir = split_dir(self.config, task_name)
        split_frames = {
            split: read_split_frame(
                task_split_dir / f"{split}.csv",
                targets,
                required_columns=FROZEN_IMAGE_FEATURE_REQUIRED_COLUMNS,
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
        report["rgb_audit"] = {
            "rows_with_pre_target_rgb_asset": int(
                (targets["rgb_has_pre_target_image"] > 0).sum()
            ),
            "rows_with_encoded_rgb": int((targets["rgb_n_encoded_images"] > 0).sum()),
            "rows_after_required_modality_subset": int(len(targets)),
            "rows_before_required_modality_subset": int(rows_before_subset),
            "image_modalities": list(RGB_IMAGE_MODALITIES),
            "embedding_model": IMAGE_EMBEDDING_MODEL,
            "embedding_dim": IMAGE_EMBEDDING_DIM,
            "input_asset_policy": "image embeddings are limited to legal pre-target inputs_index assets for each target",
            "required_modality_policy": "targets without at least one successfully encoded legal pre-target field-camera image are excluded",
        }
        return BaselineTaskData(
            task_name=task_name, validation_report=report, **split_frames
        )

    def _prepare_targets(
        self,
        task_name: str,
        targets: pd.DataFrame,
        inputs: pd.DataFrame,
        image_embeddings: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        missing = set(FROZEN_IMAGE_FEATURE_TARGET_COLUMNS) - set(targets.columns)
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

        if image_embeddings is None:
            image_embeddings = self._load_image_embeddings(
                task_name, self._legal_image_asset_uids(frame, inputs)
            )
        rgb = self._build_rgb_features(frame, inputs, image_embeddings)
        frame = frame.merge(rgb, on="target_uid", how="left", validate="one_to_one")
        rgb_cols = [col for col in rgb.columns if col != "target_uid"]
        frame[rgb_cols] = frame[rgb_cols].fillna(0.0)
        frame["rgb_days_since_last_image"] = frame["rgb_days_since_last_image"].replace(
            0.0, 999.0
        )
        return frame

    def _load_image_embeddings(
        self, task_name: str, image_asset_uids: set[str]
    ) -> pd.DataFrame:
        requested_assets = set(map(str, image_asset_uids))
        if not requested_assets:
            return empty_image_embeddings()

        cached_assets = (
            set(self._image_embedding_cache["asset_uid"].astype(str).tolist())
            if not self._image_embedding_cache.empty
            else set()
        )
        missing_assets = requested_assets - cached_assets
        if missing_assets:
            new_embeddings = load_payload_image_embeddings(
                self.data_root,
                asset_uids=missing_assets,
                modalities=RGB_IMAGE_MODALITIES,
            )
            if not new_embeddings.empty:
                if self._image_embedding_cache.empty:
                    self._image_embedding_cache = new_embeddings.copy()
                else:
                    self._image_embedding_cache = pd.concat(
                        [self._image_embedding_cache, new_embeddings],
                        ignore_index=True,
                    ).drop_duplicates("asset_uid", keep="last")

        output = self._image_embedding_cache.loc[
            self._image_embedding_cache["asset_uid"].isin(requested_assets)
        ].copy()
        output.attrs["task_name"] = task_name
        return output

    def _legal_image_asset_uids(
        self, targets: pd.DataFrame, inputs: pd.DataFrame
    ) -> set[str]:
        input_frame = inputs[list(FROZEN_IMAGE_FEATURE_INPUT_COLUMNS)].copy()
        input_frame["asset_uid"] = normalize_categorical(input_frame["asset_uid"])
        input_frame["modality_verified"] = normalize_categorical(
            input_frame["modality_verified"]
        )
        input_frame["acquisition_date"] = pd.to_datetime(
            input_frame["acquisition_date"], errors="coerce"
        )
        legal = input_frame.merge(
            targets[["target_uid", "target_date", "source_asset_uid"]],
            on="target_uid",
            how="inner",
            validate="many_to_one",
        )
        legal["source_asset_uid"] = normalize_categorical(legal["source_asset_uid"])
        legal = legal.loc[
            legal["modality_verified"].isin(RGB_MODALITIES)
            & legal["acquisition_date"].notna()
            & legal["acquisition_date"].lt(legal["target_date"])
            & legal["asset_uid"].ne(legal["source_asset_uid"])
        ]
        return set(legal["asset_uid"].tolist())

    def _build_rgb_features(
        self,
        targets: pd.DataFrame,
        inputs: pd.DataFrame,
        image_embeddings: pd.DataFrame,
        *,
        max_images_per_target: int = 3,
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
        input_frame["asset_uid"] = normalize_categorical(input_frame["asset_uid"])
        input_frame["source_asset_uid"] = normalize_categorical(
            input_frame["source_asset_uid"]
        )
        input_frame["acquisition_date"] = pd.to_datetime(
            input_frame["acquisition_date"], errors="coerce"
        )
        input_frame["target_date"] = pd.to_datetime(
            input_frame["target_date"], errors="raise"
        )
        legal = input_frame.loc[
            input_frame["modality_verified"].isin(RGB_MODALITIES)
            & input_frame["acquisition_date"].notna()
            & input_frame["acquisition_date"].lt(input_frame["target_date"])
            & input_frame["asset_uid"].ne(input_frame["source_asset_uid"])
        ].copy()

        output = _empty_rgb_feature_frame(targets)
        if legal.empty:
            return output

        counts = legal.groupby("target_uid").size().rename("rgb_n_pre_target_images")
        output["rgb_n_pre_target_images"] = (
            output["target_uid"].map(counts).fillna(0.0)
        )

        last_dates = legal.groupby("target_uid")["acquisition_date"].max()
        target_dates = targets.set_index("target_uid")["target_date"]
        output["rgb_days_since_last_image"] = output["target_uid"].map(
            (target_dates - last_dates).dt.days
        )
        output["rgb_days_since_last_image"] = output[
            "rgb_days_since_last_image"
        ].fillna(999.0)
        output["rgb_has_pre_target_image"] = (
            output["rgb_n_pre_target_images"] > 0
        ).astype(float)

        if image_embeddings.empty:
            return output

        embedding_cols = [
            col for col in IMAGE_EMBEDDING_COLUMNS if col in image_embeddings.columns
        ]
        if not embedding_cols:
            return output

        embeddings = image_embeddings[
            ["asset_uid", "image_embedding_available", *embedding_cols]
        ].copy()
        embeddings["asset_uid"] = normalize_categorical(embeddings["asset_uid"])
        embeddings["image_embedding_available"] = pd.to_numeric(
            embeddings["image_embedding_available"], errors="coerce"
        ).fillna(0.0)
        embeddings = embeddings.loc[embeddings["image_embedding_available"] > 0].copy()
        if embeddings.empty:
            return output

        legal["days_before_target"] = (
            legal["target_date"] - legal["acquisition_date"]
        ).dt.days.astype(float)
        selected = (
            legal.sort_values(
                ["target_uid", "days_before_target", "asset_uid"],
                ascending=[True, True, True],
            )
            .groupby("target_uid", sort=False)
            .head(max_images_per_target)
        )
        selected = selected.merge(embeddings, on="asset_uid", how="inner")
        if selected.empty:
            return output

        encoded_counts = (
            selected.groupby("target_uid").size().rename("rgb_n_encoded_images")
        )
        output["rgb_n_encoded_images"] = (
            output["target_uid"].map(encoded_counts).fillna(0.0)
        )

        grouped = selected.groupby("target_uid", sort=False)[embedding_cols]
        means = grouped.mean().rename(
            columns={
                source: target
                for source, target in zip(
                    IMAGE_EMBEDDING_COLUMNS, RGB_EMBEDDING_MEAN_COLUMNS, strict=True
                )
                if source in embedding_cols
            }
        )
        stds = grouped.std(ddof=0).rename(
            columns={
                source: target
                for source, target in zip(
                    IMAGE_EMBEDDING_COLUMNS, RGB_EMBEDDING_STD_COLUMNS, strict=True
                )
                if source in embedding_cols
            }
        )
        output = output.set_index("target_uid")
        output.loc[means.index, means.columns] = means
        output.loc[stds.index, stds.columns] = stds
        output = output.reset_index()

        feature_cols = (
            list(RGB_SUMMARY_COLUMNS)
            + list(RGB_EMBEDDING_MEAN_COLUMNS)
            + list(RGB_EMBEDDING_STD_COLUMNS)
        )
        output[feature_cols] = output[feature_cols].fillna(0.0)
        output["rgb_days_since_last_image"] = output[
            "rgb_days_since_last_image"
        ].replace(0.0, 999.0)
        return output[["target_uid", *feature_cols]]


def _empty_rgb_feature_frame(targets: pd.DataFrame) -> pd.DataFrame:
    output = pd.DataFrame(
        {
            "target_uid": targets["target_uid"].to_numpy(),
            "rgb_n_pre_target_images": 0.0,
            "rgb_n_encoded_images": 0.0,
            "rgb_days_since_last_image": 999.0,
            "rgb_has_pre_target_image": 0.0,
        }
    )
    embeddings = pd.DataFrame(
        0.0,
        index=output.index,
        columns=list(RGB_EMBEDDING_MEAN_COLUMNS) + list(RGB_EMBEDDING_STD_COLUMNS),
    )
    return pd.concat([output, embeddings], axis=1)
