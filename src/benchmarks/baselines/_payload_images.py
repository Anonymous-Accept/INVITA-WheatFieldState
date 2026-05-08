"""Payload-backed frozen image embeddings for RGB baselines."""

from __future__ import annotations

import io
import logging
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.loaders.payload_loader import PayloadLoader

logger = logging.getLogger(__name__)

IMAGE_EMBEDDING_DIM = 512
IMAGE_EMBEDDING_MODEL = "torchvision_squeezenet1_1_imagenet1k_features_avgpool"
IMAGE_EMBEDDING_COLUMNS = tuple(
    f"image_emb_{idx:03d}" for idx in range(IMAGE_EMBEDDING_DIM)
)
RGB_IMAGE_MODALITIES = ("field_camera",)
IMAGE_EMBEDDING_BASE_COLUMNS = (
    "asset_uid",
    "modality_verified",
    "payload_rel_path",
    "image_embedding_available",
)


def load_payload_image_embeddings(
    data_root: Path,
    *,
    asset_uids: pd.Series | list[str] | set[str] | None = None,
    modalities: tuple[str, ...] = RGB_IMAGE_MODALITIES,
    batch_size: int = 32,
    max_file_bytes: int = 50_000_000,
) -> pd.DataFrame:
    """
    Decode image payloads and extract frozen ImageNet SqueezeNet embeddings.

    The function reads true payload bytes through the asset locator. It skips
    assets that are not regular files, are larger than ``max_file_bytes``, or
    cannot be decoded as field-camera images.
    """

    data_root = Path(data_root)
    assets_path = data_root / "shared" / "assets.parquet"
    locator_path = data_root / "payload" / "asset_locator.parquet"
    payload_path = data_root / "payload" / "payload.db"
    if not assets_path.exists():
        raise FileNotFoundError(f"Missing assets table: {assets_path}")
    if not locator_path.exists():
        raise FileNotFoundError(f"Missing asset locator: {locator_path}")
    if not payload_path.exists():
        raise FileNotFoundError(f"Missing payload database: {payload_path}")

    assets = pd.read_parquet(
        assets_path,
        columns=["asset_uid", "modality_verified"],
    )
    locator = pd.read_parquet(
        locator_path,
        columns=["asset_uid", "kind", "payload_rel_path", "modality_verified"],
    ).rename(columns={"modality_verified": "locator_modality"})
    frame = assets.merge(locator, on="asset_uid", how="inner", validate="one_to_one")

    if asset_uids is not None:
        uid_set = set(map(str, asset_uids))
        frame = frame.loc[frame["asset_uid"].isin(uid_set)].copy()
    frame = frame.loc[
        frame["modality_verified"].isin(modalities) & frame["kind"].eq("file")
    ].copy()
    frame = frame.drop_duplicates("asset_uid").sort_values("asset_uid")
    if frame.empty:
        return empty_image_embeddings()

    encoder = _get_squeezenet_encoder()
    rows: list[dict[str, object]] = []
    pending_images = []
    pending_meta: list[tuple[str, str, str]] = []

    def flush() -> None:
        nonlocal pending_images, pending_meta
        if not pending_images:
            return
        embeddings = encoder.encode(pending_images)
        for (asset_uid, modality, payload_rel_path), embedding in zip(
            pending_meta, embeddings, strict=True
        ):
            row: dict[str, object] = {
                "asset_uid": asset_uid,
                "modality_verified": modality,
                "payload_rel_path": payload_rel_path,
                "image_embedding_available": 1.0,
            }
            row.update(
                {
                    column: float(value)
                    for column, value in zip(
                        IMAGE_EMBEDDING_COLUMNS, embedding, strict=True
                    )
                }
            )
            rows.append(row)
        pending_images = []
        pending_meta = []

    with PayloadLoader(
        payload_db_path=payload_path,
        asset_locator_path=locator_path,
    ) as loader:
        for item in frame.itertuples(index=False):
            file_bytes = loader._reconstruct_file(str(item.payload_rel_path))
            if file_bytes is None:
                continue
            if len(file_bytes) > max_file_bytes:
                logger.info(
                    "Skipping oversized image asset %s (%d bytes)",
                    item.asset_uid,
                    len(file_bytes),
                )
                continue
            image = _decode_rgb_image(file_bytes)
            if image is None:
                logger.info("Skipping undecodable image asset %s", item.asset_uid)
                continue
            pending_images.append(image)
            pending_meta.append(
                (str(item.asset_uid), str(item.modality_verified), str(item.payload_rel_path))
            )
            if len(pending_images) >= batch_size:
                flush()
        flush()

    if not rows:
        return empty_image_embeddings()
    return pd.DataFrame(rows, columns=list(IMAGE_EMBEDDING_BASE_COLUMNS) + list(IMAGE_EMBEDDING_COLUMNS))


def empty_image_embeddings() -> pd.DataFrame:
    """Return an empty image embedding frame with the full schema."""

    return pd.DataFrame(
        columns=list(IMAGE_EMBEDDING_BASE_COLUMNS) + list(IMAGE_EMBEDDING_COLUMNS)
    )


class _SqueezeNetEncoder:
    """Frozen SqueezeNet feature extractor."""

    def __init__(self) -> None:
        import torch
        from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1

        weights = SqueezeNet1_1_Weights.IMAGENET1K_V1
        model = squeezenet1_1(weights=weights)
        model.eval()
        self._torch = torch
        self._model = model.features
        self._transform = weights.transforms()

    def encode(self, images: list[object]) -> np.ndarray:
        """Encode RGB PIL images into 512-dimensional average-pooled features."""

        torch = self._torch
        batch = torch.stack([self._transform(image) for image in images])
        with torch.no_grad():
            features = self._model(batch)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return pooled.flatten(1).cpu().numpy().astype(np.float32)


@lru_cache(maxsize=1)
def _get_squeezenet_encoder() -> _SqueezeNetEncoder:
    return _SqueezeNetEncoder()


def _decode_rgb_image(file_bytes: bytes):
    try:
        from PIL import Image, ImageOps, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Pillow is required for Frozen image-feature model image decoding") from exc

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (OSError, UnidentifiedImageError, ValueError):
        return None
