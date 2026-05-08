"""Frozen image-feature model."""

from .dataloader import FrozenImageFeatureDataLoader
from .predictor import FrozenImageFeatureConfig, FrozenImageFeatureModel

__all__ = ["FrozenImageFeatureConfig", "FrozenImageFeatureDataLoader", "FrozenImageFeatureModel"]
