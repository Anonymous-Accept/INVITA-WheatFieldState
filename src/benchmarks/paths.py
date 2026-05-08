"""Shared path defaults for the anonymous reviewer code package."""

from __future__ import annotations

import os
from pathlib import Path

CANONICAL_SPLIT_NAME = "plot_disjoint"
DEFAULT_RELEASE_DIR = Path("..") / "INVITA-WheatFieldState-derived_release"
DEFAULT_OUTPUT_DIR = Path("outputs")


def default_data_root() -> Path:
    """Return the governed derived-release root supplied by the reviewer."""

    return Path(os.environ.get("INVITA_DATA_ROOT", str(DEFAULT_RELEASE_DIR)))


def default_split_root() -> Path:
    """Return the canonical split directory for the governed release."""

    default = default_data_root() / "splits" / CANONICAL_SPLIT_NAME
    return Path(os.environ.get("INVITA_SPLIT_ROOT", str(default)))


def default_output_root() -> Path:
    """Return the local output directory for generated experiment artifacts."""

    return Path(os.environ.get("INVITA_OUTPUT_ROOT", str(DEFAULT_OUTPUT_DIR)))
