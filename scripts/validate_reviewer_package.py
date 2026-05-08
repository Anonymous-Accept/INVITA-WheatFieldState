"""Validate that the repository is a code-only reviewer package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REQUIRED_PATHS = (
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "index.html",
    "assets/styles.css",
    "docs/reviewer_guide.md",
    "src/benchmarks/baselines",
    "src/benchmarks/neural_baselines",
    "src/data_processing/loaders",
    "src/data_processing/qa_generation",
    "experiments/runs",
    "scripts/audit_anonymity.py",
    "scripts/compare_zip_remote_manifest.py",
)

FORBIDDEN_PATHS = (
    "data",
    "outputs",
    "results",
    "logs",
    "metrics",
    "predictions",
    "checkpoints",
    "coverage",
    "cache",
    ".cache",
    "__pycache__",
    "extract",
    "extracted",
    "unzipped",
    "zip_extract",
    "remote_sync",
    "scratch",
    "tmp",
    "temp",
    "experiments/results",
    "docs/submission",
)

FORBIDDEN_FILE_NAMES = {
    "coverage.csv",
    "experiment_matrix.csv",
    "fallback_usage.csv",
    "metrics.csv",
    "metrics.json",
    "metrics_by_target_name.csv",
    "model_config.json",
    "run_config.json",
    "training_summary.csv",
    "training_summary.json",
}

FORBIDDEN_SUFFIXES = {
    ".jsonl",
    ".log",
    ".parquet",
    ".csv",
    ".pt",
    ".pth",
    ".ckpt",
    ".pyc",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".7z",
    ".db",
    ".sqlite",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
}

ALLOWED_HIDDEN = {
    ".git",
    ".gitignore",
}


def validate(root: Path) -> list[str]:
    errors: list[str] = []

    for rel in REQUIRED_PATHS:
        if not (root / rel).exists():
            errors.append(f"missing required path: {rel}")

    for rel in FORBIDDEN_PATHS:
        if (root / rel).exists():
            errors.append(f"forbidden package path exists: {rel}")

    for path in root.rglob("*"):
        rel = path.relative_to(root)
        rel_text = str(rel)
        if ".git" in rel.parts:
            continue
        if path.name.startswith(".") and path.name not in ALLOWED_HIDDEN:
            errors.append(f"unexpected hidden path: {rel_text}")
        if path.is_file() and path.name in FORBIDDEN_FILE_NAMES:
            errors.append(f"forbidden result artifact: {rel_text}")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden generated/data file: {rel_text}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    errors = validate(root)
    if errors:
        print("Reviewer package validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Reviewer package validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
