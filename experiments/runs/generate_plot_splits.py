"""Generate deterministic plot-level train/val/test splits for INVITA tasks."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks.paths import default_data_root, default_output_root, default_split_root  # noqa: E402
from src.benchmarks.splits.plot_splitter import (  # noqa: E402
    DEFAULT_RATIOS,
    TASKS,
    PlotSplitGenerator,
    SplitConfig,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=default_data_root(),
        help="Root directory of the INVITA dataset build.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_split_root(),
        help="Directory where generated split CSVs and audits are written.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(TASKS),
        choices=list(TASKS),
        help="Tasks to split.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    ratios = {
        "train": args.train_ratio,
        "val": args.val_ratio,
        "test": args.test_ratio,
    }
    if ratios != DEFAULT_RATIOS:
        logger.info("Using custom ratios: %s", ratios)
    config = SplitConfig(
        data_root=args.data_root,
        output_root=args.output_root,
        ratios=ratios,
        seed=args.seed,
    )
    reports = PlotSplitGenerator(config).generate_all(tuple(args.tasks))
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
