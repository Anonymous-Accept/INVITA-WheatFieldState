"""
Parquet reader for the governed derived release

Reads and provides convenient access to all parquet files in the INVITA dataset:
- shared/plots.parquet (79,482 plots)
- shared/assets.parquet (multimodal asset registry)
- shared/provenance_manifest.parquet
- tasks/*/targets.parquet (task-specific targets)
- tasks/*/inputs_index.parquet (multimodal input links)

Usage:
    from data.loaders.parquet_reader import ParquetReader

    reader = ParquetReader()

    # Load plots
    plots_df = reader.load_plots()

    # Load task data
    targets, inputs = reader.load_task('NDVI')

    # Get task statistics
    stats = reader.get_task_stats('NDVI')
"""

import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import logging

from src.benchmarks.paths import default_data_root

logger = logging.getLogger(__name__)


class ParquetReader:
    """
    Convenient reader for all INVITA parquet files.

    Provides cached access to:
    - Shared metadata (plots, assets, provenance)
    - Task-specific data (targets, inputs_index)
    - Task metadata (task cards)
    """

    DEFAULT_DATA_ROOT = default_data_root()

    # Task names
    TASKS = [
        'NDVI',
        'LAI',
        'FCover',
        'Zadoks'
    ]

    def __init__(self, data_root: Optional[Path] = None):
        """
        Initialize ParquetReader.

        Args:
            data_root: Root directory of the governed derived release.
        """
        self.data_root = Path(data_root) if data_root else self.DEFAULT_DATA_ROOT

        if not self.data_root.exists():
            raise FileNotFoundError(f"Data root not found: {self.data_root}")

        # Define paths
        self.shared_dir = self.data_root / "shared"
        self.tasks_dir = self.data_root / "tasks"
        self.payload_dir = self.data_root / "payload"

        # Cache for loaded dataframes
        self._cache = {}

    # ========================================
    # Shared Metadata
    # ========================================

    def load_plots(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load plots.parquet (79,482 plot records).

        Returns:
            DataFrame with columns: plot_uid, trial_code, polygon_ref, etc.
        """
        cache_key = 'plots'
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        path = self.shared_dir / "plots.parquet"
        logger.info(f"Loading plots from {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df):,} plots")

        if use_cache:
            self._cache[cache_key] = df
        return df

    def load_assets(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load assets.parquet (multimodal asset registry).

        Returns:
            DataFrame with columns: asset_uid, file_type, source, etc.
        """
        cache_key = 'assets'
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        path = self.shared_dir / "assets.parquet"
        logger.info(f"Loading assets from {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df):,} assets")

        if use_cache:
            self._cache[cache_key] = df
        return df

    def load_provenance(self, use_cache: bool = True) -> pd.DataFrame:
        """
        Load provenance_manifest.parquet.

        Returns:
            DataFrame with provenance tracking information
        """
        cache_key = 'provenance'
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        path = self.shared_dir / "provenance_manifest.parquet"
        logger.info(f"Loading provenance from {path}")
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df):,} provenance records")

        if use_cache:
            self._cache[cache_key] = df
        return df

    # ========================================
    # Task Data
    # ========================================

    def load_task(
        self,
        task_name: str,
        use_cache: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load task targets and inputs_index.

        Args:
            task_name: One of ['NDVI', 'LAI', 'FCover', 'Zadoks']
            use_cache: Whether to use cached data

        Returns:
            Tuple of (targets_df, inputs_index_df)
        """
        if task_name not in self.TASKS:
            raise ValueError(f"Invalid task: {task_name}. Must be one of {self.TASKS}")

        task_dir = self.tasks_dir / task_name

        # Load targets
        targets_key = f'{task_name}_targets'
        if use_cache and targets_key in self._cache:
            targets = self._cache[targets_key]
        else:
            targets_path = task_dir / "targets.parquet"
            logger.info(f"Loading {task_name} targets from {targets_path}")
            targets = pd.read_parquet(targets_path)
            logger.info(f"Loaded {len(targets):,} targets")
            if use_cache:
                self._cache[targets_key] = targets

        # Load inputs_index
        inputs_key = f'{task_name}_inputs'
        if use_cache and inputs_key in self._cache:
            inputs = self._cache[inputs_key]
        else:
            inputs_path = task_dir / "inputs_index.parquet"
            logger.info(f"Loading {task_name} inputs from {inputs_path}")
            inputs = pd.read_parquet(inputs_path)
            logger.info(f"Loaded {len(inputs):,} input links")
            if use_cache:
                self._cache[inputs_key] = inputs

        return targets, inputs

    def load_targets(self, task_name: str, use_cache: bool = True) -> pd.DataFrame:
        """Load only targets for a task."""
        targets, _ = self.load_task(task_name, use_cache=use_cache)
        return targets

    def load_inputs_index(self, task_name: str, use_cache: bool = True) -> pd.DataFrame:
        """Load only inputs_index for a task."""
        _, inputs = self.load_task(task_name, use_cache=use_cache)
        return inputs

    def load_task_card(self, task_name: str) -> str:
        """
        Load task card markdown file.

        Args:
            task_name: Task name

        Returns:
            Task card content as string
        """
        if task_name not in self.TASKS:
            raise ValueError(f"Invalid task: {task_name}")

        task_dir = self.tasks_dir / task_name
        card_path = task_dir / "task_card.md"

        if not card_path.exists():
            logger.warning(f"Task card not found: {card_path}")
            return ""

        with open(card_path, 'r') as f:
            return f.read()

    # ========================================
    # Statistics and Utilities
    # ========================================

    def get_task_stats(self, task_name: str) -> Dict[str, Any]:
        """
        Get statistics for a task.

        Args:
            task_name: Task name

        Returns:
            Dict with statistics
        """
        targets, inputs = self.load_task(task_name)

        stats = {
            'task_name': task_name,
            'num_targets': len(targets),
            'num_plots': targets['plot_uid'].nunique() if 'plot_uid' in targets.columns else 'N/A',
            'num_input_links': len(inputs),
            'num_unique_assets': inputs['asset_uid'].nunique() if 'asset_uid' in inputs.columns else 'N/A',
        }

        # Target value statistics (if numeric)
        target_col = self._get_target_column(task_name, targets)
        if target_col and pd.api.types.is_numeric_dtype(targets[target_col]):
            stats['target_mean'] = float(targets[target_col].mean())
            stats['target_std'] = float(targets[target_col].std())
            stats['target_min'] = float(targets[target_col].min())
            stats['target_max'] = float(targets[target_col].max())
            stats['target_median'] = float(targets[target_col].median())

        # Input coverage per target
        inputs_per_target = inputs.groupby('target_uid').size()
        stats['avg_inputs_per_target'] = float(inputs_per_target.mean())
        stats['median_inputs_per_target'] = float(inputs_per_target.median())
        stats['max_inputs_per_target'] = int(inputs_per_target.max())
        stats['min_inputs_per_target'] = int(inputs_per_target.min())

        return stats

    def get_all_task_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all tasks.

        Returns:
            Dict mapping task_name -> task_stats
        """
        all_stats = {}
        for task_name in self.TASKS:
            logger.info(f"Computing stats for {task_name}")
            all_stats[task_name] = self.get_task_stats(task_name)
        return all_stats

    def _get_target_column(self, task_name: str, targets_df: pd.DataFrame) -> Optional[str]:
        """Infer the target value column name."""
        # Common target column names
        possible_names = [
            'target_value',
            'value',
            'ndvi',
            'lai',
            'fcover',
            'growth_stage',
            'zadoks_score'
        ]

        for name in possible_names:
            if name in targets_df.columns:
                return name

        return None

    def get_modality_coverage(self, task_name: str) -> pd.DataFrame:
        """
        Analyze modality coverage for a task.

        Args:
            task_name: Task name

        Returns:
            DataFrame with modality counts per target
        """
        _, inputs = self.load_task(task_name)
        assets = self.load_assets()

        # Merge to get file_type for each input
        inputs_with_type = inputs.merge(
            assets[['asset_uid', 'file_type']],
            on='asset_uid',
            how='left'
        )

        # Count modalities per target
        modality_counts = inputs_with_type.groupby(['target_uid', 'file_type']).size().unstack(fill_value=0)

        return modality_counts

    def clear_cache(self):
        """Clear all cached dataframes."""
        self._cache = {}
        logger.info("Cache cleared")

    def get_cache_info(self) -> Dict[str, int]:
        """Get information about cached data."""
        info = {}
        for key, df in self._cache.items():
            if isinstance(df, pd.DataFrame):
                info[key] = len(df)
        return info


def test_parquet_reader():
    """Test the ParquetReader functionality."""
    print("Testing ParquetReader...")

    reader = ParquetReader()

    # Test 1: Load shared metadata
    print("\n=== Loading Shared Metadata ===")
    plots = reader.load_plots()
    print(f"Plots: {len(plots):,} records")
    print(f"Columns: {list(plots.columns)[:5]}...")

    assets = reader.load_assets()
    print(f"Assets: {len(assets):,} records")
    print(f"Columns: {list(assets.columns)[:5]}...")

    # Test 2: Load task data
    print("\n=== Loading Task Data (NDVI) ===")
    targets, inputs = reader.load_task('NDVI')
    print(f"Targets: {len(targets):,} records")
    print(f"Inputs: {len(inputs):,} links")

    # Test 3: Get task statistics
    print("\n=== Task Statistics ===")
    for task_name in reader.TASKS:
        stats = reader.get_task_stats(task_name)
        print(f"\n{task_name}:")
        print(f"  Targets: {stats['num_targets']:,}")
        print(f"  Plots: {stats['num_plots']}")
        print(f"  Input links: {stats['num_input_links']:,}")
        print(f"  Avg inputs/target: {stats['avg_inputs_per_target']:.1f}")

    # Test 4: Modality coverage
    print("\n=== Modality Coverage (NDVI) ===")
    coverage = reader.get_modality_coverage('NDVI')
    print(f"Shape: {coverage.shape}")
    print(f"Modalities: {list(coverage.columns)[:10]}...")

    # Test 5: Cache info
    print("\n=== Cache Info ===")
    cache_info = reader.get_cache_info()
    for key, count in cache_info.items():
        print(f"  {key}: {count:,} records")


if __name__ == "__main__":
    # Run tests
    test_parquet_reader()
