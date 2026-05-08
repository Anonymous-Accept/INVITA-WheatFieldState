"""
Multimodal aligner for plot-date examples

Aligns targets with their multimodal inputs and loads the actual data.
This is the core data loading component that brings together:
- Target labels (from targets.parquet)
- Input asset UIDs (from inputs_index.parquet)
- Actual multimodal data (from payload database)
- Metadata (plots, assets, provenance)

Usage:
    from data.loaders.multimodal_aligner import MultimodalAligner

    aligner = MultimodalAligner(task_name='NDVI')

    # Get a single aligned sample
    sample = aligner.get_sample(target_uid='some_target_uid')

    # Iterate through all samples
    for sample in aligner.iter_samples(batch_size=32):
        # sample contains: target_data, inputs_data, metadata
        pass
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Iterator, Tuple
import logging
from dataclasses import dataclass, field

from src.data_processing.loaders.payload_loader import PayloadLoader
from src.data_processing.loaders.parquet_reader import ParquetReader

logger = logging.getLogger(__name__)


@dataclass
class MultimodalSample:
    """
    A single multimodal sample for a task.

    Attributes:
        target_uid: Unique identifier for the target
        target_value: The target label/value
        target_metadata: Additional target metadata (date, plot_uid, etc.)
        inputs: List of input assets with their data
        plot_info: Plot metadata
    """
    target_uid: str
    target_value: Any
    target_metadata: Dict[str, Any] = field(default_factory=dict)
    inputs: List[Dict[str, Any]] = field(default_factory=list)
    plot_info: Optional[Dict[str, Any]] = None

    def get_inputs_by_modality(self, modality: str) -> List[Dict[str, Any]]:
        """Filter inputs by modality."""
        return [inp for inp in self.inputs if inp.get('modality') == modality]

    def has_modality(self, modality: str) -> bool:
        """Check if sample has a specific modality."""
        return any(inp.get('modality') == modality for inp in self.inputs)

    def count_modalities(self) -> Dict[str, int]:
        """Count available modalities."""
        counts = {}
        for inp in self.inputs:
            modality = inp.get('modality', 'unknown')
            counts[modality] = counts.get(modality, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'target_uid': self.target_uid,
            'target_value': self.target_value,
            'target_metadata': self.target_metadata,
            'num_inputs': len(self.inputs),
            'modality_counts': self.count_modalities(),
            'plot_info': self.plot_info,
        }


class MultimodalAligner:
    """
    Aligns and loads multimodal data for a specific task.

    This class:
    1. Loads target labels from targets.parquet
    2. Loads input links from inputs_index.parquet
    3. Retrieves actual multimodal data from payload database
    4. Joins with metadata (plots, assets, provenance)
    5. Returns aligned multimodal samples
    """

    def __init__(
        self,
        task_name: str,
        data_root: Optional[Path] = None,
        lazy_load: bool = True
    ):
        """
        Initialize MultimodalAligner.

        Args:
            task_name: Task name (e.g., 'NDVI')
            data_root: Root directory of INVITA data
            lazy_load: If True, only load asset metadata, not actual payload data
        """
        self.task_name = task_name
        self.lazy_load = lazy_load

        # Initialize loaders
        self.parquet_reader = ParquetReader(data_root=data_root)
        self.payload_loader = PayloadLoader()

        # Load task data
        logger.info(f"Initializing MultimodalAligner for {task_name}")
        self.targets, self.inputs_index = self.parquet_reader.load_task(task_name)
        self.assets = self.parquet_reader.load_assets()
        self.plots = self.parquet_reader.load_plots()

        # Index for fast lookup
        self._inputs_by_target = self.inputs_index.groupby('target_uid')
        self._targets_index = self.targets.set_index('target_uid')
        self._plots_index = self.plots.set_index('plot_uid')
        self._assets_index = self.assets.set_index('asset_uid')

        logger.info(f"Loaded {len(self.targets):,} targets with {len(self.inputs_index):,} input links")

    def get_sample(
        self,
        target_uid: str,
        load_payload: bool = None
    ) -> Optional[MultimodalSample]:
        """
        Get a single aligned multimodal sample.

        Args:
            target_uid: Target unique identifier
            load_payload: Whether to load actual payload data (overrides lazy_load)

        Returns:
            MultimodalSample or None if not found
        """
        if load_payload is None:
            load_payload = not self.lazy_load

        # Get target data
        if target_uid not in self._targets_index.index:
            logger.warning(f"Target not found: {target_uid}")
            return None

        target_row = self._targets_index.loc[target_uid]

        # Determine target value column
        target_value = self._extract_target_value(target_row)

        # Get target metadata
        target_metadata = target_row.to_dict()

        # Get plot info
        plot_uid = target_row.get('plot_uid')
        plot_info = None
        if plot_uid and plot_uid in self._plots_index.index:
            plot_info = self._plots_index.loc[plot_uid].to_dict()

        # Get inputs for this target
        inputs = []
        if target_uid in self._inputs_by_target.groups:
            input_rows = self._inputs_by_target.get_group(target_uid)

            for _, input_row in input_rows.iterrows():
                asset_uid = input_row['asset_uid']

                # Get asset metadata
                asset_info = None
                if asset_uid in self._assets_index.index:
                    asset_info = self._assets_index.loc[asset_uid].to_dict()

                input_data = {
                    'asset_uid': asset_uid,
                    'input_metadata': input_row.to_dict(),
                    'asset_info': asset_info,
                    'modality': asset_info.get('modality_verified') if asset_info else 'unknown',
                    'kind': asset_info.get('kind') if asset_info else 'unknown',
                }

                # Optionally load actual payload data
                if load_payload:
                    try:
                        payload = self.payload_loader.extract_asset(asset_uid, return_type='auto')
                        input_data['payload'] = payload
                    except Exception as e:
                        logger.error(f"Failed to load payload for {asset_uid}: {e}")
                        input_data['payload'] = None

                inputs.append(input_data)

        return MultimodalSample(
            target_uid=target_uid,
            target_value=target_value,
            target_metadata=target_metadata,
            inputs=inputs,
            plot_info=plot_info
        )

    def get_batch(
        self,
        target_uids: List[str],
        load_payload: bool = None
    ) -> List[MultimodalSample]:
        """
        Get multiple aligned samples.

        Args:
            target_uids: List of target UIDs
            load_payload: Whether to load actual payload data

        Returns:
            List of MultimodalSample objects
        """
        samples = []
        for target_uid in target_uids:
            sample = self.get_sample(target_uid, load_payload=load_payload)
            if sample:
                samples.append(sample)
        return samples

    def iter_samples(
        self,
        batch_size: int = 1,
        load_payload: bool = None,
        filter_fn: Optional[callable] = None,
        shuffle: bool = False,
        limit: Optional[int] = None
    ) -> Iterator[List[MultimodalSample]]:
        """
        Iterate through all samples in batches.

        Args:
            batch_size: Number of samples per batch
            load_payload: Whether to load actual payload data
            filter_fn: Optional filter function(sample) -> bool
            shuffle: Whether to shuffle targets before iteration
            limit: Maximum number of samples to yield

        Yields:
            Batches of MultimodalSample objects
        """
        target_uids = self.targets['target_uid'].tolist()

        if shuffle:
            import random
            random.shuffle(target_uids)

        if limit:
            target_uids = target_uids[:limit]

        batch = []
        total_yielded = 0

        for target_uid in target_uids:
            if limit and total_yielded >= limit:
                break

            sample = self.get_sample(target_uid, load_payload=load_payload)

            if sample is None:
                continue

            # Apply filter
            if filter_fn and not filter_fn(sample):
                continue

            batch.append(sample)

            if len(batch) >= batch_size:
                yield batch
                total_yielded += len(batch)
                batch = []

        # Yield remaining
        if batch:
            yield batch

    def get_split_samples(
        self,
        split: str,
        load_payload: bool = None
    ) -> List[MultimodalSample]:
        """
        Get all samples for a specific split (train/val/test).

        Args:
            split: Split name ('train', 'val', 'test')
            load_payload: Whether to load actual payload data

        Returns:
            List of samples for the split
        """
        # Load split info
        split_path = self.parquet_reader.tasks_dir / self.task_name / "splits" / f"{split}.parquet"

        if not split_path.exists():
            logger.warning(f"Split file not found: {split_path}")
            return []

        split_df = pd.read_parquet(split_path)
        target_uids = split_df['target_uid'].tolist()

        return self.get_batch(target_uids, load_payload=load_payload)

    def _extract_target_value(self, target_row: pd.Series) -> Any:
        """Extract the target value from a target row."""
        # Standard INVITA target value columns
        if 'target_value_num' in target_row.index:
            val = target_row['target_value_num']
            # Return numeric value if not NaN/None
            if pd.notna(val):
                return val

        if 'target_value_cat' in target_row.index:
            val = target_row['target_value_cat']
            # Return categorical value if not empty/None
            if pd.notna(val) and val != '':
                return val

        # Try target_name column for task-specific values
        if 'target_name' in target_row.index:
            target_name = target_row['target_name']
            if target_name in target_row.index:
                return target_row[target_name]

        # Fallback: return the whole row as dict
        logger.warning(f"Could not determine target value column for task {self.task_name}")
        return target_row.to_dict()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get alignment statistics.

        Returns:
            Dict with statistics about the aligned data
        """
        stats = {
            'task_name': self.task_name,
            'num_targets': len(self.targets),
            'num_input_links': len(self.inputs_index),
            'num_unique_assets': self.inputs_index['asset_uid'].nunique(),
        }

        # Coverage statistics
        targets_with_inputs = self.inputs_index['target_uid'].nunique()
        stats['targets_with_inputs'] = targets_with_inputs
        stats['targets_without_inputs'] = len(self.targets) - targets_with_inputs
        stats['coverage_rate'] = targets_with_inputs / len(self.targets) if len(self.targets) > 0 else 0

        # Inputs per target statistics
        inputs_per_target = self.inputs_index.groupby('target_uid').size()
        stats['avg_inputs_per_target'] = float(inputs_per_target.mean())
        stats['median_inputs_per_target'] = float(inputs_per_target.median())
        stats['max_inputs_per_target'] = int(inputs_per_target.max())
        stats['min_inputs_per_target'] = int(inputs_per_target.min())

        # Modality distribution
        inputs_with_assets = self.inputs_index.merge(
            self.assets[['asset_uid', 'modality_verified']],
            on='asset_uid',
            how='left'
        )
        stats['modality_distribution'] = inputs_with_assets['modality_verified'].value_counts().to_dict()

        return stats

    def close(self):
        """Close payload loader connection."""
        self.payload_loader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def test_multimodal_aligner():
    """Test the MultimodalAligner functionality."""
    print("Testing MultimodalAligner...")

    # Test with NDVI
    with MultimodalAligner(task_name='NDVI', lazy_load=True) as aligner:
        # Test 1: Get statistics
        print("\n=== Alignment Statistics ===")
        stats = aligner.get_stats()
        print(f"Task: {stats['task_name']}")
        print(f"Targets: {stats['num_targets']:,}")
        print(f"Input links: {stats['num_input_links']:,}")
        print(f"Coverage rate: {stats['coverage_rate']:.2%}")
        print(f"Avg inputs/target: {stats['avg_inputs_per_target']:.1f}")
        print(f"\nModality distribution:")
        for modality, count in list(stats['modality_distribution'].items())[:5]:
            print(f"  {modality}: {count:,}")

        # Test 2: Get a single sample (lazy load)
        print("\n=== Single Sample (Lazy Load) ===")
        sample_uid = aligner.targets.iloc[0]['target_uid']
        sample = aligner.get_sample(sample_uid, load_payload=False)

        if sample:
            print(f"Target UID: {sample.target_uid}")
            print(f"Target value: {sample.target_value}")
            print(f"Number of inputs: {len(sample.inputs)}")
            print(f"Modality counts: {sample.count_modalities()}")

        # Test 3: Iterate through a few samples
        print("\n=== Batch Iteration (First 3 batches) ===")
        for i, batch in enumerate(aligner.iter_samples(batch_size=10, load_payload=False, limit=30)):
            print(f"Batch {i+1}: {len(batch)} samples")
            if batch:
                first_sample = batch[0]
                print(f"  First sample modalities: {first_sample.count_modalities()}")

        # Test 4: Load a sample with actual payload
        print("\n=== Sample with Payload (First target) ===")
        sample_with_data = aligner.get_sample(sample_uid, load_payload=True)
        if sample_with_data and sample_with_data.inputs:
            first_input = sample_with_data.inputs[0]
            print(f"First input modality: {first_input['modality']}")
            print(f"First input kind: {first_input['kind']}")
            payload = first_input.get('payload')
            if payload is not None:
                if isinstance(payload, pd.DataFrame):
                    print(f"Loaded DataFrame with {len(payload)} rows")
                elif isinstance(payload, pd.Series):
                    print(f"Loaded Series with {len(payload)} items")
                elif isinstance(payload, bytes):
                    print(f"Loaded {len(payload)} bytes")
                else:
                    print(f"Loaded payload of type: {type(payload)}")


if __name__ == "__main__":
    # Run tests
    test_multimodal_aligner()
