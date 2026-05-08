"""
Payload loader for governed field-trial assets

Extracts files from the governed SQLite payload database.
This is a critical component for accessing the 39GB multimodal asset collection.

Usage:
    from data.loaders.payload_loader import PayloadLoader

    loader = PayloadLoader()

    # Extract a single asset
    data = loader.extract_asset(asset_uid='some_asset_uid')

    # Extract multiple assets
    assets = loader.extract_batch(['uid1', 'uid2', 'uid3'])
"""

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import io
import json
import logging

from src.benchmarks.paths import default_data_root

logger = logging.getLogger(__name__)


class PayloadLoader:
    """
    Loads data from the SQLite payload database.

    The payload database stores all multimodal assets:
    - Satellite images (GeoTIFF)
    - UAV images and products
    - CSV sensor readings (GreenSeeker, UAV-MS, etc.)
    - Field camera images
    - Biological surveys
    """

    DEFAULT_PAYLOAD_PATH = default_data_root() / "payload" / "payload.db"
    DEFAULT_LOCATOR_PATH = default_data_root() / "payload" / "asset_locator.parquet"

    def __init__(
        self,
        payload_db_path: Optional[Union[str, Path]] = None,
        asset_locator_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize PayloadLoader.

        Args:
            payload_db_path: Optional path to payload.db.
            asset_locator_path: Optional path to asset_locator.parquet.
        """
        self.payload_db_path = Path(payload_db_path) if payload_db_path else self.DEFAULT_PAYLOAD_PATH
        self.asset_locator_path = Path(asset_locator_path) if asset_locator_path else self.DEFAULT_LOCATOR_PATH

        # Validate paths
        if not self.payload_db_path.exists():
            raise FileNotFoundError(f"Payload database not found: {self.payload_db_path}")
        if not self.asset_locator_path.exists():
            raise FileNotFoundError(f"Asset locator not found: {self.asset_locator_path}")

        # Load asset locator index
        logger.info(f"Loading asset locator from {self.asset_locator_path}")
        self.asset_locator = pd.read_parquet(self.asset_locator_path)
        logger.info(f"Loaded {len(self.asset_locator)} asset records")

        # Cache for database connections (thread-local would be better for multi-threading)
        self._conn = None

    def _get_connection(self) -> sqlite3.Connection:
        """Get or create SQLite connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.payload_db_path))
        return self._conn

    def close(self):
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def get_asset_info(self, asset_uid: str) -> Optional[Dict[str, Any]]:
        """
        Get asset metadata from locator index.

        Args:
            asset_uid: Asset unique identifier

        Returns:
            Dict with asset metadata or None if not found
        """
        matches = self.asset_locator[self.asset_locator['asset_uid'] == asset_uid]
        if len(matches) == 0:
            logger.warning(f"Asset not found in locator: {asset_uid}")
            return None
        return matches.iloc[0].to_dict()

    def _reconstruct_file(self, rel_path: str) -> Optional[bytes]:
        """
        Reconstruct a file from chunks in the database.

        Args:
            rel_path: Relative path in the files table

        Returns:
            Complete file bytes or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        # Get file metadata
        cursor.execute(
            "SELECT chunk_count FROM files WHERE rel_path = ?",
            (rel_path,)
        )
        file_info = cursor.fetchone()

        if file_info is None:
            logger.warning(f"File not found in database: {rel_path}")
            return None

        chunk_count = file_info[0]

        # Fetch all chunks in order
        cursor.execute(
            """
            SELECT chunk_index, data, data_compression, data_uncompressed_size
            FROM file_chunks
            WHERE rel_path = ?
            ORDER BY chunk_index
            """,
            (rel_path,)
        )
        chunks = cursor.fetchall()

        if len(chunks) != chunk_count:
            logger.error(f"Chunk count mismatch for {rel_path}: expected {chunk_count}, got {len(chunks)}")
            return None

        # Reconstruct file from chunks
        file_bytes = b''
        for chunk_index, data, compression, uncompressed_size in chunks:
            if compression == 'zlib':
                import zlib
                chunk_data = zlib.decompress(data)
            elif compression == 'zstd':
                import zstandard as zstd
                dctx = zstd.ZstdDecompressor()
                chunk_data = dctx.decompress(data, max_output_size=uncompressed_size)
            elif compression == 'none' or compression is None:
                chunk_data = data
            else:
                logger.error(f"Unknown compression: {compression}")
                return None

            file_bytes += chunk_data

        return file_bytes

    def extract_asset(
        self,
        asset_uid: str,
        return_type: str = 'auto'
    ) -> Optional[Any]:
        """
        Extract a single asset from payload database.

        Args:
            asset_uid: Asset unique identifier
            return_type: How to return the data
                - 'auto': Automatically detect based on kind
                - 'bytes': Return raw bytes
                - 'dataframe': Parse as DataFrame (for CSV)
                - 'series': Return single row as Series (for csv_row)
                - 'geotiff': Return rasterio-compatible data
                - 'image': Return PIL Image

        Returns:
            Extracted data in requested format, or None if not found
        """
        # Get asset info from locator
        asset_info = self.get_asset_info(asset_uid)
        if asset_info is None:
            return None

        payload_rel_path = asset_info['payload_rel_path']
        kind = asset_info['kind']
        csv_row_index = asset_info.get('csv_row_index')

        # Reconstruct file from chunks
        file_bytes = self._reconstruct_file(payload_rel_path)
        if file_bytes is None:
            return None

        # For csv_row kind, extract specific row
        if kind == 'csv_row' and csv_row_index is not None:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes))
                if csv_row_index < len(df):
                    row = df.iloc[csv_row_index]
                    if return_type == 'auto' or return_type == 'series':
                        return row
                    elif return_type == 'dataframe':
                        return df.iloc[[csv_row_index]]
                    else:
                        return row
                else:
                    logger.error(f"Row index {csv_row_index} out of range for {payload_rel_path}")
                    return None
            except Exception as e:
                logger.error(f"Failed to extract CSV row for {asset_uid}: {e}")
                return None

        # Determine return type for regular files
        if return_type == 'bytes':
            return file_bytes

        # Infer file type from path
        file_ext = payload_rel_path.split('.')[-1].lower()

        if return_type == 'auto':
            # Auto-detect based on file extension
            if file_ext in ['csv']:
                return_type = 'dataframe'
            elif file_ext in ['tif', 'tiff']:
                return_type = 'geotiff'
            elif file_ext in ['jpg', 'jpeg', 'png']:
                return_type = 'image'
            else:
                return_type = 'bytes'

        # Convert based on return_type
        if return_type == 'dataframe':
            # Parse CSV
            try:
                df = pd.read_csv(io.BytesIO(file_bytes))
                return df
            except Exception as e:
                logger.error(f"Failed to parse CSV for {asset_uid}: {e}")
                return file_bytes

        elif return_type == 'geotiff':
            # Return bytes for now - rasterio will handle it
            # User can use: with rasterio.open(io.BytesIO(data)) as src: ...
            return file_bytes

        elif return_type == 'image':
            # Parse as PIL Image
            try:
                from PIL import Image
                img = Image.open(io.BytesIO(file_bytes))
                return img
            except Exception as e:
                logger.error(f"Failed to parse image for {asset_uid}: {e}")
                return file_bytes

        return file_bytes

    def extract_batch(
        self,
        asset_uids: List[str],
        return_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Extract multiple assets in batch.

        Args:
            asset_uids: List of asset UIDs
            return_type: How to return data (same as extract_asset)

        Returns:
            Dict mapping asset_uid -> extracted data
        """
        results = {}
        for asset_uid in asset_uids:
            try:
                data = self.extract_asset(asset_uid, return_type=return_type)
                if data is not None:
                    results[asset_uid] = data
            except Exception as e:
                logger.error(f"Failed to extract {asset_uid}: {e}")
        return results

    def extract_by_kind(
        self,
        kind: str,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract all assets of a specific kind.

        Args:
            kind: Asset kind to filter (e.g., 'file', 'csv_row')
            limit: Maximum number of assets to extract

        Returns:
            Dict mapping asset_uid -> extracted data
        """
        # Filter asset locator by kind
        filtered = self.asset_locator[self.asset_locator['kind'] == kind]

        if limit is not None:
            filtered = filtered.head(limit)

        asset_uids = filtered['asset_uid'].tolist()
        logger.info(f"Extracting {len(asset_uids)} assets of kind '{kind}'")

        return self.extract_batch(asset_uids)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get payload database statistics.

        Returns:
            Dict with statistics about the payload
        """
        stats = {
            'total_assets': len(self.asset_locator),
            'asset_kinds': self.asset_locator['kind'].value_counts().to_dict(),
            'db_size_bytes': self.payload_db_path.stat().st_size,
            'db_size_gb': self.payload_db_path.stat().st_size / (1024**3),
        }
        return stats

    def list_tables(self) -> List[str]:
        """List all tables in the payload database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        return tables

    def inspect_table_schema(self, table_name: str = 'files') -> List[tuple]:
        """
        Inspect table schema.

        Args:
            table_name: Name of the table to inspect

        Returns:
            List of (column_id, name, type, not_null, default_value, primary_key)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        schema = cursor.fetchall()
        return schema


def test_payload_loader():
    """Test the PayloadLoader functionality."""
    print("Testing PayloadLoader...")

    with PayloadLoader() as loader:
        # Test 1: Get stats
        print("\n=== Payload Statistics ===")
        stats = loader.get_stats()
        print(f"Total assets: {stats['total_assets']:,}")
        print(f"Database size: {stats['db_size_gb']:.2f} GB")
        print(f"\nAsset kinds:")
        for kind, count in stats['asset_kinds'].items():
            print(f"  {kind}: {count:,}")

        # Test 2: List tables
        print("\n=== Database Tables ===")
        tables = loader.list_tables()
        print(f"Tables: {tables}")

        # Test 3: Inspect schema
        print("\n=== Table Schema (files) ===")
        schema = loader.inspect_table_schema('files')
        for col in schema:
            print(f"  {col[1]} ({col[2]})")

        print("\n=== Table Schema (file_chunks) ===")
        schema = loader.inspect_table_schema('file_chunks')
        for col in schema:
            print(f"  {col[1]} ({col[2]})")

        # Test 4: Extract a sample asset
        print("\n=== Sample Asset Extraction ===")
        # Get first asset_uid
        sample_uid = loader.asset_locator.iloc[0]['asset_uid']
        print(f"Extracting asset: {sample_uid}")

        asset_info = loader.get_asset_info(sample_uid)
        print(f"Asset info: {asset_info}")

        data = loader.extract_asset(sample_uid, return_type='bytes')
        if data:
            print(f"Extracted {len(data)} bytes")
        else:
            print("Failed to extract asset")


if __name__ == "__main__":
    # Run tests
    test_payload_loader()
