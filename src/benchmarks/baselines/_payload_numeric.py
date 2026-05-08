"""Payload-backed numeric observation extraction for sensor-history baselines."""

from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from src.data_processing.loaders.payload_loader import PayloadLoader

NUMERIC_OBSERVATION_COLUMNS = (
    "asset_uid",
    "plot_uid",
    "observation_date",
    "metric",
    "value",
    "modality_verified",
    "payload_rel_path",
)

SENSOR_NUMERIC_MODALITIES = (
    "greenseeker_handheld",
    "uav_multispectral_l2_inversion",
)

UAV_METRIC_COLUMNS = {
    "ala": "ALA",
    "cab": "Cab",
    "fcover": "FCOVER",
    "fipar": "FIPAR",
    "lai": "LAI",
    "laicab": "LAICab",
}


def load_payload_numeric_observations(
    data_root: Path,
    *,
    asset_uids: pd.Series | list[str] | set[str] | None = None,
    modalities: tuple[str, ...] = SENSOR_NUMERIC_MODALITIES,
) -> pd.DataFrame:
    """
    Extract numeric sensor observations from payload files.

    This function reads true payload contents. It does not use task target
    values as features. Asset metadata is used only for row lookup, plot
    identity, acquisition date, and metric naming.
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
        columns=[
            "asset_uid",
            "source_asset_id",
            "modality_verified",
            "acquisition_date",
            "plot_uid",
            "metric",
            "relative_path",
        ],
    )
    locator = pd.read_parquet(
        locator_path,
        columns=["asset_uid", "kind", "payload_rel_path", "modality_verified"],
    ).rename(columns={"modality_verified": "locator_modality"})
    frame = assets.merge(locator, on="asset_uid", how="inner", validate="one_to_one")

    if asset_uids is not None:
        uid_set = set(map(str, asset_uids))
        frame = frame.loc[frame["asset_uid"].isin(uid_set)].copy()
    frame = frame.loc[frame["modality_verified"].isin(modalities)].copy()
    if frame.empty:
        return _empty_observations()

    rows: list[pd.DataFrame] = []
    with PayloadLoader(
        payload_db_path=payload_path,
        asset_locator_path=locator_path,
    ) as loader:
        for payload_rel_path, group in frame.groupby("payload_rel_path", sort=False):
            modality = str(group["modality_verified"].iloc[0])
            file_bytes = loader._reconstruct_file(str(payload_rel_path))
            if file_bytes is None:
                continue
            if modality == "greenseeker_handheld":
                extracted = _extract_greenseeker(group, file_bytes)
            elif modality == "uav_multispectral_l2_inversion":
                extracted = _extract_uav_ms(group, file_bytes)
            else:
                continue
            if not extracted.empty:
                rows.append(extracted)

    if not rows:
        return _empty_observations()

    output = pd.concat(rows, ignore_index=True)
    output["observation_date"] = pd.to_datetime(
        output["observation_date"], errors="coerce"
    )
    output["value"] = pd.to_numeric(output["value"], errors="coerce")
    output = output.dropna(subset=["observation_date", "value"])
    output["metric"] = output["metric"].map(_safe_metric_name)
    return output[list(NUMERIC_OBSERVATION_COLUMNS)].reset_index(drop=True)


def _extract_greenseeker(asset_rows: pd.DataFrame, file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    if "NDVI" not in df.columns:
        return _empty_observations()

    rows = asset_rows.copy()
    rows["_source_row"] = pd.to_numeric(rows["source_asset_id"], errors="coerce")
    rows = rows.dropna(subset=["_source_row"])
    rows["_source_row"] = rows["_source_row"].astype(int)
    rows = rows.loc[rows["_source_row"].between(0, len(df) - 1)].copy()
    if rows.empty:
        return _empty_observations()

    values = df.iloc[rows["_source_row"].to_numpy()]["NDVI"].to_numpy()
    return pd.DataFrame(
        {
            "asset_uid": rows["asset_uid"].to_numpy(),
            "plot_uid": rows["plot_uid"].to_numpy(),
            "observation_date": rows["acquisition_date"].to_numpy(),
            "metric": rows["metric"].fillna("NDVI").to_numpy(),
            "value": values,
            "modality_verified": rows["modality_verified"].to_numpy(),
            "payload_rel_path": rows["payload_rel_path"].to_numpy(),
        }
    )


def _extract_uav_ms(asset_rows: pd.DataFrame, file_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(file_bytes), sep=";")
    required = {"plot_id", *UAV_METRIC_COLUMNS.values()}
    if not required.issubset(df.columns):
        return _empty_observations()

    values = df.copy()
    values["_plot_token"] = values["plot_id"].map(_plot_token_from_uav_plot_id)
    value_by_plot = values.dropna(subset=["_plot_token"]).set_index("_plot_token")

    output_rows: list[dict] = []
    for row in asset_rows.itertuples(index=False):
        metric = _safe_metric_name(row.metric)
        metric_col = UAV_METRIC_COLUMNS.get(metric)
        plot_token = _plot_token_from_plot_uid(row.plot_uid)
        if metric_col is None or plot_token not in value_by_plot.index:
            continue
        match = value_by_plot.loc[plot_token]
        if isinstance(match, pd.DataFrame):
            match = match.iloc[0]
        output_rows.append(
            {
                "asset_uid": row.asset_uid,
                "plot_uid": row.plot_uid,
                "observation_date": row.acquisition_date,
                "metric": metric,
                "value": match[metric_col],
                "modality_verified": row.modality_verified,
                "payload_rel_path": row.payload_rel_path,
            }
        )

    if not output_rows:
        return _empty_observations()
    return pd.DataFrame(output_rows)


def _plot_token_from_plot_uid(plot_uid: object) -> str | None:
    match = re.search(r"_(\d+)_(\d+)$", str(plot_uid))
    if not match:
        return None
    return f"x{int(match.group(2))}_y{int(match.group(1))}"


def _plot_token_from_uav_plot_id(plot_id: object) -> str | None:
    match = re.search(r"_x(\d+)_y(\d+)$", str(plot_id).lower())
    if not match:
        return None
    return f"x{int(match.group(1))}_y{int(match.group(2))}"


def _safe_metric_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _empty_observations() -> pd.DataFrame:
    return pd.DataFrame(columns=list(NUMERIC_OBSERVATION_COLUMNS))
