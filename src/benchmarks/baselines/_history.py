"""Leakage-controlled history feature builders for temporal baselines."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.benchmarks.baselines._shared import normalize_categorical

DEFAULT_HISTORY_WINDOWS = (14, 30, 60)
DEFAULT_SEQUENCE_BINS = 12
DEFAULT_SEQUENCE_HORIZON_DAYS = 60


def build_history_features(
    targets: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    prefix: str,
    windows: tuple[int, ...] = DEFAULT_HISTORY_WINDOWS,
    sequence_bins: int = DEFAULT_SEQUENCE_BINS,
    sequence_horizon_days: int = DEFAULT_SEQUENCE_HORIZON_DAYS,
) -> pd.DataFrame:
    """Build rolling summary and fixed-bin sequence features for target rows."""

    required_targets = {"target_uid", "plot_uid", "target_date", "source_asset_uid"}
    missing_targets = required_targets - set(targets.columns)
    if missing_targets:
        raise ValueError(f"Targets missing history columns: {sorted(missing_targets)}")

    required_obs = {"plot_uid", "observation_date", "metric", "value", "asset_uid"}
    missing_obs = required_obs - set(observations.columns)
    if missing_obs:
        raise ValueError(f"Observations missing columns: {sorted(missing_obs)}")

    output = pd.DataFrame({"target_uid": targets["target_uid"].to_numpy()})
    if observations.empty:
        output[f"{prefix}_has_history"] = 0.0
        output[f"{prefix}_n_observations"] = 0.0
        return output

    target_frame = targets[
        ["target_uid", "plot_uid", "target_date", "source_asset_uid"]
    ].copy()
    target_frame["plot_uid"] = normalize_categorical(target_frame["plot_uid"])
    target_frame["source_asset_uid"] = normalize_categorical(
        target_frame["source_asset_uid"]
    )
    target_frame["target_date"] = pd.to_datetime(
        target_frame["target_date"], errors="raise"
    )

    obs = observations.copy()
    obs["plot_uid"] = normalize_categorical(obs["plot_uid"])
    obs["asset_uid"] = normalize_categorical(obs["asset_uid"])
    obs["metric"] = normalize_categorical(obs["metric"]).map(_safe_name)
    obs["observation_date"] = pd.to_datetime(obs["observation_date"], errors="raise")
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    obs = obs.dropna(subset=["value"]).copy()

    metrics = sorted(obs["metric"].unique().tolist())
    feature_store: dict[str, np.ndarray] = {}
    n_rows = len(target_frame)

    for metric in metrics:
        safe_metric = _safe_name(metric)
        for window in windows:
            for stat in ("count", "mean", "std", "min", "max", "last", "slope"):
                feature_store[f"{prefix}_{safe_metric}_{window}d_{stat}"] = np.full(
                    n_rows, np.nan, dtype=float
                )
        feature_store[f"{prefix}_{safe_metric}_days_since_last"] = np.full(
            n_rows, np.nan, dtype=float
        )
        for bin_idx in range(sequence_bins):
            feature_store[f"{prefix}_{safe_metric}_seq_{bin_idx:02d}_mean"] = np.full(
                n_rows, np.nan, dtype=float
            )
            feature_store[f"{prefix}_{safe_metric}_seq_{bin_idx:02d}_count"] = np.zeros(
                n_rows, dtype=float
            )

    obs_by_plot_metric = {
        key: group.sort_values("observation_date")
        for key, group in obs.groupby(["plot_uid", "metric"], sort=False)
    }

    for plot_uid, target_group in target_frame.groupby("plot_uid", sort=False):
        target_indices = target_group.index.to_numpy()
        target_dates = target_group["target_date"].to_numpy(dtype="datetime64[ns]")
        source_assets = target_group["source_asset_uid"].to_numpy()

        for metric in metrics:
            group = obs_by_plot_metric.get((plot_uid, metric))
            if group is None or group.empty:
                continue

            obs_dates = group["observation_date"].to_numpy(dtype="datetime64[ns]")
            values = group["value"].to_numpy(dtype=float)
            asset_uids = group["asset_uid"].to_numpy()
            safe_metric = _safe_name(metric)

            for local_pos, row_index in enumerate(target_indices):
                target_date = target_dates[local_pos]
                source_asset = source_assets[local_pos]
                right = np.searchsorted(obs_dates, target_date, side="left")
                if right == 0:
                    continue

                valid_mask = asset_uids[:right] != source_asset
                valid_dates = obs_dates[:right][valid_mask]
                valid_values = values[:right][valid_mask]
                if len(valid_values) == 0:
                    continue

                days_since = _timedelta_days(target_date - valid_dates[-1])
                feature_store[f"{prefix}_{safe_metric}_days_since_last"][row_index] = (
                    days_since
                )

                for window in windows:
                    cutoff = target_date - np.timedelta64(window, "D")
                    left = np.searchsorted(valid_dates, cutoff, side="left")
                    window_values = valid_values[left:]
                    window_dates = valid_dates[left:]
                    _write_window_stats(
                        feature_store,
                        prefix=prefix,
                        metric=safe_metric,
                        window=window,
                        row_index=row_index,
                        dates=window_dates,
                        values=window_values,
                    )

                sequence_cutoff = target_date - np.timedelta64(
                    sequence_horizon_days, "D"
                )
                seq_left = np.searchsorted(valid_dates, sequence_cutoff, side="left")
                seq_dates = valid_dates[seq_left:]
                seq_values = valid_values[seq_left:]
                _write_sequence_bins(
                    feature_store,
                    prefix=prefix,
                    metric=safe_metric,
                    row_index=row_index,
                    target_date=target_date,
                    dates=seq_dates,
                    values=seq_values,
                    sequence_bins=sequence_bins,
                    horizon_days=sequence_horizon_days,
                )

    feature_frame = pd.DataFrame(feature_store, index=output.index)
    output = pd.concat([output, feature_frame], axis=1)

    count_cols = [col for col in feature_frame.columns if col.endswith("_count")]
    if count_cols:
        output[f"{prefix}_n_observations"] = feature_frame[count_cols].sum(axis=1)
    else:
        output[f"{prefix}_n_observations"] = 0.0
    output[f"{prefix}_has_history"] = (output[f"{prefix}_n_observations"] > 0).astype(
        float
    )
    return output.copy()


def build_input_history_features(
    targets: pd.DataFrame,
    inputs: pd.DataFrame,
    observations: pd.DataFrame,
    *,
    prefix: str,
    windows: tuple[int, ...] = DEFAULT_HISTORY_WINDOWS,
    sequence_bins: int = DEFAULT_SEQUENCE_BINS,
    sequence_horizon_days: int = DEFAULT_SEQUENCE_HORIZON_DAYS,
    modality_pattern: str | None = None,
) -> pd.DataFrame:
    """
    Build target-specific history features from legal input assets only.

    Every historical observation must be linked through the target's own
    inputs_index row, precede the target timestamp, belong to the same plot,
    and differ from the current source asset.
    """

    required_targets = {"target_uid", "plot_uid", "target_date", "source_asset_uid"}
    missing_targets = required_targets - set(targets.columns)
    if missing_targets:
        raise ValueError(f"Targets missing history columns: {sorted(missing_targets)}")

    required_inputs = {
        "target_uid",
        "asset_uid",
        "modality_verified",
        "acquisition_date",
    }
    missing_inputs = required_inputs - set(inputs.columns)
    if missing_inputs:
        raise ValueError(f"Inputs missing history columns: {sorted(missing_inputs)}")

    required_obs = {"plot_uid", "observation_date", "metric", "value", "asset_uid"}
    missing_obs = required_obs - set(observations.columns)
    if missing_obs:
        raise ValueError(f"Observations missing columns: {sorted(missing_obs)}")

    target_frame = targets[
        ["target_uid", "plot_uid", "target_date", "source_asset_uid"]
    ].copy()
    target_frame["plot_uid"] = normalize_categorical(target_frame["plot_uid"])
    target_frame["source_asset_uid"] = normalize_categorical(
        target_frame["source_asset_uid"]
    )
    target_frame["target_date"] = pd.to_datetime(
        target_frame["target_date"], errors="raise"
    )
    output = pd.DataFrame({"target_uid": target_frame["target_uid"].to_numpy()})

    if observations.empty:
        return _finalize_history_output(output, {}, prefix)

    input_frame = inputs[list(required_inputs)].copy()
    input_frame["asset_uid"] = normalize_categorical(input_frame["asset_uid"])
    input_frame["modality_verified"] = normalize_categorical(
        input_frame["modality_verified"]
    )
    input_frame["acquisition_date"] = pd.to_datetime(
        input_frame["acquisition_date"], errors="coerce"
    )
    if modality_pattern:
        pattern = re.compile(modality_pattern, flags=re.IGNORECASE)
        input_frame = input_frame.loc[
            input_frame["modality_verified"].map(
                lambda value: bool(pattern.search(str(value)))
            )
        ].copy()

    legal_inputs = input_frame.merge(
        target_frame,
        on="target_uid",
        how="inner",
        validate="many_to_one",
    )
    legal_inputs = legal_inputs.loc[
        legal_inputs["acquisition_date"].notna()
        & legal_inputs["acquisition_date"].lt(legal_inputs["target_date"])
        & legal_inputs["asset_uid"].ne(legal_inputs["source_asset_uid"])
    ].copy()
    if legal_inputs.empty:
        return _finalize_history_output(output, {}, prefix)

    obs = observations.copy()
    obs["plot_uid"] = normalize_categorical(obs["plot_uid"])
    obs["asset_uid"] = normalize_categorical(obs["asset_uid"])
    obs["metric"] = normalize_categorical(obs["metric"]).map(_safe_name)
    obs["observation_date"] = pd.to_datetime(obs["observation_date"], errors="raise")
    obs["value"] = pd.to_numeric(obs["value"], errors="coerce")
    obs = obs.dropna(subset=["value"]).copy()

    linked = legal_inputs.merge(
        obs,
        on="asset_uid",
        how="inner",
        suffixes=("_target", "_obs"),
    )
    linked = linked.loc[
        linked["plot_uid_target"].eq(linked["plot_uid_obs"])
        & linked["observation_date"].lt(linked["target_date"])
    ].copy()
    if linked.empty:
        return _finalize_history_output(output, {}, prefix)

    linked = linked.drop_duplicates(
        subset=["target_uid", "asset_uid", "metric", "observation_date"]
    )
    metrics = sorted(linked["metric"].unique().tolist())
    feature_store = _initialize_feature_store(
        prefix=prefix,
        metrics=metrics,
        n_rows=len(target_frame),
        windows=windows,
        sequence_bins=sequence_bins,
    )
    row_index = pd.Series(
        target_frame.index, index=target_frame["target_uid"]
    ).to_dict()

    for (target_uid, metric), group in linked.groupby(
        ["target_uid", "metric"], sort=False
    ):
        current_row = row_index.get(target_uid)
        if current_row is None:
            continue
        group = group.sort_values("observation_date")
        target_date = group["target_date"].iloc[0].to_datetime64()
        dates = group["observation_date"].to_numpy(dtype="datetime64[ns]")
        values = group["value"].to_numpy(dtype=float)
        safe_metric = _safe_name(metric)

        feature_store[f"{prefix}_{safe_metric}_days_since_last"][current_row] = (
            _timedelta_days(target_date - dates[-1])
        )
        for window in windows:
            cutoff = target_date - np.timedelta64(window, "D")
            left = np.searchsorted(dates, cutoff, side="left")
            _write_window_stats(
                feature_store,
                prefix=prefix,
                metric=safe_metric,
                window=window,
                row_index=current_row,
                dates=dates[left:],
                values=values[left:],
            )

        sequence_cutoff = target_date - np.timedelta64(sequence_horizon_days, "D")
        seq_left = np.searchsorted(dates, sequence_cutoff, side="left")
        _write_sequence_bins(
            feature_store,
            prefix=prefix,
            metric=safe_metric,
            row_index=current_row,
            target_date=target_date,
            dates=dates[seq_left:],
            values=values[seq_left:],
            sequence_bins=sequence_bins,
            horizon_days=sequence_horizon_days,
        )

    return _finalize_history_output(output, feature_store, prefix)


def history_feature_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    """Return numeric history feature columns for one prefix."""

    return sorted(col for col in frame.columns if col.startswith(f"{prefix}_"))


def sequence_feature_columns(frame: pd.DataFrame, prefix: str) -> list[str]:
    """Return fixed-bin sequence value/count columns for one prefix."""

    return sorted(
        col for col in frame.columns if f"{prefix}_" in col and "_seq_" in col
    )


def _initialize_feature_store(
    *,
    prefix: str,
    metrics: list[str],
    n_rows: int,
    windows: tuple[int, ...],
    sequence_bins: int,
) -> dict[str, np.ndarray]:
    feature_store: dict[str, np.ndarray] = {}
    for metric in metrics:
        safe_metric = _safe_name(metric)
        for window in windows:
            for stat in ("count", "mean", "std", "min", "max", "last", "slope"):
                feature_store[f"{prefix}_{safe_metric}_{window}d_{stat}"] = np.full(
                    n_rows, np.nan, dtype=float
                )
        feature_store[f"{prefix}_{safe_metric}_days_since_last"] = np.full(
            n_rows, np.nan, dtype=float
        )
        for bin_idx in range(sequence_bins):
            feature_store[f"{prefix}_{safe_metric}_seq_{bin_idx:02d}_mean"] = np.full(
                n_rows, np.nan, dtype=float
            )
            feature_store[f"{prefix}_{safe_metric}_seq_{bin_idx:02d}_count"] = np.zeros(
                n_rows, dtype=float
            )
    return feature_store


def _finalize_history_output(
    output: pd.DataFrame,
    feature_store: dict[str, np.ndarray],
    prefix: str,
) -> pd.DataFrame:
    if feature_store:
        feature_frame = pd.DataFrame(feature_store, index=output.index)
        output = pd.concat([output, feature_frame], axis=1)
        count_cols = [col for col in feature_frame.columns if col.endswith("_count")]
        output[f"{prefix}_n_observations"] = feature_frame[count_cols].sum(axis=1)
    else:
        output[f"{prefix}_n_observations"] = 0.0
    output[f"{prefix}_has_history"] = (output[f"{prefix}_n_observations"] > 0).astype(
        float
    )
    return output.copy()


def _write_window_stats(
    feature_store: dict[str, np.ndarray],
    *,
    prefix: str,
    metric: str,
    window: int,
    row_index: int,
    dates: np.ndarray,
    values: np.ndarray,
) -> None:
    base = f"{prefix}_{metric}_{window}d"
    if len(values) == 0:
        feature_store[f"{base}_count"][row_index] = 0.0
        return

    feature_store[f"{base}_count"][row_index] = float(len(values))
    feature_store[f"{base}_mean"][row_index] = float(np.mean(values))
    feature_store[f"{base}_std"][row_index] = (
        float(np.std(values)) if len(values) > 1 else 0.0
    )
    feature_store[f"{base}_min"][row_index] = float(np.min(values))
    feature_store[f"{base}_max"][row_index] = float(np.max(values))
    feature_store[f"{base}_last"][row_index] = float(values[-1])
    if len(values) >= 2:
        x = np.array([_timedelta_days(date - dates[0]) for date in dates], dtype=float)
        if np.var(x) > 0:
            y = values.astype(float)
            feature_store[f"{base}_slope"][row_index] = float(
                np.mean((x - x.mean()) * (y - y.mean())) / np.var(x)
            )
        else:
            feature_store[f"{base}_slope"][row_index] = 0.0
    else:
        feature_store[f"{base}_slope"][row_index] = 0.0


def _write_sequence_bins(
    feature_store: dict[str, np.ndarray],
    *,
    prefix: str,
    metric: str,
    row_index: int,
    target_date: np.datetime64,
    dates: np.ndarray,
    values: np.ndarray,
    sequence_bins: int,
    horizon_days: int,
) -> None:
    if len(values) == 0:
        return

    days_before = np.array([_timedelta_days(target_date - date) for date in dates])
    bin_width = horizon_days / sequence_bins
    # Bin 0 is oldest; the final bin is closest to target_date.
    bin_indices = np.floor((horizon_days - days_before) / bin_width).astype(int)
    bin_indices = np.clip(bin_indices, 0, sequence_bins - 1)
    for bin_idx in range(sequence_bins):
        mask = bin_indices == bin_idx
        if not mask.any():
            continue
        feature_store[f"{prefix}_{metric}_seq_{bin_idx:02d}_mean"][row_index] = float(
            np.mean(values[mask])
        )
        feature_store[f"{prefix}_{metric}_seq_{bin_idx:02d}_count"][row_index] = float(
            mask.sum()
        )


def _timedelta_days(delta: np.timedelta64) -> float:
    return float(delta / np.timedelta64(1, "D"))


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
