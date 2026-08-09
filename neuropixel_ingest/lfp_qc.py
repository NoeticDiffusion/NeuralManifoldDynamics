"""Streaming LFP quality control and depth-balanced channel selection."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .anatomy import read_lfp_series_channels
from .lfp_state import LfpStateContract


SELECTION_SCHEMA = "mndm.lfp_channel_selection.v1"


def run_lfp_channel_qc(
    nwb_path: str | Path,
    *,
    series_path: str,
    output_dir: str | Path,
    dataset_id: str,
    probe_id: str,
    depth_bin_width: float = 640.0,
    per_depth_target: int = 8,
    chunk_sec: float = 60.0,
    max_failed_fraction: float = 0.20,
    state_contract: LfpStateContract | None = None,
) -> tuple[Path, Path]:
    """Stream one LFP series, write QC parquet and a selection JSON artifact."""
    nwb_path = Path(nwb_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = read_lfp_series_channels(nwb_path, series_path)
    if geometry.empty:
        raise ValueError(f"No electrode mapping for {series_path!r}")
    metrics, sampling_rate, duration = _stream_metrics(
        nwb_path, series_path, chunk_sec, state_contract=state_contract
    )
    qc = geometry.merge(metrics, on="channel_index", how="left")
    qc["metadata_valid"] = qc["is_data_valid"].fillna(False).astype(bool)
    qc["finite_geometry"] = np.isfinite(pd.to_numeric(qc["probe_vertical_position"], errors="coerce"))
    qc["depth_bin"] = np.floor(
        pd.to_numeric(qc["probe_vertical_position"], errors="coerce") / float(depth_bin_width)
    ).astype("Int64")
    qc["qc_failed_fraction"] = qc["failed_windows"] / qc["n_windows"].clip(lower=1)
    qc["qc_status"] = np.where(
        qc["metadata_valid"]
        & qc["finite_geometry"]
        & (qc["qc_failed_fraction"] <= float(max_failed_fraction)),
        "good",
        "rejected",
    )
    qc["rejection_reason"] = qc.apply(_rejection_reason, axis=1)
    selected = select_depth_balanced(qc, per_depth_target=per_depth_target)
    qc["in_selection"] = qc["channel_index"].isin(selected["channel_index"])

    stem = nwb_path.stem
    qc_path = output_dir / f"{stem}_{probe_id}_lfp_channel_qc.parquet"
    qc.to_parquet(qc_path, index=False)
    selected.to_parquet(output_dir / f"{stem}_{probe_id}_selected_probe_geometry.parquet", index=False)
    selection_path = output_dir / f"{stem}_{probe_id}_lfp_channel_selection.json"
    artifact = {
        "schema": SELECTION_SCHEMA,
        "dataset_id": dataset_id,
        "nwb_path": str(nwb_path),
        "series_path": series_path,
        "probe_id": probe_id,
        "original_n_channels": int(len(qc)),
        "target_n_channels": int(len(selected)),
        "selection_policy": {
            "valid_mask_source": "electrodes.is_data_valid",
            "depth_column": "probe_vertical_position",
            "depth_bin_width": float(depth_bin_width),
            "per_depth_target": int(per_depth_target),
            "horizontal_balance": "two_best_per_column_then_depth_fallback",
            "max_failed_fraction": float(max_failed_fraction),
        },
        "selected_channels": _records(selected),
        "rejected_channels": _records(qc.loc[~qc["in_selection"]]),
        "ensemble_groups": _ensemble_groups(selected),
        "qc_windows": {
            "duration_sec": float(duration),
            "chunk_sec": float(chunk_sec),
            "n_chunks": int(qc["n_windows"].max()),
            "sampling_rate_hz": float(sampling_rate),
        },
        "state_qc": _state_qc_payload(qc, state_contract),
        "provenance": {"created_at": datetime.now(timezone.utc).isoformat()},
    }
    selection_path.write_text(json.dumps(artifact, indent=2, default=_json_default), encoding="utf-8")
    return selection_path, qc_path


def select_depth_balanced(qc: pd.DataFrame, *, per_depth_target: int) -> pd.DataFrame:
    """Select QC-passing channels evenly across depth and horizontal columns."""
    eligible = qc.loc[qc["qc_status"].eq("good")].copy()
    selected: list[pd.DataFrame] = []
    for _, depth_frame in eligible.groupby("depth_bin", dropna=True, sort=True):
        ranked = depth_frame.sort_values(["qc_score", "channel_index"], ascending=[False, True])
        picks: list[pd.DataFrame] = []
        for _, horizontal_frame in ranked.groupby("probe_horizontal_position", dropna=False, sort=True):
            picks.append(horizontal_frame.head(2))
        chosen = pd.concat(picks).drop_duplicates("channel_index") if picks else ranked.iloc[0:0]
        if len(chosen) < per_depth_target:
            chosen = pd.concat([chosen, ranked.loc[~ranked["channel_index"].isin(chosen["channel_index"])]])
        selected.append(chosen.head(per_depth_target))
    return pd.concat(selected, ignore_index=True) if selected else eligible.iloc[0:0].copy()


def write_qc_selection_sensitivity(qc_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Report threshold and contact-count sensitivity without re-reading raw LFP."""
    qc = pd.read_parquet(qc_path)
    result: dict[str, Any] = {"schema": "mndm.lfp_qc_selection_sensitivity.v1", "thresholds": {}, "contact_targets": {}}
    for threshold in (0.10, 0.20, 0.30):
        passing = qc["metadata_valid"] & qc["finite_geometry"] & (qc["qc_failed_fraction"] <= threshold)
        result["thresholds"][str(threshold)] = {
            "n_passing": int(passing.sum()),
            "depth_bin_counts": {str(key): int(value) for key, value in qc.loc[passing].groupby("depth_bin").size().items()},
        }
    for target in (4, 8):
        selected = select_depth_balanced(qc, per_depth_target=target)
        result["contact_targets"][str(target)] = {
            "n_selected": int(len(selected)),
            "depth_bin_counts": {str(key): int(value) for key, value in selected.groupby("depth_bin").size().items()},
        }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def write_selection_from_qc(
    qc_path: str | Path,
    base_selection_path: str | Path,
    output_path: str | Path,
    *,
    per_depth_target: int,
) -> dict[str, Any]:
    """Materialize a versioned contact-count selection without re-reading LFP."""
    qc = pd.read_parquet(qc_path)
    base = json.loads(Path(base_selection_path).read_text(encoding="utf-8"))
    selected = select_depth_balanced(qc, per_depth_target=per_depth_target)
    policy = dict(base.get("selection_policy", {}))
    policy.update(
        {
            "per_depth_target": int(per_depth_target),
            "sensitivity_parent_artifact": str(base_selection_path),
            "sensitivity_axis": "contacts_per_depth",
        }
    )
    artifact = {
        **base,
        "target_n_channels": int(len(selected)),
        "selection_policy": policy,
        "selected_channels": _records(selected),
        "rejected_channels": _records(qc.loc[~qc["channel_index"].isin(selected["channel_index"])]),
        "ensemble_groups": _ensemble_groups(selected),
        "provenance": {
            **dict(base.get("provenance", {})),
            "sensitivity_source_qc": str(qc_path),
            "sensitivity_variant": f"contacts{per_depth_target}_native",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, default=_json_default), encoding="utf-8")
    selected.to_parquet(target.with_name(target.stem + "_selected_probe_geometry.parquet"), index=False)
    return artifact


def _stream_metrics(
    path: Path,
    series_path: str,
    chunk_sec: float,
    *,
    state_contract: LfpStateContract | None = None,
) -> tuple[pd.DataFrame, float, float]:
    import h5py

    with h5py.File(path, "r") as handle:
        group = handle[series_path.strip("/")]
        data = group["data"]
        n_samples, n_channels = data.shape
        sfreq = _series_sfreq(group)
        chunk_samples = max(1, int(round(float(chunk_sec) * sfreq)))
        conversion = _number(group.get("conversion"), default=float(data.attrs.get("conversion", 1.0)))
        sums = np.zeros(n_channels)
        sums_sq = np.zeros(n_channels)
        corr_sum = np.zeros(n_channels)
        flat = np.zeros(n_channels)
        rails = np.zeros(n_channels)
        failed = np.zeros(n_channels, dtype=int)
        windows = 0
        state_counts: dict[str, int] = {}
        state_failed: dict[str, np.ndarray] = {}
        for start in range(0, n_samples, chunk_samples):
            values = np.asarray(data[start : min(n_samples, start + chunk_samples), :], dtype=np.float32)
            values *= conversion
            finite = np.isfinite(values)
            per_var = np.nanvar(values, axis=0)
            median_var = float(np.nanmedian(per_var[np.isfinite(per_var)])) if np.isfinite(per_var).any() else 0.0
            flat_window = per_var <= max(1e-18, median_var * 1e-4)
            high_window = per_var >= median_var * 25.0 if median_var > 0 else np.zeros(n_channels, dtype=bool)
            rail_window = np.mean(np.abs(values) >= np.iinfo(np.int16).max * abs(conversion) * 0.999, axis=0)
            subsample = values[:: max(1, values.shape[0] // 4096)]
            probe_median = np.nanmedian(subsample, axis=1)
            corr = _column_correlation(subsample, probe_median)
            finite_counts = finite.sum(axis=0)
            sums += np.nansum(values, axis=0)
            sums_sq += np.nansum(values**2, axis=0)
            corr_sum += corr
            flat += flat_window
            rails += rail_window
            failed += (flat_window | high_window | (rail_window > 0.001) | (corr < 0.05)).astype(int)
            if state_contract is not None:
                midpoint = state_contract.time_origin_sec + (start + values.shape[0] / 2.0) / sfreq
                label = _state_at(midpoint, state_contract.blocks)
                if label is not None:
                    state_counts[label] = state_counts.get(label, 0) + 1
                    state_failed.setdefault(label, np.zeros(n_channels, dtype=int))
                    state_failed[label] += (
                        flat_window | high_window | (rail_window > 0.001) | (corr < 0.05)
                    ).astype(int)
            windows += 1
    counts = max(1, n_samples)
    variance = np.maximum(0.0, sums_sq / counts - (sums / counts) ** 2)
    qc_score = (corr_sum / max(1, windows)) - (failed / max(1, windows))
    records: dict[str, Any] = {
                "channel_index": np.arange(n_channels, dtype=int),
                "variance": variance,
                "median_corr": corr_sum / max(1, windows),
                "flat_windows": flat.astype(int),
                "rail_fraction": rails / max(1, windows),
                "failed_windows": failed,
                "n_windows": windows,
                "qc_score": qc_score,
            }
    for label, values in state_failed.items():
        records[f"state_{label}_failed_fraction"] = values / max(1, state_counts[label])
    return (
        pd.DataFrame(records),
        sfreq,
        n_samples / sfreq,
    )


def _series_sfreq(group: Any) -> float:
    if "starting_time" in group and "rate" in group["starting_time"].attrs:
        return float(group["starting_time"].attrs["rate"])
    timestamps = group.get("timestamps")
    if timestamps is not None:
        sample = np.asarray(timestamps[: min(len(timestamps), 4096)], dtype=float)
        return float(1.0 / np.median(np.diff(sample)))
    raise ValueError("LFP series has no usable sampling clock")


def _column_correlation(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = reference - np.nanmean(reference)
    ref_norm = np.linalg.norm(reference)
    out = np.zeros(values.shape[1])
    for index in range(values.shape[1]):
        channel = values[:, index] - np.nanmean(values[:, index])
        denominator = np.linalg.norm(channel) * ref_norm
        out[index] = float(np.nansum(channel * reference) / denominator) if denominator > 0 else 0.0
    return out


def _rejection_reason(row: pd.Series) -> str:
    reasons = []
    if not bool(row.get("metadata_valid", False)):
        reasons.append("metadata_invalid")
    if not bool(row.get("finite_geometry", False)):
        reasons.append("missing_depth")
    if float(row.get("qc_failed_fraction", 1.0)) > 0.20:
        reasons.append("signal_qc")
    return "|".join(reasons)


def _state_at(time_sec: float, blocks: pd.DataFrame) -> str | None:
    matched = blocks.loc[(blocks["start_sec"] <= time_sec) & (time_sec < blocks["stop_sec"]), "state"]
    return str(matched.iloc[0]) if len(matched) else None


def _state_qc_payload(qc: pd.DataFrame, contract: LfpStateContract | None) -> dict[str, Any]:
    if contract is None:
        return {"status": "not_assigned"}
    output: dict[str, Any] = {"status": "assigned", "states": {}, "policy": "reported_only"}
    for state in contract.blocks["state"].unique():
        column = f"state_{state}_failed_fraction"
        if column in qc:
            output["states"][str(state)] = {
                "median_failed_fraction": float(qc[column].median()),
                "n_channels_above_25pct": int((qc[column] > 0.25).sum()),
            }
    return output


def _ensemble_groups(selected: pd.DataFrame) -> dict[str, list[str]]:
    return {
        f"depth_{int(depth)}": [f"lfp_ch{int(index):03d}" for index in group["channel_index"]]
        for depth, group in selected.groupby("depth_bin", dropna=True, sort=True)
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    fields = [
        "channel_index",
        "electrode_id",
        "probe_vertical_position",
        "probe_horizontal_position",
        "location_raw",
        "qc_status",
        "rejection_reason",
        "qc_score",
    ]
    return [{key: row.get(key) for key in fields} for row in frame.to_dict(orient="records")]


def _number(dataset: Any, *, default: float) -> float:
    if dataset is None:
        return default
    try:
        return float(np.asarray(dataset[()]).reshape(-1)[0])
    except Exception:
        return default


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):
        return None
    raise TypeError(f"Not JSON serializable: {type(value)!r}")
