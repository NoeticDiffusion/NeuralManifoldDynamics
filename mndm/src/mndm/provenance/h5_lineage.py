"""Fail-closed HDF5 row-lineage readers for multimodal comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import pandas as pd


REQUIRED_ROW_SOURCE_COLUMNS = ("has_eeg", "has_meg", "raw_file", "source_format")


@dataclass(frozen=True)
class SimultaneousFeatureRows:
    """A transform-ready HDF5 feature frame selected by explicit provenance."""

    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    h5_path: Path


def _decode_strings(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in {"S", "O", "U"}:
        return array
    return np.asarray(
        [item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item) for item in array],
        dtype=object,
    )


def _require_dataset(h5: h5py.File, path: str) -> h5py.Dataset:
    if path not in h5 or not isinstance(h5[path], h5py.Dataset):
        raise ValueError(f"{h5.filename}: required dataset missing: /{path}")
    return h5[path]


def read_feature_surface(h5: h5py.File, surface: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read one aligned feature surface with strict shape/name checks."""
    values = np.asarray(_require_dataset(h5, f"{surface}/values"))
    names = tuple(str(value) for value in _decode_strings(_require_dataset(h5, f"{surface}/names")[:]))
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError(f"{h5.filename}: invalid /{surface} values/names shape")
    if len(set(names)) != len(names):
        raise ValueError(f"{h5.filename}: duplicate feature names in /{surface}")
    return values, names


def load_simultaneous_feature_rows(h5_path: Path, *, feature_surface: str = "features_raw") -> SimultaneousFeatureRows:
    """Load only rows explicitly marked as containing both EEG and MEG.

    Missing provenance is an error: callers must never infer simultaneous rows
    from stacked-array position.
    """
    path = Path(h5_path)
    with h5py.File(path, "r") as h5:
        values, names = read_feature_surface(h5, feature_surface)
        row_source = h5.get("row_source")
        if not isinstance(row_source, h5py.Group):
            raise ValueError(f"{path}: /row_source is required; positional fallback is forbidden")
        missing = [name for name in REQUIRED_ROW_SOURCE_COLUMNS if name not in row_source]
        if missing:
            raise ValueError(f"{path}: /row_source missing required columns: {missing}")
        n_rows = values.shape[0]
        columns: dict[str, np.ndarray] = {}
        for name in REQUIRED_ROW_SOURCE_COLUMNS:
            data = np.asarray(row_source[name])
            if data.ndim != 1 or len(data) != n_rows:
                raise ValueError(f"{path}: /row_source/{name} is not aligned to feature rows")
            columns[name] = _decode_strings(data)
        for name in ("window_start", "window_end"):
            data = np.asarray(_require_dataset(h5, name))
            if data.ndim != 1 or len(data) != n_rows:
                raise ValueError(f"{path}: /{name} is not aligned to feature rows")
            columns[name] = data
        if "epoch_id" in h5:
            data = np.asarray(h5["epoch_id"])
            if data.ndim != 1 or len(data) != n_rows:
                raise ValueError(f"{path}: /epoch_id is not aligned to feature rows")
            columns["epoch_id"] = data

    simultaneous = np.asarray(columns["has_eeg"], dtype=bool) & np.asarray(columns["has_meg"], dtype=bool)
    if not np.any(simultaneous):
        raise ValueError(f"{path}: no explicitly simultaneous EEG+MEG rows")
    frame = pd.DataFrame(values[simultaneous], columns=names)
    for name, data in columns.items():
        frame.insert(0, name, np.asarray(data)[simultaneous])
    return SimultaneousFeatureRows(frame=frame, feature_names=names, h5_path=path)


def window_key_frame(frame: pd.DataFrame, *, subject: str, run: str) -> pd.DataFrame:
    """Create an exact non-positional key for independently read H5 streams."""
    required = ("raw_file", "source_format", "window_start", "window_end")
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Cannot build window keys; missing {missing}")
    keys = frame.loc[:, required].copy()
    if "epoch_id" in frame:
        keys.insert(2, "epoch_id", frame["epoch_id"].to_numpy())
    keys.insert(0, "run", str(run))
    keys.insert(0, "subject", str(subject))
    if keys.duplicated().any():
        raise ValueError("Simultaneous-row keys are not unique")
    return keys


def require_exact_window_match(eeg_frame: pd.DataFrame, meg_frame: pd.DataFrame, *, subject: str, run: str) -> None:
    """Reject comparison inputs whose provenance/time keys differ."""
    if not window_key_frame(eeg_frame, subject=subject, run=run).equals(
        window_key_frame(meg_frame, subject=subject, run=run)
    ):
        raise ValueError(f"EEG/MEG simultaneous windows do not align exactly for sub-{subject} run-{run}")


def required_feature_columns(
    available: Iterable[str], *, modality: str, sector_names: Sequence[str], weighted_features: Sequence[str]
) -> list[str]:
    """Require global and frozen grouped features before residual analysis."""
    available_set = set(map(str, available))
    missing: list[str] = []
    for feature in weighted_features:
        if feature not in available_set:
            missing.append(feature)
        for sector in sector_names:
            column = f"{feature}__g_{modality}_{sector}"
            if column not in available_set:
                missing.append(column)
    if missing:
        raise ValueError(f"Missing frozen residual feature columns: {sorted(missing)}")
    return [str(name) for name in weighted_features]
