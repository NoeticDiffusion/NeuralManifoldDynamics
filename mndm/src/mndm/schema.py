"""Shared schema contracts for MNPS tensor outputs.

This module defines lightweight dataclasses and helpers that normalize the
payload passed between the MNPS computation stack and the IO writers. The
shapes and dtypes follow the reference specification in
``docs/MNPS_Jacobian_tensor_spec.md``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np


MNPS_AXIS_NAMES = ("m", "d", "e")
mnps_9d_CANONICAL_ORDER: Sequence[str] = (
    "m_a",
    "m_e",
    "m_o",
    "d_n",
    "d_l",
    "d_s",
    "e_e",
    "e_s",
    "e_m",
)


@dataclass
class MNPSPayload:
    """Container for MNPS tensors destined for disk writers.

    Attributes:
        time: 1-D array ``[T]``, monotonically increasing timestamps (seconds).
        x: 2-D array ``[T, 3]``, MNPS coordinates ``[m, d, e]`` (Jacobian code
            supports other widths when needed).
        x_dot: 2-D array ``[T, 3]``, time derivatives of ``x``.
        stage: Optional 1-D ``[T]`` integer-coded sleep/task labels.
        z: Optional 2-D ``[T, K]`` embodied channels.
        events: Mapping from event name to 1-D index or timestamp arrays.
        nn_indices: Optional ``[T, k]`` neighbour indices (``int32``).
        jacobian: Optional ``[W, D, D]`` windowed Jacobians ``J_hat`` (often ``D=3``).
        jacobian_dot: Optional ``[W, D, D]`` Jacobian rates ``J_dot``.
        jacobian_centers: Optional ``[W]`` window center indices.
        jacobian_9D: Optional ``[W2, K, K]`` stratified MNPS Jacobians (often ``K=9``).
        jacobian_9D_dot: Optional ``[W2, K, K]`` stratified Jacobian rates.
        jacobian_9D_centers: Optional ``[W2]`` stratified window centers.
        attrs: Free-form metadata for HDF5 attrs / JSON headers.
    """

    time: np.ndarray
    x: np.ndarray
    x_dot: np.ndarray
    stage: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    events: MutableMapping[str, np.ndarray] = field(default_factory=dict)
    nn_indices: Optional[np.ndarray] = None
    jacobian: Optional[np.ndarray] = None
    jacobian_dot: Optional[np.ndarray] = None
    jacobian_centers: Optional[np.ndarray] = None
    jacobian_9D: Optional[np.ndarray] = None
    jacobian_9D_dot: Optional[np.ndarray] = None
    jacobian_9D_centers: Optional[np.ndarray] = None
    # Optional time series extracted from jacobian_9D: mapping "out__in" -> [W2] series
    jacobian_9D_cross_partials: MutableMapping[str, np.ndarray] = field(default_factory=dict)
    # Optional per-feature baseline metadata (captured before normalization)
    feature_baselines: MutableMapping[str, Dict[str, float | str]] = field(default_factory=dict)
    # Optional first-class feature export surfaces aligned to the MNPS time axis.
    features_raw_values: Optional[np.ndarray] = None
    features_raw_names: Optional[list[str]] = None
    features_robust_z_values: Optional[np.ndarray] = None
    features_robust_z_names: Optional[list[str]] = None
    # Machine-readable feature metadata shared by the exported feature surfaces.
    feature_metadata: MutableMapping[str, Any] = field(default_factory=dict)
    attrs: MutableMapping[str, Any] = field(default_factory=dict)
    # Optional per-window time bounds (seconds)
    window_start: Optional[np.ndarray] = None
    window_end: Optional[np.ndarray] = None
    # Optional binary labels aligned to MNPS time (e.g., mapped events)
    labels: MutableMapping[str, np.ndarray] = field(default_factory=dict)
    # Optional stratified MNPS coordinates (typically 9D)
    coords_9d: Optional[np.ndarray] = None
    coords_9d_names: Optional[list[str]] = None
    # Optional raw regional signals (e.g. fMRI ROI×time). These are supporting
    # inputs for some regional analyses, but are not the canonical regional
    # output contract exposed to downstream readers.
    regions_bold: Optional[np.ndarray] = None
    regions_names: Optional[list[str]] = None
    regions_sfreq: Optional[float] = None
    # Optional extended coordinates (E-Kappa, RFM, O-Koh, TIG, etc.).
    # The structure is a free-form nested mapping that writers interpret
    # when creating HDF5 groups under ``/extensions``.
    extensions: MutableMapping[str, Any] = field(default_factory=dict)
    # Canonical modality-agnostic regional output. Each entry maps network
    # label to a dict containing 'mnps' [T,3], 'jacobian' [W,3,3], and
    # optional supporting fields such as 'stratified' and 'metrics'.
    regional_mnps: MutableMapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow dict representation for serialization."""

        return {
            "time": self.time,
            "x": self.x,
            "x_dot": self.x_dot,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "stage": self.stage,
            "z": self.z,
            "events": dict(self.events),
            "labels": dict(self.labels),
            "nn_indices": self.nn_indices,
            "jacobian": self.jacobian,
            "jacobian_dot": self.jacobian_dot,
            "jacobian_centers": self.jacobian_centers,
            "jacobian_9D": self.jacobian_9D,
            "jacobian_9D_dot": self.jacobian_9D_dot,
            "jacobian_9D_centers": self.jacobian_9D_centers,
            "jacobian_9D_cross_partials": dict(self.jacobian_9D_cross_partials),
            "feature_baselines": dict(self.feature_baselines),
            "features_raw_values": self.features_raw_values,
            "features_raw_names": list(self.features_raw_names) if self.features_raw_names is not None else None,
            "features_robust_z_values": self.features_robust_z_values,
            "features_robust_z_names": list(self.features_robust_z_names) if self.features_robust_z_names is not None else None,
            "feature_metadata": dict(self.feature_metadata),
            "attrs": dict(self.attrs),
            "coords_9d": self.coords_9d,
            "coords_9d_names": list(self.coords_9d_names) if self.coords_9d_names is not None else None,
            "regions_bold": self.regions_bold,
            "regions_names": list(self.regions_names) if self.regions_names is not None else None,
            "regions_sfreq": self.regions_sfreq,
            "extensions": dict(self.extensions),
            "regional_mnps": dict(self.regional_mnps),
        }


def _as_float_array(arr: Optional[np.ndarray], dtype: np.dtype, shape_hint: Optional[tuple[int, ...]] = None) -> Optional[np.ndarray]:
    """Normalize arrays to the requested floating dtype.

    Args:
        arr: Input array, or None.
        dtype: Target NumPy dtype (for example ``np.float32``).
        shape_hint: If set, required shape after conversion.

    Returns:
        Converted array, or None if ``arr`` is None.

    Raises:
        ValueError: If ``shape_hint`` does not match.
    """

    if arr is None:
        return None

    array = np.asarray(arr, dtype=dtype)
    if shape_hint and array.shape != shape_hint:
        raise ValueError(f"Expected shape {shape_hint}, got {array.shape}")
    return array


def _validate_optional_array(name: str, arr: Optional[np.ndarray], ndim: int) -> Optional[np.ndarray]:
    """Internal helper: validate optional array."""
    if arr is None:
        return None
    array = np.asarray(arr)
    if array.ndim != ndim:
        raise ValueError(f"{name} expected {ndim}D, got {array.ndim}D")
    return array


def _normalize_feature_surface(
    values: Optional[np.ndarray],
    names: Optional[Sequence[str]],
    *,
    t_len: int,
    label: str,
) -> tuple[Optional[np.ndarray], Optional[list[str]]]:
    """Internal helper: normalize feature surface."""
    if values is None:
        if names is not None:
            raise ValueError(f"{label}_names requires {label}_values to be present")
        return None, None
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != t_len:
        raise ValueError(f"{label}_values must be 2-D and align with time axis")
    if names is None:
        raise ValueError(f"{label}_names must be provided when {label}_values is present")
    name_list = [str(n) for n in names]
    if len(name_list) != matrix.shape[1]:
        raise ValueError(f"{label}_names length must match {label}_values columns")
    return matrix, name_list


def _normalize_feature_metadata(
    feature_metadata: MutableMapping[str, Any],
    *,
    n_features: int,
) -> MutableMapping[str, np.ndarray]:
    """Internal helper: normalize feature metadata."""
    normalized: Dict[str, np.ndarray] = {}
    for key, value in feature_metadata.items():
        arr = np.asarray(value)
        if arr.ndim != 1 or arr.shape[0] != n_features:
            raise ValueError(f"feature_metadata['{key}'] must be 1-D with length {n_features}")
        if arr.dtype.kind in {"U", "O"}:
            arr = arr.astype(str, copy=False)
        elif np.issubdtype(arr.dtype, np.bool_):
            arr = arr.astype(np.int8, copy=False)
        elif np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int32, copy=False)
        elif np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32, copy=False)
        normalized[str(key)] = arr
    return normalized


def normalize_payload(payload: MNPSPayload) -> MNPSPayload:
    """Coerce arrays to canonical dtypes and validate shapes (mutates ``payload`` in place).

    Args:
        payload: MNPS payload to normalize.

    Returns:
        The same ``MNPSPayload`` instance for chaining.
    """

    t = np.asarray(payload.time, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError("time must be 1-D")

    payload.time = t
    payload.x = _as_float_array(payload.x, np.float32, (t.shape[0], 3))
    payload.x_dot = _as_float_array(payload.x_dot, np.float32, (t.shape[0], 3))
    if payload.window_start is not None:
        payload.window_start = _as_float_array(payload.window_start, np.float32, (t.shape[0],))
    if payload.window_end is not None:
        payload.window_end = _as_float_array(payload.window_end, np.float32, (t.shape[0],))

    if payload.stage is not None:
        stage = np.asarray(payload.stage)
        if stage.shape != (t.shape[0],):
            raise ValueError("stage must align with time axis")
        payload.stage = stage.astype(np.int8, copy=False)

    if payload.z is not None:
        z = np.asarray(payload.z, dtype=np.float32)
        if z.shape[0] != t.shape[0]:
            raise ValueError("z must align with time axis")
        payload.z = z

    if payload.nn_indices is not None:
        nn = np.asarray(payload.nn_indices, dtype=np.int32)
        if nn.shape[0] != t.shape[0]:
            raise ValueError("nn_indices must align with time axis")
        payload.nn_indices = nn

    payload.features_raw_values, payload.features_raw_names = _normalize_feature_surface(
        payload.features_raw_values,
        payload.features_raw_names,
        t_len=t.shape[0],
        label="features_raw",
    )
    payload.features_robust_z_values, payload.features_robust_z_names = _normalize_feature_surface(
        payload.features_robust_z_values,
        payload.features_robust_z_names,
        t_len=t.shape[0],
        label="features_robust_z",
    )
    if payload.features_raw_names is not None and payload.features_robust_z_names is not None:
        if list(payload.features_raw_names) != list(payload.features_robust_z_names):
            raise ValueError("features_raw_names and features_robust_z_names must match")
    feature_names = payload.features_raw_names or payload.features_robust_z_names
    if payload.feature_metadata:
        if feature_names is None:
            raise ValueError("feature_metadata requires an exported feature surface")
        payload.feature_metadata = _normalize_feature_metadata(
            payload.feature_metadata,
            n_features=len(feature_names),
        )

    payload.jacobian = _validate_optional_array("jacobian", payload.jacobian, 3)
    payload.jacobian_dot = _validate_optional_array("jacobian_dot", payload.jacobian_dot, 3)
    if payload.jacobian is not None and payload.jacobian_dot is not None:
        if payload.jacobian_dot.shape != payload.jacobian.shape:
            raise ValueError(
                f"jacobian_dot must match jacobian shape, got {payload.jacobian_dot.shape} vs {payload.jacobian.shape}"
            )

    # Optional Stratified (v2) Jacobians
    payload.jacobian_9D = _validate_optional_array("jacobian_9D", payload.jacobian_9D, 3)
    payload.jacobian_9D_dot = _validate_optional_array("jacobian_9D_dot", payload.jacobian_9D_dot, 3)
    if payload.jacobian_9D is not None and payload.jacobian_9D_dot is not None:
        if payload.jacobian_9D_dot.shape != payload.jacobian_9D.shape:
            raise ValueError(
                "jacobian_9D_dot must match jacobian_9D shape, "
                f"got {payload.jacobian_9D_dot.shape} vs {payload.jacobian_9D.shape}"
            )

    # Optional jacobian_9D cross-partials
    if getattr(payload, "jacobian_9D_cross_partials", None):
        if payload.jacobian_9D is None or payload.jacobian_9D.size == 0:
            raise ValueError("jacobian_9D_cross_partials requires jacobian_9D to be present")
        w2 = int(payload.jacobian_9D.shape[0])
        normalized_cp: Dict[str, np.ndarray] = {}
        for key, value in payload.jacobian_9D_cross_partials.items():
            arr = np.asarray(value, dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] != w2:
                raise ValueError(f"jacobian_9D_cross_partials['{key}'] must be 1-D with length {w2}")
            normalized_cp[str(key)] = arr
        payload.jacobian_9D_cross_partials = normalized_cp

    if payload.jacobian_centers is not None:
        centers = np.asarray(payload.jacobian_centers, dtype=np.int32)
        payload.jacobian_centers = centers

    if payload.jacobian_9D_centers is not None:
        centers_v2 = np.asarray(payload.jacobian_9D_centers, dtype=np.int32)
        payload.jacobian_9D_centers = centers_v2

    normalized_events: Dict[str, np.ndarray] = {}
    for key, value in payload.events.items():
        arr = np.asarray(value)
        if arr.ndim != 1:
            raise ValueError(f"event '{key}' must be 1-D")
        if np.issubdtype(arr.dtype, np.integer):
            arr = arr.astype(np.int64, copy=False)
        elif np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float64, copy=False)
        else:
            raise TypeError(
                f"Event '{key}' contains non-numeric data ({arr.dtype}). "
                "MNPS events must be indices (int) or timestamps (float)."
            )
        normalized_events[key] = arr
    payload.events = normalized_events

    if payload.labels:
        normalized_labels: Dict[str, np.ndarray] = {}
        for key, value in payload.labels.items():
            arr = np.asarray(value)
            if arr.shape != (t.shape[0],):
                raise ValueError(f"label '{key}' must align with time axis")
            if np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.bool_):
                arr = arr.astype(np.int8, copy=False)
            elif np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
            else:
                # Preserve categorical/string labels (writer handles UTF-8 conversion).
                arr = arr.astype(str, copy=False)
            normalized_labels[key] = arr
        payload.labels = normalized_labels

    # Normalize coords_9d if present
    if payload.coords_9d is not None:
        cv2 = np.asarray(payload.coords_9d, dtype=np.float32)
        if cv2.ndim != 2 or cv2.shape[0] != t.shape[0]:
            raise ValueError("coords_9d must align with time axis and be 2-D")
        payload.coords_9d = cv2
        if payload.coords_9d_names is None:
            raise ValueError("coords_9d_names must be provided when coords_9d is present")
        if len(payload.coords_9d_names) != cv2.shape[1]:
            raise ValueError("coords_9d_names length must match coords_9d columns")
        names = [str(n) for n in payload.coords_9d_names]
        allow_all_non_finite = bool(payload.attrs.get("coords_9d_allow_all_non_finite_columns", False))
        # Since v2 values are pre-standardized, we only use _normalize_coords_9d for shape/NaN checks and canonical ordering.
        (
            payload.coords_9d,
            payload.coords_9d_names,
            coords_9d_diag,
        ) = _normalize_coords_9d(
            payload.coords_9d,
            names,
            allow_all_non_finite_columns=allow_all_non_finite,
            return_diagnostics=True,
        )
        if coords_9d_diag.get("all_non_finite_names"):
            payload.attrs["coords_9d_all_non_finite_names"] = coords_9d_diag.get("all_non_finite_names")
            payload.attrs["coords_9d_all_non_finite_count"] = int(coords_9d_diag.get("all_non_finite_count", 0))

    # Normalize optional raw regional signals (e.g. fMRI ROI×time). These are
    # intentionally decoupled from the MNPS time axis: the BOLD sampling
    # frequency and windowing can differ from the MNPS window grid.
    if payload.regions_bold is not None:
        reg = np.asarray(payload.regions_bold, dtype=np.float32)
        if reg.ndim != 2:
            raise ValueError("regions_bold must be 2-D (n_regions, n_times)")
        payload.regions_bold = reg

    if payload.regions_names is not None:
        names = [str(n) for n in payload.regions_names]
        if payload.regions_bold is not None and len(names) != payload.regions_bold.shape[0]:
            raise ValueError("regions_names length must match regions_bold first dimension")
        payload.regions_names = names

    if payload.regions_sfreq is not None:
        try:
            payload.regions_sfreq = float(payload.regions_sfreq)
        except Exception as exc:
            raise ValueError(f"regions_sfreq must be convertible to float, got {payload.regions_sfreq!r}") from exc

    # Explicit schema marker helps downstream readers avoid silent shape/contract drift.
    payload.attrs.setdefault("schema_version", "mnps_tensor_spec_v2")

    return payload


def _normalize_coords_9d(
    values: np.ndarray,
    names: Sequence[str],
    *,
    allow_all_non_finite_columns: bool = False,
    return_diagnostics: bool = False,
) -> tuple[np.ndarray, list[str]] | tuple[np.ndarray, list[str], Dict[str, Any]]:
    """Internal helper: normalize coords 9d."""
    if values.ndim != 2:
        raise ValueError(f"coords_9d must be 2-D, got {values.ndim}D")
    if values.shape[1] != len(names):
        raise ValueError("coords_9d names/values column mismatch")
    if values.shape[1] != len(mnps_9d_CANONICAL_ORDER):
        raise ValueError(
            f"coords_9d must contain {len(mnps_9d_CANONICAL_ORDER)} columns; got {values.shape[1]}"
        )

    name_list = list(names)
    if len(set(name_list)) != len(name_list):
        raise ValueError(f"Duplicate coords_9d names detected: {name_list}")

    missing = [key for key in mnps_9d_CANONICAL_ORDER if key not in name_list]
    if missing:
        raise ValueError(f"Missing coords_9d subcoordinates: {missing}")

    ordered_indices = [name_list.index(key) for key in mnps_9d_CANONICAL_ORDER]
    ordered_values = values[:, ordered_indices]

    finite_mask = np.isfinite(ordered_values)
    finite_counts = finite_mask.sum(axis=0).astype(np.int32)
    all_non_finite_mask = finite_counts == 0
    all_non_finite_names = [
        mnps_9d_CANONICAL_ORDER[idx] for idx, is_all_missing in enumerate(all_non_finite_mask) if bool(is_all_missing)
    ]
    if all_non_finite_names and not allow_all_non_finite_columns:
        raise ValueError(
            "Non-finite variance detected in coords_9d values "
            f"(all-non-finite columns: {all_non_finite_names})"
        )

    sentinel = np.float32(9.999e9)
    seen: Dict[str, int] = {}
    duplicates: Dict[str, str] = {}
    for idx in range(ordered_values.shape[1]):
        # Degraded all-NaN columns are allowed in tolerant mode and must not
        # trigger false duplicate collisions via sentinel hashing.
        if bool(all_non_finite_mask[idx]):
            continue
        # Hash contiguous float32 bytes for fast duplicate detection without
        # constructing massive Python tuples.
        col = np.nan_to_num(ordered_values[:, idx], nan=sentinel).astype(np.float32, copy=False)
        sig = hashlib.sha256(np.ascontiguousarray(col).view(np.uint8)).hexdigest()
        if sig in seen:
            duplicates[mnps_9d_CANONICAL_ORDER[idx]] = mnps_9d_CANONICAL_ORDER[seen[sig]]
        else:
            seen[sig] = idx
    if duplicates:
        raise ValueError(f"Duplicate coords_9d columns detected: {duplicates}")

    out_values = np.ascontiguousarray(ordered_values, dtype=np.float32)
    out_names = list(mnps_9d_CANONICAL_ORDER)
    if return_diagnostics:
        diagnostics: Dict[str, Any] = {
            "all_non_finite_names": all_non_finite_names,
            "all_non_finite_count": int(len(all_non_finite_names)),
            "finite_count_by_name": {
                out_names[idx]: int(finite_counts[idx]) for idx in range(len(out_names))
            },
        }
        return out_values, out_names, diagnostics
    return out_values, out_names


def compute_meta_indices(jacobian: Optional[np.ndarray]) -> Dict[str, float]:
    """Compute aggregate MNPS meta indices from a Jacobian tensor."""

    if jacobian is None or jacobian.size == 0:
        return {"mean_trace": float("nan"), "mean_rotation_fro": float("nan"), "windows": 0}

    trace = np.trace(jacobian, axis1=1, axis2=2)
    omega = 0.5 * (jacobian - np.swapaxes(jacobian, 1, 2))
    rot_norm = np.linalg.norm(omega, axis=(1, 2))

    def _safe_nanmean_1d(arr: np.ndarray) -> float:
        """Internal helper: safe nanmean 1d."""
        a = np.asarray(arr, dtype=float).ravel()
        a = a[np.isfinite(a)]
        return float(np.mean(a)) if a.size else float("nan")

    return {
        # Avoid RuntimeWarning: "Mean of empty slice" when all entries are NaN.
        "mean_trace": _safe_nanmean_1d(trace),
        "mean_rotation_fro": _safe_nanmean_1d(rot_norm),
        "windows": int(jacobian.shape[0]),
    }


