"""Strict, read-only Gate E/Q input loader for W_Q diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


FIXED_DT_TOLERANCE_SEC = 1.0e-6
TRANSITION_SCHEMA = "mndm.transition_residuals.v1"
Q_SCHEMA = "mndm.transition_residual_covariance_proxy.v1"


def _scalar(value: Any) -> Any:
    value = value[()] if isinstance(value, h5py.Dataset) else value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _scalar(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def _typed_hash(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    header = json.dumps({"dtype": arr.dtype.str, "shape": list(arr.shape)}, separators=(",", ":"), sort_keys=True).encode()
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_scalar(group: h5py.Group, name: str) -> Any:
    if name not in group or not isinstance(group[name], h5py.Dataset) or group[name].shape != ():
        raise ValueError(f"missing scalar metadata: {group.name}/{name}")
    return _scalar(group[name])


def _read_metadata(group: h5py.Group, names: tuple[str, ...]) -> dict[str, Any]:
    return {name: _require_scalar(group, name) for name in names}


def load_gate_e_arrays(
    h5_path: str | Path,
    branch: str = "primary",
    *,
    expected_q_dt_sec: float | None = None,
    dt_tolerance_sec: float = FIXED_DT_TOLERANCE_SEC,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load and strictly validate Gate E propagators and recording-level Q.

    This function never changes the HDF5 source and never derives a tolerance
    from the observed Q deviation. Transition IDs must be adjacent both within
    each edge and between rows, so a diagnostic cannot bridge a gap.
    """
    path = Path(h5_path)
    if not path.is_file():
        raise ValueError(f"missing HDF5 input: {path}")
    if not np.isfinite(dt_tolerance_sec) or dt_tolerance_sec < 0:
        raise ValueError("dt_tolerance_sec must be finite and non-negative")
    with h5py.File(path, "r") as handle:
        base = f"/transition_residuals/v1/{branch}"
        qbase = f"/transition_residual_covariance_proxy/v1/{branch}"
        series_path = f"{base}/series"
        for required in (
            f"{series_path}/phi_one_step",
            f"{series_path}/source_window_id",
            f"{series_path}/target_window_id",
            f"{series_path}/dt_sec",
            f"{qbase}/covariance",
        ):
            if required not in handle:
                raise ValueError(f"missing diagnostic input: {required}")
        series = handle[series_path]
        phi = np.asarray(series["phi_one_step"][:], dtype=np.float64)
        source = np.asarray(series["source_window_id"][:])
        target = np.asarray(series["target_window_id"][:])
        dt = np.asarray(series["dt_sec"][:], dtype=np.float64).reshape(-1)
        q = np.asarray(handle[f"{qbase}/covariance"][:], dtype=np.float64)
        if phi.ndim != 3 or phi.shape[1] != phi.shape[2] or phi.shape[0] == 0 or not np.isfinite(phi).all():
            raise ValueError("invalid finite phi_one_step shape")
        if source.ndim != 1 or target.ndim != 1 or source.shape != target.shape or source.size != phi.shape[0]:
            raise ValueError("invalid transition ID alignment")
        if source.dtype.kind not in {"i", "u"} or target.dtype.kind not in {"i", "u"}:
            raise ValueError("transition IDs must be integer datasets")
        if np.any(source < 0) or np.any(target != source + 1):
            raise ValueError("transition edge IDs are not adjacent")
        if source.size > 1 and np.any(source[1:] != target[:-1]):
            raise ValueError("transition rows are not adjacent; refusing gap bridging")
        if dt.shape != (phi.shape[0],) or not np.isfinite(dt).all() or np.any(dt <= 0):
            raise ValueError("invalid transition dt")
        q_dt = float(_require_scalar(handle[qbase], "q_dt_sec"))
        q_max_deviation = float(_require_scalar(handle[qbase], "q_max_dt_deviation_sec"))
        if not np.isfinite(q_dt) or q_dt <= 0 or not np.isfinite(q_max_deviation) or q_max_deviation < 0:
            raise ValueError("invalid serialized Q dt metadata")
        expected = q_dt if expected_q_dt_sec is None else float(expected_q_dt_sec)
        if not np.isfinite(expected) or expected <= 0:
            raise ValueError("expected_q_dt_sec must be finite and positive")
        if abs(q_dt - expected) > dt_tolerance_sec or np.max(np.abs(dt - expected)) > dt_tolerance_sec:
            raise ValueError("transition dt does not meet fixed expected Q dt tolerance")
        if q.ndim != 2 or q.shape != (phi.shape[1], phi.shape[1]) or not np.isfinite(q).all():
            raise ValueError("invalid finite Q covariance shape")
        qmeta = _read_metadata(
            handle[qbase],
            (
                "schema_version", "computation_status", "q_time_semantics", "q_units",
                "conversion_model", "q_scope", "q_semantics", "measurement_validity", "claim_status",
            ),
        )
        if qmeta["schema_version"] != Q_SCHEMA or qmeta["computation_status"] != "computed":
            raise ValueError("Q contract is not a computed transition covariance")
        if qmeta["q_time_semantics"] != "one_step_transition_covariance" or qmeta["q_units"] != "state_squared":
            raise ValueError("Q contract has non-admissible time semantics or units")
        if qmeta["conversion_model"] != "not_applicable" or qmeta["q_scope"] != "recording":
            raise ValueError("Q contract has non-admissible scope or conversion model")
        transition_meta = _read_metadata(handle[base], ("schema_version", "computation_status", "measurement_validity", "claim_status"))
        if transition_meta["schema_version"] != TRANSITION_SCHEMA or transition_meta["computation_status"] != "computed":
            raise ValueError("transition residual source is not computed")
        reachability_meta: dict[str, Any] = {}
        reach_base = f"/stochastic_reachability/v1/{branch}"
        if reach_base in handle:
            for name in ("schema_version", "computation_status", "failure_reason", "measurement_validity", "claim_status"):
                if name in handle[reach_base]:
                    reachability_meta[name] = _scalar(handle[reach_base][name])
        source_out = {
            "h5_path": str(path),
            "branch": branch,
            "source_sha256": _file_hash(path),
            "phi_sha256": _typed_hash(phi),
            "q_sha256": _typed_hash(q),
            "source_window_id_sha256": _typed_hash(source.astype(np.int64)),
            "target_window_id_sha256": _typed_hash(target.astype(np.int64)),
            "dt_sha256": _typed_hash(dt),
            "source_window_ids": source.astype(np.int64).tolist(),
            "target_window_ids": target.astype(np.int64).tolist(),
            "dt_sec": dt.tolist(),
            "q_dt_sec": q_dt,
            "q_max_dt_deviation_sec": q_max_deviation,
            "dt_tolerance_sec": float(dt_tolerance_sec),
            "transition_metadata": transition_meta,
            "q_metadata": qmeta,
            "reachability_metadata": reachability_meta,
        }
    return phi, q, source_out
