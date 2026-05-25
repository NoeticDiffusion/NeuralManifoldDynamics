"""Feature-anchor utilities for MNDM 2.1 coordinate contracts.

The functions in this module intentionally operate on exported ``/features_raw``
surfaces so anchors can be fitted post-hoc before the summarize pipeline is
rewired.  Anchor statistics are subject-balanced: each subject contributes one
summary value per feature, so long recordings do not dominate the reference.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np
import pandas as pd

from .projection import select_export_feature_columns


ROBUST_MAD_TO_SIGMA = 1.4826
IQR_TO_SIGMA = 1.0 / 1.349


def _json_default(value: Any) -> Any:
    """Convert NumPy scalars/arrays to JSON-friendly values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def stable_anchor_hash(payload: Mapping[str, Any]) -> str:
    """Return a deterministic hash for an anchor payload."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _decode_names(raw: Any) -> list[str]:
    """Decode HDF5 string/bytes arrays into Python strings."""
    out: list[str] = []
    for item in raw:
        if isinstance(item, (bytes, bytearray, np.bytes_)):
            out.append(bytes(item).decode("utf-8", errors="replace"))
        else:
            out.append(str(item))
    return out


def _resolve_feature_pipeline(
    col: str,
    pipeline_map: Optional[Mapping[str, Sequence[str]]] = None,
) -> list[str]:
    """Resolve the configured feature transform pipeline for an anchor fit."""
    pipeline = None
    if pipeline_map and col in pipeline_map:
        pipeline = pipeline_map[col]
    elif pipeline_map and "default" in pipeline_map:
        pipeline = pipeline_map["default"]
    if pipeline is None:
        pipeline = ["robust_z", "clip"]
    return [str(step).strip().lower() for step in pipeline if str(step).strip()]


def transform_values_for_anchor(
    values: np.ndarray,
    feature_name: str,
    *,
    feature_standardization: Optional[Mapping[str, Sequence[str]]] = None,
) -> np.ndarray:
    """Apply only pre-anchor transforms for one feature.

    For a pipeline such as ``log10 -> robust_z -> clip``, anchors must be fitted
    on the log-transformed values.  Scaling and clipping are intentionally not
    applied here because the anchor itself supplies the center/scale.
    """
    out = np.asarray(values, dtype=np.float32).copy()
    mask = np.isfinite(out)
    for step in _resolve_feature_pipeline(feature_name, feature_standardization):
        if step == "log10":
            out[mask] = np.log10(np.clip(out[mask], 1e-9, None))
        elif step in {"robust_z", "robust", "z"}:
            break
        elif step == "clip":
            # Clip belongs after scaling; ignore it while fitting anchors.
            continue
    out[~np.isfinite(out)] = np.nan
    return out


def _safe_sigma_from_mad(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    med = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - med))) * ROBUST_MAD_TO_SIGMA
    return mad if np.isfinite(mad) and mad > 0 else float("nan")


def _safe_sigma_from_iqr(values: np.ndarray) -> tuple[float, float, float, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    q25, q50, q75 = np.nanpercentile(finite, [25.0, 50.0, 75.0])
    iqr_sigma = float((q75 - q25) * IQR_TO_SIGMA)
    if not np.isfinite(iqr_sigma) or iqr_sigma <= 0:
        iqr_sigma = float("nan")
    return float(q25), float(q50), float(q75), iqr_sigma


def _safe_qn(values: np.ndarray, *, max_n: int = 2048) -> float:
    """Compute a small-sample Qn robust scale estimate.

    This exact implementation is intended for subject-level medians, not millions
    of raw epochs.  It returns NaN above ``max_n`` to avoid quadratic surprises.
    """
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    n = int(finite.size)
    if n < 2 or n > int(max_n):
        return float("nan")
    diffs = []
    for i in range(n - 1):
        diffs.extend(np.abs(finite[i + 1 :] - finite[i]).tolist())
    if not diffs:
        return float("nan")
    h = n // 2 + 1
    k = h * (h - 1) // 2
    arr = np.asarray(diffs, dtype=np.float64)
    idx = max(0, min(int(k) - 1, arr.size - 1))
    qn = 2.2219 * float(np.partition(arr, idx)[idx])
    return qn if np.isfinite(qn) and qn > 0 else float("nan")


def find_h5_files(root: Path, *, dataset_id: Optional[str] = None) -> list[Path]:
    """Find candidate H5 files under a dataset/run root."""
    root = Path(root)
    if dataset_id:
        ds_root = root / str(dataset_id)
        if ds_root.exists():
            root = ds_root
    if root.is_file() and root.suffix.lower() == ".h5":
        return [root]
    if not root.exists():
        return []
    return sorted(p for p in root.glob("**/*.h5") if p.is_file())


def _subject_id_from_h5(h5: h5py.File, path: Path) -> str:
    for key in ("subject_id", "subject", "meta_participant_id"):
        raw = h5.attrs.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return path.parent.name.split("_")[0] or path.stem


def _group_from_h5(h5: h5py.File) -> str:
    for key in ("group", "meta_Group", "meta_group", "mapped_group", "condition"):
        raw = h5.attrs.get(key)
        if raw is not None and str(raw).strip():
            return str(raw).strip()
    return ""


_SUBJECT_TOKEN_RE = re.compile(r"(sub-[A-Za-z0-9]+)")


def _normalize_subject_id(value: Any) -> str:
    """Coerce subject-like identifiers into stable ``sub-*`` tokens."""
    text = str(value or "").strip()
    if not text:
        return "sub-unknown"
    if text.startswith("sub-"):
        return text
    token = text
    if token.endswith(".0"):
        try:
            as_float = float(token)
            if as_float.is_integer():
                token = str(int(as_float))
        except Exception:
            pass
    if token.isdigit():
        return f"sub-{int(token):03d}"
    return f"sub-{token}"


def _subject_ids_from_features_df(
    features_df: pd.DataFrame,
    subject_ids: Optional[Sequence[Any]],
) -> pd.Series:
    """Resolve per-row subject ids for DataFrame-based anchor fitting."""
    if subject_ids is not None:
        if len(subject_ids) != len(features_df):
            raise ValueError("subject_ids must have the same length as features_df")
        return pd.Series([_normalize_subject_id(v) for v in subject_ids], index=features_df.index, dtype="string")
    for col in ("subject_id", "subject"):
        if col in features_df.columns:
            return pd.Series(
                [_normalize_subject_id(v) for v in features_df[col].tolist()],
                index=features_df.index,
                dtype="string",
            )
    if "file" in features_df.columns:
        out = []
        for raw in features_df["file"].tolist():
            text = str(raw or "")
            match = _SUBJECT_TOKEN_RE.search(text)
            out.append(match.group(1) if match else "sub-unknown")
        return pd.Series(out, index=features_df.index, dtype="string")
    return pd.Series(["sub-unknown"] * len(features_df), index=features_df.index, dtype="string")


def _file_ids_from_features_df(
    features_df: pd.DataFrame,
    file_ids: Optional[Sequence[Any]],
) -> pd.Series:
    """Resolve per-row file ids for DataFrame-based anchor fitting."""
    if file_ids is not None:
        if len(file_ids) != len(features_df):
            raise ValueError("file_ids must have the same length as features_df")
        return pd.Series([str(v or "") for v in file_ids], index=features_df.index, dtype="string")
    if "file" in features_df.columns:
        return features_df["file"].astype("string")
    return pd.Series([f"row_{i}" for i in range(len(features_df))], index=features_df.index, dtype="string")


def _group_by_subject_from_features_df(
    features_df: pd.DataFrame,
    subject_series: pd.Series,
    group_by_subject: Optional[Mapping[str, Any]],
) -> dict[str, str]:
    """Resolve optional subject->group provenance for DataFrame-based anchor fitting."""
    resolved: dict[str, str] = {}
    if isinstance(group_by_subject, Mapping):
        for key, value in group_by_subject.items():
            subject = _normalize_subject_id(key)
            group = str(value or "").strip()
            if group:
                resolved[subject] = group
    if "group" not in features_df.columns:
        return resolved
    group_series = features_df["group"].astype("string")
    temp = pd.DataFrame({"subject": subject_series, "group": group_series}, index=features_df.index)
    temp = temp[temp["group"].notna() & temp["group"].astype(str).str.strip().ne("")]
    if temp.empty:
        return resolved
    for subject, group in temp.groupby("subject", sort=False)["group"].first().items():
        subject_key = _normalize_subject_id(subject)
        if subject_key not in resolved:
            resolved[subject_key] = str(group).strip()
    return resolved


def _build_anchor_artifact(
    *,
    subject_values: Mapping[str, Mapping[str, Sequence[float]]],
    feature_epoch_counts: Mapping[str, int],
    group_by_subject: Mapping[str, str],
    files_used: int,
    anchor_id: str,
    anchor_source: str,
    cohort_filter: str,
    min_subjects: int,
    scale_method: str,
) -> dict[str, Any]:
    """Assemble a stable feature-anchor artifact from per-subject summaries."""
    feature_names = sorted({name for subj in subject_values.values() for name in subj.keys()})
    features: list[dict[str, Any]] = []
    for name in feature_names:
        vals = []
        for subject in sorted(subject_values.keys()):
            raw = subject_values[subject].get(name, [])
            if raw:
                vals.append(float(np.nanmedian(np.asarray(raw, dtype=np.float64))))
        arr = np.asarray(vals, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        q25, q50, q75, iqr_sigma = _safe_sigma_from_iqr(arr)
        mad_sigma = _safe_sigma_from_mad(arr)
        qn_sigma = _safe_qn(arr)
        fallback_scale = next(
            (v for v in (iqr_sigma, mad_sigma, qn_sigma, float(np.nanstd(arr)) if arr.size else float("nan")) if np.isfinite(v) and v > 0),
            float("nan"),
        )
        features.append(
            {
                "feature_name": name,
                "center": q50,
                "scale": fallback_scale,
                "scale_method_default": str(scale_method),
                "q25": q25,
                "q50": q50,
                "q75": q75,
                "iqr_sigma": iqr_sigma,
                "mad_sigma": mad_sigma,
                "qn_sigma": qn_sigma,
                "n_subjects": int(arr.size),
                "n_epochs": int(feature_epoch_counts.get(name, 0)),
                "usable": bool(arr.size >= int(min_subjects) and np.isfinite(fallback_scale) and fallback_scale > 0),
            }
        )

    spec = {
        "schema_version": "mndm.feature_anchors.v2.1",
        "anchor_id": str(anchor_id),
        "anchor_source": str(anchor_source),
        "cohort_filter": str(cohort_filter),
        "scale_method": str(scale_method),
        "subject_balanced": True,
        "min_subjects": int(min_subjects),
        "n_subjects": int(len(subject_values)),
        "n_files": int(files_used),
        "groups": {k: int(sum(1 for v in group_by_subject.values() if v == k)) for k in sorted(set(group_by_subject.values())) if k},
    }
    artifact = {"spec": spec, "features": features}
    artifact["spec"]["anchor_hash"] = stable_anchor_hash({"spec": spec, "features": features})
    return artifact


def fit_feature_anchors_from_h5(
    h5_paths: Sequence[Path],
    *,
    anchor_id: str,
    anchor_source: str = "all_subjects",
    cohort_filter: str = "",
    feature_standardization: Optional[Mapping[str, Sequence[str]]] = None,
    min_subjects: int = 3,
    scale_method: str = "iqr",
) -> dict[str, Any]:
    """Fit a subject-balanced feature-anchor artifact from H5 feature exports."""
    subject_values: Dict[str, Dict[str, list[float]]] = {}
    feature_epoch_counts: Dict[str, int] = {}
    group_by_subject: Dict[str, str] = {}
    files_used = 0

    for path in h5_paths:
        try:
            with h5py.File(path, "r") as h5:
                if "features_raw" not in h5:
                    continue
                grp = h5["features_raw"]
                if "values" not in grp or "names" not in grp:
                    continue
                values = np.asarray(grp["values"], dtype=np.float32)
                names = _decode_names(grp["names"][...])
                if values.ndim != 2 or values.shape[1] != len(names):
                    continue
                subject = _subject_id_from_h5(h5, path)
                group_by_subject.setdefault(subject, _group_from_h5(h5))
                slot = subject_values.setdefault(subject, {})
                for idx, name in enumerate(names):
                    transformed = transform_values_for_anchor(
                        values[:, idx],
                        name,
                        feature_standardization=feature_standardization,
                    )
                    finite = transformed[np.isfinite(transformed)]
                    if finite.size == 0:
                        continue
                    slot.setdefault(name, []).append(float(np.nanmedian(finite)))
                    feature_epoch_counts[name] = feature_epoch_counts.get(name, 0) + int(finite.size)
                files_used += 1
        except Exception:
            continue
    return _build_anchor_artifact(
        subject_values=subject_values,
        feature_epoch_counts=feature_epoch_counts,
        group_by_subject=group_by_subject,
        files_used=files_used,
        anchor_id=anchor_id,
        anchor_source=anchor_source,
        cohort_filter=cohort_filter,
        min_subjects=min_subjects,
        scale_method=scale_method,
    )


def fit_feature_anchors_from_features_df(
    features_df: pd.DataFrame,
    *,
    anchor_id: str,
    anchor_source: str = "all_subjects",
    cohort_filter: str = "",
    feature_standardization: Optional[Mapping[str, Sequence[str]]] = None,
    min_subjects: int = 3,
    scale_method: str = "iqr",
    subject_ids: Optional[Sequence[Any]] = None,
    file_ids: Optional[Sequence[Any]] = None,
    group_by_subject: Optional[Mapping[str, Any]] = None,
    feature_columns: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Fit a subject-balanced feature-anchor artifact from a merged features table."""
    if features_df is None or features_df.empty:
        return _build_anchor_artifact(
            subject_values={},
            feature_epoch_counts={},
            group_by_subject={},
            files_used=0,
            anchor_id=anchor_id,
            anchor_source=anchor_source,
            cohort_filter=cohort_filter,
            min_subjects=min_subjects,
            scale_method=scale_method,
        )

    feature_cols = [str(col) for col in (feature_columns or select_export_feature_columns(features_df))]
    subject_series = _subject_ids_from_features_df(features_df, subject_ids)
    file_series = _file_ids_from_features_df(features_df, file_ids)
    subject_values: Dict[str, Dict[str, list[float]]] = {}
    feature_epoch_counts: Dict[str, int] = {}
    group_map = _group_by_subject_from_features_df(features_df, subject_series, group_by_subject)
    files_used: set[tuple[str, str]] = set()

    for name in feature_cols:
        if name not in features_df.columns:
            continue
        raw_values = pd.to_numeric(features_df[name], errors="coerce").to_numpy(dtype=np.float32, copy=True)
        transformed = transform_values_for_anchor(
            raw_values,
            name,
            feature_standardization=feature_standardization,
        )
        finite_mask = np.isfinite(transformed) & subject_series.notna().to_numpy() & file_series.notna().to_numpy()
        feature_epoch_counts[name] = int(finite_mask.sum())
        if not finite_mask.any():
            continue
        temp = pd.DataFrame(
            {
                "subject": subject_series.to_numpy()[finite_mask],
                "file": file_series.to_numpy()[finite_mask],
                "value": transformed[finite_mask],
            }
        )
        file_medians = temp.groupby(["subject", "file"], sort=False)["value"].median()
        for (subject, file_id), median in file_medians.items():
            try:
                value = float(median)
            except Exception:
                continue
            if not np.isfinite(value):
                continue
            subject_key = _normalize_subject_id(subject)
            file_key = str(file_id).strip()
            if not file_key:
                continue
            subject_values.setdefault(subject_key, {}).setdefault(name, []).append(value)
            files_used.add((subject_key, file_key))

    return _build_anchor_artifact(
        subject_values=subject_values,
        feature_epoch_counts=feature_epoch_counts,
        group_by_subject=group_map,
        files_used=len(files_used),
        anchor_id=anchor_id,
        anchor_source=anchor_source,
        cohort_filter=cohort_filter,
        min_subjects=min_subjects,
        scale_method=scale_method,
    )


def save_anchor_file(anchor: Mapping[str, Any], path: Path) -> Path:
    """Write an anchor artifact as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(anchor, indent=2, sort_keys=True, ensure_ascii=False, default=_json_default), encoding="utf-8")
    return path


def load_anchor_file(path: Path) -> dict[str, Any]:
    """Load a JSON anchor artifact."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "features" not in data:
        raise ValueError(f"Invalid feature anchor file: {path}")
    return data


def anchor_mapping(
    anchor: Mapping[str, Any],
    *,
    scale_method: Optional[str] = None,
    min_subjects: int = 1,
) -> dict[str, dict[str, float | str]]:
    """Convert an anchor artifact to the mapping consumed by projection code."""
    spec = anchor.get("spec", {}) if isinstance(anchor, Mapping) else {}
    method = str(scale_method or (spec.get("scale_method") if isinstance(spec, Mapping) else None) or "iqr").strip().lower()
    key_by_method = {
        "iqr": "iqr_sigma",
        "mad": "mad_sigma",
        "qn": "qn_sigma",
        "std": "std_sigma",
    }
    scale_key = key_by_method.get(method, "scale")
    out: dict[str, dict[str, float | str]] = {}
    for item in anchor.get("features", []):  # type: ignore[union-attr]
        if not isinstance(item, Mapping):
            continue
        name = str(item.get("feature_name", "")).strip()
        if not name:
            continue
        n_subjects = int(item.get("n_subjects", 0) or 0)
        if n_subjects < int(min_subjects):
            continue
        center = item.get("center", item.get("q50"))
        scale = item.get(scale_key, item.get("scale"))
        try:
            center_f = float(center)
            scale_f = float(scale)
        except Exception:
            continue
        if not (np.isfinite(center_f) and np.isfinite(scale_f) and scale_f > 0):
            continue
        out[name] = {
            "center": center_f,
            "scale": scale_f,
            "scale_method": method,
            "anchor_id": str(spec.get("anchor_id", "")) if isinstance(spec, Mapping) else "",
            "anchor_hash": str(spec.get("anchor_hash", "")) if isinstance(spec, Mapping) else "",
        }
    return out


def summarize_group_separation(rows: Sequence[Mapping[str, Any]], value_keys: Iterable[str], group_key: str = "group") -> dict[str, Any]:
    """Compute simple two-or-more-group separation summaries for smoke reports."""
    result: dict[str, Any] = {}
    groups = sorted({str(r.get(group_key, "")).strip() for r in rows if str(r.get(group_key, "")).strip()})
    for key in value_keys:
        by_group: dict[str, np.ndarray] = {}
        for group in groups:
            vals = [float(r[key]) for r in rows if str(r.get(group_key, "")).strip() == group and key in r and np.isfinite(float(r[key]))]
            if vals:
                by_group[group] = np.asarray(vals, dtype=np.float64)
        if len(by_group) < 2:
            continue
        means = {g: float(np.nanmean(v)) for g, v in by_group.items()}
        counts = {g: int(v.size) for g, v in by_group.items()}
        ordered = sorted(means.items(), key=lambda kv: kv[1])
        low_g, low_m = ordered[0]
        high_g, high_m = ordered[-1]
        pooled = np.concatenate(list(by_group.values()))
        pooled_std = float(np.nanstd(pooled, ddof=1)) if pooled.size > 1 else float("nan")
        result[key] = {
            "means": means,
            "counts": counts,
            "max_min_delta": float(high_m - low_m),
            "max_group": high_g,
            "min_group": low_g,
            "pooled_std_effect": float((high_m - low_m) / pooled_std) if np.isfinite(pooled_std) and pooled_std > 0 else float("nan"),
        }
    return result
