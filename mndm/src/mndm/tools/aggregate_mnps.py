from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd


def _find_latest_mnps_run(ds_dir: Path) -> Optional[Path]:
    """Internal helper: find latest mnps run."""
    runs = [
        p
        for pattern in ("neuralmanifolddynamics_*", "mnps_*")
        for p in ds_dir.glob(pattern)
        if p.is_dir()
    ]
    if not runs:
        return None
    return max(runs, key=lambda p: p.stat().st_mtime)


def _find_summary_paths(processed_dir: Path, dataset_id: str, run_dir: Optional[Path] = None) -> List[Path]:
    """Internal helper: find summary paths."""
    ds_dir = processed_dir / dataset_id
    if run_dir is not None:
        target_run = run_dir if run_dir.is_absolute() else ds_dir / run_dir
        return sorted(target_run.glob("*/summary.json")) if target_run.exists() else []
    if not ds_dir.exists():
        return []
    latest = _find_latest_mnps_run(ds_dir)
    if latest is None:
        return []
    return sorted(latest.glob("*/summary.json"))


def _read_manifest(path: Path) -> Dict[str, Any]:
    """Internal helper: read manifest."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _attr_str(value: Any) -> str:
    """Internal helper: attr str."""
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    return str(value)


def _nanmean_columns(ds: h5py.Dataset, chunk_rows: int = 8192) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-column nanmean components (sum, count) chunk-wise."""
    shape = tuple(ds.shape)
    if len(shape) != 2 or shape[0] == 0:
        return np.zeros((shape[1] if len(shape) > 1 else 0,), dtype=np.float64), np.zeros((shape[1] if len(shape) > 1 else 0,), dtype=np.float64)
    n_cols = int(shape[1])
    sums = np.zeros((n_cols,), dtype=np.float64)
    counts = np.zeros((n_cols,), dtype=np.float64)
    for start in range(0, int(shape[0]), int(chunk_rows)):
        stop = min(int(shape[0]), start + int(chunk_rows))
        block = np.asarray(ds[start:stop, :], dtype=np.float64)
        finite = np.isfinite(block)
        sums += np.where(finite, block, 0.0).sum(axis=0)
        counts += finite.sum(axis=0)
    return sums, counts


def _read_h5_means(h5_path: Path) -> Dict[str, Any]:
    """Internal helper: read h5 means."""
    out: Dict[str, Any] = {}
    with h5py.File(h5_path, "r") as h5:
        # Model/provenance fields to prevent silent model-mix downstream.
        for attr in (
            "x_definition",
            "v2_definition",
            "weights_hash_direct",
            "subcoords_hash_v2",
            "normalize_mode",
            "env_hash",
        ):
            if attr in h5.attrs:
                out[attr] = _attr_str(h5.attrs.get(attr))
        # m/d/e composites
        if "mnps_3d" in h5:
            ds_x = h5["mnps_3d"]
            if isinstance(ds_x, h5py.Dataset) and len(ds_x.shape) == 2 and ds_x.shape[1] >= 3:
                sums, counts = _nanmean_columns(ds_x)
                means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
                out["m_mean"] = float(means[0]) if means.size > 0 else float("nan")
                out["d_mean"] = float(means[1]) if means.size > 1 else float("nan")
                out["e_mean"] = float(means[2]) if means.size > 2 else float("nan")
        # v2 subcoordinates
        if "coords_9d" in h5:
            grp = h5["coords_9d"]
            if "values" in grp:
                ds_vals = grp["values"]
                names: Optional[List[str]] = None
                if "names" in grp:
                    # stored as bytes; decode
                    raw = grp["names"][...]
                    names = [
                        bytes(n).decode("utf-8", errors="replace")
                        if isinstance(n, (bytes, bytearray, np.bytes_))
                        else str(n)
                        for n in raw
                    ]
                if isinstance(ds_vals, h5py.Dataset) and len(ds_vals.shape) == 2 and names:
                    sums, counts = _nanmean_columns(ds_vals)
                    means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
                    total_rows = int(ds_vals.shape[0]) if ds_vals.shape[0] else 0
                    for name, mean_val in zip(names, means):
                        out[f"v2_{name}_mean"] = float(mean_val)
                    for name, cnt in zip(names, counts):
                        out[f"v2_{name}_n_finite"] = int(cnt)
                        out[f"v2_{name}_nan_frac"] = float(1.0 - (cnt / total_rows)) if total_rows > 0 else 1.0
    return out


def _flatten_participant_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Internal helper: flatten participant meta."""
    flat: Dict[str, Any] = {}
    for k, v in meta.items():
        # simple scalar fields only
        if isinstance(v, (str, int, float)):
            flat[f"meta_{k}"] = v
    return flat


def _derive_category(meta: Dict[str, Any], preferred_key: Optional[str] = None) -> Dict[str, Any]:
    """Derive generic category fields from participant_meta.

    Returns both raw and normalized labels when possible.
    """
    # Candidate keys to consider in order
    key_candidates = []
    if preferred_key:
        key_candidates.append(preferred_key)
        key_candidates.append(preferred_key.lower())
    key_candidates += ["Group", "group", "Diagnosis", "diagnosis", "condition", "Condition", "Cohort", "cohort"]

    raw_val: Optional[str] = None
    for key in key_candidates:
        if key in meta and isinstance(meta[key], (str, int, float)):
            raw_val = str(meta[key]).strip()
            break

    result: Dict[str, Any] = {"category_raw": raw_val, "category_normalized": None}
    if not raw_val:
        return result

    # Simple normalization map (extend as needed)
    norm_map = {
        "pd": "Parkinson",
        "parkinson": "Parkinson",
        "ctl": "Control",
        "control": "Control",
        "hc": "Healthy",
        "healthy": "Healthy",
        "ad": "AD",
        "alzheimer": "AD",
        "patient": "Patient",
        "case": "Patient",
        "subject": "Subject",
    }
    key = raw_val.lower().replace("-", "").replace("_", "").strip()
    result["category_normalized"] = norm_map.get(key, raw_val)
    return result


def _match_h5_for_summary(summary_path: Path, manifest: Mapping[str, Any]) -> Optional[Path]:
    """Internal helper: match h5 for summary."""
    direct = summary_path.with_suffix(".h5")
    if direct.exists():
        return direct
    for key in ("h5_path", "h5_name", "run_h5_name"):
        raw = manifest.get(key) if isinstance(manifest, Mapping) else None
        if isinstance(raw, str) and raw.strip():
            p = Path(raw)
            cand = p if p.is_absolute() else (summary_path.parent / p)
            if cand.exists():
                return cand
    candidates = sorted(summary_path.parent.glob("*.h5"))
    if not candidates:
        return None
    stem_matches = [p for p in candidates if p.stem == summary_path.stem]
    if stem_matches:
        return stem_matches[0]
    return candidates[0]


def aggregate(
    processed_dir: Path,
    dataset_id: str,
    out_csv: Path,
    category_key: Optional[str] = None,
    run_dir: Optional[Path] = None,
) -> int:
    """Handle aggregate."""
    summaries = _find_summary_paths(processed_dir, dataset_id, run_dir=run_dir)
    if not summaries:
        print(f"No summary.json files found under {processed_dir / dataset_id}", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    for summary_path in summaries:
        try:
            manifest = _read_manifest(summary_path)
            row: Dict[str, Any] = {}

            # Basic identifiers
            row["dataset_label"] = manifest.get("dataset_id")
            row["subject"] = manifest.get("subject")
            row["session"] = manifest.get("session")
            row["samples"] = manifest.get("samples")
            row["summary_relpath"] = str(summary_path)

            # Jacobian meta (from summary)
            meta_indices = manifest.get("meta_indices", {}) or {}
            row["jac_mean_trace"] = meta_indices.get("mean_trace")
            row["jac_mean_rotation_fro"] = meta_indices.get("mean_rotation_fro")
            row["jac_windows"] = meta_indices.get("windows")

            # Participant meta (flatten a few useful fields if present)
            pmeta = manifest.get("participant_meta", {}) or {}
            row.update(_flatten_participant_meta(pmeta))
            row.update(_derive_category(pmeta, preferred_key=category_key))

            h5_path = _match_h5_for_summary(summary_path, manifest)
            if h5_path is not None:
                row["h5_path"] = str(h5_path)
                h5_means = _read_h5_means(h5_path)
                row.update(h5_means)
            else:
                print(f"Warning: no H5 matched for {summary_path}", file=sys.stderr)

            rows.append(row)
        except Exception as exc:
            print(f"Warning: failed to aggregate {summary_path}: {exc}", file=sys.stderr)
            continue

    if not rows:
        print("No rows aggregated", file=sys.stderr)
        return 1

    # Union of all keys
    all_keys: List[str] = sorted({k for r in rows for k in r.keys()})
    df = pd.DataFrame(rows, columns=all_keys)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote aggregate: {out_csv} ({len(df)} rows, {len(df.columns)} columns)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Program entry point."""
    p = argparse.ArgumentParser(description="Aggregate MNPS summaries (JSON+H5) into a CSV")
    p.add_argument("--processed-dir", type=Path, required=True, help="Base processed directory (e.g., E:/Science_Datasets/openneuro/processed)")
    p.add_argument("--dataset", required=True, help="Dataset id (e.g., ds003490)")
    p.add_argument("--run-dir", type=Path, default=None, help="Optional specific run directory (absolute or relative to <processed>/<dataset>); default uses latest summarized run")
    p.add_argument("--out", type=Path, default=None, help="Output CSV path (default: <processed>/<dataset>/mnps_aggregate.csv)")
    p.add_argument("--category-key", type=str, default=None, help="Preferred participant_meta key to use for category (e.g., Group, Diagnosis)")
    args = p.parse_args(argv)

    out_path = args.out or (args.processed_dir / args.dataset / "mnps_aggregate.csv")
    return aggregate(args.processed_dir, args.dataset, out_path, category_key=args.category_key, run_dir=args.run_dir)


if __name__ == "__main__":
    raise SystemExit(main())

