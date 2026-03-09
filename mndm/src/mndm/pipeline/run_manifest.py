"""Run-level manifest writer for MNPS outputs.

Goal: write a single JSON file per mnps_<dataset>_<timestamp> run directory that
summarizes what artifacts exist and what capabilities are present (MNPS 3D/9D,
MNJ/Jacobians, regional exports, counts, and a small config excerpt + digest).
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from core.io import json_writer
from .. import bids_index

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_json_dumps(obj: Any) -> str:
    # json_writer already knows how to coerce numpy scalars/arrays, so we re-use that.
    coerced = json_writer._to_jsonable(obj)  # type: ignore[attr-defined]
    return json.dumps(coerced, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pick(d: Mapping[str, Any], key: str, default: Any = None) -> Any:
    try:
        return d.get(key, default)
    except Exception:
        return default


def _config_excerpt(config: Mapping[str, Any], ds_id: str) -> Dict[str, Any]:
    """Keep this intentionally small + high-signal."""
    download_ds = _pick(_pick(_pick(config, "download", {}), "datasets", {}), ds_id, None)
    preprocess_ds = _pick(_pick(_pick(config, "preprocess", {}), "datasets", {}), ds_id, None)
    fmri_pre_ds = _pick(_pick(_pick(_pick(config, "preprocess", {}), "fmri", {}), "datasets", {}), ds_id, None)
    epoch_ds = _pick(_pick(_pick(config, "epoching", {}), "datasets", {}), ds_id, None)
    meta_ds = _pick(_pick(_pick(config, "metadata_extraction", {}), "datasets", {}), ds_id, None)
    mnps_9d_ds = _pick(_pick(_pick(config, "mnps_9d", {}), "datasets", {}), ds_id, None)

    robustness = _pick(config, "robustness", {})
    coverage = _pick(robustness, "coverage", {})
    coverage_ds = _pick(_pick(coverage, "datasets", {}), ds_id, None)

    excerpt: Dict[str, Any] = {
        "mnps": _pick(config, "mnps", {}),
        "mnps_9d": {
            "enabled": bool(_pick(_pick(config, "mnps_9d", {}), "enabled", False)),
            "jacobian": _pick(_pick(config, "mnps_9d", {}), "jacobian", {}),
            "subcoords_keys": sorted(list(_pick(_pick(config, "mnps_9d", {}), "subcoords", {}).keys()))
            if isinstance(_pick(_pick(config, "mnps_9d", {}), "subcoords", {}), Mapping)
            else [],
            "dataset_override": mnps_9d_ds,
        },
        "mnps_projection": _pick(config, "mnps_projection", {}),
        "mnps_extensions": {"keys": sorted(list(_pick(config, "mnps_extensions", {}).keys()))}
        if isinstance(_pick(config, "mnps_extensions", {}), Mapping)
        else {},
        "robustness": {"coverage": {"default": {k: coverage.get(k) for k in ("min_seconds", "min_epochs")}, "dataset_override": coverage_ds}},
        "dataset_cfg": {
            "download": download_ds,
            "preprocess": preprocess_ds,
            "preprocess_fmri": fmri_pre_ds,
            "epoching": epoch_ds,
            "metadata_extraction": meta_ds,
        },
        "source": _pick(config, "source", {}),
    }
    return excerpt


def _list_files(root: Path, pattern: str) -> list[Path]:
    try:
        return [p for p in root.rglob(pattern) if p.is_file()]
    except Exception:
        return []


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _field_guide() -> Dict[str, Any]:
    """Human-friendly dictionary for key H5 paths/fields."""
    return {
        "mnps_axes": {
            "order": ["m", "d", "e"],
            "definitions": {
                "m": "metastability / mobility",
                "d": "deviation from optimal integration-segregation balance",
                "e": "entropy / entropic energy",
            },
        },
        "h5_paths": {
            "time": "Per-epoch time index [T] in seconds.",
            "mnps_3d": "Canonical 3D MNPS trajectory [T,3] in axis order [m,d,e].",
            "mnps_3d_dot": "Temporal derivative of mnps_3d [T,3].",
            "window_start": "Per-epoch window start times [T] in seconds.",
            "window_end": "Per-epoch window end times [T] in seconds.",
            "nn/indices": "k-nearest-neighbor indices used for local Jacobian estimation.",
            "jacobian/J_hat": "Windowed Jacobian tensor [W,3,3] on primary MNPS.",
            "jacobian/J_dot": "Temporal derivative of Jacobian [W,3,3].",
            "jacobian/centers": "Center indices of Jacobian windows [W].",
            "jacobian_9D/J_hat": "Windowed Jacobian on stratified coordinates (dimension depends on config).",
            "jacobian_9D/J_dot": "Temporal derivative of jacobian_9D.",
            "jacobian_9D/cross_partials/*": "Named cross-partial timeseries extracted from jacobian_9D.",
            "coords_9d/values": "Stratified MNPS trajectory matrix [T,K] (often K=9).",
            "coords_9d/names": "Column names for coords_9d/values.",
            "features_raw/values": "Per-epoch raw feature matrix [T,K] in original scale.",
            "features_raw/names": "Column names for features_raw/values.",
            "features_raw/metadata/*": "Machine-readable per-feature metadata aligned to features_raw/names.",
            "features_robust_z/values": "Per-epoch strict robust-z feature matrix [T,K] (no log10/clip baked in).",
            "features_robust_z/names": "Column names for features_robust_z/values.",
            "features_robust_z/metadata/*": "Machine-readable per-feature metadata aligned to features_robust_z/names.",
            "regional_mnps/<network>/mnps": "Per-network MNPS trajectory [T,3].",
            "regional_mnps/<network>/mnps_dot": "Derivative of regional MNPS trajectory [T,3].",
            "regional_mnps/<network>/jacobian": "Per-network Jacobian tensor [W,3,3].",
            "regional_mnps/<network>/stratified": "Per-network stratified trajectory [T,K] when enabled.",
            "extensions/tabular_exports/*": "Columnar exports of CSV-style summary tables embedded into H5.",
            "labels/stage": "Optional stage labels aligned to time.",
            "events/*": "Optional event arrays (indices or timestamps).",
            "regions/bold": "Optional regional signals (e.g., fMRI ROI x time).",
            "regions/names": "Names corresponding to rows in regions/bold.",
        },
        "naming_notes": {
            "jacobian": "MNJ estimate from primary 3D MNPS.",
            "coords_9d": "Stratified MNPS coordinate group (often 9D).",
            "mnps_9d": "Configuration/runtime term for stratified MNPS coordinate system.",
            "capabilities.regional_outputs": "True when H5 files embed derived `/regional_mnps/*` network-level outputs.",
            "capabilities.raw_region_signals": "True when H5 files embed raw `/regions/*` signals (typically fMRI ROI x time).",
            "capabilities.regional_outputs_path": "Canonical regional output path for all modalities.",
            "capabilities.raw_features": "True when H5 files embed `/features_raw/*` feature exports.",
            "capabilities.robust_z_features": "True when H5 files embed `/features_robust_z/*` feature exports.",
        },
        "source": {
            "source": "Dataset provenance metadata declared in the active YAML config.",
            "doi": "Dataset DOI copied from config when available for acknowledgements and traceability.",
        },
    }


def _resolve_source_info(config: Mapping[str, Any], ds_id: str) -> Dict[str, Any]:
    raw = _pick(config, "source", {})
    src = dict(raw) if isinstance(raw, Mapping) else {}

    if not src.get("dataset_id"):
        src["dataset_id"] = ds_id

    name = str(src.get("name", "") or "").strip().lower()
    if name in {"openneuro", "open_neuro"} and not src.get("url"):
        src["url"] = f"https://openneuro.org/datasets/{ds_id}"

    return src


def _merge_summary_meta(d: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge common summary metadata containers into one lookup dict."""
    meta: Dict[str, Any] = dict(d) if isinstance(d, Mapping) else {}
    for key in ("attrs", "manifest_extra", "meta", "payload"):
        block = d.get(key) if isinstance(d, Mapping) else None
        if isinstance(block, Mapping):
            meta.update(dict(block))
    return meta


def _summarize_subject_jsons(summary_jsons: Sequence[Path]) -> Dict[str, Any]:
    subjects: set[str] = set()
    tasks: set[str] = set()
    conditions: set[str] = set()
    groups: set[str] = set()
    labels_keys: set[str] = set()
    present_counts: Dict[str, int] = {
        "mnps": 0,
        "jacobian": 0,
        "tier2_jacobian": 0,
        "meta_indices": 0,
        "meta_indices_v2": 0,
        "jacobian_9D": 0,
        "coords_9d": 0,
    }

    for p in summary_jsons:
        d = _read_json(p)
        if not isinstance(d, dict):
            continue
        meta = _merge_summary_meta(d)
        sub = meta.get("subject")
        if isinstance(sub, str) and sub:
            subjects.add(sub)
        task = meta.get("task")
        if isinstance(task, str) and task:
            tasks.add(task)
        cond = meta.get("condition")
        if isinstance(cond, str) and cond:
            conditions.add(cond)
        grp = meta.get("group")
        if isinstance(grp, str) and grp:
            groups.add(grp)
        labels = meta.get("labels")
        if isinstance(labels, dict):
            labels_keys.update(str(k) for k in labels.keys())
        elif isinstance(labels, list):
            labels_keys.update(str(k) for k in labels)
        payload_labels = _pick(_pick(meta, "payload", {}), "labels", None)
        if isinstance(payload_labels, dict):
            labels_keys.update(str(k) for k in payload_labels.keys())

        # Presence flags for quick capability inference from JSON alone
        for k in list(present_counts.keys()):
            try:
                if k in meta and meta.get(k) is not None:
                    present_counts[k] += 1
            except Exception:
                pass

    return {
        "subjects": {"count": len(subjects), "unique": sorted(subjects)},
        "tasks": {"count": len(tasks), "unique": sorted(tasks)},
        "conditions": {"count": len(conditions), "unique": sorted(conditions)},
        "groups": {"count": len(groups), "unique": sorted(groups)},
        "labels_keys": sorted(labels_keys),
        "summary_presence_counts": present_counts,
    }


def _probe_h5_capabilities(h5_paths: Sequence[Path], max_files: int = 200) -> Dict[str, Any]:
    """Lightweight probe: open some H5s and check for key paths + shapes.

    We only read shapes/keys (no heavy data reads).
    """
    try:
        import h5py  # type: ignore
    except Exception as exc:
        return {"error": f"h5py unavailable: {exc}", "probed": 0}

    probed = 0
    mnps_3d_dims: Dict[int, int] = {}
    jac_dims: Dict[int, int] = {}
    v2_dims: Dict[int, int] = {}
    jac_v2_dims: Dict[int, int] = {}
    has_raw_region_signals_count = 0
    has_regional_outputs_count = 0
    has_stage_count = 0
    has_v2_like_count = 0
    has_raw_features_count = 0
    has_robust_z_features_count = 0
    bad_files: list[str] = []

    for p in h5_paths[: max_files if max_files > 0 else len(h5_paths)]:
        try:
            with h5py.File(p, "r") as f:
                probed += 1
                # MNPS coordinate matrix (primary)
                if "mnps_3d" in f:
                    try:
                        d = int(f["mnps_3d"].shape[1])
                        mnps_3d_dims[d] = mnps_3d_dims.get(d, 0) + 1
                    except Exception:
                        pass
                # Jacobian estimate for MNJ / dynamics summaries
                if "jacobian" in f and "J_hat" in f["jacobian"]:
                    try:
                        d = int(f["jacobian"]["J_hat"].shape[1])
                        jac_dims[d] = jac_dims.get(d, 0) + 1
                    except Exception:
                        pass

                # MNPS v2 subcoordinates (typically 9D: 3 axes × 3 subcoords)
                if "coords_9d" in f and "values" in f["coords_9d"]:
                    try:
                        d = int(f["coords_9d"]["values"].shape[1])
                        v2_dims[d] = v2_dims.get(d, 0) + 1
                        has_v2_like_count += 1
                    except Exception:
                        pass

                if "features_raw" in f:
                    try:
                        g = f["features_raw"]
                        if "values" in g and "names" in g:
                            has_raw_features_count += 1
                    except Exception:
                        pass

                if "features_robust_z" in f:
                    try:
                        g = f["features_robust_z"]
                        if "values" in g and "names" in g:
                            has_robust_z_features_count += 1
                    except Exception:
                        pass

                # Optional Stratified Jacobian computed on v2 coordinates
                if "jacobian_9D" in f and "J_hat" in f["jacobian_9D"]:
                    try:
                        d = int(f["jacobian_9D"]["J_hat"].shape[1])
                        jac_v2_dims[d] = jac_v2_dims.get(d, 0) + 1
                        has_v2_like_count += 1
                    except Exception:
                        pass

                # Stage labels (sleep)
                if "labels" in f and "stage" in f["labels"]:
                    has_stage_count += 1

                # Raw region signals export (typically fMRI ROI time series)
                if "regions" in f:
                    try:
                        g = f["regions"]
                        if "bold" in g or "names" in g:
                            has_raw_region_signals_count += 1
                    except Exception:
                        pass

                # Derived regional MNPS/MNJ outputs (EEG or fMRI)
                if "regional_mnps" in f:
                    try:
                        if len(f["regional_mnps"].keys()) > 0:
                            has_regional_outputs_count += 1
                    except Exception:
                        pass

        except Exception:
            bad_files.append(str(p))

    # Primary MNPS trajectory is always 3D in this pipeline; "9D MNPS" refers to coords_9d.
    mnps_3d = any(dim == 3 for dim in mnps_3d_dims.keys())
    mnps_9d = any(dim == 9 for dim in v2_dims.keys())
    jac_3d = any(dim == 3 for dim in jac_dims.keys())
    jac_9d = any(dim == 9 for dim in jac_v2_dims.keys())

    return {
        "probed": probed,
        "bad_files": bad_files[:20],
        "mnps_3d_dims_counts": {str(k): int(v) for k, v in sorted(mnps_3d_dims.items())},
        "jacobian_dims_counts": {str(k): int(v) for k, v in sorted(jac_dims.items())},
        "coords_9d_dims_counts": {str(k): int(v) for k, v in sorted(v2_dims.items())},
        "jacobian_9D_dims_counts": {str(k): int(v) for k, v in sorted(jac_v2_dims.items())},
        "mnps3d": bool(mnps_3d),
        "mnps9d": bool(mnps_9d),
        "mnj": bool(jac_3d or jac_9d),
        "mnj_3d": bool(jac_3d),
        "mnj_9d": bool(jac_9d),
        "regional_outputs": bool(has_regional_outputs_count > 0),
        "regional_outputs_path": "/regional_mnps",
        "raw_region_signals": bool(has_raw_region_signals_count > 0),
        "raw_features": bool(has_raw_features_count > 0),
        "raw_features_path": "/features_raw",
        "robust_z_features": bool(has_robust_z_features_count > 0),
        "robust_z_features_path": "/features_robust_z",
        "labels_stage": bool(has_stage_count > 0),
        "v2_like_artifacts": bool(has_v2_like_count > 0),
        "counts": {
            "h5_with_regional_outputs": int(has_regional_outputs_count),
            "h5_with_raw_region_signals": int(has_raw_region_signals_count),
            "h5_with_raw_features": int(has_raw_features_count),
            "h5_with_robust_z_features": int(has_robust_z_features_count),
            "h5_with_stage": int(has_stage_count),
            "h5_with_v2_like": int(has_v2_like_count),
        },
    }


def write_run_manifest(
    *,
    mnps_dir: Path,
    config: Mapping[str, Any],
    ds_id: str,
    received_dir: Path,
    processed_dir: Path,
    h5_mode: str,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    mnps_dir = Path(mnps_dir)
    ds_received_root = bids_index.resolve_dataset_root(config, Path(received_dir), ds_id)
    ds_processed_root = Path(processed_dir) / ds_id

    # Enumerate artifacts
    h5_files = _list_files(mnps_dir, "*.h5")
    summary_jsons = _list_files(mnps_dir, "summary.json")
    qc_summary_jsons = _list_files(mnps_dir, "qc_summary.json")
    qc_reliability_jsons = _list_files(mnps_dir, "qc_reliability.json")
    regional_csvs = _list_files(mnps_dir, "*regional*.csv")
    block_csvs = _list_files(mnps_dir, "*block*jacobian*.csv")

    # Split block-jacobian CSVs into "regional" vs "stratified/v2" variants
    regional_block_csvs = [p for p in block_csvs if "regional" in p.name.lower()]
    stratified_block_csvs = [p for p in block_csvs if "stratified" in p.name.lower() or "v2" in p.name.lower()]

    excerpt = _config_excerpt(config, ds_id)
    config_digest = _sha256_text(_safe_json_dumps(config))
    excerpt_digest = _sha256_text(_safe_json_dumps(excerpt))
    source_info = _resolve_source_info(config, ds_id)

    probe = _probe_h5_capabilities(h5_files, max_files=200)
    subj = _summarize_subject_jsons(summary_jsons)

    # Best-effort git revision (optional)
    git_rev = None
    try:
        import subprocess

        def _find_git_root(start: Path) -> Optional[Path]:
            cur = start.resolve()
            for candidate in [cur, *cur.parents]:
                if (candidate / ".git").exists():
                    return candidate
            return None

        git_root = _find_git_root(Path(__file__).resolve())
        if git_root is not None:
            git_rev = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(git_root))
                .decode("utf-8")
                .strip()
            )
    except Exception:
        git_rev = None

    manifest: Dict[str, Any] = {
        "schema": "mndm.run_manifest.v2",
        "created_at": _utc_now_iso(),
        "dataset_id": ds_id,
        "run_dir": str(mnps_dir),
        "run_name": mnps_dir.name,
        "h5_mode": str(h5_mode),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_rev": git_rev,
        },
        "paths": {
            "received_dir": str(received_dir),
            "processed_dir": str(processed_dir),
            "dataset_received_root": str(ds_received_root),
            "dataset_processed_root": str(ds_processed_root),
        },
        "counts": {
            "h5": int(len(h5_files)),
            "summary_json": int(len(summary_jsons)),
            "qc_summary_json": int(len(qc_summary_jsons)),
            "qc_reliability_json": int(len(qc_reliability_jsons)),
            "regional_csv": int(len(regional_csvs)),
            "block_csv": int(len(block_csvs)),
        },
        "capabilities": {
            **probe,
            "block_jacobians": {
                "any": bool(block_csvs),
                "regional_any": bool(regional_block_csvs),
                "stratified_any": bool(stratified_block_csvs),
                "regional_csv_paths": [str(p.relative_to(mnps_dir)) for p in sorted(regional_block_csvs)],
                "stratified_csv_paths": [str(p.relative_to(mnps_dir)) for p in sorted(stratified_block_csvs)],
            },
            "regional_exports": {
                "any": bool(regional_csvs),
                "csv_paths": [str(p.relative_to(mnps_dir)) for p in sorted(regional_csvs)],
            },
        },
        "subjects_index": subj,
        "config": {
            "digest_sha256": config_digest,
            "excerpt_digest_sha256": excerpt_digest,
            "excerpt": excerpt,
            "top_level_keys": sorted(list(config.keys())) if isinstance(config, Mapping) else [],
        },
        "source": source_info,
        "doi": source_info.get("doi"),
        "field_guide": _field_guide(),
    }

    if extra:
        try:
            manifest["extra"] = dict(extra)
        except Exception:
            manifest["extra"] = {"note": "extra was not JSON-serializable"}

    out_path = mnps_dir / "run_manifest.json"
    return json_writer.write_json_summary(manifest, out_path)

