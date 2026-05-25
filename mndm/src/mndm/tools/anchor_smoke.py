from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import h5py
import numpy as np
import pandas as pd

from core import config_loader

from .. import anchors, projection


DEFAULT_SMOKE_DATASETS = ("ds003478", "ds003944", "ds004504")
SMOKE_SUBCOORDS = ("e_e", "e_s", "d_l")


def _repo_config_path(dataset_id: str) -> Path:
    return Path(__file__).resolve().parents[3] / "config" / f"config_ingest_{dataset_id}.yaml"


def _load_dataset_config(dataset_id: str, config: Optional[Path]) -> dict[str, Any]:
    path = Path(config) if config is not None else _repo_config_path(dataset_id)
    return config_loader.load_config(path)


def _raw_dataset_exists(data_dir: Optional[Path], dataset_id: str) -> bool:
    if data_dir is None:
        return False
    root = Path(data_dir)
    return (root / dataset_id).exists()


def _group_from_h5(h5: h5py.File) -> str:
    for key in ("group", "meta_Group", "meta_group", "mapped_group", "condition"):
        val = h5.attrs.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _subject_from_h5(h5: h5py.File, path: Path) -> str:
    for key in ("subject_id", "subject", "meta_participant_id"):
        val = h5.attrs.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return path.parent.name.split("_")[0] or path.stem


def _decode_names(raw: Any) -> list[str]:
    names: list[str] = []
    for item in raw:
        if isinstance(item, (bytes, bytearray, np.bytes_)):
            names.append(bytes(item).decode("utf-8", errors="replace"))
        else:
            names.append(str(item))
    return names


def _frame_from_features_raw(h5: h5py.File) -> Optional[pd.DataFrame]:
    if "features_raw" not in h5:
        return None
    grp = h5["features_raw"]
    if "values" not in grp or "names" not in grp:
        return None
    values = np.asarray(grp["values"], dtype=np.float32)
    names = _decode_names(grp["names"][...])
    if values.ndim != 2 or values.shape[1] != len(names):
        return None
    return pd.DataFrame(values, columns=names)


def _projection_rows_for_h5(
    h5_path: Path,
    *,
    config: Mapping[str, Any],
    anchor_map: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    proj_cfg = config.get("mnps_projection", {}) if isinstance(config, Mapping) else {}
    v2_cfg = config.get("mnps_9d", {}) if isinstance(config, Mapping) else {}
    feature_standardization = proj_cfg.get("feature_standardization", {}) if isinstance(proj_cfg, Mapping) else {}
    subcoords = v2_cfg.get("subcoords", {}) if isinstance(v2_cfg, Mapping) else {}
    if not isinstance(subcoords, Mapping) or not subcoords:
        return None
    normalize_mode = str(proj_cfg.get("normalize", "robust_z")) if isinstance(proj_cfg, Mapping) else "robust_z"
    clip_threshold = float(proj_cfg.get("clip_threshold", 6.0)) if isinstance(proj_cfg, Mapping) else 6.0

    with h5py.File(h5_path, "r") as h5:
        frame = _frame_from_features_raw(h5)
        if frame is None or frame.empty:
            return None
        local, names, _ = projection.project_features_v2(
            frame,
            subcoords,
            normalize=normalize_mode,
            feature_standardization=feature_standardization if isinstance(feature_standardization, Mapping) else None,
            clip_threshold=clip_threshold,
        )
        cohort, cohort_names, _ = projection.project_features_v2(
            frame,
            subcoords,
            normalize=normalize_mode,
            feature_standardization=feature_standardization if isinstance(feature_standardization, Mapping) else None,
            clip_threshold=clip_threshold,
            external_anchor=anchor_map,
        )
        row: dict[str, Any] = {
            "h5_path": str(h5_path),
            "subject": _subject_from_h5(h5, h5_path),
            "group": _group_from_h5(h5),
            "n_epochs": int(frame.shape[0]),
        }
        for subcoord in SMOKE_SUBCOORDS:
            if subcoord in names:
                idx = names.index(subcoord)
                row[f"subject_{subcoord}_mean"] = float(np.nanmean(local[:, idx]))
            if subcoord in cohort_names:
                idx = cohort_names.index(subcoord)
                row[f"cohort_{subcoord}_mean"] = float(np.nanmean(cohort[:, idx]))
        return row


def smoke_dataset(
    dataset_id: str,
    *,
    data_dir: Optional[Path],
    h5_root: Optional[Path],
    config: Optional[Path],
    min_subjects: int = 3,
    max_files: int = 500,
) -> dict[str, Any]:
    """Run a lightweight anchored-coordinate smoke test for one dataset."""
    cfg = _load_dataset_config(dataset_id, config)
    h5_paths = (
        anchors.find_h5_files(Path(h5_root), dataset_id=dataset_id)[: int(max_files)]
        if h5_root is not None
        else []
    )
    proj_cfg = cfg.get("mnps_projection", {}) if isinstance(cfg, Mapping) else {}
    feature_standardization = proj_cfg.get("feature_standardization", {}) if isinstance(proj_cfg, Mapping) else {}
    anchor = anchors.fit_feature_anchors_from_h5(
        h5_paths,
        anchor_id=f"{dataset_id}_smoke_v2_1",
        anchor_source="smoke_all_subjects",
        cohort_filter="all summarized H5 files discovered for dataset",
        feature_standardization=feature_standardization if isinstance(feature_standardization, Mapping) else None,
        min_subjects=min_subjects,
        scale_method="iqr",
    )
    anchor_map = anchors.anchor_mapping(anchor, scale_method="iqr", min_subjects=min_subjects)
    rows = []
    for path in h5_paths:
        row = _projection_rows_for_h5(path, config=cfg, anchor_map=anchor_map)
        if row:
            rows.append(row)
    value_keys = [
        f"{layer}_{subcoord}_mean"
        for layer in ("subject", "cohort")
        for subcoord in SMOKE_SUBCOORDS
    ]
    return {
        "dataset_id": dataset_id,
        "raw_dataset_exists": _raw_dataset_exists(data_dir, dataset_id),
        "h5_files_found": int(len(h5_paths)),
        "subjects_with_rows": int(len({r.get("subject") for r in rows})),
        "groups": sorted({str(r.get("group", "")) for r in rows if str(r.get("group", "")).strip()}),
        "anchor": anchor.get("spec", {}),
        "anchor_usable_features": int(len(anchor_map)),
        "rows": rows,
        "separation": anchors.summarize_group_separation(rows, value_keys),
    }


def smoke_datasets(
    *,
    datasets: Sequence[str] = DEFAULT_SMOKE_DATASETS,
    data_dir: Optional[Path] = None,
    h5_root: Optional[Path] = None,
    config: Optional[Path] = None,
    min_subjects: int = 3,
    max_files: int = 500,
) -> dict[str, Any]:
    return {
        "schema_version": "mndm.anchor_smoke.v2.1",
        "datasets": [
            smoke_dataset(
                ds,
                data_dir=data_dir,
                h5_root=h5_root,
                config=config,
                min_subjects=min_subjects,
                max_files=max_files,
            )
            for ds in datasets
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smoke-test MNDM 2.1 cohort anchoring on summarized H5 outputs.")
    parser.add_argument("--dataset", nargs="*", default=list(DEFAULT_SMOKE_DATASETS))
    parser.add_argument("--data-dir", type=Path, default=Path("M:/datasets/received/openneuro"))
    parser.add_argument("--h5-root", type=Path, default=None, help="Processed root or run directory containing H5 files.")
    parser.add_argument("--config", type=Path, default=None, help="Optional single config path; normally inferred per dataset.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON report path.")
    parser.add_argument("--min-subjects", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=500)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = smoke_datasets(
        datasets=args.dataset,
        data_dir=args.data_dir,
        h5_root=args.h5_root,
        config=args.config,
        min_subjects=args.min_subjects,
        max_files=args.max_files,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote anchor smoke report: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
