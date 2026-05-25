from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import numpy as np

from core import config_loader

from .. import anchors
from .anchor_smoke import _projection_rows_for_h5


def run_anchor_sensitivity(
    *,
    h5_root: Path,
    dataset_id: str,
    config: Path,
    scale_methods: list[str],
    clip_thresholds: list[float],
    min_subjects: int = 3,
    max_files: int = 500,
) -> dict[str, Any]:
    """Run a small anchor scale/clip sensitivity sweep over summarized H5 files."""
    cfg = config_loader.load_config(Path(config))
    h5_paths = anchors.find_h5_files(Path(h5_root), dataset_id=dataset_id)[: int(max_files)]
    proj_cfg = cfg.get("mnps_projection", {}) if isinstance(cfg, Mapping) else {}
    fs = proj_cfg.get("feature_standardization", {}) if isinstance(proj_cfg, Mapping) else {}
    base_anchor = anchors.fit_feature_anchors_from_h5(
        h5_paths,
        anchor_id=f"{dataset_id}_sensitivity_v2_1",
        anchor_source="sensitivity_all_subjects",
        cohort_filter="all summarized H5 files discovered for dataset",
        feature_standardization=fs if isinstance(fs, Mapping) else None,
        min_subjects=min_subjects,
        scale_method=scale_methods[0] if scale_methods else "iqr",
    )
    runs = []
    for scale_method in scale_methods:
        anchor_map = anchors.anchor_mapping(base_anchor, scale_method=scale_method, min_subjects=min_subjects)
        for clip_threshold in clip_thresholds:
            cfg_variant = dict(cfg)
            proj_variant = dict(cfg_variant.get("mnps_projection", {}) or {})
            proj_variant["clip_threshold"] = float(clip_threshold)
            cfg_variant["mnps_projection"] = proj_variant
            rows = []
            for path in h5_paths:
                row = _projection_rows_for_h5(path, config=cfg_variant, anchor_map=anchor_map)
                if row:
                    rows.append(row)
            value_keys = [key for row in rows for key in row.keys() if key.endswith("_mean")]
            runs.append(
                {
                    "scale_method": scale_method,
                    "clip_threshold": float(clip_threshold),
                    "anchor_usable_features": int(len(anchor_map)),
                    "n_rows": int(len(rows)),
                    "separation": anchors.summarize_group_separation(rows, sorted(set(value_keys))),
                }
            )
    return {
        "schema_version": "mndm.anchor_sensitivity.v2.1",
        "dataset_id": dataset_id,
        "h5_files_found": int(len(h5_paths)),
        "anchor": base_anchor.get("spec", {}),
        "runs": runs,
    }


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MNDM 2.1 anchor sensitivity sweeps.")
    parser.add_argument("--h5-root", type=Path, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--scale-methods", type=str, default="iqr,mad,qn")
    parser.add_argument("--clip-thresholds", type=str, default="4,6,9")
    parser.add_argument("--min-subjects", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=500)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_anchor_sensitivity(
        h5_root=args.h5_root,
        dataset_id=args.dataset,
        config=args.config,
        scale_methods=[x.strip() for x in args.scale_methods.split(",") if x.strip()],
        clip_thresholds=_parse_float_list(args.clip_thresholds),
        min_subjects=args.min_subjects,
        max_files=args.max_files,
    )
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote anchor sensitivity report: {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
