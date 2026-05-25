from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional

from core import config_loader

from .. import anchors


def _feature_standardization_from_config(config_path: Optional[Path]) -> Mapping[str, Any]:
    if config_path is None:
        return {}
    cfg = config_loader.load_config(Path(config_path))
    proj_cfg = cfg.get("mnps_projection", {}) if isinstance(cfg, Mapping) else {}
    fs = proj_cfg.get("feature_standardization", {}) if isinstance(proj_cfg, Mapping) else {}
    return fs if isinstance(fs, Mapping) else {}


def fit_anchors_command(
    *,
    h5_root: Path,
    out: Path,
    dataset_id: Optional[str] = None,
    config: Optional[Path] = None,
    anchor_id: str,
    anchor_source: str = "all_subjects",
    cohort_filter: str = "",
    scale_method: str = "iqr",
    min_subjects: int = 3,
) -> Path:
    """Fit and write a MNDM 2.1 feature-anchor artifact."""
    h5_paths = anchors.find_h5_files(Path(h5_root), dataset_id=dataset_id)
    feature_standardization = _feature_standardization_from_config(config)
    artifact = anchors.fit_feature_anchors_from_h5(
        h5_paths,
        anchor_id=anchor_id,
        anchor_source=anchor_source,
        cohort_filter=cohort_filter,
        feature_standardization=feature_standardization,
        min_subjects=min_subjects,
        scale_method=scale_method,
    )
    return anchors.save_anchor_file(artifact, Path(out))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit MNDM 2.1 feature anchors from exported /features_raw H5 surfaces.")
    parser.add_argument("--h5-root", type=Path, required=True, help="H5 file, run directory, processed root, or dataset directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output anchor JSON path.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset id when h5-root is a processed root.")
    parser.add_argument("--config", type=Path, default=None, help="Config used to resolve feature_standardization.")
    parser.add_argument("--anchor-id", type=str, required=True, help="Stable anchor identifier.")
    parser.add_argument("--anchor-source", type=str, default="all_subjects", help="Human-readable source policy.")
    parser.add_argument("--cohort-filter", type=str, default="", help="Human-readable cohort filter/provenance.")
    parser.add_argument("--scale-method", choices=["iqr", "mad", "qn"], default="iqr")
    parser.add_argument("--min-subjects", type=int, default=3)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = fit_anchors_command(
        h5_root=args.h5_root,
        out=args.out,
        dataset_id=args.dataset,
        config=args.config,
        anchor_id=args.anchor_id,
        anchor_source=args.anchor_source,
        cohort_filter=args.cohort_filter,
        scale_method=args.scale_method,
        min_subjects=args.min_subjects,
    )
    print(f"Wrote feature anchors: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
