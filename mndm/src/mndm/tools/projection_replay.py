"""Byte-close replay of production MNPS coordinates from cached features.

Resolves Stage 0 of ``project/ideas/fmri-EEG-harmonization/sciencelead/003.md``.

Root cause (see project/diary/167_*.md for the full writeup):

    A prior remap-only script (project/scripts/16_ds007216_fmri_v25_remap_eval.py)
    called ``derive_mde_from_v2(coords_9d, names, mnps_3d.from_v2.map, ...)``,
    i.e. it assumed the final 3D ``mnps_3d`` is an equal-weight mean of all 9
    ``mnps_9d`` subcoordinates (3 per axis). That is NOT what the real
    pipeline does when ``mnps_3d.from_v2.aggregation == "fixed_weighted_projection"``
    (the production setting): it instead builds the axis_map from
    ``mnps_projection.v1_mapping`` -- a legacy, *raw-feature-name*-keyed 3-axis
    weighting (e.g. ``m: {fmri_FC_mean: 0.7, fmri_kuramoto_global: 0.3}``) --
    coerced onto whichever v2 subcoord happens to contain each named raw
    feature, via ``mndm.pipeline.summary._coerce_v1_mapping_to_v2_subcoords``.

    For ds007216's production mapping this means the final ``mnps_3d`` only
    ever draws on 4 of the 9 ``coords_9d`` subcoordinates (m_a, d_s, d_n, e_e);
    the other 5 (m_e, m_o, d_l, e_s, e_m) are computed into ``coords_9d`` (and
    exported) but carry zero weight in ``mnps_3d``. ``coords_9d`` itself was
    never the problem -- reproducing it from cached features was always exact.

    This also explains why ``resummarize`` silently dropped every epoch for a
    modified ``mnps_9d.subcoords`` config: if a v1_mapping raw-feature name
    (e.g. ``fmri_variance_global``) is removed from every subcoord,
    ``_coerce_v1_mapping_to_v2_subcoords`` cannot resolve it to any subcoord
    and (per its own "preserve unknown keys so runtime validation can fail
    explicitly" comment) leaves a dead key in the axis_map that matches no
    real ``coords_9d`` column. The projection matrix ends up with zero
    nonzero weights for that axis, so per-axis coverage is never set (stays
    NaN) -- and NaN never satisfies ``coverage >= min_axis_coverage``, so
    *every* epoch fails the ``nan_mask_v1`` gate. This is a config-authoring
    hazard (an implicit coupling between ``mnps_projection.v1_mapping`` and
    ``mnps_9d.subcoords``), not a bug in the coverage/masking logic itself --
    but the failure mode is silent (0 epochs, no fatal error) rather than a
    clear config-validation error. ``validate_v1_mapping_coverage`` below
    turns it into a hard, explicit error in this tool (opt-in for pipeline
    integration; not modifying the shared `summarize` code path in this pass).

Scope boundary: this module reproduces the *mapping* stage (features.parquet
epoch rows -> coords_9d -> mnps_3d) exactly, reusing the real production
helper functions (`mndm.projection.project_features_v2`/`derive_mde_from_v2`,
`mndm.pipeline.summary._resolve_mnps_9d_runtime_config`/
`_resolve_mnps_3d_cfg`/`_coerce_v1_mapping_to_v2_subcoords`). It does NOT
re-derive FD-censoring/epoch-selection from raw confounds (that logic lives
on a much heavier `summarize`-internal component class). Instead it takes the
retained-epoch set from an existing *reference* H5 for the same run (FD
censoring is mapping-independent, so reusing it from any already-summarized
run -- e.g. the current production baseline -- is exact, not an
approximation). Testing a brand-new dataset with no reference H5 yet still
requires one real `mndm.cli summarize` pass to establish that reference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import h5py
import numpy as np
import pandas as pd

from .. import projection
from ..pipeline.summary import (
    _coerce_v1_mapping_to_v2_subcoords,
    _resolve_mnps_3d_cfg,
    _resolve_mnps_9d_runtime_config,
)

DEFAULT_NORMALIZE = "robust_z"
DEFAULT_CLIP_THRESHOLD = 6.0


class ProjectionReplayError(ValueError):
    """Raised when a config cannot be replayed with an explicit, actionable cause."""


def resolve_effective_mapping(config: Mapping[str, Any], dataset_id: str) -> dict[str, Any]:
    """Resolve everything needed to reproduce production coords_9d/mnps_3d
    from a loaded ingest config, exactly as `pipeline/summary.py`'s summarize
    component does at H5-write time."""
    v2_cfg = config.get("mnps_9d", {}) if isinstance(config, Mapping) else {}
    v2_enabled, v2_definition_version, selected_v2_cfg, subcoords_spec = _resolve_mnps_9d_runtime_config(
        v2_cfg if isinstance(v2_cfg, Mapping) else {}, dataset_id,
    )
    m3d_cfg = _resolve_mnps_3d_cfg(config if isinstance(config, Mapping) else {})
    proj_cfg = config.get("mnps_projection", {}) if isinstance(config, Mapping) else {}
    clip_threshold = float(proj_cfg.get("clip_threshold", DEFAULT_CLIP_THRESHOLD)) if isinstance(proj_cfg, Mapping) else DEFAULT_CLIP_THRESHOLD
    feature_standardization = proj_cfg.get("feature_standardization", {}) if isinstance(proj_cfg, Mapping) else {}
    normalize_mode = str(proj_cfg.get("normalize", DEFAULT_NORMALIZE)) if isinstance(proj_cfg, Mapping) else DEFAULT_NORMALIZE

    aggregation_effective = str(m3d_cfg.get("aggregation", "fixed_weighted_projection"))
    if aggregation_effective == "fixed_weighted_projection":
        axis_map = _coerce_v1_mapping_to_v2_subcoords(
            m3d_cfg.get("v1_mapping", {}), subcoords_spec if isinstance(subcoords_spec, Mapping) else {},
        )
    else:
        axis_map = m3d_cfg.get("map", {})

    return {
        "v2_enabled": v2_enabled,
        "v2_definition_version": v2_definition_version,
        "subcoords_spec": subcoords_spec,
        "m3d_cfg": m3d_cfg,
        "axis_map": axis_map,
        "aggregation_effective": aggregation_effective,
        "pooling": str(m3d_cfg.get("legacy_pooling", "mean")),
        "normalize_mode": normalize_mode,
        "feature_standardization": feature_standardization,
        "clip_threshold": clip_threshold,
    }


def validate_v1_mapping_coverage(effective_mapping: Mapping[str, Any]) -> list[str]:
    """Explicit, fail-fast check for the config-authoring hazard documented in
    this module's docstring: every raw feature name referenced by
    `mnps_projection.v1_mapping` must resolve to at least one real
    `mnps_9d.subcoords` entry, else that axis silently gets zero coverage and
    every epoch in the run is dropped by the `nan_mask_v1` policy gate.
    Returns a list of human-readable problems (empty if none)."""
    subcoords_spec = effective_mapping.get("subcoords_spec") or {}
    known_subcoord_names = set(subcoords_spec.keys())
    axis_map = effective_mapping.get("axis_map") or {}
    problems: list[str] = []
    for axis, weights in axis_map.items():
        if not isinstance(weights, Mapping) or not weights:
            problems.append(f"axis '{axis}': v1_mapping resolved to no usable subcoord weights at all (empty axis_map).")
            continue
        for key in weights.keys():
            if key not in known_subcoord_names:
                problems.append(
                    f"axis '{axis}': v1_mapping references '{key}', which does not match any "
                    f"mnps_9d.subcoords entry ({sorted(known_subcoord_names)}). This raw-feature "
                    "name was likely moved/removed from subcoords without updating "
                    "mnps_projection.v1_mapping to match -- every epoch in this run will be "
                    "dropped by the nan_mask_v1 coverage gate for this axis."
                )
    return problems


def replay_run(
    features_run_df: pd.DataFrame,
    config: Mapping[str, Any],
    dataset_id: str,
    *,
    retained_window_start: Optional[np.ndarray] = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Recompute coords_9d/mnps_3d for one run's cached feature rows.

    Args:
        features_run_df: rows from features.parquet for exactly one run,
            any order (will be sorted by t_start).
        config: loaded ingest config (see core.config_loader.load_config).
        dataset_id: dataset id, for mnps_9d per-dataset override resolution.
        retained_window_start: optional epoch mask (in seconds, matched
            against `t_start`) sourced from a reference H5's `window_start`.
            If omitted, all rows in `features_run_df` are used as-is (caller
            is responsible for having already applied FD-censoring/QC).
        strict: if True (default), raise ProjectionReplayError when
            `validate_v1_mapping_coverage` finds a problem, instead of
            silently producing an all-NaN axis.
    """
    effective = resolve_effective_mapping(config, dataset_id)
    if strict:
        problems = validate_v1_mapping_coverage(effective)
        if problems:
            raise ProjectionReplayError(
                "Refusing to replay: v1_mapping / mnps_9d.subcoords are inconsistent for "
                f"dataset '{dataset_id}':\n  - " + "\n  - ".join(problems)
            )

    run_df = features_run_df.sort_values("t_start").reset_index(drop=True)
    if retained_window_start is not None:
        run_df = run_df[run_df["t_start"].round(3).isin(np.round(retained_window_start, 3))].reset_index(drop=True)

    coords_9d, names, _baselines = projection.project_features_v2(
        run_df,
        effective["subcoords_spec"],
        normalize=effective["normalize_mode"],
        feature_standardization=effective["feature_standardization"],
        clip_threshold=effective["clip_threshold"],
    )
    window_start = run_df["t_start"].to_numpy(dtype=float)
    if coords_9d.size == 0 or not names:
        n = len(run_df)
        return {
            "window_start": window_start,
            "coords_9d": coords_9d,
            "coords_9d_names": names,
            "mnps_3d": np.full((n, 3), np.nan, dtype=np.float32),
            "coverage": np.full((n, 3), np.nan, dtype=np.float32),
            "effective_mapping": effective,
        }

    mnps_3d, coverage = projection.derive_mde_from_v2(
        coords_9d,
        names,
        effective["axis_map"],
        pooling=effective["pooling"],
        normalize_columns_l2=True,
        enforce_block_selective=False,
    )
    return {
        "window_start": window_start,
        "coords_9d": coords_9d,
        "coords_9d_names": names,
        "mnps_3d": mnps_3d,
        "coverage": coverage,
        "effective_mapping": effective,
    }


def load_reference_window_start(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return np.asarray(f["window_start"][:], dtype=float)


def regression_check_against_h5(
    replay_result: Mapping[str, Any],
    reference_h5_path: Path,
) -> dict[str, Any]:
    """Compare a replay result against a reference H5's stored coords_9d/mnps_3d.
    Used for Stage 0's required regression check: unmodified production config
    should replay with ~1.0 correlation and exact window counts."""
    with h5py.File(reference_h5_path, "r") as f:
        h5_window_start = np.asarray(f["window_start"][:], dtype=float)
        h5_mnps_3d = np.asarray(f["mnps_3d"][:], dtype=float)
        h5_coords_9d = np.asarray(f["coords_9d"]["values"][:], dtype=float)
        h5_coords_9d_names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["coords_9d"]["names"][:]]

    window_start = replay_result["window_start"]
    mnps_3d = np.asarray(replay_result["mnps_3d"], dtype=float)
    coords_9d = np.asarray(replay_result["coords_9d"], dtype=float)
    names = list(replay_result["coords_9d_names"])

    out: dict[str, Any] = {"n_h5": int(h5_window_start.shape[0]), "n_replay": int(window_start.shape[0])}
    out["window_count_matches"] = out["n_h5"] == out["n_replay"]
    if not out["window_count_matches"]:
        out["status"] = "window_count_mismatch"
        return out
    out["window_start_matches"] = bool(np.allclose(h5_window_start, window_start, atol=1e-3))
    out["coords_9d_names_match"] = names == h5_coords_9d_names

    finite_9d = np.isfinite(h5_coords_9d) & np.isfinite(coords_9d)
    out["coords_9d_max_abs_diff"] = float(np.nanmax(np.abs(h5_coords_9d[finite_9d] - coords_9d[finite_9d]))) if finite_9d.any() else float("nan")

    finite_3d = np.isfinite(h5_mnps_3d) & np.isfinite(mnps_3d)
    out["mnps_3d_max_abs_diff"] = float(np.nanmax(np.abs(h5_mnps_3d[finite_3d] - mnps_3d[finite_3d]))) if finite_3d.any() else float("nan")
    out["mnps_3d_corr"] = float(np.corrcoef(h5_mnps_3d[finite_3d], mnps_3d[finite_3d])[0, 1]) if finite_3d.sum() > 1 else float("nan")
    out["status"] = "ok"
    return out
