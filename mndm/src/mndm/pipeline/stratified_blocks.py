"""
stratified_blocks.py
Helpers for Stratified MNPS (v2) Jacobian block summaries and cross-partials.

This module implements "Option B":
- Use the already-estimated Stratified Jacobian field J_v2(t) in 9D subcoordinate space.
- Derive block-level summaries for arbitrary, config-defined groupings of the 9 subcoords.
- Optionally save selected cross-partials J_{out,in}(t) as time series for downstream analysis.

The grouping is *not* constrained to the default prefix split (m_*, d_*, e_*).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


_CANONICAL_SUBcoords_9d: Tuple[str, ...] = (
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


def _deep_merge_dict(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep merge dictionaries: override wins; nested mappings are merged recursively."""
    out: Dict[str, Any] = dict(base) if isinstance(base, Mapping) else {}
    if not isinstance(override, Mapping):
        return out
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge_dict(out.get(k, {}), v)
        else:
            out[k] = v
    return out


@lru_cache(maxsize=128)
def _load_mnps_9d_policy(policy_dir: str, dataset_id: str) -> Dict[str, Any]:
    """Load optional per-dataset mnps_9d policy from YAML."""
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    if not policy_dir or not dataset_id:
        return {}

    root = Path(policy_dir)
    ds_path = root / f"{dataset_id}_mnps_9d.yml"
    map_path = root / "datasets.yml"
    path = ds_path if ds_path.exists() else (map_path if map_path.exists() else None)
    if path is None:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    payload: Dict[str, Any] = {}
    if path.name == "datasets.yml":
        if isinstance(data, Mapping):
            if isinstance(data.get("mnps_9d"), Mapping):
                v2 = data.get("mnps_9d") or {}
                ds_map = v2.get("datasets", {}) if isinstance(v2, Mapping) else {}
                if isinstance(ds_map, Mapping):
                    payload = dict(ds_map.get(dataset_id, {}) or {})
            if not payload and isinstance(data.get("datasets"), Mapping):
                payload = dict((data.get("datasets") or {}).get(dataset_id, {}) or {})
            if not payload and isinstance(data.get(dataset_id), Mapping):
                payload = dict(data.get(dataset_id) or {})
    else:
        if isinstance(data, Mapping) and isinstance(data.get("mnps_9d"), Mapping):
            payload = dict(data.get("mnps_9d") or {})
        else:
            payload = dict(data) if isinstance(data, Mapping) else {}

    for k in ("schema", "schema_version", "dataset_id", "description"):
        payload.pop(k, None)
    return payload


def _dedup_pairs(pairs: Sequence[Sequence[str]]) -> List[List[str]]:
    """Deduplicate ordered pairs while preserving first-seen order."""
    seen: set[tuple[str, str]] = set()
    out: List[List[str]] = []
    for p in pairs:
        a, b = str(p[0]), str(p[1])
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        out.append([a, b])
    return out


def _preset_block_jacobians(name: str) -> Dict[str, Any]:
    """Return a config snippet for mnps_9d.block_jacobians presets."""
    preset = str(name).strip()
    if preset in ("mde_families_v1", "ndt_mde_families_v1"):
        return {
            "enabled": True,
            "include_self": True,
            "include_sym_rot": True,
            "pairs": "all",
            "groups": {
                "m": ["m_a", "m_e", "m_o"],
                "d": ["d_n", "d_l", "d_s"],
                "e": ["e_e", "e_s", "e_m"],
            },
        }
    if preset in ("mde_diag_v1", "ndt_mde_diag_v1"):
        return {
            "enabled": True,
            "include_self": True,
            "include_sym_rot": False,
            "pairs": [["m", "m"], ["d", "d"], ["e", "e"]],
            "groups": {
                "m": ["m_a", "m_e", "m_o"],
                "d": ["d_n", "d_l", "d_s"],
                "e": ["e_e", "e_s", "e_m"],
            },
        }
    if preset in ("mde_offdiag_v1", "ndt_mde_offdiag_v1"):
        # Cross-family only (no self blocks)
        return {
            "enabled": True,
            "include_self": False,
            "include_sym_rot": True,
            "pairs": "all",
            "groups": {
                "m": ["m_a", "m_e", "m_o"],
                "d": ["d_n", "d_l", "d_s"],
                "e": ["e_e", "e_s", "e_m"],
            },
        }
    raise ValueError(
        f"Unknown block_jacobians preset '{preset}'. "
        "Known: mde_families_v1, mde_diag_v1, mde_offdiag_v1"
    )


def _preset_cross_partials(name: str) -> Dict[str, Any]:
    """Return a config snippet for mnps_9d.save_cross_partials presets.

    Each preset returns a dict with:
    - core_pairs: list of [out, in] pairs considered part of the fixed "core"
    - pairs: the fully expanded pairs list (core plus any always-included additions)
    """
    preset = str(name).strip()

    # Theoridrivet kärnset (dataset-agnostic) – minimal, stable list
    core_pairs: List[List[str]] = [
        # Mobility ↔ Diffusion (MD)
        ["m_a", "d_l"],
        ["m_o", "d_s"],
        ["d_l", "m_a"],
        ["d_s", "m_o"],
        # Diffusion ↔ Entropy (DE)
        ["e_s", "d_s"],
        ["e_e", "d_n"],
        ["d_s", "e_s"],
        # Mobility ↔ Entropy (ME)
        ["m_o", "e_s"],
        ["m_a", "e_e"],
    ]

    if preset in ("ndt_core_v1", "core_ndt_v1"):
        return {"core_pairs": core_pairs, "pairs": _dedup_pairs(core_pairs)}

    if preset in ("ndt_core_plus_diag_v1", "core_ndt_plus_diag_v1"):
        diag = [[c, c] for c in _CANONICAL_SUBcoords_9d]
        pairs = _dedup_pairs([*core_pairs, *diag])
        return {"core_pairs": core_pairs, "pairs": pairs, "includes": "core + self-couplings (diag)"}

    if preset in ("ndt_core_plus_intra3x3_v1", "core_ndt_plus_intra3x3_v1"):
        # Full 3×3 intra-family blocks as explicit element series
        intra: List[List[str]] = []
        families = [
            ["m_a", "m_e", "m_o"],
            ["d_n", "d_l", "d_s"],
            ["e_e", "e_s", "e_m"],
        ]
        for fam in families:
            for out in fam:
                for inn in fam:
                    intra.append([out, inn])
        pairs = _dedup_pairs([*core_pairs, *intra])
        return {"core_pairs": core_pairs, "pairs": pairs, "includes": "core + intra-family 3×3 blocks"}

    raise ValueError(
        f"Unknown save_cross_partials preset '{preset}'. "
        "Known: ndt_core_v1, ndt_core_plus_diag_v1, ndt_core_plus_intra3x3_v1"
    )


@dataclass
class StratifiedBlocksResult:
    block_rows: List[Dict[str, Any]]
    blocks_manifest: Dict[str, Any]
    cross_partials_series: Dict[str, np.ndarray]
    cross_partials_manifest: Dict[str, Any]


def write_stratified_block_jacobians_csv(
    rows: Sequence[Mapping[str, Any]],
    output_path: "Any",
    *,
    append: bool = False,
) -> None:
    """Write stratified block Jacobian rows to a CSV file."""
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to write block Jacobian CSVs") from exc

    if not rows:
        return

    df = pd.DataFrame(list(rows))

    # Consistent, analysis-friendly column order
    column_order = [
        "dataset_id",
        "subject_id",
        "session_id",
        "condition_label",
        "task_label",
        "out_group",
        "in_group",
        "out_dim",
        "in_dim",
        "n_timepoints",
        "block_trace_mean",
        "block_frobenius_mean",
        "block_anisotropy_mean",
        "c_sym_mean",
        "c_rot_mean",
    ]
    columns = [c for c in column_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in columns]
    columns.extend(extra_cols)
    df = df[columns]

    mode = "a" if append and getattr(output_path, "exists", lambda: False)() else "w"
    header = not (append and getattr(output_path, "exists", lambda: False)())
    df.to_csv(output_path, mode=mode, header=header, index=False)


def _safe_nanmean(x: np.ndarray) -> float:
    if x.size == 0 or np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanmean(x))


def _safe_nanstd(x: np.ndarray) -> float:
    if x.size == 0 or np.all(np.isnan(x)):
        return float("nan")
    return float(np.nanstd(x))


def _compute_block_metrics(block_field: np.ndarray, is_diagonal_block: bool) -> Dict[str, float]:
    """Summaries for an ordered block field J^{(A,B)} over time.

    Parameters
    ----------
    block_field
        Array of shape [T, p_out, p_in].
    is_diagonal_block
        True only when out_group == in_group (e.g. M→M, D→D, E→E).

        For *diagonal* blocks the trace equals the flow divergence
        (∑ ∂ṁ_i/∂m_i) and anisotropy measures phase-space curvature —
        both are geometrically meaningful.

        For *off-diagonal* blocks (e.g. E→M) the trace is a physically
        meaningless sum of randomly paired cross-derivatives, and anisotropy
        loses its curvature interpretation.  Only the Frobenius norm (total
        cross-coupling energy) is valid for off-diagonal blocks.

    Returns
    -------
    dict
        Contains:
        - block_frobenius_mean  (always valid)
        - block_trace_mean      (NaN for off-diagonal blocks)
        - block_anisotropy_mean (NaN for off-diagonal blocks)
    """
    if block_field.size == 0 or block_field.shape[0] == 0:
        return {
            "block_trace_mean": float("nan"),
            "block_frobenius_mean": float("nan"),
            "block_anisotropy_mean": float("nan"),
        }

    frob = np.linalg.norm(block_field, ord="fro", axis=(1, 2))
    metrics: Dict[str, float] = {
        "block_frobenius_mean": _safe_nanmean(frob),
        "block_trace_mean": float("nan"),
        "block_anisotropy_mean": float("nan"),
    }

    # Trace and anisotropy are only meaningful for intra-block (diagonal) Jacobians.
    if not is_diagonal_block:
        return metrics

    p_out = int(block_field.shape[1])
    p_in = int(block_field.shape[2])
    if p_out != p_in:
        return metrics

    traces = np.trace(block_field, axis1=1, axis2=2)
    aniso_vals: list[float] = []
    for t in range(block_field.shape[0]):
        try:
            svd = np.linalg.svd(block_field[t], compute_uv=False)
            if svd[-1] > 1e-10:
                aniso_vals.append(float(svd[0] / svd[-1]))
            else:
                aniso_vals.append(float("nan"))
        except np.linalg.LinAlgError:
            aniso_vals.append(float("nan"))

    metrics["block_trace_mean"] = _safe_nanmean(np.asarray(traces, dtype=float))
    metrics["block_anisotropy_mean"] = _safe_nanmean(np.asarray(aniso_vals, dtype=float))
    return metrics


def _resolve_group_indices(
    coords_9d_names: Sequence[str],
    groups: Mapping[str, Sequence[str]],
) -> Dict[str, List[int]]:
    name_to_idx = {str(n): i for i, n in enumerate(coords_9d_names)}
    resolved: Dict[str, List[int]] = {}
    for group_name, members in groups.items():
        idxs: List[int] = []
        for m in members:
            m_str = str(m)
            if m_str not in name_to_idx:
                raise ValueError(f"Unknown Stratified subcoord '{m_str}' in group '{group_name}'")
            idxs.append(int(name_to_idx[m_str]))
        if len(idxs) != len(set(idxs)):
            raise ValueError(f"Duplicate subcoords in group '{group_name}': {list(members)}")
        resolved[str(group_name)] = idxs
    return resolved


def _iter_pairs(
    groups: Sequence[str],
    pairs_cfg: Any,
    include_self: bool,
) -> Iterable[Tuple[str, str]]:
    if pairs_cfg == "all":
        for out_g in groups:
            for in_g in groups:
                if not include_self and out_g == in_g:
                    continue
                yield out_g, in_g
        return

    if isinstance(pairs_cfg, Sequence):
        for item in pairs_cfg:
            if not (isinstance(item, Sequence) and len(item) == 2):
                raise ValueError("block_jacobians.pairs must be 'all' or a list of [out_group, in_group]")
            out_g, in_g = str(item[0]), str(item[1])
            if not include_self and out_g == in_g:
                continue
            yield out_g, in_g
        return

    raise ValueError("block_jacobians.pairs must be 'all' or a list of [out_group, in_group]")


def compute_stratified_blocks_and_cross_partials(
    *,
    ds_id: str,
    dataset_label: str,
    subject: str,
    session: Optional[str],
    condition: Optional[str],
    task: Optional[str],
    coords_9d_names: Sequence[str],
    jacobian_9D: np.ndarray,
    config: Mapping[str, Any],
) -> StratifiedBlocksResult:
    """Compute config-driven block summaries and cross-partials from a 9D Jacobian field."""
    mnps_9d_cfg = config.get("mnps_9d", {}) if isinstance(config, Mapping) else {}

    # Merge dataset overrides (if present) on top of global defaults
    ds_overrides: Mapping[str, Any] = {}
    if isinstance(mnps_9d_cfg, Mapping):
        datasets_cfg = mnps_9d_cfg.get("datasets", {}) or {}
        if isinstance(datasets_cfg, Mapping):
            ds_cfg = datasets_cfg.get(ds_id, {}) or {}
            if isinstance(ds_cfg, Mapping):
                ds_overrides = dict(ds_cfg)
        policy_dir = mnps_9d_cfg.get("policy_dir")
        if isinstance(policy_dir, (str, Path)):
            policy_cfg = _load_mnps_9d_policy(str(policy_dir), ds_id)
            if policy_cfg:
                ds_overrides = _deep_merge_dict(ds_overrides, policy_cfg)

    # Accept both keys (block_jacobians preferred; jacobian_blocks alias)
    blocks_cfg: Mapping[str, Any] = {}
    if isinstance(mnps_9d_cfg, Mapping):
        blocks_cfg = mnps_9d_cfg.get("block_jacobians", {}) or mnps_9d_cfg.get("jacobian_blocks", {}) or {}
    if isinstance(ds_overrides, Mapping):
        blocks_cfg = dict(blocks_cfg)
        blocks_cfg.update(ds_overrides.get("block_jacobians", {}) or ds_overrides.get("jacobian_blocks", {}) or {})
    # Optional preset expansion (explicit config keys always override the preset)
    if isinstance(blocks_cfg, Mapping) and blocks_cfg.get("preset"):
        preset_cfg = _preset_block_jacobians(str(blocks_cfg.get("preset")))
        merged = dict(preset_cfg)
        merged.update(dict(blocks_cfg))
        blocks_cfg = merged
    blocks_enabled = bool(getattr(blocks_cfg, "get", lambda *_: False)("enabled", False))

    cross_cfg: Mapping[str, Any] = {}
    if isinstance(mnps_9d_cfg, Mapping):
        cross_cfg = mnps_9d_cfg.get("save_cross_partials", {}) or {}
    if isinstance(ds_overrides, Mapping):
        cross_cfg = dict(cross_cfg)
        cross_cfg.update(ds_overrides.get("save_cross_partials", {}) or {})
    # Optional preset expansion (explicit config keys always override the preset)
    cross_preset_info: Dict[str, Any] = {}
    if isinstance(cross_cfg, Mapping) and cross_cfg.get("preset"):
        cross_preset_info = _preset_cross_partials(str(cross_cfg.get("preset")))
        merged = dict(cross_preset_info)
        merged.update(dict(cross_cfg))
        cross_cfg = merged
    cross_enabled = bool(getattr(cross_cfg, "get", lambda *_: False)("enabled", False))

    block_rows: List[Dict[str, Any]] = []
    blocks_manifest: Dict[str, Any] = {}
    cross_series: Dict[str, np.ndarray] = {}
    cross_manifest: Dict[str, Any] = {}

    # ---- Block summaries ----
    if blocks_enabled:
        groups_cfg = blocks_cfg.get("groups", {}) if isinstance(blocks_cfg, Mapping) else {}
        if not isinstance(groups_cfg, Mapping) or not groups_cfg:
            raise ValueError("mnps_9d.block_jacobians.groups must be a non-empty mapping")

        include_self = bool(blocks_cfg.get("include_self", True))
        include_sym_rot = bool(blocks_cfg.get("include_sym_rot", True))
        pairs_cfg = blocks_cfg.get("pairs", "all")

        group_indices = _resolve_group_indices(coords_9d_names, groups_cfg)
        group_names = list(group_indices.keys())

        # Precompute for c_sym/c_rot lookups
        n_timepoints = int(jacobian_9D.shape[0])

        for out_g, in_g in _iter_pairs(group_names, pairs_cfg, include_self=include_self):
            out_idxs = group_indices[out_g]
            in_idxs = group_indices[in_g]

            block_out_in = jacobian_9D[:, out_idxs, :][:, :, in_idxs]  # [T, p_out, p_in]
            metrics = _compute_block_metrics(block_out_in, is_diagonal_block=(out_g == in_g))

            c_sym_mean = float("nan")
            c_rot_mean = float("nan")
            if include_sym_rot and out_g != in_g:
                p_out = len(out_idxs)
                p_in = len(in_idxs)
                # Only defined when p_out == p_in so block_out_in and block_in_out^T match
                if p_out == p_in and n_timepoints > 0:
                    block_in_out = jacobian_9D[:, in_idxs, :][:, :, out_idxs]  # [T, p_in, p_out]
                    sym_vals: list[float] = []
                    rot_vals: list[float] = []
                    for t in range(n_timepoints):
                        J_out_in = block_out_in[t]
                        J_in_out = block_in_out[t]
                        sym_block = 0.5 * (J_out_in + J_in_out.T)
                        rot_block = 0.5 * (J_out_in - J_in_out.T)
                        sym_vals.append(float(np.linalg.norm(sym_block, ord="fro")))
                        rot_vals.append(float(np.linalg.norm(rot_block, ord="fro")))
                    c_sym_mean = _safe_nanmean(np.asarray(sym_vals, dtype=float))
                    c_rot_mean = _safe_nanmean(np.asarray(rot_vals, dtype=float))

            row = {
                "dataset_id": dataset_label,
                "subject_id": subject,
                "session_id": session,
                "condition_label": condition,
                "task_label": task,
                "out_group": out_g,
                "in_group": in_g,
                "out_dim": int(len(out_idxs)),
                "in_dim": int(len(in_idxs)),
                "n_timepoints": int(block_out_in.shape[0]),
                "c_sym_mean": float(c_sym_mean),
                "c_rot_mean": float(c_rot_mean),
            }
            row.update(metrics)
            block_rows.append(row)

        blocks_manifest = {
            "enabled": True,
            "preset": str(blocks_cfg.get("preset")) if isinstance(blocks_cfg, Mapping) and blocks_cfg.get("preset") else None,
            "groups": {k: list(v) for k, v in groups_cfg.items()},
            "pairs": "all" if pairs_cfg == "all" else list(pairs_cfg),
            "include_self": bool(include_self),
            "include_sym_rot": bool(include_sym_rot),
            "n_blocks": int(len(block_rows)),
        }

    # ---- Cross partials ----
    if cross_enabled:
        pairs: Any = cross_cfg.get("pairs", []) if isinstance(cross_cfg, Mapping) else []
        extra_pairs: Any = cross_cfg.get("extra_pairs", []) if isinstance(cross_cfg, Mapping) else []
        core_pairs_from_preset: Any = cross_cfg.get("core_pairs", []) if isinstance(cross_cfg, Mapping) else []

        # Back-compat precedence:
        # - If 'pairs' is explicitly provided as a non-empty list, use it verbatim.
        # - Else: use preset-expanded pairs (already in cross_cfg['pairs']) and append extra_pairs if provided.
        if not (isinstance(pairs, Sequence) and len(pairs) > 0):
            # If preset was used, cross_cfg['pairs'] is already expanded.
            pairs = cross_preset_info.get("pairs", pairs)
            if isinstance(extra_pairs, Sequence) and len(extra_pairs) > 0:
                pairs = _dedup_pairs([*list(pairs), *list(extra_pairs)])

        if not isinstance(pairs, Sequence) or not pairs:
            raise ValueError(
                "mnps_9d.save_cross_partials requires either a non-empty 'pairs' list "
                "or a valid 'preset' (optionally with 'extra_pairs')."
            )
        name_to_idx = {str(n): i for i, n in enumerate(coords_9d_names)}
        items: List[Dict[str, Any]] = []
        for pair in pairs:
            if not (isinstance(pair, Sequence) and len(pair) == 2):
                raise ValueError("Each save_cross_partials entry must be [out_coord, in_coord]")
            out_name, in_name = str(pair[0]), str(pair[1])
            if out_name not in name_to_idx or in_name not in name_to_idx:
                raise ValueError(f"Unknown cross-partial pair [{out_name}, {in_name}] for coords_9d_names={list(coords_9d_names)}")
            i = int(name_to_idx[out_name])
            j = int(name_to_idx[in_name])
            series = np.asarray(jacobian_9D[:, i, j], dtype=np.float32)
            key = f"{out_name}__{in_name}"
            cross_series[key] = series
            items.append(
                {
                    "name": key,
                    "out": out_name,
                    "in": in_name,
                    "mean": _safe_nanmean(series.astype(float)),
                    "std": _safe_nanstd(series.astype(float)),
                    "abs_mean": _safe_nanmean(np.abs(series.astype(float))),
                    "n_timepoints": int(series.shape[0]),
                }
            )

        cross_manifest = {
            "enabled": True,
            "preset": str(cross_cfg.get("preset")) if isinstance(cross_cfg, Mapping) and cross_cfg.get("preset") else None,
            # "Core vs extra" bookkeeping to support prereg / Methods language
            "core_pairs": _dedup_pairs(list(core_pairs_from_preset)) if isinstance(core_pairs_from_preset, Sequence) else [],
            "extra_pairs": _dedup_pairs(list(extra_pairs)) if isinstance(extra_pairs, Sequence) else [],
            "pairs": [[str(p[0]), str(p[1])] for p in pairs],
            "rationale": str(cross_cfg.get("rationale")) if isinstance(cross_cfg, Mapping) and cross_cfg.get("rationale") else None,
            "items": items,
        }

    return StratifiedBlocksResult(
        block_rows=block_rows,
        blocks_manifest=blocks_manifest,
        cross_partials_series=cross_series,
        cross_partials_manifest=cross_manifest,
    )


