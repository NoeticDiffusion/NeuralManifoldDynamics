"""Helpers for config-driven conventional EEG comparator summaries."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd

from .robustness_helpers import _distributional_descriptives


def _deep_merge_mapping(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge two config mappings."""
    merged: Dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge_mapping(dict(merged.get(key, {})), dict(value))
        else:
            merged[key] = value
    return merged


def resolve_conventional_eeg_cfg(config: Mapping[str, Any], dataset_id: Optional[str]) -> Dict[str, Any]:
    """Resolve conventional EEG config with optional dataset overrides."""
    root = config.get("conventional_eeg", {}) if isinstance(config, Mapping) else {}
    if not isinstance(root, Mapping):
        return {}
    merged: Dict[str, Any] = {k: root[k] for k in root if k != "datasets"}
    ds_map = root.get("datasets", {})
    if dataset_id and isinstance(ds_map, Mapping):
        ds_cfg = ds_map.get(dataset_id)
        if isinstance(ds_cfg, Mapping):
            merged = _deep_merge_mapping(merged, dict(ds_cfg))
    return merged


def _normalize_packs(conventional_cfg: Mapping[str, Any]) -> set[str]:
    """Return normalized conventional EEG pack names."""
    packs_raw = conventional_cfg.get("packs", ["tier1"]) if isinstance(conventional_cfg, Mapping) else ["tier1"]
    if isinstance(packs_raw, (str, bytes)):
        return {str(packs_raw).strip().lower()}
    if isinstance(packs_raw, list):
        return {str(v).strip().lower() for v in packs_raw if str(v).strip()}
    return {"tier1"}


def _coma_clinical_markers_payload(dataset_id: Optional[str]) -> Dict[str, Any]:
    """Return explicit availability metadata for non-EEG coma biomarkers."""
    source_hint = "requires_external_clinical_data"
    if str(dataset_id or "").strip() == "physionet_icare_2_1":
        source_hint = "not_present_in_public_icare_2_1_release"
    marker_payload = {
        "status": "unavailable",
        "reason": source_hint,
    }
    return {
        "mode": "eeg_only_proxy",
        "markers": {
            "ssep": dict(marker_payload),
            "nse": dict(marker_payload),
            "gcs": dict(marker_payload),
            "s100b": dict(marker_payload),
        },
    }


def _artifact_qc_status_payload(artifact_qc_applied: Optional[bool]) -> Dict[str, Any]:
    """Explicit, non-blocking artifact-QC provenance for this extension.

    This is a soft flag, not a write gate: the extension is still written
    when an artifact-reduction method is not confirmed to have run (e.g.
    I-CARE preprocess.artifacts.method=none), because withholding it
    entirely would make the family unusable on every dataset that has not
    yet enabled ICA/EOG regression. Readers must consult this field before
    trusting family-average band-power / connectivity descriptives that can
    be outlier-heavy without artifact rejection (see
    project/mnps_v3/tests/ingest_jacobian_fidelity_handover_2.md items 9 and
    12).
    """
    if artifact_qc_applied is True:
        return {"status": "confirmed_applied", "reason": "artifact_method_confirmed_applied"}
    if artifact_qc_applied is False:
        return {"status": "not_confirmed", "reason": "artifact_method_not_configured_or_not_applied"}
    return {"status": "not_assessed", "reason": "no_artifact_qc_sidecar_evidence"}


def compute_conventional_eeg_summary(
    *,
    sub_frame: pd.DataFrame,
    config: Mapping[str, Any],
    dataset_id: Optional[str],
    artifact_qc_applied: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    """Compute summarize-time descriptives for conventional EEG comparator columns."""
    conventional_cfg = resolve_conventional_eeg_cfg(config, dataset_id)
    if not conventional_cfg or not bool(conventional_cfg.get("enabled", False)):
        return None

    packs = _normalize_packs(conventional_cfg)
    coma_cfg = conventional_cfg.get("coma", {})
    coma_cfg = coma_cfg if isinstance(coma_cfg, Mapping) else {}
    suppression_cfg = coma_cfg.get("suppression", {})
    suppression_cfg = suppression_cfg if isinstance(suppression_cfg, Mapping) else {}

    export_cfg = conventional_cfg.get("export", {})
    if isinstance(export_cfg, Mapping) and not bool(export_cfg.get("summaries", True)):
        return None

    conventional_cols = [
        str(col)
        for col in sub_frame.columns
        if str(col).startswith("eeg_conventional_")
    ]
    if not conventional_cols:
        return None

    families: Dict[str, Dict[str, Any]] = {}
    for col in sorted(conventional_cols):
        suffix = str(col)[len("eeg_conventional_") :]
        family, sep, feature_name = suffix.partition("_")
        if not sep:
            family = "misc"
            feature_name = suffix
        families.setdefault(family or "misc", {})[feature_name] = col

    family_payload: Dict[str, Any] = {}
    for family_name, feature_cols in families.items():
        ordered_items = sorted(feature_cols.items(), key=lambda item: item[0])
        feature_names = [feature_name for feature_name, _ in ordered_items]
        values = np.column_stack(
            [
                pd.to_numeric(sub_frame[col_name], errors="coerce").to_numpy(dtype=float)
                for _, col_name in ordered_items
            ]
        )
        technical_eligibility: Optional[Dict[str, Any]] = None
        if family_name == "coma":
            eligible = np.ones(len(sub_frame), dtype=bool)
            exclusion_reasons = []
            if "technical_eeg_status" in sub_frame.columns:
                technical_valid = sub_frame["technical_eeg_status"].astype(str).str.lower().eq("valid").to_numpy()
                eligible &= technical_valid
                exclusion_reasons.append("technical_eeg_status_must_be_valid")
            else:
                exclusion_reasons.append("technical_eeg_status_unavailable; historical values retained")
            if "technical_eeg_channel_status" in sub_frame.columns:
                channel_supported = sub_frame["technical_eeg_channel_status"].astype(str).str.lower().eq("computed").to_numpy()
                eligible &= channel_supported
                exclusion_reasons.append("technical_eeg_channel_status_must_be_computed")
            values[~eligible, :] = np.nan
            technical_eligibility = {
                "status": "applied" if any(
                    col in sub_frame.columns
                    for col in ("technical_eeg_status", "technical_eeg_channel_status")
                ) else "not_available",
                "included_row_count": int(np.sum(eligible)),
                "excluded_row_count": int(np.sum(~eligible)),
                "criteria": exclusion_reasons,
            }
        descriptives = _distributional_descriptives(values, feature_names)
        family_payload[family_name] = {
            feature_name: {
                "column": feature_cols[feature_name],
                **stats,
            }
            for feature_name, stats in descriptives.items()
        }
        if technical_eligibility is not None:
            family_payload[family_name]["technical_eligibility"] = technical_eligibility

    payload = {
        "schema_version": "mndm.conventional_eeg.v2",
        "packs": sorted(packs),
        "column_count": int(len(conventional_cols)),
        "columns": sorted(conventional_cols),
        "families": family_payload,
        "artifact_qc": _artifact_qc_status_payload(artifact_qc_applied),
    }
    row_status_cols = [
        "technical_eeg_status", "technical_eeg_reason",
        "technical_eeg_channel_status", "technical_eeg_channel_reason",
        "technical_eeg_channel_count", "technical_eeg_finite_fraction",
        "technical_eeg_gap_count", "technical_eeg_flatline_channel_count",
        "technical_eeg_clipping_status", "technical_eeg_clipping_reason",
        "technical_eeg_adc_rail_fraction", "coma_proxy_suppression_status",
        "coma_proxy_suppression_reason", "coma_proxy_unit_status",
        "coma_proxy_amplitude_unit", "coma_proxy_reference_status",
        "coma_proxy_reference_mode", "coma_proxy_amplitude_status",
        "coma_proxy_amplitude_reason", "coma_proxy_valid_sample_fraction",
        "coma_proxy_gap_count", "coma_proxy_segment_count",
        "coma_proxy_threshold_value",
    ]
    available_status_cols = [col for col in row_status_cols if col in sub_frame.columns]
    row_measure_cols = [col for col in conventional_cols if col.startswith("eeg_conventional_coma_")]
    row_level_cols = available_status_cols + [col for col in row_measure_cols if col not in available_status_cols]
    if row_level_cols:
        identity_cols = [col for col in ("file", "recording_id", "epoch_id", "t_start", "t_end") if col in sub_frame.columns]
        status_rows = []
        for row_index, (_, row) in enumerate(sub_frame.iterrows()):
            item: Dict[str, Any] = {"row_index": int(row_index)}
            for col in identity_cols + row_level_cols:
                value = row.get(col)
                if isinstance(value, np.generic):
                    value = value.item()
                try:
                    missing_value = value is None or bool(pd.isna(value))
                except (TypeError, ValueError):
                    missing_value = value is None
                if missing_value:
                    value = None
                item[col] = value
            status_rows.append(item)
        payload["row_level_status"] = {
            "status": "computed" if available_status_cols else "not_assessed",
            "reason": None if available_status_cols else "technical_status_columns_unavailable",
            "columns": row_level_cols,
            "rows": status_rows,
            "physiological_proxy_scope": "algorithmic_threshold_proxy; not an ACNS adjudication",
            "measure_definitions": {
                "technical_eeg": {
                    "finite_fraction": "fraction of channel-samples that are finite within the nominal EEG window",
                    "gap_count": "count of contiguous sample intervals where at least one EEG channel is non-finite",
                    "flatline": "per-channel exact zero range among finite samples; reported as suspect and never used to erase the physiological proxy",
                    "clipping": "fraction of finite samples at or beyond verified ADC minimum/maximum rails; not assessed when ADC rails are unavailable",
                },
                "coma_suppression_ratio": {
                    "definition": "fraction of finite smoothed absolute-amplitude samples at or below max(amplitude_floor, relative_to_rms * epoch_rms)",
                    "gap_policy": "smooth each contiguous finite segment independently; burst transitions are counted only within segments",
                    "parameters": {
                        "amplitude_floor": suppression_cfg.get("amplitude_floor", 0.01) if isinstance(suppression_cfg, Mapping) else 0.01,
                        "relative_to_rms": suppression_cfg.get("relative_to_rms", 0.20) if isinstance(suppression_cfg, Mapping) else 0.20,
                        "envelope_smoothing_sec": suppression_cfg.get("envelope_smoothing_sec", 0.25) if isinstance(suppression_cfg, Mapping) else 0.25,
                    },
                    "amplitude_unit_source": "preprocessing EEG unit/scaling provenance; numeric suppression and burst proxies are withheld when unverified",
                    "clinical_status": "algorithmic proxy only; not an ACNS adjudication",
                },
                "coma_amplitude_rms": {
                    "definition": "root mean square amplitude of the global median EEG montage over finite samples in the nominal window",
                    "status_field": "coma_proxy_amplitude_status",
                    "required_provenance": ["verified physical amplitude unit", "verified preprocessing reference mode"],
                    "unit_field": "coma_proxy_amplitude_unit",
                },
            },
        }
    else:
        payload["row_level_status"] = {
            "status": "not_assessed",
            "reason": "row_level_technical_and_proxy_status_columns_unavailable",
            "columns": [],
            "rows": [],
        }
    if "coma" in packs:
        payload["clinical_markers"] = _coma_clinical_markers_payload(dataset_id)
    return payload
