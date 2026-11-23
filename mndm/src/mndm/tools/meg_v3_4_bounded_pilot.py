"""Run the fixed ds003645 MEG MNPS 3.4 measurement comparison pilot.

This pilot emits all configured variants from one preprocessing pass per
recording. It does not inspect condition contrasts and does not fit anchors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from core import config_loader
from core.io.h5_writer import write_measurement_export_sidecar

from ..features.meg import compute_meg_features
from ..mnps.v3_4.meg_candidates import MEG_V34_VARIANTS
from ..mnps.v3_4.meg_export import build_meg_measurement_export
from ..preprocess import preprocess_file


_SPLIT_A = ("meg_left_anterior", "meg_midline_posterior", "meg_right_anterior")
_SPLIT_B = ("meg_left_posterior", "meg_midline_anterior", "meg_right_posterior")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _spearman(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    from scipy.stats import spearmanr

    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    if int(finite.sum()) < 3:
        return float("nan"), int(finite.sum())
    value = float(spearmanr(a[finite], b[finite]).statistic)
    return value, int(finite.sum())


def _spatial_split_indices(
    labels: list[str],
    family: str,
    groups: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the frozen ds003645 sector checkerboard to explicit sensors."""
    region_to_half: dict[str, str] = {}
    for group_name in _SPLIT_A:
        values = groups.get(group_name, [])
        for value in values if isinstance(values, list) else []:
            token = str(value)
            region_to_half[token[:-1]] = "A"
    for group_name in _SPLIT_B:
        values = groups.get(group_name, [])
        for value in values if isinstance(values, list) else []:
            token = str(value)
            region_to_half[token[:-1]] = "B"

    indices: dict[str, list[int]] = {"A": [], "B": []}
    unmatched: list[str] = []
    for idx, label_raw in enumerate(labels):
        label = str(label_raw).strip()
        if family == "mag":
            coil_prefix = label[:-1] if label.endswith("1") else ""
        else:
            coil_prefix = label[:-1] if label.endswith(("2", "3")) else ""
        half = region_to_half.get(coil_prefix)
        if half is None:
            unmatched.append(label)
            continue
        indices[half].append(idx)
    info = {
        "split_definition": {
            "A": list(_SPLIT_A),
            "B": list(_SPLIT_B),
            "spatial_coverage": "each split includes left, midline, and right sectors and both anterior/posterior sectors",
        },
        "channel_count_A": len(indices["A"]),
        "channel_count_B": len(indices["B"]),
        "channel_set_hash_A": _json_hash(sorted(labels[i] for i in indices["A"])),
        "channel_set_hash_B": _json_hash(sorted(labels[i] for i in indices["B"])),
        "unmatched_channel_count": len(unmatched),
        "unmatched_channel_labels": unmatched,
    }
    return np.asarray(indices["A"], dtype=int), np.asarray(indices["B"], dtype=int), info


def _feature_series(frame: pd.DataFrame, family: str, coordinate: str) -> np.ndarray:
    name = f"mnps_v3_4_meg_{family}_{coordinate}"
    if name not in frame:
        return np.full(len(frame), np.nan, dtype=np.float64)
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)


def _make_spatial_split_payload(
    preprocessed: Any,
    config: Mapping[str, Any],
    family: str,
    indices: np.ndarray,
) -> dict[str, Any]:
    signals = dict(preprocessed.signals)
    channels = dict(preprocessed.channels or {})
    family_key = f"meg_{family}"
    signals = {key: value for key, value in signals.items() if key not in {"meg_mag", "meg_grad"}}
    channels = {key: value for key, value in channels.items() if key not in {"meg_mag", "meg_grad"}}
    family_data = np.asarray(preprocessed.signals[family_key])[indices]
    family_labels = [str(value) for value in (preprocessed.channels or {}).get(family_key, [])]
    signals[family_key] = family_data
    channels[family_key] = [family_labels[int(idx)] for idx in indices]
    return {
        "signals": signals,
        "channels": channels,
        "sfreq": float(preprocessed.sfreq),
        "dataset_id": "ds003645",
        "file_path": str(preprocessed.meta.get("file", "")),
        "meta": dict(preprocessed.meta or {}),
    }


def _run_recording(
    source_file: Path,
    *,
    config: Mapping[str, Any],
    output_root: Path,
    sensor_groups: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    preprocessed = preprocess_file(source_file, config)
    payload = {
        "signals": preprocessed.signals,
        "channels": preprocessed.channels or {},
        "sfreq": float(preprocessed.sfreq),
        "dataset_id": "ds003645",
        "file_path": str(source_file),
        "meta": dict(preprocessed.meta or {}),
    }
    full_frame = compute_meg_features(payload, config)
    measurement = build_meg_measurement_export(full_frame, config)
    if not isinstance(measurement, Mapping):
        raise RuntimeError(f"MEG 3.4 export missing for {source_file.name}")

    source_hash = _sha256_file(source_file)
    measurement = dict(measurement)
    measurement["provenance"] = {
        **dict(measurement.get("provenance", {})),
        "source_file": source_file.name,
        "source_file_sha256": source_hash,
        "source_file_size_bytes": int(source_file.stat().st_size),
        "preprocessing_support_status": str(
            (preprocessed.meta or {}).get("signal_support_provenance", {}).get("status", "unknown")
        ),
    }
    recording_root = output_root / source_file.parent.parent.name
    recording_root.mkdir(parents=True, exist_ok=True)
    export_path = recording_root / f"{source_file.stem}_mnps_v3_4_meg.h5"
    write_measurement_export_sidecar(
        export_path,
        measurement_id="v3_4_meg",
        measurement_payload=measurement,
        recording_identity={
            "dataset_id": "ds003645",
            "subject_id": source_file.parent.parent.name,
            "task": "FacePerception",
            "run": "1",
            "recording_id": source_file.stem,
        },
        skip_reason="bounded_measurement_pilot_without_canonical_MNPS_v2_summary",
    )

    coverage_rows: list[dict[str, Any]] = []
    redundancy_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    subject = source_file.parent.parent.name
    families = list(measurement.get("sensor_families", {}))
    for family in families:
        family_payload = measurement["sensor_families"][family]
        for variant_id, variant in family_payload["variants"].items():
            support = np.asarray(variant["coordinate_support"], dtype=bool)
            coverage_rows.append({
                "subject": subject,
                "source_file": source_file.name,
                "sensor_family": family,
                "variant": variant_id,
                "coordinate_count": len(variant["coordinate_names"]),
                "window_count": int(len(full_frame)),
                "complete_window_count": int(np.all(support, axis=1).sum()),
                "coordinate_support_fraction": float(support.mean()) if support.size else float("nan"),
                "export_h5": str(export_path),
            })
        analysis_names = list(dict.fromkeys([
            *MEG_V34_VARIANTS["existing_coordinate_set"],
            *MEG_V34_VARIANTS["alpha_replacement"],
        ]))
        for candidate in analysis_names:
            values = _feature_series(full_frame, family, candidate)
            statuses = full_frame.get(
                f"mnps_v3_4_meg_{family}_{candidate}_status", pd.Series(["missing"] * len(full_frame))
            ).astype(str)
            for status, count in statuses.value_counts(dropna=False).items():
                redundancy_rows.append({
                    "subject": subject,
                    "sensor_family": family,
                    "feature": candidate,
                    "record_type": "estimator_status_count",
                    "status": str(status),
                    "window_count": int(count),
                    "finite_value_count": int(np.isfinite(values).sum()),
                })
        for left_idx, left_name in enumerate(analysis_names):
            for right_name in analysis_names[left_idx + 1 :]:
                correlation, n_common = _spearman(
                    _feature_series(full_frame, family, left_name),
                    _feature_series(full_frame, family, right_name),
                )
                redundancy_rows.append({
                    "subject": subject,
                    "sensor_family": family,
                    "feature": left_name,
                    "other_feature": right_name,
                    "record_type": "within_recording_spearman",
                    "spearman_r": correlation,
                    "common_window_count": n_common,
                })

        labels = [str(value) for value in (preprocessed.channels or {}).get(f"meg_{family}", [])]
        if not labels:
            split_rows.append({
                "subject": subject,
                "sensor_family": family,
                "status": "missing_channel_labels",
            })
            continue
        split_a, split_b, split_info = _spatial_split_indices(labels, family, sensor_groups)
        if len(split_a) == 0 or len(split_b) == 0:
            split_rows.append({
                "subject": subject,
                "sensor_family": family,
                "status": "spatial_split_unsupported",
                **split_info,
            })
            continue
        split_config = copy.deepcopy(dict(config))
        split_robustness = split_config.get("robustness", {})
        if isinstance(split_robustness, dict) and isinstance(split_robustness.get("meg_ensembles"), dict):
            split_robustness["meg_ensembles"]["enabled"] = False
        left_payload = _make_spatial_split_payload(preprocessed, split_config, family, split_a)
        right_payload = _make_spatial_split_payload(preprocessed, split_config, family, split_b)
        left_frame = compute_meg_features(left_payload, split_config)
        right_frame = compute_meg_features(right_payload, split_config)
        for variant_id, coordinate_names in MEG_V34_VARIANTS.items():
            for coordinate in coordinate_names:
                left = _feature_series(left_frame, family, coordinate)
                right = _feature_series(right_frame, family, coordinate)
                correlation, n_common = _spearman(left, right)
                split_rows.append({
                    "subject": subject,
                    "sensor_family": family,
                    "variant": variant_id,
                    "coordinate": coordinate,
                    "status": "computed" if np.isfinite(correlation) else "insufficient_common_windows",
                    "split_spearman_r": correlation,
                    "common_window_count": n_common,
                    **split_info,
                })
    summary = {
        "subject": subject,
        "source_file": source_file.name,
        "source_file_sha256": source_hash,
        "source_file_size_bytes": int(source_file.stat().st_size),
        "preprocessed_sfreq_hz": float(preprocessed.sfreq),
        "window_count": int(len(full_frame)),
        "window_length_sec": float(np.nanmedian(pd.to_numeric(full_frame["t_end"], errors="coerce") - pd.to_numeric(full_frame["t_start"], errors="coerce"))),
        "window_step_sec": float(np.nanmedian(pd.to_numeric(full_frame["t_start"], errors="coerce").diff())) if len(full_frame) > 1 else None,
        "sensor_families": families,
        "legacy_output_columns_retained": bool("meg_sample_entropy" in full_frame.columns),
        "meg_v3_4_export": str(export_path),
    }
    return summary, coverage_rows, redundancy_rows + split_rows


def run_bounded_meg_v3_4_pilot(
    *,
    config_path: Path,
    dataset_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run the preselected five-recording pilot and emit H5/CSV/JSON results."""
    config = config_loader.load_config(Path(config_path))
    pilot = config.get("pilot_selection", {})
    if not isinstance(pilot, Mapping):
        raise ValueError("pilot_selection is required")
    subjects = [str(value) for value in pilot.get("subjects", [])]
    if not subjects or list(pilot.get("runs", [])) != [1]:
        raise ValueError("This bounded runner requires the frozen subject list and run [1]")
    task = str(pilot.get("task", "FacePerception"))
    if task != "FacePerception":
        raise ValueError("This runner only accepts the frozen FacePerception task selection")
    if not str(pilot.get("selection_status", "")) == "frozen_before_candidate_inspection":
        raise ValueError("Pilot selection must be marked frozen before candidate inspection")

    groups_cfg = config.get("robustness", {}).get("meg_ensembles", {})
    groups = groups_cfg.get("groups", {}) if isinstance(groups_cfg, Mapping) else {}
    if not isinstance(groups, Mapping):
        raise ValueError("The frozen geometry sector groups are required for sensor-split stability")
    selected_files: list[Path] = []
    for subject in subjects:
        file_path = Path(dataset_root) / subject / "meg" / f"{subject}_task-{task}_run-1_meg.fif"
        if not file_path.is_file():
            raise FileNotFoundError(f"Preselected pilot recording is missing: {file_path}")
        selected_files.append(file_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    analysis_rows: list[dict[str, Any]] = []
    for source_file in selected_files:
        summary, coverage, analysis = _run_recording(
            source_file,
            config=config,
            output_root=output_root,
            sensor_groups=groups,
        )
        summaries.append(summary)
        coverage_rows.extend(coverage)
        analysis_rows.extend(analysis)
    pd.DataFrame(coverage_rows).to_csv(output_root / "variant_coverage.csv", index=False)
    pd.DataFrame(analysis_rows).to_csv(output_root / "estimator_and_stability.csv", index=False)
    report = {
        "schema": "mndm.meg_mnps_v3_4.bounded_pilot_report.v1",
        "status": "complete",
        "definition_version": "3.4-meg-exp.1",
        "dataset_id": "ds003645",
        "selection": dict(pilot),
        "selected_source_files": [
            {"path": str(path), "size_bytes": int(path.stat().st_size), "sha256": item["source_file_sha256"]}
            for path, item in zip(selected_files, summaries)
        ],
        "preprocessing_config_path": str(Path(config_path).resolve()),
        "output_root": str(output_root.resolve()),
        "execution_environment": {
            "python_version": platform.python_version(),
            "numba_disable_jit": os.environ.get("NUMBA_DISABLE_JIT", "0"),
            "execution_note": "NUMBA_DISABLE_JIT was set to avoid import-time antropy specialization during this bounded pilot.",
        },
        "variants": list(config["mnps_v3_4_meg"].get("variants", [])),
        "estimand_boundary": {
            "inherited_features": "median waveform within MAG or GRAD, then existing estimator",
            "new_alpha_candidates": "channel-wise estimator, median across channels within one family",
            "families_combined": False,
            "event_contrast_inspected": False,
            "anchors_fitted": False,
        },
        "recordings": summaries,
        "summary_files": ["variant_coverage.csv", "estimator_and_stability.csv"],
    }
    (output_root / "pilot_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_bounded_meg_v3_4_pilot(
        config_path=args.config,
        dataset_root=args.dataset_root,
        output_root=args.output_root,
    )
    print(json.dumps({"status": report["status"], "recordings": len(report["recordings"]), "output_root": report["output_root"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
