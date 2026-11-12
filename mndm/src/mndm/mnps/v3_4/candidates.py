"""Window-supported EEG estimators used by the experimental MNPS 3.4 surface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import signal


def resolve_v3_4_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve either the standalone v3.4 YAML or a nested v3.4 overlay."""
    nested = config.get("mnps_v3_4")
    if isinstance(nested, Mapping):
        return nested
    if config.get("schema") == "mnps_flat_coordinates" and str(config.get("definition_version")) == "3.4":
        return config
    return {}


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def runtime_implementation_hashes() -> dict[str, str]:
    """Hash the estimator and row assembly code used for this realization."""
    module_dir = Path(__file__).resolve().parent
    eeg_module = module_dir.parents[1] / "features" / "eeg.py"
    coordinate_module = module_dir / "coordinates.py"
    preprocess_module = module_dir.parents[1] / "preprocess.py"
    return {
        "candidate_estimators_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "feature_row_assembly_sha256": hashlib.sha256(eeg_module.read_bytes()).hexdigest(),
        "coordinate_assembly_sha256": hashlib.sha256(coordinate_module.read_bytes()).hexdigest(),
        "preprocessing_pipeline_sha256": hashlib.sha256(preprocess_module.read_bytes()).hexdigest(),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_component_hashes(source_file: str | None) -> dict[str, dict[str, Any]]:
    """Hash a raw recording and the ordinary same-stem BrainVision/BIDS components."""
    if not source_file:
        return {}
    path = Path(str(source_file))
    candidates: list[Path] = [path]
    if path.suffix.lower() == ".vhdr":
        candidates.extend((path.with_suffix(".eeg"), path.with_suffix(".vmrk")))
        stem = path.stem.removesuffix("_eeg")
        for suffix in ("_eeg.json", "_channels.tsv", "_events.tsv", "_electrodes.tsv", "_coordsystem.json"):
            candidates.append(path.with_name(f"{stem}{suffix}"))
    components: dict[str, dict[str, Any]] = {}
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        components[candidate.name] = {
            "sha256": _sha256_file(candidate),
            "size_bytes": int(candidate.stat().st_size),
            "component_role": (
                "signal_samples" if candidate.suffix.lower() == ".eeg"
                else "event_markers" if candidate.suffix.lower() == ".vmrk"
                else "primary_source_or_bids_sidecar"
            ),
        }
    return components


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    ends = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _as_json_numeric(values: Sequence[float]) -> str:
    return json.dumps([float(v) if np.isfinite(v) else None for v in values], separators=(",", ":"))


def _unsupported(reason: str, n_channels: int) -> dict[str, Any]:
    missing = [float("nan")] * n_channels
    return {
        "alpha_centroid_hz": float("nan"),
        "alpha_centroid_status": reason,
        "alpha_centroid_channel_count": 0,
        "alpha_centroid_hz_per_channel": list(missing),
        "alpha_envelope_cv": float("nan"),
        "alpha_envelope_status": reason,
        "alpha_envelope_channel_count": 0,
        "alpha_envelope_cv_per_channel": list(missing),
        "spatial_logvariance_dispersion": float("nan"),
        "spatial_logvariance_status": reason,
        "spatial_logvariance_channel_count": 0,
        "spatial_logvariance_per_channel": list(missing),
    }


def compute_eeg_candidates(
    eeg_data: np.ndarray,
    *,
    sfreq: float,
    windows: Sequence[tuple[int, int, int]],
    channel_labels: Sequence[str],
    config: Mapping[str, Any],
    dataset_id: str | None,
    reference: str | None,
    reference_set_hash: str | None,
    source_file: str | None,
    preprocessed_segmented_epochs: bool = False,
) -> list[dict[str, Any]]:
    """Compute the 3.4 candidate estimators from each true continuous segment.

    ``windows`` contains ``(epoch_id, start_sample, stop_sample)`` triples on
    the same sample grid as ``eeg_data``. Per-channel candidates are retained
    alongside the channel-median scalars. A full channel set is required for
    each window; missing channels are never dropped or imputed.
    """
    cfg = resolve_v3_4_config(config)
    if not cfg or not bool(cfg.get("enabled", False)):
        return []
    surface = cfg.get("measurement_surface", {})
    estimators = cfg.get("estimators", {})
    overlay = surface.get("dataset_overlay") if isinstance(surface, Mapping) else None
    if not overlay:
        raise ValueError("MNPS 3.4 requires a dataset_overlay before feature extraction")
    required_overlay = list(surface.get("required_overlay_fields", []) or []) if isinstance(surface, Mapping) else []
    missing_overlay = [key for key in required_overlay if not _overlay_value(cfg, str(key))]
    if missing_overlay:
        raise ValueError(f"MNPS 3.4 dataset overlay is incomplete: {missing_overlay}")

    label_compatibility = str(_overlay_value(cfg, "channel_label_set_compatibility") or "").strip()
    if label_compatibility != "casefold_for_anchor_matching; exact_labels_preserved":
        raise ValueError(
            "MNPS 3.4 requires channel_label_set_compatibility="
            "'casefold_for_anchor_matching; exact_labels_preserved'"
        )

    data = np.asarray(eeg_data, dtype=np.float64)
    labels = [str(value) for value in channel_labels]
    normalized_labels = [value.strip().casefold() for value in labels]
    if data.ndim != 2 or data.shape[0] != len(labels) or not labels:
        raise ValueError("MNPS 3.4 requires ordered EEG channel labels matching the signal array")
    if len(set(normalized_labels)) != len(normalized_labels):
        raise ValueError("MNPS 3.4 native channel labels must be unique")
    reference_known = bool(reference and reference_set_hash)
    reference_value = str(reference) if reference else "unknown_reference"
    reference_hash_value = str(reference_set_hash) if reference_set_hash else "unknown_reference_set"

    sfreq = float(sfreq)
    if not np.isfinite(sfreq) or sfreq <= 0:
        raise ValueError("MNPS 3.4 requires a positive realized sampling rate")
    candidate_cfg = estimators.get("candidates", {}) if isinstance(estimators, Mapping) else {}
    centroid_cfg = candidate_cfg.get("alpha_centroid_hz", {}) if isinstance(candidate_cfg, Mapping) else {}
    envelope_cfg = candidate_cfg.get("alpha_envelope_cv", {}) if isinstance(candidate_cfg, Mapping) else {}
    centroid_psd = centroid_cfg.get("psd", {}) if isinstance(centroid_cfg, Mapping) else {}
    segment_sec = float(centroid_psd.get("segment_sec", 4.0))
    overlap_fraction = float(centroid_psd.get("overlap_fraction", 0.5))
    fmin, fmax = [float(x) for x in centroid_cfg.get("frequency_range_hz", (7.0, 13.0))]
    nperseg = int(round(segment_sec * sfreq))
    noverlap = int(round(nperseg * overlap_fraction))
    if nperseg < 4 or not 0 <= noverlap < nperseg:
        raise ValueError("MNPS 3.4 Welch segment/overlap configuration is invalid for realized sfreq")
    if fmax > sfreq / 2.0:
        raise ValueError("MNPS 3.4 alpha-centroid upper frequency exceeds Nyquist")

    alpha_band = envelope_cfg.get("band_hz", (8.0, 12.0))
    low, high = float(alpha_band[0]), float(alpha_band[1])
    order = int((envelope_cfg.get("bandpass", {}) or {}).get("order", 4))
    guard_sec = float(envelope_cfg.get("edge_support_guard_sec", 10.0))
    guard = int(np.ceil(guard_sec * sfreq))
    sos = signal.butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    # The zero-phase filter and segment-wide Hilbert transform have finite
    # recording-segment support. The declared guard is conservative and is
    # carried as an explicit support policy; it is not a biological validity
    # threshold.
    envelope_by_sample = np.full(data.shape, np.nan, dtype=np.float64)
    finite_sample = np.isfinite(data).all(axis=0)
    continuous_runs = [] if preprocessed_segmented_epochs else _true_runs(finite_sample)
    padlen = 3 * (2 * len(sos) + 1)
    for run_start, run_end in continuous_runs:
        segment = data[:, run_start:run_end]
        if segment.shape[1] <= padlen:
            continue
        try:
            filtered = signal.sosfiltfilt(sos, segment, axis=1)
            envelope_by_sample[:, run_start:run_end] = np.abs(signal.hilbert(filtered, axis=1))
        except ValueError:
            continue

    output: list[dict[str, Any]] = []
    implementation_hashes = runtime_implementation_hashes()
    source_component_hashes = _source_component_hashes(source_file)
    for _, start, stop in windows:
        start, stop = int(start), int(stop)
        width = stop - start
        record = _unsupported("insufficient_full_channel_support", len(labels))
        if start < 0 or stop > data.shape[1] or width <= 0:
            record["alpha_centroid_status"] = "window_outside_source_support"
            record["alpha_envelope_status"] = "window_outside_source_support"
            record["spatial_logvariance_status"] = "window_outside_source_support"
            output.append(record)
            continue
        if preprocessed_segmented_epochs:
            record["alpha_envelope_status"] = "continuous_source_boundaries_unavailable_for_concatenated_epochs"
        window = data[:, start:stop]
        all_finite = bool(window.size and np.isfinite(window).all())
        if not all_finite:
            record["alpha_centroid_status"] = "nonfinite_required_channel_support"
            record["alpha_envelope_status"] = "nonfinite_required_channel_support"
            record["spatial_logvariance_status"] = "nonfinite_required_channel_support"
            output.append(record)
            continue

        # Candidate 1: Welch PSD per channel, median across channels at every
        # frequency, then the frozen 7-13 Hz power-weighted spectral centroid.
        nperseg_window = min(nperseg, width)
        overlap_window = min(noverlap, max(0, nperseg_window - 1))
        freqs, psd_by_channel = signal.welch(
            window,
            fs=sfreq,
            window="hann",
            nperseg=nperseg_window,
            noverlap=overlap_window,
            detrend="constant",
            scaling="density",
            axis=1,
        )
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        if band_mask.any():
            median_psd = np.median(psd_by_channel[:, band_mask], axis=0)
            band_freqs = freqs[band_mask]
            denominator = float(np.sum(median_psd))
            if np.isfinite(denominator) and denominator > 0:
                record["alpha_centroid_hz"] = float(np.sum(band_freqs * median_psd) / denominator)
                record["alpha_centroid_status"] = "computed"
                channel_denominator = np.sum(psd_by_channel[:, band_mask], axis=1)
                channel_numerator = np.sum(psd_by_channel[:, band_mask] * band_freqs[None, :], axis=1)
                per_channel = np.divide(
                    channel_numerator,
                    channel_denominator,
                    out=np.full(len(labels), np.nan, dtype=np.float64),
                    where=np.isfinite(channel_denominator) & (channel_denominator > 0),
                )
                record["alpha_centroid_hz_per_channel"] = per_channel.tolist()
                record["alpha_centroid_channel_count"] = int(np.isfinite(per_channel).sum())
            else:
                record["alpha_centroid_status"] = "nonpositive_psd_denominator"
        else:
            record["alpha_centroid_status"] = "alpha_band_outside_welch_grid"

        # Auxiliary sp_disp: robust spread of channel log-variance values.
        variances = np.var(window, axis=1, ddof=0)
        valid_variances = np.isfinite(variances) & (variances > 0)
        log_variances = np.full(len(labels), np.nan, dtype=np.float64)
        log_variances[valid_variances] = np.log10(variances[valid_variances])
        if bool(valid_variances.all()):
            center = float(np.median(log_variances))
            record["spatial_logvariance_dispersion"] = float(1.4826 * np.median(np.abs(log_variances - center)))
            record["spatial_logvariance_status"] = "computed"
            record["spatial_logvariance_per_channel"] = log_variances.tolist()
            record["spatial_logvariance_channel_count"] = int(len(labels))
        else:
            record["spatial_logvariance_status"] = "nonpositive_channel_variance"
            record["spatial_logvariance_per_channel"] = log_variances.tolist()
            record["spatial_logvariance_channel_count"] = int(valid_variances.sum())

        # Candidate 2: filter the full finite source segment before extracting
        # each window, then compute per-channel Hilbert-envelope CV.
        supporting_run = next(((a, b) for a, b in continuous_runs if a <= start and stop <= b), None)
        if preprocessed_segmented_epochs:
            pass
        elif supporting_run is None:
            record["alpha_envelope_status"] = "window_crosses_continuous_segment_boundary"
        else:
            run_start, run_end = supporting_run
            if start - run_start < guard or run_end - stop < guard:
                record["alpha_envelope_status"] = "within_declared_segment_edge_guard"
            else:
                env = envelope_by_sample[:, start:stop]
                mean = np.mean(env, axis=1)
                std = np.std(env, axis=1, ddof=0)
                valid = np.isfinite(mean) & np.isfinite(std) & (mean > 0)
                per_channel_cv = np.full(len(labels), np.nan, dtype=np.float64)
                per_channel_cv[valid] = std[valid] / mean[valid]
                if bool(valid.all()):
                    record["alpha_envelope_cv"] = float(np.median(per_channel_cv))
                    record["alpha_envelope_status"] = "computed"
                else:
                    record["alpha_envelope_status"] = "nonpositive_envelope_mean"
                record["alpha_envelope_cv_per_channel"] = per_channel_cv.tolist()
                record["alpha_envelope_channel_count"] = int(valid.sum())
        output.append(record)

    for record in output:
        record["channel_labels_json"] = json.dumps(labels, ensure_ascii=False, separators=(",", ":"))
        record["channel_set_hash"] = _stable_hash(sorted(labels))
        record["channel_compatibility_set_hash"] = _stable_hash(sorted(normalized_labels))
        record["channel_order_hash"] = _stable_hash(labels)
        record["reference"] = reference_value
        record["reference_set_hash"] = reference_hash_value
        record["reference_status"] = "known" if reference_known else "unknown_input_preserved"
        record["dataset_id"] = str(dataset_id or "unknown_dataset")
        record["source_file"] = Path(str(source_file)).name if source_file else "unknown_source"
        record["alpha_centroid_hz_per_channel_json"] = _as_json_numeric(record["alpha_centroid_hz_per_channel"])
        record["alpha_envelope_cv_per_channel_json"] = _as_json_numeric(record["alpha_envelope_cv_per_channel"])
        record["spatial_logvariance_per_channel_json"] = _as_json_numeric(record["spatial_logvariance_per_channel"])
        record["alpha_envelope_edge_guard_sec"] = guard_sec
        record["implementation_hashes_json"] = json.dumps(
            implementation_hashes, sort_keys=True, separators=(",", ":")
        )
        record["source_component_hashes_json"] = json.dumps(
            source_component_hashes, sort_keys=True, separators=(",", ":")
        )
        record["source_surface_hash"] = _stable_hash(
            {
                "definition_version": str(cfg.get("definition_version", "3.4")),
                "dataset_id": str(dataset_id or "unknown_dataset"),
                "dataset_version": _overlay_value(cfg, "dataset_version"),
                "sfreq_hz": sfreq,
                "channel_compatibility_set_hash": record["channel_compatibility_set_hash"],
                "channel_label_set_compatibility": label_compatibility,
                "reference": reference_value,
                "reference_set_hash": reference_hash_value,
                "preprocess": config.get("preprocess", {}),
                "features": config.get("features", {}),
                "epoching": config.get("epoching", {}),
                "window_length_sec": surface.get("window_length_sec"),
                "window_step_sec": surface.get("window_step_sec"),
                "estimators": estimators,
                "implementation_hashes": implementation_hashes,
            }
        )
    return output


def _overlay_value(config: Mapping[str, Any], key: str) -> Any:
    surface = config.get("measurement_surface", {})
    overlay = surface.get("dataset_overlay", {}) if isinstance(surface, Mapping) else {}
    if isinstance(overlay, Mapping) and key in overlay:
        return overlay[key]
    if isinstance(overlay, str) and key == "dataset_version":
        return overlay
    return None


__all__ = ["compute_eeg_candidates", "resolve_v3_4_config", "runtime_implementation_hashes"]
