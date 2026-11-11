"""Bounded deterministic-noise sensitivity check for WINDOW-SUPPORT-001."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np

from mndm.mnps.v3_2.candidate_grid import CandidateWindow
from mndm.mnps.v3_2.experiments import full_vector_challenge_001 as gate_b
from mndm.mnps.v3_2.experiments import full_vector_challenge_001_gate_c1 as gate_c1
from mndm.mnps.v3_2.experiments import window_support_001 as parent


PROTOCOL_ID = "WINDOW-SUPPORT-001-PERTURBATION-001"
NOISE_FRACTIONS = (0.001, 0.01)
BASELINE_MANIFEST_SHA256 = "d3cdcd72205279ae604d651c26a618d5323a4b613fa3f663af0c03173fa83365"
BASELINE_MANIFEST_DEFAULT = Path(
    r"C:\Users\Robin\.codex\visualizations\2026\10\11\01a12bb4-ebf8-7840-a3c6-bbd60bb89c7a"
    r"\WINDOW-SUPPORT-001\run_20261015_v1\manifest.json"
)
BASELINE_RUN_DEFAULT = BASELINE_MANIFEST_DEFAULT.parent
DEFAULT_PROTOCOL_PATH = (
    parent.REPO_ROOT / "project" / "new_mnps_v.3.2" / "protocols"
    / "WINDOW-SUPPORT-001-PERTURBATION-001.md"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_earliest_cadence_windows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Choose one earliest cadence interval per participant/condition/duration."""
    selected: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        if "cadence" not in row.get("window_grid_roles", []):
            continue
        key = (
            row.get("participant_id"), row.get("run_id"), row.get("condition"),
            row.get("duration_seconds"),
        )
        old = selected.get(key)
        if old is None or (
            int(row["support_start_sample"]), str(row["window_id"])
        ) < (
            int(old["support_start_sample"]), str(old["window_id"])
        ):
            selected[key] = row
    return [
        dict(row) for _, row in sorted(
            selected.items(), key=lambda item: tuple(str(value) for value in item[0])
        )
    ]


def deterministic_noise_seed(
    participant_id: str, run_id: str, window_id: str, noise_fraction: float
) -> tuple[str, int]:
    text = (
        f"{PROTOCOL_ID}|{participant_id}|{run_id}|{window_id}|"
        f"{noise_fraction:.6f}"
    )
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest, int(digest[:16], 16)


def perturb_signal(signals: np.ndarray, noise_fraction: float, seed: int) -> np.ndarray:
    """Add seeded white noise at a channel-specific fraction of window RMS."""
    if noise_fraction not in NOISE_FRACTIONS:
        raise ValueError("noise_fraction_not_in_frozen_protocol")
    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("signals_must_be_finite_channels_by_samples")
    rms = np.sqrt(np.mean(values * values, axis=1, dtype=np.float64))
    scales = noise_fraction * rms
    noise = np.random.default_rng(seed).normal(0.0, scales[:, None], size=values.shape)
    return values + noise


def _value(row: Mapping[str, Any]) -> Any:
    return row.get("component_values") if row.get("family_id") == "AR" else row.get("value")


def _is_finite(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(float(value))


def _array_deltas(left: Any, right: Any) -> tuple[list[float | None] | None, list[float | None] | None]:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return None, None
    deltas: list[float | None] = []
    absolute: list[float | None] = []
    for before, after in zip(left, right):
        if _is_finite(before) and _is_finite(after):
            delta = float(after) - float(before)
            deltas.append(delta)
            absolute.append(abs(delta))
        else:
            deltas.append(None)
            absolute.append(None)
    return deltas, absolute


def build_effect_row(
    baseline: Mapping[str, Any], perturbed: Mapping[str, Any], *,
    noise_fraction: float, seed_sha256: str, perturbed_window_sha256: str,
) -> dict[str, Any]:
    family = str(baseline["family_id"])
    before = _value(baseline)
    after = _value(perturbed)
    shape_match = (
        isinstance(before, list) and isinstance(after, list) and len(before) == len(after)
        if family == "AR" else True
    )
    signed_delta: Any = None
    absolute_delta: Any = None
    relative_delta: Any = None
    if family == "AR":
        signed_delta, absolute_delta = _array_deltas(before, after)
        if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
            relative_delta = [
                (float(right) - float(left)) / abs(float(left))
                if _is_finite(left) and _is_finite(right) and float(left) != 0.0 else None
                for left, right in zip(before, after)
            ]
    elif _is_finite(before) and _is_finite(after):
        signed_delta = float(after) - float(before)
        absolute_delta = abs(signed_delta)
        relative_delta = signed_delta / abs(float(before)) if float(before) != 0.0 else None
    return {
        "protocol_id": PROTOCOL_ID,
        "participant_id": baseline.get("participant_id"),
        "run_id": baseline.get("run_id"),
        "condition": baseline.get("condition"),
        "state_block_id": baseline.get("state_block_id"),
        "window_id": baseline.get("window_id"),
        "duration_seconds": baseline.get("duration_seconds"),
        "support_start_sample": baseline.get("support_start_sample"),
        "support_end_sample_exclusive": baseline.get("support_end_sample_exclusive"),
        "family_id": family,
        "duration_realization_id": baseline.get("duration_realization_id"),
        "noise_fraction_of_channel_rms": noise_fraction,
        "noise_model": "independent_zero_mean_gaussian_per_channel_and_sample",
        "noise_seed_sha256": seed_sha256,
        "baseline_window_signal_sha256": baseline.get("window_signal_sha256"),
        "perturbed_window_signal_sha256": perturbed_window_sha256,
        "baseline_computed": baseline.get("computed"),
        "baseline_valid": baseline.get("valid"),
        "baseline_status": baseline.get("status"),
        "baseline_reason": baseline.get("reason"),
        "perturbed_computed": perturbed.get("computed"),
        "perturbed_valid": perturbed.get("valid"),
        "perturbed_status": perturbed.get("status"),
        "perturbed_reason": perturbed.get("reason"),
        "shape_match": shape_match,
        "baseline_value": before,
        "perturbed_value": after,
        "signed_delta": signed_delta,
        "absolute_delta": absolute_delta,
        "relative_delta": relative_delta,
        "baseline_component_names": baseline.get("component_names") if family == "AR" else None,
        "perturbed_component_names": perturbed.get("component_names") if family == "AR" else None,
        "qualified": False,
    }


def summarize_effects(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["family_id"], int(row["duration_seconds"]), float(row["noise_fraction_of_channel_rms"]))
        groups.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        family, duration, noise_fraction = key
        component_indices: Sequence[int | None]
        if family == "AR":
            names = next((row.get("baseline_component_names") for row in group if row.get("baseline_component_names")), [])
            component_indices = range(len(names))
        else:
            component_indices = (None,)
        for index in component_indices:
            absolute: list[float] = []
            relative: list[float] = []
            for row in group:
                value = row.get("absolute_delta")
                rel = row.get("relative_delta")
                if index is None:
                    if _is_finite(value):
                        absolute.append(float(value))
                    if _is_finite(rel):
                        relative.append(abs(float(rel)))
                elif isinstance(value, list) and index < len(value) and _is_finite(value[index]):
                    absolute.append(float(value[index]))
                    if isinstance(rel, list) and index < len(rel) and _is_finite(rel[index]):
                        relative.append(abs(float(rel[index])))
            name = None
            if index is not None:
                name = next((row["baseline_component_names"][index] for row in group
                             if len(row.get("baseline_component_names") or []) > index), None)
            summary.append({
                "protocol_id": PROTOCOL_ID,
                "family_id": family,
                "duration_seconds": duration,
                "noise_fraction_of_channel_rms": noise_fraction,
                "component_index": index,
                "component_name": name,
                "windows_requested": len(group),
                "windows_with_finite_delta": len(absolute),
                "median_absolute_delta": float(np.median(absolute)) if absolute else None,
                "median_absolute_relative_delta": float(np.median(relative)) if relative else None,
                "status_flips": sum(
                    (row.get("baseline_computed"), row.get("baseline_valid"), row.get("baseline_status"), row.get("baseline_reason"))
                    != (row.get("perturbed_computed"), row.get("perturbed_valid"), row.get("perturbed_status"), row.get("perturbed_reason"))
                    for row in group
                ),
                "shape_flips": sum(row.get("shape_match") is False for row in group),
                "inference": "descriptive_single_seed_per_window_no_reliability_claim",
                "qualified": False,
            })
    return summary


def _read_baseline_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if "cadence" in row.get("window_grid_roles", []):
                rows.append(row)
    return rows


def _verify_baseline(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != BASELINE_MANIFEST_SHA256:
        raise ValueError("frozen_parent_manifest_missing_or_hash_changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_ids = ["sub-079", "sub-047", "sub-026", "sub-094"]
    if (
        manifest.get("protocol_id") != parent.PROTOCOL_ID
        or manifest.get("participants_processed") != expected_ids
        or manifest.get("qualified") is not False
        or manifest.get("c1_20s_parity_pass") is not True
    ):
        raise ValueError("parent_run_not_the_frozen_four_recording_pilot")
    for name, expected_output_hash in manifest.get("output_hashes", {}).items():
        output_path = run_dir / name
        if not output_path.is_file() or sha256_file(output_path) != expected_output_hash:
            raise ValueError(f"parent_output_hash_mismatch:{name}")
    measurements_path = run_dir / "window_measurements.jsonl.gz"
    expected_hash = manifest.get("output_hashes", {}).get("window_measurements.jsonl.gz")
    if expected_hash is None or sha256_file(measurements_path) != expected_hash:
        raise ValueError("parent_measurement_output_hash_mismatch")
    return manifest, json.loads((run_dir / "selection_manifest.json").read_text(encoding="utf-8"))


def _baseline_parity(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "computed_match": expected.get("computed") == actual.get("computed"),
        "valid_match": expected.get("valid") == actual.get("valid"),
        "status_match": expected.get("status") == actual.get("status"),
        "reason_match": expected.get("reason") == actual.get("reason"),
        "estimator_match": expected.get("estimator_id") == actual.get("estimator_id"),
        "signal_hash_match": expected.get("window_signal_sha256") == actual.get("window_signal_sha256"),
        "channel_hashes_match": (
            expected.get("channel_order_sha256") == actual.get("channel_order_sha256")
            and expected.get("channel_set_sha256") == actual.get("channel_set_sha256")
        ),
        "value_match": parent._values_match(parent._row_value_payload(expected), parent._row_value_payload(actual)),
    }
    return {**checks, "parity_pass": all(checks.values())}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                for key, value in row.items()
            })


def run(
    *, output_dir: Path, baseline_run_dir: Path = BASELINE_RUN_DEFAULT,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing_to_overwrite_nonempty_output_dir:{output_dir}")
    if not protocol_path.is_file():
        raise FileNotFoundError("frozen_perturbation_protocol_missing")
    parent_manifest, parent_selection = _verify_baseline(baseline_run_dir)
    protocol_sha256 = sha256_file(protocol_path)
    baseline_rows = _read_baseline_rows(baseline_run_dir / "window_measurements.jsonl.gz")
    selected_windows = choose_earliest_cadence_windows(baseline_rows)
    expected_count = 4 * 2 * len(parent.DURATIONS_SECONDS)
    if len(selected_windows) != expected_count:
        raise ValueError(f"unexpected_selected_window_count:{len(selected_windows)}")
    if {row["participant_id"] for row in selected_windows} != set(parent_manifest["participants_processed"]):
        raise ValueError("selected_windows_do_not_match_parent_participants")
    expected_keys = {
        (participant, condition, duration)
        for participant in parent_manifest["participants_processed"]
        for condition in ("EO", "EC")
        for duration in parent.DURATIONS_SECONDS
    }
    actual_keys = {
        (row["participant_id"], row["condition"], int(row["duration_seconds"]))
        for row in selected_windows
    }
    if actual_keys != expected_keys:
        raise ValueError("selected_windows_do_not_cover_frozen_participant_condition_duration_grid")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol_frozen.md").write_bytes(protocol_path.read_bytes())
    (output_dir / "selected_windows.json").write_text(
        json.dumps({
            "protocol_id": PROTOCOL_ID,
            "selection_rule": "earliest_cadence_start_per_participant_condition_duration_from_hash_verified_parent_output",
            "windows": [{key: row.get(key) for key in (
                "participant_id", "run_id", "condition", "state_block_id", "window_id",
                "duration_seconds", "support_start_sample", "support_end_sample_exclusive",
                "window_signal_sha256", "duration_realization_id",
            )} for row in selected_windows],
            "clinical_metadata_read": False,
            "qualified": False,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    roster = parent_selection.get("records")
    if not isinstance(roster, list):
        raise ValueError("parent_selection_manifest_missing_technical_records")
    roster_by_id = {str(record["participant_id"]): record for record in roster}
    rows_by_participant: dict[str, list[dict[str, Any]]] = {}
    for row in selected_windows:
        rows_by_participant.setdefault(str(row["participant_id"]), []).append(row)
    parity_rows: list[dict[str, Any]] = []
    effect_rows: list[dict[str, Any]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    for participant_id in parent_manifest["participants_processed"]:
        run_metadata = gate_c1._metadata_record_to_run(roster_by_id[participant_id])
        gate_b._verify_frozen_inputs(run_metadata)
        eeg_json = gate_b._read_json(Path(run_metadata.eeg_json_path))
        acquisition_reference = str(eeg_json.get("EEGReference", "unknown"))
        data, source_fdt_sha256 = gate_b._source_memmap(run_metadata)
        try:
            participant_windows = rows_by_participant[participant_id]
            for record in participant_windows:
                duration = int(record["duration_seconds"])
                window = CandidateWindow(
                    window_id=str(record["window_id"]),
                    block_id=str(record["state_block_id"]),
                    condition=str(record["condition"]),
                    start_sample=int(record["support_start_sample"]),
                    end_sample_exclusive=int(record["support_end_sample_exclusive"]),
                )
                signals, labels = gate_b._window_data(data, run_metadata, window)
                if labels != run_metadata.eeg_channel_labels:
                    raise ValueError("frozen_EEG_label_order_changed_since_parent_run")
                source_signal_sha256 = gate_b._window_hash(
                    signals, labels, run_metadata.sample_rate_hz, parent.SOURCE_REFERENCE_ID
                )
                expected_base_rows = {
                    str(row["family_id"]): row for row in baseline_rows
                    if row.get("participant_id") == participant_id
                    and row.get("window_id") == record["window_id"]
                    and row.get("family_id") in parent.FAMILY_IDS
                }
                scheduled = parent.ScheduledWindow(
                    window=window, duration_seconds=duration,
                    grid_roles=tuple(record.get("window_grid_roles", [])),
                    matched_anchor_id=record.get("matched_anchor_window_id"),
                    matched_anchor_start_sample=record.get("matched_anchor_start_sample"),
                    matched_anchor_end_sample_exclusive=record.get("matched_anchor_end_sample_exclusive"),
                )
                recomputed = parent.compute_window_rows(
                    run_metadata, scheduled, data, source_fdt_sha256, acquisition_reference
                )
                recomputed_by_family = {str(row["family_id"]): row for row in recomputed}
                for family in parent.FAMILY_IDS:
                    expected = expected_base_rows[family]
                    actual = recomputed_by_family[family]
                    parity = _baseline_parity(expected, actual)
                    parity_rows.append({
                        "participant_id": participant_id, "run_id": run_metadata.run_id,
                        "condition": window.condition, "state_block_id": window.block_id,
                        "window_id": window.window_id, "duration_seconds": duration,
                        "family_id": family, **parity,
                    })
                    if not parity["parity_pass"]:
                        raise ValueError(f"baseline_recomputation_parity_failed:{participant_id}:{window.window_id}:{family}")

                for noise_fraction in NOISE_FRACTIONS:
                    seed_sha256, seed = deterministic_noise_seed(
                        participant_id, run_metadata.run_id, window.window_id, noise_fraction
                    )
                    perturbed_signal = perturb_signal(signals, noise_fraction, seed)
                    perturbed_signal_sha256 = gate_b._window_hash(
                        perturbed_signal, labels, run_metadata.sample_rate_hz, parent.SOURCE_REFERENCE_ID
                    )
                    perturbed_results = parent._family_kernel_results(
                        perturbed_signal, labels, fs=run_metadata.sample_rate_hz,
                        run=run_metadata, window=window, source_fdt_sha256=source_fdt_sha256,
                    )
                    for family in parent.FAMILY_IDS:
                        effect_rows.append(build_effect_row(
                            expected_base_rows[family], perturbed_results[family],
                            noise_fraction=noise_fraction,
                            seed_sha256=seed_sha256,
                            perturbed_window_sha256=perturbed_signal_sha256,
                        ))
            source_hashes[participant_id] = {
                "set": run_metadata.set_sha256,
                "fdt": source_fdt_sha256,
                "events": run_metadata.events_sha256,
                "channels": run_metadata.channels_sha256,
                "eeg_json": run_metadata.eeg_json_sha256,
            }
        finally:
            del data

    _write_csv(output_dir / "baseline_recomputation_parity.csv", parity_rows)
    _write_csv(output_dir / "perturbation_effects.csv", effect_rows)
    summary_rows = summarize_effects(effect_rows)
    _write_csv(output_dir / "perturbation_summary.csv", summary_rows)
    output_names = [
        "protocol_frozen.md", "selected_windows.json", "baseline_recomputation_parity.csv",
        "perturbation_effects.csv", "perturbation_summary.csv",
    ]
    runner_path = Path(__file__).resolve()
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETED_BOUNDED_PERTURBATION_AUDIT",
        "qualified": False,
        "clinical_metadata_read": False,
        "analyst_blinded": False,
        "parent_protocol_id": parent.PROTOCOL_ID,
        "parent_manifest_sha256": BASELINE_MANIFEST_SHA256,
        "parent_protocol_sha256": parent_manifest["analysis_protocol_sha256"],
        "parent_runner_sha256": parent_manifest["analysis_runner_sha256"],
        "perturbation_protocol_sha256": protocol_sha256,
        "perturbation_runner_sha256": sha256_file(runner_path),
        "participants_processed": parent_manifest["participants_processed"],
        "participant_count": len(parent_manifest["participants_processed"]),
        "selected_windows": len(selected_windows),
        "noise_fractions_of_channel_rms": list(NOISE_FRACTIONS),
        "families": list(parent.FAMILY_IDS),
        "baseline_recomputation_rows": len(parity_rows),
        "baseline_recomputation_parity_pass": bool(parity_rows) and all(row["parity_pass"] for row in parity_rows),
        "perturbation_effect_rows": len(effect_rows),
        "source_hashes": source_hashes,
        "implementation_code_sha256": {
            "perturbation_runner": sha256_file(runner_path),
            "parent_runner": sha256_file(Path(parent.__file__)),
            "gate_b_runner": sha256_file(Path(gate_b.__file__)),
            "gate_c1_runner": sha256_file(Path(gate_c1.__file__)),
        },
        "software": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
        },
        "output_hashes": {name: sha256_file(output_dir / name) for name in output_names},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run-dir", type=Path, default=BASELINE_RUN_DEFAULT)
    parser.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = run(
        output_dir=args.output_dir,
        baseline_run_dir=args.baseline_run_dir,
        protocol_path=args.protocol_path,
    )
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "status": manifest["status"],
        "output_dir": str(args.output_dir.resolve()),
        "selected_windows": manifest["selected_windows"],
        "effect_rows": manifest["perturbation_effect_rows"],
        "baseline_parity_pass": manifest["baseline_recomputation_parity_pass"],
        "qualified": False,
    }, indent=2))
    return 0 if manifest["baseline_recomputation_parity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
