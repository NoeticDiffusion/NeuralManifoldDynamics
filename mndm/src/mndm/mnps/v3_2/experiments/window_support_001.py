"""Bounded, outcome-excluded multiscale support benchmark for ds003478."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import signal

from mndm.mnps.v3_2.ar import compute_ar_activity_transition_vector
from mndm.mnps.v3_2.candidate_grid import (
    CandidateWindow,
    ConditionBlock,
    enumerate_full_windows,
    reconstruct_condition_blocks,
)
from mndm.mnps.v3_2.ed import compute_ed_raw_sensor_covariance_participation_ratio
from mndm.mnps.v3_2.oi import compute_oi_matrix
from mndm.mnps.v3_2.pcv import compute_pcv_phase_order_parameter_sd
from mndm.mnps.v3_2.rd import compute_rd_centered_log_psd_dispersion
from mndm.mnps.v3_2.registry import get_full_vector_registry
from mndm.mnps.v3_2.sc import compute_sc_lz78_exact_multiset_ratio
from mndm.mnps.v3_2.tp import compute_tp_log_rms_envelope_persistence
from mndm.mnps.v3_2.experiments import full_vector_challenge_001 as gate_b
from mndm.mnps.v3_2.experiments import full_vector_challenge_001_gate_c1 as gate_c1


PROTOCOL_ID = "WINDOW-SUPPORT-001"
PROTOCOL_VERSION = "v1"
DURATIONS_SECONDS = (4, 8, 12, 20)
AR_BLOCK_SECONDS = 2
MATCHED_ANCHOR_SECONDS = 20
SELECTED_PARTICIPANT_COUNT = 4
SELECTION_PREFIX = "WINDOW-SUPPORT-001|"
SELECTION_SHA256 = "d6016fa5baa6922cd432542ca92dd958766c640cf7544fb149389ef8698b5964"
C1_MANIFEST_SHA256 = "f11990ce65c70e8ddc51511088c4099eb42332bc5f01573f7699313b27489c2d"
C1_MANIFEST_DEFAULT = Path(
    r"C:\Users\Robin\.codex\visualizations\2026\10\11\01a12bb4-ebf8-7840-a3c6-bbd60bb89c7a"
    r"\FULL-VECTOR-CHALLENGE-001\GATE-C1\full_20261014T140444Z\manifest.json"
)
C1_RUN_DEFAULT = C1_MANIFEST_DEFAULT.parent
REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_SELECTION_PATH = REPO_ROOT / "mndm" / "config" / (
    "mnps_v3_2_full_vector_challenge_001_gate_c1_selection_20261014_r1.json"
)
DEFAULT_PROTOCOL_PATH = REPO_ROOT / "project" / "new_mnps_v.3.2" / "protocols" / "WINDOW-SUPPORT-001.md"
FAMILY_IDS = ("PCV", "TP", "AR", "NIE", "RD", "OI", "SC", "ED")
SOURCE_REFERENCE_ID = "acquisition_reference_from_BIDS_sidecar"
MATCH_RTOL = 1e-12
MATCH_ATOL = 1e-12


@dataclass(frozen=True)
class ScheduledWindow:
    window: CandidateWindow
    duration_seconds: int
    grid_roles: tuple[str, ...]
    matched_anchor_id: str | None = None
    matched_anchor_start_sample: int | None = None
    matched_anchor_end_sample_exclusive: int | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_technical_roster(roster: Sequence[Mapping[str, Any]], count: int = SELECTED_PARTICIPANT_COUNT) -> list[Mapping[str, Any]]:
    """Choose a deterministic subset using participant IDs only."""
    if count < 1 or count > len(roster):
        raise ValueError("selection_count_outside_technical_roster")
    ids = [str(row.get("participant_id", "")) for row in roster]
    if any(not pid for pid in ids) or len(set(ids)) != len(ids):
        raise ValueError("technical_roster_has_missing_or_duplicate_participant_ids")
    return sorted(
        roster,
        key=lambda row: hashlib.sha256(
            (SELECTION_PREFIX + str(row["participant_id"])).encode("utf-8")
        ).hexdigest(),
    )[:count]


def _boundary_sample_indices(events: Iterable[Mapping[str, Any]], sample_rate_hz: float) -> tuple[int, ...]:
    samples: set[int] = set()
    for event in events:
        is_boundary = (
            str(event.get("trial_type", "")).strip().casefold() == "boundary"
            or str(event.get("value", "")).strip().casefold() == "boundary"
        )
        if not is_boundary:
            continue
        try:
            onset = float(event.get("onset", "nan"))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(onset) and onset >= 0:
            samples.add(int(round(onset * sample_rate_hz)))
    return tuple(sorted(samples))


def split_blocks_at_boundaries(
    blocks: Iterable[ConditionBlock], boundary_samples: Iterable[int]
) -> tuple[ConditionBlock, ...]:
    """Split condition blocks at explicit source boundary markers."""
    boundaries = tuple(sorted(set(int(value) for value in boundary_samples)))
    output: list[ConditionBlock] = []
    for block in blocks:
        interior = [value for value in boundaries if block.start_sample < value < block.end_sample_exclusive]
        if not interior:
            output.append(block)
            continue
        edges = [block.start_sample, *interior, block.end_sample_exclusive]
        for index, (start, end) in enumerate(zip(edges, edges[1:]), start=1):
            if end > start:
                output.append(ConditionBlock(
                    block_id=f"{block.block_id}-boundary-{index:02d}",
                    condition=block.condition,
                    start_sample=start,
                    end_sample_exclusive=end,
                    start_event_onset_seconds=block.start_event_onset_seconds,
                ))
    return tuple(output)


def schedule_duration_windows(
    blocks: Iterable[ConditionBlock], *, sample_rate_hz: float, duration_seconds: int
) -> tuple[ScheduledWindow, ...]:
    """Union the nonoverlapping cadence grid with matched 20-second prefixes."""
    if duration_seconds not in DURATIONS_SECONDS:
        raise ValueError("duration_not_in_frozen_grid")
    block_rows = tuple(blocks)
    scheduled: dict[tuple[str, int, int], dict[str, Any]] = {}

    def add(window: CandidateWindow, role: str, anchor: CandidateWindow | None = None) -> None:
        key = (window.block_id, window.start_sample, window.end_sample_exclusive)
        item = scheduled.get(key)
        if item is None:
            item = {"window": window, "roles": set(), "anchor": None}
            scheduled[key] = item
        item["roles"].add(role)
        if anchor is not None:
            item["anchor"] = anchor

    prefix = "fv20" if duration_seconds == 20 else f"ws{duration_seconds:02d}"
    cadence = enumerate_full_windows(
        block_rows,
        sample_rate_hz=sample_rate_hz,
        window_seconds=float(duration_seconds),
        window_prefix=prefix,
    )
    for window in cadence:
        add(window, "cadence")

    anchors = enumerate_full_windows(
        block_rows,
        sample_rate_hz=sample_rate_hz,
        window_seconds=float(MATCHED_ANCHOR_SECONDS),
        window_prefix="fv20",
    )
    window_samples = int(round(duration_seconds * sample_rate_hz))
    for anchor_index, anchor in enumerate(anchors, start=1):
        matched = CandidateWindow(
            window_id=(anchor.window_id if duration_seconds == 20 else
                       f"{anchor.block_id}-m20p{duration_seconds:02d}-{anchor_index:03d}"),
            block_id=anchor.block_id,
            condition=anchor.condition,
            start_sample=anchor.start_sample,
            end_sample_exclusive=anchor.start_sample + window_samples,
        )
        add(matched, "matched_20s_anchor_prefix", anchor)

    output: list[ScheduledWindow] = []
    for item in scheduled.values():
        window = item["window"]
        anchor = item["anchor"]
        output.append(ScheduledWindow(
            window=window,
            duration_seconds=duration_seconds,
            grid_roles=tuple(sorted(item["roles"])),
            matched_anchor_id=anchor.window_id if anchor else None,
            matched_anchor_start_sample=anchor.start_sample if anchor else None,
            matched_anchor_end_sample_exclusive=anchor.end_sample_exclusive if anchor else None,
        ))
    return tuple(sorted(output, key=lambda row: (row.window.start_sample, row.window.end_sample_exclusive, row.window.window_id)))


def ar_realization_metadata(duration_seconds: int) -> dict[str, Any]:
    if duration_seconds not in DURATIONS_SECONDS:
        raise ValueError("duration_not_in_frozen_grid")
    blocks = duration_seconds // AR_BLOCK_SECONDS
    return {
        "duration_realization_id": f"WINDOW-SUPPORT-001.AR.window{duration_seconds:02d}s_rms2s_{blocks}blocks.v1",
        "block_duration_seconds": AR_BLOCK_SECONDS,
        "block_count": blocks,
        "component_count": blocks - 1,
        "vector_construction_id": (
            gate_b.AR_VECTOR_CONSTRUCTION_ID if blocks == 10 else
            f"per_channel_rms_{blocks}_contiguous_2s_blocks.window_support_001.v1"
        ),
        "scaling_id": gate_b.AR_SCALING_ID,
    }


def duration_realization_id(family_id: str, duration_seconds: int) -> str:
    if family_id == "AR":
        return str(ar_realization_metadata(duration_seconds)["duration_realization_id"])
    if family_id not in FAMILY_IDS or duration_seconds not in DURATIONS_SECONDS:
        raise ValueError("unknown_family_or_duration")
    return f"WINDOW-SUPPORT-001.{family_id}.window{duration_seconds:02d}s.source_reference.v1"


def _family_kernel_results(
    signals: np.ndarray,
    labels: Sequence[str],
    *,
    fs: float,
    run: gate_b.RunMetadata,
    window: CandidateWindow,
    source_fdt_sha256: str,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    sos = signal.butter(4, (8.0, 13.0), btype="bandpass", fs=fs, output="sos")

    try:
        alpha = signal.sosfiltfilt(sos, signals, axis=-1, padtype=None, padlen=0)
        phase, reason = gate_b._phase_from_alpha(alpha)
        results["PCV"] = (
            gate_b._unsupported_reason("PCV", reason) if reason is not None else
            gate_b._metric_record("PCV", compute_pcv_phase_order_parameter_sd(
                phase,
                phase_estimator_id=gate_b.PCV_PHASE_ESTIMATOR_ID,
                band_id=gate_b.PCV_BAND_ID,
                channel_labels=labels,
            ), "source_reference")
        )
    except Exception as exc:
        results["PCV"] = gate_b._unsupported_exception("PCV", exc)

    try:
        # Recompute as in Gate C1 so this path preserves the frozen estimator
        # implementation and per-window filter realization.
        alpha = signal.sosfiltfilt(sos, signals, axis=-1, padtype=None, padlen=0)
        results["TP"] = gate_b._metric_record(
            "TP",
            compute_tp_log_rms_envelope_persistence(
                alpha, labels, sample_rate_hz=fs, envelope_samples=500,
                lag_samples=500, epsilon_amplitude=1e-12, minimum_pair_count=20,
            ),
            "source_reference",
        )
    except Exception as exc:
        results["TP"] = gate_b._unsupported_exception("TP", exc)

    try:
        metadata = ar_realization_metadata(int(round(window.sample_count / fs)))
        block_samples = int(round(AR_BLOCK_SECONDS * fs))
        if window.sample_count % block_samples:
            raise ValueError("window_not_divisible_into_complete_2s_AR_blocks")
        block_vectors = np.asarray([
            np.sqrt(np.mean(signals[:, start:start + block_samples] ** 2, axis=1, dtype=np.float64))
            for start in range(0, window.sample_count, block_samples)
        ], dtype=np.float64)
        start_sec = window.start_sample / fs
        intervals = [
            (start_sec + index * AR_BLOCK_SECONDS, start_sec + (index + 1) * AR_BLOCK_SECONDS)
            for index in range(block_vectors.shape[0])
        ]
        ar_result = compute_ar_activity_transition_vector(
            block_vectors, intervals,
            vector_construction_id=str(metadata["vector_construction_id"]),
            scaling_id=str(metadata["scaling_id"]),
            channel_labels=labels,
        )
        results["AR"] = gate_b._metric_record("AR", ar_result, "source_reference")
    except Exception as exc:
        results["AR"] = gate_b._unsupported_exception("AR", exc)

    for family, compute in (
        ("NIE", lambda: gate_b._compute_nie(signals, labels)),
        ("RD", lambda: compute_rd_centered_log_psd_dispersion(
            signals, labels, sample_rate_hz=fs, welch_segment_seconds=4.0,
            frequency_min_hz=1.0, frequency_max_hz=45.0,
        )),
        ("OI", lambda: compute_oi_matrix(signals, labels)),
        ("SC", lambda: compute_sc_lz78_exact_multiset_ratio(
            signals, labels,
            source_identity=f"ds003478|{run.participant_id}|{run.run_id}|{source_fdt_sha256}",
            segment_id=window.window_id,
            start_seconds=window.start_sample / fs,
            end_seconds=window.end_sample_exclusive / fs,
            reference_count=32,
            master_seed=310930,
        )),
        ("ED", lambda: compute_ed_raw_sensor_covariance_participation_ratio(signals, labels)),
    ):
        try:
            result = compute()
            results[family] = gate_b._metric_record(family, result, "source_reference")
        except Exception as exc:
            results[family] = gate_b._unsupported_exception(family, exc)
    return results


def compute_window_rows(
    run: gate_b.RunMetadata,
    scheduled: ScheduledWindow,
    data: np.memmap,
    source_fdt_sha256: str,
    acquisition_reference: str,
) -> list[dict[str, Any]]:
    signals, labels = gate_b._window_data(data, run, scheduled.window)
    if labels != run.eeg_channel_labels:
        raise ValueError("frozen_EEG_label_order_changed_since_selection")
    reference_id = SOURCE_REFERENCE_ID
    reference_hash, source_reference_hash, reference_set_hash = gate_b._reference_provenance(
        labels, reference_id, acquisition_reference
    )
    signal_hash = gate_b._window_hash(signals, labels, run.sample_rate_hz, reference_id)
    base = gate_b._base_record(
        run,
        scheduled.window,
        source_fdt_sha256=source_fdt_sha256,
        source_ref=reference_id,
        acquisition_reference=acquisition_reference,
        reference_realization_hash=reference_hash,
        source_reference_metadata_hash=source_reference_hash,
        reference_set_hash=reference_set_hash,
        eeg_labels=labels,
        window_sha256=signal_hash,
    )
    rows: list[dict[str, Any]] = []
    results = _family_kernel_results(
        signals, labels, fs=run.sample_rate_hz, run=run,
        window=scheduled.window, source_fdt_sha256=source_fdt_sha256,
    )
    ar_metadata = ar_realization_metadata(scheduled.duration_seconds)
    for family_id in FAMILY_IDS:
        row = {
            **base,
            **results[family_id],
            "protocol_id": PROTOCOL_ID,
            "duration_seconds": scheduled.duration_seconds,
            "duration_realization_id": duration_realization_id(family_id, scheduled.duration_seconds),
            "window_grid_roles": list(scheduled.grid_roles),
            "matched_anchor_window_id": scheduled.matched_anchor_id,
            "matched_anchor_start_sample": scheduled.matched_anchor_start_sample,
            "matched_anchor_end_sample_exclusive": scheduled.matched_anchor_end_sample_exclusive,
            "clinical_metadata_read": False,
        }
        if family_id == "AR":
            row["duration_parameters"] = ar_metadata
        rows.append(row)
    return rows


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool) and math.isfinite(float(value))


def _row_value_payload(row: Mapping[str, Any]) -> Any:
    if row.get("family_id") == "AR":
        return row.get("component_values")
    return row.get("value")


def _values_match(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_values_match(a, b) for a, b in zip(left, right))
    if _is_finite_number(left) and _is_finite_number(right):
        return math.isclose(float(left), float(right), rel_tol=MATCH_RTOL, abs_tol=MATCH_ATOL)
    return left == right


def _parity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("participant_id"), row.get("run_id"), row.get("condition"),
        row.get("state_block_id"), row.get("support_start_sample"),
        row.get("support_end_sample_exclusive"), row.get("family_id"), row.get("reference_id"),
    )


def audit_c1_parity(rows: Sequence[Mapping[str, Any]], c1_run_dir: Path, participant_ids: Sequence[str]) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    wanted = set(participant_ids)
    expected_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for participant_id in participant_ids:
        shard = c1_run_dir / "participants" / f"{participant_id}.jsonl.gz"
        if not shard.is_file():
            raise FileNotFoundError(f"c1_participant_shard_missing:{participant_id}")
        with gzip.open(shard, "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if (
                    row.get("participant_id") in wanted
                    and row.get("reference_id") == SOURCE_REFERENCE_ID
                    and row.get("family_id") in FAMILY_IDS
                ):
                    expected_by_key[_parity_key(row)] = row
    actual_by_key = {
        _parity_key(row): row for row in rows
        if row.get("duration_seconds") == 20
        and row.get("reference_id") == SOURCE_REFERENCE_ID
        and "cadence" in row.get("window_grid_roles", [])
        and row.get("family_id") in FAMILY_IDS
    }
    all_keys = sorted(set(expected_by_key) | set(actual_by_key), key=lambda key: tuple(str(value) for value in key))
    for key in all_keys:
        expected = expected_by_key.get(key)
        actual = actual_by_key.get(key)
        checks: dict[str, bool] = {
            "key_present_both": expected is not None and actual is not None,
        }
        if expected is not None and actual is not None:
            checks.update({
                "status_match": expected.get("computed") == actual.get("computed")
                    and expected.get("valid") == actual.get("valid")
                    and expected.get("status") == actual.get("status")
                    and expected.get("reason") == actual.get("reason"),
                "estimator_match": expected.get("estimator_id") == actual.get("estimator_id"),
                "source_window_match": expected.get("window_signal_sha256") == actual.get("window_signal_sha256")
                    and expected.get("source_fdt_sha256") == actual.get("source_fdt_sha256"),
                "labels_match": expected.get("channel_order_sha256") == actual.get("channel_order_sha256")
                    and expected.get("channel_set_sha256") == actual.get("channel_set_sha256"),
                "value_match": _values_match(_row_value_payload(expected), _row_value_payload(actual)),
            })
        audit.append({
            "participant_id": key[0], "run_id": key[1], "condition": key[2],
            "state_block_id": key[3], "start_sample": key[4], "end_sample_exclusive": key[5],
            "family_id": key[6], "reference_id": key[7], **checks,
            "parity_pass": all(checks.values()),
        })
    return audit


def _support_is_valid(row: Mapping[str, Any]) -> bool:
    if row.get("computed") is not True or row.get("valid") is not True:
        return False
    if row.get("family_id") == "AR":
        values = row.get("component_values")
        statuses = row.get("component_status")
        names = row.get("component_names")
        return (
            isinstance(values, list) and isinstance(names, list) and len(values) == len(names)
            and all(_is_finite_number(value) for value in values)
            and (statuses is None or (
                isinstance(statuses, list) and len(statuses) == len(values)
                and all(status == "computed_exploratory" for status in statuses)
            ))
        )
    return _is_finite_number(row.get("value"))


def _ar_component_value(row: Mapping[str, Any], index: int) -> float | None:
    values = row.get("component_values")
    statuses = row.get("component_status")
    if not isinstance(values, list) or index >= len(values):
        return None
    if isinstance(statuses, list) and (index >= len(statuses) or statuses[index] != "computed_exploratory"):
        return None
    value = values[index]
    return float(value) if _is_finite_number(value) else None


def build_support_and_dynamics(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    support_groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    cadence_windows: dict[tuple[Any, ...], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        key = (
            row.get("participant_id"), row.get("run_id"), row.get("condition"),
            row.get("state_block_id"), row.get("duration_seconds"),
        )
        support_groups.setdefault((*key, row.get("family_id")), []).append(row)
        if "cadence" in row.get("window_grid_roles", []):
            wkey = (*key, row.get("support_start_sample"), row.get("support_end_sample_exclusive"))
            cadence_windows.setdefault(wkey, {})[str(row.get("family_id"))] = row

    support_rows: list[dict[str, Any]] = []
    for key, group in sorted(support_groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        cadence = [row for row in group if "cadence" in row.get("window_grid_roles", [])]
        matched = [row for row in group if "matched_20s_anchor_prefix" in row.get("window_grid_roles", [])]
        reasons: dict[str, int] = {}
        component_valid: dict[str, int] = {}
        for row in cadence:
            reason = row.get("reason")
            if reason:
                reasons[str(reason)] = reasons.get(str(reason), 0) + 1
            if row.get("family_id") == "AR":
                for index in range(len(row.get("component_names") or [])):
                    if _ar_component_value(row, index) is not None:
                        key_name = str(row["component_names"][index])
                        component_valid[key_name] = component_valid.get(key_name, 0) + 1
        support_rows.append({
            "participant_id": key[0], "run_id": key[1], "condition": key[2],
            "state_block_id": key[3], "duration_seconds": key[4], "family_id": key[5],
            "cadence_windows_requested": len(cadence),
            "cadence_computed": sum(row.get("computed") is True for row in cadence),
            "cadence_valid_finite": sum(_support_is_valid(row) for row in cadence),
            "cadence_unsupported": sum(row.get("status") == "unsupported" for row in cadence),
            "matched_anchor_prefix_count": len(matched),
            "matched_valid_finite": sum(_support_is_valid(row) for row in matched),
            "component_valid_counts_json": json.dumps(component_valid, sort_keys=True),
            "reason_counts_json": json.dumps(reasons, sort_keys=True),
        })

    by_block: dict[tuple[Any, ...], list[tuple[tuple[Any, ...], dict[str, Mapping[str, Any]]]]] = {}
    for key, family_rows in cadence_windows.items():
        group_key = key[:5]
        by_block.setdefault(group_key, []).append((key, family_rows))
    dynamics_rows: list[dict[str, Any]] = []
    for group_key, entries in sorted(by_block.items(), key=lambda item: tuple(str(value) for value in item[0])):
        participant, run_id, condition, block_id, duration = group_key
        dimension = 7 + duration // AR_BLOCK_SECONDS - 1
        minimum = dimension + 1
        complete: list[tuple[tuple[Any, ...], dict[str, Mapping[str, Any]]]] = []
        for key, family_rows in sorted(entries, key=lambda item: int(item[0][5])):
            all_complete = all(
                family in family_rows and _support_is_valid(family_rows[family])
                for family in FAMILY_IDS
            )
            if all_complete and len(family_rows["AR"].get("component_values", [])) == duration // AR_BLOCK_SECONDS - 1:
                complete.append((key, family_rows))
        max_run = max_transitions = current_run = 0
        previous_end: int | None = None
        for key, _ in complete:
            start = int(key[5])
            end = int(key[6])
            if previous_end == start:
                current_run += 1
            else:
                current_run = 1
            max_run = max(max_run, current_run)
            max_transitions = max(max_transitions, current_run - 1)
            previous_end = end
        dynamics_rows.append({
            "participant_id": participant, "run_id": run_id, "condition": condition,
            "state_block_id": block_id, "duration_seconds": duration,
            "state_dimension_if_complete": dimension,
            "ar_component_count": duration // AR_BLOCK_SECONDS - 1,
            "complete_vector_windows": len(complete),
            "max_exactly_adjacent_complete_run_windows": max_run,
            "max_exact_sample_adjacent_transitions": max_transitions,
            "necessary_affine_full_rank_minimum_transitions": minimum,
            "meets_necessary_rank_count_only": max_transitions >= minimum,
            "physical_gap_and_bad_interval_status": "unmarked_source_gaps_and_bad_intervals_not_assessed",
            "jacobian_estimated": False,
        })
    return support_rows, dynamics_rows


def _scalar_value(row: Mapping[str, Any]) -> float | None:
    value = row.get("value")
    return float(value) if _support_is_valid(row) and _is_finite_number(value) else None


def _component_values(row: Mapping[str, Any]) -> list[float] | None:
    if not _support_is_valid(row) or row.get("family_id") != "AR":
        return None
    return [float(value) for value in row.get("component_values", [])]


def build_matched_duration_comparisons(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    anchors: dict[tuple[Any, ...], dict[int, Mapping[str, Any]]] = {}
    for row in rows:
        if "matched_20s_anchor_prefix" not in row.get("window_grid_roles", []):
            continue
        key = (
            row.get("participant_id"), row.get("run_id"), row.get("condition"),
            row.get("state_block_id"), row.get("matched_anchor_window_id"),
            row.get("matched_anchor_start_sample"), row.get("family_id"),
        )
        anchors.setdefault(key, {})[int(row["duration_seconds"])] = row
    output: list[dict[str, Any]] = []
    for key, by_duration in sorted(anchors.items(), key=lambda item: tuple(str(value) for value in item[0])):
        family = str(key[6])
        ref = by_duration.get(20)
        ref_value = _scalar_value(ref) if ref is not None and family != "AR" else None
        ref_components = _component_values(ref) if ref is not None and family == "AR" else None
        for duration in DURATIONS_SECONDS:
            row = by_duration.get(duration)
            value = _scalar_value(row) if row is not None and family != "AR" else None
            components = _component_values(row) if row is not None and family == "AR" else None
            delta = value - ref_value if value is not None and ref_value is not None else None
            output.append({
                "participant_id": key[0], "run_id": key[1], "condition": key[2],
                "state_block_id": key[3], "matched_anchor_window_id": key[4],
                "matched_anchor_start_sample": key[5], "family_id": family,
                "duration_seconds": duration,
                "candidate_status": row.get("status") if row is not None else "missing_anchor_candidate",
                "candidate_valid": _support_is_valid(row) if row is not None else False,
                "candidate_value": value,
                "candidate_component_count": len(components) if components is not None else None,
                "reference_20s_value": ref_value,
                "reference_20s_component_count": len(ref_components) if ref_components is not None else None,
                "delta_from_20s": delta,
                "cross_duration_AR_comparison": "not_compared_duration_specific_shape" if family == "AR" else None,
            })
    return output


def build_temporal_and_condition_descriptives(rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cadence = [row for row in rows if "cadence" in row.get("window_grid_roles", [])]
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in cadence:
        key = (row.get("participant_id"), row.get("run_id"), row.get("condition"), row.get("duration_seconds"), row.get("family_id"))
        grouped.setdefault(key, []).append(row)

    temporal_rows: list[dict[str, Any]] = []
    participant_condition_values: dict[tuple[Any, ...], dict[str, Any]] = {}
    for key, group in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        family = str(key[4])
        if family == "AR":
            component_count = int(key[3]) // AR_BLOCK_SECONDS - 1
            component_indices = range(component_count)
            for index in component_indices:
                valid_rows = [row for row in group if _ar_component_value(row, index) is not None]
                values = [float(_ar_component_value(row, index)) for row in valid_rows]
                changes: list[float] = []
                ordered = sorted(valid_rows, key=lambda row: (row["state_block_id"], row["support_start_sample"]))
                for left, right in zip(ordered, ordered[1:]):
                    if left["state_block_id"] == right["state_block_id"] and left["support_end_sample_exclusive"] == right["support_start_sample"]:
                        changes.append(abs(float(_ar_component_value(right, index)) - float(_ar_component_value(left, index))))
                component_name = next((row["component_names"][index] for row in group if len(row.get("component_names", [])) > index), f"component_{index:03d}")
                temporal_rows.append(_descriptive_row(key, component_name, len(group), values, changes, index))
                vals = np.asarray(values, dtype=np.float64)
                participant_condition_values[(*key, index)] = {
                    "median": float(np.median(vals)) if vals.size else None,
                    "valid_windows": len(values),
                    "component_name": component_name,
                }
            continue
        valid_rows = [row for row in group if _scalar_value(row) is not None]
        values = [float(row["value"]) for row in valid_rows]
        changes = []
        ordered = sorted(valid_rows, key=lambda row: (row["state_block_id"], row["support_start_sample"]))
        for left, right in zip(ordered, ordered[1:]):
            if left["state_block_id"] == right["state_block_id"] and left["support_end_sample_exclusive"] == right["support_start_sample"]:
                changes.append(abs(float(right["value"]) - float(left["value"])))
        temporal_rows.append(_descriptive_row(key, None, len(group), values, changes, None))
        vals = np.asarray(values, dtype=np.float64)
        participant_condition_values[key] = {
            "median": float(np.median(vals)) if vals.size else None,
            "valid_windows": len(values),
        }

    contrast_rows: list[dict[str, Any]] = []
    keys = sorted({(k[0], k[1], k[3], k[4], *k[5:]) for k in participant_condition_values}, key=lambda key: tuple(str(value) for value in key))
    for contrast_key in keys:
        participant, run_id, duration, family = contrast_key[:4]
        component_index = contrast_key[4] if len(contrast_key) > 4 else None
        eo_key = (participant, run_id, "EO", duration, family, *contrast_key[4:])
        ec_key = (participant, run_id, "EC", duration, family, *contrast_key[4:])
        eo = participant_condition_values.get(eo_key)
        ec = participant_condition_values.get(ec_key)
        component_name = (eo or ec or {}).get("component_name")
        contrast_rows.append({
            "participant_id": participant, "run_id": run_id, "duration_seconds": duration,
            "family_id": family, "component_index": component_index,
            "component_name": component_name,
            "eo_median": eo["median"] if eo else None,
            "ec_median": ec["median"] if ec else None,
            "ec_minus_eo": (ec["median"] - eo["median"])
                if eo and ec and eo["median"] is not None and ec["median"] is not None else None,
            "eo_valid_windows": eo["valid_windows"] if eo else 0,
            "ec_valid_windows": ec["valid_windows"] if ec else 0,
            "inference": "descriptive_only_no_p_values",
        })
    return temporal_rows, contrast_rows


def _descriptive_row(key: tuple[Any, ...], component_name: str | None, requested: int,
                     values: Sequence[float], changes: Sequence[float], component_index: int | None) -> dict[str, Any]:
    data = np.asarray(values, dtype=np.float64)
    change_data = np.asarray(changes, dtype=np.float64)
    q25, q75 = np.quantile(data, [0.25, 0.75]) if data.size else (None, None)
    return {
        "participant_id": key[0], "run_id": key[1], "condition": key[2],
        "duration_seconds": key[3], "family_id": key[4],
        "component_index": component_index, "component_name": component_name,
        "cadence_windows_requested": requested, "valid_values": int(data.size),
        "median": float(np.median(data)) if data.size else None,
        "q25": float(q25) if data.size else None,
        "q75": float(q75) if data.size else None,
        "median_absolute_adjacent_change": float(np.median(change_data)) if change_data.size else None,
        "adjacent_valid_pairs": int(change_data.size),
        "interpretation": "within_recording_temporal_dispersion_not_reliability",
    }


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


def _write_jsonl_gzip(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n", compresslevel=6) as stream:
        for row in rows:
            stream.write(json.dumps(gate_b._jsonable(row), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _load_c1_manifest(c1_run_dir: Path) -> dict[str, Any]:
    manifest_path = c1_run_dir / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != C1_MANIFEST_SHA256:
        raise ValueError("frozen_c1_manifest_missing_or_hash_changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("gate_id") != "C1" or manifest.get("qualified") is not False:
        raise ValueError("c1_run_is_not_the_expected_unqualified_gate")
    if manifest.get("clinical_metadata_read") is not False or manifest.get("outcome_blind") is not True:
        raise ValueError("c1_source_run_is_not_outcome_blind")
    if manifest.get("selection_manifest_sha256") != SELECTION_SHA256:
        raise ValueError("c1_selection_hash_does_not_match_frozen_selection")
    return manifest


def _validate_selection(selection: Mapping[str, Any], selection_path: Path) -> list[Mapping[str, Any]]:
    if sha256_file(selection_path) != SELECTION_SHA256:
        raise ValueError("frozen_technical_selection_hash_changed")
    if selection.get("gate_id") != "C1" or int(selection.get("eligible_count", -1)) != 103:
        raise ValueError("frozen_gate_c1_roster_mismatch")
    roster = selection.get("eligible_roster")
    if not isinstance(roster, list) or len(roster) != 103:
        raise ValueError("frozen_gate_c1_roster_missing")
    selected = choose_technical_roster(roster)
    expected = ("sub-079", "sub-047", "sub-026", "sub-094")
    if tuple(str(row["participant_id"]) for row in selected) != expected:
        raise ValueError("deterministic_technical_selection_changed")
    return selected


def _participant_shard_hash(c1_manifest: Mapping[str, Any], participant_id: str, shard: Path) -> str:
    expected = None
    for item in c1_manifest.get("gzip_member_shards", []):
        if item.get("participant_id") == participant_id:
            expected = item.get("sha256")
            break
    actual = sha256_file(shard)
    if expected is not None and actual != expected:
        raise ValueError(f"c1_participant_shard_hash_mismatch:{participant_id}")
    return actual


def _family_window_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {family: 0 for family in FAMILY_IDS}
    for row in rows:
        if row.get("family_id") in counts and "cadence" in row.get("window_grid_roles", []):
            counts[str(row["family_id"])] += 1
    return counts


def _render_report(
    selection_ids: Sequence[str], participant_status: Sequence[Mapping[str, Any]],
    parity_rows: Sequence[Mapping[str, Any]], support_rows: Sequence[Mapping[str, Any]],
    dynamics_rows: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]],
) -> str:
    parity_pass = bool(parity_rows) and all(row.get("parity_pass") for row in parity_rows)
    processed_ids = [str(row["participant_id"]) for row in participant_status]
    smoke_run = len(processed_ids) < len(selection_ids)
    run_status = (
        "SMOKE_PARITY_PASS" if parity_pass and smoke_run else
        "SMOKE_PARITY_FAILED" if smoke_run else
        "COMPLETED_BOUNDED_TECHNICAL_PILOT" if parity_pass else
        "C1_20S_PARITY_FAILED"
    )
    lines = [
        "# WINDOW-SUPPORT-001 — bounded multiscale support benchmark", "",
        f"**Status:** {run_status}  ",
        "**Qualification:** false  ",
        "**Clinical metadata loaded by this run:** no  ",
        f"**Frozen participant selection:** {', '.join(selection_ids)}  ",
        f"**Recordings processed in this run:** {', '.join(processed_ids)}  ",
        "**C1 20-second parity:** pass" if parity_pass else "**C1 20-second parity:** fail",
        "",
        f"This run processed {len(processed_ids)} of {len(selection_ids)} recordings from the frozen Gate C1 technical roster. The source reference and all eight existing families are retained. No clinical fields or p-values were used. Before protocol freeze, rows for sub-001 through sub-004 were inadvertently displayed during metadata preflight; none is in the selected set. The analyst has also seen previous ds003478 clinical analyses, so this is outcome-excluded execution, not analyst-blinded work.",
        "",
        "## Coverage by duration and family", "",
        f"Cadence rows: {json.dumps(_family_window_counts(rows), sort_keys=True)}.",
        "Unsupported and invalid rows remain explicit in `window_measurements.jsonl.gz` and `window_support.csv`; no value was imputed or replaced by zero.",
        "",
        "## Matched duration and temporal variation", "",
        "`matched_duration_comparisons.csv` compares 4/8/12-second prefixes with the 20-second value at the same 20-second anchor start. Prefixes have unequal sample support by design. Per-window and within-block summaries are descriptive. Neighbor-window changes mix estimator variation with real temporal variation and are not reliability or test–retest estimates.",
        "",
        "## Condition and trajectory support", "",
        "`condition_descriptives.csv` contains per-participant EC-minus-EO descriptions only; there are no p-values or cohort-level inferential claims.",
        "`trajectory_support.csv` reports exact sample-adjacent complete-window runs and the necessary affine rank count only. Physical gaps not represented by explicit event boundaries and bad intervals remain unassessed; no Jacobian was estimated.",
        "",
        "## Limits", "",
        "The C1 20-second realization is a parity reference, not a gold standard. Duration-specific estimator behavior can differ, and AR output shapes are 1/3/5/9 components at 4/8/12/20 seconds. These are separate realizations; no cross-duration AR scalarization or canonical chart was created. Four recordings cannot establish reliability, a preferred universal duration, cross-dataset equivalence, clinical usefulness, or qualification.",
        "",
        "See `manifest.json` for frozen protocol, source, code, environment and output hashes. See `participant_status.json` for source verification and elapsed time.",
        "",
    ]
    if not parity_pass:
        failed = sum(not row.get("parity_pass") for row in parity_rows)
        lines.insert(3, f"**Parity failures:** {failed}")
    return "\n".join(lines)


def run(
    *,
    selection_path: Path = DEFAULT_SELECTION_PATH,
    c1_run_dir: Path = C1_RUN_DEFAULT,
    output_dir: Path,
    protocol_path: Path = DEFAULT_PROTOCOL_PATH,
    participant_limit: int | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing_to_overwrite_nonempty_output_dir:{output_dir}")
    if participant_limit is not None and (participant_limit < 1 or participant_limit > SELECTED_PARTICIPANT_COUNT):
        raise ValueError("participant_limit_must_be_between_1_and_frozen_selection_size")
    if not protocol_path.is_file():
        raise FileNotFoundError("frozen_window_support_protocol_missing")
    protocol_hash = sha256_file(protocol_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = _validate_selection(selection, selection_path)
    c1_manifest = _load_c1_manifest(c1_run_dir)
    selected = selected[:participant_limit] if participant_limit is not None else selected
    participant_ids = [str(record["participant_id"]) for record in selected]
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol_frozen.md").write_bytes(protocol_path.read_bytes())
    selection_record = {
        "protocol_id": PROTOCOL_ID,
        "selection_rule": "ascending_sha256_utf8_WINDOW-SUPPORT-001|participant_id_from_frozen_C1_roster",
        "selection_manifest_sha256": SELECTION_SHA256,
        "frozen_selected_participant_ids": [str(record["participant_id"]) for record in choose_technical_roster(selection["eligible_roster"])],
        "participants_processed": participant_ids,
        "technical_only_selection": True,
        "clinical_metadata_read": False,
        "clinical_metadata_context": (
            "No clinical metadata loaded by this run; prior analyst exposure and the pre-freeze preview of sub-001 through sub-004 are disclosed in the frozen protocol."
        ),
        "records": [dict(record) for record in selected],
    }
    (output_dir / "selection_manifest.json").write_text(json.dumps(selection_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []
    participant_status: list[dict[str, Any]] = []
    source_hashes: dict[str, dict[str, str]] = {}
    c1_shard_hashes: dict[str, str] = {}
    roster_lookup = {str(record["participant_id"]): record for record in selected}
    for participant_id in participant_ids:
        record = roster_lookup[participant_id]
        run_metadata = gate_c1._metadata_record_to_run(record)
        gate_b._verify_frozen_inputs(run_metadata)
        eeg_metadata = gate_b._read_json(Path(run_metadata.eeg_json_path))
        acquisition_reference = str(eeg_metadata.get("EEGReference", "unknown"))
        events = gate_b._events_for_run(Path(run_metadata.events_path))
        blocks = reconstruct_condition_blocks(
            events, sample_rate_hz=run_metadata.sample_rate_hz,
            recording_sample_count=run_metadata.sample_count,
        )
        boundaries = _boundary_sample_indices(events, run_metadata.sample_rate_hz)
        blocks = split_blocks_at_boundaries(blocks, boundaries)
        all_scheduled = [
            scheduled
            for duration in DURATIONS_SECONDS
            for scheduled in schedule_duration_windows(
                blocks, sample_rate_hz=run_metadata.sample_rate_hz,
                duration_seconds=duration,
            )
        ]
        data, source_fdt_sha256 = gate_b._source_memmap(run_metadata)
        try:
            participant_rows: list[dict[str, Any]] = []
            for scheduled in all_scheduled:
                participant_rows.extend(compute_window_rows(
                    run_metadata, scheduled, data, source_fdt_sha256, acquisition_reference
                ))
            rows.extend(participant_rows)
            shard_path = c1_run_dir / "participants" / f"{participant_id}.jsonl.gz"
            c1_shard_hashes[participant_id] = _participant_shard_hash(c1_manifest, participant_id, shard_path)
            source_hashes[participant_id] = {
                "set": run_metadata.set_sha256,
                "fdt": source_fdt_sha256,
                "events": run_metadata.events_sha256,
                "channels": run_metadata.channels_sha256,
                "eeg_json": run_metadata.eeg_json_sha256,
            }
            participant_status.append({
                "participant_id": participant_id, "run_id": run_metadata.run_id,
                "status": "processed", "condition_block_count": len(blocks),
                "window_rows": sum(len(FAMILY_IDS) for _ in all_scheduled),
                "scheduled_windows_by_duration": {
                    str(duration): sum(item.duration_seconds == duration for item in all_scheduled)
                    for duration in DURATIONS_SECONDS
                },
                "source_fdt_sha256": source_fdt_sha256,
            })
        finally:
            del data
    elapsed_seconds = time.perf_counter() - start_time

    parity_rows = audit_c1_parity(rows, c1_run_dir, participant_ids)
    support_rows, dynamics_rows = build_support_and_dynamics(rows)
    matched_rows = build_matched_duration_comparisons(rows)
    temporal_rows, condition_rows = build_temporal_and_condition_descriptives(rows)
    parity_pass = bool(parity_rows) and all(row.get("parity_pass") for row in parity_rows)
    smoke_run = len(participant_ids) < SELECTED_PARTICIPANT_COUNT
    run_status = (
        "SMOKE_PARITY_PASS" if parity_pass and smoke_run else
        "SMOKE_PARITY_FAILED" if smoke_run else
        "COMPLETED_BOUNDED_TECHNICAL_PILOT" if parity_pass else
        "C1_20S_PARITY_FAILED"
    )

    _write_jsonl_gzip(output_dir / "window_measurements.jsonl.gz", rows)
    _write_csv(output_dir / "window_support.csv", support_rows)
    _write_csv(output_dir / "trajectory_support.csv", dynamics_rows)
    _write_csv(output_dir / "matched_duration_comparisons.csv", matched_rows)
    _write_csv(output_dir / "temporal_descriptives.csv", temporal_rows)
    _write_csv(output_dir / "condition_descriptives.csv", condition_rows)
    _write_csv(output_dir / "c1_20s_parity.csv", parity_rows)
    (output_dir / "participant_status.json").write_text(json.dumps({
        "elapsed_seconds": elapsed_seconds,
        "participant_status": participant_status,
        "clinical_metadata_read": False,
        "source_hashes": source_hashes,
        "c1_participant_shard_hashes": c1_shard_hashes,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = _render_report(
        [str(record["participant_id"]) for record in choose_technical_roster(selection["eligible_roster"])],
        participant_status, parity_rows, support_rows, dynamics_rows, rows,
    )
    (output_dir / "WINDOW-SUPPORT-001_REPORT.md").write_text(report, encoding="utf-8")

    repo_root = REPO_ROOT
    source_paths = {
        "runner": Path(__file__).resolve(),
        "gate_b_runner": Path(gate_b.__file__).resolve(),
        "gate_c1_runner": Path(gate_c1.__file__).resolve(),
        "candidate_grid": repo_root / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "candidate_grid.py",
        "validation_plan": repo_root / "project" / "new_mnps_v.3.2" / "VALIDATION_PLAN.md",
        "registry": repo_root / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "registry.py",
        "pcv": repo_root / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "pcv.py",
        "tp": repo_root / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "tp.py",
        "ar": repo_root / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "ar.py",
        "nie": repo_root / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "nie.py",
        "rd": repo_root / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "rd.py",
        "oi": repo_root / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "oi.py",
        "sc": repo_root / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "sc.py",
        "ed": repo_root / "mndm" / "src" / "mndm" / "mnps" / "v3_2" / "ed.py",
    }
    output_names = [
        "protocol_frozen.md", "selection_manifest.json", "window_measurements.jsonl.gz",
        "window_support.csv", "trajectory_support.csv", "matched_duration_comparisons.csv",
        "temporal_descriptives.csv", "condition_descriptives.csv", "c1_20s_parity.csv",
        "participant_status.json", "WINDOW-SUPPORT-001_REPORT.md",
    ]
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "protocol_version": PROTOCOL_VERSION,
        "status": run_status,
        "run_mode": "smoke" if smoke_run else "bounded_four_recording_pilot",
        "qualified": False,
        "clinical_metadata_read": False,
        "analyst_blinded": False,
        "technical_selection_only": True,
        "selected_participant_ids_frozen": selection_record["frozen_selected_participant_ids"],
        "participants_processed": participant_ids,
        "duration_grid_seconds": list(DURATIONS_SECONDS),
        "cadence": "nonoverlapping_step_equals_duration_inside_condition_and_boundary_blocks",
        "matched_duration_design": "nested_prefixes_at_common_20s_anchor_start",
        "ar_components_by_duration": {
            str(duration): duration // AR_BLOCK_SECONDS - 1 for duration in DURATIONS_SECONDS
        },
        "selection_manifest_sha256": SELECTION_SHA256,
        "c1_manifest_sha256": C1_MANIFEST_SHA256,
        "analysis_protocol_sha256": protocol_hash,
        "validation_plan_sha256": sha256_file(source_paths["validation_plan"]),
        "analysis_runner_sha256": sha256_file(source_paths["runner"]),
        "implementation_code_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
        "source_hashes": source_hashes,
        "c1_participant_shard_hashes": c1_shard_hashes,
        "c1_20s_parity_pass": parity_pass,
        "c1_20s_parity_row_count": len(parity_rows),
        "all_window_rows": len(rows),
        "elapsed_seconds": elapsed_seconds,
        "software": {
            "python": sys.version.split()[0], "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
        },
        "registry_family_ids": [item["family_id"] for item in get_full_vector_registry() if item["family_id"] != "SCG"],
        "output_hashes": {name: sha256_file(output_dir / name) for name in output_names},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-path", type=Path, default=DEFAULT_SELECTION_PATH)
    parser.add_argument("--c1-run-dir", type=Path, default=C1_RUN_DEFAULT)
    parser.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--participant-limit", type=int)
    args = parser.parse_args(argv)
    manifest = run(
        selection_path=args.selection_path,
        c1_run_dir=args.c1_run_dir,
        output_dir=args.output_dir,
        protocol_path=args.protocol_path,
        participant_limit=args.participant_limit,
    )
    print(json.dumps({
        "protocol_id": PROTOCOL_ID,
        "status": manifest["status"],
        "output_dir": str(args.output_dir.resolve()),
        "participants_processed": manifest["participants_processed"],
        "window_rows": manifest["all_window_rows"],
        "c1_20s_parity_pass": manifest["c1_20s_parity_pass"],
        "qualified": False,
    }, indent=2))
    return 0 if manifest["c1_20s_parity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
