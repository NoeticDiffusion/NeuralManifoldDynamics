"""Fold-scoped manifest validation and feature-anchor fitting.

This module deliberately implements only the replayable cohort-anchor step.
Transforms without a saved fit/apply implementation (currently the summarize
ComBat path) must be refused by the caller in fold-safe mode.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "mndm.fold_split.v1"
_SUBJECT_TOKEN_RE = re.compile(r"(sub-[A-Za-z0-9]+)")


class FoldSplitError(ValueError):
    """Raised when the supplied split cannot define disjoint fit/apply rows."""


@dataclass(frozen=True)
class FoldSplit:
    """Validated split manifest and its byte-level identity."""

    payload: Mapping[str, Any]
    sha256: str
    path: Path


@dataclass(frozen=True)
class FoldSelection:
    """One selected manifest partition, retaining source DataFrame indices."""

    frame: pd.DataFrame
    role: str
    row_records: tuple[Mapping[str, Any], ...]
    manifest_sha256: str

    @property
    def row_ids(self) -> list[str]:
        return [str(row["row_id"]) for row in self.row_records]


@dataclass(frozen=True)
class FoldSignalSupportLedger:
    """Validated finite-support ledger bound to its on-disk bytes."""

    payload: Mapping[str, Any]
    sha256: str
    path: Path


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def transform_config_sha256(value: Mapping[str, Any]) -> str:
    """Hash the normalized transform settings that must match during apply."""
    return _sha256_text(_canonical_json(value))


def implementation_sha256(paths: Sequence[Path | str]) -> str:
    """Hash the source files that implement the saved chart transform."""
    digest = hashlib.sha256()
    normalized = sorted((Path(path).resolve() for path in paths), key=lambda item: item.name)
    for path in normalized:
        if not path.is_file():
            raise FoldSplitError(f"transform implementation source is missing: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()


def verify_source_file_hashes(
    expected_hashes: Mapping[str, str],
    source_paths: Mapping[str, Path | str],
) -> dict[str, str]:
    """Hash selected raw inputs and require equality with the split manifest."""
    actual_hashes: dict[str, str] = {}
    for raw_name, raw_expected in expected_hashes.items():
        name = _normalize_recording(raw_name)
        path = source_paths.get(name)
        if path is None:
            raise FoldSplitError(f"cannot resolve raw source file for {name}")
        source_path = Path(path)
        digest = hashlib.sha256()
        try:
            if not source_path.is_file():
                raise FoldSplitError(f"raw source file is missing for {name}: {source_path}")
            with source_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
        except OSError as exc:
            raise FoldSplitError(f"could not read raw source file for {name}: {source_path}") from exc
        actual = digest.hexdigest()
        expected = str(raw_expected).strip().lower()
        if actual != expected:
            raise FoldSplitError(f"raw source SHA256 mismatch for {name}")
        actual_hashes[name] = actual
    return actual_hashes


def _normalize_subject(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.startswith("sub-"):
        return raw
    return f"sub-{raw.zfill(3)}" if raw.isdigit() else f"sub-{raw}"


def _normalize_recording(value: Any) -> str:
    return Path(str(value or "").replace("\\", "/")).name


def _validated_ranges(payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    raw_ranges = payload.get(key)
    if not isinstance(raw_ranges, list) or not raw_ranges:
        raise FoldSplitError(f"{key} must be a non-empty list of chronological ranges")
    out: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_ranges):
        if not isinstance(raw, Mapping):
            raise FoldSplitError(f"{key}[{index}] must be an object")
        subject = _normalize_subject(raw.get("subject_id"))
        recording = _normalize_recording(raw.get("recording_id"))
        try:
            start = float(raw["start_sec"])
            end = float(raw["end_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FoldSplitError(f"{key}[{index}] must contain numeric start_sec and end_sec") from exc
        if not subject or not recording:
            raise FoldSplitError(f"{key}[{index}] must identify subject_id and recording_id")
        if not (np.isfinite(start) and np.isfinite(end) and end > start):
            raise FoldSplitError(f"{key}[{index}] must have finite end_sec > start_sec")
        out.append({"subject_id": subject, "recording_id": recording, "start_sec": start, "end_sec": end})
    return out


def _check_ranges_disjoint(training: Sequence[Mapping[str, Any]], apply: Sequence[Mapping[str, Any]]) -> None:
    same_partition = training is apply
    for left_index, left in enumerate(training):
        for right_index, right in enumerate(apply):
            if same_partition and right_index <= left_index:
                continue
            if left["subject_id"] != right["subject_id"] or left["recording_id"] != right["recording_id"]:
                continue
            if max(float(left["start_sec"]), float(right["start_sec"])) < min(float(left["end_sec"]), float(right["end_sec"])):
                raise FoldSplitError(
                    "training/apply ranges overlap for "
                    f"{left['subject_id']} / {left['recording_id']}"
                )


def load_fold_split(path: Path | str) -> FoldSplit:
    """Load and validate an explicit, verified-clock fold manifest.

    Manifest shape::

        {
          "schema_version": "mndm.fold_split.v1",
          "split_id": "pilot-a",
          "time_reference": {
            "status": "verified", "start_column": "t_start",
            "end_column": "t_end", "unit": "seconds", "origin": "recording_start"
          },
          "source_file_sha256": {"recording.edf": "<64 hex chars>"},
          "training_ranges": [{"subject_id": "sub-001", "recording_id": "recording.edf",
                               "start_sec": 0, "end_sec": 120}],
          "apply_ranges": [{...}]
        }

    Recording IDs are matched to the basename of the feature table's ``file``
    column. Ranges are half-open [start_sec, end_sec); feature rows intersecting
    but not contained by a boundary are rejected.
    """
    manifest_path = Path(path).expanduser().resolve()
    try:
        raw_bytes = manifest_path.read_bytes()
    except OSError as exc:
        raise FoldSplitError(f"Could not read fold split manifest: {manifest_path}") from exc
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FoldSplitError(f"Could not parse fold split manifest: {manifest_path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        raise FoldSplitError(f"schema_version must be {SCHEMA_VERSION}")
    if not str(payload.get("split_id", "")).strip():
        raise FoldSplitError("split_id is required")
    time_reference = payload.get("time_reference")
    if not isinstance(time_reference, Mapping) or time_reference.get("status") != "verified":
        raise FoldSplitError("time_reference.status must be verified")
    if str(time_reference.get("unit", "")).lower() not in {"s", "sec", "second", "seconds"}:
        raise FoldSplitError("time_reference.unit must be seconds")
    if not str(time_reference.get("origin", "")).strip():
        raise FoldSplitError("time_reference.origin is required")
    for key in ("start_column", "end_column"):
        if not str(time_reference.get(key, "")).strip():
            raise FoldSplitError(f"time_reference.{key} is required")

    training = _validated_ranges(payload, "training_ranges")
    apply = _validated_ranges(payload, "apply_ranges")
    _check_ranges_disjoint(training, apply)
    # Duplicate/overlapping ranges within either partition would make fit
    # counts depend on range ordering. Reject them before row selection.
    _check_ranges_disjoint(training, training)
    _check_ranges_disjoint(apply, apply)

    source_hashes = payload.get("source_file_sha256")
    if not isinstance(source_hashes, Mapping):
        raise FoldSplitError("source_file_sha256 must map recording IDs to source-file SHA256 values")
    normalized_hashes: dict[str, str] = {}
    for name, digest in source_hashes.items():
        recording = _normalize_recording(name)
        value = str(digest or "").strip().lower()
        if not recording or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise FoldSplitError(f"invalid SHA256 source identity for recording {name!r}")
        if recording in normalized_hashes and normalized_hashes[recording] != value:
            raise FoldSplitError(f"conflicting source hashes for recording {recording}")
        normalized_hashes[recording] = value
    required_recordings = {str(item["recording_id"]) for item in training + apply}
    missing_hashes = sorted(required_recordings - set(normalized_hashes))
    if missing_hashes:
        raise FoldSplitError(f"source_file_sha256 missing recording(s): {', '.join(missing_hashes)}")

    normalized_payload = dict(payload)
    normalized_payload["training_ranges"] = training
    normalized_payload["apply_ranges"] = apply
    normalized_payload["source_file_sha256"] = normalized_hashes
    return FoldSplit(
        payload=normalized_payload,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        path=manifest_path,
    )


def load_signal_support_ledger(
    path: Path | str,
    *,
    expected_sha256: str,
) -> FoldSignalSupportLedger:
    """Load the exact support ledger named by a fold manifest.

    A fold-safe chart transform must bind the support ledger by digest. The
    support checks below intentionally accept only the producer's validated
    finite-causal conservative-bound export; unknown/noncausal support cannot
    certify a train/apply split.
    """
    ledger_path = Path(path).expanduser().resolve()
    try:
        raw_bytes = ledger_path.read_bytes()
    except OSError as exc:
        raise FoldSplitError(f"Could not read signal support ledger: {ledger_path}") from exc
    actual_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    expected = str(expected_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise FoldSplitError("signal_support_ledger_sha256 must be a SHA256 digest in the split manifest")
    if actual_sha256 != expected:
        raise FoldSplitError("signal support ledger SHA256 does not match the split manifest")
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FoldSplitError(f"Could not parse signal support ledger: {ledger_path}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema") != "mndm.signal_support_export.v1":
        raise FoldSplitError("signal support ledger schema must be mndm.signal_support_export.v1")
    if payload.get("temporal_support_status") != "finite_causal_conservative_bound":
        raise FoldSplitError("signal support ledger does not certify finite causal conservative bounds")
    if not isinstance(payload.get("per_epoch_input_extent"), list):
        raise FoldSplitError("signal support ledger has no per_epoch_input_extent list")
    if not isinstance(payload.get("execution_records"), list):
        raise FoldSplitError("signal support ledger has no execution_records list")
    return FoldSignalSupportLedger(payload=payload, sha256=actual_sha256, path=ledger_path)


def _recording_stem(value: Any) -> str:
    return Path(_normalize_recording(value)).stem.casefold()


def _epoch_key(value: Any) -> str:
    try:
        numeric = float(value)
        if np.isfinite(numeric) and numeric.is_integer():
            return str(int(numeric))
    except (TypeError, ValueError):
        pass
    return str(value or "").strip()


def validate_fold_signal_support(
    training: FoldSelection,
    apply: FoldSelection,
    split: FoldSplit,
    ledger: FoldSignalSupportLedger,
) -> dict[str, Any]:
    """Bind both selected partitions to ledger rows and reject shared samples.

    The interval comparison uses conservative, half-open source-array sample
    bounds. It proves only that these feature rows do not share raw source
    samples across fit/apply. Estimator history, forecast horizon, and the
    preregistered temporal purge remain separate analysis-side requirements.
    """
    if training.role != "training" or apply.role != "apply":
        raise FoldSplitError("signal-support validation requires training and apply selections")
    if training.manifest_sha256 != split.sha256 or apply.manifest_sha256 != split.sha256:
        raise FoldSplitError("signal-support selections must come from the exact split manifest")
    declared_ledger_sha256 = str(split.payload.get("signal_support_ledger_sha256", "")).strip().lower()
    if declared_ledger_sha256 != ledger.sha256:
        raise FoldSplitError("signal support ledger SHA256 does not match the split manifest")

    expected_source_hashes = split.payload["source_file_sha256"]
    ledger_rows_by_identity: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row_index, raw_row in enumerate(ledger.payload["per_epoch_input_extent"]):
        if not isinstance(raw_row, Mapping):
            raise FoldSplitError(f"signal support ledger row {row_index} is not an object")
        raw_file = _normalize_recording(raw_row.get("raw_file"))
        epoch = _epoch_key(raw_row.get("epoch_id"))
        if not raw_file or not epoch:
            raise FoldSplitError(f"signal support ledger row {row_index} lacks raw_file or epoch_id")
        ledger_rows_by_identity.setdefault((_recording_stem(raw_file), epoch), []).append(raw_row)

    execution_by_recording: dict[str, Mapping[str, Any]] = {}
    for raw_execution in ledger.payload["execution_records"]:
        if not isinstance(raw_execution, Mapping):
            raise FoldSplitError("signal support ledger has a malformed execution record")
        raw_file = _normalize_recording(raw_execution.get("raw_file"))
        if not raw_file:
            raise FoldSplitError("signal support execution record lacks raw_file")
        key = _recording_stem(raw_file)
        if key in execution_by_recording:
            raise FoldSplitError(f"signal support ledger has duplicate execution records for {raw_file}")
        execution_by_recording[key] = raw_execution

    selected_intervals: dict[str, dict[str, list[tuple[int, int]]]] = {
        "training": {},
        "apply": {},
    }
    verified_counts = {"training": 0, "apply": 0}
    for selection in (training, apply):
        role_intervals = selected_intervals[selection.role]
        for selected in selection.row_records:
            recording = _normalize_recording(selected["recording_id"])
            recording_key = _recording_stem(recording)
            epoch = _epoch_key(selected.get("epoch_id"))
            candidates = ledger_rows_by_identity.get((recording_key, epoch), [])
            if len(candidates) != 1:
                raise FoldSplitError(
                    f"selected {selection.role} row has {len(candidates)} matching support ledger rows "
                    f"for {recording} / {epoch}"
                )
            support = candidates[0]
            try:
                ledger_start = float(support["nominal_time_start_sec"])
                ledger_end = float(support["nominal_time_end_sec"])
                selected_start = float(selected["start_sec"])
                selected_end = float(selected["end_sec"])
            except (KeyError, TypeError, ValueError) as exc:
                raise FoldSplitError("selected row or support ledger has invalid nominal time bounds") from exc
            if not (
                np.isfinite(ledger_start)
                and np.isfinite(ledger_end)
                and np.isclose(ledger_start, selected_start, rtol=0.0, atol=1e-9)
                and np.isclose(ledger_end, selected_end, rtol=0.0, atol=1e-9)
            ):
                raise FoldSplitError(
                    f"support ledger nominal clock does not match selected {selection.role} row "
                    f"for {recording} / {epoch}"
                )
            if support.get("sample_mapping_status") != "nominal_bounds_mapped":
                raise FoldSplitError(f"support ledger row has no verified source-sample mapping for {recording} / {epoch}")
            if support.get("actual_sample_contributors_status") != "finite_causal_conservative_bound":
                raise FoldSplitError(f"support ledger row has unknown/noncausal support for {recording} / {epoch}")
            if support.get("actual_contributor_interval_convention") != (
                "half_open_0_based_source_array_indices; conservative_dependency_superset"
            ):
                raise FoldSplitError(f"support ledger row has an unsupported contributor interval convention for {recording}")
            if support.get("original_sample_interval_convention") != (
                "half_open_0_based_source_array_indices_within_nominal_window"
            ):
                raise FoldSplitError(f"support ledger row has an unsupported nominal interval convention for {recording}")
            source_hash = str(support.get("source_file_sha256", "")).strip().lower()
            if source_hash != str(expected_source_hashes.get(recording, "")).strip().lower():
                raise FoldSplitError(f"support ledger source header hash differs from the split for {recording}")
            execution = execution_by_recording.get(recording_key)
            if execution is None or execution.get("validation_status") != "valid":
                raise FoldSplitError(f"support ledger execution provenance is absent/invalid for {recording}")
            execution_record = execution.get("record", {})
            execution_record = execution_record if isinstance(execution_record, Mapping) else {}
            execution_source = execution_record.get("source", {})
            execution_source_hash = str(execution_source.get("sha256", "")).strip().lower() if isinstance(execution_source, Mapping) else ""
            if execution_source_hash != source_hash:
                raise FoldSplitError(f"support ledger execution source hash disagrees for {recording}")
            raw_clock = execution_record.get("raw_clock_before_crop", {})
            raw_clock = raw_clock if isinstance(raw_clock, Mapping) else {}
            try:
                source_sample_count = _strict_integer(raw_clock["n_times"])
                execution_sample_rate = float(raw_clock["sfreq_hz"])
                nominal_sample_start = _strict_integer(support["original_sample_start"])
                nominal_sample_end = _strict_integer(support["original_sample_end"])
                contributor_start = _strict_integer(support["actual_source_sample_start"])
                contributor_end = _strict_integer(support["actual_source_sample_end_exclusive"])
                source_sample_rate = float(support["source_sample_rate_hz"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise FoldSplitError(f"support ledger row has invalid source-sample bounds for {recording} / {epoch}") from exc
            if not (
                np.isfinite(execution_sample_rate)
                and execution_sample_rate > 0
                and np.isfinite(source_sample_rate)
                and source_sample_rate > 0
                and np.isclose(source_sample_rate, execution_sample_rate, rtol=1e-9, atol=1e-12)
            ):
                raise FoldSplitError(f"support ledger sample rate disagrees with execution raw clock for {recording}")
            if not (
                source_sample_count > 0
                and 0 <= contributor_start <= nominal_sample_start < nominal_sample_end <= source_sample_count
                and nominal_sample_start < contributor_end <= nominal_sample_end
                and abs(nominal_sample_start / source_sample_rate - ledger_start) <= 1.0 / source_sample_rate
                and abs(nominal_sample_end / source_sample_rate - ledger_end) <= 1.0 / source_sample_rate
            ):
                raise FoldSplitError(f"support ledger row has invalid or future-reaching bounds for {recording} / {epoch}")
            role_intervals.setdefault(recording, []).append((contributor_start, contributor_end))
            verified_counts[selection.role] += 1

    min_gap_by_recording: dict[str, int] = {}
    shared_count = 0
    for recording in sorted(set(selected_intervals["training"]) & set(selected_intervals["apply"])):
        train_bounds = sorted(selected_intervals["training"][recording])
        apply_bounds = sorted(selected_intervals["apply"][recording])
        i = j = 0
        min_gap: Optional[int] = None
        while i < len(train_bounds) and j < len(apply_bounds):
            train_start, train_end = train_bounds[i]
            apply_start, apply_end = apply_bounds[j]
            if max(train_start, apply_start) < min(train_end, apply_end):
                shared_count += 1
                raise FoldSplitError(
                    f"training/apply conservative raw-sample support overlaps for {recording} "
                    f"([{train_start}, {train_end}) vs [{apply_start}, {apply_end}))"
                )
            gap = apply_start - train_end if train_end <= apply_start else train_start - apply_end
            min_gap = gap if min_gap is None else min(min_gap, gap)
            if train_end <= apply_end:
                i += 1
            else:
                j += 1
        if min_gap is not None:
            min_gap_by_recording[recording] = int(min_gap)

    return {
        "schema_version": "mndm.fold_signal_support_validation.v1",
        "status": "verified_no_shared_source_samples",
        "signal_support_ledger_path": str(ledger.path),
        "signal_support_ledger_sha256": ledger.sha256,
        "training_rows_verified": int(verified_counts["training"]),
        "apply_rows_verified": int(verified_counts["apply"]),
        "shared_source_sample_intersection_count": int(shared_count),
        "minimum_source_sample_gap_by_recording": min_gap_by_recording,
        "scope": "conservative feature-row source-sample support only; estimator history/horizon embargo is not certified",
    }


def _strict_integer(value: Any) -> int:
    """Parse a JSON sample index without silently truncating fractional values."""
    if isinstance(value, bool):
        raise ValueError("boolean is not an integer sample index")
    numeric = float(value)
    if not np.isfinite(numeric) or not numeric.is_integer():
        raise ValueError("sample index is not a finite integer")
    return int(numeric)


def _row_identity_columns(features_df: pd.DataFrame, split: FoldSplit) -> tuple[str, str, str, str]:
    subject_col = next((name for name in ("subject", "subject_id") if name in features_df.columns), "")
    recording_col = "recording_id" if "recording_id" in features_df.columns else "file" if "file" in features_df.columns else ""
    time_reference = split.payload["time_reference"]
    start_col = str(time_reference["start_column"])
    end_col = str(time_reference["end_column"])
    if not subject_col and "file" not in features_df.columns:
        raise FoldSplitError("features require subject/subject_id or a BIDS file column")
    if not recording_col or start_col not in features_df.columns or end_col not in features_df.columns:
        raise FoldSplitError(
            "features lack the declared recording and time columns "
            f"({recording_col or 'recording_id/file'}, {start_col}, {end_col})"
        )
    return subject_col, recording_col, start_col, end_col


def select_fold_rows(features_df: pd.DataFrame, split: FoldSplit, role: str) -> FoldSelection:
    """Select exactly the rows wholly contained in the requested split ranges."""
    if role not in {"training", "apply"}:
        raise FoldSplitError("role must be 'training' or 'apply'")
    if features_df is None or features_df.empty:
        raise FoldSplitError("cannot select fold rows from an empty feature table")
    subject_col, recording_col, start_col, end_col = _row_identity_columns(features_df, split)
    ranges = split.payload["training_ranges" if role == "training" else "apply_ranges"]
    selected_indices: list[Any] = []
    row_records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, row in features_df.iterrows():
        raw_subject = row.get(subject_col) if subject_col else None
        if raw_subject is None or pd.isna(raw_subject) or not str(raw_subject).strip():
            match = _SUBJECT_TOKEN_RE.search(str(row.get("file", "")))
            raw_subject = match.group(1) if match else ""
        subject = _normalize_subject(raw_subject)
        recording = _normalize_recording(row.get(recording_col))
        try:
            start = float(row[start_col])
            end = float(row[end_col])
        except (TypeError, ValueError):
            raise FoldSplitError(f"row {index!r} has missing/non-numeric time bounds")
        if not (np.isfinite(start) and np.isfinite(end) and end > start):
            raise FoldSplitError(f"row {index!r} has invalid time bounds")
        matched = []
        intersects_any = False
        for interval in ranges:
            if interval["subject_id"] != subject or interval["recording_id"] != recording:
                continue
            intersects = start < interval["end_sec"] and end > interval["start_sec"]
            contained = start >= interval["start_sec"] and end <= interval["end_sec"]
            intersects_any = intersects_any or intersects
            if intersects and not contained:
                raise FoldSplitError(
                    f"row {index!r} straddles a {role} range boundary; refusing ambiguous assignment"
                )
            if contained:
                matched.append(interval)
        if intersects_any and len(matched) != 1:
            raise FoldSplitError(f"row {index!r} is assigned to multiple {role} ranges")
        if not matched:
            continue
        if "recording_id" in features_df.columns and "file" in features_df.columns:
            raw_file = row.get("file")
            if raw_file is None or pd.isna(raw_file) or not str(raw_file).strip():
                raise FoldSplitError(f"row {index!r} has recording_id but no source file identity")
            if recording != _normalize_recording(raw_file):
                raise FoldSplitError(
                    f"row {index!r} recording_id does not match its source file basename"
                )
        epoch_id = str(row.get("epoch_id", row.get("window_id", "")) or "")
        identity = {
            "subject_id": subject,
            "recording_id": recording,
            "epoch_id": epoch_id,
            "start_sec": start,
            "end_sec": end,
        }
        row_id = _sha256_text(_canonical_json(identity))
        if row_id in seen_ids:
            raise FoldSplitError(f"duplicate feature-row identity in {role} partition: {row_id}")
        seen_ids.add(row_id)
        selected_indices.append(index)
        row_records.append({"row_id": row_id, **identity})
    if not row_records:
        raise FoldSplitError(f"no feature rows matched the {role} ranges")
    return FoldSelection(
        frame=features_df.loc[selected_indices].copy(),
        role=role,
        row_records=tuple(row_records),
        manifest_sha256=split.sha256,
    )


def fit_fold_anchor_artifact(
    selection: FoldSelection,
    split: FoldSplit,
    *,
    anchor_id: str,
    anchor_source: str,
    cohort_filter: str,
    feature_standardization: Optional[Mapping[str, Sequence[str]]],
    min_subjects: int,
    scale_method: str,
    transform_config: Mapping[str, Any],
    code_version: str,
    implementation_hash: str,
    verified_source_file_sha256: Mapping[str, str],
    feature_columns: Sequence[str],
    signal_support_validation: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Fit a cohort anchor from training rows and bind it to the split inputs."""
    if selection.role != "training" or selection.manifest_sha256 != split.sha256:
        raise FoldSplitError("anchor fit requires the training partition from this exact manifest")
    expected_support_hash = str(split.payload.get("signal_support_ledger_sha256", "")).strip().lower()
    if expected_support_hash:
        if not isinstance(signal_support_validation, Mapping):
            raise FoldSplitError("fold-safe anchor fit requires verified train/apply signal-support validation")
        if (
            signal_support_validation.get("status") != "verified_no_shared_source_samples"
            or str(signal_support_validation.get("signal_support_ledger_sha256", "")).strip().lower()
            != expected_support_hash
        ):
            raise FoldSplitError("fold-safe anchor fit signal-support validation does not match the split manifest")
    from .. import anchors

    frame = selection.frame
    # The selector resolves missing subject cells from the source filename and
    # records canonical IDs. Reuse those IDs here; stringifying missing cells
    # would pool distinct subjects under labels such as "nan".
    subject_ids = [str(item["subject_id"]) for item in selection.row_records]
    file_ids = frame["file"].astype(str).tolist() if "file" in frame.columns else [
        item["recording_id"] for item in selection.row_records
    ]
    group_by_subject: dict[str, str] = {}
    if "group" in frame.columns:
        for subject, group in zip(subject_ids, frame["group"].tolist()):
            if group is not None and not pd.isna(group) and str(group).strip():
                group_by_subject.setdefault(_normalize_subject(subject), str(group).strip())

    artifact = anchors.fit_feature_anchors_from_features_df(
        frame,
        anchor_id=anchor_id,
        anchor_source=anchor_source,
        cohort_filter=cohort_filter,
        feature_standardization=feature_standardization,
        min_subjects=min_subjects,
        scale_method=scale_method,
        subject_ids=subject_ids,
        file_ids=file_ids,
        group_by_subject=group_by_subject,
        feature_columns=feature_columns,
    )
    spec = dict(artifact.get("spec", {}))
    source_hashes = split.payload["source_file_sha256"]
    fit_recordings = sorted({str(item["recording_id"]) for item in selection.row_records})
    missing_verified = sorted(set(fit_recordings) - set(verified_source_file_sha256))
    if missing_verified:
        raise FoldSplitError("raw source hashes were not verified for fit recording(s): " + ", ".join(missing_verified))
    for recording in fit_recordings:
        actual = str(verified_source_file_sha256[recording]).lower()
        if actual != str(source_hashes[recording]).lower():
            raise FoldSplitError(f"verified raw source SHA256 does not match split manifest for {recording}")
    input_config_hash = transform_config_sha256(transform_config)
    fit_rows_payload = _canonical_json(list(selection.row_records))
    feature_fit: dict[str, dict[str, Any]] = {}
    for name in sorted(str(v) for v in feature_columns):
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        transformed = anchors.transform_values_for_anchor(
            values,
            name,
            feature_standardization=feature_standardization,
        )
        finite = np.isfinite(transformed)
        row_ids = [selection.row_records[pos]["row_id"] for pos, ok in enumerate(finite) if ok]
        if not row_ids:
            continue
        rows = pd.DataFrame(
            {
                "subject_id": [_normalize_subject(value) for value in subject_ids],
                "file_id": file_ids,
                "value": transformed,
                "row_id": [item["row_id"] for item in selection.row_records],
            }
        )
        rows = rows.loc[finite]
        file_estimates: list[dict[str, Any]] = []
        subject_file_medians: dict[str, list[float]] = {}
        for (subject, file_id), group in rows.groupby(["subject_id", "file_id"], sort=False):
            median = float(group["value"].median())
            subject = str(subject)
            subject_file_medians.setdefault(subject, []).append(median)
            file_estimates.append(
                {
                    "subject_id": subject,
                    "file_id": str(file_id),
                    "row_count": int(len(group)),
                    "row_ids": group["row_id"].astype(str).tolist(),
                    "median": median,
                }
            )
        subject_estimates = [
            {"subject_id": subject, "median_of_file_medians": float(np.nanmedian(file_values))}
            for subject, file_values in sorted(subject_file_medians.items())
        ]
        feature_fit[name] = {
            "fit_status": "recorded",
            "fit_count": int(len(subject_estimates)),
            "fit_count_unit": "subject_medians",
            "fit_population_hash": _sha256_text(_canonical_json(subject_estimates)),
            "fit_population_ids": [item["subject_id"] for item in subject_estimates],
            "fit_row_count": int(len(row_ids)),
            "fit_row_population_hash": _sha256_text(_canonical_json(row_ids)),
            "fit_row_ids": row_ids,
            "file_median_inputs": file_estimates,
            "subject_median_inputs": subject_estimates,
        }
    provenance = {
        "schema_version": "mndm.fold_transform_anchor.v1",
        "method": "subject_balanced_feature_anchor",
        "method_version": str(spec.get("schema_version", "mndm.feature_anchors.v2.1")),
        "inputs": [str(name) for name in feature_columns],
        "fit_status": "complete",
        "apply_status": "pending",
        "split_id": str(split.payload["split_id"]),
        "split_manifest_path": str(split.path),
        "split_manifest_sha256": split.sha256,
        "signal_support_ledger_sha256": expected_support_hash or None,
        "signal_support_validation_status": (
            signal_support_validation.get("status") if isinstance(signal_support_validation, Mapping) else None
        ),
        "fit_population_count": len(selection.row_records),
        "fit_population_hash": _sha256_text(fit_rows_payload),
        "fit_row_ids": selection.row_ids,
        "fit_manifest_ranges": split.payload["training_ranges"],
        "fit_time_ranges": _selection_time_ranges(selection),
        "time_reference": dict(split.payload["time_reference"]),
        "fit_source_recordings": fit_recordings,
        "fit_source_file_sha256": {name: source_hashes[name] for name in fit_recordings},
        "fit_source_hash_verification": {"status": "verified", "actual_sha256": dict(verified_source_file_sha256)},
        "feature_fit_populations": feature_fit,
        "config_sha256": input_config_hash,
        "code_version": str(code_version),
        "implementation_sha256": str(implementation_hash),
        "reference_population": "manifest-selected training rows, subject-balanced within each feature",
        "feature_standardization_scope": "training partition only; fixed anchor applied unchanged to the apply partition",
    }
    spec["fold_safe_fit"] = provenance
    spec.pop("anchor_hash", None)
    artifact["spec"] = spec
    # The artifact digest binds both feature parameters and their fit provenance.
    digest_payload = {"spec": spec, "features": artifact.get("features", [])}
    parameter_hash = anchors.stable_anchor_hash(digest_payload)
    spec["anchor_hash"] = parameter_hash
    spec["fold_safe_fit"]["parameter_artifact_hash"] = parameter_hash
    return artifact


def validate_fold_anchor_artifact(artifact: Mapping[str, Any], split: FoldSplit) -> Mapping[str, Any]:
    """Require a complete train-only anchor fitted for the supplied split."""
    spec = artifact.get("spec", {}) if isinstance(artifact, Mapping) else {}
    record = spec.get("fold_safe_fit", {}) if isinstance(spec, Mapping) else {}
    if not isinstance(record, Mapping) or record.get("schema_version") != "mndm.fold_transform_anchor.v1":
        raise FoldSplitError("anchor artifact has no fold-safe fit provenance")
    if record.get("fit_status") != "complete":
        raise FoldSplitError(f"anchor fit status is not complete: {record.get('fit_status')}")
    if record.get("split_manifest_sha256") != split.sha256:
        raise FoldSplitError("anchor artifact was fitted against a different split manifest")
    expected_support_hash = str(split.payload.get("signal_support_ledger_sha256", "")).strip().lower()
    if expected_support_hash and (
        str(record.get("signal_support_ledger_sha256", "")).strip().lower() != expected_support_hash
        or record.get("signal_support_validation_status") != "verified_no_shared_source_samples"
    ):
        raise FoldSplitError("anchor artifact has no matching verified signal-support validation")
    source_verification = record.get("fit_source_hash_verification", {})
    if not isinstance(source_verification, Mapping) or source_verification.get("status") != "verified":
        raise FoldSplitError("anchor artifact has no verified raw-source hashes")
    actual_hashes = source_verification.get("actual_sha256", {})
    expected_hashes = split.payload["source_file_sha256"]
    if not isinstance(actual_hashes, Mapping):
        raise FoldSplitError("anchor artifact has invalid raw-source hash verification")
    for raw_name in record.get("fit_source_recordings", []):
        name = _normalize_recording(raw_name)
        if str(actual_hashes.get(name, "")).lower() != str(expected_hashes.get(name, "")).lower():
            raise FoldSplitError(f"anchor artifact source hash verification does not match the split for {name}")
    fit_ids = set(str(value) for value in record.get("fit_row_ids", []))
    if not fit_ids:
        raise FoldSplitError("anchor artifact has no recorded fit row identities")
    from .. import anchors

    hash_spec = dict(spec)
    hash_record = dict(record)
    expected_hash = str(hash_spec.pop("anchor_hash", ""))
    hash_record.pop("parameter_artifact_hash", None)
    hash_spec["fold_safe_fit"] = hash_record
    actual_hash = anchors.stable_anchor_hash({"spec": hash_spec, "features": artifact.get("features", [])})
    if not expected_hash or actual_hash != expected_hash or record.get("parameter_artifact_hash") != expected_hash:
        raise FoldSplitError("anchor parameter artifact hash does not match its saved contents")
    return record


def _selection_time_ranges(selection: FoldSelection) -> list[dict[str, Any]]:
    """List the exact row intervals without merging across gaps or overlap."""
    return [
        {
            "subject_id": str(item["subject_id"]),
            "recording_id": str(item["recording_id"]),
            "start_sec": float(item["start_sec"]),
            "end_sec": float(item["end_sec"]),
        }
        for item in selection.row_records
    ]


def build_apply_provenance(
    artifact: Mapping[str, Any],
    split: FoldSplit,
    selection: FoldSelection,
    *,
    verified_source_file_sha256: Mapping[str, str],
) -> dict[str, Any]:
    """Build the per-apply-record transform record stored with each chart."""
    fit_record = validate_fold_anchor_artifact(artifact, split)
    apply_ids = selection.row_ids
    fit_ids = set(str(value) for value in fit_record.get("fit_row_ids", []))
    overlap = fit_ids.intersection(apply_ids)
    if overlap:
        raise FoldSplitError(f"fit/apply row IDs overlap ({len(overlap)} shared rows)")
    source_hashes = split.payload["source_file_sha256"]
    apply_recordings = sorted({str(item["recording_id"]) for item in selection.row_records})
    missing_verified = sorted(set(apply_recordings) - set(verified_source_file_sha256))
    if missing_verified:
        raise FoldSplitError("raw source hashes were not verified for apply recording(s): " + ", ".join(missing_verified))
    for recording in apply_recordings:
        actual = str(verified_source_file_sha256[recording]).lower()
        if actual != str(source_hashes[recording]).lower():
            raise FoldSplitError(f"verified raw source SHA256 does not match split manifest for {recording}")
    return {
        "schema_version": "mndm.fold_transform_apply.v1",
        "method": fit_record.get("method"),
        "method_version": fit_record.get("method_version"),
        "fit_status": "complete",
        "apply_status": "applied",
        "split_id": str(split.payload["split_id"]),
        "split_manifest_sha256": split.sha256,
        "signal_support_ledger_sha256": fit_record.get("signal_support_ledger_sha256"),
        "parameter_artifact_hash": artifact.get("spec", {}).get("anchor_hash"),
        "fit_population_hash": fit_record.get("fit_population_hash"),
        "fit_population_count": fit_record.get("fit_population_count"),
        "fit_row_ids": fit_record.get("fit_row_ids"),
        "fit_time_ranges": fit_record.get("fit_time_ranges"),
        "time_reference": dict(split.payload["time_reference"]),
        "apply_population_count": len(selection.row_records),
        "apply_population_hash": _sha256_text(_canonical_json(selection.row_records)),
        "apply_row_ids": apply_ids,
        "apply_time_ranges": _selection_time_ranges(selection),
        "apply_source_file_sha256": {name: source_hashes[name] for name in apply_recordings},
        "apply_source_hash_verification": {"status": "verified", "actual_sha256": dict(verified_source_file_sha256)},
        "refusal_reason": None,
    }


def build_training_replay_provenance(
    artifact: Mapping[str, Any], split: FoldSplit, selection: FoldSelection
) -> dict[str, Any]:
    """Describe training rows replayed through their own frozen fit artifact."""
    fit_record = validate_fold_anchor_artifact(artifact, split)
    if selection.role != "training":
        raise FoldSplitError("training replay provenance requires training rows")
    selection_ids = set(selection.row_ids)
    artifact_ids = set(str(value) for value in fit_record.get("fit_row_ids", []))
    if not selection_ids.issubset(artifact_ids):
        raise FoldSplitError("training replay rows are not a subset of the anchor fit population")
    return {
        "schema_version": "mndm.fold_transform_apply.v1",
        "method": fit_record.get("method"),
        "method_version": fit_record.get("method_version"),
        "fit_status": "complete",
        "apply_status": "training_replay",
        "split_id": str(split.payload["split_id"]),
        "split_manifest_sha256": split.sha256,
        "signal_support_ledger_sha256": fit_record.get("signal_support_ledger_sha256"),
        "parameter_artifact_hash": artifact.get("spec", {}).get("anchor_hash"),
        "fit_population_hash": fit_record.get("fit_population_hash"),
        "fit_population_count": fit_record.get("fit_population_count"),
        "fit_row_ids": fit_record.get("fit_row_ids"),
        "fit_time_ranges": fit_record.get("fit_time_ranges"),
        "time_reference": dict(split.payload["time_reference"]),
        "apply_population_count": len(selection.row_records),
        "apply_population_hash": _sha256_text(_canonical_json(selection.row_records)),
        "apply_row_ids": selection.row_ids,
        "apply_time_ranges": _selection_time_ranges(selection),
        "refusal_reason": None,
    }
