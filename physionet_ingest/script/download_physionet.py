from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import fnmatch
import hashlib
import json
import logging
import os
import random
import re
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

LOGGER = logging.getLogger("physionet_ingest")
_ICARE_EEG_RECORD_RE = re.compile(
    r"^(?P<patient>\d{3,8})_(?P<run>\d+)_(?P<acq>\d+)_EEG\.(?P<ext>hea|mat|dat)$",
    re.IGNORECASE,
)


def load_yaml_config(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return config


def deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def normalize_relative_path(relative_path: str) -> str:
    normalized = relative_path.strip().lstrip("./")
    if not normalized:
        raise ValueError("Encountered empty relative path")
    posix_path = PurePosixPath(normalized)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"Unsafe relative path: {relative_path}")
    return posix_path.as_posix()


def build_dataset_root_url(base_url: str, remote_root: str) -> str:
    return f"{base_url.rstrip('/')}/{remote_root.strip('/')}"


def build_remote_file_url(dataset_root_url: str, relative_path: str, with_download_query: bool = True) -> str:
    safe_rel = normalize_relative_path(relative_path)
    encoded = urllib_parse.quote(safe_rel, safe="/")
    suffix = "?download" if with_download_query else ""
    return f"{dataset_root_url}/{encoded}{suffix}"


def parse_records_text(records_text: str) -> list[str]:
    records: list[str] = []
    for line in records_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        records.append(normalize_relative_path(stripped))
    return records


def parse_sha256sums_text(checksums_text: str) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in checksums_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        digest = parts[0].lower()
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            continue
        remote_path = parts[-1]
        if remote_path.startswith("*"):
            remote_path = remote_path[1:]
        try:
            checksums[normalize_relative_path(remote_path)] = digest
        except ValueError:
            continue
    return checksums


def expand_record_paths(record_paths: list[str], extensions: list[str]) -> list[str]:
    output: list[str] = []
    for record_path in record_paths:
        normalized = normalize_relative_path(record_path)
        suffix = PurePosixPath(normalized).suffix
        if suffix:
            output.append(normalized)
            continue
        if not extensions:
            output.append(normalized)
            continue
        for extension in extensions:
            ext = extension if extension.startswith(".") else f".{extension}"
            output.append(f"{normalized}{ext}")
    return dedupe_keep_order(output)


def path_matches_any_glob(path: str, include_globs: list[str]) -> bool:
    if not include_globs:
        return True
    normalized = normalize_relative_path(path)
    basename = PurePosixPath(normalized).name
    for pattern in include_globs:
        if fnmatch.fnmatch(normalized, pattern) or fnmatch.fnmatch(basename, pattern):
            return True
    return False


def expand_selected_records_to_files(
    selected_records: list[str],
    checksums_lookup: dict[str, str],
    include_file_globs: list[str],
    fallback_extensions: list[str],
    max_files_per_patient: int | None = None,
) -> list[str]:
    checksum_keys = list(checksums_lookup.keys())
    expanded: list[str] = []

    for record_path in selected_records:
        normalized_record = normalize_relative_path(record_path).rstrip("/")
        prefix = f"{normalized_record}/"
        matched_files = [key for key in checksum_keys if key.startswith(prefix)]
        if matched_files:
            filtered = [path for path in matched_files if path_matches_any_glob(path, include_file_globs)]
            if max_files_per_patient is not None and max_files_per_patient > 0:
                filtered = filtered[:max_files_per_patient]
            expanded.extend(filtered)
            continue

        expanded.extend(expand_record_paths([normalized_record], extensions=fallback_extensions))

    return dedupe_keep_order(expanded)


def parse_icare_eeg_file_identity(relative_path: str) -> tuple[str, int, int, str] | None:
    normalized = normalize_relative_path(relative_path)
    name = PurePosixPath(normalized).name
    match = _ICARE_EEG_RECORD_RE.match(name)
    if not match:
        return None
    patient = match.group("patient")
    run_idx = int(match.group("run"))
    acq_idx = int(match.group("acq"))
    extension = match.group("ext").lower()
    return patient, run_idx, acq_idx, extension


def parse_wfdb_duration_seconds(header_text: str) -> float | None:
    for line in header_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 4:
            return None
        fs_token = parts[2].split("/")[0]
        fs_match = re.match(r"^([0-9]+(?:\.[0-9]+)?)", fs_token)
        siglen_match = re.match(r"^([0-9]+)", parts[3])
        if not fs_match or not siglen_match:
            return None
        fs_hz = float(fs_match.group(1))
        siglen = float(siglen_match.group(1))
        if fs_hz <= 0 or siglen <= 0:
            return None
        return siglen / fs_hz
    return None


def _read_or_fetch_wfdb_header_text(
    *,
    relative_header_path: str,
    output_dir: Path,
    dataset_root_url: str,
    timeout_seconds: float,
    retries: int,
    retry_backoff_seconds: float,
    user_agent: str,
) -> str:
    normalized = normalize_relative_path(relative_header_path)
    local_path = output_dir.joinpath(*PurePosixPath(normalized).parts)
    if local_path.exists():
        try:
            return local_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return local_path.read_text(encoding="latin-1", errors="ignore")

    remote_url = build_remote_file_url(dataset_root_url, normalized, with_download_query=True)
    return fetch_text_with_retries(
        url=remote_url,
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        user_agent=user_agent,
    )


def limit_icare_eeg_files_to_first_hours(
    *,
    selected_paths: list[str],
    max_hours_per_patient: float,
    min_hours_per_patient: float = 0.0,
    output_dir: Path,
    dataset_root_url: str,
    timeout_seconds: float,
    retries: int,
    retry_backoff_seconds: float,
    user_agent: str,
) -> tuple[list[str], dict[str, Any]]:
    if max_hours_per_patient <= 0:
        raise ValueError("subset.max_eeg_hours_per_patient must be > 0")
    if min_hours_per_patient < 0:
        raise ValueError("subset.min_eeg_hours_per_patient must be >= 0")
    if min_hours_per_patient >= max_hours_per_patient:
        raise ValueError("subset.min_eeg_hours_per_patient must be < subset.max_eeg_hours_per_patient")

    start_seconds = min_hours_per_patient * 3600.0
    end_seconds = max_hours_per_patient * 3600.0
    normalized_paths = [normalize_relative_path(path) for path in selected_paths]
    headers_by_patient: dict[str, list[tuple[int, int, str]]] = {}

    for path in normalized_paths:
        identity = parse_icare_eeg_file_identity(path)
        if identity is None:
            continue
        patient_id, run_idx, acq_idx, extension = identity
        if extension != "hea":
            continue
        headers_by_patient.setdefault(patient_id, []).append((run_idx, acq_idx, path))

    if not headers_by_patient:
        return dedupe_keep_order(normalized_paths), {
            "enabled": False,
            "reason": "no_icare_eeg_headers_found",
            "min_hours_per_patient": min_hours_per_patient,
            "max_hours_per_patient": max_hours_per_patient,
        }

    selected_stems: set[str] = set()
    per_patient_summary: list[dict[str, Any]] = []
    duration_parse_failures = 0
    header_fetches = 0

    for patient_id in sorted(headers_by_patient):
        ordered_headers = sorted(headers_by_patient[patient_id], key=lambda row: (row[0], row[1], row[2]))
        cumulative_seconds = 0.0
        selected_header_records = 0
        selected_window_seconds = 0.0
        total_header_records = len(ordered_headers)

        for _, _, header_path in ordered_headers:
            stem = header_path.rsplit(".", 1)[0]
            try:
                header_text = _read_or_fetch_wfdb_header_text(
                    relative_header_path=header_path,
                    output_dir=output_dir,
                    dataset_root_url=dataset_root_url,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    user_agent=user_agent,
                )
                header_fetches += 1
                duration_seconds = parse_wfdb_duration_seconds(header_text)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "Could not inspect WFDB header for %s: %s. Keeping record to avoid accidental data loss.",
                    header_path,
                    exc,
                )
                selected_stems.add(stem)
                selected_header_records += 1
                continue

            if duration_seconds is None:
                duration_parse_failures += 1
                LOGGER.warning(
                    "Could not parse WFDB duration for %s. Keeping record to avoid accidental data loss.",
                    header_path,
                )
                selected_stems.add(stem)
                selected_header_records += 1
                continue

            record_start = cumulative_seconds
            record_end = cumulative_seconds + duration_seconds
            cumulative_seconds = record_end

            if record_end <= start_seconds + 1e-6:
                continue
            if record_start >= end_seconds - 1e-6:
                break

            overlap_seconds = max(
                0.0,
                min(record_end, end_seconds) - max(record_start, start_seconds),
            )
            if overlap_seconds > 0.0:
                selected_stems.add(stem)
                selected_header_records += 1
                selected_window_seconds += overlap_seconds
                continue

        per_patient_summary.append(
            {
                "patient_id": patient_id,
                "header_records_total": total_header_records,
                "header_records_selected": selected_header_records,
                "selected_hours_approx": round(selected_window_seconds / 3600.0, 3),
            }
        )

    filtered_paths: list[str] = []
    eeg_files_before = 0
    eeg_files_kept = 0
    eeg_files_dropped = 0

    for path in normalized_paths:
        identity = parse_icare_eeg_file_identity(path)
        if identity is None:
            filtered_paths.append(path)
            continue

        eeg_files_before += 1
        stem = path.rsplit(".", 1)[0]
        if stem in selected_stems:
            filtered_paths.append(path)
            eeg_files_kept += 1
        else:
            eeg_files_dropped += 1

    filtered_deduped = dedupe_keep_order(filtered_paths)
    summary: dict[str, Any] = {
        "enabled": True,
        "min_hours_per_patient": min_hours_per_patient,
        "max_hours_per_patient": max_hours_per_patient,
        "window_seconds_per_patient": [start_seconds, end_seconds],
        "patients_seen": len(headers_by_patient),
        "header_fetches": header_fetches,
        "duration_parse_failures": duration_parse_failures,
        "eeg_files_before": eeg_files_before,
        "eeg_files_kept": eeg_files_kept,
        "eeg_files_dropped": eeg_files_dropped,
        "patient_examples": per_patient_summary[:10],
    }
    return filtered_deduped, summary


def extract_patient_id(record_path: str) -> str | None:
    normalized = normalize_relative_path(record_path)
    basename = PurePosixPath(normalized).name
    match = re.match(r"^(\d{3,8})[_-]", basename)
    if match:
        return match.group(1)
    if basename.isdigit() and 3 <= len(basename) <= 8:
        return basename
    for part in reversed(PurePosixPath(normalized).parts):
        if part.isdigit() and 3 <= len(part) <= 8:
            return part
    return None


def collect_patients_in_order(record_paths: list[str]) -> list[str]:
    patient_order: list[str] = []
    seen: set[str] = set()
    for record_path in record_paths:
        patient_id = extract_patient_id(record_path)
        if not patient_id or patient_id in seen:
            continue
        patient_order.append(patient_id)
        seen.add(patient_id)
    return patient_order


def collect_records_for_patients(record_paths: list[str], selected_patients: list[str]) -> list[str]:
    selected_patient_set = set(selected_patients)
    selected_records: list[str] = []
    for record_path in record_paths:
        patient_id = extract_patient_id(record_path)
        if patient_id in selected_patient_set:
            selected_records.append(normalize_relative_path(record_path))
    return dedupe_keep_order(selected_records)


def select_first_n_patients(record_paths: list[str], patient_count: int) -> tuple[list[str], list[str]]:
    if patient_count <= 0:
        raise ValueError("patient_count must be positive")

    patient_order = collect_patients_in_order(record_paths)
    selected_patients = patient_order[:patient_count]
    selected_records = collect_records_for_patients(record_paths, selected_patients)
    return selected_patients, selected_records


def select_random_n_patients(record_paths: list[str], patient_count: int, random_seed: int) -> tuple[list[str], list[str]]:
    if patient_count <= 0:
        raise ValueError("patient_count must be positive")

    patient_order = collect_patients_in_order(record_paths)
    if not patient_order:
        return [], []

    sample_count = min(patient_count, len(patient_order))
    rng = random.Random(random_seed)
    selected_patients = rng.sample(patient_order, sample_count)
    selected_records = collect_records_for_patients(record_paths, selected_patients)
    return selected_patients, selected_records


def _extract_patient_token(value: Any) -> str | None:
    token = str(value).strip()
    if not token:
        return None
    if token.isdigit() and 1 <= len(token) <= 12:
        return token
    match = re.search(r"(\d{3,8})", token)
    if match:
        return match.group(1)
    return None


def _coerce_explicit_patient_ids(raw_value: Any) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        values = [part.strip() for part in raw_value.split(",")]
    elif isinstance(raw_value, (list, tuple, set)):
        values = list(raw_value)
    else:
        raise ValueError("subset.patient_ids must be a list/tuple/set or comma-separated string")

    coerced: list[str] = []
    for item in values:
        token = _extract_patient_token(item)
        if token is not None:
            coerced.append(token)
    return dedupe_keep_order(coerced)


def select_explicit_patient_ids(record_paths: list[str], requested_patient_ids: list[str]) -> tuple[list[str], list[str]]:
    if not requested_patient_ids:
        raise ValueError("subset.patient_ids is empty for strategy=explicit_patient_ids")

    patient_order = collect_patients_in_order(record_paths)
    if not patient_order:
        return [], []

    by_exact = {patient_id: patient_id for patient_id in patient_order}
    by_int: dict[int, str] = {}
    for patient_id in patient_order:
        if patient_id.isdigit():
            by_int[int(patient_id)] = patient_id

    selected_patients: list[str] = []
    missing_requested: list[str] = []

    for requested in requested_patient_ids:
        normalized_request = _extract_patient_token(requested) or requested
        resolved = by_exact.get(normalized_request)
        if resolved is None and normalized_request.isdigit():
            resolved = by_int.get(int(normalized_request))
        if resolved is None:
            missing_requested.append(str(requested))
            continue
        if resolved not in selected_patients:
            selected_patients.append(resolved)

    if missing_requested:
        preview = ", ".join(missing_requested[:10])
        suffix = "..." if len(missing_requested) > 10 else ""
        raise ValueError(
            "Requested explicit patient IDs not found in RECORDS/checksum pool: "
            f"{preview}{suffix}"
        )

    selected_records = collect_records_for_patients(record_paths, selected_patients)
    return selected_patients, selected_records


def select_patients_by_strategy(
    record_paths: list[str],
    strategy: str,
    patient_count: int,
    random_seed: int,
    explicit_patient_ids: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    if strategy == "first_n_patients":
        return select_first_n_patients(record_paths, patient_count=patient_count)
    if strategy == "random_n_patients":
        return select_random_n_patients(
            record_paths,
            patient_count=patient_count,
            random_seed=random_seed,
        )
    if strategy == "explicit_patient_ids":
        return select_explicit_patient_ids(record_paths, requested_patient_ids=explicit_patient_ids or [])
    raise ValueError(f"Unsupported subset.strategy: {strategy}")


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        ordered.append(value)
        seen.add(value)
    return ordered


def compute_sha256(path: Path, chunk_size_bytes: int = 1_048_576) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def fetch_bytes_with_retries(
    url: str,
    timeout_seconds: float,
    retries: int,
    retry_backoff_seconds: float,
    user_agent: str,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib_request.Request(url, headers={"User-Agent": user_agent})
            with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
                return response.read()
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            sleep_time = retry_backoff_seconds * attempt
            LOGGER.warning("Request failed (%s/%s) for %s: %s", attempt, retries, url, exc)
            time.sleep(sleep_time)
    raise RuntimeError(f"Failed to fetch URL after {retries} attempts: {url} ({last_error})")


def fetch_text_with_retries(
    url: str,
    timeout_seconds: float,
    retries: int,
    retry_backoff_seconds: float,
    user_agent: str,
) -> str:
    return fetch_bytes_with_retries(
        url=url,
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        user_agent=user_agent,
    ).decode("utf-8")


def fetch_remote_content_length_with_retries(
    url: str,
    timeout_seconds: float,
    retries: int,
    retry_backoff_seconds: float,
    user_agent: str,
) -> int | None:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib_request.Request(url, method="HEAD", headers={"User-Agent": user_agent})
            with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
                header = response.headers.get("Content-Length")
                if header is None:
                    return None
                return int(header)
        except (urllib_error.HTTPError, urllib_error.URLError, TimeoutError, OSError, ValueError) as exc:
            last_error = exc
            if attempt >= retries:
                break
            sleep_time = retry_backoff_seconds * attempt
            LOGGER.warning("HEAD request failed (%s/%s) for %s: %s", attempt, retries, url, exc)
            time.sleep(sleep_time)
    LOGGER.warning("Could not fetch remote content length for %s (%s)", url, last_error)
    return None


def download_file_with_retries(
    url: str,
    output_path: Path,
    timeout_seconds: float,
    retries: int,
    retry_backoff_seconds: float,
    chunk_size_bytes: int,
    user_agent: str,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            total_written = 0
            req = urllib_request.Request(url, headers={"User-Agent": user_agent})
            with urllib_request.urlopen(req, timeout=timeout_seconds) as response:
                with tmp_path.open("wb") as handle:
                    while True:
                        block = response.read(chunk_size_bytes)
                        if not block:
                            break
                        handle.write(block)
                        total_written += len(block)
            tmp_path.replace(output_path)
            return total_written
        except (urllib_error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt >= retries:
                break
            sleep_time = retry_backoff_seconds * attempt
            LOGGER.warning("Download failed (%s/%s) for %s: %s", attempt, retries, url, exc)
            time.sleep(sleep_time)

    raise RuntimeError(f"Failed download after {retries} attempts: {url} ({last_error})")


def write_manifest_jsonl(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_manifest_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "remote_path",
        "local_path",
        "patient_id",
        "status",
        "bytes",
        "checksum_expected",
        "checksum_actual",
        "error",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def write_lines(lines: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def download_or_skip_relative_path(
    relative_path: str,
    output_dir: Path,
    dataset_root_url: str,
    checksums: dict[str, str],
    overwrite: bool,
    verify_checksum: bool,
    verify_existing_size: bool,
    timeout_seconds: float,
    retries: int,
    retry_backoff_seconds: float,
    chunk_size_bytes: int,
    user_agent: str,
) -> dict[str, Any]:
    normalized_path = normalize_relative_path(relative_path)
    patient_id = extract_patient_id(normalized_path) or ""
    expected_checksum = checksums.get(normalized_path)
    local_path = output_dir.joinpath(*PurePosixPath(normalized_path).parts)

    row: dict[str, Any] = {
        "remote_path": normalized_path,
        "local_path": str(local_path),
        "patient_id": patient_id,
        "status": "",
        "bytes": 0,
        "checksum_expected": expected_checksum or "",
        "checksum_actual": "",
        "error": "",
    }

    try:
        should_download = True
        if local_path.exists() and not overwrite:
            if verify_checksum and expected_checksum:
                row["checksum_actual"] = compute_sha256(local_path, chunk_size_bytes=chunk_size_bytes)
                if row["checksum_actual"] == expected_checksum:
                    row["status"] = "skipped_exists_verified"
                    should_download = False
                else:
                    row["status"] = "checksum_mismatch_redownload"
            elif verify_existing_size:
                remote_url = build_remote_file_url(dataset_root_url, normalized_path, with_download_query=True)
                remote_size = fetch_remote_content_length_with_retries(
                    url=remote_url,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    user_agent=user_agent,
                )
                local_size = local_path.stat().st_size
                if remote_size is None:
                    row["status"] = "skipped_exists_size_unverified"
                    should_download = False
                elif local_size == remote_size:
                    row["status"] = "skipped_exists_size_match"
                    should_download = False
                else:
                    row["status"] = "size_mismatch_redownload"
            else:
                row["status"] = "skipped_exists"
                should_download = False

        if should_download:
            remote_url = build_remote_file_url(dataset_root_url, normalized_path, with_download_query=True)
            bytes_written = download_file_with_retries(
                url=remote_url,
                output_path=local_path,
                timeout_seconds=timeout_seconds,
                retries=retries,
                retry_backoff_seconds=retry_backoff_seconds,
                chunk_size_bytes=chunk_size_bytes,
                user_agent=user_agent,
            )
            row["bytes"] = bytes_written
            row["status"] = "downloaded"
            if verify_checksum and expected_checksum:
                row["checksum_actual"] = compute_sha256(local_path, chunk_size_bytes=chunk_size_bytes)
                if row["checksum_actual"] != expected_checksum:
                    row["status"] = "error"
                    row["error"] = "checksum_mismatch_after_download"
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = str(exc)

    return row


def maybe_create_physionet_client(base_url: str, auth_env: dict[str, Any]) -> Any | None:
    try:
        from physionet import PhysioNetClient  # type: ignore
    except Exception as exc:
        LOGGER.warning("physionet package unavailable, continuing without API client: %s", exc)
        return None

    username_env = str(auth_env.get("username", "PHYSIONET_USERNAME"))
    password_env = str(auth_env.get("password", "PHYSIONET_PASSWORD"))
    username = os.getenv(username_env, "")
    password = os.getenv(password_env, "")
    if username and password:
        LOGGER.info("Using PhysioNet API client with credentials from env vars.")
        return PhysioNetClient(base_url=base_url, username=username, password=password)
    LOGGER.info("Using PhysioNet API client without credentials.")
    return PhysioNetClient(base_url=base_url)


def fetch_checksums_text(
    client: Any | None,
    dataset_root_url: str,
    slug: str,
    version: str,
    checksums_file: str,
    timeout_seconds: float,
    retries: int,
    retry_backoff_seconds: float,
    user_agent: str,
) -> str:
    if client is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        try:
            client.projects.download_checksums(slug, version, str(tmp_path))
            if tmp_path.exists() and tmp_path.stat().st_size > 0:
                return tmp_path.read_text(encoding="utf-8")
        except Exception as exc:
            LOGGER.warning("PhysioNet API checksum download failed, falling back to direct file: %s", exc)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    fallback_url = build_remote_file_url(dataset_root_url, checksums_file, with_download_query=True)
    return fetch_text_with_retries(
        url=fallback_url,
        timeout_seconds=timeout_seconds,
        retries=retries,
        retry_backoff_seconds=retry_backoff_seconds,
        user_agent=user_agent,
    )


def build_checksum_lookup(raw_checksums: dict[str, str], remote_root: str) -> dict[str, str]:
    lookup = dict(raw_checksums)
    normalized_root = remote_root.strip("/")
    for key, value in raw_checksums.items():
        if key.startswith(f"{normalized_root}/"):
            stripped = key[len(normalized_root) + 1 :]
            lookup[stripped] = value
    return lookup


def estimate_subset_size_from_global_total(
    total_size_bytes: float, total_patients: int, selected_patients: int
) -> dict[str, float]:
    if total_size_bytes <= 0 or total_patients <= 0 or selected_patients <= 0:
        return {}

    bytes_per_patient = total_size_bytes / float(total_patients)
    selected_bytes = bytes_per_patient * float(selected_patients)
    return {
        "global_total_bytes": total_size_bytes,
        "estimated_bytes_per_patient": bytes_per_patient,
        "estimated_selected_bytes": selected_bytes,
        "estimated_selected_gb": selected_bytes / float(1024**3),
    }


def run_download(config_general_path: Path, config_dataset_path: Path, dry_run_override: bool | None = None) -> int:
    config_general = load_yaml_config(config_general_path)
    config_dataset = load_yaml_config(config_dataset_path)
    config = deep_merge_dict(config_general, config_dataset)

    log_level = str(config.get("logging", {}).get("level", "INFO")).upper()
    logging.basicConfig(level=getattr(logging, log_level, logging.INFO), format="%(levelname)s: %(message)s")

    paths_cfg = config.get("paths", {})
    network_cfg = config.get("network", {})
    auth_env_cfg = config.get("auth_env", {})
    download_cfg = config.get("download", {})
    dataset_cfg = config.get("dataset", {})
    subset_cfg = config.get("subset", {})

    strategy = str(subset_cfg.get("strategy", "first_n_patients"))
    if strategy not in {"first_n_patients", "random_n_patients", "explicit_patient_ids"}:
        raise ValueError(
            "Supported subset.strategy values: first_n_patients, random_n_patients, explicit_patient_ids"
        )
    explicit_patient_ids = _coerce_explicit_patient_ids(subset_cfg.get("patient_ids"))
    if strategy == "explicit_patient_ids":
        if not explicit_patient_ids:
            raise ValueError("subset.patient_ids must contain at least one patient for explicit_patient_ids strategy")
        patient_count = len(explicit_patient_ids)
        requested_patient_count = patient_count
    else:
        patient_count = int(subset_cfg.get("patient_count", 1))
        if patient_count <= 0:
            raise ValueError("subset.patient_count must be positive")
        requested_patient_count = patient_count
    random_seed = int(subset_cfg.get("random_seed", 42))
    max_total_gb_raw = subset_cfg.get("max_total_gb")
    max_total_gb = float(max_total_gb_raw) if max_total_gb_raw is not None else None
    enforce_budget = bool(subset_cfg.get("enforce_budget", False))
    min_patient_count = int(subset_cfg.get("min_patient_count", 1))
    if min_patient_count <= 0:
        raise ValueError("subset.min_patient_count must be positive")
    include_file_globs = [str(x) for x in subset_cfg.get("include_file_globs", [])]
    fallback_extensions = [str(x) for x in subset_cfg.get("expand_record_extensions", [".hea", ".mat"])]
    max_files_per_patient_raw = subset_cfg.get("max_files_per_patient")
    max_files_per_patient = int(max_files_per_patient_raw) if max_files_per_patient_raw is not None else None
    if max_files_per_patient is not None and max_files_per_patient <= 0:
        raise ValueError("subset.max_files_per_patient must be positive when set")
    min_eeg_hours_raw = subset_cfg.get("min_eeg_hours_per_patient")
    min_eeg_hours_per_patient = float(min_eeg_hours_raw) if min_eeg_hours_raw is not None else 0.0
    if min_eeg_hours_per_patient < 0:
        raise ValueError("subset.min_eeg_hours_per_patient must be >= 0 when set")
    max_eeg_hours_raw = subset_cfg.get("max_eeg_hours_per_patient")
    max_eeg_hours_per_patient = float(max_eeg_hours_raw) if max_eeg_hours_raw is not None else None
    if max_eeg_hours_per_patient is not None and max_eeg_hours_per_patient <= 0:
        raise ValueError("subset.max_eeg_hours_per_patient must be > 0 when set")
    if max_eeg_hours_per_patient is None and min_eeg_hours_per_patient > 0:
        raise ValueError(
            "subset.min_eeg_hours_per_patient requires subset.max_eeg_hours_per_patient to also be set"
        )
    if (
        max_eeg_hours_per_patient is not None
        and min_eeg_hours_per_patient >= max_eeg_hours_per_patient
    ):
        raise ValueError("subset.min_eeg_hours_per_patient must be < subset.max_eeg_hours_per_patient when set")

    timeout_seconds = float(network_cfg.get("timeout_seconds", 60))
    retries = int(network_cfg.get("retries", 3))
    retry_backoff_seconds = float(network_cfg.get("retry_backoff_seconds", 2.0))
    chunk_size_bytes = int(network_cfg.get("chunk_size_bytes", 1_048_576))
    user_agent = str(network_cfg.get("user_agent", "NeuralManifoldDynamics-PhysioNetIngest/1.0"))

    slug = str(dataset_cfg.get("slug", "i-care"))
    version = str(dataset_cfg.get("version", "2.1"))
    base_url = str(dataset_cfg.get("base_url", "https://physionet.org"))
    remote_root = str(dataset_cfg.get("remote_root", f"files/{slug}/{version}"))
    records_file = str(dataset_cfg.get("records_file", "RECORDS"))
    checksums_file = str(dataset_cfg.get("checksums_file", "SHA256SUMS.txt"))
    dataset_root_url = build_dataset_root_url(base_url, remote_root)

    output_subdir = str(download_cfg.get("output_subdir", f"{slug}_{version}"))
    download_root = Path(str(paths_cfg.get("download_root", "E:/Science_Datasets/physionet/received")))
    metadata_root = Path(str(paths_cfg.get("metadata_root", "E:/Science_Datasets/physionet/metadata")))
    output_dir = download_root / output_subdir
    metadata_dir = metadata_root / output_subdir
    metadata_dir.mkdir(parents=True, exist_ok=True)

    overwrite = bool(download_cfg.get("overwrite", False))
    verify_checksum = bool(download_cfg.get("verify_checksum", True))
    verify_existing_size = bool(download_cfg.get("verify_existing_size", False))
    write_manifest_csv_flag = bool(download_cfg.get("write_manifest_csv", True))
    write_manifest_jsonl_flag = bool(download_cfg.get("write_manifest_jsonl", True))
    max_parallel_downloads_raw = download_cfg.get("max_parallel_downloads", 1)
    max_parallel_downloads = int(max_parallel_downloads_raw)
    if max_parallel_downloads <= 0:
        raise ValueError("download.max_parallel_downloads must be >= 1")
    dry_run = bool(download_cfg.get("dry_run", False))
    if dry_run_override is not None:
        dry_run = dry_run_override
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Dataset: %s v%s", slug, version)
    LOGGER.info("Subset strategy: %s, patient_count=%s", strategy, patient_count)
    if strategy == "random_n_patients":
        LOGGER.info("Random seed: %s", random_seed)
    if strategy == "explicit_patient_ids":
        LOGGER.info("Explicit patient IDs requested: %s", len(explicit_patient_ids))
    LOGGER.info(
        "Existing-file verification mode: checksum=%s, size_check=%s",
        verify_checksum,
        verify_existing_size,
    )
    if verify_checksum and verify_existing_size:
        LOGGER.info(
            "download.verify_checksum=true takes precedence over download.verify_existing_size for existing files."
        )
    LOGGER.info("Dry run: %s", dry_run)

    project_details: Any | None = None
    client = maybe_create_physionet_client(base_url, auth_env_cfg)
    if client is not None:
        try:
            project_details = client.projects.get_details(slug, version)
            LOGGER.info("PhysioNet project details fetched: %s", getattr(project_details, "title", slug))
        except Exception as exc:
            LOGGER.warning("Could not fetch project details via PhysioNet API client: %s", exc)

    records_cache_path = output_dir.joinpath(*PurePosixPath(normalize_relative_path(records_file)).parts)
    if records_cache_path.exists() and records_cache_path.stat().st_size > 0:
        LOGGER.info("Using cached records file: %s", records_cache_path)
        records_text = records_cache_path.read_text(encoding="utf-8")
    else:
        records_url = build_remote_file_url(dataset_root_url, records_file, with_download_query=True)
        records_text = fetch_text_with_retries(
            url=records_url,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            user_agent=user_agent,
        )
    record_paths = parse_records_text(records_text)
    patient_pool = collect_patients_in_order(record_paths)
    selected_patients, selected_record_paths = select_patients_by_strategy(
        record_paths=record_paths,
        strategy=strategy,
        patient_count=patient_count,
        random_seed=random_seed,
        explicit_patient_ids=explicit_patient_ids,
    )
    if not selected_patients:
        raise RuntimeError("Could not select any patients from RECORDS")

    budget_estimation: dict[str, float] = {}
    if max_total_gb is not None:
        total_size_bytes = 0.0
        if project_details is not None:
            try:
                total_size_bytes = float(getattr(project_details, "main_storage_size", 0.0) or 0.0)
            except Exception:
                total_size_bytes = 0.0
        budget_estimation = estimate_subset_size_from_global_total(
            total_size_bytes=total_size_bytes,
            total_patients=len(patient_pool),
            selected_patients=len(selected_patients),
        )
        if budget_estimation:
            LOGGER.info(
                "Estimated subset size: %.2f GB (budget %.2f GB).",
                budget_estimation["estimated_selected_gb"],
                max_total_gb,
            )
            if enforce_budget:
                budget_bytes = max_total_gb * float(1024**3)
                estimated_bytes_per_patient = budget_estimation["estimated_bytes_per_patient"]
                max_patients_by_budget = int(budget_bytes // estimated_bytes_per_patient)
                max_patients_by_budget = max(1, max_patients_by_budget)
                if len(selected_patients) > max_patients_by_budget:
                    if max_patients_by_budget < min_patient_count:
                        raise RuntimeError(
                            "Budget cap allows fewer patients than subset.min_patient_count. "
                            f"Allowed={max_patients_by_budget}, min={min_patient_count}"
                        )
                    LOGGER.warning(
                        "Budget cap enforces patient_count reduction from %s to %s.",
                        len(selected_patients),
                        max_patients_by_budget,
                    )
                    patient_count = max_patients_by_budget
                    selected_patients, selected_record_paths = select_patients_by_strategy(
                        record_paths=record_paths,
                        strategy=strategy,
                        patient_count=patient_count,
                        random_seed=random_seed,
                    )
                    budget_estimation = estimate_subset_size_from_global_total(
                        total_size_bytes=total_size_bytes,
                        total_patients=len(patient_pool),
                        selected_patients=len(selected_patients),
                    )
        else:
            LOGGER.warning(
                "Could not estimate subset size for budget cap (missing project size metadata)."
            )

    checksums_cache_path = output_dir.joinpath(*PurePosixPath(normalize_relative_path(checksums_file)).parts)
    if checksums_cache_path.exists() and checksums_cache_path.stat().st_size > 0:
        LOGGER.info("Using cached checksums file: %s", checksums_cache_path)
        checksums_text = checksums_cache_path.read_text(encoding="utf-8")
    else:
        checksums_text = fetch_checksums_text(
            client=client,
            dataset_root_url=dataset_root_url,
            slug=slug,
            version=version,
            checksums_file=checksums_file,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            user_agent=user_agent,
        )
    checksums_raw = parse_sha256sums_text(checksums_text)
    checksums = build_checksum_lookup(checksums_raw, remote_root=remote_root)
    top_level_files = [normalize_relative_path(str(x)) for x in subset_cfg.get("include_top_level_files", [])]
    selected_patient_files = expand_selected_records_to_files(
        selected_records=selected_record_paths,
        checksums_lookup=checksums,
        include_file_globs=include_file_globs,
        fallback_extensions=fallback_extensions,
        max_files_per_patient=max_files_per_patient,
    )
    eeg_hours_filter_summary: dict[str, Any] = {}
    if max_eeg_hours_per_patient is not None:
        selected_before = len(selected_patient_files)
        selected_patient_files, eeg_hours_filter_summary = limit_icare_eeg_files_to_first_hours(
            selected_paths=selected_patient_files,
            max_hours_per_patient=max_eeg_hours_per_patient,
            min_hours_per_patient=min_eeg_hours_per_patient,
            output_dir=output_dir,
            dataset_root_url=dataset_root_url,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_backoff_seconds=retry_backoff_seconds,
            user_agent=user_agent,
        )
        LOGGER.info(
            "Applied EEG hour window [%.2f, %.2f]; candidate patient files %s -> %s.",
            min_eeg_hours_per_patient,
            max_eeg_hours_per_patient,
            selected_before,
            len(selected_patient_files),
        )
    download_paths = dedupe_keep_order(selected_patient_files + top_level_files)
    LOGGER.info(
        "Selected %s patient(s), %s candidate file(s) after filters.",
        len(selected_patients),
        len(download_paths),
    )

    if client is not None:
        try:
            client.close()
        except Exception:
            pass

    write_lines(selected_patients, metadata_dir / "selected_patients.txt")
    write_lines(download_paths, metadata_dir / "planned_files.txt")

    manifest_rows: list[dict[str, Any]] = []
    error_count = 0
    if dry_run:
        for relative_path in download_paths:
            patient_id = extract_patient_id(relative_path) or ""
            expected_checksum = checksums.get(relative_path)
            local_path = output_dir.joinpath(*PurePosixPath(relative_path).parts)
            row: dict[str, Any] = {
                "remote_path": relative_path,
                "local_path": str(local_path),
                "patient_id": patient_id,
                "status": "planned",
                "bytes": 0,
                "checksum_expected": expected_checksum or "",
                "checksum_actual": "",
                "error": "",
            }
            manifest_rows.append(row)
    elif max_parallel_downloads == 1:
        for relative_path in download_paths:
            row = download_or_skip_relative_path(
                relative_path=relative_path,
                output_dir=output_dir,
                dataset_root_url=dataset_root_url,
                checksums=checksums,
                overwrite=overwrite,
                verify_checksum=verify_checksum,
                verify_existing_size=verify_existing_size,
                timeout_seconds=timeout_seconds,
                retries=retries,
                retry_backoff_seconds=retry_backoff_seconds,
                chunk_size_bytes=chunk_size_bytes,
                user_agent=user_agent,
            )
            manifest_rows.append(row)
            if row["status"] == "error":
                error_count += 1
    else:
        LOGGER.info("Parallel downloads enabled (max_workers=%s).", max_parallel_downloads)
        rows_by_index: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=max_parallel_downloads) as executor:
            future_to_index = {
                executor.submit(
                    download_or_skip_relative_path,
                    relative_path=relative_path,
                    output_dir=output_dir,
                    dataset_root_url=dataset_root_url,
                    checksums=checksums,
                    overwrite=overwrite,
                    verify_checksum=verify_checksum,
                    verify_existing_size=verify_existing_size,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                    chunk_size_bytes=chunk_size_bytes,
                    user_agent=user_agent,
                ): idx
                for idx, relative_path in enumerate(download_paths)
            }
            completed = 0
            total = len(download_paths)
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                row = future.result()
                rows_by_index[idx] = row
                if row["status"] == "error":
                    error_count += 1
                completed += 1
                if completed % 100 == 0 or completed == total:
                    LOGGER.info("Download progress: %s/%s complete (errors=%s)", completed, total, error_count)
        manifest_rows = [rows_by_index[idx] for idx in range(len(download_paths))]

    if write_manifest_jsonl_flag:
        write_manifest_jsonl(manifest_rows, metadata_dir / "download_manifest.jsonl")
    if write_manifest_csv_flag:
        write_manifest_csv(manifest_rows, metadata_dir / "download_manifest.csv")

    summary = {
        "dataset": {"slug": slug, "version": version, "remote_root": remote_root},
        "subset": {
            "strategy": strategy,
            "patient_count": len(selected_patients),
            "patient_count_requested": requested_patient_count,
            "patient_pool_size": len(patient_pool),
            "selected_patients": selected_patients,
            "records_total": len(record_paths),
            "records_selected": len(selected_record_paths),
            "files_planned": len(download_paths),
            "random_seed": random_seed if strategy == "random_n_patients" else None,
            "explicit_patient_ids_requested_count": (
                len(explicit_patient_ids) if strategy == "explicit_patient_ids" else None
            ),
            "explicit_patient_ids_requested_preview": (
                explicit_patient_ids[:20] if strategy == "explicit_patient_ids" else None
            ),
            "max_total_gb": max_total_gb,
            "enforce_budget": enforce_budget,
            "min_patient_count": min_patient_count,
            "min_eeg_hours_per_patient": min_eeg_hours_per_patient,
            "max_eeg_hours_per_patient": max_eeg_hours_per_patient,
            "eeg_hours_filter": eeg_hours_filter_summary,
            "budget_estimation": budget_estimation,
        },
        "execution": {
            "dry_run": dry_run,
            "max_parallel_downloads": max_parallel_downloads,
            "verify_checksum": verify_checksum,
            "verify_existing_size": verify_existing_size,
            "output_dir": str(output_dir),
            "metadata_dir": str(metadata_dir),
            "errors": error_count,
        },
    }
    (metadata_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")

    if dry_run:
        LOGGER.info("Dry run complete. Planned files: %s", len(download_paths))
        return 0
    LOGGER.info("Run complete. Files in manifest: %s, errors: %s", len(manifest_rows), error_count)
    return 0 if error_count == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    script_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Download a subset of PhysioNet I-CARE records.")
    parser.add_argument(
        "--config-general",
        type=Path,
        default=script_root / "config" / "config_ingest.yml",
        help="Path to generic PhysioNet ingest config.",
    )
    parser.add_argument(
        "--config-dataset",
        type=Path,
        default=script_root / "config" / "config_i-care_2_1.yml",
        help="Path to dataset-specific ingest config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Force dry-run regardless of YAML setting.",
    )
    parser.add_argument(
        "--no-dry-run",
        action="store_true",
        help="Force real download regardless of YAML setting.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.dry_run and args.no_dry_run:
        parser.error("Use only one of --dry-run or --no-dry-run")

    dry_run_override: bool | None = None
    if args.dry_run:
        dry_run_override = True
    elif args.no_dry_run:
        dry_run_override = False

    return run_download(
        config_general_path=args.config_general,
        config_dataset_path=args.config_dataset,
        dry_run_override=dry_run_override,
    )


if __name__ == "__main__":
    raise SystemExit(main())
