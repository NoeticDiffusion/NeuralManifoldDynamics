"""Select Zenodo files and download them into a received directory."""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from zenodo_ingest.client import DownloadError, ZenodoClient, ZenodoError, clamp_workers, local_path_for_key
from zenodo_ingest.contracts import DownloadOptions, FileEntry, FileTransfer, RecordDownload, ZenodoRecord

logger = logging.getLogger(__name__)

_RECEIPT_NAME = "acquisition_receipt.json"


def select_files(entries: Sequence[FileEntry], selection: Mapping[str, Any]) -> tuple[FileEntry, ...]:
    """Apply include and exclude globs. An empty selection keeps every file."""
    includes = _string_list(selection.get("include", selection.get("include_patterns")))
    excludes = _string_list(selection.get("exclude", selection.get("exclude_patterns")))
    names = _string_list(selection.get("files", selection.get("keys")))
    selected: list[FileEntry] = []
    for entry in entries:
        basename = PurePosixPath(entry.key).name
        if names and entry.key not in names and basename not in names:
            continue
        if includes and not any(_glob_match(entry.key, basename, pattern) for pattern in includes):
            continue
        if excludes and any(_glob_match(entry.key, basename, pattern) for pattern in excludes):
            continue
        selected.append(entry)
    if not selected:
        raise ZenodoError("zenodo_selection_empty")
    return tuple(selected)


def build_options(
    download_cfg: Mapping[str, Any],
    *,
    workers: int | None = None,
    overwrite: bool | None = None,
) -> DownloadOptions:
    worker_value = workers if workers is not None else download_cfg.get("workers", 2)
    return DownloadOptions(
        chunk_bytes=_positive_int(download_cfg.get("chunk_bytes", 8 * 1024 * 1024), "chunk_bytes"),
        retries=_positive_int(download_cfg.get("retries", 5), "retries"),
        retry_backoff_seconds=_nonnegative_float(
            download_cfg.get("retry_backoff_seconds", 2.0),
            "retry_backoff_seconds",
        ),
        timeout_seconds=_positive_float(download_cfg.get("timeout_seconds", 120), "timeout_seconds"),
        resume=bool(download_cfg.get("resume", True)),
        verify_checksum=bool(download_cfg.get("verify_checksum", True)),
        overwrite=bool(download_cfg.get("overwrite", False) if overwrite is None else overwrite),
        progress_bytes=_nonnegative_int(download_cfg.get("progress_bytes", 256 * 1024 * 1024), "progress_bytes"),
        workers=clamp_workers(_positive_int(worker_value, "workers")),
        continue_on_error=bool(download_cfg.get("continue_on_error", True)),
    )


def download_records(
    record_refs: Iterable[str],
    config: Mapping[str, Any],
    out_dir: Path,
    *,
    client: ZenodoClient | None = None,
    dry_run: bool = False,
) -> dict[str, RecordDownload]:
    results: dict[str, RecordDownload] = {}
    active = client or client_from_config(config)
    for record_ref in record_refs:
        result = download_record(record_ref, config, out_dir, client=active, dry_run=dry_run)
        results[result.record_id] = result
    return results


def download_record(
    record_ref: str,
    config: Mapping[str, Any],
    out_dir: Path,
    *,
    client: ZenodoClient | None = None,
    dry_run: bool = False,
) -> RecordDownload:
    """Download one record under ``out_dir/<resolved record id>``."""
    active = client or client_from_config(config)
    selection = _mapping(config, "selection")
    options = build_options(_mapping(config, "download"))
    record = active.get_record(record_ref)
    selected = select_files(record.files, selection)
    root = Path(out_dir) / record.record_id
    if dry_run:
        return RecordDownload(
            record_id=record.record_id,
            root=str(root),
            title=record.title,
            doi=record.doi,
            files=tuple(_planned(entry, root) for entry in selected),
            dry_run=True,
        )

    root.mkdir(parents=True, exist_ok=True)
    _guard_existing_receipt(root, record.record_id)
    trusted = _trusted_digests(_read_receipt(root), record.record_id)
    completed: list[FileTransfer] = []
    try:
        if options.workers == 1 or not options.continue_on_error:
            for entry in selected:
                completed.append(_download_one(active, entry, root, options, trusted))
        else:
            completed.extend(_download_parallel(active, selected, root, options, trusted))
    finally:
        if completed:
            _write_receipt(root, record, completed)
    failed = tuple(item.key for item in completed if item.status == "error")
    return RecordDownload(
        record_id=record.record_id,
        root=str(root),
        title=record.title,
        doi=record.doi,
        files=tuple(completed),
        failed_files=failed,
    )


def client_from_config(config: Mapping[str, Any]) -> ZenodoClient:
    download_cfg = _mapping(config, "download")
    token_env = str(download_cfg.get("token_env") or "ZENODO_TOKEN")
    token = os.environ.get(token_env, "")
    return ZenodoClient(
        api_base=str(download_cfg.get("api_base") or "https://zenodo.org/api"),
        token=token,
        timeout=float(download_cfg.get("timeout_seconds") or 120),
    )


def configured_records(config: Mapping[str, Any], cli_records: Sequence[str] | None) -> list[str]:
    if cli_records:
        return [str(item).strip() for item in cli_records if str(item).strip()]
    single = config.get("record")
    if single not in (None, ""):
        return [str(single).strip()]
    records = config.get("records") or []
    if isinstance(records, str):
        records = [records]
    if not isinstance(records, Sequence):
        raise ZenodoError("zenodo_records_must_be_a_list")
    return [str(item).strip() for item in records if str(item).strip()]


def _download_one(
    client: ZenodoClient,
    entry: FileEntry,
    root: Path,
    options: DownloadOptions,
    trusted: Mapping[str, str],
) -> FileTransfer:
    try:
        return client.download_file(
            entry,
            root,
            options,
            trusted_digest=trusted.get(entry.key),
        )
    except DownloadError as exc:
        if not options.continue_on_error:
            raise
        logger.error("Zenodo file failed after retries; continuing: %s", exc)
        return FileTransfer(
            key=entry.key,
            local_path=str(local_path_for_key(root, entry.key)),
            size=entry.size,
            checksum=entry.checksum,
            status="error",
            error=str(exc),
        )


def _download_parallel(
    client: ZenodoClient,
    selected: Sequence[FileEntry],
    root: Path,
    options: DownloadOptions,
    trusted: Mapping[str, str],
) -> list[FileTransfer]:
    from concurrent.futures import ThreadPoolExecutor

    workers = max(1, options.workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_download_one, client, entry, root, options, trusted)
            for entry in selected
        ]
        return [future.result() for future in futures]


def _planned(entry: FileEntry, root: Path) -> FileTransfer:
    return FileTransfer(
        key=entry.key,
        local_path=str(root / Path(*PurePosixPath(entry.key).parts)),
        size=entry.size,
        checksum=entry.checksum,
        status="planned",
    )


def _glob_match(key: str, basename: str, pattern: str) -> bool:
    normalized = pattern.replace("\\", "/")
    return fnmatch.fnmatchcase(key, normalized) or fnmatch.fnmatchcase(basename, normalized)


def _string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        return (text,) if text else ()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise ZenodoError("zenodo_selection_values_must_be_strings")


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ZenodoError(f"zenodo_config_{key}_must_be_a_mapping")
    return value


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ZenodoError(f"invalid_{name}:{value!r}") from exc
    if parsed <= 0:
        raise ZenodoError(f"invalid_{name}:{value!r}")
    return parsed


def _nonnegative_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ZenodoError(f"invalid_{name}:{value!r}") from exc
    if parsed < 0:
        raise ZenodoError(f"invalid_{name}:{value!r}")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ZenodoError(f"invalid_{name}:{value!r}") from exc
    if parsed <= 0:
        raise ZenodoError(f"invalid_{name}:{value!r}")
    return parsed


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ZenodoError(f"invalid_{name}:{value!r}") from exc
    if parsed < 0:
        raise ZenodoError(f"invalid_{name}:{value!r}")
    return parsed


def _receipt_path(root: Path) -> Path:
    return root / _RECEIPT_NAME


def _read_receipt(root: Path) -> Mapping[str, Any] | None:
    path = _receipt_path(root)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ZenodoError(f"zenodo_receipt_invalid:{path}") from exc
    if not isinstance(payload, Mapping):
        raise ZenodoError(f"zenodo_receipt_invalid:{path}")
    return payload


def _guard_existing_receipt(root: Path, record_id: str) -> None:
    payload = _read_receipt(root)
    if payload is None:
        return
    existing = str(payload.get("record_id") or "")
    if existing and existing != record_id:
        raise ZenodoError(
            f"zenodo_receipt_record_mismatch:{_receipt_path(root)}:"
            f"existing={existing}:requested={record_id}"
        )


def _trusted_digests(payload: Mapping[str, Any] | None, record_id: str) -> dict[str, str]:
    if not payload or str(payload.get("record_id") or "") != record_id:
        return {}
    files = payload.get("files")
    if not isinstance(files, list):
        return {}
    trusted: dict[str, str] = {}
    for item in files:
        if not isinstance(item, Mapping):
            continue
        if item.get("status") not in {"downloaded", "resumed", "skipped_verified"}:
            continue
        if not item.get("checksum_verified"):
            continue
        key = item.get("key")
        checksum = item.get("checksum")
        if isinstance(key, str) and isinstance(checksum, str) and ":" in checksum:
            trusted[key] = checksum.split(":", 1)[1]
    return trusted


def _file_payload(item: FileTransfer) -> dict[str, Any]:
    return {
        "key": item.key,
        "local_path": item.local_path,
        "size": item.size,
        "checksum": item.checksum,
        "status": item.status,
        "checksum_verified": item.checksum_verified,
        "error": item.error,
    }


def _write_receipt(root: Path, record: ZenodoRecord, transfers: Sequence[FileTransfer]) -> None:
    """Write the receipt, keeping verified rows for files this attempt did not touch."""
    previous_files: list[Mapping[str, Any]] = []
    existing = _read_receipt(root)
    if existing and str(existing.get("record_id") or "") == record.record_id:
        raw_files = existing.get("files")
        if isinstance(raw_files, list):
            previous_files = [item for item in raw_files if isinstance(item, Mapping)]
    by_key: dict[str, Mapping[str, Any]] = {}
    for item in previous_files:
        key = item.get("key")
        if isinstance(key, str):
            by_key[key] = item
    for item in transfers:
        by_key[item.key] = _file_payload(item)
    ordered: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for item in transfers:
        ordered.append(by_key[item.key])
        seen.add(item.key)
    for item in previous_files:
        key = item.get("key")
        if isinstance(key, str) and key not in seen:
            ordered.append(by_key[key])
            seen.add(key)
    payload = {
        "schema_version": 1,
        "source": "zenodo",
        "record_id": record.record_id,
        "title": record.title,
        "doi": record.doi,
        "concept_doi": record.concept_doi,
        "api_url": record.api_url,
        "files": ordered,
    }
    root.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".acquisition_receipt.",
            suffix=".tmp",
            dir=root,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, _receipt_path(root))
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
