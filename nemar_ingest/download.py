"""Manifest- and CLI-backed NEMAR downloads with BIDS-aware selection."""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from .client import DownloadError, NemarClient, NemarError, validate_dataset_id, validate_version, validate_relative_path
from .contracts import DatasetDownload, ManifestEntry

logger = logging.getLogger(__name__)


class NemarCLIUnavailable(NemarError):
    """Raised only when the configured NEMAR CLI executable is absent."""


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    return value if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise ValueError("nemar_selection_values_must_be_strings")


def _strip_entity_prefix(value: str, prefix: str) -> str:
    text = str(value).strip()
    return text[len(prefix):] if text.lower().startswith(prefix.lower()) else text


def _path_entities(path: str) -> dict[str, str]:
    entities: dict[str, str] = {}
    for token in PurePosixPath(path).name.split("_"):
        if "-" not in token:
            continue
        key, value = token.split("-", 1)
        if key in {"sub", "ses", "task", "run", "acq", "datatype"}:
            entities[key] = value.split(".", 1)[0]
    parts = PurePosixPath(path).parts
    for part in parts:
        if part.startswith("sub-"):
            entities.setdefault("sub", part[4:])
        elif part.startswith("ses-"):
            entities.setdefault("ses", part[4:])
        elif part in {"eeg", "ieeg", "meg", "func", "anat", "beh", "emg", "motion"}:
            entities.setdefault("datatype", part)
    return entities


def _matches_value(actual: str | None, requested: Sequence[str], prefix: str) -> bool:
    if not requested:
        return True
    if actual is None:
        return False
    normalized = actual.lower()
    return any(_strip_entity_prefix(item, prefix).lower() == normalized for item in requested)


def _is_dataset_root_metadata(path: str) -> bool:
    """Keep BIDS root bookkeeping files with entity-filtered selections."""
    name = PurePosixPath(path).name.lower()
    return PurePosixPath(path).parent == PurePosixPath(".") and (
        name in {"dataset_description.json", "participants.tsv", "participants.json", "changes", "license"}
        or name.startswith("readme")
    )


def select_manifest_entries(entries: Sequence[ManifestEntry], selection: Mapping[str, Any]) -> tuple[ManifestEntry, ...]:
    """Apply explicit path globs and BIDS entity filters to a manifest."""
    includes = _string_list(selection.get("include", selection.get("include_patterns")))
    excludes = _string_list(selection.get("exclude", selection.get("exclude_patterns")))
    subjects = _string_list(selection.get("subjects", selection.get("subject")))
    sessions = _string_list(selection.get("sessions", selection.get("session")))
    tasks = _string_list(selection.get("tasks", selection.get("task")))
    runs = _string_list(selection.get("runs", selection.get("run")))
    datatypes = _string_list(selection.get("datatypes", selection.get("datatype")))

    def glob_matches(path: str, pattern: str) -> bool:
        normalized = pattern.replace("\\", "/")
        return fnmatch.fnmatchcase(path, normalized) or PurePosixPath(path).match(normalized)

    selected: list[ManifestEntry] = []
    for entry in entries:
        path = validate_relative_path(entry.path)
        # NEMAR's BIDS download contract always retains root bookkeeping files,
        # even when include/exclude or entity filters target a subset.
        if not _is_dataset_root_metadata(path):
            if includes and not any(glob_matches(path, pattern) for pattern in includes):
                continue
            if excludes and any(glob_matches(path, pattern) for pattern in excludes):
                continue
            entities = _path_entities(path)
            if not _matches_value(entities.get("sub"), subjects, "sub-"):
                continue
            if not _matches_value(entities.get("ses"), sessions, "ses-"):
                continue
            if not _matches_value(entities.get("task"), tasks, "task-"):
                continue
            if not _matches_value(entities.get("run"), runs, "run-"):
                continue
            if datatypes and not _matches_value(entities.get("datatype"), datatypes, "datatype-"):
                continue
        selected.append(entry)
    if not selected:
        raise NemarError("nemar_selection_empty")
    return tuple(selected)


def _resolve_cli(command: str | Sequence[str] | None) -> list[str] | None:
    if command is None:
        return None
    if isinstance(command, str):
        command = [command]
    prefix = [str(part) for part in command]
    if not prefix:
        return None
    executable = prefix[0]
    resolved = shutil.which(executable)
    if resolved:
        prefix[0] = resolved
        return prefix
    return None


def _cli_download(dataset_id: str, root: Path, version: str, selection: Mapping[str, Any], download_config: Mapping[str, Any]) -> Path:
    command = download_config.get("cli_command", "nemar")
    prefix = _resolve_cli(command)
    if prefix is None:
        raise NemarCLIUnavailable(f"nemar_cli_not_found:{command}")
    cmd = prefix + ["dataset", "download", validate_dataset_id(dataset_id)]
    # The current CLI uses --version for an explicit release; omit it for
    # latest so the server resolves the current release as documented.
    if version != "latest":
        cmd.extend(["--version", validate_version(version)])
    cmd.extend(["--output", str(root)])
    flag_map = {
        "subjects": "--subjects", "sessions": "--sessions", "tasks": "--tasks",
        "runs": "--runs", "datatypes": "--datatypes", "include": "--include", "exclude": "--exclude",
    }
    for key, flag in flag_map.items():
        for value in _string_list(selection.get(key)):
            cmd.extend([flag, value])
    if bool(download_config.get("stimuli", False)):
        cmd.append("--stimuli")
    if bool(download_config.get("derivatives", False)):
        cmd.append("--derivatives")
    logger.info("Using NEMAR CLI: %s", " ".join(cmd))
    child_env = os.environ.copy()
    subprocess.run(cmd, check=True, cwd=str(root.parent), env=child_env)
    return root


_RECEIPT_NAME = "acquisition_receipt.json"


def _receipt_path(root: Path) -> Path:
    return root / _RECEIPT_NAME


def _guard_existing_receipt(root: Path, dataset_id: str, version: str) -> None:
    receipt = _receipt_path(root)
    if not receipt.exists():
        return
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NemarError(f"nemar_receipt_invalid:{receipt}") from exc
    if payload.get("dataset_id") != dataset_id or payload.get("version") != version:
        raise NemarError(
            f"nemar_receipt_version_mismatch:{receipt}:"
            f"existing={payload.get('dataset_id')}@{payload.get('version')}:"
            f"requested={dataset_id}@{version}"
        )


def _write_receipt(
    root: Path,
    *,
    dataset_id: str,
    version: str,
    manifest_url: str,
    manifest_digest: str | None,
    selected: Sequence[ManifestEntry],
    failed: Sequence[tuple[str, str]] = (),
) -> None:
    payload = {
        "schema_version": 1,
        "dataset_id": dataset_id,
        "version": version,
        "manifest_url": manifest_url,
        "manifest_sha256": manifest_digest,
        "failed_entries": [{"path": path, "error": error} for path, error in failed],
        "selected_entries": [
            {
                "path": entry.path,
                "url": entry.url,
                "size_bytes": entry.size,
                "sha256": entry.sha256,
                "checksum": entry.checksum if entry.checksum is not None else entry.sha256,
                "checksum_algorithm": entry.checksum_algorithm or ("sha256" if entry.sha256 else None),
            }
            for entry in selected
        ],
    }
    root.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".acquisition_receipt.", suffix=".tmp",
            dir=root, delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, _receipt_path(root))
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def download_dataset(
    dataset_id: str,
    config: Mapping[str, Any],
    out_dir: Path,
    *,
    client: NemarClient | None = None,
) -> DatasetDownload:
    """Download one configured dataset and return auditable selection metadata."""
    dataset = validate_dataset_id(dataset_id)
    dataset_cfg = _mapping(config, "dataset")
    download_cfg = _mapping(config, "download")
    selection = _mapping(config, "selection")
    version = validate_version(str(dataset_cfg.get("version", download_cfg.get("version", "latest"))))
    root = Path(out_dir) / dataset
    root.mkdir(parents=True, exist_ok=True)
    backend = str(download_cfg.get("backend", "manifest")).strip().lower()

    _guard_existing_receipt(root, dataset, version)

    active_client = client or NemarClient(data_base_url=str(download_cfg.get("data_base_url", "https://data.nemar.org")))
    manifest_url, entries, manifest_digest = active_client.get_manifest_with_digest(dataset, version)
    selected = select_manifest_entries(entries, selection)

    if backend in {"cli", "auto"}:
        try:
            _cli_download(dataset, root, version, selection, download_cfg)
            missing = [entry.path for entry in selected if not (root / Path(*entry.path.split("/"))).is_file()]
            if missing:
                raise NemarError(f"nemar_cli_selection_incomplete:{missing[0]}")
            _write_receipt(root, dataset_id=dataset, version=version, manifest_url=manifest_url, manifest_digest=manifest_digest, selected=selected)
            return DatasetDownload(dataset, version, str(root), tuple(entry.path for entry in selected), manifest_url)
        except NemarCLIUnavailable:
            if backend == "cli":
                raise
            logger.info("NEMAR CLI unavailable; falling back to manifest downloader")

    if backend not in {"manifest", "auto"}:
        raise ValueError(f"unsupported_nemar_backend:{backend}")
    retries = max(1, int(download_cfg.get("retries", 3)))
    verify_checksum = bool(download_cfg.get("verify_checksum", True))
    verify_size = bool(download_cfg.get("verify_size", True))
    continue_on_error = bool(download_cfg.get("continue_on_error", True))
    failed: list[tuple[str, str]] = []
    for entry in selected:
        try:
            active_client.download_entry(entry, root, verify_checksum=verify_checksum, verify_size=verify_size, retries=retries)
        except DownloadError as exc:
            if not continue_on_error:
                raise
            logger.error("NEMAR file failed after retries; continuing: %s", exc)
            failed.append((entry.path, str(exc)))
    _write_receipt(
        root,
        dataset_id=dataset,
        version=version,
        manifest_url=manifest_url,
        manifest_digest=manifest_digest,
        selected=selected,
        failed=failed,
    )
    return DatasetDownload(
        dataset,
        version,
        str(root),
        tuple(entry.path for entry in selected),
        manifest_url,
        failed_files=tuple(path for path, _error in failed),
    )


def download_datasets(dataset_ids: Iterable[str], config: Mapping[str, Any], out_dir: Path) -> dict[str, DatasetDownload]:
    results: dict[str, DatasetDownload] = {}
    for dataset_id in dataset_ids:
        result = download_dataset(dataset_id, config, out_dir)
        results[result.dataset_id] = result
    return results
