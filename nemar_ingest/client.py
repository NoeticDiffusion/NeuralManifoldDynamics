"""Public NEMAR data-plane client.

NEMAR publishes a versioned BIDS tree at ``data.nemar.org``.  This module
intentionally uses the stable data-plane contract rather than scraping the
dataset page or depending on the JavaScript CLI.  The manifest is fetched
first, paths are validated before touching disk, and each payload is written
atomically with optional size and checksum verification. NEMAR publishes
SHA-256 for annexed payloads and git blob SHA-1 (``checksum_algorithm: git``)
for files stored directly in the git tree.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

from .contracts import ManifestEntry

logger = logging.getLogger(__name__)

_DATASET_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_VERSION_RE = re.compile(r"^(?:latest|v[0-9]+\.[0-9]+\.[0-9]+)$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class NemarError(RuntimeError):
    """Base exception for NEMAR acquisition failures."""


class ManifestError(NemarError):
    """Raised when a release manifest is malformed or unsafe."""


class DownloadError(NemarError):
    """Raised when a file cannot be downloaded or fails integrity checks."""


def validate_dataset_id(dataset_id: str) -> str:
    value = str(dataset_id).strip()
    if not _DATASET_RE.fullmatch(value):
        raise ValueError(f"invalid_nemar_dataset_id:{dataset_id!r}")
    return value


def validate_version(version: str) -> str:
    value = str(version).strip()
    if not _VERSION_RE.fullmatch(value):
        raise ValueError(
            f"invalid_nemar_version:{version!r}; expected latest or vX.Y.Z"
        )
    return value


def validate_relative_path(path: str) -> str:
    """Return a normalized BIDS-relative path, rejecting traversal."""
    raw = str(path).replace("\\", "/").strip()
    if not raw or "\x00" in raw:
        raise ManifestError(f"invalid_manifest_path:{path!r}")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ManifestError(f"manifest_path_traversal:{path!r}")
    normalized = str(candidate)
    if normalized in {".", ""} or normalized.startswith("/"):
        raise ManifestError(f"invalid_manifest_path:{path!r}")
    return normalized


def _coerce_size(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"invalid_manifest_size:{value!r}") from exc
    if parsed < 0:
        raise ManifestError(f"invalid_manifest_size:{value!r}")
    return parsed


def _normalize_sha256(value: Any) -> str:
    parsed = str(value).strip().lower()
    if parsed.startswith("sha256:"):
        parsed = parsed[7:]
    if not _SHA256_RE.fullmatch(parsed):
        raise ManifestError(f"invalid_manifest_sha256:{value!r}")
    return parsed


def _parse_checksum(item: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return ``(algorithm, digest)`` for one manifest record.

    ``git`` is the SHA-1 of the git blob header plus file bytes, not a raw
    SHA-1 of the payload. Bare 40-character digests without an algorithm are
    rejected so a SHA-1 is never checked as SHA-256.
    """
    raw_algorithm = item.get("checksum_algorithm")
    if raw_algorithm not in (None, ""):
        algorithm = str(raw_algorithm).strip().lower()
        raw_value = item.get("checksum", item.get("sha256"))
        if raw_value in (None, ""):
            raise ManifestError("manifest_checksum_missing")
        if algorithm == "sha256":
            return "sha256", _normalize_sha256(raw_value)
        if algorithm == "git":
            digest = str(raw_value).strip().lower()
            if not _GIT_SHA1_RE.fullmatch(digest):
                raise ManifestError(f"invalid_manifest_git_checksum:{raw_value!r}")
            return "git", digest
        raise ManifestError(f"unsupported_manifest_checksum_algorithm:{raw_algorithm!r}")

    raw_value = item.get("sha256", item.get("checksum"))
    if raw_value in (None, ""):
        return None, None
    return "sha256", _normalize_sha256(raw_value)


def _entry_from_mapping(item: Mapping[str, Any], *, base_url: str) -> ManifestEntry:
    path_value = item.get("path") or item.get("name") or item.get("key")
    if not isinstance(path_value, str):
        raise ManifestError("manifest_entry_missing_path")
    path = validate_relative_path(path_value)

    # Current NEMAR manifests expose bytes_url.  Accept stable data_url/url as
    # well so older releases and test fixtures remain readable.
    raw_url = item.get("bytes_url") or item.get("data_url") or item.get("url")
    url = str(raw_url).strip() if raw_url else None
    if url:
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ManifestError(f"manifest_url_not_https:{url!r}")
    else:
        url = f"{base_url.rstrip('/')}/{quote(path, safe='/') }"

    algorithm, digest = _parse_checksum(item)
    return ManifestEntry(
        path=path,
        url=url,
        size=_coerce_size(item.get("size_bytes", item.get("size", item.get("bytes")))),
        sha256=digest if algorithm == "sha256" else None,
        metadata=dict(item),
        checksum=digest,
        checksum_algorithm=algorithm,
    )


def parse_manifest(payload: Any, *, base_url: str) -> tuple[ManifestEntry, ...]:
    """Parse list- and wrapper-shaped NEMAR manifests into safe entries."""
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, Mapping):
        records = payload.get("files") or payload.get("entries") or payload.get("manifest")
        if records is None and isinstance(payload.get("data"), list):
            records = payload["data"]
    else:
        records = None
    if not isinstance(records, list):
        raise ManifestError("manifest_expected_file_list")

    entries: list[ManifestEntry] = []
    seen: set[str] = set()
    for item in records:
        if not isinstance(item, Mapping):
            raise ManifestError("manifest_entry_expected_object")
        entry = _entry_from_mapping(item, base_url=base_url)
        if entry.path in seen:
            raise ManifestError(f"manifest_duplicate_path:{entry.path}")
        seen.add(entry.path)
        entries.append(entry)
    return tuple(entries)


class NemarClient:
    """Minimal client for the anonymous NEMAR data API."""

    def __init__(
        self,
        *,
        data_base_url: str = "https://data.nemar.org",
        opener: Callable[..., Any] | None = None,
        timeout: float = 60.0,
    ) -> None:
        parsed = urlparse(data_base_url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("data_base_url_must_be_https")
        self.data_base_url = data_base_url.rstrip("/")
        self._opener = opener or urlopen
        self.timeout = float(timeout)

    def manifest_url(self, dataset_id: str, version: str = "latest") -> str:
        dataset = validate_dataset_id(dataset_id)
        release = validate_version(version)
        return f"{self.data_base_url}/{quote(dataset, safe='')}/{quote(release, safe='')}/manifest.json"

    def file_url(self, dataset_id: str, version: str, path: str) -> str:
        dataset = validate_dataset_id(dataset_id)
        release = validate_version(version)
        relative = validate_relative_path(path)
        return f"{self.data_base_url}/{quote(dataset, safe='')}/{quote(release, safe='')}/{quote(relative, safe='/')}"

    def _get_json(self, url: str) -> Any:
        payload, _ = self._get_json_with_digest(url)
        return payload

    def _get_json_with_digest(self, url: str) -> tuple[Any, str]:
        try:
            response = self._opener(Request(url, headers={"Accept": "application/json", "User-Agent": "NeuralManifoldDynamics/nemar-ingest"}), timeout=self.timeout)
            with response:
                raw = response.read()
            return json.loads(raw), hashlib.sha256(raw).hexdigest()
        except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            raise NemarError(f"nemar_manifest_fetch_failed:{url}:{exc}") from exc

    def get_manifest(self, dataset_id: str, version: str = "latest") -> tuple[str, tuple[ManifestEntry, ...]]:
        url, entries, _ = self.get_manifest_with_digest(dataset_id, version)
        return url, entries

    def get_manifest_with_digest(
        self, dataset_id: str, version: str = "latest"
    ) -> tuple[str, tuple[ManifestEntry, ...], str]:
        url = self.manifest_url(dataset_id, version)
        payload, digest = self._get_json_with_digest(url)
        return url, parse_manifest(payload, base_url=f"{self.data_base_url}/{dataset_id}/{version}"), digest

    def download_entry(
        self,
        entry: ManifestEntry,
        destination_root: Path,
        *,
        verify_checksum: bool = True,
        verify_size: bool = True,
        retries: int = 3,
    ) -> Path:
        relative = validate_relative_path(entry.path)
        root = Path(destination_root).resolve()
        target = (root / Path(*PurePosixPath(relative).parts)).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise DownloadError(f"destination_path_traversal:{entry.path}") from exc
        if not entry.url:
            raise DownloadError(f"manifest_entry_missing_url:{entry.path}")

        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.is_file():
            if self._matches_integrity(target, entry, verify_checksum=verify_checksum, verify_size=verify_size):
                return target

        last_error: Exception | None = None
        for attempt in range(max(1, int(retries))):
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=f".{target.name}.", suffix=".part", dir=target.parent, delete=False
                ) as handle:
                    temporary = Path(handle.name)
                    response = self._opener(Request(entry.url, headers={"User-Agent": "NeuralManifoldDynamics/nemar-ingest"}), timeout=self.timeout)
                    with response:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                if not self._matches_integrity(temporary, entry, verify_checksum=verify_checksum, verify_size=verify_size):
                    raise DownloadError(f"integrity_check_failed:{entry.path}")
                os.replace(temporary, target)
                temporary = None
                return target
            except (HTTPError, URLError, TimeoutError, OSError, DownloadError) as exc:
                last_error = exc
                logger.warning("NEMAR file download attempt %s/%s failed for %s: %s", attempt + 1, retries, entry.path, exc)
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
        raise DownloadError(f"nemar_file_download_failed:{entry.path}:{last_error}") from last_error

    @staticmethod
    def _matches_integrity(path: Path, entry: ManifestEntry, *, verify_checksum: bool, verify_size: bool) -> bool:
        if verify_size and entry.size is not None and path.stat().st_size != entry.size:
            return False
        if verify_checksum:
            algorithm, expected = _checksum_spec(entry)
            if algorithm is None and expected is None and (entry.checksum or entry.checksum_algorithm or entry.sha256):
                return False
            if expected:
                return _file_checksum(path, algorithm) == expected
        return True


def _checksum_spec(entry: ManifestEntry) -> tuple[str | None, str | None]:
    """Resolve the digest that ``verify_checksum`` must reproduce."""
    algorithm = (entry.checksum_algorithm or "").strip().lower()
    if algorithm == "git":
        return ("git", entry.checksum.lower()) if entry.checksum else (None, None)
    if algorithm == "sha256":
        digest = entry.checksum or entry.sha256
        return ("sha256", digest.lower()) if digest else (None, None)
    if not algorithm and entry.sha256:
        return "sha256", entry.sha256.lower()
    return None, None


def _file_checksum(path: Path, algorithm: str | None) -> str:
    if algorithm == "sha256":
        digest = hashlib.sha256()
    elif algorithm == "git":
        # Git blob id: SHA-1("blob {nbytes}\\0" + bytes). usedforsecurity=False
        # marks this as a published content id, not a signature.
        digest = hashlib.sha1(usedforsecurity=False)
        digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    else:
        raise DownloadError(f"unsupported_checksum_algorithm:{algorithm}")
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
