"""Zenodo records API client with streamed, resumable file transfer.

The records API is limited to ``zenodo.org`` and ``sandbox.zenodo.org``.
File bytes are written to a sibling ``.partial`` and renamed only after the
advertised size matches and, when requested, the published checksum matches.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from zenodo_ingest.contracts import DownloadOptions, FileEntry, FileTransfer, ZenodoRecord

logger = logging.getLogger(__name__)

USER_AGENT = "NeuralManifoldDynamics/zenodo-ingest"
ALLOWED_API_HOSTS = frozenset({"zenodo.org", "sandbox.zenodo.org"})
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_CHECKSUM_ALGORITHMS = frozenset({"md5", "sha1", "sha256"})
_NUMERIC_ID = re.compile(r"^\d{1,12}$")
_DOI_REF = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/)?10\.5281/zenodo\.(\d{1,12})/?$",
    re.IGNORECASE,
)
_RECORD_URL = re.compile(
    r"^https?://(?P<host>(?:sandbox\.)?zenodo\.org)/(?:records|record|api/records)/(?P<id>\d{1,12})(?:/.*)?$",
    re.IGNORECASE,
)
_CONTENT_RANGE = re.compile(r"bytes\s+(\d+)-(\d+)/(\d+|\*)", re.IGNORECASE)
_UNSAFE_KEY = re.compile(r"[\x00-\x1f<>:\"|?*]")
_MAX_WORKERS = 8


class ZenodoError(RuntimeError):
    """Base exception for Zenodo acquisition failures."""


class DownloadError(ZenodoError):
    """Raised when a file cannot be downloaded or fails integrity checks."""


class RetryableDownloadError(ZenodoError):
    """Internal signal that the same file should be tried again."""

    def __init__(self, message: str, *, delay: float | None = None) -> None:
        super().__init__(message)
        self.delay = delay


def normalize_api_base(value: str) -> str:
    """Return ``https://<zenodo-host>/api`` or raise."""
    text = str(value).strip().rstrip("/")
    parsed = urlparse(text)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or host not in ALLOWED_API_HOSTS:
        raise ZenodoError(f"unsupported_zenodo_api_base:{value}")
    if parsed.path.rstrip("/") not in {"", "/api"}:
        raise ZenodoError(f"unsupported_zenodo_api_base:{value}")
    return f"https://{host}/api"


def parse_record_ref(value: str, *, default_api_base: str = "https://zenodo.org/api") -> tuple[str, str]:
    """Parse a recid, DOI, or Zenodo record URL into ``(record_id, api_base)``."""
    text = str(value).strip()
    api_base = normalize_api_base(default_api_base)
    url_match = _RECORD_URL.match(text)
    if url_match:
        host = url_match.group("host").lower()
        return url_match.group("id"), normalize_api_base(f"https://{host}/api")
    doi_match = _DOI_REF.match(text)
    if doi_match:
        return doi_match.group(1), api_base
    if _NUMERIC_ID.fullmatch(text):
        return text, api_base
    raise ZenodoError(f"invalid_zenodo_record:{value}")


def validate_file_key(key: str) -> str:
    """Return a relative Zenodo file key, rejecting traversal and Windows-unsafe names."""
    raw = str(key).replace("\\", "/").strip()
    if not raw or raw in {".", ".."}:
        raise ZenodoError(f"invalid_zenodo_file_key:{key!r}")
    if _UNSAFE_KEY.search(raw):
        raise ZenodoError(f"invalid_zenodo_file_key:{key!r}")
    candidate = PurePosixPath(raw)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ZenodoError(f"invalid_zenodo_file_key:{key!r}")
    return candidate.as_posix()


def local_path_for_key(root: Path, key: str) -> Path:
    """Map a validated key onto ``root``, and refuse paths that escape it."""
    relative = validate_file_key(key)
    base = Path(root).resolve()
    target = (base / Path(*PurePosixPath(relative).parts)).resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise ZenodoError(f"destination_path_traversal:{key}") from exc
    return target


def parse_checksum(value: Any) -> tuple[str | None, str | None]:
    """Return ``(algorithm, hex digest)`` for a Zenodo checksum token."""
    if value in (None, ""):
        return None, None
    text = str(value).strip().lower()
    if ":" in text:
        algorithm, digest = text.split(":", 1)
        algorithm = algorithm.strip()
        digest = digest.strip()
    elif re.fullmatch(r"[0-9a-f]{32}", text):
        algorithm, digest = "md5", text
    elif re.fullmatch(r"[0-9a-f]{64}", text):
        algorithm, digest = "sha256", text
    else:
        raise ZenodoError(f"invalid_zenodo_checksum:{value!r}")
    if algorithm not in _CHECKSUM_ALGORITHMS or not re.fullmatch(r"[0-9a-f]+", digest):
        raise ZenodoError(f"invalid_zenodo_checksum:{value!r}")
    expected_length = {"md5": 32, "sha1": 40, "sha256": 64}[algorithm]
    if len(digest) != expected_length:
        raise ZenodoError(f"invalid_zenodo_checksum:{value!r}")
    return algorithm, digest


def checksum_token(algorithm: str | None, digest: str | None) -> str | None:
    if algorithm and digest:
        return f"{algorithm}:{digest}"
    return None


def unwrap_file_list(payload: Any) -> list[Mapping[str, Any]]:
    """Accept the file-list shapes Zenodo has published."""
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, Mapping):
        if isinstance(payload.get("entries"), list):
            items = payload["entries"]
        elif isinstance(payload.get("hits"), Mapping) and isinstance(payload["hits"].get("hits"), list):
            items = payload["hits"]["hits"]
        elif isinstance(payload.get("files"), list):
            items = payload["files"]
        else:
            items = []
    else:
        raise ZenodoError("zenodo_files_payload_invalid")
    for item in items:
        if not isinstance(item, Mapping):
            raise ZenodoError("zenodo_file_entry_invalid")
    return list(items)


def file_content_url(item: Mapping[str, Any], *, record_id: str, api_base: str) -> str:
    """Pick the HTTPS content URL for one file object."""
    links = item.get("links") if isinstance(item.get("links"), Mapping) else {}
    for field in ("content", "download", "self"):
        candidate = links.get(field)
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        url = candidate.strip()
        if not _is_https(url):
            raise ZenodoError(f"zenodo_file_url_not_https:{url}")
        if field != "self" or _path_endswith(url, "/content"):
            return url
    key = validate_file_key(str(item.get("key", "")))
    return f"{api_base}/records/{quote(record_id, safe='')}/files/{quote(key, safe='/')}/content"


def parse_file_entry(item: Mapping[str, Any], *, record_id: str, api_base: str) -> FileEntry:
    if not isinstance(item.get("key"), str):
        raise ZenodoError("zenodo_file_missing_key")
    key = validate_file_key(item["key"])
    size = _coerce_size(item.get("size"))
    algorithm, digest = parse_checksum(item.get("checksum"))
    return FileEntry(
        key=key,
        url=file_content_url(item, record_id=record_id, api_base=api_base),
        size=size,
        checksum=checksum_token(algorithm, digest),
        algorithm=algorithm,
        digest=digest,
    )


def file_digest(path: Path, algorithm: str, chunk_bytes: int) -> str:
    """Stream a content digest. MD5 and SHA-1 are used only as published checksums."""
    if algorithm == "md5":
        digest = hashlib.md5(usedforsecurity=False)
    elif algorithm == "sha1":
        digest = hashlib.sha1(usedforsecurity=False)
    elif algorithm == "sha256":
        digest = hashlib.sha256()
    else:
        raise ZenodoError(f"unsupported_checksum_algorithm:{algorithm}")
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def format_bytes(size: int | None) -> str:
    if size is None:
        return "unknown"
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{size} B"


def clamp_workers(value: int) -> int:
    workers = int(value)
    if workers < 1:
        raise ZenodoError("invalid_workers")
    if workers > _MAX_WORKERS:
        logger.info("Clamping Zenodo workers from %s to %s", workers, _MAX_WORKERS)
        return _MAX_WORKERS
    return workers


class ZenodoClient:
    """Anonymous or token-backed client for the Zenodo records API."""

    def __init__(
        self,
        *,
        api_base: str = "https://zenodo.org/api",
        token: str | None = None,
        opener: Callable[..., Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_base = normalize_api_base(api_base)
        self.token = _clean_token(token)
        self._opener = opener or urlopen
        self._sleep = sleeper or time.sleep
        self.timeout = float(timeout)

    def get_record(self, record_ref: str) -> ZenodoRecord:
        record_id, api_base = parse_record_ref(record_ref, default_api_base=self.api_base)
        url = f"{api_base}/records/{quote(record_id, safe='')}"
        payload = self._get_json(url)
        if not isinstance(payload, Mapping):
            raise ZenodoError("zenodo_record_payload_invalid")
        resolved = str(payload.get("id") or payload.get("recid") or record_id).strip()
        if not _NUMERIC_ID.fullmatch(resolved):
            raise ZenodoError(f"invalid_zenodo_record_id:{resolved}")
        files = self._collect_files(payload, record_id=resolved, api_base=api_base)
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else {}
        title = str(payload.get("title") or metadata.get("title") or "").strip()
        doi = _clean_doi(payload.get("doi") or metadata.get("doi"))
        concept_doi = _clean_doi(payload.get("conceptdoi") or metadata.get("conceptdoi"))
        return ZenodoRecord(
            record_id=resolved,
            files=files,
            title=title,
            doi=doi,
            concept_doi=concept_doi,
            api_url=f"{api_base}/records/{resolved}",
        )

    def download_file(
        self,
        entry: FileEntry,
        destination_root: Path,
        options: DownloadOptions,
        *,
        trusted_digest: str | None = None,
    ) -> FileTransfer:
        """Stream one file. A matching local file is left in place."""
        root = Path(destination_root).resolve()
        target = local_path_for_key(root, entry.key)
        partial = target.with_name(target.name + ".partial")
        target.parent.mkdir(parents=True, exist_ok=True)
        if options.overwrite:
            partial.unlink(missing_ok=True)
        else:
            skipped = self._skip_existing(target, entry, options, trusted_digest)
            if skipped is not None:
                partial.unlink(missing_ok=True)
                return skipped

        last_error: Exception | None = None
        checksum_failures = 0
        attempts = max(1, int(options.retries))
        for attempt in range(1, attempts + 1):
            try:
                prior_bytes = partial.stat().st_size if partial.exists() else 0
                self._transfer_once(entry, partial, options)
                self._assert_size(partial, entry)
                verified = False
                if options.verify_checksum and entry.digest and entry.algorithm:
                    actual = file_digest(partial, entry.algorithm, options.chunk_bytes)
                    if actual != entry.digest:
                        partial.unlink(missing_ok=True)
                        checksum_failures += 1
                        if checksum_failures >= 2:
                            raise DownloadError(f"checksum_mismatch:{entry.key}")
                        raise RetryableDownloadError(f"checksum_mismatch:{entry.key}")
                    verified = True
                elif not options.verify_checksum:
                    verified = False
                _fsync(partial)
                os.replace(partial, target)
                status = "resumed" if prior_bytes > 0 else "downloaded"
                logger.info(
                    "Finished %s status=%s bytes=%s",
                    entry.key,
                    status,
                    target.stat().st_size,
                )
                return _transfer(entry, target, status, checksum_verified=verified)
            except RetryableDownloadError as exc:
                last_error = exc
                logger.warning(
                    "Zenodo transfer attempt %s/%s for %s: %s",
                    attempt,
                    attempts,
                    entry.key,
                    exc,
                )
                if attempt < attempts:
                    self._sleep(_delay(exc.delay, options.retry_backoff_seconds, attempt))
            except DownloadError:
                raise
            except (URLError, TimeoutError, OSError) as exc:
                if isinstance(exc, OSError) and getattr(exc, "errno", None) == 28:
                    raise DownloadError(f"disk_full:{entry.key}") from exc
                last_error = exc
                logger.warning(
                    "Zenodo transfer attempt %s/%s for %s: %s",
                    attempt,
                    attempts,
                    entry.key,
                    exc,
                )
                if attempt < attempts:
                    self._sleep(_delay(None, options.retry_backoff_seconds, attempt))
        raise DownloadError(f"zenodo_download_failed:{entry.key}:{last_error}") from last_error

    def _collect_files(self, payload: Mapping[str, Any], *, record_id: str, api_base: str) -> tuple[FileEntry, ...]:
        links = payload.get("links") if isinstance(payload.get("links"), Mapping) else {}
        files_url = links.get("files") if isinstance(links, Mapping) else None
        raw_items: list[Mapping[str, Any]] = []
        if isinstance(files_url, str) and files_url.strip():
            try:
                raw_items = self._paginate_files(_absolute_api_url(files_url.strip(), api_base), api_base)
            except ZenodoError as exc:
                logger.warning("Zenodo files endpoint failed, using embedded files: %s", exc)
                raw_items = []
        if not raw_items:
            raw_items = unwrap_file_list(payload.get("files") or [])
        entries: list[FileEntry] = []
        seen: set[str] = set()
        for item in raw_items:
            entry = parse_file_entry(item, record_id=record_id, api_base=api_base)
            if entry.key in seen:
                raise ZenodoError(f"duplicate_file_key:{entry.key}")
            seen.add(entry.key)
            entries.append(entry)
        return tuple(entries)

    def _paginate_files(self, url: str, api_base: str) -> list[Mapping[str, Any]]:
        items: list[Mapping[str, Any]] = []
        seen_urls: set[str] = set()
        while url and url not in seen_urls:
            seen_urls.add(url)
            payload = self._get_json(url)
            items.extend(unwrap_file_list(payload))
            next_url = None
            if isinstance(payload, Mapping) and isinstance(payload.get("links"), Mapping):
                candidate = payload["links"].get("next")
                if isinstance(candidate, str) and candidate.strip():
                    next_url = _absolute_api_url(candidate.strip(), api_base)
            url = next_url or ""
        return items

    def _get_json(self, url: str, *, retries: int = 4) -> Any:
        if not _is_https(url) or urlparse(url).hostname not in ALLOWED_API_HOSTS:
            raise ZenodoError(f"unsupported_zenodo_url:{_url_without_query(url)}")
        last_error: Exception | None = None
        attempts = max(1, int(retries))
        for attempt in range(1, attempts + 1):
            try:
                request = Request(url, headers=self._headers({"Accept": "application/json"}))
                with self._opener(request, timeout=self.timeout) as response:
                    raw = response.read()
                return json.loads(raw.decode("utf-8"))
            except HTTPError as exc:
                last_error = exc
                delay = _retry_after_seconds(getattr(exc, "headers", None))
                try:
                    if exc.code not in RETRY_STATUS or attempt >= attempts:
                        raise ZenodoError(f"zenodo_http_{exc.code}:{_url_without_query(url)}") from exc
                finally:
                    exc.close()
                logger.warning("Zenodo JSON attempt %s/%s failed for %s: HTTP %s", attempt, attempts, _url_without_query(url), exc.code)
                self._sleep(_delay(delay, 2.0, attempt))
            except (URLError, TimeoutError, OSError, json.JSONDecodeError, UnicodeError) as exc:
                last_error = exc
                logger.warning("Zenodo JSON attempt %s/%s failed for %s: %s", attempt, attempts, _url_without_query(url), exc)
                if attempt >= attempts:
                    break
                self._sleep(_delay(None, 2.0, attempt))
        raise ZenodoError(f"zenodo_request_failed:{_url_without_query(url)}:{last_error}") from last_error

    def _transfer_once(self, entry: FileEntry, partial: Path, options: DownloadOptions) -> None:
        existing = partial.stat().st_size if partial.exists() else 0
        if entry.size is not None and existing > entry.size:
            partial.unlink()
            existing = 0
        if entry.size is not None and existing == entry.size:
            return
        if existing > 0 and not options.resume:
            partial.unlink()
            existing = 0
        range_header = f"bytes={existing}-" if existing > 0 and options.resume else None
        request = Request(entry.url, headers=self._headers({"Accept": "*/*"}, range_header=range_header))
        if not _is_https(entry.url):
            raise DownloadError(f"zenodo_file_url_not_https:{entry.key}")
        try:
            response = self._opener(request, timeout=options.timeout_seconds)
        except HTTPError as exc:
            self._handle_http_error(exc, partial, entry)
            return
        with response:
            status = _response_status(response)
            if status == 206:
                parsed = _parse_content_range(_header(response, "Content-Range"))
                if parsed is None or parsed[0] != existing:
                    partial.unlink(missing_ok=True)
                    raise RetryableDownloadError(f"unexpected_content_range:{entry.key}")
                if parsed[2] is not None and entry.size is not None and parsed[2] != entry.size:
                    raise DownloadError(f"content_range_size_mismatch:{entry.key}")
                mode = "ab"
            elif status == 200:
                mode = "wb"
                existing = 0
            else:
                raise RetryableDownloadError(f"unexpected_http_status:{status}:{entry.key}")
            written = existing
            last_report = existing
            logger.info(
                "Downloading %s from byte %s (%s)",
                entry.key,
                existing,
                format_bytes(entry.size),
            )
            with partial.open(mode) as handle:
                while True:
                    chunk = response.read(options.chunk_bytes)
                    if not chunk:
                        break
                    handle.write(chunk)
                    written += len(chunk)
                    if options.progress_bytes > 0 and written - last_report >= options.progress_bytes:
                        handle.flush()
                        logger.info(
                            "%s %s / %s",
                            entry.key,
                            format_bytes(written),
                            format_bytes(entry.size),
                        )
                        last_report = written
                handle.flush()
        final_size = partial.stat().st_size if partial.exists() else 0
        if entry.size is not None and final_size != entry.size:
            kind = "short_read" if final_size < entry.size else "long_read"
            if kind == "long_read":
                partial.unlink(missing_ok=True)
            raise RetryableDownloadError(f"{kind}:{entry.key}:{final_size}/{entry.size}")

    def _handle_http_error(self, exc: HTTPError, partial: Path, entry: FileEntry) -> None:
        delay = _retry_after_seconds(getattr(exc, "headers", None))
        try:
            if exc.code == 416:
                if entry.size is not None and partial.exists() and partial.stat().st_size == entry.size:
                    return
                partial.unlink(missing_ok=True)
                raise RetryableDownloadError(f"range_not_satisfiable:{entry.key}", delay=delay)
            if exc.code in RETRY_STATUS:
                raise RetryableDownloadError(f"http_{exc.code}:{entry.key}", delay=delay)
            raise DownloadError(f"http_{exc.code}:{entry.key}") from exc
        finally:
            exc.close()

    def _skip_existing(
        self,
        target: Path,
        entry: FileEntry,
        options: DownloadOptions,
        trusted_digest: str | None,
    ) -> FileTransfer | None:
        if not target.is_file() or entry.size is None:
            return None
        if target.stat().st_size != entry.size:
            return None
        if options.verify_checksum and entry.digest and entry.algorithm:
            if trusted_digest == entry.digest:
                logger.info("Skipping %s; receipt matches size and checksum", entry.key)
                return _transfer(entry, target, "skipped_verified", checksum_verified=True)
            actual = file_digest(target, entry.algorithm, options.chunk_bytes)
            if actual == entry.digest:
                logger.info("Skipping %s; local checksum matches", entry.key)
                return _transfer(entry, target, "skipped_verified", checksum_verified=True)
            logger.info("Replacing %s; local checksum differs", entry.key)
            return None
        logger.info("Skipping %s; local size matches", entry.key)
        return _transfer(entry, target, "skipped_size", checksum_verified=False)

    def _headers(self, extra: Mapping[str, str], *, range_header: str | None = None) -> dict[str, str]:
        headers = {"User-Agent": USER_AGENT}
        headers.update(extra)
        if range_header:
            headers["Range"] = range_header
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    @staticmethod
    def _assert_size(path: Path, entry: FileEntry) -> None:
        if entry.size is None:
            return
        actual = path.stat().st_size if path.exists() else 0
        if actual != entry.size:
            raise RetryableDownloadError(f"size_mismatch:{entry.key}:{actual}/{entry.size}")


def _transfer(entry: FileEntry, path: Path, status: str, *, checksum_verified: bool) -> FileTransfer:
    return FileTransfer(
        key=entry.key,
        local_path=str(path),
        size=path.stat().st_size if path.exists() else entry.size,
        checksum=entry.checksum,
        status=status,
        checksum_verified=checksum_verified,
    )


def _fsync(path: Path) -> None:
    """Flush file data before rename. A read-only handle fails fsync on Windows."""
    fd = os.open(path, os.O_RDWR)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _response_status(response: Any) -> int:
    status = getattr(response, "status", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        status = getcode() if callable(getcode) else 200
    return int(status)


def _header(response: Any, name: str) -> str | None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    direct = getter(name)
    if direct:
        return str(direct)
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            if str(key).lower() == name.lower() and value:
                return str(value)
    return None


def _parse_content_range(value: str | None) -> tuple[int, int, int | None] | None:
    if not value:
        return None
    match = _CONTENT_RANGE.search(value)
    if not match:
        return None
    total = None if match.group(3) == "*" else int(match.group(3))
    return int(match.group(1)), int(match.group(2)), total


def _retry_after_seconds(headers: Any) -> float | None:
    if headers is None or not hasattr(headers, "get"):
        return None
    raw = headers.get("Retry-After")
    if raw in (None, ""):
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def _delay(retry_after: float | None, backoff: float, attempt: int) -> float:
    if retry_after is not None:
        return min(max(retry_after, 0.0), 120.0)
    return min(max(backoff, 0.0) * attempt, 60.0)


def _coerce_size(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ZenodoError(f"invalid_zenodo_file_size:{value!r}") from exc
    if parsed < 0:
        raise ZenodoError(f"invalid_zenodo_file_size:{value!r}")
    return parsed


def _clean_token(token: str | None) -> str | None:
    if token is None:
        return None
    text = str(token).strip()
    if not text:
        return None
    if any(character.isspace() for character in text):
        raise ZenodoError("zenodo_token_invalid")
    return text


def _clean_doi(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text, flags=re.IGNORECASE)
    return text or None


def _is_https(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme == "https" and bool(parsed.netloc)


def _path_endswith(url: str, suffix: str) -> bool:
    return urlparse(url).path.rstrip("/").endswith(suffix)


def _absolute_api_url(url: str, api_base: str) -> str:
    if url.startswith("https://"):
        absolute = url
    elif url.startswith("/"):
        origin = api_base[: -len("/api")] if api_base.endswith("/api") else api_base
        absolute = origin + url
    else:
        raise ZenodoError(f"unsupported_zenodo_url:{url}")
    if not _is_https(absolute) or urlparse(absolute).hostname not in ALLOWED_API_HOSTS:
        raise ZenodoError(f"unsupported_zenodo_url:{_url_without_query(absolute)}")
    return absolute


def _url_without_query(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
