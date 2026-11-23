from __future__ import annotations

import hashlib
import io
import json
from http.client import HTTPMessage
from pathlib import Path
from urllib.error import HTTPError

import pytest

from zenodo_ingest.client import (
    DownloadError,
    ZenodoClient,
    ZenodoError,
    file_content_url,
    parse_file_entry,
    parse_record_ref,
    validate_file_key,
)
from zenodo_ingest.contracts import DownloadOptions


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers: dict[str, str] | None = None) -> None:
        self._body = body
        self._offset = 0
        self.status = status
        self.headers = headers or {}

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._body) - self._offset
        chunk = self._body[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def close(self) -> None:
        return None


def _json_response(payload: object) -> FakeResponse:
    return FakeResponse(json.dumps(payload).encode("utf-8"))


def _options(**overrides: object) -> DownloadOptions:
    values: dict[str, object] = {
        "chunk_bytes": 4,
        "retries": 3,
        "retry_backoff_seconds": 0.0,
        "timeout_seconds": 5.0,
        "resume": True,
        "verify_checksum": True,
        "overwrite": False,
        "progress_bytes": 0,
        "workers": 1,
        "continue_on_error": True,
    }
    values.update(overrides)
    return DownloadOptions(**values)  # type: ignore[arg-type]


def test_parse_record_ref_accepts_recid_doi_and_sandbox_url() -> None:
    assert parse_record_ref("4456079") == ("4456079", "https://zenodo.org/api")
    assert parse_record_ref("10.5281/zenodo.4456079") == ("4456079", "https://zenodo.org/api")
    assert parse_record_ref("https://doi.org/10.5281/zenodo.4456079") == ("4456079", "https://zenodo.org/api")
    assert parse_record_ref("https://sandbox.zenodo.org/records/15") == ("15", "https://sandbox.zenodo.org/api")
    with pytest.raises(ZenodoError, match="invalid_zenodo_record"):
        parse_record_ref("https://example.com/records/15")
    with pytest.raises(ZenodoError, match="unsupported_zenodo_api_base"):
        parse_record_ref("15", default_api_base="http://zenodo.org/api")


def test_file_key_rejects_traversal_and_keeps_spaces() -> None:
    assert validate_file_key("Exp1_Raw Data.zip") == "Exp1_Raw Data.zip"
    assert validate_file_key("dir/./file.zip") == "dir/file.zip"
    for key in ("../escape.zip", "/tmp/escape.zip", "a/../../b.zip", "bad:name.zip"):
        with pytest.raises(ZenodoError, match="invalid_zenodo_file_key"):
            validate_file_key(key)


def test_content_url_keeps_api_self_link_and_builds_a_fallback() -> None:
    published = "https://zenodo.org/api/records/4456079/files/GX_Subject%20Info%20&%20Behavioral%20Data.xlsx/content"
    entry = parse_file_entry(
        {
            "key": "GX_Subject Info & Behavioral Data.xlsx",
            "size": 452229,
            "checksum": "md5:7e432db18dea5176c93c96c511d45a83",
            "links": {"self": published},
        },
        record_id="4456079",
        api_base="https://zenodo.org/api",
    )
    assert entry.url == published
    assert entry.algorithm == "md5"
    assert entry.digest == "7e432db18dea5176c93c96c511d45a83"
    fallback = file_content_url(
        {"key": "a b.zip", "links": {"self": "https://zenodo.org/api/records/1/files/a%20b.zip"}},
        record_id="1",
        api_base="https://zenodo.org/api",
    )
    assert fallback == "https://zenodo.org/api/records/1/files/a%20b.zip/content"
    with pytest.raises(ZenodoError, match="zenodo_file_url_not_https"):
        file_content_url(
            {"key": "a.zip", "links": {"content": "http://zenodo.org/api/records/1/files/a.zip/content"}},
            record_id="1",
            api_base="https://zenodo.org/api",
        )


def test_get_record_reads_entries_shape_and_follows_next_page() -> None:
    seen: list[str] = []

    def opener(request, timeout):  # noqa: ANN001
        del timeout
        seen.append(request.full_url)
        if request.full_url.endswith("/records/9"):
            return _json_response(
                {
                    "id": 9,
                    "title": "Paged record",
                    "doi": "10.5281/zenodo.9",
                    "links": {"files": "https://zenodo.org/api/records/9/files"},
                    "files": [{"key": "ignored.zip", "size": 1, "links": {"self": "https://zenodo.org/api/records/9/files/ignored.zip/content"}}],
                }
            )
        if request.full_url.endswith("/files"):
            return _json_response(
                {
                    "entries": [
                        {
                            "key": "Exp1_Raw Data.zip",
                            "size": 4,
                            "checksum": "md5:" + "ab" * 16,
                            "links": {"self": "https://zenodo.org/api/records/9/files/Exp1_Raw%20Data.zip/content"},
                        }
                    ],
                    "links": {"next": "https://zenodo.org/api/records/9/files?page=2"},
                }
            )
        if request.full_url.endswith("/files?page=2"):
            return _json_response(
                {
                    "hits": {
                        "hits": [
                            {
                                "key": "notes.pdf",
                                "size": 2,
                                "checksum": "md5:" + "cd" * 16,
                                "links": {"self": "https://zenodo.org/api/records/9/files/notes.pdf/content"},
                            }
                        ]
                    },
                    "links": {},
                }
            )
        raise AssertionError(request.full_url)

    record = ZenodoClient(opener=opener, sleeper=lambda delay: None).get_record("https://zenodo.org/records/9")
    assert record.record_id == "9"
    assert record.doi == "10.5281/zenodo.9"
    assert [item.key for item in record.files] == ["Exp1_Raw Data.zip", "notes.pdf"]
    assert seen[0] == "https://zenodo.org/api/records/9"
    assert "ignored.zip" not in [item.key for item in record.files]


def test_download_verifies_md5_and_renames_partial(tmp_path: Path) -> None:
    body = b"large-file-bytes"
    digest = hashlib.md5(body, usedforsecurity=False).hexdigest()
    entry = parse_file_entry(
        {
            "key": "Exp1_Raw Data.zip",
            "size": len(body),
            "checksum": f"md5:{digest}",
            "links": {"self": "https://zenodo.org/api/records/9/files/Exp1_Raw%20Data.zip/content"},
        },
        record_id="9",
        api_base="https://zenodo.org/api",
    )

    def opener(request, timeout):  # noqa: ANN001
        del request, timeout
        return FakeResponse(body, status=200)

    target = ZenodoClient(opener=opener, sleeper=lambda delay: None).download_file(entry, tmp_path, _options())
    assert target.status == "downloaded"
    assert target.checksum_verified is True
    assert Path(target.local_path).read_bytes() == body
    assert list(tmp_path.glob("*.partial")) == []


def test_short_read_resumes_with_range(tmp_path: Path) -> None:
    body = b"abcdefghijklmnopqrstuvwxyz"
    digest = hashlib.md5(body, usedforsecurity=False).hexdigest()
    entry = parse_file_entry(
        {
            "key": "signal.zip",
            "size": len(body),
            "checksum": f"md5:{digest}",
            "links": {"content": "https://zenodo.org/api/records/9/files/signal.zip/content"},
        },
        record_id="9",
        api_base="https://zenodo.org/api",
    )
    calls: list[str | None] = []

    def opener(request, timeout):  # noqa: ANN001
        del timeout
        range_header = request.get_header("Range")
        calls.append(range_header)
        if range_header is None:
            return FakeResponse(body[:10], status=200)
        assert range_header == "bytes=10-"
        return FakeResponse(
            body[10:],
            status=206,
            headers={"Content-Range": f"bytes 10-{len(body) - 1}/{len(body)}"},
        )

    result = ZenodoClient(opener=opener, sleeper=lambda delay: None).download_file(entry, tmp_path, _options())
    assert calls == [None, "bytes=10-"]
    assert result.status == "resumed"
    assert Path(result.local_path).read_bytes() == body


def test_ignored_range_restarts_from_the_full_body(tmp_path: Path) -> None:
    body = b"hello world"
    digest = hashlib.md5(body, usedforsecurity=False).hexdigest()
    entry = parse_file_entry(
        {
            "key": "signal.zip",
            "size": len(body),
            "checksum": f"md5:{digest}",
            "links": {"content": "https://zenodo.org/api/records/9/files/signal.zip/content"},
        },
        record_id="9",
        api_base="https://zenodo.org/api",
    )
    calls: list[str | None] = []

    def opener(request, timeout):  # noqa: ANN001
        del timeout
        range_header = request.get_header("Range")
        calls.append(range_header)
        if range_header is None:
            return FakeResponse(b"hello", status=200)
        return FakeResponse(body, status=200)

    result = ZenodoClient(opener=opener, sleeper=lambda delay: None).download_file(entry, tmp_path, _options())
    assert calls == [None, "bytes=5-"]
    assert result.status == "resumed"
    assert Path(result.local_path).read_bytes() == body


def test_checksum_mismatch_does_not_publish_the_file(tmp_path: Path) -> None:
    entry = parse_file_entry(
        {
            "key": "signal.zip",
            "size": 4,
            "checksum": "md5:" + "ab" * 16,
            "links": {"content": "https://zenodo.org/api/records/9/files/signal.zip/content"},
        },
        record_id="9",
        api_base="https://zenodo.org/api",
    )

    def opener(request, timeout):  # noqa: ANN001
        del request, timeout
        return FakeResponse(b"data", status=200)

    with pytest.raises(DownloadError, match="checksum_mismatch:signal.zip"):
        ZenodoClient(opener=opener, sleeper=lambda delay: None).download_file(entry, tmp_path, _options(retries=2))
    assert list(tmp_path.iterdir()) == []


def test_http_404_is_not_retried_and_429_is_retried(tmp_path: Path) -> None:
    entry = parse_file_entry(
        {
            "key": "signal.zip",
            "size": 4,
            "checksum": "md5:" + hashlib.md5(b"data", usedforsecurity=False).hexdigest(),
            "links": {"content": "https://zenodo.org/api/records/9/files/signal.zip/content"},
        },
        record_id="9",
        api_base="https://zenodo.org/api",
    )
    missing_calls = {"n": 0}

    def missing(request, timeout):  # noqa: ANN001
        del timeout
        missing_calls["n"] += 1
        raise HTTPError(request.full_url, 404, "missing", HTTPMessage(), io.BytesIO(b""))

    with pytest.raises(DownloadError, match="http_404"):
        ZenodoClient(opener=missing, sleeper=lambda delay: None).download_file(entry, tmp_path, _options())
    assert missing_calls["n"] == 1

    rate_calls = {"n": 0}
    sleeps: list[float] = []

    def limited(request, timeout):  # noqa: ANN001
        del timeout
        rate_calls["n"] += 1
        if rate_calls["n"] == 1:
            headers = HTTPMessage()
            headers["Retry-After"] = "0"
            raise HTTPError(request.full_url, 429, "rate", headers, io.BytesIO(b""))
        return FakeResponse(b"data", status=200)

    result = ZenodoClient(opener=limited, sleeper=sleeps.append).download_file(entry, tmp_path, _options())
    assert rate_calls["n"] == 2
    assert sleeps == [0.0]
    assert result.status == "downloaded"


def test_receipt_digest_skips_a_second_transfer(tmp_path: Path) -> None:
    body = b"data"
    digest = hashlib.md5(body, usedforsecurity=False).hexdigest()
    entry = parse_file_entry(
        {
            "key": "signal.zip",
            "size": len(body),
            "checksum": f"md5:{digest}",
            "links": {"content": "https://zenodo.org/api/records/9/files/signal.zip/content"},
        },
        record_id="9",
        api_base="https://zenodo.org/api",
    )
    calls = {"n": 0}

    def opener(request, timeout):  # noqa: ANN001
        del request, timeout
        calls["n"] += 1
        return FakeResponse(body, status=200)

    client = ZenodoClient(opener=opener, sleeper=lambda delay: None)
    first = client.download_file(entry, tmp_path, _options())
    second = client.download_file(entry, tmp_path, _options(), trusted_digest=digest)
    assert first.status == "downloaded"
    assert second.status == "skipped_verified"
    assert calls["n"] == 1
