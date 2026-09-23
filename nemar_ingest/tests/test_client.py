from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest

from nemar_ingest.client import ManifestError, NemarClient, parse_manifest, validate_relative_path
from nemar_ingest.contracts import ManifestEntry


def test_version_and_dataset_are_embedded_as_quoted_path_segments() -> None:
    client = NemarClient()
    assert client.manifest_url("on005385", "v1.0.0") == "https://data.nemar.org/on005385/v1.0.0/manifest.json"
    with pytest.raises(ValueError, match="invalid_nemar_version"):
        client.manifest_url("on005385", "../../secret")
    with pytest.raises(ValueError, match="invalid_nemar_dataset_id"):
        client.manifest_url("../secret", "latest")


def test_manifest_rejects_path_traversal_and_non_https_urls() -> None:
    with pytest.raises(ManifestError, match="manifest_path_traversal"):
        parse_manifest([{"path": "sub-01/../../escape.edf"}], base_url="https://data.nemar.org/on005385/latest")
    with pytest.raises(ManifestError, match="manifest_url_not_https"):
        parse_manifest([{"path": "participants.tsv", "bytes_url": "http://evil.invalid/x"}], base_url="https://data.nemar.org/on005385/latest")
    with pytest.raises(ManifestError, match="manifest_path_traversal"):
        validate_relative_path("/absolute/file.edf")


def test_download_entry_writes_atomically_and_verifies_size_and_sha256(tmp_path: Path) -> None:
    body = b"participants\nsub-01\n"
    digest = hashlib.sha256(body).hexdigest()

    def opener(request, timeout):
        return io.BytesIO(body)

    client = NemarClient(opener=opener)
    target = client.download_entry(ManifestEntry("participants.tsv", "https://data.nemar.org/x", len(body), digest), tmp_path)
    assert target.read_bytes() == body
    assert not list(tmp_path.glob("*.part"))


def test_parse_manifest_accepts_git_blob_and_sha256_checksums() -> None:
    git_digest = "66fda97533307ffdbd0d7995ab97297aa16b54d4"
    sha_digest = "ab" * 32
    entries = parse_manifest(
        [
            {"path": ".bidsignore", "size": 8, "checksum_algorithm": "git", "checksum": git_digest},
            {"path": "sub-001/ses-1/eeg/sub-001_ses-1_task-rest_eeg.edf", "size": 12, "checksum_algorithm": "sha256", "checksum": sha_digest},
        ],
        base_url="https://data.nemar.org/on005385/v1.0.0",
    )
    assert entries[0].checksum_algorithm == "git"
    assert entries[0].checksum == git_digest
    assert entries[0].sha256 is None
    assert entries[1].checksum_algorithm == "sha256"
    assert entries[1].sha256 == sha_digest


def test_parse_manifest_rejects_unknown_or_bare_short_checksums() -> None:
    with pytest.raises(ManifestError, match="unsupported_manifest_checksum_algorithm"):
        parse_manifest(
            [{"path": "participants.tsv", "checksum_algorithm": "md5", "checksum": "ab" * 16}],
            base_url="https://data.nemar.org/on005385/v1.0.0",
        )
    with pytest.raises(ManifestError, match="invalid_manifest_sha256"):
        parse_manifest(
            [{"path": "participants.tsv", "checksum": "66fda97533307ffdbd0d7995ab97297aa16b54d4"}],
            base_url="https://data.nemar.org/on005385/v1.0.0",
        )


def test_download_entry_verifies_git_blob_checksum(tmp_path: Path) -> None:
    body = b".nemar/\n"
    prefix = f"blob {len(body)}\0".encode("ascii")
    digest = hashlib.sha1(prefix + body, usedforsecurity=False).hexdigest()

    def opener(request, timeout):
        return io.BytesIO(body)

    client = NemarClient(opener=opener)
    entry = ManifestEntry(
        ".bidsignore",
        "https://data.nemar.org/on005385/v1.0.0/.bidsignore",
        len(body),
        checksum=digest,
        checksum_algorithm="git",
    )
    target = client.download_entry(entry, tmp_path, retries=1)
    assert target.read_bytes() == body

    mismatched = ManifestEntry(
        "README.md",
        "https://data.nemar.org/on005385/v1.0.0/README.md",
        len(body),
        checksum="0" * 40,
        checksum_algorithm="git",
    )
    with pytest.raises(Exception, match="integrity_check_failed"):
        client.download_entry(mismatched, tmp_path, retries=1)


def test_parse_manifest_accepts_nemar_bytes_url_shape() -> None:
    entries = parse_manifest(
        [{"path": "sub-01/eeg/sub-01_task-rest_eeg.set", "bytes_url": "https://s3.example/set", "bytes": 4}],
        base_url="https://data.nemar.org/on005385/v1.0.0",
    )
    assert entries[0].path == "sub-01/eeg/sub-01_task-rest_eeg.set"
    assert entries[0].size == 4

