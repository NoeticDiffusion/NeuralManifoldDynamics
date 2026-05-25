from __future__ import annotations

import json
from pathlib import Path

import pytest

from physionet_ingest.script import download_physionet as mod


def test_parse_records_and_extract_patient_ids() -> None:
    records_text = """
    # comment
    training/0284/0284_001_004_EEG
    training/0284/0284_001_005_EEG
    training/0300/0300_001_001_EEG
    """
    parsed = mod.parse_records_text(records_text)

    assert parsed == [
        "training/0284/0284_001_004_EEG",
        "training/0284/0284_001_005_EEG",
        "training/0300/0300_001_001_EEG",
    ]
    assert mod.extract_patient_id(parsed[0]) == "0284"
    assert mod.extract_patient_id(parsed[2]) == "0300"
    assert mod.extract_patient_id("SHA256SUMS.txt") is None


def test_select_first_n_patients_preserves_order() -> None:
    records = [
        "training/0284/0284_001_004_EEG",
        "training/0300/0300_001_001_EEG",
        "training/0284/0284_001_005_EEG",
        "training/0400/0400_001_001_EEG",
    ]

    patients, selected = mod.select_first_n_patients(records, patient_count=2)

    assert patients == ["0284", "0300"]
    assert selected == [
        "training/0284/0284_001_004_EEG",
        "training/0300/0300_001_001_EEG",
        "training/0284/0284_001_005_EEG",
    ]


def test_select_random_n_patients_is_deterministic() -> None:
    records = [
        "training/0284/0284_001_004_EEG",
        "training/0300/0300_001_001_EEG",
        "training/0400/0400_001_001_EEG",
        "training/0500/0500_001_001_EEG",
    ]
    patients_a, selected_a = mod.select_random_n_patients(records, patient_count=3, random_seed=7)
    patients_b, selected_b = mod.select_random_n_patients(records, patient_count=3, random_seed=7)
    patients_c, _ = mod.select_random_n_patients(records, patient_count=3, random_seed=8)

    assert patients_a == patients_b
    assert selected_a == selected_b
    assert patients_a != patients_c


def test_select_explicit_patient_ids_supports_normalization() -> None:
    records = [
        "training/0284/0284_001_004_EEG",
        "training/0300/0300_001_001_EEG",
        "training/0400/0400_001_001_EEG",
    ]
    selected_patients, selected_records = mod.select_explicit_patient_ids(
        records,
        requested_patient_ids=["284", "sub-300", "0400"],
    )
    assert selected_patients == ["0284", "0300", "0400"]
    assert len(selected_records) == 3


def test_select_explicit_patient_ids_raises_on_missing() -> None:
    records = [
        "training/0284/0284_001_004_EEG",
        "training/0300/0300_001_001_EEG",
    ]
    with pytest.raises(ValueError, match="not found"):
        mod.select_explicit_patient_ids(records, requested_patient_ids=["9999"])


def test_expand_record_paths_for_wfdb_files() -> None:
    expanded = mod.expand_record_paths(
        record_paths=["training/0284/0284_001_004_EEG"],
        extensions=[".hea", ".mat", ".txt"],
    )
    assert expanded == [
        "training/0284/0284_001_004_EEG.hea",
        "training/0284/0284_001_004_EEG.mat",
        "training/0284/0284_001_004_EEG.txt",
    ]


def test_parse_sha256sums_text() -> None:
    checksums_text = """
    # sha256 checksums
    aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  training/0284/0284_001_004_EEG.hea
    bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb *training/0284/0284_001_004_EEG.mat
    """
    parsed = mod.parse_sha256sums_text(checksums_text)

    assert parsed["training/0284/0284_001_004_EEG.hea"] == "a" * 64
    assert parsed["training/0284/0284_001_004_EEG.mat"] == "b" * 64


def test_parse_wfdb_duration_seconds() -> None:
    header_text = "\n".join(
        [
            "0332_002_023_EEG 20 200 720000",
            "#Start time: 23:00:00",
            "#End time: 23:59:59",
        ]
    )
    duration = mod.parse_wfdb_duration_seconds(header_text)
    assert duration == pytest.approx(3600.0)


def test_build_checksum_lookup_strips_dataset_prefix() -> None:
    lookup = mod.build_checksum_lookup(
        {
            "files/i-care/2.1/training/0284/0284_001_004_EEG.hea": "a" * 64,
            "training/0300/0300_001_001_EEG.hea": "b" * 64,
        },
        remote_root="files/i-care/2.1",
    )
    assert lookup["training/0284/0284_001_004_EEG.hea"] == "a" * 64
    assert lookup["training/0300/0300_001_001_EEG.hea"] == "b" * 64


def test_estimate_subset_size_from_global_total() -> None:
    estimate = mod.estimate_subset_size_from_global_total(
        total_size_bytes=float(1_500 * 1024**3),
        total_patients=607,
        selected_patients=40,
    )
    assert estimate["estimated_selected_gb"] > 90.0
    assert estimate["estimated_selected_gb"] < 110.0


def test_expand_selected_records_from_checksum_prefix_with_globs_and_cap() -> None:
    checksums = {
        "training/0284/0284.txt": "a" * 64,
        "training/0284/0284_001_004_EEG.hea": "b" * 64,
        "training/0284/0284_001_004_EEG.mat": "c" * 64,
        "training/0284/0284_001_004_ECG.mat": "d" * 64,
    }
    expanded = mod.expand_selected_records_to_files(
        selected_records=["training/0284"],
        checksums_lookup=checksums,
        include_file_globs=["*.txt", "*_EEG.hea", "*_EEG.mat"],
        fallback_extensions=[".hea", ".mat"],
        max_files_per_patient=3,
    )

    assert expanded == [
        "training/0284/0284.txt",
        "training/0284/0284_001_004_EEG.hea",
        "training/0284/0284_001_004_EEG.mat",
    ]


def test_limit_icare_eeg_files_to_first_hours_keeps_matching_pairs(monkeypatch, tmp_path: Path) -> None:
    selected_paths = [
        "training/0001/0001_001_001_EEG.hea",
        "training/0001/0001_001_001_EEG.mat",
        "training/0001/0001_002_002_EEG.hea",
        "training/0001/0001_002_002_EEG.mat",
        "training/0001/0001_003_003_EEG.hea",
        "training/0001/0001_003_003_EEG.mat",
        "training/0001/0001.txt",
    ]

    def fake_fetch_text_with_retries(url: str, **_: object) -> str:
        if "0001_001_001_EEG.hea" in url:
            return "0001_001_001_EEG 20 200 360000\n"
        if "0001_002_002_EEG.hea" in url:
            return "0001_002_002_EEG 20 200 720000\n"
        if "0001_003_003_EEG.hea" in url:
            return "0001_003_003_EEG 20 200 720000\n"
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(mod, "fetch_text_with_retries", fake_fetch_text_with_retries)

    filtered, summary = mod.limit_icare_eeg_files_to_first_hours(
        selected_paths=selected_paths,
        max_hours_per_patient=1.5,
        output_dir=tmp_path / "received",
        dataset_root_url="https://physionet.org/files/i-care/2.1",
        timeout_seconds=5.0,
        retries=1,
        retry_backoff_seconds=0.0,
        user_agent="test-agent",
    )

    assert "training/0001/0001_001_001_EEG.hea" in filtered
    assert "training/0001/0001_001_001_EEG.mat" in filtered
    assert "training/0001/0001_002_002_EEG.hea" in filtered
    assert "training/0001/0001_002_002_EEG.mat" in filtered
    assert "training/0001/0001_003_003_EEG.hea" not in filtered
    assert "training/0001/0001_003_003_EEG.mat" not in filtered
    assert "training/0001/0001.txt" in filtered
    assert summary["eeg_files_dropped"] == 2


def test_limit_icare_eeg_files_to_hour_window_supports_12_24(monkeypatch, tmp_path: Path) -> None:
    selected_paths = [
        "training/0001/0001_001_001_EEG.hea",
        "training/0001/0001_001_001_EEG.mat",
        "training/0001/0001_002_002_EEG.hea",
        "training/0001/0001_002_002_EEG.mat",
        "training/0001/0001_003_003_EEG.hea",
        "training/0001/0001_003_003_EEG.mat",
        "training/0001/0001_004_004_EEG.hea",
        "training/0001/0001_004_004_EEG.mat",
    ]

    def fake_fetch_text_with_retries(url: str, **_: object) -> str:
        # Four sequential 8-hour runs.
        if "0001_001_001_EEG.hea" in url:
            return "0001_001_001_EEG 20 200 5760000\n"
        if "0001_002_002_EEG.hea" in url:
            return "0001_002_002_EEG 20 200 5760000\n"
        if "0001_003_003_EEG.hea" in url:
            return "0001_003_003_EEG 20 200 5760000\n"
        if "0001_004_004_EEG.hea" in url:
            return "0001_004_004_EEG 20 200 5760000\n"
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(mod, "fetch_text_with_retries", fake_fetch_text_with_retries)

    filtered, summary = mod.limit_icare_eeg_files_to_first_hours(
        selected_paths=selected_paths,
        min_hours_per_patient=12.0,
        max_hours_per_patient=24.0,
        output_dir=tmp_path / "received",
        dataset_root_url="https://physionet.org/files/i-care/2.1",
        timeout_seconds=5.0,
        retries=1,
        retry_backoff_seconds=0.0,
        user_agent="test-agent",
    )

    # Keep runs that overlap [12, 24] hours: run2 and run3.
    assert "training/0001/0001_001_001_EEG.hea" not in filtered
    assert "training/0001/0001_001_001_EEG.mat" not in filtered
    assert "training/0001/0001_002_002_EEG.hea" in filtered
    assert "training/0001/0001_002_002_EEG.mat" in filtered
    assert "training/0001/0001_003_003_EEG.hea" in filtered
    assert "training/0001/0001_003_003_EEG.mat" in filtered
    assert "training/0001/0001_004_004_EEG.hea" not in filtered
    assert "training/0001/0001_004_004_EEG.mat" not in filtered

    assert summary["min_hours_per_patient"] == pytest.approx(12.0)
    assert summary["max_hours_per_patient"] == pytest.approx(24.0)
    assert summary["eeg_files_kept"] == 4


def test_download_or_skip_uses_size_check_when_checksum_disabled(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "received"
    local_path = output_dir / "training" / "0284" / "0284_001_004_EEG.mat"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"abcdef")

    monkeypatch.setattr(mod, "fetch_remote_content_length_with_retries", lambda **_: 6)

    row = mod.download_or_skip_relative_path(
        relative_path="training/0284/0284_001_004_EEG.mat",
        output_dir=output_dir,
        dataset_root_url="https://physionet.org/files/i-care/2.1",
        checksums={},
        overwrite=False,
        verify_checksum=False,
        verify_existing_size=True,
        timeout_seconds=5.0,
        retries=1,
        retry_backoff_seconds=0.0,
        chunk_size_bytes=1024,
        user_agent="test-agent",
    )

    assert row["status"] == "skipped_exists_size_match"
    assert row["bytes"] == 0


def test_download_or_skip_size_check_skips_when_length_unknown(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "received"
    local_path = output_dir / "training" / "0284" / "0284_001_004_EEG.mat"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"abcdef")

    monkeypatch.setattr(mod, "fetch_remote_content_length_with_retries", lambda **_: None)

    row = mod.download_or_skip_relative_path(
        relative_path="training/0284/0284_001_004_EEG.mat",
        output_dir=output_dir,
        dataset_root_url="https://physionet.org/files/i-care/2.1",
        checksums={},
        overwrite=False,
        verify_checksum=False,
        verify_existing_size=True,
        timeout_seconds=5.0,
        retries=1,
        retry_backoff_seconds=0.0,
        chunk_size_bytes=1024,
        user_agent="test-agent",
    )

    assert row["status"] == "skipped_exists_size_unverified"
    assert row["bytes"] == 0


def test_download_or_skip_existing_name_only_skips_without_remote_size_check(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "received"
    local_path = output_dir / "training" / "0284" / "0284_001_004_EEG.mat"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(b"abcdef")

    def _unexpected_remote_size_call(**_: object) -> int:
        raise AssertionError("Remote size check should not be called in name-only mode")

    monkeypatch.setattr(mod, "fetch_remote_content_length_with_retries", _unexpected_remote_size_call)

    row = mod.download_or_skip_relative_path(
        relative_path="training/0284/0284_001_004_EEG.mat",
        output_dir=output_dir,
        dataset_root_url="https://physionet.org/files/i-care/2.1",
        checksums={},
        overwrite=False,
        verify_checksum=False,
        verify_existing_size=False,
        timeout_seconds=5.0,
        retries=1,
        retry_backoff_seconds=0.0,
        chunk_size_bytes=1024,
        user_agent="test-agent",
    )

    assert row["status"] == "skipped_exists"
    assert row["bytes"] == 0


def test_run_download_dry_run_writes_manifests(tmp_path: Path, monkeypatch) -> None:
    general_cfg = tmp_path / "config_ingest.yml"
    dataset_cfg = tmp_path / "config_i-care_2_1.yml"

    general_cfg.write_text(
        "\n".join(
            [
                "paths:",
                f"  download_root: {tmp_path.as_posix()}/received",
                f"  metadata_root: {tmp_path.as_posix()}/metadata",
                "network:",
                "  timeout_seconds: 5",
                "  retries: 1",
                "  retry_backoff_seconds: 0.0",
                "  chunk_size_bytes: 1024",
                "  user_agent: test-agent",
                "download:",
                "  overwrite: false",
                "  verify_checksum: true",
                "  write_manifest_csv: true",
                "  write_manifest_jsonl: true",
            ]
        ),
        encoding="utf-8",
    )
    dataset_cfg.write_text(
        "\n".join(
            [
                "dataset:",
                "  slug: i-care",
                "  version: '2.1'",
                "  base_url: https://physionet.org",
                "  remote_root: files/i-care/2.1",
                "  records_file: RECORDS",
                "  checksums_file: SHA256SUMS.txt",
                "subset:",
                "  strategy: first_n_patients",
                "  patient_count: 1",
                "  expand_record_extensions: ['.hea', '.mat']",
                "  include_top_level_files: ['LICENSE.txt']",
                "download:",
                "  output_subdir: smoke",
                "  dry_run: true",
            ]
        ),
        encoding="utf-8",
    )

    records_text = "\n".join(
        [
            "training/0284/0284_001_004_EEG",
            "training/0284/0284_001_005_EEG",
            "training/0300/0300_001_001_EEG",
        ]
    )
    checksums_text = "\n".join(
        [
            f"{'a' * 64}  training/0284/0284_001_004_EEG.hea",
            f"{'b' * 64}  training/0284/0284_001_004_EEG.mat",
            f"{'c' * 64}  training/0284/0284_001_005_EEG.hea",
            f"{'d' * 64}  training/0284/0284_001_005_EEG.mat",
            f"{'e' * 64}  LICENSE.txt",
        ]
    )

    def fake_fetch_text_with_retries(url: str, **_: object) -> str:
        if "RECORDS" in url:
            return records_text
        return checksums_text

    monkeypatch.setattr(mod, "fetch_text_with_retries", fake_fetch_text_with_retries)
    monkeypatch.setattr(mod, "fetch_checksums_text", lambda **_: checksums_text)
    monkeypatch.setattr(mod, "maybe_create_physionet_client", lambda *_, **__: None)

    exit_code = mod.run_download(general_cfg, dataset_cfg)
    assert exit_code == 0

    metadata_dir = tmp_path / "metadata" / "smoke"
    selected_patients = (metadata_dir / "selected_patients.txt").read_text(encoding="utf-8").strip().splitlines()
    planned_files = (metadata_dir / "planned_files.txt").read_text(encoding="utf-8").strip().splitlines()
    summary = json.loads((metadata_dir / "run_summary.json").read_text(encoding="utf-8"))

    assert selected_patients == ["0284"]
    assert "training/0284/0284_001_004_EEG.hea" in planned_files
    assert "training/0284/0284_001_004_EEG.mat" in planned_files
    assert "LICENSE.txt" in planned_files
    assert summary["execution"]["dry_run"] is True

    csv_manifest = (metadata_dir / "download_manifest.csv").read_text(encoding="utf-8")
    assert "planned" in csv_manifest


def test_run_download_name_only_override_disables_existing_size_checks(tmp_path: Path, monkeypatch) -> None:
    general_cfg = tmp_path / "config_ingest.yml"
    dataset_cfg = tmp_path / "config_i-care_2_1.yml"

    general_cfg.write_text(
        "\n".join(
            [
                "paths:",
                f"  download_root: {tmp_path.as_posix()}/received",
                f"  metadata_root: {tmp_path.as_posix()}/metadata",
                "network:",
                "  timeout_seconds: 5",
                "  retries: 1",
                "  retry_backoff_seconds: 0.0",
                "  chunk_size_bytes: 1024",
                "  user_agent: test-agent",
                "download:",
                "  overwrite: false",
                "  verify_checksum: false",
                "  verify_existing_size: true",
                "  write_manifest_csv: true",
                "  write_manifest_jsonl: true",
            ]
        ),
        encoding="utf-8",
    )
    dataset_cfg.write_text(
        "\n".join(
            [
                "dataset:",
                "  slug: i-care",
                "  version: '2.1'",
                "  base_url: https://physionet.org",
                "  remote_root: files/i-care/2.1",
                "  records_file: RECORDS",
                "  checksums_file: SHA256SUMS.txt",
                "subset:",
                "  strategy: first_n_patients",
                "  patient_count: 1",
                "  expand_record_extensions: ['.hea', '.mat']",
                "download:",
                "  output_subdir: smoke",
                "  dry_run: false",
                "  max_parallel_downloads: 1",
            ]
        ),
        encoding="utf-8",
    )

    records_text = "training/0284/0284_001_004_EEG\n"
    checksums_text = "\n".join(
        [
            f"{'a' * 64}  training/0284/0284_001_004_EEG.hea",
            f"{'b' * 64}  training/0284/0284_001_004_EEG.mat",
        ]
    )

    def fake_fetch_text_with_retries(url: str, **_: object) -> str:
        if "RECORDS" in url:
            return records_text
        return checksums_text

    def fake_download_file_with_retries(url: str, output_path: Path, **_: object) -> int:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"x")
        return 1

    def _unexpected_remote_size_call(**_: object) -> int:
        raise AssertionError("Remote size check should not be called with name-only override")

    local_existing = tmp_path / "received" / "smoke" / "training" / "0284" / "0284_001_004_EEG.mat"
    local_existing.parent.mkdir(parents=True, exist_ok=True)
    local_existing.write_bytes(b"already-here")

    monkeypatch.setattr(mod, "fetch_text_with_retries", fake_fetch_text_with_retries)
    monkeypatch.setattr(mod, "fetch_checksums_text", lambda **_: checksums_text)
    monkeypatch.setattr(mod, "maybe_create_physionet_client", lambda *_, **__: None)
    monkeypatch.setattr(mod, "download_file_with_retries", fake_download_file_with_retries)
    monkeypatch.setattr(mod, "fetch_remote_content_length_with_retries", _unexpected_remote_size_call)

    exit_code = mod.run_download(
        general_cfg,
        dataset_cfg,
        name_only_existing_check_override=True,
    )
    assert exit_code == 0

    metadata_dir = tmp_path / "metadata" / "smoke"
    csv_manifest = (metadata_dir / "download_manifest.csv").read_text(encoding="utf-8")
    run_summary = json.loads((metadata_dir / "run_summary.json").read_text(encoding="utf-8"))

    assert "skipped_exists" in csv_manifest
    assert run_summary["execution"]["existing_file_check_mode"] == "name_only"
    assert run_summary["execution"]["verify_existing_size"] is False


def test_run_download_rejects_non_positive_parallel_workers(tmp_path: Path) -> None:
    general_cfg = tmp_path / "config_ingest.yml"
    dataset_cfg = tmp_path / "config_i-care_2_1.yml"

    general_cfg.write_text(
        "\n".join(
            [
                "paths:",
                f"  download_root: {tmp_path.as_posix()}/received",
                f"  metadata_root: {tmp_path.as_posix()}/metadata",
                "network:",
                "  timeout_seconds: 5",
                "  retries: 1",
                "  retry_backoff_seconds: 0.0",
                "  chunk_size_bytes: 1024",
                "  user_agent: test-agent",
                "download:",
                "  max_parallel_downloads: 0",
            ]
        ),
        encoding="utf-8",
    )
    dataset_cfg.write_text(
        "\n".join(
            [
                "dataset:",
                "  slug: i-care",
                "  version: '2.1'",
                "subset:",
                "  strategy: first_n_patients",
                "  patient_count: 1",
                "download:",
                "  output_subdir: smoke",
                "  dry_run: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="download.max_parallel_downloads must be >= 1"):
        mod.run_download(general_cfg, dataset_cfg)


def test_run_download_rejects_non_positive_max_eeg_hours(tmp_path: Path) -> None:
    general_cfg = tmp_path / "config_ingest.yml"
    dataset_cfg = tmp_path / "config_i-care_2_1.yml"

    general_cfg.write_text(
        "\n".join(
            [
                "paths:",
                f"  download_root: {tmp_path.as_posix()}/received",
                f"  metadata_root: {tmp_path.as_posix()}/metadata",
                "network:",
                "  timeout_seconds: 5",
                "  retries: 1",
                "  retry_backoff_seconds: 0.0",
                "  chunk_size_bytes: 1024",
                "  user_agent: test-agent",
            ]
        ),
        encoding="utf-8",
    )
    dataset_cfg.write_text(
        "\n".join(
            [
                "dataset:",
                "  slug: i-care",
                "  version: '2.1'",
                "subset:",
                "  strategy: first_n_patients",
                "  patient_count: 1",
                "  max_eeg_hours_per_patient: 0",
                "download:",
                "  output_subdir: smoke",
                "  dry_run: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="subset.max_eeg_hours_per_patient must be > 0 when set"):
        mod.run_download(general_cfg, dataset_cfg)


def test_run_download_requires_max_when_min_eeg_hours_is_set(tmp_path: Path) -> None:
    general_cfg = tmp_path / "config_ingest.yml"
    dataset_cfg = tmp_path / "config_i-care_2_1.yml"

    general_cfg.write_text(
        "\n".join(
            [
                "paths:",
                f"  download_root: {tmp_path.as_posix()}/received",
                f"  metadata_root: {tmp_path.as_posix()}/metadata",
                "network:",
                "  timeout_seconds: 5",
                "  retries: 1",
                "  retry_backoff_seconds: 0.0",
                "  chunk_size_bytes: 1024",
                "  user_agent: test-agent",
            ]
        ),
        encoding="utf-8",
    )
    dataset_cfg.write_text(
        "\n".join(
            [
                "dataset:",
                "  slug: i-care",
                "  version: '2.1'",
                "subset:",
                "  strategy: first_n_patients",
                "  patient_count: 1",
                "  min_eeg_hours_per_patient: 12",
                "download:",
                "  output_subdir: smoke",
                "  dry_run: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires subset.max_eeg_hours_per_patient"):
        mod.run_download(general_cfg, dataset_cfg)


def test_run_download_explicit_strategy_requires_patient_ids(tmp_path: Path) -> None:
    general_cfg = tmp_path / "config_ingest.yml"
    dataset_cfg = tmp_path / "config_i-care_2_1.yml"

    general_cfg.write_text(
        "\n".join(
            [
                "paths:",
                f"  download_root: {tmp_path.as_posix()}/received",
                f"  metadata_root: {tmp_path.as_posix()}/metadata",
                "network:",
                "  timeout_seconds: 5",
                "  retries: 1",
                "  retry_backoff_seconds: 0.0",
                "  chunk_size_bytes: 1024",
                "  user_agent: test-agent",
            ]
        ),
        encoding="utf-8",
    )
    dataset_cfg.write_text(
        "\n".join(
            [
                "dataset:",
                "  slug: i-care",
                "  version: '2.1'",
                "subset:",
                "  strategy: explicit_patient_ids",
                "download:",
                "  output_subdir: smoke",
                "  dry_run: true",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="subset.patient_ids must contain at least one patient"):
        mod.run_download(general_cfg, dataset_cfg)
