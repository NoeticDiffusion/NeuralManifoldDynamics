from __future__ import annotations

import json
from pathlib import Path
import struct
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families import (  # noqa: E402
    far_ext_002c_ds003670_native_rail_methods as far_ext,
)


def _write_flat_set(path: Path, *, embedded: bool = False) -> None:
    import scipy.io as sio

    path.parent.mkdir(parents=True, exist_ok=True)
    data: object = np.zeros((2, 2), dtype=np.float32) if embedded else "x.fdt"
    sio.savemat(
        str(path),
        {
            "srate": 2000.0,
            "pnts": 1000.0,
            "nbchan": 2.0,
            "xmin": 0.0,
            "data": data,
            "chanlocs": np.array(
                [{"labels": "E1"}, {"labels": "E2"}],
                dtype=object,
            ),
        },
    )


def _write_fixture(
    tmp_path: Path,
    *,
    embedded: bool = False,
) -> tuple[Path, Path, Path]:
    metadata_root = tmp_path / "metadata" / "ds003670"
    eeg_root = tmp_path / "eeg" / "ds003670"
    far002b_path = tmp_path / "far_ext_002b.json"
    sidecar = metadata_root / "sub-001" / "ses-01" / "eeg"
    sidecar.mkdir(parents=True)
    (sidecar / "sub-001_ses-01_task-test_eeg.json").write_text(
        json.dumps(
            {
                "ManufacturersModelName": (
                    "ANT Neuro-eego, Ref Input Range: 150-1000 mV pp"
                ),
                "EEGChannelCount": 32,
                "SoftwareFilters": "n/a",
            }
        ),
        encoding="utf-8",
    )
    (sidecar / "sub-001_ses-01_task-test_channels.tsv").write_text(
        "name\ttype\tunits\nE1\tEEG\tn/a\n",
        encoding="utf-8",
    )
    set_path = (
        eeg_root
        / "sub-001"
        / "ses-01"
        / "eeg"
        / "sub-001_ses-01_task-test_eeg.set"
    )
    _write_flat_set(set_path, embedded=embedded)
    (set_path.with_suffix(".fdt")).write_bytes(b"fixture payload is never opened")
    set_relative = set_path.relative_to(eeg_root).as_posix()
    fdt_relative = set_path.with_suffix(".fdt").relative_to(eeg_root).as_posix()
    far002b_path.write_text(
        json.dumps(
            {
                "protocol_id": "FAR-EXT-002B",
                "dataset_id": "ds003670",
                "global_status": "NOT_TESTABLE",
                "payload_binding": {
                    "status": "PASS",
                    "expected_payloads": [set_relative, fdt_relative],
                },
            }
        ),
        encoding="utf-8",
    )
    return metadata_root, eeg_root, far002b_path


def test_methods_audit_is_unresolved_without_float_export_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root, eeg_root, far002b_path = _write_fixture(tmp_path)
    monkeypatch.setattr(
        far_ext,
        "EXPECTED_ENTRY_CERTIFICATE_SHA256",
        far_ext.sha256_file(far002b_path),
    )
    result = far_ext.audit_far_ext_002c(
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        far002b_path=far002b_path,
        fetch_documents=False,
        probe_remote_header=False,
    )
    assert result["global_status"] == "NOT_TESTABLE"
    assert result["global_reason"] == "native_float_rail_criterion_unresolved"
    assert result["documentation_status"] == "PARTIAL"
    assert result["criterion_assessment"]["field_ledger"]["export_unit"][
        "status"
    ] == "UNRESOLVED"
    assert result["local_sidecar_audit"]["range_menu_not_selected"] is True
    assert result["local_set_header_audit"]["fdt_payload_opened"] is False
    assert result["audit_scope"]["fdt_samples_inspected"] is False


def test_metadata_sidecars_outside_payload_allowlist_are_ignored(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root, eeg_root, far002b_path = _write_fixture(tmp_path)
    extra = metadata_root / "sub-011" / "ses-01" / "eeg"
    extra.mkdir(parents=True)
    (extra / "sub-011_ses-01_task-extra_eeg.json").write_text(
        json.dumps({"EEGChannelCount": 32}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        far_ext,
        "EXPECTED_ENTRY_CERTIFICATE_SHA256",
        far_ext.sha256_file(far002b_path),
    )
    result = far_ext.audit_far_ext_002c(
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        far002b_path=far002b_path,
        fetch_documents=False,
        probe_remote_header=False,
    )
    assert result["local_sidecar_audit"]["status"] == "PASS"
    assert result["local_sidecar_audit"][
        "ignored_sidecars_outside_payload_allowlist"
    ] == ["sub-011/ses-01/eeg/sub-011_ses-01_task-extra_eeg.json"]


def test_embedded_set_samples_fail_closed_before_mat_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_root, eeg_root, far002b_path = _write_fixture(
        tmp_path,
        embedded=True,
    )
    monkeypatch.setattr(
        far_ext,
        "EXPECTED_ENTRY_CERTIFICATE_SHA256",
        far_ext.sha256_file(far002b_path),
    )
    result = far_ext.audit_far_ext_002c(
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        far002b_path=far002b_path,
        fetch_documents=False,
        probe_remote_header=False,
    )
    assert result["global_status"] == "AUDIT_SCOPE_FAILED"
    assert "set_header_contains_embedded_samples" in str(
        result["local_set_header_audit"]["errors"]
    )


def test_zip64_extra_consumes_only_present_fields() -> None:
    all_fields = struct.pack(
        "<HHQQQI",
        0x0001,
        28,
        11,
        22,
        33,
        44,
    )
    assert far_ext._zip64_extra(
        all_fields,
        uncompressed_overflow=True,
        compressed_overflow=True,
        offset_overflow=True,
        disk_overflow=True,
    ) == {
        "uncompressed_size": 11,
        "compressed_size": 22,
        "local_header_offset": 33,
        "disk_start": 44,
    }
    sizes_only = struct.pack("<HHQQ", 0x0001, 16, 11, 22)
    assert far_ext._zip64_extra(
        sizes_only,
        uncompressed_overflow=True,
        compressed_overflow=True,
        offset_overflow=False,
        disk_overflow=False,
    ) == {
        "uncompressed_size": 11,
        "compressed_size": 22,
    }


def test_cnt_parser_stops_at_data_chunk_header() -> None:
    header = (
        b"RIFF"
        + struct.pack("<I", 40)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 12)
        + b"metadata1234"
        + b"data"
        + struct.pack("<I", 999)
        + b"signal bytes must not be parsed"
    )
    data_offset, data_size, chunk_id = far_ext._cnt_data_chunk_info(header)
    assert header[data_offset:data_offset + 4] == b"data"
    assert chunk_id == b"data"
    assert data_size == 999
    assert data_offset + 8 < len(header)


def test_cnt_parser_accepts_eeprobe_form_and_stops_at_list() -> None:
    header = (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"CNT "
        + b"LIST"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"raw3"
        + b"sample bytes are never consumed"
    )
    boundary_offset, boundary_size, chunk_id = far_ext._cnt_data_chunk_info(
        header
    )
    assert boundary_offset == 12
    assert boundary_size == 0xFFFFFFFF
    assert chunk_id == b"LIST"


def test_cnt_parser_does_not_require_standard_wave_form() -> None:
    header = (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"ABCD"
        + b"LIST"
        + struct.pack("<I", 0xFFFFFFFF)
    )
    boundary_offset, _boundary_size, chunk_id = far_ext._cnt_data_chunk_info(
        header
    )
    assert boundary_offset == 12
    assert chunk_id == b"LIST"


def test_cnt_parser_stops_at_cnt_header_chunk() -> None:
    header = (
        b"RIFF"
        + struct.pack("<I", 0xFFFFFFFF)
        + b"ABCD"
        + b"CNT "
        + struct.pack("<I", 0xFFFFFFFF)
        + b"raw3"
    )
    boundary_offset, _boundary_size, chunk_id = far_ext._cnt_data_chunk_info(
        header
    )
    assert boundary_offset == 12
    assert chunk_id == b"CNT "


def test_criterion_can_pass_only_with_complete_unique_binding() -> None:
    binding = {
        "export_unit": "uV",
        "selected_input_range": "1000 mVpp",
        "adc_or_export_transform": "documented transform",
        "rail_low": -500.0,
        "rail_high": 500.0,
        "rail_test": "documented tolerance",
    }
    result = far_ext._criterion_assessment(
        sidecars={},
        sets={"explicit_native_rail_binding": binding},
        documents=[],
        remote={},
    )
    assert result["native_float_rail_status"] == "PASS"
    assert result["reason"] == "native_float_rail_criterion_bound"
