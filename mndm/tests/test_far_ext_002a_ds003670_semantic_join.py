from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mndm.dynamical_families import far_ext_002a_ds003670_semantic_join as far_ext


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_events(
    source: Path,
    file_num: str,
    *,
    one_block: bool = False,
    path_file_num: str | None = None,
) -> Path:
    path_file_num = path_file_num or file_num
    subject = path_file_num[:2]
    session = path_file_num[2:]
    eeg_dir = source / f"sub-{int(subject):03d}" / f"ses-{int(session):02d}" / "eeg"
    eeg_dir.mkdir(parents=True, exist_ok=True)
    rows = [(0.0, "2")]
    block_starts = [10.0] if one_block else [10.0, 1000.0, 2000.0]
    for block_start in block_starts:
        rows.append((block_start, "2"))
        for trial in range(4):
            onset = block_start + 1.0 + trial * 30.0
            rows.extend(((onset, "16"), (onset + 10.0, "32")))
    body = "onset\tvalue\ttrial_type\n"
    body += "\n".join(f"{onset}\t{value}\tstim" for onset, value in rows) + "\n"
    path = eeg_dir / f"sub-{int(subject):03d}_ses-{int(session):02d}_task-GXtESCTT_events.tsv"
    path.write_text(body, encoding="utf-8")
    return path


def _write_bound_fixture(
    tmp_path: Path,
    files: tuple[str, ...] = ("0104", "0801"),
    path_file_nums: dict[str, str] | None = None,
) -> tuple[Path, dict]:
    source = tmp_path / "ds003670"
    source.mkdir()
    (source / "dataset_description.json").write_text(
        json.dumps({"Name": "fixture"}), encoding="utf-8"
    )
    for file_num in files:
        events_path = _write_events(
            source,
            file_num,
            path_file_num=(path_file_nums or {}).get(file_num),
        )
        protocol = far_ext._protocol_for_file(file_num)
        assert protocol is not None
        sidecar = events_path.with_name(
            events_path.name.replace("_events.tsv", "_eeg.json")
        )
        sidecar.write_text(
            json.dumps(
                {
                    "SubjectArtefactDescription": (
                        f"File: {file_num};"
                        f"Montages: {', '.join(block['stim_type'] for block in protocol['blocks'])};"
                        f"Stim Amplitude(mA): "
                        f"{', '.join(str(block['rho']) for block in protocol['blocks'])};"
                    )
                }
            ),
            encoding="utf-8",
        )
    records = [
        {
            "relative_path": str(path.relative_to(source)).replace("\\", "/"),
            "sha256": _sha256(path),
        }
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]
    far001 = {
        "protocol_id": "FAR-EXT-001",
        "nmd_window_sec": {"eeg": 8.0},
        "datasets": [
            {
                "dataset_id": "ds003670",
                "source_root": str(source),
                "metadata_inventory": {"metadata_file_records": records},
                "audit_scope": {"signal_payloads_present": False},
            }
        ],
    }
    return source, far001


def test_protocol_join_keeps_frequency_in_v_and_finds_limited_curve(tmp_path: Path):
    source, far001 = _write_bound_fixture(
        tmp_path,
        files=("0104", "0201", "0403", "0801", "1003"),
    )

    result = far_ext.audit_ds003670(source, far001)

    assert result["source_binding"]["status"] == "PASS"
    assert result["global_status"] == "DS003670_LIMITED_CURVE_SEMANTICS_PASS"
    families = {
        (family["v"]["target"], family["v"]["frequency_hz"]): family
        for family in result["families"]
    }
    frontal_5 = families[("frontal", 5)]
    assert frontal_5["rho_levels_mA"] == [0.5, 1.0]
    assert frontal_5["min_biological_units_across_rho"] == 2
    assert frontal_5["semantic_status"] == "DS003670_LIMITED_CURVE_SEMANTICS_PASS"
    assert ("frontal", 30) in families


def test_support_requires_two_biological_units_per_rho(tmp_path: Path):
    source, far001 = _write_bound_fixture(tmp_path, files=("0104",))

    result = far_ext.audit_ds003670(source, far001)

    assert result["global_status"] == "POINT_SEMANTICS_ONLY"
    assert all(
        family["min_biological_units_across_rho"] < 2
        for family in result["families"]
    )


def test_block_cardinality_is_fail_closed(tmp_path: Path):
    source, far001 = _write_bound_fixture(tmp_path, files=())
    event_path = _write_events(source, "0104", one_block=True)
    far001["datasets"][0]["metadata_inventory"]["metadata_file_records"] = [
        {
            "relative_path": str(path.relative_to(source)).replace("\\", "/"),
            "sha256": _sha256(path),
        }
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]

    result = far_ext.audit_ds003670(source, far001)

    assert result["source_binding"]["status"] == "PASS"
    assert result["global_status"] == "SOURCE_SEMANTICS_UNRESOLVED"
    assert result["families"] == []
    assert event_path.exists()


def test_0102_technical_f0_block_is_not_semantically_promoted(tmp_path: Path):
    source, far001 = _write_bound_fixture(tmp_path, files=("0102",))

    result = far_ext.audit_ds003670(source, far001)

    technical = [
        event
        for event in result["events"]
        if event["block"] == 3
    ]
    assert technical
    assert all(event["semantic_status"] == "TECHNICAL_ERROR_REPORTED" for event in technical)


def test_0201_only_first_trial_uses_documented_30_second_ramp(tmp_path: Path):
    source, far001 = _write_bound_fixture(tmp_path, files=("0201",))

    result = far_ext.audit_ds003670(source, far001)

    block_one = [
        event for event in result["events"]
        if event["block"] == 1
    ]
    assert block_one[0]["ramp_down_sec"] == 30.0
    assert block_one[0]["semantic_status"] == "TECHNICAL_ERROR_REPORTED"
    assert block_one[1]["ramp_down_sec"] == 5.0
    assert block_one[1]["semantic_status"] == "PASS"


def test_payload_binding_rejects_fdt_and_does_not_return_families(tmp_path: Path):
    source, far001 = _write_bound_fixture(tmp_path)
    payload = source / "sub-001" / "ses-04" / "eeg" / "sub-001_ses-04_task-GXtESCTT_eeg.fdt"
    payload.write_bytes(b"not opened")

    result = far_ext.audit_ds003670(source, far001)

    assert result["source_binding"]["status"] == "SOURCE_BINDING_FAILED"
    assert str(payload.relative_to(source)).replace("\\", "/") in result["source_binding"]["signal_payloads"]
    assert result["families"] == []
    assert result["global_status"] == "SOURCE_SEMANTICS_UNRESOLVED"


def test_publication_and_sidecar_rho_conflict_is_unresolved(tmp_path: Path):
    source, far001 = _write_bound_fixture(tmp_path, files=("0603",))
    sidecar = next(source.rglob("*_eeg.json"))
    sidecar.write_text(
        sidecar.read_text(encoding="utf-8").replace(
            "Stim Amplitude(mA): 1.0, 1.0, 1.0",
            "Stim Amplitude(mA): 1.0, 1.0, 0.5",
        ),
        encoding="utf-8",
    )
    far001["datasets"][0]["metadata_inventory"]["metadata_file_records"] = [
        {
            "relative_path": str(path.relative_to(source)).replace("\\", "/"),
            "sha256": _sha256(path),
        }
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]

    result = far_ext.audit_ds003670(source, far001)

    assert result["source_binding"]["status"] == "PASS"
    assert result["families"] == []
    assert result["global_status"] == "SOURCE_SEMANTICS_UNRESOLVED"
    assert all(
        event["semantic_status"] == "SOURCE_SEMANTICS_UNRESOLVED"
        for event in result["events"]
    )


def test_binding_normalizes_far001_paths_and_ignores_non_inventory_notes(tmp_path: Path):
    source, far001 = _write_bound_fixture(tmp_path, files=("0104",))
    (source / "README").write_text("metadata note", encoding="utf-8")
    (source / "CHANGES").write_text("metadata note", encoding="utf-8")
    for record in far001["datasets"][0]["metadata_inventory"]["metadata_file_records"]:
        record["relative_path"] = record["relative_path"].replace("/", "\\")

    result = far_ext.audit_ds003670(source, far001)

    assert result["source_binding"]["status"] == "PASS"


def test_matching_source_file_can_differ_from_bids_session_label(tmp_path: Path):
    source, far001 = _write_bound_fixture(
        tmp_path,
        files=("0505",),
        path_file_nums={"0505": "0503"},
    )

    result = far_ext.audit_ds003670(source, far001)

    assert result["source_binding"]["status"] == "PASS"
    assert result["events"][0]["path_file_num"] == "0503"
    assert result["events"][0]["source_file_num"] == "0505"
    assert all(event["semantic_status"] == "PASS" for event in result["events"])
