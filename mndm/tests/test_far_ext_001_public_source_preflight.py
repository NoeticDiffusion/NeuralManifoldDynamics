from __future__ import annotations

import json
from pathlib import Path

from mndm.dynamical_families import far_ext_001_public_source_preflight as far_ext


def _write_curve_fixture(root: Path, *, with_units: bool = True) -> Path:
    source = root / "ds003670"
    source.mkdir(parents=True)
    (source / "dataset_description.json").write_text(
        json.dumps({"Name": "fixture"}),
        encoding="utf-8",
    )
    (source / "participants.tsv").write_text("participant_id\nsub-01\nsub-02\n", encoding="utf-8")
    for subject in ("01", "02"):
        eeg_dir = source / f"sub-{subject}" / "eeg"
        eeg_dir.mkdir(parents=True)
        (eeg_dir / f"sub-{subject}_task-stim_eeg.json").write_text(
            json.dumps({"SamplingFrequency": 2000, "RecordingDuration": 100.0}),
            encoding="utf-8",
        )
        rows = [
            ("0", "0.5 mA" if with_units else "0.5", "motor", "dc", "stim_start", "16"),
            ("35", "0.5 mA" if with_units else "0.5", "motor", "dc", "stim_stop", "32"),
            ("50", "1 mA" if with_units else "1", "motor", "dc", "stim_start", "16"),
            ("85", "1 mA" if with_units else "1", "motor", "dc", "stim_stop", "32"),
            ("100", "0.5 mA" if with_units else "0.5", "motor", "dc", "stim_start", "16"),
            ("135", "0.5 mA" if with_units else "0.5", "motor", "dc", "stim_stop", "32"),
        ]
        body = "onset\tcurrent\ttarget\twaveform\ttrial_type\ttrigger\n"
        body += "\n".join("\t".join(row) for row in rows) + "\n"
        events = eeg_dir / f"sub-{subject}_task-stim_events.tsv"
        events.write_text(body, encoding="utf-8")
        sidecar = {"current": {"Units": "mA"}} if with_units else {}
        (eeg_dir / f"sub-{subject}_task-stim_events.json").write_text(
            json.dumps(sidecar),
            encoding="utf-8",
        )
    return source


def test_curve_census_requires_explicit_units_and_time_support(tmp_path: Path):
    source = _write_curve_fixture(tmp_path)

    result = far_ext.run_inventory({"ds003670": source})

    dataset = result["datasets"][0]
    assert result["gate_status"] == "NOT_TESTABLE"
    assert dataset["classification"] == "CURVE_CANDIDATE"
    family = dataset["families"][0]
    assert family["rho_levels"] == [0.5, 1.0]
    assert family["min_subjects_across_rho"] == 2
    assert family["clean_isolated_post_horizon_sec"] == 10.0
    assert family["nmd_window_sec"] == 8.0
    assert family["isolated_to_nmd_ratio"] > 1.0
    assert family["zero_or_sham_present"] is False
    assert dataset["audit_scope"]["signal_payloads_opened"] is False


def test_missing_units_remains_source_uncertain(tmp_path: Path):
    source = _write_curve_fixture(tmp_path, with_units=False)

    result = far_ext.run_inventory({"ds003670": source})

    dataset = result["datasets"][0]
    assert dataset["classification"] == "SOURCE_UNCERTAIN"
    assert dataset["families"][0]["rho_units_known"] is False


def test_payload_presence_is_fail_closed(tmp_path: Path):
    source = _write_curve_fixture(tmp_path)
    (source / "sub-01" / "eeg" / "sub-01_task-stim_eeg.edf").write_bytes(b"not opened")

    result = far_ext.run_inventory({"ds003670": source})

    dataset = result["datasets"][0]
    assert dataset["classification"] == "SOURCE_UNCERTAIN"
    assert dataset["audit_scope"]["signal_payloads_present"] is True
    assert result["gate_status"] == "NOT_TESTABLE"


def test_all_four_sources_are_in_scope_and_missing_roots_are_uncertain():
    result = far_ext.run_inventory({})

    assert set(result["source_scope"]) == {
        "ds003670",
        "ds006519",
        "ds005169",
        "ds008037",
    }
    assert result["gate_status"] == "NOT_TESTABLE"
    assert result["classification_counts"] == {"SOURCE_UNCERTAIN": 4}


def test_download_contract_has_required_and_optional_metadata_only_patterns():
    config = far_ext.metadata_download_config()
    required = config["download"]["metadata_only_patterns"]
    optional = config["download"]["metadata_only_optional_patterns"]

    assert required == ["**/dataset_description.json", "**/*_events.tsv"]
    assert all(pattern.endswith((".json", ".tsv")) for pattern in optional)


def _write_simple_ds003670_events(
    root: Path,
    body: str,
) -> dict[str, Path]:
    source = root / "ds003670"
    eeg_dir = source / "sub-01" / "eeg"
    eeg_dir.mkdir(parents=True)
    (source / "dataset_description.json").write_text("{}", encoding="utf-8")
    (eeg_dir / "sub-01_task-stim_events.tsv").write_text(body, encoding="utf-8")
    (eeg_dir / "sub-01_task-stim_eeg.json").write_text(
        json.dumps({"RecordingDuration": 100.0}),
        encoding="utf-8",
    )
    return {"ds003670": source}


def test_unresolved_v_cannot_be_promoted_to_curve(tmp_path: Path):
    roots = _write_simple_ds003670_events(
        tmp_path,
        "onset\tcurrent\ttrigger\n"
        "0\t0.5 mA\t16\n"
        "35\t0.5 mA\t32\n"
        "50\t1 mA\t16\n"
        "85\t1 mA\t32\n",
    )

    result = far_ext.run_inventory(roots)

    family = result["datasets"][0]["families"][0]
    assert family["v_resolved"] is False
    assert family["classification"] == "SOURCE_UNCERTAIN"


def test_overlapping_trigger_pairs_are_not_given_recording_remainder(tmp_path: Path):
    roots = _write_simple_ds003670_events(
        tmp_path,
        "onset\tcurrent\ttrigger\ttarget\n"
        "0\t0.5 mA\t16\tmotor\n"
        "10\t0.5 mA\t32\tmotor\n"
        "12\t1 mA\t16\tmotor\n"
        "20\t1 mA\t32\tmotor\n",
    )

    result = far_ext.run_inventory(roots)

    dataset = result["datasets"][0]
    assert dataset["classification"] == "TIMING_UNRESOLVED"
    assert dataset["event_tuples"][0]["overlap_with_next_stim"] is True
    assert dataset["event_tuples"][0]["T_isolated"] is None


def test_ds003670_without_locked_trigger_grammar_is_timing_unresolved(tmp_path: Path):
    roots = _write_simple_ds003670_events(
        tmp_path,
        "onset\tduration\tcurrent\ttarget\ttrial_type\n"
        "0\t5\t0.5 mA\tmotor\tstim\n"
        "20\t5\t1 mA\tmotor\tstim\n",
    )

    result = far_ext.run_inventory(roots)

    dataset = result["datasets"][0]
    assert dataset["classification"] == "TIMING_UNRESOLVED"
    assert all(
        event["timing_grammar"] == "unresolved_missing_start_stop_trigger_grammar"
        for event in dataset["event_tuples"]
    )


def test_exact_nmd_window_is_incompatible(tmp_path: Path):
    roots = _write_simple_ds003670_events(
        tmp_path,
        "onset\tcurrent\ttrigger\ttarget\n"
        "0\t0.5 mA\t16\tmotor\n"
        "3\t0.5 mA\t32\tmotor\n"
        "16\t1 mA\t16\tmotor\n"
        "19\t1 mA\t32\tmotor\n",
    )

    result = far_ext.run_inventory(roots)

    family = result["datasets"][0]["families"][0]
    assert family["clean_isolated_post_horizon_sec"] == 8.0
    assert family["classification"] == "NMD_TIMEBASE_INCOMPATIBLE"


def test_ds003670_public_trigger_schema_is_timed_but_not_amplitude_promoted(
    tmp_path: Path,
):
    roots = _write_simple_ds003670_events(
        tmp_path,
        "onset\tvalue\ttrial_type\n"
        "0\t16\tStimulation Start\n"
        "35\t32\tStimulation Stop\n"
        "50\t16\tStimulation Start\n"
        "85\t32\tStimulation Stop\n",
    )

    result = far_ext.run_inventory(roots)

    dataset = result["datasets"][0]
    assert len(dataset["event_tuples"]) == 2
    assert dataset["event_tuples"][0]["timing_grammar"].startswith("trigger_16_32")
    assert dataset["event_tuples"][0]["t_on"] == 0.0
    assert dataset["event_tuples"][0]["t_off"] == 40.0
    assert dataset["event_tuples"][0]["rho"] is None
    assert dataset["families"][0]["v_resolved"] is False
    assert dataset["classification"] == "SOURCE_UNCERTAIN"
    assert "ds003670" not in result["promotion_candidates"]


def test_ds006519_prestim_poststim_rows_are_paired_without_poststim_as_stim(
    tmp_path: Path,
):
    source = tmp_path / "ds006519" / "sub-01" / "ieeg"
    source.mkdir(parents=True)
    (source.parent.parent / "dataset_description.json").write_text("{}", encoding="utf-8")
    (source / "sub-01_task-stim_events.tsv").write_text(
        "label\tonset\tduration\tstim_contacts\tstim_frequency\tstim_intensity\n"
        "prestim\t0\t9\tn/a\tn/a\tn/a\n"
        "poststim\t14\t0\tA'09-A'10\t50\t3\n",
        encoding="utf-8",
    )

    result = far_ext.run_inventory({"ds006519": source.parent.parent})

    dataset = next(item for item in result["datasets"] if item["dataset_id"] == "ds006519")
    assert len(dataset["event_tuples"]) == 1
    event = dataset["event_tuples"][0]
    assert event["t_on"] == 9.0
    assert event["t_off"] == 14.0
    assert event["rho"] == 3.0
    assert event["v"]["stim_contacts"] == "A'09-A'10"
    assert dataset["classification"] == "SOURCE_UNCERTAIN"


def test_ds005169_stim_row_uses_poststim_boundary(tmp_path: Path):
    source = tmp_path / "ds005169" / "sub-01" / "ieeg"
    source.mkdir(parents=True)
    (source.parent.parent / "dataset_description.json").write_text("{}", encoding="utf-8")
    (source / "sub-01_task-stim_events.tsv").write_text(
        "label\tonset\tduration\tstim_intensity\n"
        "prestim\t0\t8\tn/a\n"
        "B'03-B'04\t8\t0\t0.75\n"
        "poststim\t13\t0\tn/a\n",
        encoding="utf-8",
    )

    result = far_ext.run_inventory({"ds005169": source.parent.parent})

    dataset = next(item for item in result["datasets"] if item["dataset_id"] == "ds005169")
    assert len(dataset["event_tuples"]) == 1
    event = dataset["event_tuples"][0]
    assert event["t_on"] == 8.0
    assert event["t_off"] == 13.0
    assert event["rho"] == 0.75
    assert event["v"]["label"] == "B'03-B'04"
