"""Synthetic tests for the FAR-001B metadata-only source audit."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.far_source_metadata_audit import (  # noqa: E402
    CURVE_CANDIDATE,
    DIRECTION_ONLY,
    POINT_OR_LIMITED_CANDIDATE,
    SOURCE_UNCERTAIN,
    _classify_levels,
    _audit_dandi_000009,
    _audit_dandi_000458,
    _audit_ds005917,
    _audit_ds006623,
    run_inventory,
)
from od_far_001b_source_metadata_gate import run_gate  # noqa: E402


h5py = pytest.importorskip("h5py")


def _write_nwb_fixture(
    path: Path,
    *,
    currents: list[str] | None = None,
    powers: list[float] | None = None,
    validity: list[bool] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        trials = handle.create_group("intervals").create_group("trials")
        if currents is not None:
            values = trials.create_dataset(
                "estim_current",
                data=currents,
                dtype=string_dtype,
            )
            values.attrs["description"] = (
                "electrical stimulation current (μA), if applicable"
            )
            trials.create_dataset(
                "stimulus_type",
                data=["electrical"] * len(currents),
                dtype=string_dtype,
            )
            trials.create_dataset(
                "behavioral_epoch",
                data=["awake"] * len(currents),
                dtype=string_dtype,
            )
            trials.create_dataset(
                "estim_target_depth",
                data=["superficial"] * len(currents),
                dtype=string_dtype,
            )
            trials.create_dataset(
                "estim_target_region",
                data=["MOs"] * len(currents),
                dtype=string_dtype,
            )
            trials.create_dataset(
                "stimulus_description",
                data=["biphasic"] * len(currents),
                dtype=string_dtype,
            )
            trials.create_dataset(
                "is_valid",
                data=validity or [True] * len(currents),
            )
            trials.create_dataset(
                "is_running",
                data=[False] * len(currents),
            )
            trials.create_dataset(
                "start_time",
                data=[float(index) for index in range(len(currents))],
            )
            trials.create_dataset(
                "stop_time",
                data=[float(index) + 0.001 for index in range(len(currents))],
            )
        if powers is not None:
            values = trials.create_dataset("photo_stim_power", data=powers)
            values.attrs["description"] = "(mW) stimulation power"
            trials.create_dataset(
                "photo_stim_period",
                data=["N/A"] * len(powers),
                dtype=string_dtype,
            )
            trials.create_dataset(
                "photo_stim_type",
                data=["stimulation"] * len(powers),
                dtype=string_dtype,
            )
            trials.create_dataset(
                "stim_present",
                data=[0 if power == 0 else 1 for power in powers],
            )
            trials.create_dataset(
                "start_time",
                data=[float(index) for index in range(len(powers))],
            )
            trials.create_dataset(
                "stop_time",
                data=[float(index) + 1.0 for index in range(len(powers))],
            )
        handle.create_group("acquisition")


def _write_bids_metadata(root: Path) -> None:
    root.mkdir(parents=True)
    (root / "README.md").write_text(
        "The study uses graded propofol sedation and effect-site concentrations.\n",
        encoding="utf-8",
    )
    (root / "dataset_description.json").write_text(
        json.dumps({"Name": "fixture"}),
        encoding="utf-8",
    )


def test_dandi_000458_requires_native_units_and_timing_for_curve(
    tmp_path: Path,
) -> None:
    root = tmp_path / "000458"
    _write_nwb_fixture(
        root / "sub-001" / "sub-001.nwb",
        currents=["10", "20", "30", "20"],
    )

    result = _audit_dandi_000458(root)

    assert result["classification"] == CURVE_CANDIDATE
    identity = result["identities"][0]
    assert identity["classification"] == CURVE_CANDIDATE
    assert identity["levels"] == [10.0, 20.0, 30.0]
    assert identity["rho_unit"] == "μA"
    assert identity["timing_complete"] is True


def test_level_count_and_units_control_point_classification() -> None:
    assert (
        _classify_levels(
            levels=[10.0],
            unit="μA",
            timing_complete=True,
            repetitions=3,
        )
        == POINT_OR_LIMITED_CANDIDATE
    )
    assert (
        _classify_levels(
            levels=[10.0, 20.0],
            unit=None,
            timing_complete=True,
            repetitions=3,
        )
        == SOURCE_UNCERTAIN
    )


def test_dandi_000458_excludes_invalid_trials_from_rho_census(
    tmp_path: Path,
) -> None:
    root = tmp_path / "000458"
    _write_nwb_fixture(
        root / "sub-001" / "sub-001.nwb",
        currents=["10", "20", "30", "40"],
        validity=[True, True, False, False],
    )

    result = _audit_dandi_000458(root)

    assert result["identities"][0]["levels"] == [10.0, 20.0]
    assert result["identities"][0]["classification"] == POINT_OR_LIMITED_CANDIDATE
    assert result["identities"][0]["invalid_electrical_rows_excluded"] == 2


def test_dandi_000009_does_not_promote_trial_window_to_pulse_timing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "000009"
    _write_nwb_fixture(
        root / "sub-001" / "sub-001.nwb",
        powers=[0.0, 1.5, 0.0],
    )

    result = _audit_dandi_000009(root)

    assert result["classification"] == SOURCE_UNCERTAIN
    assert result["rho_unit"] == "mW"
    assert result["rho_zero_or_sham"] is True
    assert result["pulse_onset_offset_fields"] == []
    assert result["conditional_classification_if_trial_window_accepted"] == (
        POINT_OR_LIMITED_CANDIDATE
    )


def test_ds006623_numeric_protocol_without_units_is_uncertain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ds006623"
    _write_bids_metadata(root)
    (root / "Participant_Info.csv").write_text(
        "Subjects,Infusion Protocol,LOR ESC,Propofol dosage,Infusion Stop\n"
        "sub-01,0.4 to 2.4 (0.4),1.6,LOR 1.6; Maintain 2.4,Right at task3\n",
        encoding="utf-8",
    )
    (root / "LOR_ROR_Timing.csv").write_text(
        "Subject,LOR time (TR in task2),ROR time (TR in task3)\n"
        "sub-01,100,200\n",
        encoding="utf-8",
    )
    (root / "sub-01").mkdir()
    (root / "sub-01" / "sub-01_task-rest_bold.json").write_text(
        json.dumps({"RepetitionTime": 0.8}),
        encoding="utf-8",
    )

    result = _audit_ds006623(root)

    assert result["classification"] == SOURCE_UNCERTAIN
    assert result["candidate_rho_units"] == []
    assert result["timed_dose_series_present"] is False
    assert result["effect_site_concentration_text_claim"] is True


def test_ds005917_condition_labels_remain_direction_only(tmp_path: Path) -> None:
    root = tmp_path / "ds005917"
    _write_bids_metadata(root)
    (root / "README").write_text(
        "Baseline and post-infusion ketamine/placebo scans.\n",
        encoding="utf-8",
    )
    (root / "participants.tsv").write_text(
        "participant_id\tinfusion_1\tinfusion_2\n"
        "sub-01\td\tp\n",
        encoding="utf-8",
    )
    (root / "phenotype").mkdir()
    (root / "phenotype" / "phenotype.tsv").write_text(
        "participant_id\tsession_id\tMADRS_Total\n"
        "sub-01\tses-b0\t20\n",
        encoding="utf-8",
    )
    for session in ("ses-b0", "ses-d2", "ses-p2"):
        (root / "sub-01" / session / "func").mkdir(parents=True)

    result = _audit_ds005917(root)

    assert result["classification"] == DIRECTION_ONLY
    assert result["infusion_labels"] == ["d", "p"]
    assert result["rho_known"] is False
    assert result["placebo_sessions_present"] is True


def test_inventory_opens_far_002_only_after_explicit_candidate(
    tmp_path: Path,
) -> None:
    roots = {
        "dandi_000458": tmp_path / "000458",
        "dandi_000009": tmp_path / "000009",
        "ds006623": tmp_path / "ds006623",
        "ds005917": tmp_path / "ds005917",
    }
    _write_nwb_fixture(
        roots["dandi_000458"] / "sub-001" / "sub-001.nwb",
        currents=["10", "20", "30"],
    )
    _write_nwb_fixture(
        roots["dandi_000009"] / "sub-001" / "sub-001.nwb",
        powers=[0.0, 1.5],
    )
    _write_bids_metadata(roots["ds006623"])
    (roots["ds006623"] / "Participant_Info.csv").write_text(
        "Subjects,Infusion Protocol,LOR ESC,Propofol dosage,Infusion Stop\n",
        encoding="utf-8",
    )
    (roots["ds006623"] / "LOR_ROR_Timing.csv").write_text(
        "Subject,LOR time (TR in task2),ROR time (TR in task3)\n",
        encoding="utf-8",
    )
    _write_bids_metadata(roots["ds005917"])
    (roots["ds005917"] / "README").write_text("ketamine placebo\n", encoding="utf-8")
    (roots["ds005917"] / "participants.tsv").write_text(
        "participant_id\tinfusion_1\tinfusion_2\nsub-01\td\tp\n",
        encoding="utf-8",
    )
    (roots["ds005917"] / "phenotype").mkdir()
    (roots["ds005917"] / "phenotype" / "phenotype.tsv").write_text(
        "participant_id\tsession_id\nsub-01\tses-b0\n",
        encoding="utf-8",
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("FAR-001B fixture\n", encoding="utf-8")

    result = run_inventory(source_roots=roots, protocol_path=protocol)

    assert result["gate_status"] == "PASS"
    assert result["far_002_authorized"] is True
    assert all(dataset.get("identities") for dataset in result["datasets"])
    assert result["eligible_candidates"] == [
        {
            "dataset_id": "dandi_000458",
            "identity": "electrical_stimulation_current",
            "classification": CURVE_CANDIDATE,
        }
    ]
    assert result["fail_closed_assertions"]["signal_payloads_opened"] is False


def test_gate_requires_far000_and_far001_entry_certificates(tmp_path: Path) -> None:
    protocol = (
        tmp_path
        / "project/orthagonal_axis/orthagonal_dynamics/"
        "finite-amplitude_resilience/003_far_001b_source_specific_prereg.md"
    )
    protocol.parent.mkdir(parents=True)
    protocol.write_text("FAR-001B fixture\n", encoding="utf-8")
    far000 = tmp_path / "far000.json"
    far000.write_text(
        json.dumps({"status": "BLOCKED", "decision": {"far_001_authorized": False}}),
        encoding="utf-8",
    )
    far001 = tmp_path / "far001.json"
    far001.write_text(
        json.dumps({"protocol_id": "FAR-001", "gate_status": "NOT_TESTABLE"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="far_000_entry_criterion"):
        run_gate(
            repo_root=tmp_path,
            output_json=tmp_path / "result.json",
            output_report=tmp_path / "result.md",
            far000_certificate=far000,
            far001_certificate=far001,
            source_roots={
                dataset_id: tmp_path / f"missing-{dataset_id}"
                for dataset_id in (
                    "dandi_000458",
                    "dandi_000009",
                    "ds006623",
                    "ds005917",
                )
            },
        )


def test_gate_writes_archive_and_refuses_overwrite(tmp_path: Path) -> None:
    protocol = (
        tmp_path
        / "project/orthagonal_axis/orthagonal_dynamics/"
        "finite-amplitude_resilience/003_far_001b_source_specific_prereg.md"
    )
    protocol.parent.mkdir(parents=True)
    protocol.write_text("FAR-001B fixture\n", encoding="utf-8")
    far000 = tmp_path / "far000.json"
    far000.write_text(
        json.dumps({"status": "PASS", "decision": {"far_001_authorized": True}}),
        encoding="utf-8",
    )
    far001 = tmp_path / "far001.json"
    far001.write_text(
        json.dumps({"protocol_id": "FAR-001", "gate_status": "NOT_TESTABLE"}),
        encoding="utf-8",
    )
    roots = {
        dataset_id: tmp_path / f"missing-{dataset_id}"
        for dataset_id in (
            "dandi_000458",
            "dandi_000009",
            "ds006623",
            "ds005917",
        )
    }
    output_json = tmp_path / "result.json"
    output_report = tmp_path / "result.md"

    result = run_gate(
        repo_root=tmp_path,
        output_json=output_json,
        output_report=output_report,
        far000_certificate=far000,
        far001_certificate=far001,
        source_roots=roots,
    )

    assert result["gate_status"] == "NOT_TESTABLE"
    assert output_json.is_file()
    with pytest.raises(FileExistsError, match="refusing_to_overwrite"):
        run_gate(
            repo_root=tmp_path,
            output_json=output_json,
            output_report=tmp_path / "new-report.md",
            far000_certificate=far000,
            far001_certificate=far001,
            source_roots=roots,
        )
