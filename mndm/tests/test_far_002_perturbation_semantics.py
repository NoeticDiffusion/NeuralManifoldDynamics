"""Synthetic tests for the FAR-002 metadata-only semantics gate."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.far_perturbation_family_semantics import (  # noqa: E402
    TIMING_UNRESOLVED,
    audit_dandi_000458,
)
from od_far_002_perturbation_semantics_gate import run_gate  # noqa: E402


h5py = pytest.importorskip("h5py")


def _write_nwb_fixture(
    path: Path,
    *,
    currents: list[float],
    states: list[str],
    target_region: str = "MOs",
    target_depth: str = "superficial",
    stimulus_text: str = "single pulse electrical stimuli targeted to MOs",
    durations: list[float] | None = None,
    optional_columns: dict[str, list[str]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as handle:
        intervals = handle.create_group("intervals")
        trials = intervals.create_group("trials")
        n_rows = len(currents)
        trials.create_dataset(
            "behavioral_epoch",
            data=states,
            dtype=string_dtype,
        )
        current_dataset = trials.create_dataset("estim_current", data=currents)
        current_dataset.attrs["description"] = (
            "electrical stimulation current (μA), if applicable"
        )
        for name, values in {
            "estim_target_depth": [target_depth] * n_rows,
            "estim_target_region": [target_region] * n_rows,
            "stimulus_description": ["biphasic"] * n_rows,
            "stimulus_type": ["electrical"] * n_rows,
        }.items():
            trials.create_dataset(name, data=values, dtype=string_dtype)
        trials.create_dataset("is_valid", data=[True] * n_rows)
        trials.create_dataset(
            "start_time",
            data=[float(index) for index in range(n_rows)],
        )
        trials.create_dataset(
            "stop_time",
            data=[
                float(index) + (durations[index] if durations else 0.001)
                for index in range(n_rows)
            ],
        )
        for name, values in (optional_columns or {}).items():
            trials.create_dataset(name, data=values, dtype=string_dtype)
        general = handle.create_group("general")
        general.create_dataset(
            "stimulus",
            data=stimulus_text,
            dtype=string_dtype,
        )
        general.create_dataset(
            "experiment_description",
            data="in vivo electrophysiology during cortical electrical microstimulation",
            dtype=string_dtype,
        )
        handle.create_dataset(
            "session_description",
            data="EEG during stimulation",
            dtype=string_dtype,
        )
        acquisition = handle.create_group("acquisition")
        series = acquisition.create_group("ElectricalSeriesEEG")
        series.attrs["description"] = "voltage measured over time"
        series.create_group("data")


def _write_entry_fixture(tmp_path: Path) -> tuple[Path, Path]:
    protocol = (
        tmp_path
        / "project/orthagonal_axis/orthagonal_dynamics/"
        "finite-amplitude_resilience/004_far_002_perturbation_family_semantics_prereg.md"
    )
    protocol.parent.mkdir(parents=True)
    protocol.write_text("FAR-002 fixture\n", encoding="utf-8")
    certificate = tmp_path / "far001b.json"
    certificate.write_text(
        json.dumps(
            {
                "protocol_id": "FAR-001B",
                "gate_status": "PASS",
                "eligible_candidates": [
                    {
                        "dataset_id": "dandi_000458",
                        "identity": "electrical_stimulation_current",
                        "classification": "CURVE_CANDIDATE",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return protocol, certificate


def test_missing_optional_pulse_fields_are_recorded_without_blocking(
    tmp_path: Path,
) -> None:
    root = tmp_path / "000458"
    for subject in ("sub-001", "sub-002"):
        _write_nwb_fixture(
            root / subject / f"{subject}_ses-1.nwb",
            currents=[10.0, 20.0, 30.0],
            states=["awake", "awake", "awake"],
        )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("FAR-002 fixture\n", encoding="utf-8")

    result = audit_dandi_000458(source_root=root, protocol_path=protocol)

    assert result["global_status"] == "PASS"
    assert result["unresolved_v_fields"] == [
        "n_pulses",
        "polarity",
        "pulse_frequency",
        "pulse_width",
        "train_duration",
        "waveform",
    ]
    assert any(
        row["curve_status"] == "CURVE_SEMANTICS_PASS"
        for row in result["perturbation_family_ledger"]
    )
    assert result["perturbation_family_ledger"][0]["rho_level_ledger"][0][
        "n_subjects"
    ] == 2


def test_broad_trial_window_is_timing_unresolved(tmp_path: Path) -> None:
    root = tmp_path / "000458"
    _write_nwb_fixture(
        root / "sub-001" / "sub-001_ses-1.nwb",
        currents=[10.0, 20.0, 30.0],
        states=["awake", "awake", "awake"],
        stimulus_text="electrical stimulation during behavioral trial",
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("FAR-002 fixture\n", encoding="utf-8")

    result = audit_dandi_000458(source_root=root, protocol_path=protocol)

    assert result["global_status"] == "NOT_TESTABLE"
    assert result["perturbation_family_ledger"][0]["curve_status"] == (
        TIMING_UNRESOLVED
    )


def test_duration_classes_split_observed_timing_variants(tmp_path: Path) -> None:
    root = tmp_path / "000458"
    for subject in ("sub-001", "sub-002"):
        _write_nwb_fixture(
            root / subject / f"{subject}_ses-1.nwb",
            currents=[10.0, 20.0, 30.0],
            states=["awake", "awake", "awake"],
            durations=[0.0005, 0.0005, 0.0005],
        )
    _write_nwb_fixture(
        root / "sub-003" / "sub-003_ses-1.nwb",
        currents=[20.0],
        states=["awake"],
        durations=[0.01],
    )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("FAR-002 fixture\n", encoding="utf-8")

    result = audit_dandi_000458(source_root=root, protocol_path=protocol)

    classes = {
        row["perturbation_identity_v"]["observed_stim_duration_class"]
        for row in result["perturbation_family_ledger"]
    }
    assert classes == {"sub_2ms", "2_to_100ms"}
    assert any(
        row["curve_status"] == "CURVE_SEMANTICS_PASS"
        and row["perturbation_identity_v"]["observed_stim_duration_class"]
        == "sub_2ms"
        for row in result["perturbation_family_ledger"]
    )


def test_duration_above_100ms_is_timing_unresolved(tmp_path: Path) -> None:
    root = tmp_path / "000458"
    for subject in ("sub-001", "sub-002"):
        _write_nwb_fixture(
            root / subject / f"{subject}_ses-1.nwb",
            currents=[10.0, 20.0, 30.0],
            states=["awake", "awake", "awake"],
            durations=[0.101, 0.101, 0.101],
        )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("FAR-002 fixture\n", encoding="utf-8")

    result = audit_dandi_000458(source_root=root, protocol_path=protocol)

    assert result["global_status"] == "NOT_TESTABLE"
    assert all(
        row["curve_status"] == TIMING_UNRESOLVED
        for row in result["perturbation_family_ledger"]
    )


def test_present_optional_native_field_enters_v_key(tmp_path: Path) -> None:
    root = tmp_path / "000458"
    for subject, polarity in (("sub-001", "positive"), ("sub-002", "negative")):
        _write_nwb_fixture(
            root / subject / f"{subject}_ses-1.nwb",
            currents=[10.0, 20.0, 30.0],
            states=["awake", "awake", "awake"],
            optional_columns={"polarity": [polarity] * 3},
        )
    protocol = tmp_path / "protocol.md"
    protocol.write_text("FAR-002 fixture\n", encoding="utf-8")

    result = audit_dandi_000458(source_root=root, protocol_path=protocol)

    assert len(result["perturbation_family_ledger"]) == 2
    assert all(
        "polarity=" in row["v_key"]
        for row in result["perturbation_family_ledger"]
    )


def test_gate_requires_far001b_and_refuses_overwrite(tmp_path: Path) -> None:
    protocol, certificate = _write_entry_fixture(tmp_path)
    root = tmp_path / "000458"
    _write_nwb_fixture(
        root / "sub-001" / "sub-001_ses-1.nwb",
        currents=[10.0, 20.0, 30.0],
        states=["awake", "awake", "awake"],
    )
    output_json = tmp_path / "result.json"
    output_report = tmp_path / "result.md"

    result = run_gate(
        repo_root=tmp_path,
        output_json=output_json,
        output_report=output_report,
        far001b_certificate=certificate,
        source_root=root,
    )

    assert result["global_status"] == "NOT_TESTABLE"
    assert output_json.is_file()
    assert output_report.is_file()
    with pytest.raises(FileExistsError, match="refusing_to_overwrite"):
        run_gate(
            repo_root=tmp_path,
            output_json=output_json,
            output_report=tmp_path / "new-report.md",
            far001b_certificate=certificate,
            source_root=root,
        )
