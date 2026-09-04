"""Synthetic tests for FAR-003A signal/time-base eligibility."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.far_perturbation_family_semantics import (  # noqa: E402
    _family_id,
    _v_key,
)
from mndm.dynamical_families.far_signal_timebase_eligibility import (  # noqa: E402
    STIM_ARTIFACT_FREE,
    STIM_ARTIFACT_UNRESOLVED,
    audit_far_003a,
)
from od_far_003a_signal_timebase_gate import run_gate  # noqa: E402


h5py = pytest.importorskip("h5py")


def _v_values() -> dict[str, str]:
    return {
        "stimulus_type": "electrical",
        "stimulus_description": "biphasic",
        "estim_target_region": "MOs",
        "estim_target_depth": "superficial",
        "observed_stim_duration_class": "sub_2ms",
    }


def _write_nwb_fixture(
    path: Path,
    *,
    gap: bool = False,
    native_recovery: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = 10_001
    timestamps = np.arange(n_samples, dtype=float) / 1000.0
    if gap:
        timestamps[2500:] += 0.25
    with h5py.File(path, "w") as handle:
        trials = handle.create_group("intervals").create_group("trials")
        string_dtype = h5py.string_dtype(encoding="utf-8")
        rows = 3
        trials.create_dataset(
            "behavioral_epoch",
            data=["awake"] * rows,
            dtype=string_dtype,
        )
        current = trials.create_dataset("estim_current", data=[10.0, 20.0, 30.0])
        current.attrs["description"] = (
            "electrical stimulation current (μA), if applicable"
        )
        for name, values in {
            "estim_target_depth": ["superficial"] * rows,
            "estim_target_region": ["MOs"] * rows,
            "stimulus_description": ["biphasic"] * rows,
            "stimulus_type": ["electrical"] * rows,
        }.items():
            trials.create_dataset(name, data=values, dtype=string_dtype)
        trials.create_dataset("is_valid", data=[True] * rows)
        trials.create_dataset("start_time", data=[1.0, 2.0, 3.0])
        trials.create_dataset("stop_time", data=[1.0005, 2.0005, 3.0005])
        acquisition = handle.create_group("acquisition")
        series = acquisition.create_group("ElectricalSeriesEEG")
        series.create_dataset(
            "data",
            data=np.zeros((n_samples, 2), dtype=np.int16),
            chunks=(1000, 1),
        )
        series.create_dataset("timestamps", data=timestamps)
        if native_recovery:
            series.attrs["artifact_recovery_sec"] = 0.0


def _write_far002_fixture(
    tmp_path: Path,
    *,
    protocol: Path,
    source_root: Path,
) -> Path:
    v = _v_values()
    family_id = _family_id(_v_key(v))
    rows = [
        {
            "family_id": family_id if index == 0 else f"fake-family-{index}",
            "operating_state_w": "awake",
            "perturbation_identity_v": v,
            "n_nonzero_levels": 3,
            "rho_levels": [10.0, 20.0, 30.0],
            "curve_status": "CURVE_SEMANTICS_PASS",
        }
        for index in range(10)
    ]
    certificate = tmp_path / "far002.json"
    certificate.write_text(
        json.dumps(
            {
                "protocol_id": "FAR-002",
                "global_status": "PASS",
                "source_root": str(source_root),
                "native_nwb_files": [
                    {
                        "relative_path": str(path.relative_to(source_root)).replace(
                            "\\", "/"
                        ),
                        "size_bytes": path.stat().st_size,
                    }
                    for path in sorted(source_root.rglob("*.nwb"))
                ],
                "perturbation_family_ledger": rows,
            }
        ),
        encoding="utf-8",
    )
    protocol.write_text("FAR-003A fixture\n", encoding="utf-8")
    return certificate


def test_signal_audit_keeps_nmd_status_separate_and_fail_closed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "000458"
    for subject in ("sub-001", "sub-002"):
        _write_nwb_fixture(
            root / subject / f"{subject}_ses-1.nwb",
        )
    protocol = (
        tmp_path
        / "project/orthagonal_axis/orthagonal_dynamics/"
        "finite-amplitude_resilience/005_far_003a_signal_timebase_prereg.md"
    )
    protocol.parent.mkdir(parents=True)
    far002 = _write_far002_fixture(
        tmp_path,
        protocol=protocol,
        source_root=root,
    )

    result = audit_far_003a(
        source_root=root,
        far002_path=far002,
        protocol_path=protocol,
        repo_root=tmp_path,
    )

    assert result["frozen_curve_family_count"] == 10
    assert result["nmd_timebase"]["status"] == "NMD_TIMEBASE_NOT_TESTABLE"
    assert result["global_status"] == "METHOD_LIMITED"
    family = next(row for row in result["family_ledger"] if row["n_subjects"])
    assert family["artifact_status"][STIM_ARTIFACT_FREE] == 6
    assert family["artifact_recovery_status"][STIM_ARTIFACT_UNRESOLVED] == 6
    assert family["artifact_probe_status"][STIM_ARTIFACT_FREE] == 6
    assert all(
        record["artifact_status"] == STIM_ARTIFACT_FREE
        for record in result["event_records"]
    )
    assert result["far_003b_authorized"] is False
    assert result["fail_closed_assertions"]["response_statistics_computed"] is False


def test_timestamp_gap_is_recorded_without_response_statistics(tmp_path: Path) -> None:
    root = tmp_path / "000458"
    for subject in ("sub-001", "sub-002"):
        _write_nwb_fixture(
            root / subject / f"{subject}_ses-1.nwb",
            gap=True,
        )
    protocol = (
        tmp_path
        / "project/orthagonal_axis/orthagonal_dynamics/"
        "finite-amplitude_resilience/005_far_003a_signal_timebase_prereg.md"
    )
    protocol.parent.mkdir(parents=True)
    far002 = _write_far002_fixture(
        tmp_path,
        protocol=protocol,
        source_root=root,
    )

    result = audit_far_003a(
        source_root=root,
        far002_path=far002,
        protocol_path=protocol,
        repo_root=tmp_path,
    )

    assert any(record["gap_crossing"] for record in result["event_records"])
    assert result["fail_closed_assertions"]["home_away_constructed"] is False


def test_native_recovery_metadata_unlocks_outcome_blind_horizon_selection(
    tmp_path: Path,
) -> None:
    root = tmp_path / "000458"
    for subject in ("sub-001", "sub-002"):
        _write_nwb_fixture(
            root / subject / f"{subject}_ses-1.nwb",
            native_recovery=True,
        )
    protocol = (
        tmp_path
        / "project/orthagonal_axis/orthagonal_dynamics/"
        "finite-amplitude_resilience/005_far_003a_signal_timebase_prereg.md"
    )
    protocol.parent.mkdir(parents=True)
    far002 = _write_far002_fixture(
        tmp_path,
        protocol=protocol,
        source_root=root,
    )

    result = audit_far_003a(
        source_root=root,
        far002_path=far002,
        protocol_path=protocol,
        repo_root=tmp_path,
    )

    family = next(row for row in result["family_ledger"] if row["n_subjects"])
    assert family["status"] == "RAW_SIGNAL_ONLY_PASS"
    assert family["candidate_horizon"] == 0.5
    assert family["n_surviving_rho_levels"] == 3
    assert result["global_status"] == "METHOD_LIMITED"
    assert result["far_003b_authorized"] is False


def test_gate_requires_far002_and_refuses_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "000458"
    _write_nwb_fixture(root / "sub-001" / "sub-001_ses-1.nwb")
    protocol = (
        tmp_path
        / "project/orthagonal_axis/orthagonal_dynamics/"
        "finite-amplitude_resilience/005_far_003a_signal_timebase_prereg.md"
    )
    protocol.parent.mkdir(parents=True)
    far002 = _write_far002_fixture(
        tmp_path,
        protocol=protocol,
        source_root=root,
    )
    output_json = tmp_path / "result.json"
    output_report = tmp_path / "result.md"

    result = run_gate(
        repo_root=tmp_path,
        output_json=output_json,
        output_report=output_report,
        far002_certificate=far002,
        source_root=root,
    )

    assert result["global_status"] == "NOT_TESTABLE"
    assert output_json.is_file()
    with pytest.raises(FileExistsError, match="refusing_to_overwrite"):
        run_gate(
            repo_root=tmp_path,
            output_json=output_json,
            output_report=tmp_path / "new-report.md",
            far002_certificate=far002,
            source_root=root,
        )
