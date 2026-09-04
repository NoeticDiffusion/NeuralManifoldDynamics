from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families import (  # noqa: E402
    far_ext_002d_ds003670_exported_signal as far_ext,
)
from mndm.dynamical_families.far_ext_002b_ds003670_signal_timebase import (  # noqa: E402
    NativeEEG,
)


def _native_fixture(
    tmp_path: Path,
    *,
    dtype: np.dtype = np.dtype("float64"),
    n_samples: int = 100,
) -> NativeEEG:
    time = np.arange(n_samples, dtype=np.float64) / 20.0
    data = np.vstack(
        [
            np.sin(2.0 * np.pi * (1.0 + index / 10.0) * time)
            + index * 0.01
            for index in range(32)
        ]
        + [np.zeros((3, n_samples), dtype=np.float64)]
    ).astype(dtype)
    return NativeEEG(
        set_path=tmp_path / "sub-001_ses-01_task-GXtESCTT_eeg.set",
        sfreq=20.0,
        n_samples=n_samples,
        n_channels=35,
        channel_names=list(far_ext._EEG_CHANNEL_NAMES) + ["ECG", "EOG", "RESP"],
        events=[],
        boundary_times=[],
        bad_intervals=[],
        bad_channel_indices=[],
        data=data,
        data_dtype=np.dtype(dtype),
        fdt_path=None,
        rail_min=None,
        rail_max=None,
        xmin=0.0,
    )


def _candidate_row(
    *,
    eligible_horizons: set[float],
    technical_horizons: set[float] | None = None,
) -> dict[str, object]:
    technical_horizons = (
        eligible_horizons if technical_horizons is None else technical_horizons
    )
    rows: dict[str, dict[str, object]] = {}
    for horizon in far_ext.HORIZONS_SEC:
        eligible = horizon in eligible_horizons
        technical = horizon in technical_horizons
        rows[str(horizon)] = {
            "eligible": eligible,
            "technical_eligible": technical,
            "horizon_sec": horizon,
            "pre_qc": {"status": far_ext.TECHNICALLY_ADMISSIBLE},
            "post_qc": {"status": far_ext.TECHNICALLY_ADMISSIBLE},
            "nmd": {
                "pre_count": 1,
                "post_count": 2,
            },
            "reason": "eligible" if eligible else "fixture",
        }
    return {
        "event_key": "fixture",
        "clock_status": "PASS",
        "pre_qc": {"status": far_ext.TECHNICALLY_ADMISSIBLE},
        "candidate_by_horizon": rows,
    }


def test_002a_canonicalization_is_002a_only_and_normalized() -> None:
    payload = {
        "protocol_id": "FAR-EXT-002A",
        "dataset_id": "ds003670",
        "global_status": "DS003670_LIMITED_CURVE_SEMANTICS_PASS",
        "events": [
            {
                "semantic_status": "PASS",
                "rho_contrast_eligible": True,
                "experiment": 1,
                "T_isolated": 9.0,
                "source_file": r"sub-001\ses-01\eeg\record_events.tsv",
                "subject": "001",
                "session": "01",
                "block": 1,
                "trial_in_block": 1,
                "trigger_start": 10.0,
                "v": {
                    "target": "frontal",
                    "waveform": "DC",
                    "frequency_hz": 0,
                },
                "rho": 0.5,
            },
            {
                "semantic_status": "PASS",
                "rho_contrast_eligible": False,
                "experiment": 1,
                "T_isolated": 9.0,
            },
        ],
    }
    events, digest = far_ext.canonical_002a_promoted_events(payload)
    assert len(events) == 1
    assert events[0]["source_file"] == "sub-001/ses-01/eeg/record_events.tsv"
    assert digest == far_ext._json_hash(events)
    assert "event_key" not in events[0]
    assert "family_id" not in events[0]


def test_grid_and_nmd_window_support_are_recording_anchored() -> None:
    assert far_ext.grid_starts(0.0, 30.0) == [
        0.0,
        4.0,
        8.0,
        12.0,
        16.0,
        20.0,
    ]
    support = far_ext.nmd_window_support(
        trigger_start=30.0,
        t0=40.0,
        horizon=16.0,
    )
    assert support["pre_starts"] == [0.0, 4.0, 8.0, 12.0, 16.0, 20.0]
    assert support["post_starts"] == [40.0, 44.0, 48.0]
    assert support["pre_required"] is True
    assert support["post_required"] is True


def test_interval_is_sliced_then_qcd_as_absolute_float_surface(
    tmp_path: Path,
) -> None:
    native = _native_fixture(
        tmp_path,
        dtype=np.dtype("float64"),
        n_samples=200,
    )
    native.data[:, :40] = np.nan
    native.data[:, 100:] = np.nan
    result = far_ext.audit_interval(
        native,
        channel_indices=list(range(32)),
        channel_names=list(far_ext._EEG_CHANNEL_NAMES),
        start_sec=2.0,
        end_sec=5.0,
        input_label="pre",
    )
    assert result["status"] == far_ext.TECHNICALLY_ADMISSIBLE
    assert result["sample_start"] == 40
    assert result["sample_end"] == 100
    assert result["qc"]["input_shape"] == [32, 60]
    assert result["qc"]["required_channel_indices"] == list(range(32))
    assert result["qc"]["required_channel_names"] == list(
        far_ext._EEG_CHANNEL_NAMES
    )
    assert result["qc"]["time_dtype"] == "float64"


def test_interval_qc_receives_absolute_time_for_the_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _native_fixture(
        tmp_path,
        dtype=np.dtype("float64"),
        n_samples=200,
    )
    native.data[:, :40] = np.nan
    native.data[:, 100:] = np.nan
    original_audit = far_ext.audit_exported_float
    captured: dict[str, object] = {}

    def capture_audit(signal: object, time: object, **kwargs: object) -> dict[str, object]:
        captured["signal"] = np.asarray(signal)
        captured["time"] = np.asarray(time)
        return original_audit(signal, time, **kwargs)

    monkeypatch.setattr(far_ext, "audit_exported_float", capture_audit)
    result = far_ext.audit_interval(
        native,
        channel_indices=list(range(32)),
        channel_names=list(far_ext._EEG_CHANNEL_NAMES),
        start_sec=2.0,
        end_sec=5.0,
        input_label="pre",
    )
    assert result["status"] == far_ext.TECHNICALLY_ADMISSIBLE
    assert captured["signal"].shape == (32, 60)
    np.testing.assert_allclose(
        captured["time"],
        np.arange(40, 100, dtype=np.float64) / 20.0,
    )


def test_interval_qc_does_not_hide_nonfinite_values_inside_slice(
    tmp_path: Path,
) -> None:
    native = _native_fixture(
        tmp_path,
        dtype=np.dtype("float64"),
        n_samples=200,
    )
    native.data[:, :40] = np.nan
    native.data[:, 100:] = np.nan
    native.data[3, 50] = np.nan
    result = far_ext.audit_interval(
        native,
        channel_indices=list(range(32)),
        channel_names=list(far_ext._EEG_CHANNEL_NAMES),
        start_sec=2.0,
        end_sec=5.0,
        input_label="pre",
    )
    assert result["status"] == far_ext.TECHNICAL_INVALID
    assert "NONFINITE_SAMPLES" in result["qc"]["invalid_reasons"]


def test_interval_passes_r8_indices_and_omits_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _native_fixture(tmp_path)
    original_audit = far_ext.audit_exported_float
    captured: dict[str, object] = {}

    def capture_audit(signal: object, time: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return original_audit(signal, time, **kwargs)

    monkeypatch.setattr(far_ext, "audit_exported_float", capture_audit)
    result = far_ext.audit_interval(
        native,
        channel_indices=list(range(32)),
        channel_names=list(far_ext._EEG_CHANNEL_NAMES),
        start_sec=0.0,
        end_sec=5.0,
        input_label="pre",
    )
    assert result["status"] == far_ext.TECHNICALLY_ADMISSIBLE
    assert captured["required_channel_indices"] == list(range(32))
    assert captured.get("required_channel_names") is None
    assert captured["channel_selection_provenance"]


def test_interval_preserves_technical_invalid_and_unsupported_dtype(
    tmp_path: Path,
) -> None:
    native = _native_fixture(tmp_path, dtype=np.dtype("float64"))
    native.data[3, 10] = np.nan
    invalid = far_ext.audit_interval(
        native,
        channel_indices=list(range(32)),
        channel_names=list(far_ext._EEG_CHANNEL_NAMES),
        start_sec=0.0,
        end_sec=5.0,
        input_label="post_16s",
    )
    assert invalid["status"] == far_ext.TECHNICAL_INVALID
    assert "NONFINITE_SAMPLES" in invalid["qc"]["invalid_reasons"]

    integer_native = _native_fixture(tmp_path, dtype=np.dtype("int16"))
    unresolved = far_ext.audit_interval(
        integer_native,
        channel_indices=list(range(32)),
        channel_names=list(far_ext._EEG_CHANNEL_NAMES),
        start_sec=0.0,
        end_sec=5.0,
        input_label="post_16s",
    )
    assert unresolved["status"] == far_ext.TECHNICAL_STATUS_UNRESOLVED
    assert unresolved["qc"]["unresolved_reasons"] == [
        "SIGNAL_DTYPE_UNSUPPORTED"
    ]


def test_family_horizon_is_common_and_support_is_same_horizon() -> None:
    events = []
    candidates = []
    for index, (rho, biological_unit) in enumerate(
        (
            (0.5, "bio-001"),
            (0.5, "bio-002"),
            (1.0, "bio-003"),
            (1.0, "bio-004"),
        )
    ):
        events.append(
            {
                "event_key": f"event-{index}",
                "biological_unit": biological_unit,
                "rho": rho,
                "v": {
                    "target": "frontal",
                    "waveform": "sinusoidal",
                    "frequency_hz": 5,
                },
                "target": "frontal",
                "waveform": "sinusoidal",
                "frequency_hz": 5,
            }
        )
        candidates.append(
            _candidate_row(
                eligible_horizons={30.0}
                if rho == 1.0
                else {60.0, 30.0},
            )
        )
    family = far_ext._family_result(
        "frontal_sinusoidal_5hz",
        events,
        candidates,
        nmd_config_pass=True,
        nmd_binding={"status": "PASS"},
    )
    assert family["selected_post_horizon"] == 30.0
    assert family["subjects_per_rho"] == {
        "0.5": ["bio-001", "bio-002"],
        "1.0": ["bio-003", "bio-004"],
    }
    assert family["events_per_rho"] == {"0.5": 2, "1.0": 2}
    assert family["eligibility_status"] == far_ext.SIGNAL_TIMEBASE_PASS


def test_family_does_not_select_private_event_fallback() -> None:
    events = [
        {
            "event_key": "event-0",
            "biological_unit": "bio-001",
            "rho": 0.5,
            "v": {"target": "frontal", "waveform": "DC", "frequency_hz": 0},
        },
        {
            "event_key": "event-1",
            "biological_unit": "bio-002",
            "rho": 0.5,
            "v": {"target": "frontal", "waveform": "DC", "frequency_hz": 0},
        },
        {
            "event_key": "event-2",
            "biological_unit": "bio-003",
            "rho": 1.0,
            "v": {"target": "frontal", "waveform": "DC", "frequency_hz": 0},
        },
        {
            "event_key": "event-3",
            "biological_unit": "bio-004",
            "rho": 1.0,
            "v": {"target": "frontal", "waveform": "DC", "frequency_hz": 0},
        },
    ]
    candidates = [
        _candidate_row(eligible_horizons={60.0}),
        _candidate_row(eligible_horizons={60.0}),
        _candidate_row(eligible_horizons={30.0}),
        _candidate_row(eligible_horizons={30.0}),
    ]
    family = far_ext._family_result(
        "frontal_dc_0hz",
        events,
        candidates,
        nmd_config_pass=True,
        nmd_binding={"status": "PASS"},
    )
    assert family["selected_post_horizon"] is None
    assert family["eligibility_status"] == (
        far_ext.INSUFFICIENT_BIOLOGICAL_SUPPORT
    )


def test_lattice_config_failure_is_method_limited() -> None:
    event = {
        "event_key": "event-0",
        "biological_unit": "bio-001",
        "rho": 0.5,
        "v": {"target": "frontal", "waveform": "DC", "frequency_hz": 0},
    }
    family = far_ext._family_result(
        "frontal_dc_0hz",
        [event],
        [_candidate_row(eligible_horizons=set(), technical_horizons={60.0})],
        nmd_config_pass=False,
        nmd_binding={"status": "NMD_TIMEBASE_METHOD_LIMITED"},
    )
    assert family["selected_post_horizon"] is None
    assert family["eligibility_status"] == far_ext.NMD_TIMEBASE_METHOD_LIMITED
    assert family["reason"] == "frozen_nmd_configuration_not_auditable"
    assert family["nmd_timebase_status"] == far_ext.NMD_TIMEBASE_METHOD_LIMITED


def test_next_stimulation_rejects_horizon_but_still_audits_post_slice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _native_fixture(tmp_path, n_samples=2000)
    calls: list[tuple[float, float, str]] = []

    def fake_audit(
        _native: NativeEEG,
        *,
        start_sec: float,
        end_sec: float,
        input_label: str,
        **kwargs: object,
    ) -> dict[str, object]:
        del _native, kwargs
        calls.append((start_sec, end_sec, input_label))
        return {
            "status": far_ext.TECHNICALLY_ADMISSIBLE,
            "reason": "fixture_pass",
            "qc": {"status": far_ext.TECHNICALLY_ADMISSIBLE},
        }

    monkeypatch.setattr(far_ext, "audit_interval", fake_audit)
    event = {
        "event_key": "event-0",
        "trigger_start": 30.0,
        "ramp_down_end": 40.0,
        "next_stim_start": 70.0,
    }
    result = far_ext._event_candidate_rows(
        event,
        native,
        channel_indices=list(range(32)),
        channel_names=list(far_ext._EEG_CHANNEL_NAMES),
        clock_status="PASS",
        nmd_config_pass=True,
    )
    assert len(calls) == 5
    horizon_60 = result["candidate_by_horizon"]["60.0"]
    assert horizon_60["post_qc"]["status"] == far_ext.TECHNICALLY_ADMISSIBLE
    assert horizon_60["eligible"] is False
    assert horizon_60["technical_eligible"] is False
    assert horizon_60["reason"] == "next_stimulation_within_horizon"
    horizon_30 = result["candidate_by_horizon"]["30.0"]
    assert horizon_30["eligible"] is True


def test_payload_inventory_detects_hash_or_extra_file(
    tmp_path: Path,
) -> None:
    eeg_root = tmp_path / "eeg"
    eeg_root.mkdir()
    set_path = eeg_root / "sub-001" / "ses-01" / "eeg" / "record_eeg.set"
    fdt_path = set_path.with_suffix(".fdt")
    set_path.parent.mkdir(parents=True)
    set_path.write_bytes(b"set")
    fdt_path.write_bytes(b"fdt")
    inventory_path = tmp_path / "inventory.json"
    inventory = {
        "files": [
            {
                "path": "sub-001/ses-01/eeg/record_eeg.fdt",
                "sha256": far_ext.sha256_file(fdt_path),
            },
            {
                "path": "sub-001/ses-01/eeg/record_eeg.set",
                "sha256": far_ext.sha256_file(set_path),
            },
        ]
    }
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    scope = {
        "payload": {
            "inventory_path": str(inventory_path),
            "inventory_sha256": far_ext.sha256_file(inventory_path),
            "existing_payload_root": str(eeg_root),
        }
    }
    ledger = [{"source_file": "sub-001/ses-01/eeg/record_events.tsv"}]
    result = far_ext.verify_payload_inventory(
        eeg_root=eeg_root,
        scope=scope,
        repo_root=tmp_path,
        ledger=ledger,
    )
    assert result["status"] == "PASS"
    set_path.write_bytes(b"changed")
    result = far_ext.verify_payload_inventory(
        eeg_root=eeg_root,
        scope=scope,
        repo_root=tmp_path,
        ledger=ledger,
    )
    assert result["status"] == far_ext.SOURCE_BINDING_FAILED
    assert result["reason"] == "payload_inventory_mismatch"
    set_path.write_bytes(b"set")
    fdt_path.unlink()
    result = far_ext.verify_payload_inventory(
        eeg_root=eeg_root,
        scope=scope,
        repo_root=tmp_path,
        ledger=ledger,
    )
    assert result["status"] == far_ext.SOURCE_BINDING_FAILED
    assert result["reason"] == "payload_inventory_mismatch"
    assert result["missing"] == [
        "sub-001/ses-01/eeg/record_eeg.fdt",
    ]
    fdt_path.write_bytes(b"fdt")
    (eeg_root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    result = far_ext.verify_payload_inventory(
        eeg_root=eeg_root,
        scope=scope,
        repo_root=tmp_path,
        ledger=ledger,
    )
    assert result["status"] == far_ext.SOURCE_BINDING_FAILED
    assert result["reason"] == "payload_inventory_mismatch"


def test_frozen_002a_certificate_ledger_hash_and_mismatch_fail_closed() -> None:
    certificate = (
        Path(__file__).resolve().parents[2]
        / "project"
        / "orthagonal_axis"
        / "orthagonal_dynamics"
        / "finite-amplitude_resilience"
        / "results"
        / "far_ext_002a"
        / "far_ext_002a.json"
    )
    payload = json.loads(certificate.read_text(encoding="utf-8"))
    events, digest = far_ext.canonical_002a_promoted_events(payload)
    assert len(events) == 323
    assert digest == (
        "bb14b77523ea9e3a7353a31bf4ad1a7dbad4e79e8e2c036d00f056dc8f3694cd"
    )
    with pytest.raises(ValueError, match="promoted_ledger_hash_mismatch"):
        far_ext.promoted_event_ledger(payload, expected_hash="0" * 64)


def test_payload_binding_fails_before_native_reader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    far_root = (
        Path(__file__).resolve().parents[2]
        / "project"
        / "orthagonal_axis"
        / "orthagonal_dynamics"
        / "finite-amplitude_resilience"
    )
    scope_path = far_root / "FAR-EXT-002D-SCOPE-0.1.json"
    certificate = far_root / "results" / "far_ext_002a" / "far_ext_002a.json"
    metadata_root = far_root / "data" / "far_ext_001" / "openneuro_metadata_only" / "ds003670"
    eeg_root = tmp_path / "wrong-payload-root"
    eeg_root.mkdir()
    called = False

    def fail_if_called(*args: object, **kwargs: object) -> None:
        del args, kwargs
        nonlocal called
        called = True
        raise AssertionError("native reader called before payload binding")

    monkeypatch.setattr(far_ext, "read_native_eeglab", fail_if_called)
    result = far_ext.audit_far_ext_002d(
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        far002_path=certificate,
        scope_path=scope_path,
        repo_root=far_root.parents[3],
    )
    assert result["global_status"] == far_ext.SOURCE_BINDING_FAILED
    assert result["global_reason"] == "payload_binding_failed"
    assert called is False


def test_clock_matches_codes_to_distinct_002a_fields_and_ignores_leftovers(
    tmp_path: Path,
) -> None:
    native = _native_fixture(tmp_path)
    native.events = [
        {
            "code": "16",
            "time_sec": 1.0,
            "sample_index_0": 20,
        },
        {
            "code": "32",
            "time_sec": 2.0,
            "sample_index_0": 40,
        },
        {
            "code": "16",
            "time_sec": 10.0,
            "sample_index_0": 200,
        },
        {
            "code": "32",
            "time_sec": 11.0,
            "sample_index_0": 220,
        },
        {
            "code": "16",
            "time_sec": 50.0,
            "sample_index_0": 1000,
        },
        {
            "code": "32",
            "time_sec": 51.0,
            "sample_index_0": 1020,
        },
    ]
    events = [
        {
            "event_key": "event-1",
            "trigger_start": 10.0,
            "trigger_max_end": 11.0,
        }
    ]
    result = far_ext._clock_matches(native, events)
    assert result["status"] == "PASS"
    assert result["matched_trigger_count"] == 2
    assert {(match["code"], match["mapped_field"]) for match in result["matches"]} == {
        ("16", "trigger_start"),
        ("32", "trigger_max_end"),
    }
    assert result["leftover_native_16_32_ignored"] is True
    assert result["by_event"]["event-1"]["status"] == "PASS"

    events[0]["trigger_max_end"] = 12.0
    unresolved = far_ext._clock_matches(native, events)
    assert unresolved["status"] == far_ext.CLOCK_UNRESOLVED
    assert unresolved["by_event"]["event-1"]["status"] == (
        far_ext.CLOCK_UNRESOLVED
    )


def test_channel_order_mismatch_is_fail_closed(tmp_path: Path) -> None:
    native = _native_fixture(tmp_path)
    native.channel_names[2] = "FP2"
    sidecar = (
        tmp_path
        / "sub-001"
        / "ses-01"
        / "eeg"
        / "sub-001_ses-01_task-GXtESCTT_eeg.json"
    )
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(json.dumps({"EEGChannelCount": 32}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="channel_order_mismatch"):
        far_ext._select_channels(
            native,
            metadata_root=tmp_path,
            relative_set="sub-001/ses-01/eeg/sub-001_ses-01_task-GXtESCTT_eeg.set",
        )


def test_global_status_precedence_and_primary_family_ranking() -> None:
    family = {
        "eligibility_status": far_ext.NOT_TESTABLE,
        "clock_status": far_ext.CLOCK_UNRESOLVED,
    }
    assert far_ext._global_status([family])[0] == far_ext.CLOCK_UNRESOLVED
    assert far_ext._global_status(
        [
            {
                "eligibility_status": far_ext.INSUFFICIENT_BIOLOGICAL_SUPPORT,
                "clock_status": "PASS",
            }
        ]
    )[0] == far_ext.INSUFFICIENT_BIOLOGICAL_SUPPORT
    assert far_ext._global_status(
        [
            {
                "eligibility_status": far_ext.NMD_TIMEBASE_METHOD_LIMITED,
                "clock_status": "PASS",
            },
            {
                "eligibility_status": far_ext.CLOCK_UNRESOLVED,
                "clock_status": far_ext.CLOCK_UNRESOLVED,
            },
        ]
    )[0] == far_ext.NMD_TIMEBASE_METHOD_LIMITED
    assert far_ext._global_status(
        [
            {
                "eligibility_status": far_ext.SIGNAL_TIMEBASE_PASS,
                "clock_status": "PASS",
            },
            {
                "eligibility_status": far_ext.CLOCK_UNRESOLVED,
                "clock_status": far_ext.CLOCK_UNRESOLVED,
            },
        ]
    )[0] == far_ext.SIGNAL_TIMEBASE_PASS

    families = [
        {
            "family_id": "frontal_dc_0hz",
            "eligibility_status": far_ext.SIGNAL_TIMEBASE_PASS,
            "min_subjects_across_rho": 2,
            "events_per_rho": {"0.5": 8, "1.0": 8},
            "selected_post_horizon": 60.0,
        },
        {
            "family_id": "motor_dc_0hz",
            "eligibility_status": far_ext.SIGNAL_TIMEBASE_PASS,
            "min_subjects_across_rho": 3,
            "events_per_rho": {"0.5": 4, "1.0": 4},
            "selected_post_horizon": 30.0,
        },
    ]
    assert far_ext._select_primary_family(families) == "motor_dc_0hz"


def test_compact_end_to_end_float_fixture_uses_frozen_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sio = pytest.importorskip("scipy.io")
    metadata_root = tmp_path / "metadata"
    eeg_root = tmp_path / "eeg"
    metadata_root.mkdir()
    eeg_root.mkdir()
    config_path = tmp_path / "config_ingest_common_eeg.yaml"
    config_path.write_text(
        "\n".join(
            [
                "mnps:",
                "  window_sec: 8.0",
                "  overlap: 0.5",
                "epoching:",
                "  length_s: 8.0",
                "  step_s: 4.0",
                "",
            ]
        ),
        encoding="utf-8",
    )

    n_samples = 2000
    sfreq = 20.0
    time = np.arange(n_samples, dtype=np.float64) / sfreq
    rng = np.random.default_rng(20260826)
    base_data = np.vstack(
        [
            (
                np.sin(2.0 * np.pi * (1.0 + index / 10.0) * time)
                + 0.01 * index
                + 0.005 * rng.normal(size=n_samples)
            )
            for index in range(32)
        ]
        + [np.zeros((3, n_samples), dtype=np.float64)]
    ).astype(np.float32)
    events: list[dict[str, object]] = []
    for index, (rho, biological_unit) in enumerate(
        (
            (0.5, "bio-001"),
            (0.5, "bio-002"),
            (1.0, "bio-003"),
            (1.0, "bio-004"),
        ),
        start=1,
    ):
        relative_events = (
            f"sub-{index:03d}/ses-01/eeg/"
            f"sub-{index:03d}_ses-01_task-GXtESCTT_events.tsv"
        )
        relative_set = relative_events.replace("_events.tsv", "_eeg.set")
        relative_sidecar = relative_events.replace("_events.tsv", "_eeg.json")
        sidecar = metadata_root / Path(relative_sidecar)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        sidecar.write_text(
            json.dumps({"EEGChannelCount": 32}),
            encoding="utf-8",
        )
        set_path = eeg_root / Path(relative_set)
        set_path.parent.mkdir(parents=True, exist_ok=True)
        labels = np.empty(35, dtype=object)
        for channel_index, name in enumerate(
            list(far_ext._EEG_CHANNEL_NAMES) + ["ECG", "EOG", "RESP"]
        ):
            labels[channel_index] = {"labels": name}
        native_events = np.empty(2, dtype=object)
        native_events[0] = {"type": "16", "latency": 601.0}
        native_events[1] = {"type": "32", "latency": 701.0}
        sio.savemat(
            str(set_path),
            {
                "EEG": {
                    "srate": sfreq,
                    "pnts": float(n_samples),
                    "nbchan": 35.0,
                    "xmin": 0.0,
                    "data": base_data,
                    "chanlocs": labels,
                    "event": native_events,
                }
            },
            do_compression=False,
        )
        events.append(
            {
                "subject": f"{index:03d}",
                "biological_unit": biological_unit,
                "experiment": 1,
                "session": "01",
                "block": 1,
                "trial_in_block": 1,
                "v": {
                    "target": "frontal",
                    "waveform": "sinusoidal",
                    "frequency_hz": 5,
                },
                "target": "frontal",
                "waveform": "sinusoidal",
                "frequency_hz": 5,
                "rho": rho,
                "rho_source": "publication_table_1",
                "rho_contrast_eligible": True,
                "trigger_start": 30.0,
                "trigger_max_end": 35.0,
                "ramp_down_end": 40.0,
                "next_stim_start": None,
                "T_isolated": 60.0,
                "semantic_status": "PASS",
                "source_file": relative_events,
            }
        )

    certificate = {
        "protocol_id": "FAR-EXT-002A",
        "dataset_id": "ds003670",
        "global_status": "DS003670_LIMITED_CURVE_SEMANTICS_PASS",
        "source_root": str(metadata_root),
        "source_binding": {
            "status": "PASS",
            "signal_payloads": [],
            "unexpected_files": [],
        },
        "events": events,
    }
    certificate_path = tmp_path / "far_ext_002a.json"
    certificate_path.write_text(
        json.dumps(certificate),
        encoding="utf-8",
    )
    canonical_events, ledger_hash = far_ext.canonical_002a_promoted_events(
        certificate
    )
    inventory_rows = []
    for event in canonical_events:
        relative_set = event["source_file"].replace(
            "_events.tsv",
            "_eeg.set",
        )
        set_path = eeg_root / Path(relative_set)
        inventory_rows.append(
            {
                "path": relative_set,
                "sha256": far_ext.sha256_file(set_path),
            }
        )
        fdt_path = set_path.with_suffix(".fdt")
        fdt_path.write_bytes(b"")
        inventory_rows.append(
            {
                "path": str(fdt_path.relative_to(eeg_root)).replace("\\", "/"),
                "sha256": far_ext.sha256_file(fdt_path),
            }
        )
    inventory = {"files": sorted(inventory_rows, key=lambda row: row["path"])}
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(
        json.dumps(inventory, separators=(",", ":")),
        encoding="utf-8",
    )
    config_hash = far_ext.sha256_file(config_path)
    scope = {
        "protocol_id": "FAR-EXT-002D",
        "execution_status": "PREREGISTRATION_FROZEN",
        "inherited_002a": {
            "certificate_sha256": far_ext.sha256_file(certificate_path),
            "certificate_global_status": certificate["global_status"],
            "promoted_event_ledger_sha256": ledger_hash,
        },
        "payload": {
            "inventory_path": str(inventory_path),
            "inventory_sha256": far_ext.sha256_file(inventory_path),
            "inventory_file_count": 8,
            "expected_payload_file_count": 8,
            "existing_payload_root": str(eeg_root),
        },
        "inherited_config": {
            "path": str(config_path),
            "sha256": config_hash,
            "required_values": {
                "mnps.window_sec": 8.0,
                "mnps.overlap": 0.5,
                "epoching.length_s": 8.0,
                "epoching.step_s": 4.0,
            },
        },
        "nmd_qc_float": {
            "contract_version": "0.3",
            "rule_manifest_sha256": far_ext.RULE_MANIFEST_SHA256,
        },
    }
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    monkeypatch.setattr(far_ext, "PROMOTED_EVENT_COUNT", 4)
    result = far_ext.audit_far_ext_002d(
        metadata_root=metadata_root,
        eeg_root=eeg_root,
        far002_path=certificate_path,
        scope_path=scope_path,
        repo_root=tmp_path,
    )
    assert result["global_status"] == far_ext.SIGNAL_TIMEBASE_PASS
    assert result["promoted_event_count"] == 4
    assert result["family_ledger"][0]["selected_post_horizon"] == 60.0
    assert result["family_ledger"][0]["pre_qc_status"]["status"] == (
        far_ext.TECHNICALLY_ADMISSIBLE
    )
    assert result["family_ledger"][0]["post_qc_status_by_horizon"]["60.0"][
        "status"
    ] == far_ext.TECHNICALLY_ADMISSIBLE
    assert result["nmd_timebase_status"] == "PASS"
    assert result["post_stimulation_biological_interpretability"] == (
        "NOT_ESTABLISHED"
    )
