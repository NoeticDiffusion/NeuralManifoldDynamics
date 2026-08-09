"""Synthetic unit tests for the versioned Sleep-EAP Phase 2 data products."""

from __future__ import annotations

import json
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core" / "src"))


def test_cardiac_phase_series_has_known_half_cycle_phase():
    from mndm.features.sleep_eap import cardiac_phase_series

    phase = cardiac_phase_series(np.array([0.0, 0.4, 0.8, 1.2]), np.array([0.0, 0.8, 1.6]))
    assert phase[0] == pytest.approx(0.0)
    assert phase[1] == pytest.approx(np.pi)
    assert phase[2] == pytest.approx(0.0)
    assert phase[3] == pytest.approx(np.pi)


def test_continuous_phase_uses_public_phase_anchor_helpers(monkeypatch):
    import mndm.features.sleep_eap.phase_continuous as phase_continuous

    monkeypatch.setattr(
        phase_continuous,
        "detect_rpeaks_chunked",
        lambda ecg, sfreq, config: np.array([0, int(0.8 * sfreq), int(1.6 * sfreq)]),
    )
    sfreq = 100.0
    time = np.arange(0, 2.0, 1.0 / sfreq)
    state = phase_continuous.build_phase_continuous_state(
        ecg_1d=np.zeros_like(time),
        resp_1d=np.sin(2 * np.pi * 0.25 * time),
        sfreq=sfreq,
        sample_hz=10.0,
    )
    assert state.time_sec.size == 20
    assert state.cardiac_valid.sum() > 0
    assert state.resp_valid.sum() == state.time_sec.size


def test_sigma_strength_detects_burst_and_uses_reference_distribution():
    from mndm.features.sleep_eap import compute_spindle_strength, sigma_power_in_window

    sfreq = 100.0
    time = np.arange(0, 12.0, 1.0 / sfreq)
    rng = np.random.default_rng(7)
    eeg = 2.0 * rng.normal(size=time.size)
    burst = (time >= 5.0) & (time < 7.0)
    eeg[burst] += 20.0 * np.sin(2 * np.pi * 13.0 * time[burst])
    burst_power, _, status = sigma_power_in_window(eeg, sfreq=sfreq, center_sec=6.0)
    baseline_power, _, _ = sigma_power_in_window(eeg, sfreq=sfreq, center_sec=2.0)
    assert status == "ok"
    assert burst_power > baseline_power
    spindles = pd.DataFrame({"onset_sec": [5.5, 6.0], "peak_sec": [5.8, 6.2]})
    result = compute_spindle_strength(
        spindles,
        eeg=eeg,
        sfreq=sfreq,
        reference_centers_sec=[1.0, 2.0, 3.0, 9.0, 10.0],
    )
    assert result["sigma_power"].notna().all()
    assert result["sigma_power_z_n2"].notna().all()


def test_sleep_phase_sidecars_survive_degenerate_sigma_and_eog_anchor_inputs():
    """Sleep-specific products remain valid while inherited anchors reject bad z-scores."""
    from mndm.features.sleep_eap import (
        PhaseContinuousState,
        build_non_event_risk_catalog,
        robust_z_against_reference,
    )
    from mndm.pipeline.anchor_state import build_anchor_state_exports
    from mndm.projection import build_feature_export_bundle

    sigma_z, _, sigma_scale = robust_z_against_reference(
        [2.0, 3.0],
        [1.0, 1.0, 1.0],
    )
    assert np.isnan(sigma_z).all()
    assert np.isnan(sigma_scale)

    phase_state = PhaseContinuousState(
        time_sec=np.arange(0.0, 120.0, 1.0),
        phi_cardiac=np.mod(np.arange(120.0) * 0.2, 2 * np.pi),
        phi_resp=np.mod(np.arange(120.0) * 0.1, 2 * np.pi),
        cardiac_valid=np.ones(120, dtype=bool),
        resp_valid=np.ones(120, dtype=bool),
        rpeaks_sec=np.arange(0.0, 121.0, 1.0),
        sample_hz=1.0,
    )
    risk = build_non_event_risk_catalog(
        phase_state,
        subject="sleep01",
        stage_codes=[2] * 4,
        spindle_onsets_sec=[60.0],
        exclusion_margin_sec=1.0,
        n_per_event=1,
        seed=7,
    )
    assert not risk.empty
    assert risk["cardiac_phase_valid"].all()
    assert risk["resp_phase_valid"].all()

    raw_features = pd.DataFrame(
        {
            "eog_blink_rate": [0.0, 0.0, 0.0, 0.875],
            "resp_regular_index": [0.2, 0.3, 0.4, 0.5],
        }
    )
    bundle = build_feature_export_bundle(raw_features)
    anchor_state, _, anchor_quality, _ = build_anchor_state_exports(
        features_df=raw_features,
        robust_z_values=bundle["robust_z_values"],
        robust_z_names=bundle["robust_z_names"],
        feature_metadata=bundle["metadata"],
        time=np.arange(4, dtype=float),
        config={"anchor_state": {"enabled": True, "v2": {"enabled": True}}},
    )
    anchor_idx = anchor_state["names"].index("anchor_index")
    valid_idx = anchor_quality["names"].index("anchor_index_valid")
    assert np.isnan(anchor_state["values"][:, anchor_idx]).all()
    assert np.all(anchor_quality["values"][:, valid_idx] == 0.0)


def test_slow_oscillation_and_coupling_preserve_no_partner_as_nan():
    from mndm.features.sleep_eap import (
        couple_spindles_to_slow_oscillations,
        detect_slow_oscillations,
        slow_oscillation_phase,
    )

    sfreq = 100.0
    time = np.arange(0, 60.0, 1.0 / sfreq)
    eeg = 45.0 * np.sin(2 * np.pi * 0.8 * time)
    stages = [2, 2]
    slow = detect_slow_oscillations(eeg, sfreq=sfreq, stage_codes=stages)
    assert not slow.empty
    known_upstate = float(slow.iloc[0]["upstate_sec"])
    spindles = pd.DataFrame(
        {
            "onset_sec": [known_upstate + 0.3, 55.0],
            "peak_sec": [known_upstate + 0.3, 55.0],
        }
    )
    result = couple_spindles_to_slow_oscillations(
        spindles,
        slow.iloc[:1],
        max_abs_latency_sec=1.0,
        so_phase=slow_oscillation_phase(eeg, sfreq=sfreq),
        sfreq=sfreq,
    )
    assert result.loc[0, "so_partner_missing"] == 0
    assert result.loc[0, "so_spindle_latency_sec"] == pytest.approx(0.3, abs=0.02)
    assert result.loc[1, "so_partner_missing"] == 1
    assert np.isnan(result.loc[1, "so_spindle_latency_sec"])


def test_non_event_risk_catalog_is_seeded_and_excludes_spindles():
    from mndm.features.sleep_eap import PhaseContinuousState, build_non_event_risk_catalog

    time = np.arange(0, 240, dtype=float)
    state = PhaseContinuousState(
        time_sec=time,
        phi_cardiac=np.mod(time * 0.2, 2 * np.pi),
        phi_resp=np.mod(time * 0.1, 2 * np.pi),
        cardiac_valid=np.ones(time.size, dtype=bool),
        resp_valid=np.ones(time.size, dtype=bool),
        rpeaks_sec=np.arange(0, 241, 1.0),
        sample_hz=1.0,
    )
    first = build_non_event_risk_catalog(
        state,
        subject="xn001",
        stage_codes=[2] * 8,
        spindle_onsets_sec=[60.0, 170.0],
        exclusion_margin_sec=10.0,
        n_per_event=2,
        seed=99,
    )
    second = build_non_event_risk_catalog(
        state,
        subject="xn001",
        stage_codes=[2] * 8,
        spindle_onsets_sec=[60.0, 170.0],
        exclusion_margin_sec=10.0,
        n_per_event=2,
        seed=99,
    )
    pd.testing.assert_frame_equal(first, second)
    assert not first.empty
    assert np.all(np.abs(first["timestamp_sec"].to_numpy()[:, None] - np.array([60.0, 170.0])) > 10.0)


def test_rem_theta_is_only_valid_during_rem():
    from mndm.features.sleep_eap import rem_theta_phase

    sfreq = 100.0
    time = np.arange(0, 60, 1 / sfreq)
    phase, valid = rem_theta_phase(
        10 * np.sin(2 * np.pi * 6 * time),
        sfreq=sfreq,
        stage_codes=[2, 4],
    )
    assert not valid[: int(30 * sfreq)].any()
    assert valid[int(30 * sfreq) :].all()
    assert np.isnan(phase[: int(30 * sfreq)]).all()
    assert np.isfinite(phase[int(30 * sfreq) :]).all()


def test_n3_so_event_phase_is_n3_gated_and_keeps_spindle_schema_separate():
    from mndm.features.sleep_eap import (
        PhaseContinuousState,
        build_n3_slow_oscillation_event_phase,
    )

    state = PhaseContinuousState(
        time_sec=np.arange(0.0, 121.0, 1.0),
        phi_cardiac=np.arange(121.0) * 0.1,
        phi_resp=-np.arange(121.0) * 0.1,
        cardiac_valid=np.ones(121, dtype=bool),
        resp_valid=np.ones(121, dtype=bool),
        rpeaks_sec=np.arange(0.0, 130.0, 10.0),
        sample_hz=1.0,
    )
    slow = pd.DataFrame(
        {
            "onset_sec": [20.0, 30.0],
            "upstate_sec": [21.0, 31.0],
            "duration_sec": [1.0, 1.0],
            "amplitude": [80.0, 90.0],
            "stage": ["2", "3"],
            "source": ["synthetic", "synthetic"],
            "so_qc_flag": ["ok", "ok"],
        }
    )
    result = build_n3_slow_oscillation_event_phase(
        slow,
        phase_state=state,
        so_phase=np.arange(121.0) * 0.2,
        sfreq=1.0,
        subject="xn001",
    )

    assert len(result) == 1
    assert result.loc[0, "stage"] == 3
    assert result.loc[0, "phi_so_trough"] == pytest.approx(6.0)
    assert result.loc[0, "phi_so_upstate"] == pytest.approx(6.2)
    assert result.loc[0, "event_phase_contract"] == "event_phase_n3_so_v1"
    assert result.loc[0, "autonomic_phase_eligible_trough"] == 1
    assert result.loc[0, "inhale_at_trough"] == 1.0
    forbidden = {
        "sigma_power",
        "frequency_hz",
        "so_spindle_latency_sec",
        "so_partner_missing",
        "strength_qc_flag",
    }
    assert forbidden.isdisjoint(result.columns)


def test_n3_so_and_rem_event_phase_preserve_missing_autonomic_phase_as_nan():
    from mndm.features.sleep_eap import (
        PhaseContinuousState,
        build_n3_slow_oscillation_event_phase,
        build_rem_theta_event_phase,
    )

    state = PhaseContinuousState(
        time_sec=np.arange(0.0, 91.0, 1.0),
        phi_cardiac=np.full(91, np.nan),
        phi_resp=np.full(91, np.nan),
        cardiac_valid=np.zeros(91, dtype=bool),
        resp_valid=np.zeros(91, dtype=bool),
        rpeaks_sec=np.empty(0, dtype=float),
        sample_hz=1.0,
    )
    slow = pd.DataFrame({"onset_sec": [30.0], "upstate_sec": [31.0], "stage": [3]})
    n3 = build_n3_slow_oscillation_event_phase(
        slow,
        phase_state=state,
        so_phase=np.zeros(91),
        sfreq=1.0,
        subject="xn001",
    )
    rem = build_rem_theta_event_phase(
        [2, 4, 4],
        phase_state=state,
        theta_phase=np.arange(91.0),
        theta_valid=np.ones(91, dtype=bool),
        sfreq=1.0,
        subject="xn001",
    )

    assert n3["phi_cardiac_trough"].isna().all()
    assert n3["phi_resp_upstate"].isna().all()
    assert (n3["autonomic_phase_eligible_trough"] == 0).all()
    assert n3.loc[0, "phase_coupling_qc_flag_trough"] == "missing_cardiac_and_resp"
    assert rem["reference_sec"].tolist() == [45.0, 75.0]
    assert rem["phi_cardiac"].isna().all()
    assert rem["phi_resp"].isna().all()
    assert (rem["autonomic_phase_eligible"] == 0).all()
    assert set(rem["phase_coupling_qc_flag"]) == {"missing_cardiac_and_resp"}
    assert set(rem["event_phase_contract"]) == {"event_phase_rem_theta_v1"}
    assert {"sigma_power", "so_spindle_coupling_score"}.isdisjoint(rem.columns)


def test_event_table_and_event_locked_meta_support_phase2_fields(tmp_path: Path):
    from mndm.pipeline.event_annotations import load_event_table_from_csv
    from mndm.pipeline.event_locked_export import _event_meta

    csv_path = tmp_path / "spindles.csv"
    pd.DataFrame(
        {
            "onset_sec": [10.0],
            "duration_sec": [1.0],
            "sigma_power_z_n2": [2.5],
            "so_spindle_latency_sec": [0.2],
            "so_partner_missing": [0],
        }
    ).to_csv(csv_path, index=False)
    table = load_event_table_from_csv(csv_path)
    metadata = _event_meta(table, 0)
    assert metadata["event_sigma_power_z_n2"] == pytest.approx(2.5)
    assert metadata["event_so_spindle_latency_sec"] == pytest.approx(0.2)
    assert metadata["event_so_partner_missing"] == pytest.approx(0.0)


def test_phase2_extension_bridge_reads_versioned_sidecars(tmp_path: Path):
    from mndm.pipeline.phase_continuous_export import build_phase2_extensions_for_run

    continuous = tmp_path / "phase_continuous"
    risk = tmp_path / "risk"
    continuous.mkdir()
    risk.mkdir()
    pd.DataFrame({"timestamp_sec": [0.0], "phi_cardiac": [0.0], "phi_resp": [1.0]}).to_parquet(
        continuous / "xn001_phase_continuous_v1.parquet", index=False
    )
    pd.DataFrame({"timestamp_sec": [15.0], "stage": [2], "seed": [42]}).to_parquet(
        risk / "xn001_non_event_risk_v1.parquet", index=False
    )
    result = build_phase2_extensions_for_run(
        config={
            "phase_continuous": {"enabled": True, "sidecar_dir": str(continuous)},
            "non_event_risk": {"enabled": True, "sidecar_dir": str(risk)},
        },
        dataset_id="RichSleep",
        subject_id="sub-xn001",
    )
    assert result["manifest"]["phase_continuous"]["status"] == "available"
    assert result["manifest"]["non_event_risk"]["status"] == "available"
    assert "phase_continuous_v1" in result["extensions"]
    assert "non_event_risk_v1" in result["extensions"]


def test_phase2_extensions_are_written_under_h5_extensions(tmp_path: Path):
    import h5py

    from core.io.h5_writer import write_h5
    from mndm.schema import MNPSPayload

    payload = MNPSPayload(
        time=np.array([0.0, 1.0]),
        x=np.zeros((2, 3), dtype=np.float32),
        x_dot=np.zeros((2, 3), dtype=np.float32),
        extensions={
            "phase_continuous_v1": {"timestamp_sec": np.array([0.0]), "phi_resp": np.array([1.0])},
            "non_event_risk_v1": {"timestamp_sec": np.array([0.5]), "seed": np.array([42])},
        },
    )
    out_path = tmp_path / "phase2.h5"
    write_h5(out_path, "RichSleep", payload)
    with h5py.File(out_path, "r") as handle:
        assert "/extensions/phase_continuous_v1/timestamp_sec" in handle
        assert "/extensions/non_event_risk_v1/seed" in handle


def test_phase2_richsleep_overlay_preserves_primary_coordinate_contract():
    from core.config_loader import load_config

    config_path = Path(__file__).resolve().parents[1] / "config" / "config_ingest_richsleep_rvc_trv_phase2.yaml"
    config = load_config(config_path)
    assert config["paths"]["processed_dir"].endswith("mndm_rvc_processed_phase2")
    assert config["source"]["coordinate_contract"] == "rvc_v1_eeg_only_e_m"
    assert config["phase_continuous"]["enabled"] is True
    assert config["non_event_risk"]["enabled"] is True
    assert config["event_locked"]["datasets"]["RichSleep"]["profile"] == "rvc_spindle_v3"
    assert config["event_locked"]["datasets"]["RichSleep"]["csv_source_glob"].endswith("_spindles_yasa_v3_*.csv")


def test_v3_delivery_merges_strength_without_mutating_v2(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[2] / "project" / "scripts" / "21_deliver_event_phase_v3.py"
    spec = importlib.util.spec_from_file_location("deliver_event_phase_v3", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v2 = tmp_path / "xn001_event_phase_v2.parquet"
    strengths = tmp_path / "xn001_spindles_yasa_v3_eeg.csv"
    output = tmp_path / "xn001_event_phase_v3.parquet"
    original = pd.DataFrame({"subject": ["xn001"], "onset_sec": [10.0], "phi_resp_onset": [0.3]})
    original.to_parquet(v2, index=False)
    pd.DataFrame(
        {
            "onset_sec": [10.01],
            "sigma_power": [1.2],
            "sigma_power_z_n2": [2.1],
            "so_partner_missing": [0],
        }
    ).to_csv(strengths, index=False)
    module._merge_subject(v2, strengths, output, 0.05)
    pd.testing.assert_frame_equal(pd.read_parquet(v2), original)
    delivered = pd.read_parquet(output)
    assert delivered["sigma_power_z_n2"].iloc[0] == pytest.approx(2.1)
    assert delivered["event_phase_contract"].iloc[0] == "rvc_v1_phase_resolved_v03"


def test_phase2_extractor_writes_orthogonal_n3_and_rem_sidecars(tmp_path: Path, monkeypatch):
    from mndm.features.sleep_eap import PhaseContinuousState

    script_path = Path(__file__).resolve().parents[2] / "project" / "scripts" / "20_sleep_eap_phase2_extract.py"
    spec = importlib.util.spec_from_file_location("sleep_eap_phase2_extract", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    subject = "xn001"
    (tmp_path / f"{subject}.edf").touch()
    (tmp_path / f"{subject}_stages.txt").write_text("N2\nN3\nREM\n", encoding="utf-8")
    spindle_frame = pd.DataFrame({"onset_sec": [10.0], "peak_sec": [10.5], "duration_sec": [1.0]})
    phase_state = PhaseContinuousState(
        time_sec=np.arange(0.0, 121.0, 1.0),
        phi_cardiac=np.zeros(121),
        phi_resp=np.zeros(121),
        cardiac_valid=np.ones(121, dtype=bool),
        resp_valid=np.ones(121, dtype=bool),
        rpeaks_sec=np.arange(0.0, 130.0, 10.0),
        sample_hz=1.0,
    )

    class FakeRaw:
        info = {"sfreq": 1.0}

    def fake_slow(*_args, **kwargs):
        stage = 3 if kwargs.get("include_stages") == (3,) else 2
        return pd.DataFrame(
            {
                "onset_sec": [30.0],
                "upstate_sec": [31.0],
                "duration_sec": [1.0],
                "amplitude": [80.0],
                "stage": [stage],
            }
        )

    monkeypatch.setattr(module, "_load_spindles", lambda *_args: (spindle_frame.copy(), tmp_path / "source.csv"))
    monkeypatch.setattr(module, "_stage_codes", lambda _path: [2, 3, 4])
    monkeypatch.setattr(module.mne.io, "read_raw_edf", lambda *_args, **_kwargs: FakeRaw())
    monkeypatch.setattr(module, "_pick_data", lambda *_args, **_kwargs: np.zeros((1, 121)))
    monkeypatch.setattr(module, "compute_spindle_strength", lambda frame, **_kwargs: frame.copy())
    monkeypatch.setattr(module, "detect_slow_oscillations", fake_slow)
    monkeypatch.setattr(module, "slow_oscillation_phase", lambda *_args, **_kwargs: np.zeros(121))
    monkeypatch.setattr(module, "couple_spindles_to_slow_oscillations", lambda frame, *_args, **_kwargs: frame.copy())
    monkeypatch.setattr(module, "build_phase_continuous_state", lambda **_kwargs: phase_state)
    monkeypatch.setattr(module, "rem_theta_phase", lambda *_args, **_kwargs: (np.zeros(121), np.ones(121, dtype=bool)))
    monkeypatch.setattr(
        module,
        "build_non_event_risk_catalog",
        lambda *_args, **_kwargs: pd.DataFrame({"timestamp_sec": [15.0], "stage": [2]}),
    )

    meta = module.process_subject(subject, root=tmp_path, output_root=tmp_path, sample_hz=1.0, seed=7)

    n3 = pd.read_parquet(tmp_path / "event_phase_n3_so_v1" / f"{subject}_event_phase_n3_so_v1.parquet")
    rem = pd.read_parquet(tmp_path / "event_phase_rem_theta_v1" / f"{subject}_event_phase_rem_theta_v1.parquet")
    assert n3["event_phase_contract"].eq("event_phase_n3_so_v1").all()
    assert rem["event_phase_contract"].eq("event_phase_rem_theta_v1").all()
    assert meta["n3_event_phase_rows"] == len(n3)
    assert meta["rem_event_phase_rows"] == len(rem)
    assert "event_phase_n3_so_contract" in json.loads(
        (tmp_path / f"{subject}_sleep_eap_phase2_meta.json").read_text(encoding="utf-8")
    )
