"""P0.8/P0.9: regional NaN policy and realized dt / derivative contract."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.regions import aggregate_group_timeseries, select_complete_group_rois
from mndm.pipeline.regional_mnps import (
    compute_all_regional_mnps,
    compute_regional_mnps_for_network,
    merge_regional_mnps_estimator_config,
)
from mndm.pipeline.summary import _regional_result_to_h5_payload
from mndm.pipeline.summary_regional import (
    build_precomputed_network_trajectories,
    compute_regional_context,
    infer_dt_realized_sec,
)
from mndm.projection import estimate_derivatives

FMRI_STEP_SEC = 15.0
FMRI_DERIV = {"method": "sav_gol", "window": 5, "polyorder": 2}


def _fake_mnps(n: int = 40, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, 3)).astype(np.float32)


def _noop_mnps_3d(_cfg):
    return {"mode": "direct"}


def _noop_coerce(_v1, _sub):
    return {}


def _noop_align(arr, _names, _ordered):
    return arr


class TestRegionalNanPolicy:
    def test_as_mnps_does_not_zero_fill_before_jacobian(self):
        mnps = _fake_mnps(40)
        mnps[7, 0] = np.nan
        result = compute_regional_mnps_for_network(
            "DMN",
            mnps,
            {"mnps": {"time_step_sec": FMRI_STEP_SEC}, "jacobian": {"enabled": True}},
            min_length=5,
        )
        assert result.valid
        assert np.isnan(result.mnps[7, 0])
        assert result.jacobian is None
        assert result.jacobian_diagnostics.get("reason") == "non_finite_mnps"
        assert result.mnps[7, 0] != 0.0

    def test_all_nan_network_is_dropped(self):
        mnps = np.full((20, 3), np.nan, dtype=np.float32)
        result = compute_regional_mnps_for_network(
            "DMN",
            mnps,
            {"jacobian": {"enabled": False}},
            min_length=5,
        )
        assert not result.valid
        assert "too few finite" in (result.drop_reason or "")

        summary = compute_all_regional_mnps(
            group_ts=None,
            sfreq=None,
            config={"min_segment_length_tr": 5, "jacobian": {"enabled": False}},
            subject="sub-001",
            network_mnps={"DMN": mnps, "VIS": _fake_mnps(20, seed=1)},
        )
        assert summary.n_dropped == 1
        assert summary.n_networks == 1
        assert "DMN" not in summary.results
        assert "VIS" in summary.results

    def test_all_nan_roi_is_skipped_not_zero_filled(self):
        bold = np.ones((3, 12), dtype=float)
        bold[0, :] = np.nan
        bold[1, :] = np.linspace(0.0, 1.0, 12)
        bold[2, :] = np.linspace(1.0, 2.0, 12)
        out = aggregate_group_timeseries(bold, {"VIS": [0, 1, 2]})
        assert "VIS" in out
        assert np.isfinite(out["VIS"]).all()
        # A zero-filled all-NaN ROI would pull the PC toward a constant-0 series.
        assert not np.allclose(out["VIS"], 0.0)

    def test_group_omitted_when_no_finite_roi(self):
        bold = np.full((2, 8), np.nan)
        out = aggregate_group_timeseries(bold, {"DMN": [0, 1]})
        assert "DMN" not in out

    def test_partial_nan_roi_is_dropped_not_median_filled(self):
        rng = np.random.default_rng(1)
        good = rng.normal(size=16)
        dirty = good + 3.0
        dirty[4] = np.nan
        bold = np.stack([dirty, good], axis=0)
        out = aggregate_group_timeseries(bold, {"DAN": [0, 1]})
        assert "DAN" in out
        centered = good - good.mean()
        if np.nansum(centered) < 0:
            centered = -centered
        assert np.allclose(out["DAN"], centered)

    def test_select_complete_group_rois_drops_incomplete_rows(self):
        bold = np.ones((3, 8), dtype=float)
        bold[0, :] = np.nan
        bold[1, 2] = np.nan
        kept, names = select_complete_group_rois(
            bold, [0, 1, 2], ["Vis_a", "Vis_b", "Vis_c"]
        )
        assert kept is not None
        assert kept.shape == (1, 8)
        assert names == ["Vis_c"]
        none_kept, none_names = select_complete_group_rois(bold, [0, 1])
        assert none_kept is None
        assert none_names is None

    def test_network_feature_path_drops_incomplete_rois(self, monkeypatch):
        captured = {}

        def _fake_features(data, _config, **kwargs):
            captured["signals"] = np.asarray(data["signals"]["fmri"])
            captured["roi_indices"] = kwargs.get("roi_indices")
            captured["session_continuous"] = kwargs.get("session_continuous")
            idx = captured["roi_indices"]
            if idx is not None:
                captured["signals"] = captured["signals"][np.asarray(idx, dtype=int)]
            captured["channels"] = list((data.get("channels") or {}).get("fmri") or [])
            return pd.DataFrame(
                {
                    "epoch_id": np.arange(2),
                    "feat_m": np.ones(2),
                    "feat_d": np.ones(2),
                    "feat_e": np.ones(2),
                    "t_start": np.arange(2) * 15.0,
                    "t_end": np.arange(2) * 15.0 + 45.0,
                }
            )

        monkeypatch.setattr(
            "mndm.pipeline.summary_regional.fmri_features.compute_fmri_features",
            _fake_features,
        )
        monkeypatch.setattr(
            "mndm.pipeline.summary_regional.projection.project_features",
            lambda *_a, **_k: (np.ones((2, 3), dtype=np.float32), {}),
        )
        bold = np.ones((3, 16), dtype=float)
        bold[0, :] = np.nan
        bold[1, 3] = np.nan
        mnps, _strat = build_precomputed_network_trajectories(
            regions_bold=bold,
            regions_names=["Vis_a", "Vis_b", "Vis_c"],
            regions_sfreq=0.5,
            region_groups={"VIS": [0, 1, 2]},
            axis_weights={"m": {"feat_m": 1.0}, "d": {"feat_d": 1.0}, "e": {"feat_e": 1.0}},
            dataset_id="ds_test",
            config={},
            sub_frame=pd.DataFrame({"epoch_id": [0, 1]}),
            proj_cfg={},
            normalize_mode=None,
            subcoords_spec={},
            v2_enabled=False,
            resolve_mnps_3d_cfg=_noop_mnps_3d,
            coerce_v1_mapping_to_v2_subcoords=_noop_coerce,
            align_v2_subcoords=_noop_align,
        )
        assert "VIS" in mnps
        assert captured["signals"].shape[0] == 1
        roi_idx = captured.get("roi_indices")
        if roi_idx is not None:
            assert list(np.atleast_1d(np.asarray(roi_idx, dtype=int))) == [2]
        assert np.isfinite(captured["signals"]).all()


class TestRegionalDtContract:
    def test_infer_dt_from_fmri_step_and_t_start(self):
        frame = pd.DataFrame(
            {
                "t_start": [0.0, 15.0, 30.0, 45.0],
                "t_end": [45.0, 60.0, 75.0, 90.0],
                "fmri_step_sec": [15.0, 15.0, 15.0, 15.0],
            }
        )
        assert infer_dt_realized_sec(frame) == pytest.approx(FMRI_STEP_SEC)
        no_step = frame.drop(columns=["fmri_step_sec"])
        assert infer_dt_realized_sec(no_step) == pytest.approx(FMRI_STEP_SEC)

    def test_merge_inherits_parent_derivative_and_realized_dt(self):
        merged = merge_regional_mnps_estimator_config(
            {"enabled": True},
            parent_mnps_cfg={"derivative": FMRI_DERIV},
            dt_realized_sec=FMRI_STEP_SEC,
            derivative_cfg=FMRI_DERIV,
        )
        assert merged["mnps"]["time_step_sec"] == pytest.approx(FMRI_STEP_SEC)
        assert merged["mnps"]["dt_source"] == "measured"
        assert merged["mnps"]["derivative"]["window"] == 5
        assert merged["mnps"]["derivative"]["polyorder"] == 2
        assert merged["mnps"]["derivative"]["method"] == "sav_gol"

    def test_regional_explicit_derivative_wins(self):
        merged = merge_regional_mnps_estimator_config(
            {"mnps": {"derivative": {"method": "central", "window": 3, "polyorder": 1}}},
            parent_mnps_cfg={"derivative": FMRI_DERIV},
            dt_realized_sec=FMRI_STEP_SEC,
            derivative_cfg=FMRI_DERIV,
        )
        assert merged["mnps"]["derivative"]["method"] == "central"
        assert merged["mnps"]["time_step_sec"] == pytest.approx(FMRI_STEP_SEC)

    def test_central_dot_uses_dt_15_not_unit_step(self):
        t = np.arange(20, dtype=np.float32)
        mnps = np.stack([t, t, t], axis=1)
        result = compute_regional_mnps_for_network(
            "DMN",
            mnps,
            {
                "mnps": {
                    "time_step_sec": FMRI_STEP_SEC,
                    "derivative": {"method": "central"},
                },
                "jacobian": {"enabled": False},
            },
            min_length=5,
        )
        assert result.valid
        assert result.dt_realized_sec == pytest.approx(FMRI_STEP_SEC)
        interior = result.mnps_dot[1:-1]
        assert np.allclose(interior, 1.0 / FMRI_STEP_SEC, atol=1e-6)
        unit_step = estimate_derivatives(mnps, 1.0, "central")
        assert not np.allclose(result.mnps_dot[1:-1], unit_step[1:-1])

    def test_sg_window_matches_common_fmri_5_2(self):
        mnps = _fake_mnps(40)
        cfg = merge_regional_mnps_estimator_config(
            {"jacobian": {"enabled": False}},
            parent_mnps_cfg={"derivative": FMRI_DERIV},
            dt_realized_sec=FMRI_STEP_SEC,
            derivative_cfg=FMRI_DERIV,
        )
        result = compute_regional_mnps_for_network("DMN", mnps, cfg, min_length=5)
        expected = estimate_derivatives(mnps, FMRI_STEP_SEC, "sav_gol", 5, 2)
        defaulted = estimate_derivatives(mnps, 1.0, "sav_gol", 7, 3)
        assert result.derivative_window == 5
        assert result.derivative_polyorder == 2
        assert result.dt_realized_sec == pytest.approx(FMRI_STEP_SEC)
        assert result.dt_source == "measured"
        assert np.allclose(result.mnps_dot, expected, equal_nan=True)
        assert not np.allclose(result.mnps_dot, defaulted, equal_nan=True)

    def test_h5_payload_serializes_dt_realized_sec(self):
        result = compute_regional_mnps_for_network(
            "DMN",
            _fake_mnps(20),
            {
                "mnps": {"time_step_sec": FMRI_STEP_SEC, "derivative": FMRI_DERIV},
                "jacobian": {"enabled": False},
            },
            min_length=5,
        )
        payload = _regional_result_to_h5_payload(result, coordinate_contract="subject_anchored")
        assert payload["attrs"]["dt_realized_sec"] == pytest.approx(FMRI_STEP_SEC)
        assert payload["attrs"]["dt_source"] == "config"
        assert payload["attrs"]["derivative_window"] == 5
        assert payload["attrs"]["derivative_polyorder"] == 2

    def test_compute_regional_context_passes_realized_dt(self, monkeypatch):
        captured = {}

        def _fake_compute_all(**kwargs):
            captured["config"] = kwargs["config"]
            from mndm.pipeline.regional_mnps import RegionalMNPSSummary

            return RegionalMNPSSummary(
                subject=kwargs["subject"],
                session=kwargs.get("session"),
                condition=kwargs.get("condition"),
                task=kwargs.get("task"),
            )

        def _fake_build(**_kwargs):
            return {"DMN": _fake_mnps(12)}, {}

        monkeypatch.setattr(
            "mndm.pipeline.summary_regional.compute_all_regional_mnps",
            _fake_compute_all,
        )
        monkeypatch.setattr(
            "mndm.pipeline.summary_regional.build_precomputed_network_trajectories",
            _fake_build,
        )
        sub_frame = pd.DataFrame(
            {
                "t_start": np.arange(8, dtype=float) * FMRI_STEP_SEC,
                "t_end": np.arange(8, dtype=float) * FMRI_STEP_SEC + 45.0,
                "fmri_step_sec": np.full(8, FMRI_STEP_SEC),
            }
        )
        regions_bold = np.ones((2, 16), dtype=float)
        compute_regional_context(
            sub_frame=sub_frame,
            regions_bold=regions_bold,
            regions_names=["Vis_left", "Vis_right"],
            regions_sfreq=1.0 / 2.0,
            config={
                "modality": "fmri",
                "mnps": {"derivative": FMRI_DERIV},
            },
            regional_mnps_cfg={"enabled": True, "min_regions_required": 0},
            subcoords_spec={},
            axis_weights={},
            dataset_id="ds_test",
            dataset_label="ds_test/sub-001",
            proj_cfg={},
            normalize_mode=None,
            external_anchor=None,
            subject="sub-001",
            session=None,
            condition=None,
            task="rest",
            resolve_mnps_3d_cfg=_noop_mnps_3d,
            coerce_v1_mapping_to_v2_subcoords=_noop_coerce,
            align_v2_subcoords=_noop_align,
            dt_realized_sec=FMRI_STEP_SEC,
            parent_mnps_cfg={"derivative": FMRI_DERIV},
            derivative_cfg=FMRI_DERIV,
        )
        assert captured["config"]["mnps"]["time_step_sec"] == pytest.approx(FMRI_STEP_SEC)
        assert captured["config"]["mnps"]["dt_source"] == "measured"
        assert captured["config"]["mnps"]["derivative"]["window"] == 5
        assert captured["config"]["mnps"]["derivative"]["polyorder"] == 2

    def test_invalid_dt_is_not_silently_replaced_with_one(self):
        result = compute_regional_mnps_for_network(
            "DMN",
            _fake_mnps(20),
            {"mnps": {"time_step_sec": 0.0}, "jacobian": {"enabled": False}},
            min_length=5,
        )
        assert not result.valid
        assert "invalid_dt_realized_sec" in (result.drop_reason or "")

    def test_standalone_default_dt_is_labelled_not_measured(self):
        result = compute_regional_mnps_for_network(
            "DMN",
            _fake_mnps(20),
            {"jacobian": {"enabled": False}},
            min_length=5,
        )
        assert result.valid
        assert result.dt_realized_sec == pytest.approx(1.0)
        assert result.dt_source == "default_1"
        payload = _regional_result_to_h5_payload(result, coordinate_contract="subject_anchored")
        assert payload["attrs"]["dt_source"] == "default_1"
        assert payload["attrs"]["dt_realized_sec"] == pytest.approx(1.0)
