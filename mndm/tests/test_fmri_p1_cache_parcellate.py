"""P1: atlas/ROI-TS cache, bincount parcellation, regional session reuse."""

from pathlib import Path
import json
import logging
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("nibabel")
import nibabel as nib  # noqa: E402

from mndm.features.fmri import compute_fmri_features, prepare_session_continuous
from mndm.features.fmri_epoch_metrics import fc_mean_for_indices
from mndm.pipeline.fmri_parcellate import (
    parcellate_label_means,
    parcellate_label_means_loop,
    parcellate_label_means_scatter,
)
from mndm.pipeline.fmri_roi_cache import (
    atlas_cache_size,
    build_roi_ts_cache_key,
    clear_atlas_cache,
    try_load_roi_ts,
)
from mndm.pipeline.summary_selectors import load_regional_fmri_signals
from mndm.preprocess import preprocess_fmri, roi_ts_cache_lookup_args

AFFINE = np.diag([2.0, 2.0, 2.0, 1.0]).astype(float)
TR_SEC = 2.0


def _write_nifti(path: Path, data: np.ndarray, zooms=None) -> Path:
    img = nib.Nifti1Image(np.asarray(data), AFFINE)
    hdr = img.header
    if zooms is None:
        zooms = (2.0, 2.0, 2.0, TR_SEC)
    hdr.set_zooms(zooms[: data.ndim])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(path))
    return path


def _write_sidecar(bold: Path, tr: float = TR_SEC) -> Path:
    side = bold.with_name(bold.name.replace(".nii.gz", ".json").replace(".nii", ".json"))
    side.write_text(json.dumps({"RepetitionTime": tr}), encoding="utf-8")
    return side


def _constant_roi_volume():
    """4 labels, each a 2x2x2 block, constant BOLD = label id + 0.01 * t."""
    atlas = np.zeros((4, 4, 4), dtype=np.int16)
    atlas[:2, :2, :] = 1
    atlas[2:, :2, :] = 2
    atlas[:2, 2:, :] = 3
    atlas[2:, 2:, :] = 4
    n_t = 12
    bold = np.zeros((4, 4, 4, n_t), dtype=np.float32)
    t = np.arange(n_t, dtype=np.float32)
    for lab in (1, 2, 3, 4):
        bold[atlas == lab, :] = lab + 0.01 * t
    return atlas, bold


def _fmri_config(tmp_path: Path, atlas: Path, dataset_id: str = "ds999999") -> dict:
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    return {
        "datasets": [dataset_id],
        "paths": {"processed_dir": str(processed), "received_dir": str(tmp_path / "received")},
        "preprocess": {
            "fmri": {
                "atlas_path": str(atlas),
                "min_regions_required": 1,
                "min_region_fraction": 0.0,
                "fallback_tr": TR_SEC,
            }
        },
        "preprocessing": {"bandpass": [0.01, 0.1], "compute_phase": True},
        "features": {"fmri": {"window_sec": 10.0, "step_sec": 5.0}},
        "epoching": {"length_s": 10.0, "step_s": 5.0},
    }


class TestParcellateBincount:
    def test_known_roi_constants_match_loop_exactly(self):
        atlas, bold = _constant_roi_volume()
        bold = bold.astype(np.float64)
        labels_a, ts_a = parcellate_label_means_loop(bold, atlas)
        labels_b, ts_b = parcellate_label_means_scatter(bold, atlas)
        labels_c, ts_c = parcellate_label_means(bold, atlas)
        assert np.array_equal(labels_a, labels_b)
        assert np.array_equal(ts_a, ts_b)
        assert np.array_equal(labels_a, labels_c)
        assert np.array_equal(ts_a, ts_c)
        t = np.arange(bold.shape[3], dtype=np.float64)
        for i, lab in enumerate(labels_b):
            expected = np.asarray(float(lab) + 0.01 * t, dtype=np.float32)
            assert np.array_equal(ts_b[i], expected)

    def test_nan_voxels_match_nanmean_not_zero_fill(self):
        atlas, bold = _constant_roi_volume()
        bold = bold.astype(np.float64, copy=True)
        mask = atlas == 1
        vol = bold[:, :, :, 3]
        vol[mask] = np.nan
        bold[:, :, :, 3] = vol
        _, ts_loop = parcellate_label_means_loop(bold, atlas)
        _, ts_bin = parcellate_label_means_scatter(bold, atlas)
        assert np.array_equal(np.isnan(ts_loop), np.isnan(ts_bin))
        assert np.allclose(ts_loop, ts_bin, equal_nan=True, atol=0, rtol=0)
        assert np.isnan(ts_bin[0, 3])


class TestAtlasAndRoiTsCache:
    def setup_method(self):
        clear_atlas_cache()

    def test_second_preprocess_hits_atlas_and_roi_ts_cache(self, tmp_path: Path):
        atlas_vol, bold_vol = _constant_roi_volume()
        ds = tmp_path / "ds999999"
        atlas_path = ds / "atlas.nii.gz"
        bold_path = ds / "sub-001_task-rest_bold.nii.gz"
        _write_nifti(atlas_path, atlas_vol)
        _write_nifti(bold_path, bold_vol)
        _write_sidecar(bold_path)
        cfg = _fmri_config(tmp_path, atlas_path)
        first = preprocess_fmri(bold_path, cfg)
        assert first.meta.get("roi_ts_cache_hit") == 0
        assert atlas_cache_size() == 1
        second = preprocess_fmri(bold_path, cfg)
        assert second.meta.get("roi_ts_cache_hit") == 1
        assert np.array_equal(first.signals["fmri"], second.signals["fmri"])
        assert first.channels["fmri"] == second.channels["fmri"]

    def test_basename_collision_does_not_share_cache(self, tmp_path: Path):
        atlas_vol, bold_a = _constant_roi_volume()
        bold_b = bold_a + 5.0
        atlas_path = tmp_path / "atlas.nii.gz"
        _write_nifti(atlas_path, atlas_vol)
        a_path = tmp_path / "runA" / "sub-001_task-rest_bold.nii.gz"
        b_path = tmp_path / "runB" / "sub-001_task-rest_bold.nii.gz"
        _write_nifti(a_path, bold_a)
        _write_nifti(b_path, bold_b)
        _write_sidecar(a_path)
        _write_sidecar(b_path)
        cfg = _fmri_config(tmp_path, atlas_path)
        out_a = preprocess_fmri(a_path, cfg)
        out_b = preprocess_fmri(b_path, cfg)
        assert out_a.meta.get("roi_ts_cache_key") != out_b.meta.get("roi_ts_cache_key")
        assert not np.allclose(out_a.signals["fmri"], out_b.signals["fmri"])

    def test_summarize_loads_roi_ts_without_nifti(self, tmp_path: Path):
        atlas_vol, bold_vol = _constant_roi_volume()
        ds = tmp_path / "ds999999"
        atlas_path = ds / "atlas.nii.gz"
        bold_path = ds / "sub-001_task-rest_bold.nii.gz"
        _write_nifti(atlas_path, atlas_vol)
        _write_nifti(bold_path, bold_vol)
        _write_sidecar(bold_path)
        cfg = _fmri_config(tmp_path, atlas_path)
        pre = preprocess_fmri(bold_path, cfg)
        bold_path.unlink()
        assert not bold_path.exists()
        lookup = roi_ts_cache_lookup_args(bold_path, cfg)
        assert lookup is not None
        cached = try_load_roi_ts(
            bold_path=bold_path,
            config=cfg,
            dataset_id="ds999999",
            config_identity=lookup["identity"],
            require_bold_exists=False,
        )
        assert cached is not None
        assert try_load_roi_ts(
            bold_path=bold_path,
            config=cfg,
            dataset_id="ds999999",
            require_bold_exists=False,
        ) is None
        regions, names, sfreq = load_regional_fmri_signals(
            sub_id="sub-001",
            dataset_label="ds999999:sub-001",
            config=cfg,
            sub_frame=pd.DataFrame({"file": [str(bold_path)]}),
            raw_task="rest",
            condition=None,
            session=None,
            run_id=None,
            dataset_root=tmp_path,
            index_df=None,
            lookup_rel_paths_by_file_value=lambda _v: [],
            preprocess_fmri=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("must not open 4D")),
            logger=logging.getLogger("test"),
            dataset_id="ds999999",
        )
        assert regions is not None
        assert np.array_equal(regions, pre.signals["fmri"])
        assert names == pre.channels["fmri"]
        assert sfreq == pytest.approx(pre.sfreq)

    def test_stale_atlas_does_not_reuse_roi_ts(self, tmp_path: Path):
        atlas_vol, bold_vol = _constant_roi_volume()
        ds = tmp_path / "ds999999"
        atlas_a = ds / "atlas_a.nii.gz"
        atlas_b = ds / "atlas_b.nii.gz"
        bold_path = ds / "sub-001_task-rest_bold.nii.gz"
        other = atlas_vol.copy()
        other[atlas_vol == 4] = 5
        _write_nifti(atlas_a, atlas_vol)
        _write_nifti(atlas_b, other)
        _write_nifti(bold_path, bold_vol)
        _write_sidecar(bold_path)
        cfg_a = _fmri_config(tmp_path, atlas_a)
        preprocess_fmri(bold_path, cfg_a)
        cfg_b = _fmri_config(tmp_path, atlas_b)
        lookup_b = roi_ts_cache_lookup_args(bold_path, cfg_b, nifti_zooms=(2.0, 2.0, 2.0, TR_SEC))
        assert lookup_b is not None
        cached = try_load_roi_ts(
            bold_path=bold_path,
            config=cfg_b,
            dataset_id="ds999999",
            expected_key=build_roi_ts_cache_key(**lookup_b["kwargs"]),
            config_identity=lookup_b["identity"],
            require_bold_exists=True,
        )
        assert cached is None
        regions, _names, _sfreq = load_regional_fmri_signals(
            sub_id="sub-001",
            dataset_label="ds999999:sub-001",
            config=cfg_b,
            sub_frame=pd.DataFrame({"file": [str(bold_path)]}),
            raw_task="rest",
            condition=None,
            session=None,
            run_id=None,
            dataset_root=tmp_path,
            index_df=None,
            lookup_rel_paths_by_file_value=lambda _v: [],
            preprocess_fmri=lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("stale cache must miss")),
            logger=logging.getLogger("test"),
            dataset_id="ds999999",
        )
        assert regions is None


class TestRegionalSessionReuse:
    def test_network_fc_mean_matches_global_fc_slice(self):
        rng = np.random.default_rng(0)
        n_roi, n_t = 8, 80
        sfreq = 1.0
        roi_ts = rng.normal(size=(n_roi, n_t))
        names = [f"Vis_{i}" if i < 4 else f"Default_{i}" for i in range(n_roi)]
        cfg = {
            "preprocessing": {"bandpass": [0.02, 0.2], "compute_phase": True},
            "features": {"fmri": {"window_sec": 20.0, "step_sec": 10.0}},
            "metrics": {"compute_modularity": False, "compute_dfc_variance": False},
        }
        signals = {
            "signals": {"fmri": roi_ts},
            "sfreq": sfreq,
            "channels": {"fmri": names},
        }
        session = prepare_session_continuous(signals, cfg)
        filtered = np.asarray(session["filtered_ts"])
        window_samples = 20
        epoch_fc = np.corrcoef(filtered[:, :window_samples])
        vis_idx = np.arange(4)
        expected = fc_mean_for_indices(epoch_fc, vis_idx)
        vis = compute_fmri_features(
            signals,
            cfg,
            session_continuous=session,
            roi_indices=vis_idx,
        )
        full = compute_fmri_features(signals, cfg)
        assert vis["fmri_FC_mean"].iloc[0] == pytest.approx(expected, rel=0, abs=1e-12)
        # Reuse equals slicing the same session filter, not a second filter pass.
        sliced = compute_fmri_features(
            {
                "signals": {"fmri": roi_ts[vis_idx]},
                "sfreq": sfreq,
                "channels": {"fmri": names[:4]},
            },
            cfg,
        )
        assert vis["fmri_FC_mean"].to_numpy() == pytest.approx(
            sliced["fmri_FC_mean"].to_numpy(), rel=0, abs=1e-10
        )
        assert full["fmri_FC_mean"].iloc[0] != pytest.approx(expected)
