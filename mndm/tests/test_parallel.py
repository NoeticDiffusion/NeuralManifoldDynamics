"""Tests for parallel processing helpers."""

from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_merge_temp_features_dedupes_by_file_epoch_id(tmp_path: Path):
    """Test merge temp features dedupes by file epoch id."""
    from mndm.parallel import merge_temp_features

    ds_path = tmp_path / "ds"
    ds_path.mkdir(parents=True, exist_ok=True)

    # Existing features row
    pd.DataFrame(
        [{"file": "a.edf", "epoch_id": 1, "feat": 10.0, "qc_ok_eeg": 1}]
    ).to_csv(ds_path / "features.csv", index=False)

    # Temp row updates same (file, epoch_id) + one new row
    pd.DataFrame(
        [
            {"file": "a.edf", "epoch_id": 1, "feat": 20.0, "qc_ok_eeg": 1},
            {"file": "a.edf", "epoch_id": 2, "feat": 30.0, "qc_ok_eeg": 1},
        ]
    ).to_csv(ds_path / "features_hash1.csv", index=False)

    merged = merge_temp_features(ds_path)
    assert len(merged) == 2
    row1 = merged[(merged["file"] == "a.edf") & (merged["epoch_id"] == 1)].iloc[0]
    assert float(row1["feat"]) == 20.0


def test_merge_temp_features_fallback_drop_duplicates_without_keys(tmp_path: Path):
    """Test merge temp features fallback drop duplicates without keys."""
    from mndm.parallel import merge_temp_features

    ds_path = tmp_path / "ds2"
    ds_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([{"x": 1.0}, {"x": 1.0}]).to_csv(ds_path / "features_hash2.csv", index=False)
    merged = merge_temp_features(ds_path)
    assert len(merged) == 1


def test_write_qc_json_serializes_numpy_types(tmp_path: Path):
    """Test write qc json serializes numpy types."""
    from mndm.parallel import write_qc_json

    meta = {
        "artifact": {
            "method": "asr",
            "n_bad": np.int64(2),
            "bad_frac": np.float32(0.25),
            "enabled": np.bool_(True),
            "bad_idx": np.array([1, 3], dtype=np.int32),
        }
    }
    out_dir = tmp_path / "qc"
    write_qc_json(meta, out_dir, "sub-01_task-rest_run-01_eeg.edf")

    qc_path = out_dir / "sub-01_task-rest_run-01_eeg_qc_artifacts.json"
    assert qc_path.exists()
    with qc_path.open("r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert loaded["artifact"]["n_bad"] == 2
    assert loaded["artifact"]["enabled"] is True
    assert loaded["artifact"]["bad_idx"] == [1, 3]


def test_merge_feature_frames_warns_on_length_only_alignment(caplog):
    """Test merge feature frames warns on length only alignment."""
    from mndm.parallel import _merge_feature_frames

    df_base = pd.DataFrame({"epoch_id": [10, 11, 12], "eeg_feat": [1.0, 2.0, 3.0]})
    df_other = pd.DataFrame({"ecg_feat": [0.1, 0.2, 0.3]})

    with caplog.at_level("WARNING"):
        merged = _merge_feature_frames([df_base, df_other])
    assert "Forcing epoch_id alignment by length-only fallback" in caplog.text
    assert "epoch_id" in merged.columns


def test_merge_feature_frames_adds_embodied_proxy_priority():
    """Test merge feature frames adds embodied proxy priority."""
    from mndm.parallel import _merge_feature_frames

    df_eeg = pd.DataFrame(
        {
            "epoch_id": [0, 1, 2],
            "eeg_highfreq_power_30_45": [1.0, 2.0, 3.0],
        }
    )
    df_eog = pd.DataFrame(
        {
            "epoch_id": [0, 1, 2],
            "eog_blink_rate": [10.0, np.nan, 30.0],
        }
    )
    df_ecg = pd.DataFrame(
        {
            "epoch_id": [0, 1, 2],
            "ecg_rmssd": [np.nan, 20.0, np.nan],
        }
    )
    merged = _merge_feature_frames([df_eeg, df_eog, df_ecg])
    assert "embodied_arousal_proxy" in merged.columns
    assert "embodied_arousal_proxy_source" in merged.columns
    # Priority: ecg_rmssd > eog_blink_rate > eeg_highfreq_power_30_45
    assert np.allclose(
        pd.to_numeric(merged["embodied_arousal_proxy"], errors="coerce").to_numpy(dtype=float),
        np.array([10.0, 20.0, 30.0], dtype=float),
        equal_nan=True,
    )
    assert merged["embodied_arousal_proxy_source"].tolist() == [
        "eog_blink_rate",
        "ecg_rmssd",
        "eog_blink_rate",
    ]


def test_resolve_feature_io_policy_auto_large_prefers_parquet(tmp_path: Path):
    """Test resolve feature io policy auto large prefers parquet."""
    from mndm.parallel import resolve_feature_io_policy

    cfg = {
        "feature_storage": {
            "default_format": "auto",
            "write_both": True,
            "read_prefer": "parquet",
            "auto_large_dataset": {"min_files": 10, "min_total_mb": 1},
        }
    }
    policy = resolve_feature_io_policy(cfg, tmp_path, planned_files=25)
    assert policy["is_large_dataset"] is True
    assert policy["primary_format"] == "parquet"
    assert policy["write_csv"] is True
    assert policy["write_parquet"] is True


def test_merge_temp_features_prefers_parquet_when_both_exist(monkeypatch, tmp_path: Path):
    """Test merge temp features prefers parquet when both exist."""
    from mndm.parallel import merge_temp_features

    ds_path = tmp_path / "ds3"
    ds_path.mkdir(parents=True, exist_ok=True)
    (ds_path / "features_hash3.csv").write_text("x\n1\n", encoding="utf-8")
    (ds_path / "features_hash3.parquet").write_text("not-real-parquet", encoding="utf-8")

    csv_df = pd.DataFrame([{"file": "a.edf", "epoch_id": 1, "feat": 10.0}])
    pq_df = pd.DataFrame([{"file": "a.edf", "epoch_id": 1, "feat": 99.0}])

    monkeypatch.setattr(pd, "read_csv", lambda path, *args, **kwargs: csv_df.copy())
    monkeypatch.setattr(pd, "read_parquet", lambda path, *args, **kwargs: pq_df.copy())

    merged = merge_temp_features(ds_path, io_policy={"read_prefer": "parquet"})
    assert len(merged) == 1
    assert float(merged.iloc[0]["feat"]) == 99.0
