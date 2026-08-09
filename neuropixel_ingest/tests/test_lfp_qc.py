import pandas as pd
import h5py
import numpy as np

from neuropixel_ingest.lfp_compare import _depth_columns, write_condition_sensitivity
from neuropixel_ingest.lfp_qc import select_depth_balanced, write_qc_selection_sensitivity, write_selection_from_qc


def test_depth_horizontal_selection_is_balanced_and_deterministic():
    rows = []
    for depth in (0, 1):
        for horizontal in (0, 1, 2, 3):
            for replica in range(2):
                rows.append(
                    {
                        "channel_index": depth * 100 + horizontal * 10 + replica,
                        "depth_bin": depth,
                        "probe_horizontal_position": horizontal,
                        "qc_status": "good",
                        "qc_score": 10 - replica,
                    }
                )
    selected = select_depth_balanced(pd.DataFrame(rows), per_depth_target=8)
    assert selected["channel_index"].tolist() == [row["channel_index"] for row in rows]


def test_depth_selection_falls_back_within_stratum_only():
    qc = pd.DataFrame(
        [
            {"channel_index": 0, "depth_bin": 0, "probe_horizontal_position": 0, "qc_status": "good", "qc_score": 5},
            {"channel_index": 1, "depth_bin": 0, "probe_horizontal_position": 0, "qc_status": "good", "qc_score": 4},
            {"channel_index": 2, "depth_bin": 0, "probe_horizontal_position": 1, "qc_status": "good", "qc_score": 3},
            {"channel_index": 3, "depth_bin": 1, "probe_horizontal_position": 0, "qc_status": "rejected", "qc_score": 9},
        ]
    )
    selected = select_depth_balanced(qc, per_depth_target=4)
    assert selected["channel_index"].tolist() == [0, 1, 2]


def test_shared_depth_discovery_excludes_non_ensemble_columns():
    frame = pd.DataFrame({"eeg_delta__g_depth_0": [1.0], "eeg_theta__g_depth_3": [2.0], "eeg_delta": [3.0]})
    assert _depth_columns(frame) == {"depth_0", "depth_3"}


def test_selection_sensitivity_reports_threshold_and_contact_variants(tmp_path):
    qc = pd.DataFrame(
        {
            "channel_index": [0, 1],
            "depth_bin": [0, 0],
            "probe_horizontal_position": [0, 1],
            "qc_status": ["good", "good"],
            "qc_score": [1.0, 0.5],
            "metadata_valid": [True, True],
            "finite_geometry": [True, True],
            "qc_failed_fraction": [0.0, 0.25],
        }
    )
    input_path, output_path = tmp_path / "qc.parquet", tmp_path / "sensitivity.json"
    qc.to_parquet(input_path)
    report = write_qc_selection_sensitivity(input_path, output_path)
    assert report["thresholds"]["0.1"]["n_passing"] == 1
    assert report["thresholds"]["0.3"]["n_passing"] == 2
    assert output_path.exists()


def test_contact_selection_artifact_has_requested_count(tmp_path):
    qc = pd.DataFrame(
        {
            "channel_index": range(8),
            "depth_bin": [0] * 8,
            "probe_horizontal_position": [0, 0, 1, 1, 2, 2, 3, 3],
            "qc_status": ["good"] * 8,
            "qc_score": list(reversed(range(8))),
        }
    )
    qc_path, base_path, output_path = tmp_path / "qc.parquet", tmp_path / "base.json", tmp_path / "four.json"
    qc.to_parquet(qc_path)
    base_path.write_text('{"selection_policy": {}, "provenance": {}}', encoding="utf-8")
    result = write_selection_from_qc(qc_path, base_path, output_path, per_depth_target=4)
    assert result["target_n_channels"] == 4
    assert len(result["selected_channels"]) == 4


def test_condition_sensitivity_aligns_feature_and_h5_outputs(tmp_path):
    left = pd.DataFrame({"epoch_id": [0, 1], "file": ["x", "x"], "eeg_delta": [1.0, 2.0]})
    right = left.assign(eeg_delta=[2.0, 4.0])
    left_path, right_path = tmp_path / "left.parquet", tmp_path / "right.parquet"
    left.to_parquet(left_path)
    right.to_parquet(right_path)
    for path, scale in ((tmp_path / "left.h5", 1.0), (tmp_path / "right.h5", 2.0)):
        with h5py.File(path, "w") as handle:
            handle["mnps_3d"] = np.arange(9, dtype=float).reshape(3, 3) * scale
    report = write_condition_sensitivity(left_path, right_path, tmp_path / "left.h5", tmp_path / "right.h5", tmp_path / "out.json")
    assert report["n_matched_feature_epochs"] == 2
    assert report["mnps_coordinate_correlations"] == [1.0, 1.0, 1.0]
