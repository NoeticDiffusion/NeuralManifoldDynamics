import numpy as np
import pandas as pd

from neuropixel_ingest.lfp_state import LfpStateContract, annotate_lfp_features


def test_feature_annotation_uses_midpoint_and_stim_guard():
    contract = LfpStateContract(
        blocks=pd.DataFrame(
            [
                {"state": "awake", "start_sec": 10.0, "stop_sec": 20.0},
                {"state": "isoflurane", "start_sec": 20.0, "stop_sec": 30.0},
            ]
        ),
        stim_events=pd.DataFrame({"start_time": [14.0, 24.0], "estim_current": [40, 60]}),
        time_origin_sec=10.0,
        recording_stop_sec=30.0,
    )
    features = pd.DataFrame({"t_start": [2.0, 10.0], "t_end": [6.0, 14.0]})
    output = annotate_lfp_features(features, contract, stim_guard_sec=1.0)
    assert output["lfp_behavioral_state"].tolist() == ["awake", "isoflurane"]
    assert output["lfp_stim_contains_onset"].tolist() == [True, False]
    assert output["lfp_stim_adjacent"].tolist() == [True, False]
    assert output["lfp_nearest_estim_current"].tolist() == [40.0, 60.0]
    assert output["lfp_interstim_primary"].tolist() == [False, True]


def test_feature_annotation_excludes_outside_blocks():
    contract = LfpStateContract(
        blocks=pd.DataFrame([{"state": "awake", "start_sec": 5.0, "stop_sec": 10.0}]),
        stim_events=pd.DataFrame({"start_time": [7.0]}),
        time_origin_sec=0.0,
        recording_stop_sec=10.0,
    )
    output = annotate_lfp_features(pd.DataFrame({"t_start": [0.0], "t_end": [2.0]}), contract)
    assert output["lfp_behavioral_state"].isna().all()
    assert np.isfinite(output["lfp_nearest_stim_distance_sec"]).all()


def test_induction_exclusion_uses_contract_interval_not_fixed_session_time():
    contract = LfpStateContract(
        blocks=pd.DataFrame([{"state": "isoflurane", "start_sec": 100.0, "stop_sec": 200.0}]),
        stim_events=pd.DataFrame({"start_time": [110.0]}),
        time_origin_sec=0.0,
        recording_stop_sec=200.0,
        pharmacological_intervals=pd.DataFrame(
            [{"tag": "isoflurane_induction", "start_sec": 130.0, "stop_sec": 140.0}]
        ),
    )
    output = annotate_lfp_features(
        pd.DataFrame({"t_start": [128.0, 148.0], "t_end": [132.0, 152.0]}),
        contract,
        stim_guard_sec=0.1,
        exclude_induction=True,
    )
    assert output["lfp_induction_excluded"].tolist() == [True, False]
