"""Static contracts for the JQ1 production-pipeline synthetic fixture."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_jq1_fixture_has_all_frozen_truth_families_and_fresh_seeds():
    from mndm.tools.run_jq1_synthetic import SEEDS, truth_cases

    cases = truth_cases()
    assert SEEDS == (611, 612, 613, 614)
    assert {case.identifier for case in cases if case.family == "J1"} == {
        "T0",
        "T1",
        "T2",
        "T3",
        "T4",
        "T5",
        "T8",
        "T9",
    }
    assert {
        case.identifier for case in cases if case.family == "J2"
    } == {
        "S0_P0_zero_process_noise",
        "S1_P0_zero_process_noise",
        "S2_P0_zero_process_noise",
        "S0_P1_reference_process_noise",
        "S1_P1_reference_process_noise",
        "S2_P1_reference_process_noise",
    }
    assert len(cases) == 56


def test_jq1_fixture_is_three_dimensional_with_identity_mde_mapping(tmp_path):
    from mndm.tools.run_jq1_synthetic import DT, _config, truth_cases

    case = truth_cases()[0]
    assert case.matrix_3d.shape == (3, 3)
    assert np.allclose(case.matrix_3d[:2, :2], case.matrix_2d)
    assert case.matrix_3d[2, 2] == -4.0

    config = _config(tmp_path / "processed", tmp_path / "received")
    assert config["mnps"]["window_sec"] * (1.0 - config["mnps"]["overlap"]) == DT
    assert config["mnps_projection"]["normalize"] is None
    assert config["mnps_projection"]["feature_standardization"] == {}
    assert config["mnps_projection"]["weights"] == {
        "m": {"synthetic_m": 1.0},
        "d": {"synthetic_d": 1.0},
        "e": {"synthetic_e": 1.0},
    }
