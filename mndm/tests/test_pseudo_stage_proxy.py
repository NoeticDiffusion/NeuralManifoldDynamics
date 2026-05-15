"""Tests for pseudo-stage proxy derivation."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.extractors import derive_pseudo_stage_array


def test_derive_pseudo_stage_array_labels_n2_proxy_windows():
    """Proxy should label N2-like windows when criteria are met."""
    n = 12
    df = pd.DataFrame(
        {
            "eeg_alpha": np.array([8, 9, 10, 11, 12, 11, 3, 2, 2, 1, 1, 1], dtype=float),
            "embodied_arousal_proxy": np.array([1, 1, 1, 2, 2, 2, 8, 9, 8, 9, 10, 9], dtype=float),
            "event_sleep_spindle_count": np.array([3, 3, 4, 3, 2, 2, 0, 0, 0, 0, 0, 0], dtype=float),
        }
    )
    config = {
        "pseudo_stage": {
            "datasets": {
                "physionet_icare_2_1": {
                    "enabled": True,
                    "label": "N2_PROXY",
                    "code": 2,
                    "min_criteria": 3,
                    "min_labeled_fraction": 0.05,
                    "spindle_columns": ["event_sleep_spindle_count"],
                    "spindle_burst_quantile_min": 0.6,
                    "spindle_density_window_epochs": 1,
                    "spindle_density_min": 0.5,
                    "sigma_columns": ["eeg_alpha"],
                    "sigma_quantile_min": 0.5,
                    "emg_columns": ["embodied_arousal_proxy"],
                    "emg_quantile_max": 0.5,
                }
            }
        }
    }

    stage, meta = derive_pseudo_stage_array(
        df,
        config=config,
        dataset_id="physionet_icare_2_1",
        stage_codebook={"N2": 2},
    )

    assert stage is not None
    assert stage.dtype == np.int8
    assert int(np.sum(stage == 2)) >= 4
    assert meta.get("status") == "ok"
    assert meta.get("label") == "N2_PROXY"
    assert int(meta.get("code")) == 2


def test_derive_pseudo_stage_array_returns_none_when_no_columns():
    """Proxy should disable gracefully when required source columns are missing."""
    df = pd.DataFrame({"other_signal": np.arange(10, dtype=float)})
    config = {
        "pseudo_stage": {
            "datasets": {
                "physionet_icare_2_1": {
                    "enabled": True,
                    "sigma_columns": ["missing_sigma"],
                    "emg_columns": ["missing_emg"],
                    "spindle_columns": ["missing_spindle"],
                }
            }
        }
    }

    stage, meta = derive_pseudo_stage_array(
        df,
        config=config,
        dataset_id="physionet_icare_2_1",
        stage_codebook={"N2": 2},
    )

    assert stage is None
    assert meta.get("status") == "no_proxy_columns_found"
