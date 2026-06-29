from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_merge_feature_frames_coalesces_shared_time_columns_across_nested_merges():
    from mndm.parallel import _merge_feature_frames

    ecg_df = pd.DataFrame(
        {
            "epoch_id": [0, 1],
            "t_start": [0.0, 4.0],
            "t_end": [4.0, 8.0],
            "ecg_hr_bpm": [60.0, 62.0],
        }
    )
    ppg_df = pd.DataFrame(
        {
            "epoch_id": [0, 1],
            "t_start": [0.0, 4.0],
            "t_end": [4.0, 8.0],
            "ppg_rate_bpm": [59.0, 61.0],
        }
    )
    eeg_df = pd.DataFrame(
        {
            "epoch_id": [0, 1],
            "t_start": [0.0, 4.0],
            "t_end": [4.0, 8.0],
            "eeg_highfreq_power_30_45": [0.2, 0.3],
        }
    )
    pupil_df = pd.DataFrame(
        {
            "epoch_id": [0, 1],
            "t_start": [0.0, 4.0],
            "t_end": [4.0, 8.0],
            "pupil_dilation_velocity": [1.0, 1.2],
        }
    )

    ecg_ppg = _merge_feature_frames([ecg_df, ppg_df])
    merged = _merge_feature_frames([ecg_ppg, eeg_df, pupil_df])

    assert merged.columns.is_unique
    assert "t_start" in merged.columns
    assert "t_end" in merged.columns
    assert not any(str(col).startswith("t_start__dup") for col in merged.columns)
    assert not any(str(col).startswith("t_end__dup") for col in merged.columns)
    assert "embodied_arousal_proxy" in merged.columns
    assert "embodied_arousal_proxy_source" in merged.columns
    assert np.isfinite(pd.to_numeric(merged["embodied_arousal_proxy"], errors="coerce")).all()
