"""Actual fit membership, including missing rows and external calibration."""
import hashlib
import numpy as np
import pandas as pd

from mndm.projection import _normalize_used_columns


def test_local_fit_hash_identifies_consumed_rows_and_values_unchanged():
    frame = pd.DataFrame({"file": ["a.edf"] * 4, "epoch_id": [0, 1, 2, 3],
                          "t_start": [0., 30., 60., 90.], "t_end": [30., 60., 90., 120.],
                          "feature": [1., np.nan, 3., 7.]})
    result, baselines = _normalize_used_columns(frame, ["feature"], "z", {"feature": ["z"]})
    record = baselines["feature"]
    identities = frame[["file", "epoch_id", "t_start", "t_end"]].copy()
    identities.insert(0, "input_row_position", range(4))
    payload = identities.iloc[[0, 2, 3]].to_json(orient="split", index=False, double_precision=15, force_ascii=True)
    assert record["fit_population_hash"] == hashlib.sha256(payload.encode()).hexdigest()
    assert record["fit_population_count"] == 3
    assert record["fit_input_row_count"] == 4
    selected = np.array([1., 3., 7.], dtype=np.float32)
    np.testing.assert_array_equal(result.loc[[0, 2, 3], "feature"],
                                  (selected - selected.mean()) / (selected.std() + 1e-9))
    assert bool(np.isnan(result.loc[1, "feature"]))
    changed_identity = frame.copy()
    changed_identity.loc[2, "file"] = "b.edf"
    _, other = _normalize_used_columns(changed_identity, ["feature"], "z", {"feature": ["z"]})
    assert record["fit_population_hash"] != other["feature"]["fit_population_hash"]


def test_external_anchor_does_not_claim_current_rows_as_fit_population():
    frame = pd.DataFrame({"feature": [1., 3., 7.]})
    _, baselines = _normalize_used_columns(frame, ["feature"], "z", {"feature": ["z"]},
                                          external_anchor={"feature": {"center": 2., "scale": 4., "anchor_hash": "reference"}})
    record = baselines["feature"]
    assert record["fit_population_hash"] is None
    assert record["fit_scope"] == "external_anchor_population_not_recorded_here"
    assert record["anchor_hash"] == "reference"
