from pathlib import Path

import numpy as np
import pandas as pd
import pytest

h5py = pytest.importorskip("h5py")


def _write_feature_h5(path: Path, subject: str, group: str, alpha_values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5:
        h5.attrs["subject_id"] = subject
        h5.attrs["group"] = group
        grp = h5.require_group("features_raw")
        grp.create_dataset("values", data=np.asarray(alpha_values, dtype=np.float32).reshape(-1, 1))
        grp.create_dataset("names", data=np.asarray(["eeg_alpha"], dtype=h5py.string_dtype(encoding="utf-8")))


def test_fit_feature_anchors_from_h5_is_subject_balanced(tmp_path: Path):
    """Long recordings should not dominate the fitted center."""
    from mndm.anchors import anchor_mapping, fit_feature_anchors_from_h5

    a = tmp_path / "sub-001.h5"
    b = tmp_path / "sub-002.h5"
    _write_feature_h5(a, "sub-001", "Control", [10.0] * 20)
    _write_feature_h5(b, "sub-002", "Patient", [1000.0] * 2)

    artifact = fit_feature_anchors_from_h5(
        [a, b],
        anchor_id="unit-test",
        feature_standardization={"eeg_alpha": ["log10", "robust_z", "clip"]},
        min_subjects=2,
    )
    amap = anchor_mapping(artifact, scale_method="iqr", min_subjects=2)

    assert artifact["spec"]["schema_version"] == "mndm.feature_anchors.v2.1"
    assert artifact["spec"]["subject_balanced"] is True
    assert "anchor_hash" in artifact["spec"]
    assert amap["eeg_alpha"]["center"] == pytest.approx(2.0)
    assert amap["eeg_alpha"]["scale"] > 0


def test_fit_feature_anchors_from_features_df_matches_h5(tmp_path: Path):
    """DataFrame and H5 anchor fitting should agree on the same feature surface."""
    from mndm.anchors import (
        fit_feature_anchors_from_features_df,
        fit_feature_anchors_from_h5,
    )

    a1 = tmp_path / "sub-001_run-1.h5"
    a2 = tmp_path / "sub-001_run-2.h5"
    b1 = tmp_path / "sub-002_run-1.h5"
    _write_feature_h5(a1, "sub-001", "Control", [10.0] * 5)
    _write_feature_h5(a2, "sub-001", "Control", [20.0] * 2)
    _write_feature_h5(b1, "sub-002", "Patient", [1000.0] * 3)

    artifact_h5 = fit_feature_anchors_from_h5(
        [a1, a2, b1],
        anchor_id="unit-test",
        feature_standardization={"eeg_alpha": ["log10", "robust_z", "clip"]},
        min_subjects=2,
    )

    features_df = pd.DataFrame(
        {
            "file": ["sub-001_run-1.edf"] * 5 + ["sub-001_run-2.edf"] * 2 + ["sub-002_run-1.edf"] * 3,
            "subject": ["sub-001"] * 7 + ["sub-002"] * 3,
            "group": ["Control"] * 7 + ["Patient"] * 3,
            "eeg_alpha": [10.0] * 5 + [20.0] * 2 + [1000.0] * 3,
        }
    )
    artifact_df = fit_feature_anchors_from_features_df(
        features_df,
        anchor_id="unit-test",
        feature_standardization={"eeg_alpha": ["log10", "robust_z", "clip"]},
        min_subjects=2,
    )

    assert artifact_df["spec"]["anchor_hash"] == artifact_h5["spec"]["anchor_hash"]
    assert artifact_df["spec"]["groups"] == artifact_h5["spec"]["groups"]
    assert artifact_df["features"] == artifact_h5["features"]
