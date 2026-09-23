"""P0.6 atlas vs BOLD affine/orientation: same shape is not the same space."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

pytest.importorskip("nibabel")

from mndm.pipeline.spatial_align import (
    AFFINE_ATOL_MM,
    SPACE_MATCHED,
    SPACE_RESAMPLED,
    FmriAtlasSpaceNotTestable,
    affines_close,
    fmri_atlas_space_attrs_from_frame,
    orientation_code,
    require_fmri_atlas_space_attrs,
    resolve_atlas_space,
)

SHAPE = (4, 5, 6)
RAS = np.diag([2.0, 2.0, 2.0, 1.0])


def _shifted(affine: np.ndarray, dx_mm: float) -> np.ndarray:
    out = np.array(affine, dtype=float, copy=True)
    out[0, 3] += dx_mm
    return out


def _flipped_x(affine: np.ndarray) -> np.ndarray:
    out = np.array(affine, dtype=float, copy=True)
    out[0, 0] *= -1.0
    return out


def test_identical_grid_is_matched():
    decision = resolve_atlas_space(RAS, RAS, SHAPE, SHAPE, {})
    assert decision.status == SPACE_MATCHED
    assert decision.should_resample is False
    assert decision.affine_match is True
    assert decision.ornt_match is True
    assert orientation_code(RAS) == "RAS"


def test_same_shape_different_affine_is_not_testable():
    bold = _shifted(RAS, 10.0)
    with pytest.raises(FmriAtlasSpaceNotTestable, match="NOT_TESTABLE"):
        resolve_atlas_space(RAS, bold, SHAPE, SHAPE, {})


def test_same_shape_different_orientation_is_not_testable():
    bold = _flipped_x(RAS)
    assert orientation_code(RAS) != orientation_code(bold)
    with pytest.raises(FmriAtlasSpaceNotTestable, match="orientation"):
        resolve_atlas_space(RAS, bold, SHAPE, SHAPE, {})


def test_resample_alone_does_not_skip_same_shape_affine_mismatch():
    bold = _shifted(RAS, 10.0)
    with pytest.raises(FmriAtlasSpaceNotTestable, match="assume_same_space"):
        resolve_atlas_space(
            RAS, bold, SHAPE, SHAPE, {"resample_atlas_to_bold": True}
        )


def test_assume_same_space_alone_does_not_skip_affine_mismatch():
    bold = _shifted(RAS, 10.0)
    with pytest.raises(FmriAtlasSpaceNotTestable, match="NOT_TESTABLE"):
        resolve_atlas_space(RAS, bold, SHAPE, SHAPE, {"assume_same_space": True})


def test_both_flags_allow_resample_when_shape_matches_but_affine_differs():
    bold = _shifted(RAS, 10.0)
    decision = resolve_atlas_space(
        RAS,
        bold,
        SHAPE,
        SHAPE,
        {"resample_atlas_to_bold": True, "assume_same_space": True},
    )
    assert decision.status == SPACE_RESAMPLED
    assert decision.should_resample is True
    assert decision.affine_match is False


def test_shape_mismatch_without_resample_keeps_valueerror():
    with pytest.raises(ValueError, match="does not match BOLD spatial shape"):
        resolve_atlas_space(RAS, RAS, SHAPE, (8, 8, 8), {})


def test_shape_mismatch_with_resample_is_resampled():
    decision = resolve_atlas_space(
        RAS, RAS, SHAPE, (8, 8, 8), {"resample_atlas_to_bold": True}
    )
    assert decision.status == SPACE_RESAMPLED
    assert decision.should_resample is True
    assert decision.shape_match is False


def test_within_atol_translation_is_matched():
    bold = _shifted(RAS, AFFINE_ATOL_MM * 0.5)
    assert affines_close(RAS, bold)
    decision = resolve_atlas_space(RAS, bold, SHAPE, SHAPE, {})
    assert decision.status == SPACE_MATCHED


def test_custom_atol_is_serialized_on_decision():
    bold = _shifted(RAS, 0.05)
    decision = resolve_atlas_space(RAS, bold, SHAPE, SHAPE, {}, atol=0.1)
    assert decision.status == SPACE_MATCHED
    assert decision.to_meta()["atlas_affine_atol_mm"] == pytest.approx(0.1)


def test_fmri_atlas_space_attrs_from_frame_and_eeg_empty():
    assert fmri_atlas_space_attrs_from_frame(pd.DataFrame({"feat_a": [1.0]})) == {}
    frame = pd.DataFrame(
        {
            "fmri_atlas_space_status": ["matched", "matched"],
            "fmri_atlas_ornt": ["RAS", "RAS"],
            "fmri_bold_ornt": ["RAS", "RAS"],
            "fmri_atlas_affine_match": [1, 1],
        }
    )
    attrs = fmri_atlas_space_attrs_from_frame(frame)
    assert attrs["atlas_space_status"] == "matched"
    assert attrs["atlas_ornt"] == "RAS"
    assert attrs["bold_ornt"] == "RAS"
    assert attrs["atlas_affine_match"] == 1
    mixed = pd.DataFrame(
        {
            "fmri_atlas_space_status": ["matched", "resampled"],
            "fmri_atlas_ornt": ["RAS", "RAS"],
        }
    )
    assert fmri_atlas_space_attrs_from_frame(mixed) == {}
    bogus = pd.DataFrame(
        {"fmri_atlas_space_status": ["matched", "bogus"]}
    )
    assert fmri_atlas_space_attrs_from_frame(bogus) == {}
    missing_mix = pd.DataFrame({"fmri_atlas_space_status": ["matched", None]})
    assert fmri_atlas_space_attrs_from_frame(missing_mix) == {}
    nan_mix = pd.DataFrame({"fmri_atlas_space_status": ["matched", np.nan]})
    assert fmri_atlas_space_attrs_from_frame(nan_mix) == {}
    with pytest.raises(FmriAtlasSpaceNotTestable, match="atlas_space_status"):
        require_fmri_atlas_space_attrs(missing_mix)
    with pytest.raises(FmriAtlasSpaceNotTestable, match="atlas_space_status"):
        require_fmri_atlas_space_attrs(pd.DataFrame({"feat_a": [1.0]}))
    required = require_fmri_atlas_space_attrs(frame)
    assert required["atlas_space_status"] == "matched"
    assert required["atlas_affine_atol_mm"] == pytest.approx(AFFINE_ATOL_MM)


def test_degenerate_affine_is_not_testable():
    zero = np.zeros((4, 4), dtype=float)
    zero[3, 3] = 1.0
    with pytest.raises(FmriAtlasSpaceNotTestable, match="degenerate"):
        resolve_atlas_space(zero, RAS, SHAPE, SHAPE, {})
