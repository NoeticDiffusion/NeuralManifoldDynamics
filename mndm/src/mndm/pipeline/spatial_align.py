"""fMRI atlas vs BOLD spatial alignment (P0.6).

Same voxel shape is not the same space. Affine and orientation must match
unless the caller both resamples onto the BOLD grid and explicitly sets
``assume_same_space: true``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

AFFINE_ATOL_MM = 1e-3
AFFINE_DET_MIN = 1e-6

SPACE_MATCHED = "matched"
SPACE_RESAMPLED = "resampled"

SPACE_STATUS_VALUES: frozenset[str] = frozenset({SPACE_MATCHED, SPACE_RESAMPLED})


class FmriAtlasSpaceNotTestable(ValueError):
    """Raised when atlas and BOLD are not in a declared common space."""


@dataclass(frozen=True)
class AtlasSpaceDecision:
    """Whether parcellation may proceed, and whether atlas must be resampled."""

    status: str
    should_resample: bool
    shape_match: bool
    affine_match: bool
    ornt_match: bool
    atlas_ornt: str
    bold_ornt: str
    resample_requested: bool
    assume_same_space: bool
    message: str = ""
    atol_mm: float = AFFINE_ATOL_MM

    def to_meta(self) -> dict[str, Any]:
        return {
            "atlas_space_status": self.status,
            "atlas_ornt": self.atlas_ornt,
            "bold_ornt": self.bold_ornt,
            "atlas_affine_match": int(bool(self.affine_match)),
            "atlas_ornt_match": int(bool(self.ornt_match)),
            "resample_atlas_to_bold": int(bool(self.resample_requested)),
            "assume_same_space": int(bool(self.assume_same_space)),
            "atlas_affine_atol_mm": float(self.atol_mm),
        }


def _as_affine_4x4(affine: Any) -> np.ndarray:
    arr = np.asarray(affine, dtype=float)
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=float)
        out[:3, :] = arr
        arr = out
    if arr.shape != (4, 4):
        raise FmriAtlasSpaceNotTestable(
            f"NOT_TESTABLE: affine must be 4x4 or 3x4, got shape {getattr(affine, 'shape', None)}"
        )
    if not np.isfinite(arr).all():
        raise FmriAtlasSpaceNotTestable("NOT_TESTABLE: affine contains non-finite values")
    det = float(np.linalg.det(arr[:3, :3]))
    if not np.isfinite(det) or abs(det) < AFFINE_DET_MIN:
        raise FmriAtlasSpaceNotTestable(
            f"NOT_TESTABLE: affine 3x3 is degenerate (det={det})"
        )
    return arr


def orientation_code(affine: Any) -> str:
    """Return a compact orientation string such as ``RAS``."""
    from nibabel.orientations import aff2axcodes

    codes = aff2axcodes(_as_affine_4x4(affine))
    return "".join(str(c) for c in codes)


def affines_close(atlas_affine: Any, bold_affine: Any, *, atol: float = AFFINE_ATOL_MM) -> bool:
    """True when atlas and BOLD affines agree within ``atol`` millimetres."""
    return bool(
        np.allclose(
            _as_affine_4x4(atlas_affine),
            _as_affine_4x4(bold_affine),
            atol=float(atol),
            rtol=0.0,
        )
    )


def spaces_match(atlas_affine: Any, bold_affine: Any, *, atol: float = AFFINE_ATOL_MM) -> bool:
    """True when affine and orientation codes both match."""
    return affines_close(atlas_affine, bold_affine, atol=atol) and (
        orientation_code(atlas_affine) == orientation_code(bold_affine)
    )


def _shape3(shape: Sequence[int] | np.ndarray) -> tuple[int, int, int]:
    values = tuple(int(v) for v in list(shape)[:3])
    if len(values) != 3:
        raise FmriAtlasSpaceNotTestable(
            f"NOT_TESTABLE: expected a 3-D spatial shape, got {shape}"
        )
    return values[0], values[1], values[2]


def resolve_atlas_space(
    atlas_affine: Any,
    bold_affine: Any,
    atlas_shape: Sequence[int] | np.ndarray,
    bold_shape: Sequence[int] | np.ndarray,
    fmri_cfg: Mapping[str, Any] | None = None,
    *,
    atol: float = AFFINE_ATOL_MM,
) -> AtlasSpaceDecision:
    """Decide whether atlas and BOLD share a grid, or must be resampled / fail.

    Same shape with a mismatched affine is ``NOT_TESTABLE`` unless both
    ``resample_atlas_to_bold`` and ``assume_same_space`` are true.
    """
    cfg = fmri_cfg if isinstance(fmri_cfg, Mapping) else {}
    resample_requested = bool(cfg.get("resample_atlas_to_bold", False))
    assume_same_space = bool(cfg.get("assume_same_space", False))
    atlas_shape3 = _shape3(atlas_shape)
    bold_shape3 = _shape3(bold_shape)
    shape_match = atlas_shape3 == bold_shape3
    atlas_ornt = orientation_code(atlas_affine)
    bold_ornt = orientation_code(bold_affine)
    affine_match = affines_close(atlas_affine, bold_affine, atol=atol)
    ornt_match = atlas_ornt == bold_ornt
    space_ok = bool(affine_match and ornt_match)

    if shape_match and space_ok:
        return AtlasSpaceDecision(
            status=SPACE_MATCHED,
            should_resample=False,
            shape_match=True,
            affine_match=True,
            ornt_match=True,
            atlas_ornt=atlas_ornt,
            bold_ornt=bold_ornt,
            resample_requested=resample_requested,
            assume_same_space=assume_same_space,
            message="atlas and BOLD share shape, affine, and orientation",
            atol_mm=float(atol),
        )

    if shape_match and not space_ok:
        if resample_requested and assume_same_space:
            return AtlasSpaceDecision(
                status=SPACE_RESAMPLED,
                should_resample=True,
                shape_match=True,
                affine_match=affine_match,
                ornt_match=ornt_match,
                atlas_ornt=atlas_ornt,
                bold_ornt=bold_ornt,
                resample_requested=True,
                assume_same_space=True,
                message="same shape but affine/orientation differ; resampling under assume_same_space",
                atol_mm=float(atol),
            )
        raise FmriAtlasSpaceNotTestable(
            "NOT_TESTABLE: atlas and BOLD share shape "
            f"{atlas_shape3} but affine/orientation differ "
            f"(atlas_ornt={atlas_ornt}, bold_ornt={bold_ornt}). "
            "Set resample_atlas_to_bold: true and assume_same_space: true "
            "only if they share anatomical space."
        )

    if resample_requested:
        return AtlasSpaceDecision(
            status=SPACE_RESAMPLED,
            should_resample=True,
            shape_match=False,
            affine_match=affine_match,
            ornt_match=ornt_match,
            atlas_ornt=atlas_ornt,
            bold_ornt=bold_ornt,
            resample_requested=True,
            assume_same_space=assume_same_space,
            message="shape mismatch; resampling atlas onto BOLD grid",
            atol_mm=float(atol),
        )

    raise ValueError(
        f"Atlas shape {atlas_shape3} does not match BOLD spatial shape {bold_shape3}"
    )


def fmri_atlas_space_attrs_from_frame(sub_frame: Any) -> dict[str, Any]:
    """Copy atlas-space provenance from a feature table onto HDF5 root attrs."""
    if sub_frame is None or getattr(sub_frame, "empty", True):
        return {}
    columns = getattr(sub_frame, "columns", ())
    if "fmri_atlas_space_status" not in columns:
        return {}

    def _strings(column: str) -> list[str]:
        if column not in columns:
            return []
        values = [
            str(v).strip()
            for v in sub_frame[column].tolist()
            if v is not None and str(v).strip() and str(v).strip().lower() not in {"nan", "none"}
        ]
        return list(dict.fromkeys(values))

    raw_status: list[str] = []
    for v in sub_frame["fmri_atlas_space_status"].tolist():
        if v is None:
            raw_status.append("")
            continue
        try:
            if v != v:
                raw_status.append("")
                continue
        except Exception:
            pass
        text = str(v).strip()
        raw_status.append("" if not text or text.lower() in {"nan", "none"} else text)
    unique_status = list(dict.fromkeys(raw_status))
    if len(unique_status) != 1 or unique_status[0] not in SPACE_STATUS_VALUES:
        return {}
    attrs: dict[str, Any] = {"atlas_space_status": unique_status[0]}
    if "fmri_atlas_affine_atol_mm" in columns:
        atols: list[float] = []
        for raw in list(sub_frame["fmri_atlas_affine_atol_mm"]):
            try:
                val = float(raw)
            except (TypeError, ValueError):
                return {}
            if val != val or val in (float("inf"), float("-inf")):
                return {}
            atols.append(val)
        unique_atol = list(dict.fromkeys(atols))
        if len(unique_atol) != 1:
            return {}
        attrs["atlas_affine_atol_mm"] = float(unique_atol[0])
    else:
        attrs["atlas_affine_atol_mm"] = float(AFFINE_ATOL_MM)
    atlas_ornt = _strings("fmri_atlas_ornt")
    bold_ornt = _strings("fmri_bold_ornt")
    if len(atlas_ornt) == 1:
        attrs["atlas_ornt"] = atlas_ornt[0]
    if len(bold_ornt) == 1:
        attrs["bold_ornt"] = bold_ornt[0]
    if "fmri_atlas_affine_match" in columns:
        matches = []
        for raw in list(sub_frame["fmri_atlas_affine_match"]):
            try:
                matches.append(int(bool(int(raw))))
            except (TypeError, ValueError):
                continue
        unique_match = list(dict.fromkeys(matches))
        if len(unique_match) == 1:
            attrs["atlas_affine_match"] = unique_match[0]
    return attrs


def require_fmri_atlas_space_attrs(sub_frame: Any) -> dict[str, Any]:
    """Fail closed: fMRI HDF5 must carry a single valid atlas-space status."""
    attrs = fmri_atlas_space_attrs_from_frame(sub_frame)
    if "atlas_space_status" not in attrs:
        raise FmriAtlasSpaceNotTestable(
            "NOT_TESTABLE: fMRI export requires a single valid atlas_space_status "
            "(matched or resampled); re-extract features after the P0.6 affine check"
        )
    return attrs
