"""Resolve fMRI repetition time from BIDS JSON, NIfTI zooms, or config fallback.

P0.3: do not silently guess TR=1.0. Order is BIDS ``RepetitionTime`` beside
the BOLD file, then NIfTI ``zooms[3]``, then ``preprocess.fallback_tr``.
If JSON and zooms both exist and disagree by more than ``TR_MISMATCH_ATOL_SEC``,
resolution fails closed (``NOT_TESTABLE``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

TR_SOURCE_BIDS_JSON = "bids_json"
TR_SOURCE_NIFTI_ZOOMS = "nifti_zooms"
TR_SOURCE_CONFIG_FALLBACK = "config_fallback"

TR_MISMATCH_ATOL_SEC = 1e-3


class FmriTRNotTestable(ValueError):
    """Raised when TR is missing, unreadable, or sources disagree."""


@dataclass(frozen=True)
class FmriTRResolution:
    """Resolved TR with explicit source and observed candidates."""

    tr_sec: float
    sfreq: float
    tr_source: str
    bids_tr_sec: Optional[float] = None
    nifti_tr_sec: Optional[float] = None
    fallback_tr_sec: Optional[float] = None
    sidecar_path: Optional[str] = None


def bold_json_sidecar_path(bold_path: Path | str) -> Path:
    """Return the BIDS JSON sidecar path beside a BOLD NIfTI file."""
    path = Path(bold_path)
    name = path.name
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        stem = name[:-7]
    elif lower.endswith(".nii"):
        stem = name[:-4]
    else:
        stem = path.stem
    return path.with_name(stem + ".json")


def _finite_positive(value: Any) -> Optional[float]:
    try:
        tr = float(value)
    except (TypeError, ValueError):
        return None
    if tr != tr or tr <= 0.0:  # NaN or non-positive
        return None
    if tr == float("inf"):
        return None
    return tr


def _read_bids_repetition_time(sidecar: Path) -> Optional[float]:
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FmriTRNotTestable(
            f"NOT_TESTABLE: BIDS sidecar unreadable at {sidecar}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise FmriTRNotTestable(
            f"NOT_TESTABLE: BIDS sidecar is not valid JSON at {sidecar}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise FmriTRNotTestable(
            f"NOT_TESTABLE: BIDS sidecar is not a JSON object at {sidecar}"
        )
    if "RepetitionTime" not in payload:
        return None
    raw = payload.get("RepetitionTime")
    if raw is None:
        # Derivative sidecars may include the key with a null placeholder
        # (ds007216 lightweight_norm). That is not a TR source; fall through.
        return None
    tr = _finite_positive(raw)
    if tr is None:
        raise FmriTRNotTestable(
            f"NOT_TESTABLE: BIDS sidecar RepetitionTime is missing or not a "
            f"positive finite number at {sidecar}"
        )
    return tr


def _tr_from_nifti_zooms(zooms: Optional[Sequence[Any]]) -> Optional[float]:
    if zooms is None:
        return None
    if len(zooms) <= 3:
        return None
    return _finite_positive(zooms[3])


def _fallback_tr_from_config(config: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not isinstance(config, Mapping):
        return None
    preprocess = config.get("preprocess")
    if not isinstance(preprocess, Mapping):
        return None
    return _finite_positive(preprocess.get("fallback_tr"))


def resolve_fmri_tr(
    bold_path: Path | str,
    *,
    config: Optional[Mapping[str, Any]] = None,
    nifti_zooms: Optional[Sequence[Any]] = None,
    mismatch_atol_sec: float = TR_MISMATCH_ATOL_SEC,
) -> FmriTRResolution:
    """Resolve TR in seconds and 1/TR sampling frequency.

    Args:
        bold_path: BOLD NIfTI path; sidecar is ``<stem>.json`` beside it.
        config: Ingest config; ``preprocess.fallback_tr`` is last resort.
        nifti_zooms: Header zooms from an already-loaded image. When omitted,
            zooms are treated as unavailable (callers that have the image
            should pass them).
        mismatch_atol_sec: Absolute tolerance in seconds for JSON vs zooms.

    Returns:
        :class:`FmriTRResolution` with ``tr_source`` in
        ``{bids_json, nifti_zooms, config_fallback}``.

    Raises:
        FmriTRNotTestable: JSON/zooms disagree, sidecar is corrupt, or no
            positive TR can be obtained from any source.
    """
    path = Path(bold_path)
    sidecar = bold_json_sidecar_path(path)
    sidecar_s = str(sidecar) if sidecar.exists() else None
    bids_tr = _read_bids_repetition_time(sidecar) if sidecar.exists() else None
    nifti_tr = _tr_from_nifti_zooms(nifti_zooms)
    fallback_tr = _fallback_tr_from_config(config)

    if bids_tr is not None and nifti_tr is not None:
        if abs(bids_tr - nifti_tr) > float(mismatch_atol_sec):
            raise FmriTRNotTestable(
                f"NOT_TESTABLE: BIDS JSON RepetitionTime={bids_tr}s disagrees "
                f"with NIfTI zooms[3]={nifti_tr}s for {path.name} "
                f"(atol={mismatch_atol_sec}s)"
            )

    if bids_tr is not None:
        source = TR_SOURCE_BIDS_JSON
        tr = bids_tr
    elif nifti_tr is not None:
        source = TR_SOURCE_NIFTI_ZOOMS
        tr = nifti_tr
    elif fallback_tr is not None:
        source = TR_SOURCE_CONFIG_FALLBACK
        tr = fallback_tr
    else:
        raise FmriTRNotTestable(
            f"NOT_TESTABLE: no positive TR from BIDS JSON, NIfTI zooms[3], or "
            f"preprocess.fallback_tr for {path.name}"
        )

    return FmriTRResolution(
        tr_sec=float(tr),
        sfreq=float(1.0 / tr),
        tr_source=source,
        bids_tr_sec=bids_tr,
        nifti_tr_sec=nifti_tr,
        fallback_tr_sec=fallback_tr,
        sidecar_path=sidecar_s,
    )


def resolution_to_meta(resolution: FmriTRResolution) -> dict[str, Any]:
    """Return preprocess/QC metadata keys for a resolved TR."""
    return {
        "tr_sec": float(resolution.tr_sec),
        "sfreq": float(resolution.sfreq),
        "tr_source": str(resolution.tr_source),
        "tr_bids_json_sec": resolution.bids_tr_sec,
        "tr_nifti_zooms_sec": resolution.nifti_tr_sec,
        "tr_fallback_sec": resolution.fallback_tr_sec,
        "tr_sidecar_path": resolution.sidecar_path,
    }


def fmri_tr_attrs_from_frame(sub_frame: Any) -> dict[str, Any]:
    """Copy resolved TR fields from a feature table onto HDF5 root attrs.

    Returns an empty dict when the frame has no fMRI TR provenance, so EEG
    exports are unchanged.
    """
    if sub_frame is None or getattr(sub_frame, "empty", True):
        return {}
    attrs: dict[str, Any] = {}
    if "fmri_tr_sec" in sub_frame.columns:
        tr_vals = _series_finite_positive(sub_frame["fmri_tr_sec"])
        if tr_vals:
            attrs["tr_sec"] = tr_vals[0]
            attrs["sfreq"] = float(1.0 / tr_vals[0])
    if "fmri_sfreq" in sub_frame.columns and "sfreq" not in attrs:
        sf_vals = _series_finite_positive(sub_frame["fmri_sfreq"])
        if sf_vals:
            attrs["sfreq"] = sf_vals[0]
            attrs.setdefault("tr_sec", float(1.0 / sf_vals[0]))
    if "fmri_tr_source" in sub_frame.columns:
        sources = [
            str(v).strip()
            for v in sub_frame["fmri_tr_source"].tolist()
            if v is not None and str(v).strip() and str(v).strip().lower() not in {"nan", "none"}
        ]
        if sources:
            attrs["tr_source"] = sources[0]
    return attrs


def _series_finite_positive(values: Any) -> list[float]:
    out: list[float] = []
    for raw in list(values):
        tr = _finite_positive(raw)
        if tr is not None:
            out.append(tr)
    return out
