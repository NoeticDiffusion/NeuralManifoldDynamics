"""fMRI nuisance-regression status (P0.5).

When ``nuisance_regression.enabled`` is true, ingest fail-closes unless the
status is ``applied``. Length mismatch is flagged and does not silently
truncate BOLD. Remaining NaNs after edge fill are not zero-imputed.
The confound column set and nilearn ``clean`` call are unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NUISANCE_APPLIED = "applied"
NUISANCE_SKIPPED_MISSING_FILE = "skipped_missing_file"
NUISANCE_SKIPPED_NO_COLUMNS = "skipped_no_columns"
# Length mismatch; BOLD is left untruncated (enum name is the P0.5 contract token).
NUISANCE_TRUNCATED_LENGTH = "truncated_length"
NUISANCE_FAILED_CLEAN = "failed_clean"
NUISANCE_DISABLED = "disabled"

NUISANCE_STATUS_VALUES: frozenset[str] = frozenset(
    {
        NUISANCE_APPLIED,
        NUISANCE_SKIPPED_MISSING_FILE,
        NUISANCE_SKIPPED_NO_COLUMNS,
        NUISANCE_TRUNCATED_LENGTH,
        NUISANCE_FAILED_CLEAN,
        NUISANCE_DISABLED,
    }
)

DEFAULT_NUISANCE_COLUMNS: tuple[str, ...] = (
    "trans_x",
    "trans_y",
    "trans_z",
    "rot_x",
    "rot_y",
    "rot_z",
    "wm_mean",
    "csf_mean",
    "wm_pc1",
    "wm_pc2",
    "wm_pc3",
    "csf_pc1",
    "csf_pc2",
    "csf_pc3",
)


class FmriNuisanceNotTestable(ValueError):
    """Raised when enabled nuisance regression did not apply cleanly."""


@dataclass
class NuisanceResult:
    """Outcome of one nuisance-regression attempt."""

    region_array: np.ndarray
    status: str
    confounds_path: Optional[str] = None
    columns_used: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    n_times_bold: Optional[int] = None
    n_times_confounds: Optional[int] = None
    message: str = ""

    def to_meta(self) -> dict[str, Any]:
        return {
            "nuisance_status": self.status,
            "nuisance_confounds_path": self.confounds_path,
            "nuisance_columns_used": list(self.columns_used),
            "nuisance_missing_columns": list(self.missing_columns),
            "nuisance_n_times_bold": self.n_times_bold,
            "nuisance_n_times_confounds": self.n_times_confounds,
            "nuisance_message": self.message or None,
        }


def confounds_path_for_bold(bold_path: Path | str, suffix: str) -> Path:
    """Resolve the extended-confounds TSV beside a BOLD file."""
    path = Path(bold_path)
    bold_name = path.name
    bold_base = bold_name.replace(".nii.gz", "").replace(".nii", "")
    conf_name = f"{bold_base[:-5]}{suffix}" if bold_base.endswith("_bold") else f"{bold_base}{suffix}"
    return path.parent / conf_name


def fmri_nuisance_attrs_from_frame(sub_frame: Any) -> dict[str, Any]:
    """Copy a single valid nuisance status from a feature table onto HDF5 attrs.

    Mixed or unknown statuses are omitted rather than stamping a first-row lie.
    """
    if sub_frame is None or getattr(sub_frame, "empty", True):
        return {}
    if "fmri_nuisance_status" not in getattr(sub_frame, "columns", ()):
        return {}
    statuses = [
        str(v).strip()
        for v in sub_frame["fmri_nuisance_status"].tolist()
        if v is not None and str(v).strip() and str(v).strip().lower() not in {"nan", "none"}
    ]
    known = [s for s in statuses if s in NUISANCE_STATUS_VALUES]
    unique = list(dict.fromkeys(known))
    if len(unique) != 1:
        return {}
    return {"nuisance_status": unique[0]}


def _fill_leading_trailing_nans(values: np.ndarray) -> np.ndarray:
    """Fill only leading/trailing NaNs per column. Internal holes stay NaN."""
    filled = np.array(values, dtype=float, copy=True)
    if filled.ndim != 2:
        return filled
    n_rows = filled.shape[0]
    for j in range(filled.shape[1]):
        col = filled[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            continue
        first = int(np.argmax(finite))
        last = int(n_rows - 1 - np.argmax(finite[::-1]))
        if first > 0:
            col[:first] = col[first]
        if last < n_rows - 1:
            col[last + 1 :] = col[last]
        filled[:, j] = col
    return filled


def _disabled_result(region_array: np.ndarray) -> NuisanceResult:
    return NuisanceResult(
        region_array=np.asarray(region_array),
        status=NUISANCE_DISABLED,
        message="nuisance_regression.enabled is false or absent",
    )


def apply_fmri_nuisance_regression(
    region_array: np.ndarray,
    bold_path: Path | str,
    nuisance_cfg: Mapping[str, Any],
    tr: float,
    *,
    clean_fn: Optional[Callable[..., np.ndarray]] = None,
) -> NuisanceResult:
    """Regress motion + WM/CSF confounds, or return an explicit non-applied status.

    Does not raise. Callers that set ``enabled: true`` must fail closed when
    ``status != applied``.
    """
    array = np.asarray(region_array)
    n_times = int(array.shape[1]) if array.ndim == 2 else 0
    suffix = str(nuisance_cfg.get("confounds_suffix", "_desc-confoundsextended_timeseries.tsv"))
    columns = list(nuisance_cfg.get("columns") or DEFAULT_NUISANCE_COLUMNS)
    conf_path = confounds_path_for_bold(bold_path, suffix)
    bold_name = Path(bold_path).name

    if not conf_path.exists():
        logger.warning(
            "Nuisance regression enabled but confounds file not found at %s; skipping for %s",
            conf_path,
            bold_name,
        )
        return NuisanceResult(
            region_array=array,
            status=NUISANCE_SKIPPED_MISSING_FILE,
            confounds_path=str(conf_path),
            n_times_bold=n_times,
            message=f"confounds file not found: {conf_path}",
        )

    try:
        conf_df = pd.read_csv(conf_path, sep="\t")
    except Exception as exc:
        logger.warning("Failed to read nuisance confounds %s: %s; skipping", conf_path, exc)
        return NuisanceResult(
            region_array=array,
            status=NUISANCE_FAILED_CLEAN,
            confounds_path=str(conf_path),
            n_times_bold=n_times,
            message=f"unreadable confounds TSV: {exc}",
        )

    use_cols = [c for c in columns if c in conf_df.columns]
    missing = [c for c in columns if c not in conf_df.columns]
    if missing:
        logger.warning("Nuisance confounds %s missing columns %s; using available subset", conf_path, missing)
    if not use_cols:
        logger.warning("No usable nuisance confound columns in %s; skipping regression", conf_path)
        return NuisanceResult(
            region_array=array,
            status=NUISANCE_SKIPPED_NO_COLUMNS,
            confounds_path=str(conf_path),
            missing_columns=missing,
            n_times_bold=n_times,
            n_times_confounds=int(len(conf_df)),
            message="no configured confound columns present",
        )

    try:
        conf_values = conf_df[use_cols].to_numpy(dtype=float)
    except Exception as exc:
        logger.warning("Nuisance confounds %s are not numeric: %s; not applying", conf_path, exc)
        return NuisanceResult(
            region_array=array,
            status=NUISANCE_FAILED_CLEAN,
            confounds_path=str(conf_path),
            columns_used=use_cols,
            missing_columns=missing,
            n_times_bold=n_times,
            n_times_confounds=int(len(conf_df)),
            message=f"non-numeric confound values: {exc}",
        )
    n_conf = int(conf_values.shape[0])
    if n_conf != n_times:
        logger.warning(
            "Nuisance confounds length %d != n_times %d for %s; not truncating (P0.5)",
            n_conf,
            n_times,
            bold_name,
        )
        return NuisanceResult(
            region_array=array,
            status=NUISANCE_TRUNCATED_LENGTH,
            confounds_path=str(conf_path),
            columns_used=use_cols,
            missing_columns=missing,
            n_times_bold=n_times,
            n_times_confounds=n_conf,
            message="confounds length does not match BOLD n_times",
        )

    # Leading/trailing NaNs (motion / PCA edges) may be filled; internal holes
    # and remaining NaNs are not interpolated or zero-imputed.
    conf_values = _fill_leading_trailing_nans(conf_values)
    if not np.isfinite(conf_values).all():
        logger.warning(
            "Nuisance confounds %s still contain non-finite values after edge fill; not zero-imputing",
            conf_path,
        )
        return NuisanceResult(
            region_array=array,
            status=NUISANCE_FAILED_CLEAN,
            confounds_path=str(conf_path),
            columns_used=use_cols,
            missing_columns=missing,
            n_times_bold=n_times,
            n_times_confounds=n_conf,
            message="non-finite confound values remain after edge fill",
        )

    try:
        if clean_fn is None:
            from nilearn.signal import clean as nilearn_clean

            clean_fn = nilearn_clean
        cleaned = clean_fn(
            array.T,
            confounds=conf_values,
            detrend=True,
            standardize=False,
            t_r=tr if tr and np.isfinite(tr) and tr > 0 else None,
        )
    except Exception as exc:
        logger.warning("nilearn.signal.clean failed for %s: %s; skipping nuisance regression", bold_name, exc)
        return NuisanceResult(
            region_array=array,
            status=NUISANCE_FAILED_CLEAN,
            confounds_path=str(conf_path),
            columns_used=use_cols,
            missing_columns=missing,
            n_times_bold=n_times,
            n_times_confounds=n_conf,
            message=f"nilearn.signal.clean failed: {exc}",
        )

    cleaned_arr = np.asarray(cleaned).T.astype(np.float32)
    logger.info(
        "Applied nuisance regression to %s using columns %s (n_times=%d)",
        bold_name,
        use_cols,
        cleaned_arr.shape[1],
    )
    return NuisanceResult(
        region_array=cleaned_arr,
        status=NUISANCE_APPLIED,
        confounds_path=str(conf_path),
        columns_used=use_cols,
        missing_columns=missing,
        n_times_bold=n_times,
        n_times_confounds=n_conf,
    )


def resolve_fmri_nuisance(
    region_array: np.ndarray,
    bold_path: Path | str,
    fmri_cfg: Mapping[str, Any] | None,
    tr: float,
    *,
    clean_fn: Optional[Callable[..., np.ndarray]] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply nuisance regression when enabled; fail closed if it does not apply."""
    cfg = fmri_cfg.get("nuisance_regression") if isinstance(fmri_cfg, Mapping) else None
    enabled = isinstance(cfg, Mapping) and bool(cfg.get("enabled", False))
    if not enabled:
        result = _disabled_result(region_array)
        return result.region_array, result.to_meta()

    result = apply_fmri_nuisance_regression(
        region_array,
        bold_path,
        cfg,
        tr,
        clean_fn=clean_fn,
    )
    if result.status != NUISANCE_APPLIED:
        raise FmriNuisanceNotTestable(
            f"NOT_TESTABLE: nuisance_regression.enabled is true but status="
            f"{result.status} for {Path(bold_path).name}"
            + (f" ({result.message})" if result.message else "")
        )
    return result.region_array, result.to_meta()
