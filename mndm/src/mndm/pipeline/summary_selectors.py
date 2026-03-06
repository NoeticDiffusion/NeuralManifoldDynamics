"""Path/selector helpers extracted from summary.py."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd


def _match_bold_candidate(
    path_str: str,
    *,
    raw_task: Optional[str],
    condition: Optional[str],
    session: Optional[str],
    run_id: Optional[str],
    require_bold_token: bool = True,
) -> bool:
    if not path_str:
        return False
    if path_str.startswith("._") or "/._" in path_str or "\\._" in path_str:
        return False
    path_l = path_str.lower()
    if require_bold_token and "_bold" not in path_l:
        return False
    if raw_task and raw_task.lower() not in path_l:
        return False
    if condition and not session:
        if condition.lower() not in path_l:
            compound = f"{(raw_task or '').lower()}{condition.lower()}"
            if compound and compound not in path_l:
                return False
    if session and session not in path_str:
        return False
    if run_id and run_id not in path_str:
        return False
    return path_str.endswith((".nii", ".nii.gz"))


def resolve_bold_path_for_subframe(
    *,
    sub_frame: pd.DataFrame,
    raw_task: Optional[str],
    condition: Optional[str],
    session: Optional[str],
    run_id: Optional[str],
    acq_id: Optional[str],
    dataset_root: Path,
    index_df: Optional[pd.DataFrame],
    lookup_rel_paths_by_file_value: Callable[[str], List[str]],
) -> Optional[Path]:
    """Resolve BOLD NIfTI path for the current grouped sub_frame."""
    del acq_id  # retained for interface compatibility
    if "file" not in sub_frame.columns or len(sub_frame) == 0:
        return None

    if "file" in sub_frame.columns and len(sub_frame) > 0:
        file_values = sub_frame["file"].astype(str).dropna().unique().tolist()
        for file_val in file_values:
            if not _match_bold_candidate(
                file_val,
                raw_task=raw_task,
                condition=condition,
                session=session,
                run_id=run_id,
            ):
                continue
            try:
                abs_candidate = Path(file_val)
                if abs_candidate.is_absolute() and abs_candidate.exists():
                    return abs_candidate
            except Exception:
                pass

            try:
                rel_norm = file_val.replace("\\", "/")
                if "/" in rel_norm:
                    candidate = dataset_root / rel_norm
                    if candidate.exists():
                        return candidate
            except Exception:
                pass

            try:
                rel_candidates = lookup_rel_paths_by_file_value(file_val)
            except Exception:
                rel_candidates = []
            for rel in rel_candidates:
                if not _match_bold_candidate(
                    str(rel),
                    raw_task=raw_task,
                    condition=condition,
                    session=session,
                    run_id=run_id,
                ):
                    continue
                candidate = dataset_root / str(rel)
                if candidate.exists():
                    return candidate

    if index_df is not None and "path" in index_df.columns:
        try:
            fmri_rows = index_df[index_df.get("modality", "").astype(str) == "fmri"]
            if session and "session" in fmri_rows.columns:
                fmri_rows = fmri_rows[fmri_rows["session"].astype(str) == str(session).replace("ses-", "")]
            paths = fmri_rows["path"].astype(str).tolist()
            for rel in paths:
                if not _match_bold_candidate(
                    rel,
                    raw_task=raw_task,
                    condition=condition,
                    session=session,
                    run_id=run_id,
                ):
                    continue
                candidate = dataset_root / rel
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    return None


def load_regional_fmri_signals(
    *,
    sub_id: str,
    dataset_label: str,
    config: Mapping[str, Any],
    sub_frame: pd.DataFrame,
    raw_task: Optional[str],
    condition: Optional[str],
    session: Optional[str],
    run_id: Optional[str],
    dataset_root: Path,
    index_df: Optional[pd.DataFrame],
    lookup_rel_paths_by_file_value: Callable[[str], List[str]],
    preprocess_fmri: Callable[[Path, Mapping[str, Any]], Any],
    logger: Any,
) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[float]]:
    """Load regional fMRI signals for one subject/segment."""
    regions_bold = None
    regions_names = None
    regions_sfreq = None

    def _match_candidate(path_str: str) -> bool:
        return _match_bold_candidate(
            path_str,
            raw_task=raw_task,
            condition=condition,
            session=session,
            run_id=run_id,
            require_bold_token=True,
        )

    bold_path = None
    fmri_rows_subject = None
    if index_df is not None and "path" in index_df.columns:
        try:
            subj_token_raw = str(sub_id)[4:] if str(sub_id).startswith("sub-") else str(sub_id)
            subj_tokens = [subj_token_raw]
            if subj_token_raw.isdigit():
                subj_tokens.append(str(int(subj_token_raw)))
            fmri_rows_subject = index_df[index_df.get("modality", "").astype(str) == "fmri"]
            fmri_rows_subject = fmri_rows_subject[fmri_rows_subject.get("subject", "").astype(str).isin(subj_tokens)]
            fmri_rows_subject = fmri_rows_subject[
                ~fmri_rows_subject["path"].astype(str).str.contains(r"(?:^|[\\/])\._", regex=True)
            ]
        except Exception:
            fmri_rows_subject = None

    if "file" in sub_frame.columns and len(sub_frame) > 0:
        file_values = sub_frame["file"].astype(str).dropna().unique().tolist()
        for file_val in file_values:
            if not _match_candidate(file_val):
                continue
            try:
                abs_candidate = Path(file_val)
                if abs_candidate.is_absolute() and abs_candidate.exists():
                    bold_path = abs_candidate
                    break
            except Exception:
                pass

            try:
                rel_norm = file_val.replace("\\", "/")
                if "/" in rel_norm:
                    candidate = dataset_root / rel_norm
                    if candidate.exists():
                        bold_path = candidate
                        break
            except Exception:
                pass

            if bold_path is None and fmri_rows_subject is not None and not fmri_rows_subject.empty:
                try:
                    rel_candidates = lookup_rel_paths_by_file_value(file_val)
                    if rel_candidates:
                        rel_set = set(rel_candidates)
                        match_mask = fmri_rows_subject["path"].astype(str).isin(rel_set)
                        if match_mask.any():
                            bold_rel = str(fmri_rows_subject.loc[match_mask, "path"].iloc[0])
                            candidate = dataset_root / bold_rel
                            if candidate.exists():
                                bold_path = candidate
                                break
                except Exception:
                    pass

    if bold_path is None and index_df is not None:
        try:
            subj_token_raw = str(sub_id)[4:] if str(sub_id).startswith("sub-") else str(sub_id)
            subj_tokens = [subj_token_raw]
            if subj_token_raw.isdigit():
                subj_tokens.append(str(int(subj_token_raw)))
            fmri_rows = index_df[index_df.get("modality", "").astype(str) == "fmri"]
            fmri_rows = fmri_rows[fmri_rows.get("subject", "").astype(str).isin(subj_tokens)]
            fmri_rows = fmri_rows[
                ~fmri_rows["path"].astype(str).str.contains(r"(?:^|[\\/])\._", regex=True)
            ]
            if session and "session" in fmri_rows.columns:
                fmri_rows = fmri_rows[fmri_rows["session"].astype(str) == str(session).replace("ses-", "")]

            def _path_matches(path_str: str) -> bool:
                return _match_bold_candidate(
                    path_str,
                    raw_task=raw_task,
                    condition=condition,
                    session=session,
                    run_id=run_id,
                    require_bold_token=False,
                )

            if not fmri_rows.empty and (raw_task or condition):
                path_vals = fmri_rows["path"].astype(str).tolist()
                keep_mask = [_path_matches(p) for p in path_vals]
                if keep_mask:
                    fmri_rows = fmri_rows.loc[np.asarray(keep_mask, dtype=bool)]

            if not fmri_rows.empty:
                bold_rel = fmri_rows.iloc[0]["path"]
                bold_path = dataset_root / str(bold_rel)
        except Exception:
            pass

    if bold_path is not None and bold_path.exists():
        try:
            fmri_pre = preprocess_fmri(bold_path, config)
            fmri_signals = fmri_pre.signals.get("fmri")
            fmri_ch_names = None
            if isinstance(fmri_pre.channels, dict):
                fmri_ch_names = fmri_pre.channels.get("fmri")
            if fmri_signals is not None:
                regions_bold = np.asarray(fmri_signals, dtype=np.float32)
                regions_names = list(fmri_ch_names) if fmri_ch_names else None
                regions_sfreq = float(fmri_pre.sfreq)
        except Exception:
            logger.exception("Failed to load regional fMRI signals for %s", dataset_label)

    return regions_bold, regions_names, regions_sfreq
