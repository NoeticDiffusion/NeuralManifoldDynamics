from __future__ import annotations

from typing import Sequence

import numpy as np

from dandi_analysis.contracts import (
    ActivityMatrix,
    OfflineWindow,
    PairwiseCoreactivationResult,
)


def pairwise_coactivity_matrix(activity_matrix: ActivityMatrix) -> np.ndarray:
    """Compute symmetric, time-averaged pairwise co-activity.

    Returns an (N, N) matrix where entry (i, j) is the Pearson correlation
    between unit i and unit j across time.  Self-correlations (diagonal) are
    set to NaN.
    """
    data = np.array(activity_matrix.data, dtype=float)
    n_units = data.shape[1]
    # Use numpy corrcoef (units as rows)
    corr = np.corrcoef(data.T)  # (N, N)
    np.fill_diagonal(corr, np.nan)
    return corr


def offline_coreactivation_score(
    activity_matrix: ActivityMatrix,
    window: OfflineWindow,
    trace_i: str,
    trace_j: str,
    *,
    null_matrices: Sequence[ActivityMatrix] | None = None,
) -> PairwiseCoreactivationResult:
    """Compute the offline co-reactivation score for two unit groups.

    *trace_i* and *trace_j* are unit IDs present in *activity_matrix.unit_ids*.
    The score is the Pearson correlation between the two unit traces within the
    window.  If *null_matrices* are supplied the z-score is computed against
    the null distribution of the same metric.

    If the unit IDs are not found the score is NaN.
    """
    uid_list = list(activity_matrix.unit_ids)
    data = np.array(activity_matrix.data, dtype=float)

    def _score_for(mat: np.ndarray) -> float:
        try:
            i_idx = uid_list.index(trace_i)
            j_idx = uid_list.index(trace_j)
            a = mat[:, i_idx]
            b = mat[:, j_idx]
            if a.std() == 0 or b.std() == 0:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])
        except (ValueError, IndexError):
            return float("nan")

    score = _score_for(data)

    null_scores: list[float] = []
    if null_matrices:
        for nm in null_matrices:
            null_scores.append(_score_for(np.array(nm.data, dtype=float)))

    if null_scores:
        null_arr = np.array(null_scores, dtype=float)
        null_mean = float(np.nanmean(null_arr))
        null_std = float(np.nanstd(null_arr))
        z = (score - null_mean) / null_std if null_std > 0 else float("nan")
    else:
        null_mean = float("nan")
        null_std = float("nan")
        z = float("nan")

    return PairwiseCoreactivationResult(
        session_id=activity_matrix.session_id,
        trace_i=trace_i,
        trace_j=trace_j,
        co_reactivation_score=score,
        null_mean=null_mean,
        null_std=null_std,
        z_score=z,
        window_label=window.label,
    )
