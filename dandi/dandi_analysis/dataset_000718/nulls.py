from __future__ import annotations

import numpy as np

from dandi_analysis.contracts import ActivityMatrix


def circular_time_shift(
    matrix: ActivityMatrix,
    shift: int,
) -> ActivityMatrix:
    """Return a new ActivityMatrix with each unit trace shifted by *shift* bins.

    The shift is circular (wrap-around), preserving the marginal distribution
    of each unit while disrupting temporal co-structure.
    """
    data = np.roll(np.array(matrix.data, dtype=float), shift, axis=0)
    return ActivityMatrix(
        session_id=matrix.session_id,
        data=data,
        unit_ids=matrix.unit_ids,
        timestamps=np.array(matrix.timestamps),
        sampling_rate=matrix.sampling_rate,
        window_label=matrix.window_label,
        metadata={**matrix.metadata, "null": "circular_shift", "shift": shift},
    )


def unit_label_permutation(
    matrix: ActivityMatrix,
    rng: np.random.Generator,
) -> ActivityMatrix:
    """Return a new ActivityMatrix with unit columns randomly permuted.

    This destroys the pairing between unit identities and their traces while
    preserving all single-unit statistics.
    """
    data = np.array(matrix.data, dtype=float)
    perm = rng.permutation(data.shape[1])
    data_perm = data[:, perm]
    unit_ids_perm = tuple(matrix.unit_ids[i] for i in perm)
    return ActivityMatrix(
        session_id=matrix.session_id,
        data=data_perm,
        unit_ids=unit_ids_perm,
        timestamps=np.array(matrix.timestamps),
        sampling_rate=matrix.sampling_rate,
        window_label=matrix.window_label,
        metadata={**matrix.metadata, "null": "unit_label_permutation"},
    )


def matched_count_shuffle(
    matrix: ActivityMatrix,
    rng: np.random.Generator,
) -> ActivityMatrix:
    """Return a new ActivityMatrix with spike counts shuffled within each unit.

    For each unit, the non-zero bins are randomly redistributed across the same
    bins, preserving the total count and temporal envelope but destroying fine
    temporal co-activity.
    """
    data = np.array(matrix.data, dtype=float)
    shuffled = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j].copy()
        idx = rng.permutation(len(col))
        shuffled[:, j] = col[idx]
    return ActivityMatrix(
        session_id=matrix.session_id,
        data=shuffled,
        unit_ids=matrix.unit_ids,
        timestamps=np.array(matrix.timestamps),
        sampling_rate=matrix.sampling_rate,
        window_label=matrix.window_label,
        metadata={**matrix.metadata, "null": "matched_count_shuffle"},
    )
