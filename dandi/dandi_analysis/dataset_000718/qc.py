from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from dandi_analysis.contracts import ActivityMatrix, QcIssue, SessionIndexRow


_MIN_UNITS = 5
_MIN_DURATION_SEC = 30.0


def run_qc(
    rows: Sequence[SessionIndexRow],
    activity_matrices: Sequence[ActivityMatrix | None],
) -> list[QcIssue]:
    """Return a list of QcIssue objects for detected problems."""
    issues: list[QcIssue] = []

    for row in rows:
        if not row.interval_names:
            issues.append(
                QcIssue(
                    path=str(row.local_path),
                    issue_type="missing_intervals",
                    message="No named intervals found in NWB file.",
                    severity="warning",
                )
            )

    for mat in activity_matrices:
        if mat is None:
            continue

        if mat.n_units < _MIN_UNITS:
            issues.append(
                QcIssue(
                    path=mat.session_id,
                    issue_type="too_few_units",
                    message=f"Only {mat.n_units} units found (minimum {_MIN_UNITS}).",
                    severity="warning",
                )
            )

        duration = (
            float(mat.timestamps[-1]) - float(mat.timestamps[0])
            if mat.n_time > 1
            else 0.0
        )
        if duration < _MIN_DURATION_SEC:
            issues.append(
                QcIssue(
                    path=mat.session_id,
                    issue_type="short_activity_matrix",
                    message=f"Activity matrix duration {duration:.1f}s < {_MIN_DURATION_SEC}s.",
                    severity="warning",
                )
            )

        data = np.array(mat.data, dtype=float)
        if data.size == 0 or not np.any(data != 0):
            issues.append(
                QcIssue(
                    path=mat.session_id,
                    issue_type="empty_activity_matrix",
                    message="Activity matrix contains only zeros.",
                    severity="error",
                )
            )

    return issues
