"""Quality-control helpers for DANDI 001710 sessions.

Each check returns a list of ``QcIssue`` instances; an empty list means
the check passed.  All checks are non-destructive.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dandi_analysis.contracts import QcIssue
from dandi_analysis.dataset_001710.behavior import BehaviorTable
from dandi_analysis.dataset_001710.io import SessionChannel
from dandi_analysis.dataset_001710.ophys import OphysMatrix
from dandi_analysis.dataset_001710.trials import TrialTable


_MIN_ROIS = 20
_MIN_TRIALS = 5
_MAX_LEFT_RIGHT_IMBALANCE = 4.0  # max ratio of left/right trial counts


def check_behavior_channels(
    behavior: BehaviorTable,
    *,
    required_channels: tuple[str, ...] = (
        "position",
        "trial_start",
        "trial_end",
        "arm",
        "block",
    ),
) -> list[QcIssue]:
    issues: list[QcIssue] = []
    for ch in required_channels:
        if ch not in behavior.channels:
            issues.append(
                QcIssue(
                    path=str(behavior.session_path),
                    issue_type="missing_behavior_channel",
                    message=f"Required channel '{ch}' not found in {behavior.source} container.",
                    severity="warning",
                )
            )
    return issues


def check_frame_count_consistency(
    ophys: OphysMatrix,
    behavior: BehaviorTable,
    *,
    tolerance_frames: int = 10,
) -> list[QcIssue]:
    issues: list[QcIssue] = []
    diff = abs(ophys.n_frames - behavior.n_frames)
    if diff > tolerance_frames:
        issues.append(
            QcIssue(
                path=str(ophys.session_path),
                issue_type="frame_count_mismatch",
                message=(
                    f"Ophys has {ophys.n_frames} frames, behavior has "
                    f"{behavior.n_frames} frames (|diff|={diff})."
                ),
                severity="warning",
            )
        )
    return issues


def check_annotation_blob(
    blob: dict[str, Any],
    path: Path,
    *,
    required_keys: tuple[str, ...] = ("trial_start_inds", "day", "novel_arm"),
) -> list[QcIssue]:
    issues: list[QcIssue] = []
    if not blob:
        issues.append(
            QcIssue(
                path=str(path),
                issue_type="missing_annotation_blob",
                message="trial_cell_data annotation payload is absent or empty.",
                severity="warning",
            )
        )
        return issues
    for key in required_keys:
        if key not in blob:
            issues.append(
                QcIssue(
                    path=str(path),
                    issue_type="missing_annotation_blob_key",
                    message=f"Expected key '{key}' not found in annotation blob.",
                    severity="warning",
                )
            )
    return issues


def check_trial_table(trials: TrialTable) -> list[QcIssue]:
    issues: list[QcIssue] = []
    path = str(trials.session_path)

    if len(trials) == 0:
        issues.append(
            QcIssue(
                path=path,
                issue_type="empty_trial_table",
                message="No trials reconstructed.",
                severity="error",
            )
        )
        return issues

    valid = trials.valid_trials()
    if len(valid) < _MIN_TRIALS:
        issues.append(
            QcIssue(
                path=path,
                issue_type="too_few_valid_trials",
                message=f"Only {len(valid)} valid trials (min {_MIN_TRIALS}).",
                severity="warning",
            )
        )

    # Check ordering: trial start times must be non-decreasing
    starts = [t.start_time for t in trials.trials]
    for i in range(1, len(starts)):
        if starts[i] < starts[i - 1] - 0.001:
            issues.append(
                QcIssue(
                    path=path,
                    issue_type="trial_ordering_violation",
                    message=f"Trial {i} starts before trial {i - 1} ({starts[i]:.3f} < {starts[i - 1]:.3f}).",
                    severity="warning",
                )
            )
            break  # report only first violation

    # Left/right imbalance
    n_left = len(trials.by_arm("left"))
    n_right = len(trials.by_arm("right"))
    if n_left > 0 and n_right > 0:
        ratio = max(n_left, n_right) / min(n_left, n_right)
        if ratio > _MAX_LEFT_RIGHT_IMBALANCE:
            issues.append(
                QcIssue(
                    path=path,
                    issue_type="arm_imbalance",
                    message=(
                        f"Suspicious arm imbalance: left={n_left}, right={n_right} "
                        f"(ratio={ratio:.1f}, max allowed={_MAX_LEFT_RIGHT_IMBALANCE})."
                    ),
                    severity="warning",
                )
            )
    elif n_left == 0 and n_right == 0:
        issues.append(
            QcIssue(
                path=path,
                issue_type="no_arm_labels",
                message="No trials have arm labels (all 'unknown').",
                severity="warning",
            )
        )

    return issues


def check_roi_count(ophys: OphysMatrix) -> list[QcIssue]:
    issues: list[QcIssue] = []
    if ophys.n_rois < _MIN_ROIS:
        issues.append(
            QcIssue(
                path=str(ophys.session_path),
                issue_type="too_few_rois",
                message=f"Only {ophys.n_rois} ROIs (min {_MIN_ROIS}).",
                severity="warning",
            )
        )
    return issues


def run_all_checks(
    *,
    behavior: BehaviorTable | None = None,
    ophys: OphysMatrix | None = None,
    trials: TrialTable | None = None,
    blob: dict[str, Any] | None = None,
    path: Path | None = None,
    channels: list[SessionChannel] | None = None,
    all_matrices: list[OphysMatrix] | None = None,
) -> list[QcIssue]:
    """Run all applicable QC checks and return the combined issue list."""
    all_issues: list[QcIssue] = []

    if behavior is not None:
        all_issues.extend(check_behavior_channels(behavior))

    if ophys is not None:
        all_issues.extend(check_roi_count(ophys))

    if ophys is not None and behavior is not None:
        all_issues.extend(check_frame_count_consistency(ophys, behavior))

    if trials is not None:
        all_issues.extend(check_trial_table(trials))

    if blob is not None and path is not None:
        all_issues.extend(check_annotation_blob(blob, path))

    if channels is not None and all_matrices is not None and path is not None:
        all_issues.extend(check_multichannel_structure(channels, all_matrices, path))

    return all_issues


def check_multichannel_structure(
    channels: list[SessionChannel],
    matrices: list[OphysMatrix],
    path: Path,
) -> list[QcIssue]:
    """QC checks specific to multi-channel (SparseKO-style) sessions."""
    issues: list[QcIssue] = []
    path_str = str(path)

    if not channels:
        issues.append(QcIssue(
            path=path_str,
            issue_type="no_channels_resolved",
            message="Channel resolver returned no channels.",
            severity="error",
        ))
        return issues

    if len(channels) > 1:
        # Check that all channels have the same frame count
        frame_counts = [m.n_frames for m in matrices]
        if len(set(frame_counts)) > 1:
            issues.append(QcIssue(
                path=path_str,
                issue_type="channel_frame_count_mismatch",
                message=f"Channels have different frame counts: {dict(zip([c.channel_id for c in channels], frame_counts))}.",
                severity="warning",
            ))

        # Check ROI counts per channel
        for ch, mat in zip(channels, matrices):
            if mat.n_rois < _MIN_ROIS:
                issues.append(QcIssue(
                    path=path_str,
                    issue_type="too_few_rois_channel",
                    message=f"Channel {ch.channel_id} has only {mat.n_rois} ROIs (min {_MIN_ROIS}).",
                    severity="warning",
                ))

        # Check that each channel resolved a distinct behavior container
        beh_containers = [c.behavior_container for c in channels]
        if len(set(beh_containers)) < len(channels):
            issues.append(QcIssue(
                path=path_str,
                issue_type="duplicate_behavior_containers",
                message=f"Multiple channels map to the same behavior container: {beh_containers}.",
                severity="warning",
            ))

    return issues


def format_qc_report(issues: list[QcIssue], session_label: str = "") -> str:
    """Return a compact markdown QC report."""
    header = f"# QC Report{': ' + session_label if session_label else ''}"
    if not issues:
        return header + "\n\nAll checks passed.\n"
    lines = [header, "", f"**{len(issues)} issue(s) found:**", ""]
    for issue in issues:
        lines.append(
            f"- [{issue.severity.upper()}] `{issue.issue_type}`: {issue.message}"
        )
    return "\n".join(lines) + "\n"
