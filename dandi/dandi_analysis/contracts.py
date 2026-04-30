from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DiscoveredNwbAsset:
    """A locally discovered NWB file, canonical or duplicate.

    Attributes:
        path: Relative or absolute path to the file on disk.
        size: File size in bytes.
        mtime: Last modification time of the file.
        is_canonical: True if this is the primary version of a duplicate set.
        duplicate_of: Path to the canonical file if this is a duplicate.
        metadata: Additional metadata discovered during scanning.
    """

    path: Path
    size: int
    mtime: float
    is_canonical: bool
    duplicate_of: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ReadyNwbAsset:
    """A discovered NWB file that has passed readiness checks.

    Attributes:
        path: Path to the validated NWB file.
        size: File size in bytes.
        is_h5_openable: Whether the file can be opened by the H5 library.
        is_nwb_openable: Whether the file can be opened by the PyNWB library.
        error: Descriptive error message if any check failed.
        metadata: Additional metadata extracted during validation.
    """

    path: Path
    size: int
    is_h5_openable: bool
    is_nwb_openable: bool
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ready(self) -> bool:
        """Returns True if the asset passed all readiness checks."""
        return self.is_h5_openable and self.is_nwb_openable and self.error is None


@dataclass(slots=True)
class SessionIndexRow:
    """Normalized summary of one NWB session file.

    Attributes:
        subject_id: Unique identifier for the animal/subject.
        session_label: Human-readable label for the recording session.
        local_path: Absolute path to the local file.
        size: File size in bytes.
        state: Recording state (e.g., 'rest', 'active', 'sleep').
        description: Brief description of the session content.
        start_time: ISO timestamp of the session start.
        processing_keys: Top-level processing module names.
        interval_names: Names of available time intervals (epochs).
        imaging_planes: Names of recorded imaging planes.
        metadata: Arbitrary dictionary for additional session-level data.
    """

    subject_id: str
    session_label: str
    local_path: Path
    size: int
    state: str
    description: str = ""
    start_time: str = ""
    processing_keys: tuple[str, ...] = ()
    interval_names: tuple[str, ...] = ()
    imaging_planes: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OfflineWindow:
    """A candidate offline or rest window extracted from an NWB file.

    Attributes:
        session_id: ID of the parent session.
        label: Name or type of the window.
        start_sec: Start time in seconds from session start.
        stop_sec: End time in seconds from session start.
        epoch_type: Type of epoch (e.g., 'SWS', 'REM', 'Quiet').
        unit_count: Number of active units during this window.
        metadata: Additional diagnostics for the window.
    """

    session_id: str
    label: str
    start_sec: float
    stop_sec: float
    epoch_type: str
    unit_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        """Returns the duration of the window in seconds."""
        return self.stop_sec - self.start_sec


@dataclass
class ActivityMatrix:
    """A time x units neural activity array for one session/window.

    Attributes:
        session_id: ID of the parent session.
        data: Numpy array of shape (Time, Units) containing activity values.
        unit_ids: Identifiers for each column (unit) in the data.
        timestamps: Numpy array of shape (Time,) with absolute or relative times.
        sampling_rate: Inferred or declared sampling rate in Hz.
        window_label: Optional label for the subset of data.
        metadata: Additional diagnostic data about the extraction.
    """

    session_id: str
    data: Any  # np.ndarray shape (T, N)
    unit_ids: tuple[str, ...]
    timestamps: Any  # np.ndarray shape (T,)
    sampling_rate: float
    window_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_time(self) -> int:
        """Returns the number of time points (rows) in the data."""
        return int(self.data.shape[0])

    @property
    def n_units(self) -> int:
        """Returns the number of units (columns) in the data."""
        return int(self.data.shape[1])


@dataclass(slots=True)
class PairwiseCoreactivationResult:
    """Offline co-reactivation score for one pair of unit groups.

    Attributes:
        session_id: ID of the parent session.
        trace_i: ID of the first unit group (engram trace).
        trace_j: ID of the second unit group (engram trace).
        co_reactivation_score: Measured coreactivation in the target window.
        null_mean: Mean coreactivation across shuffled null distributions.
        null_std: Standard deviation of coreactivation in the null distribution.
        z_score: Standardized score ( (observed - mean) / std ).
        window_label: Label of the window where this was measured.
        metadata: Arbitrary dictionary for additional diagnostics.
    """

    session_id: str
    trace_i: str
    trace_j: str
    co_reactivation_score: float
    null_mean: float
    null_std: float
    z_score: float
    window_label: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class QcIssue:
    """A single quality-control issue flagged for a file or matrix.

    Attributes:
        path: Path to the affected file or asset.
        issue_type: Category of the issue (e.g., 'missing_data', 'low_unit_count').
        message: Descriptive explanation of the problem.
        severity: Level of concern ('warning', 'error', 'info').
        metadata: Additional diagnostic context.
    """

    path: str
    issue_type: str
    message: str
    severity: str = "warning"
    metadata: dict[str, Any] = field(default_factory=dict)
