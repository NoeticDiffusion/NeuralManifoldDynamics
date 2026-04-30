"""Analysis package for DANDI 001710 (Y-maze context-sensitive remapping).

This package provides an analysis scaffold for discovering, probing, and
extracting place-code and remapping observables from 001710 NWB files.  It
mirrors the structure of dataset_000718 / dataset_000871 but is tailored for
longitudinal virtual-reality Y-maze calcium imaging.

Pipeline shape:
    dandi_io  →  inventory/readiness  →  index  →  behavior/trials  →
    ophys  →  placecode  →  remapping  →  nulls  →  exports
"""
from dandi_analysis.dataset_001710.behavior import BehaviorTable, load_behavior_table
from dandi_analysis.dataset_001710.exports import (
    export_dff_matrix,
    export_metadata_json,
    export_qc_report,
    export_session_index,
    export_similarity_matrix,
    export_trial_table,
    export_tuning_summary,
)
from dandi_analysis.dataset_001710.index import build_session_index, parse_subject_session
from dandi_analysis.dataset_001710.io import (
    SessionChannel,
    list_session_channels,
    open_nwb_readonly,
    read_behavior_series,
    read_plane_segmentation,
    read_roi_response_series,
    read_trial_annotation_blob,
    resolve_session_channels,
)
from dandi_analysis.dataset_001710.metadata import extract_nwb_metadata
from dandi_analysis.dataset_001710.nulls import (
    arm_label_shuffle,
    circular_time_shift,
    generate_null_distribution,
    position_bin_shuffle,
    trial_label_shuffle,
)
from dandi_analysis.dataset_001710.ophys import OphysMatrix, align_ophys_to_behavior, load_all_channel_matrices, load_ophys_matrix
from dandi_analysis.dataset_001710.placecode import (
    ArmTuning,
    TuningCurveSet,
    arm_tuning,
    compute_tuning_curves,
    reward_zone_summary,
    split_half_reliability,
)
from dandi_analysis.dataset_001710.qc import (
    QcIssue,
    check_multichannel_structure,
    format_qc_report,
    run_all_checks,
)
from dandi_analysis.dataset_001710.remapping import (
    RemappingResult,
    SimilarityMatrix,
    block_conditioned_similarity,
    build_day_similarity_matrix,
    cross_day_tuning_correlation,
    within_day_arm_separation,
)
from dandi_analysis.dataset_001710.trials import TrialRow, TrialTable, build_trial_table

__all__ = [
    # behavior
    "BehaviorTable",
    "load_behavior_table",
    # exports
    "export_dff_matrix",
    "export_metadata_json",
    "export_qc_report",
    "export_session_index",
    "export_similarity_matrix",
    "export_trial_table",
    "export_tuning_summary",
    # index
    "build_session_index",
    "parse_subject_session",
    # io
    "SessionChannel",
    "list_session_channels",
    "open_nwb_readonly",
    "read_behavior_series",
    "read_plane_segmentation",
    "read_roi_response_series",
    "read_trial_annotation_blob",
    "resolve_session_channels",
    # metadata
    "extract_nwb_metadata",
    # nulls
    "arm_label_shuffle",
    "circular_time_shift",
    "generate_null_distribution",
    "position_bin_shuffle",
    "trial_label_shuffle",
    # ophys
    "OphysMatrix",
    "align_ophys_to_behavior",
    "load_all_channel_matrices",
    "load_ophys_matrix",
    # placecode
    "ArmTuning",
    "TuningCurveSet",
    "arm_tuning",
    "compute_tuning_curves",
    "reward_zone_summary",
    "split_half_reliability",
    # qc
    "QcIssue",
    "check_multichannel_structure",
    "format_qc_report",
    "run_all_checks",
    # remapping
    "RemappingResult",
    "SimilarityMatrix",
    "block_conditioned_similarity",
    "build_day_similarity_matrix",
    "cross_day_tuning_correlation",
    "within_day_arm_separation",
    # trials
    "TrialRow",
    "TrialTable",
    "build_trial_table",
]
