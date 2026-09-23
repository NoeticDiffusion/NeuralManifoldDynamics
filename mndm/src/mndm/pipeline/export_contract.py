"""Export-contract identifiers and per-row source classification.

P0.2: fMRI HDF5 must not inherit the EEG contract stamp or ``has_eeg=1``.
EEG/MEG FIF+SET classification is unchanged. Unknown suffixes stay EEG-like
(EDF/BrainVision) so existing non-SET EEG paths keep ``has_eeg=1``.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

EEG_H5_CONTRACT_VERSION = "mndm.eeg_h5_contract.v1"
FMRI_H5_CONTRACT_VERSION = "mndm.fmri_h5_contract.v1"

ROW_SOURCE_FIF_MEEG = "fif_meeg"
ROW_SOURCE_SET_EEG = "set_eeg"
ROW_SOURCE_NIFTI_BOLD = "nifti_bold"
ROW_SOURCE_UNKNOWN = "unknown"

SOURCE_FORMAT = {
    ROW_SOURCE_FIF_MEEG: "neuromag_fif",
    ROW_SOURCE_SET_EEG: "eeglab_set",
    ROW_SOURCE_NIFTI_BOLD: "nifti_bold",
}


def resolve_export_contract_version(
    modality: Any,
    row_sources: Optional[Iterable[Any]] = None,
) -> str:
    """Return the HDF5 export-contract identifier for this recording.

    fMRI is selected when config ``modality`` is ``fmri``, or when every
    classified row source is NIfTI BOLD. Other modalities keep the historical
    EEG contract string so existing EEG/MEG readers do not break.

    Recording-level stamp follows config modality when it is ``fmri`` even if
    a mixed ``file`` list is present. Per-row ``has_eeg`` / ``row_source``
    remain the source of truth for those rows; mixed EEG+NIfTI under an fMRI
    config is pathological and is not partitioned here.
    """
    token = str(modality or "").strip().lower()
    if token == "fmri":
        return FMRI_H5_CONTRACT_VERSION
    if row_sources is not None:
        classified = [str(src) for src in row_sources]
        if classified and all(src == ROW_SOURCE_NIFTI_BOLD for src in classified):
            return FMRI_H5_CONTRACT_VERSION
    return EEG_H5_CONTRACT_VERSION


def classify_row_source(fname: str) -> str:
    """Classify a raw filename into the ``/row_source`` vocabulary."""
    fl = str(fname or "").strip().lower()
    if fl.endswith(".nii.gz") or fl.endswith(".nii"):
        return ROW_SOURCE_NIFTI_BOLD
    if fl.endswith(".fif") or fl.endswith(".fif.gz"):
        return ROW_SOURCE_FIF_MEEG
    if fl.endswith(".set") or fl.endswith(".fdt"):
        return ROW_SOURCE_SET_EEG
    return ROW_SOURCE_UNKNOWN


def source_format_for_row_source(row_source: str) -> str:
    """Map a ``row_source`` token onto ``source_format``."""
    return SOURCE_FORMAT.get(str(row_source), ROW_SOURCE_UNKNOWN)


def row_has_eeg(row_source: str) -> bool:
    """Return whether this source should be flagged as containing EEG."""
    return str(row_source) != ROW_SOURCE_NIFTI_BOLD


def row_has_meg(row_source: str) -> bool:
    """Return whether this source should be flagged as containing MEG."""
    return str(row_source) == ROW_SOURCE_FIF_MEEG
