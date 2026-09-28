"""Export-contract identifiers and per-row source classification.

P0.2: fMRI HDF5 must not inherit the EEG contract stamp or ``has_eeg=1``.
EEG/MEG FIF+SET classification is unchanged. Unknown suffixes stay EEG-like
(EDF/BrainVision) so existing non-SET EEG paths keep ``has_eeg=1``.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

EEG_H5_CONTRACT_VERSION = "mndm.eeg_h5_contract.v1"
FNIRS_H5_CONTRACT_VERSION = "mndm.fnirs_native_h5_contract.v1"
# v2 marks fMRI files written by the 3.0.2 writer (P5 Jacobian support gate
# is available). v1 and the historical EEG string remain on older files.
# The stamp follows modality; it does not prove an overlay left the gate on.
FMRI_H5_CONTRACT_VERSION = "mndm.fmri_h5_contract.v2"

ROW_SOURCE_FIF_MEEG = "fif_meeg"
ROW_SOURCE_SET_EEG = "set_eeg"
ROW_SOURCE_NIFTI_BOLD = "nifti_bold"
ROW_SOURCE_FNIRS_SNIRF = "snirf_fnirs"
ROW_SOURCE_UNKNOWN = "unknown"

SOURCE_FORMAT = {
    ROW_SOURCE_FIF_MEEG: "neuromag_fif",
    ROW_SOURCE_SET_EEG: "eeglab_set",
    ROW_SOURCE_NIFTI_BOLD: "nifti_bold",
    ROW_SOURCE_FNIRS_SNIRF: "snirf",
}


def resolve_export_contract_version(
    modality: Any,
    row_sources: Optional[Iterable[Any]] = None,
) -> str:
    """Return the HDF5 export-contract identifier for this recording.

    fMRI and fNIRS use distinct contracts selected by config modality or a
    homogeneous row-source set. EEG/MEG keep the historical contract string.

    Recording-level stamp follows config modality when it is ``fmri`` even if
    a mixed ``file`` list is present. Per-row ``has_eeg`` / ``row_source``
    remain the source of truth for those rows; mixed EEG+NIfTI under an fMRI
    config is pathological and is not partitioned here.
    """
    token = str(modality or "").strip().lower()
    if token == "fnirs":
        return FNIRS_H5_CONTRACT_VERSION
    if token == "fmri":
        return FMRI_H5_CONTRACT_VERSION
    if row_sources is not None:
        classified = [str(src) for src in row_sources]
        if classified and all(src == ROW_SOURCE_FNIRS_SNIRF for src in classified):
            return FNIRS_H5_CONTRACT_VERSION
        if classified and all(src == ROW_SOURCE_NIFTI_BOLD for src in classified):
            return FMRI_H5_CONTRACT_VERSION
    return EEG_H5_CONTRACT_VERSION


def classify_row_source(fname: str) -> str:
    """Classify a raw filename into the ``/row_source`` vocabulary."""
    fl = str(fname or "").strip().lower()
    if fl.endswith(".snirf"):
        return ROW_SOURCE_FNIRS_SNIRF
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
    return str(row_source) not in {ROW_SOURCE_NIFTI_BOLD, ROW_SOURCE_FNIRS_SNIRF}


def row_has_meg(row_source: str) -> bool:
    """Return whether this source should be flagged as containing MEG."""
    return str(row_source) == ROW_SOURCE_FIF_MEEG
