"""P0.2 export-contract helpers: fMRI vs EEG stamps and row-source flags."""

from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline.export_contract import (
    EEG_H5_CONTRACT_VERSION,
    FMRI_H5_CONTRACT_VERSION,
    classify_row_source,
    resolve_export_contract_version,
    row_has_eeg,
    row_has_meg,
    source_format_for_row_source,
)


def test_classify_row_source_nifti_and_eeg_suffixes():
    assert classify_row_source("sub-001_task-rest_bold.nii.gz") == "nifti_bold"
    assert classify_row_source("sub-001_task-rest_bold.nii") == "nifti_bold"
    assert classify_row_source("sub-001_task-rest_eeg.set") == "set_eeg"
    assert classify_row_source("sub-001_task-rest_eeg.fdt") == "set_eeg"
    assert classify_row_source("sub-001_task-rest_meg.fif") == "fif_meeg"
    assert classify_row_source("sub-001_task-rest_meg.fif.gz") == "fif_meeg"
    assert classify_row_source("sub-001_task-rest_eeg.edf") == "unknown"
    assert classify_row_source("sub-001_task-rest_eeg.vhdr") == "unknown"


def test_nifti_row_source_has_no_eeg_flag():
    assert row_has_eeg("nifti_bold") is False
    assert row_has_meg("nifti_bold") is False
    assert row_has_eeg("set_eeg") is True
    assert row_has_eeg("fif_meeg") is True
    assert row_has_meg("fif_meeg") is True
    assert row_has_eeg("unknown") is True
    assert source_format_for_row_source("nifti_bold") == "nifti_bold"


def test_export_contract_version_fmri_from_modality_or_nifti_rows():
    assert resolve_export_contract_version("fmri") == FMRI_H5_CONTRACT_VERSION
    assert resolve_export_contract_version("eeg") == EEG_H5_CONTRACT_VERSION
    assert resolve_export_contract_version("meg") == EEG_H5_CONTRACT_VERSION
    assert (
        resolve_export_contract_version(None, ["nifti_bold", "nifti_bold"])
        == FMRI_H5_CONTRACT_VERSION
    )
    assert resolve_export_contract_version(None, ["set_eeg"]) == EEG_H5_CONTRACT_VERSION
    assert resolve_export_contract_version(None, []) == EEG_H5_CONTRACT_VERSION
    mixed = np.array(["nifti_bold", "unknown"], dtype=object)
    assert resolve_export_contract_version(None, mixed) == EEG_H5_CONTRACT_VERSION
    # Config modality is the recording-level stamp; mixed rows stay visible
    # via per-row has_eeg / row_source (not partitioned in P0.2).
    assert (
        resolve_export_contract_version("fmri", ["nifti_bold", "set_eeg"])
        == FMRI_H5_CONTRACT_VERSION
    )
