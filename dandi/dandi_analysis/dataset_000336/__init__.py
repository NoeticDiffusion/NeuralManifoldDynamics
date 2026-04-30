"""Analysis package for DANDI 000336 (Allen Institute OpenScope dendritic coupling).

This package provides lightweight wrappers for discovering, probing, and extracting
coupling-relevant observables from 000336 NWB files.  It mirrors the structure of
dataset_000718 but is tailored for two-plane / somato-dendritic imaging.
"""
from dandi_analysis.dataset_000336.index import build_session_index, parse_subject_session
from dandi_analysis.dataset_000336.io import open_nwb_readonly, safe_read_session_metadata
from dandi_analysis.dataset_000336.metadata import extract_nwb_metadata

__all__ = [
    "build_session_index",
    "extract_nwb_metadata",
    "open_nwb_readonly",
    "parse_subject_session",
    "safe_read_session_metadata",
]
