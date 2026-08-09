"""Portable NWB extracellular-electrophysiology ingestion utilities.

This package turns an NWB Units table into quality-filtered, binned population
rate matrices.  Dataset acquisition remains the responsibility of
``dandi_ingest``; MNPS feature calculation remains in ``mndm``.
"""

from .contracts import EphysReadConfig, SessionBundle
from .lfp_qc import run_lfp_channel_qc
from .nwb_units import read_session_bundle, read_units

__all__ = ["EphysReadConfig", "SessionBundle", "read_session_bundle", "read_units", "run_lfp_channel_qc"]
