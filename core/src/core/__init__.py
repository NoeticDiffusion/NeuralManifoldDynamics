"""Shared core utilities for ingest and MNPS packages.

This package exposes configuration loading, dataset registry helpers, path
resolution, BIDS parsing, IO writers, graph metrics, and robust statistics used
by the MNDM and ingest pipelines.
"""

from . import config_loader  # noqa: F401
from . import datasets  # noqa: F401
from . import ensembles  # noqa: F401
from . import paths  # noqa: F401

