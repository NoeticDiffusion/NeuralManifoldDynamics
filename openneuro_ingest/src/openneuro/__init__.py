"""Top-level package for OpenNeuro ingest.

This package provides a modular pipeline to download BIDS datasets from
OpenNeuro, index files and events, preprocess multimodal signals, and extract
epoch-based features for downstream MNPS computation.
"""

from .__about__ import __version__, __description__, __author__, __contact__  # noqa: F401

__all__ = [
    "__version__",
    "__description__",
    "__author__",
    "__contact__",
]


