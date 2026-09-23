"""Config-driven downloads from the public NEMAR data plane."""

from .client import NemarClient
from .download import download_dataset, download_datasets

__all__ = ["NemarClient", "download_dataset", "download_datasets"]
