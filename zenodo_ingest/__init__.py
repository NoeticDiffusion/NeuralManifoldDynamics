"""Config-driven downloads from public Zenodo records.

Large files are streamed to disk and can resume from a ``.partial`` file
with an HTTP Range request. Checksums published by Zenodo are content
checks, not a security signature.
"""

from zenodo_ingest.client import ZenodoClient
from zenodo_ingest.download import download_record, download_records

__all__ = ["ZenodoClient", "download_record", "download_records"]
