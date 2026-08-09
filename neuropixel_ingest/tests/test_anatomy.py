"""HDF5-level anatomy and probe geometry coverage."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from neuropixel_ingest.anatomy import enrich_units_anatomy, read_probe_geometry
from neuropixel_ingest.contracts import AnatomyReadConfig


def test_geometry_and_unit_anatomy_from_electrode_references(tmp_path: Path) -> None:
    path = tmp_path / "anatomy.nwb"
    _write_h5_layout(path)
    geometry = read_probe_geometry(path)
    assert geometry["geometry_status"].eq("placeholder").all()
    assert geometry["location_acronym"].tolist() == ["ALM", "ALM"]

    units = pd.DataFrame({"unit_id": [1], "spike_times": [np.asarray([0.1])]})
    enriched = enrich_units_anatomy(path, units, AnatomyReadConfig())
    assert enriched.loc[0, "location_raw"].startswith("brain_region: ALM")
    assert enriched.loc[0, "location_acronym"] == "ALM"
    assert enriched.loc[0, "electrode_group"] == "probeA"


def _write_h5_layout(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        units = handle.create_group("units")
        units.create_dataset("electrodes", data=np.asarray([0, 1], dtype=np.int64))
        units.create_dataset("electrodes_index", data=np.asarray([2], dtype=np.int64))
        electrodes = handle.create_group("general/extracellular_ephys/electrodes")
        electrodes.create_dataset("id", data=np.asarray([0, 1], dtype=np.int64))
        electrodes.create_dataset("x", data=np.asarray([0.0, 0.0]))
        electrodes.create_dataset("y", data=np.asarray([0.0, 0.0]))
        electrodes.create_dataset("z", data=np.asarray([0.0, 0.0]))
        electrodes.create_dataset("location", data=np.asarray([b"brain_region: ALM", b"brain_region: ALM"]))
        electrodes.create_dataset("group_name", data=np.asarray([b"probeA", b"probeA"]))
