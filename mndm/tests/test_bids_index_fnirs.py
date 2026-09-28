"""BIDS index coverage for the validated continuous-wave SNIRF route."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def test_build_file_index_detects_fnirs_snirf_and_explicit_sidecars(tmp_path: Path):
    """Index one BIDS nirs run and preserve existing EEG/fMRI row semantics."""
    from mndm.bids_index import build_file_index

    dataset_root = tmp_path / "ds005963"
    nirs_dir = dataset_root / "sub-001" / "ses-left2s" / "nirs"
    eeg_dir = dataset_root / "sub-001" / "ses-left2s" / "eeg"
    func_dir = dataset_root / "sub-001" / "ses-left2s" / "func"
    nirs_dir.mkdir(parents=True)
    eeg_dir.mkdir(parents=True)
    func_dir.mkdir(parents=True)

    nirs_stem = "sub-001_ses-left2s_task-FRESHMOTOR_acq-source3_run-02"
    nirs_file = nirs_dir / f"{nirs_stem}_nirs.snirf"
    nirs_json = nirs_dir / f"{nirs_stem}_nirs.json"
    channels_tsv = nirs_dir / f"{nirs_stem}_channels.tsv"
    events_tsv = nirs_dir / f"{nirs_stem}_events.tsv"
    # BIDS NIRS has suffix-specific entity rules: optodes cannot carry task
    # or run, and coordsystem cannot carry run.
    optodes_tsv = nirs_dir / "sub-001_ses-left2s_acq-source3_optodes.tsv"
    coordsystem_json = nirs_dir / (
        "sub-001_ses-left2s_task-FRESHMOTOR_acq-source3_coordsystem.json"
    )

    nirs_file.write_bytes(b"synthetic SNIRF fixture")
    nirs_json.write_text('{"SamplingFrequency": 8.0}', encoding="utf-8")
    channels_tsv.write_text("name\ttype\nS1_D1 760\tNIRSCWAMPLITUDE\n", encoding="utf-8")
    events_tsv.write_text("onset\tduration\ttrial_type\n0\t1\trest\n", encoding="utf-8")
    optodes_tsv.write_text("name\tx\ty\tz\nS1\t0\t0\t0\n", encoding="utf-8")
    coordsystem_json.write_text('{"NIRSCoordinateSystem": "CapTrak"}', encoding="utf-8")

    eeg_file = eeg_dir / "sub-001_ses-left2s_task-FRESHMOTOR_run-02_eeg.edf"
    bold_file = func_dir / "sub-001_ses-left2s_task-FRESHMOTOR_run-02_bold.nii.gz"
    eeg_file.write_bytes(b"synthetic EDF fixture")
    bold_file.write_bytes(b"synthetic NIfTI fixture")

    index_df = build_file_index(
        dataset_root,
        config={"datasets": ["ds005963"]},
        dataset_id="ds005963",
    )

    assert len(index_df) == 3
    nirs_row = index_df[index_df["path"] == str(nirs_file.relative_to(dataset_root))].iloc[0]
    assert nirs_row["datatype"] == "nirs"
    assert nirs_row["modality"] == "fnirs"
    assert nirs_row["source_format"] == "snirf"
    assert nirs_row["subject"] == "001"
    assert nirs_row["session"] == "left2s"
    assert nirs_row["task"] == "FRESHMOTOR"
    assert nirs_row["acq"] == "source3"
    assert nirs_row["run"] == "02"
    assert nirs_row["bundle_key"] == "001|left2s|FRESHMOTOR|02|source3"

    expected_sidecars = {
        "nirs_json": nirs_json,
        "nirs_channels_tsv": channels_tsv,
        "nirs_events_tsv": events_tsv,
        "nirs_optodes_tsv": optodes_tsv,
        "nirs_coordsystem_json": coordsystem_json,
    }
    for column, sidecar in expected_sidecars.items():
        assert nirs_row[column] == str(sidecar.relative_to(dataset_root))

    # Generic signal/channel/event aliases remain usable by current index
    # consumers. Modality-specific fields make fNIRS provenance unambiguous.
    assert nirs_row["signal_json"] == nirs_row["nirs_json"]
    assert nirs_row["channels_tsv"] == nirs_row["nirs_channels_tsv"]
    assert nirs_row["events_tsv"] == nirs_row["nirs_events_tsv"]
    assert nirs_row["eeg_json"] is None
    assert nirs_row["meg_json"] is None

    eeg_row = index_df[index_df["path"] == str(eeg_file.relative_to(dataset_root))].iloc[0]
    assert eeg_row["datatype"] == "eeg"
    assert eeg_row["modality"] == "eeg"
    assert eeg_row["source_format"] != "snirf"

    fmri_row = index_df[index_df["path"] == str(bold_file.relative_to(dataset_root))].iloc[0]
    assert fmri_row["datatype"] == "func"
    assert fmri_row["modality"] == "fmri"
    assert fmri_row["fmri_json"] is None


def test_build_file_index_ignores_snirf_outside_bids_nirs_datatype(tmp_path: Path):
    """A matching suffix outside BIDS `nirs/` is not silently classified."""
    from mndm.bids_index import build_file_index

    eeg_dir = tmp_path / "sub-001" / "eeg"
    eeg_dir.mkdir(parents=True)
    (eeg_dir / "sub-001_task-rest_nirs.snirf").write_bytes(b"not a BIDS nirs path")

    index_df = build_file_index(tmp_path)

    assert index_df.empty


def test_build_file_index_resolves_fnirs_inherited_sidecars_by_suffix_entities(tmp_path: Path):
    """Resolve lower-specificity/ancestor sidecars using NIRS entity rules."""
    from mndm.bids_index import build_file_index

    dataset_root = tmp_path / "ds-fnirs-inheritance"
    subject_dir = dataset_root / "sub-002"
    session_dir = subject_dir / "ses-01"
    nirs_dir = session_dir / "nirs"
    nirs_dir.mkdir(parents=True)

    nirs_file = nirs_dir / "sub-002_ses-01_task-rest_run-01_nirs.snirf"
    nirs_json = nirs_dir / "sub-002_ses-01_task-rest_nirs.json"
    channels_tsv = nirs_dir / "sub-002_ses-01_task-rest_channels.tsv"
    events_tsv = nirs_dir / "sub-002_ses-01_task-rest_events.tsv"
    # Optodes and coordinate metadata are shared at less-specific directory
    # levels; their allowed entity sets intentionally omit task/run as required.
    optodes_tsv = session_dir / "sub-002_ses-01_optodes.tsv"
    coordsystem_parent_json = subject_dir / "sub-002_coordsystem.json"
    coordsystem_json = nirs_dir / "sub-002_ses-01_task-rest_coordsystem.json"

    nirs_file.write_bytes(b"synthetic SNIRF fixture")
    nirs_json.write_text('{"SamplingFrequency": 8.0}', encoding="utf-8")
    channels_tsv.write_text("name\ttype\nS1_D1 760\tNIRSCWAMPLITUDE\n", encoding="utf-8")
    events_tsv.write_text("onset\tduration\ttrial_type\n0\t1\trest\n", encoding="utf-8")
    optodes_tsv.write_text("name\tx\ty\tz\nS1\t0\t0\t0\n", encoding="utf-8")
    coordsystem_parent_json.write_text('{"NIRSCoordinateUnits": "mm"}', encoding="utf-8")
    coordsystem_json.write_text('{"NIRSCoordinateSystem": "CapTrak"}', encoding="utf-8")

    index_df = build_file_index(dataset_root)

    assert len(index_df) == 1
    nirs_row = index_df.iloc[0]
    assert nirs_row["path"] == str(nirs_file.relative_to(dataset_root))
    assert nirs_row["nirs_json"] == str(nirs_json.relative_to(dataset_root))
    assert nirs_row["nirs_json_sidecars"] == [str(nirs_json.relative_to(dataset_root))]
    assert not bool(nirs_row["nirs_json_sidecar_ambiguous"])
    assert nirs_row["nirs_channels_tsv"] == str(channels_tsv.relative_to(dataset_root))
    assert nirs_row["nirs_channels_tsv_candidates"] == [
        str(channels_tsv.relative_to(dataset_root))
    ]
    assert not bool(nirs_row["nirs_channels_tsv_ambiguous"])
    assert nirs_row["nirs_events_tsv"] == str(events_tsv.relative_to(dataset_root))
    assert nirs_row["nirs_events_tsv_candidates"] == [
        str(events_tsv.relative_to(dataset_root))
    ]
    assert not bool(nirs_row["nirs_events_tsv_ambiguous"])
    assert nirs_row["nirs_optodes_tsv"] == str(optodes_tsv.relative_to(dataset_root))
    assert nirs_row["nirs_optodes_tsv_candidates"] == [
        str(optodes_tsv.relative_to(dataset_root))
    ]
    assert not bool(nirs_row["nirs_optodes_tsv_ambiguous"])
    assert nirs_row["nirs_coordsystem_json"] == str(coordsystem_json.relative_to(dataset_root))
    assert nirs_row["nirs_coordsystem_sidecars"] == [
        str(coordsystem_parent_json.relative_to(dataset_root)),
        str(coordsystem_json.relative_to(dataset_root)),
    ]
    assert not bool(nirs_row["nirs_coordsystem_sidecar_ambiguous"])


def test_build_file_index_applies_fnirs_json_inheritance_and_deepest_tsv(tmp_path: Path):
    """JSON sidecars form an ancestor-first chain; TSV uses deepest metadata."""
    import json

    from mndm.bids_index import build_file_index

    dataset_root = tmp_path / "ds-fnirs-inheritance-order"
    subject_dir = dataset_root / "sub-003"
    nirs_dir = subject_dir / "nirs"
    nirs_dir.mkdir(parents=True)
    nirs_file = nirs_dir / "sub-003_task-rest_run-02_nirs.snirf"
    parent_json = subject_dir / "sub-003_task-rest_nirs.json"
    child_json = nirs_dir / "sub-003_task-rest_run-02_nirs.json"
    # Both channels files are applicable. The parent file has more entities,
    # but BIDS simple-metadata precedence selects the lower/deeper directory.
    parent_channels = subject_dir / "sub-003_task-rest_run-02_channels.tsv"
    child_channels = nirs_dir / "sub-003_task-rest_channels.tsv"

    nirs_file.write_bytes(b"synthetic SNIRF fixture")
    parent_json.write_text(
        json.dumps({"SamplingFrequency": 8.0, "Manufacturer": "fixture"}),
        encoding="utf-8",
    )
    child_json.write_text(
        json.dumps({"SoftwareVersions": "fixture-child"}), encoding="utf-8"
    )
    parent_channels.write_text("name\ttype\nparent\tNIRSCWAMPLITUDE\n", encoding="utf-8")
    child_channels.write_text("name\ttype\nchild\tNIRSCWAMPLITUDE\n", encoding="utf-8")

    index_df = build_file_index(dataset_root)

    assert len(index_df) == 1
    nirs_row = index_df.iloc[0]
    parent_rel = str(parent_json.relative_to(dataset_root))
    child_rel = str(child_json.relative_to(dataset_root))
    assert nirs_row["nirs_json"] == child_rel
    assert nirs_row["nirs_json_sidecars"] == [parent_rel, child_rel]
    assert not bool(nirs_row["nirs_json_sidecar_ambiguous"])
    merged_metadata = {}
    for sidecar_rel in nirs_row["nirs_json_sidecars"]:
        merged_metadata.update(json.loads((dataset_root / sidecar_rel).read_text(encoding="utf-8")))
    assert merged_metadata == {
        "SamplingFrequency": 8.0,
        "Manufacturer": "fixture",
        "SoftwareVersions": "fixture-child",
    }
    assert nirs_row["nirs_channels_tsv"] == str(child_channels.relative_to(dataset_root))


def test_build_file_index_reports_fnirs_json_ambiguity_at_one_level(tmp_path: Path):
    """Multiple applicable JSON files at one directory level are surfaced."""
    from mndm.bids_index import build_file_index

    dataset_root = tmp_path / "ds-fnirs-json-ambiguity"
    subject_dir = dataset_root / "sub-004"
    nirs_dir = subject_dir / "nirs"
    nirs_dir.mkdir(parents=True)
    nirs_file = nirs_dir / "sub-004_task-rest_run-01_nirs.snirf"
    parent_json_a = subject_dir / "sub-004_task-rest_nirs.json"
    parent_json_b = subject_dir / "sub-004_task-rest_run-01_nirs.json"
    child_json = nirs_dir / "sub-004_task-rest_run-01_nirs.json"

    nirs_file.write_bytes(b"synthetic SNIRF fixture")
    parent_json_a.write_text('{"Manufacturer": "a"}', encoding="utf-8")
    parent_json_b.write_text('{"SoftwareVersions": "b"}', encoding="utf-8")
    child_json.write_text('{"SamplingFrequency": 8.0}', encoding="utf-8")

    index_df = build_file_index(dataset_root)

    assert len(index_df) == 1
    nirs_row = index_df.iloc[0]
    assert bool(nirs_row["nirs_json_sidecar_ambiguous"])
    assert nirs_row["nirs_json_sidecars"] == [
        str(parent_json_a.relative_to(dataset_root)),
        str(parent_json_b.relative_to(dataset_root)),
        str(child_json.relative_to(dataset_root)),
    ]
    assert nirs_row["nirs_json"] == str(child_json.relative_to(dataset_root))


def test_build_file_index_fails_closed_for_duplicate_fnirs_tsv_sidecars(tmp_path: Path):
    """Same-level duplicate applicable TSVs are listed and not selected."""
    from mndm.bids_index import build_file_index

    dataset_root = tmp_path / "ds-fnirs-tsv-ambiguity"
    subject_dir = dataset_root / "sub-005"
    session_dir = subject_dir / "ses-02"
    nirs_dir = session_dir / "nirs"
    nirs_dir.mkdir(parents=True)
    nirs_file = nirs_dir / (
        "sub-005_ses-02_task-rest_acq-source_run-01_nirs.snirf"
    )
    channels_a = nirs_dir / "sub-005_ses-02_task-rest_channels.tsv"
    channels_b = nirs_dir / "sub-005_ses-02_task-rest_run-01_channels.tsv"
    optodes_a = nirs_dir / "sub-005_ses-02_optodes.tsv"
    optodes_b = nirs_dir / "sub-005_ses-02_acq-source_optodes.tsv"

    nirs_file.write_bytes(b"synthetic SNIRF fixture")
    for sidecar in (channels_a, channels_b, optodes_a, optodes_b):
        sidecar.write_text("name\ttype\nfixture\tNIRSCWAMPLITUDE\n", encoding="utf-8")

    index_df = build_file_index(dataset_root)

    assert len(index_df) == 1
    nirs_row = index_df.iloc[0]
    assert nirs_row["nirs_channels_tsv"] is None
    assert bool(nirs_row["nirs_channels_tsv_ambiguous"])
    assert nirs_row["nirs_channels_tsv_candidates"] == [
        str(channels_a.relative_to(dataset_root)),
        str(channels_b.relative_to(dataset_root)),
    ]
    assert nirs_row["channels_tsv"] is None
    assert nirs_row["nirs_optodes_tsv"] is None
    assert bool(nirs_row["nirs_optodes_tsv_ambiguous"])
    assert nirs_row["nirs_optodes_tsv_candidates"] == [
        str(optodes_b.relative_to(dataset_root)),
        str(optodes_a.relative_to(dataset_root)),
    ]


def test_build_file_index_reports_coordsystem_json_inheritance_ambiguity(tmp_path: Path):
    """Coordsystem exposes its JSON chain and same-level duplicate status."""
    from mndm.bids_index import build_file_index

    dataset_root = tmp_path / "ds-fnirs-coordsystem-ambiguity"
    subject_dir = dataset_root / "sub-006"
    session_dir = subject_dir / "ses-01"
    nirs_dir = session_dir / "nirs"
    nirs_dir.mkdir(parents=True)
    nirs_file = nirs_dir / "sub-006_ses-01_task-rest_acq-default_run-01_nirs.snirf"
    parent_json_a = session_dir / "sub-006_ses-01_coordsystem.json"
    parent_json_b = session_dir / "sub-006_ses-01_task-rest_coordsystem.json"
    child_json = nirs_dir / "sub-006_ses-01_task-rest_acq-default_coordsystem.json"

    nirs_file.write_bytes(b"synthetic SNIRF fixture")
    parent_json_a.write_text('{"NIRSCoordinateUnits": "mm"}', encoding="utf-8")
    parent_json_b.write_text('{"NIRSCoordinateSystem": "CapTrak"}', encoding="utf-8")
    child_json.write_text('{"NIRSCoordinateSystemDescription": "fixture"}', encoding="utf-8")

    index_df = build_file_index(dataset_root)

    assert len(index_df) == 1
    nirs_row = index_df.iloc[0]
    assert nirs_row["nirs_coordsystem_json"] == str(child_json.relative_to(dataset_root))
    assert nirs_row["nirs_coordsystem_sidecars"] == [
        str(parent_json_a.relative_to(dataset_root)),
        str(parent_json_b.relative_to(dataset_root)),
        str(child_json.relative_to(dataset_root)),
    ]
    assert bool(nirs_row["nirs_coordsystem_sidecar_ambiguous"])
