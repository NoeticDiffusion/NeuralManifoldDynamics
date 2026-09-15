"""Non-BIDS filename_parse coverage for file indexing."""

from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.bids_index import build_file_index, enrich_index_with_filename_parse, parse_filename_entities

APOLLO_REGEX = (
    r"^(?P<subject>\d+)-\d{4}-(?P<task>[A-Za-z]+)-?\s+(?P<session>\d{8})\s+"
    r"(?P<acq>\d+)\.(?P<run>\d{3})\.[^.]+$"
)
APOLLO_CONFIG = {
    "metadata_extraction": {
        "datasets": {
            "Sedation-RestingState": {
                "filename_parse": {
                    "regex": APOLLO_REGEX,
                    "subject_pad": 3,
                }
            }
        }
    }
}


def test_parse_filename_entities_apollo_eeglab_names() -> None:
    """Apollo EEGLAB names should yield padded subject plus session/task/run/acq."""
    parsed = parse_filename_entities(
        "02-2010-anest 20100210 135.003.set",
        APOLLO_CONFIG,
        "Sedation-RestingState",
    )
    assert parsed["subject"] == "002"
    assert parsed["task"] == "anest"
    assert parsed["session"] == "20100210"
    assert parsed["acq"] == "135"
    assert parsed["run"] == "003"

    hyphenated = parse_filename_entities(
        "02-2010-anest- 20100210 16.003.set",
        APOLLO_CONFIG,
        "Sedation-RestingState",
    )
    assert hyphenated["subject"] == "002"
    assert hyphenated["acq"] == "16"
    assert hyphenated["run"] == "003"


def test_build_file_index_uses_filename_parse_for_apollo_eeglab(tmp_path: Path) -> None:
    """Indexing must populate subject from filename_parse when BIDS tokens are absent."""
    (tmp_path / "02-2010-anest 20100210 135.003.set").write_bytes(b"\x00")
    (tmp_path / "02-2010-anest- 20100210 16.003.set").write_bytes(b"\x00")
    (tmp_path / "03-2010-anest 20100211 142.003.set").write_bytes(b"\x00")

    index_df = build_file_index(
        tmp_path,
        config=APOLLO_CONFIG,
        dataset_id="Sedation-RestingState",
    )

    assert len(index_df) == 3
    assert set(index_df["subject"].astype(str)) == {"002", "003"}
    sub02 = index_df[index_df["subject"].astype(str) == "002"]
    assert len(sub02) == 2
    assert set(sub02["run"].astype(str)) == {"003"}
    assert "anest" in set(sub02["task"].astype(str))


def test_enrich_index_repairs_stale_empty_subject_csv() -> None:
    """Existing Apollo indexes with blank subject cells should be fillable in place."""
    stale = pd.DataFrame(
        {
            "path": [
                "02-2010-anest 20100210 135.003.set",
                "03-2010-anest 20100211 142.003.set",
            ],
            "subject": [None, ""],
            "session": [None, None],
            "task": [None, None],
            "run": [None, None],
            "acq": [None, None],
            "bundle_key": ["||||", "||||"],
        }
    )
    filled, n_changed = enrich_index_with_filename_parse(
        stale,
        config=APOLLO_CONFIG,
        dataset_id="Sedation-RestingState",
    )
    assert n_changed == 2
    assert list(filled["subject"].astype(str)) == ["002", "003"]
    assert filled.iloc[0]["task"] == "anest"
    assert filled.iloc[0]["run"] == "003"
