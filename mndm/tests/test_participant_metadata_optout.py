from pathlib import Path
import sys

import pandas as pd
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.pipeline import extractors


def test_participant_metadata_optout_does_not_read_auto_discovered_tsv(tmp_path: Path, monkeypatch):
    dataset_root = tmp_path / "received" / "ds004100"
    dataset_root.mkdir(parents=True)
    (dataset_root / "participants.tsv").write_text(
        "participant_id\tsex\tage\nsub-HUP060\tF\t42\n", encoding="utf-8"
    )

    def fail_if_read(*_args, **_kwargs):
        raise AssertionError("participants.tsv values must not be read when metadata is disabled")

    monkeypatch.setattr(pd, "read_csv", fail_if_read)
    result = extractors.load_participant_table(
        tmp_path / "received",
        "ds004100",
        {"metadata_extraction": {"datasets": {"ds004100": {"participants": {"enabled": False}}}}},
    )

    assert result is None
