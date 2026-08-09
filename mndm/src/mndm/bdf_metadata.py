"""Build privacy-safe participant/session tables for mapped non-BIDS BDF data."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _subject_session_tokens(value: str) -> tuple[Optional[str], Optional[str]]:
    """Extract BIDS subject/session labels from one combined mapping value."""
    match = re.search(r"sub-(?P<subject>[^_]+)_ses-(?P<session>[^_]+)", str(value))
    if not match:
        return None, None
    return f"sub-{match.group('subject')}", f"ses-{match.group('session')}"


def _read_meas_date(path: Path) -> Optional[pd.Timestamp]:
    """Read a BDF acquisition timestamp without persisting it to outputs."""
    try:
        import mne

        raw = mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
        value = raw.info.get("meas_date")
        if value is None:
            return None
        return pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo else pd.Timestamp(value)
    except Exception as exc:
        logger.warning("Cannot read BDF measurement timestamp for %s: %s", path.name, exc)
        return None


def _first_nonempty(frame: pd.DataFrame, column: str) -> Any:
    """Return the first non-null table value for a column."""
    if column not in frame.columns:
        return None
    values = frame[column].dropna()
    return values.iloc[0] if not values.empty else None


def _sha256(path: Path) -> str:
    """Return SHA-256 for a small source metadata file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_bdf_metadata_tables(
    *,
    raw_dir: Path,
    mapping_path: Path,
    outcomes_path: Path,
    output_dir: Path,
    mapping_key_column: str = "old_id",
    mapping_subject_session_column: str = "subject_id",
    outcome_group_column: str = "Group",
    outcome_group_value: str = "TD",
    outcome_subject_column: str = "Infant",
    outcome_visit_column: str = "Visit",
    age_column: str = "Chronological Age (days)",
    exclude_recordings: tuple[str, ...] = ("TD25v4",),
) -> Dict[str, Any]:
    """Create BIDS-like metadata without exposing acquisition dates or DOB.

    Direct outcome rows are joined by source subject/visit.  Missing session
    ages are derived internally from the first direct age and BDF measurement
    timestamps.  Only resulting age-in-days and an ``age_source`` label are
    exported.
    """
    mapping = pd.read_csv(mapping_path)
    outcomes = pd.read_excel(outcomes_path)
    required_mapping = {mapping_key_column, mapping_subject_session_column}
    if missing := required_mapping.difference(mapping.columns):
        raise ValueError(f"Mapping table is missing columns: {sorted(missing)}")
    required_outcomes = {outcome_group_column, outcome_subject_column, outcome_visit_column, age_column}
    if missing := required_outcomes.difference(outcomes.columns):
        raise ValueError(f"Outcome sheet is missing columns: {sorted(missing)}")

    outcomes = outcomes.loc[outcomes[outcome_group_column].astype(str) == str(outcome_group_value)].copy()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Dict[str, Any]] = []
    timestamp_by_old_id: Dict[str, Optional[pd.Timestamp]] = {}
    for _, mapped in mapping.iterrows():
        old_id = str(mapped[mapping_key_column])
        participant_id, session_id = _subject_session_tokens(str(mapped[mapping_subject_session_column]))
        raw_path = raw_dir / f"{old_id}.bdf"
        timestamp = _read_meas_date(raw_path) if raw_path.exists() else None
        timestamp_by_old_id[old_id] = timestamp
        source_subject = old_id.split("v", maxsplit=1)[0]
        source_visit_match = re.search(r"v(?P<visit>\d+)", old_id)
        source_visit = int(source_visit_match.group("visit")) if source_visit_match else None
        direct = outcomes.loc[
            (outcomes[outcome_subject_column].astype(str) == source_subject)
            & (pd.to_numeric(outcomes[outcome_visit_column], errors="coerce") == source_visit)
        ]
        rows.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "original_bdf_id": old_id,
                "source_subject_id": source_subject,
                "visit": source_visit,
                "_timestamp": timestamp,
                "_direct": direct,
                "recording_timestamp_available": bool(timestamp is not None),
                "excluded_from_age_analysis": old_id in exclude_recordings,
                "exclusion_reason": "insufficient_age_provenance" if old_id in exclude_recordings else "",
            }
        )

    # Estimate DOB only in memory.  Any direct age/timestamp pair can anchor a
    # subject (some raw archives lack visit 1 but retain visit 3); it is
    # intentionally never added to any table or JSON payload.
    dob_by_subject: Dict[str, pd.Timestamp] = {}
    for row in rows:
        direct = row["_direct"]
        age = _first_nonempty(direct, age_column)
        if (
            row["_timestamp"] is not None
            and age is not None
            and pd.notna(age)
            and row["source_subject_id"] not in dob_by_subject
        ):
            dob_by_subject[row["source_subject_id"]] = row["_timestamp"] - pd.to_timedelta(int(age), unit="D")

    outcome_columns = {
        "sex": "Gender",
        "bayley_raw_fine_motor": "Bayley raw fine motor",
        "bayley_raw_gross_motor": "Bayley raw gross motor",
        "bayley_raw_receptive_language": "Bayley raw receptive language",
        "bayley_raw_expressive_language": "Bayley raw expressive language",
        "bayley_raw_cognitive": "Bayley raw cognitive",
        "bayley_total_raw_score": "Bayley total raw score",
    }
    session_rows: list[Dict[str, Any]] = []
    for row in rows:
        direct = row.pop("_direct")
        timestamp = row.pop("_timestamp")
        direct_age = _first_nonempty(direct, age_column)
        if row["excluded_from_age_analysis"]:
            age_days = None
            age_source = "unresolved"
        elif direct_age is not None and pd.notna(direct_age):
            age_days: Optional[int] = int(direct_age)
            age_source = "spreadsheet_direct"
        elif timestamp is not None and row["source_subject_id"] in dob_by_subject:
            age_days = int((timestamp - dob_by_subject[row["source_subject_id"]]).days)
            age_source = "derived_from_first_visit_age_and_meas_date"
        else:
            age_days = None
            age_source = "unresolved"
        row["age_days"] = age_days
        row["age_source"] = age_source
        for exported, source in outcome_columns.items():
            value = _first_nonempty(direct, source)
            row[exported] = value if value is not None and pd.notna(value) else None
        session_rows.append(row)

    sessions = pd.DataFrame(session_rows)
    sessions = sessions.drop(columns=["source_subject_id"])
    sessions["days_from_first_session"] = (
        sessions.groupby("participant_id", dropna=False)["age_days"].transform(
            lambda series: series - series.dropna().min() if series.notna().any() else series
        )
    )
    sessions.to_csv(output_dir / "sessions.tsv", sep="\t", index=False)

    participant_rows: list[Dict[str, Any]] = []
    for participant_id, group in sessions.groupby("participant_id", dropna=False):
        participant_rows.append(
            {
                "participant_id": participant_id,
                "group": outcome_group_value,
                "sex": _first_nonempty(group, "sex"),
                "n_recordings": int(len(group)),
                "n_age_resolved": int(group["age_days"].notna().sum()),
            }
        )
    participants = pd.DataFrame(participant_rows).sort_values("participant_id")
    participants.to_csv(output_dir / "participants.tsv", sep="\t", index=False)

    qc = {
        "schema": "mndm.non_bids_bdf_metadata_qc.v1",
        "raw_recordings": int(len(mapping)),
        "participants": int(participants["participant_id"].nunique()),
        "age_resolved": int(sessions["age_days"].notna().sum()),
        "age_unresolved": int(sessions["age_days"].isna().sum()),
        "excluded_from_age_analysis": sessions.loc[
            sessions["excluded_from_age_analysis"], "original_bdf_id"
        ].astype(str).tolist(),
        "source_files": {
            "mapping": {"path": str(mapping_path), "sha256": _sha256(mapping_path)},
            "outcomes": {"path": str(outcomes_path), "sha256": _sha256(outcomes_path)},
        },
        "privacy": {
            "exports_recording_timestamps": False,
            "exports_reconstructed_dob": False,
        },
    }
    (output_dir / "mapping_qc.json").write_text(json.dumps(qc, indent=2), encoding="utf-8")
    return qc


def main() -> None:
    """Run the metadata builder from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = build_bdf_metadata_tables(
        raw_dir=args.raw_dir,
        mapping_path=args.mapping,
        outcomes_path=args.outcomes,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
