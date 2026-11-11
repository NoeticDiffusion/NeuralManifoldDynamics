"""Coverage and candidate-local null audits for the full-vector C1 runner."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from mndm.mnps.v3_2.experiments import full_vector_challenge_001_gate_c1 as c1


def _fixture_rows():
    participant_id = "sub-001"
    windows = {
        "EO": [{"window_id": "eo-window-001"}],
        "EC": [{"window_id": "ec-window-001"}],
    }
    selection = {
        "eligible_roster": [{"participant_id": participant_id, "all_windows": windows}]
    }
    rows = []
    for condition, condition_windows in windows.items():
        window_id = condition_windows[0]["window_id"]
        for reference_id in c1.CORE_REFERENCE_IDS:
            for family_id in c1.CORE_FAMILIES:
                row = {
                    "participant_id": participant_id,
                    "condition": condition,
                    "window_id": window_id,
                    "reference_id": reference_id,
                    "family_id": family_id,
                    "value": 0.25,
                    "valid": True,
                    "status": "computed_exploratory",
                    "reason": None,
                    "qualified": False,
                }
                if family_id == "AR":
                    row.update({
                        "value": None,
                        "component_names": list(c1.AR_COMPONENT_NAMES),
                        "component_values": [0.1] * 9,
                        "component_status": ["computed_exploratory"] * 9,
                        "component_reasons": [None] * 9,
                    })
                rows.append(row)
        for family_id in c1.BASELINE_FAMILIES:
            rows.append({
                "participant_id": participant_id,
                "condition": condition,
                "window_id": window_id,
                "reference_id": c1.PRIMARY_REFERENCE_ID,
                "family_id": family_id,
                "value": 0.25,
                "valid": True,
                "status": "valid",
                "reason": None,
                "qualified": False,
            })
        rows.append({
            "participant_id": participant_id,
            "condition": condition,
            "window_id": None,
            "reference_id": "separate_scg",
            "family_id": "SCG",
            "value": None,
            "valid": False,
            "status": "unsupported",
            "reason": "electrode_coordinates_unavailable",
            "qualified": False,
        })
    return selection, rows


def test_c1_reports_ar_vector_and_component_coverage() -> None:
    selection, rows = _fixture_rows()

    audit = c1._audit_rows(rows, selection, ["sub-001"])

    assert audit["exact_key_audits_passed"] is True
    assert audit["component_audit_passed"] is True
    assert audit["null_reason_audit_passed"] is True
    ar_coverage = [item for item in audit["coverage"] if item["family_id"] == "AR"]
    assert len(ar_coverage) == 4
    assert all(item["value_representation"] == "ordered_vector" for item in ar_coverage)
    assert all(item["finite_value_count"] == 1 for item in ar_coverage)
    assert len(audit["component_coverage"]) == 2 * len(c1.CORE_REFERENCE_IDS) * 9
    assert all(item["row_count"] == 1 for item in audit["component_coverage"])
    assert all(item["finite_value_count"] == 1 for item in audit["component_coverage"])


def test_c1_keeps_component_local_nulls_with_reasons_and_rejects_missing_reasons() -> None:
    selection, rows = _fixture_rows()
    ar_row = next(row for row in rows if row["family_id"] == "AR")
    ar_row["component_values"][0] = None
    ar_row["component_status"][0] = "unsupported"
    ar_row["component_reasons"][0] = "constant_sensor_vector_correlation_undefined"
    ar_row["valid"] = False
    ar_row["status"] = "partial"
    ar_row["reason"] = "one_or_more_adjacent_pairs_undefined"

    audit = c1._audit_rows(rows, selection, ["sub-001"])

    assert audit["component_audit_passed"] is True
    assert audit["null_reason_audit_passed"] is True
    assert audit["null_reason_audit"]["component_null_count"] == 1
    assert audit["null_reason_audit"]["component_null_missing_reason_count"] == 0

    ar_row["component_reasons"][0] = None
    audit_without_reason = c1._audit_rows(rows, selection, ["sub-001"])
    assert audit_without_reason["component_audit_passed"] is False
    assert audit_without_reason["null_reason_audit_passed"] is False

