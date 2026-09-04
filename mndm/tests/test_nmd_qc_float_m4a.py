from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families.nmd_qc_float_m4a import (  # noqa: E402
    CANDIDATE_MANIFEST_SHA256,
    RULE_MANIFEST_SHA256,
    _export_evidence_summary,
    _fetch_text_evidence,
    _parse_wfdb_header,
    _source_suffix,
    audit_m4a,
)


def test_wfdb_header_parser_uses_metadata_only_fields() -> None:
    header = """0543_006_019_EEG 2 256 921600
0543_006_019_EEG.mat 16+24 18.9(14203)/nu 16 0 13587 -11575 0 C3
0543_006_019_EEG.mat 16+24 19.5(14020)/nu 16 0 13288 -23592 0 C4
#Start time: 19:00:00
#End time: 19:59:59
"""
    summary = _parse_wfdb_header(header)
    assert summary["format_16_present"] is True
    assert summary["units_nu_present"] is True
    assert summary["sampling_frequency"] == 256.0
    assert summary["channel_names"] == ["C3", "C4"]
    assert summary["start_time"] == "19:00:00"


def test_forbidden_signal_payload_is_rejected_before_fetch() -> None:
    result, text = _fetch_text_evidence(
        {
            "kind": "forbidden",
            "url": "https://example.invalid/training/record.mat",
        },
        timeout_seconds=0.1,
        max_bytes=100,
    )
    assert result["status"] == "REJECTED"
    assert result["reason"] == "forbidden_payload"
    assert text == ""


def test_versioned_page_and_export_evidence_are_classified_strictly() -> None:
    assert _source_suffix("https://physionet.org/content/i-care/2.1/") == ""
    assert _source_suffix("https://example.invalid/record.mat") == ".mat"
    summary = _export_evidence_summary(
        [
            (
                {"status": "RETRIEVED"},
                (
                    "Format 16: 16-bit two's complement integers. "
                    "physical = (digital - baseline) / gain"
                ),
            ),
            (
                {"status": "RETRIEVED"},
                "wfdbNaN=-32768",
            ),
        ]
    )
    assert summary["format16_present"] is True
    assert summary["formula_present"] is True
    assert summary["format16_missing_sentinel_present"] is True


def test_m4a_offline_mode_never_opens_samples_or_authorizes_m4b() -> None:
    result = audit_m4a(fetch_remote=False)
    assert result["rule_manifest_sha256"] == RULE_MANIFEST_SHA256
    assert result["candidate_manifest_sha256"] == CANDIDATE_MANIFEST_SHA256
    assert result["signal_samples_opened"] is False
    assert result["native_binary_payloads_opened"] is False
    assert result["nmd_qc_executed_on_real_data"] is False
    assert result["m4b_authorized"] is False
    assert result["ds003670_accessed"] is False
    selected = result["candidates"][0]
    assert selected["independent_truth_surface"]["R1"] == "TRUTH_UNRESOLVED"
    assert selected["independent_truth_surface"]["R5"] == "TRUTH_KNOWN"
    assert selected["independent_truth_surface"]["R8"] == "TRUTH_KNOWN"
