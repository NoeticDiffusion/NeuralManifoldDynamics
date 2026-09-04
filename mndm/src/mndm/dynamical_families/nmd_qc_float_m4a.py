"""Documentation-only M4A reference-semantics audit.

M4A may read text documentation, metadata, and source specifications.  It
must never open waveform payloads or call the real-signal QC implementation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


PROTOCOL_ID = "NMD-QC-FLOAT-001"
GATE_ID = "M4A"
RULE_MANIFEST_SHA256 = (
    "054394322bad951a0a7e245be657fe6842ba4f3fc5807787c2e1e47f5592a6fa"
)
CANDIDATE_MANIFEST_RELATIVE = (
    "project/orthagonal_axis/orthagonal_dynamics/nmd-qc-float/"
    "NMD-QC-FLOAT-M4A-CANDIDATES-0.1.json"
)
CANDIDATE_MANIFEST_SHA256 = (
    "b251bb1e3c7e01dd6dacb9ef7dba1b0ac016ada77903cd2c0802f162efb1cff3"
)

REFERENCE_SEMANTICS_PASS = "REFERENCE_SEMANTICS_PASS"
REFERENCE_PARTIAL = "REFERENCE_PARTIAL"
SOURCE_UNCERTAIN = "SOURCE_UNCERTAIN"
NOT_TESTABLE = "NOT_TESTABLE"

_ALLOWED_SUFFIXES = {
    "",
    ".csv",
    ".hea",
    ".html",
    ".json",
    ".m",
    ".md",
    ".py",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
_FORBIDDEN_SUFFIXES = {
    ".bdf",
    ".dat",
    ".edf",
    ".fdt",
    ".fif",
    ".mat",
    ".nwb",
    ".set",
}


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _source_suffix(value: str) -> str:
    path = urlparse(value).path
    if path.endswith("/"):
        return ""
    return Path(path).suffix.lower()


def _evidence_error(
    reference: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    output = {
        "kind": reference.get("kind"),
        "metadata_only": True,
        "status": "REJECTED" if reason == "forbidden_payload" else "UNAVAILABLE",
        "reason": reason,
    }
    if "url" in reference:
        output["url"] = reference["url"]
    if "path" in reference:
        output["path"] = reference["path"]
    return output


def _fetch_text_evidence(
    reference: Mapping[str, Any],
    *,
    timeout_seconds: float,
    max_bytes: int,
) -> tuple[dict[str, Any], str]:
    url = str(reference.get("url", ""))
    suffix = _source_suffix(url)
    if suffix in _FORBIDDEN_SUFFIXES:
        return _evidence_error(reference, "forbidden_payload"), ""
    if reference.get("allow_signal_samples") is not False:
        return _evidence_error(reference, "metadata_only_boundary_not_declared"), ""
    if suffix not in _ALLOWED_SUFFIXES:
        return _evidence_error(reference, "suffix_not_allowlisted"), ""
    request = Request(
        url,
        headers={"User-Agent": "NMD-QC-FLOAT-001-M4A/0.1"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            advertised = response.headers.get("Content-Length")
            if advertised is not None and int(advertised) > max_bytes:
                return _evidence_error(reference, "metadata_payload_too_large"), ""
            payload = response.read(max_bytes + 1)
    except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
        return (
            {
                **_evidence_error(reference, "metadata_fetch_failed"),
                "error_type": type(exc).__name__,
            },
            "",
        )
    if len(payload) > max_bytes:
        return _evidence_error(reference, "metadata_payload_too_large"), ""
    text = payload.decode("utf-8", errors="replace")
    return (
        {
            "kind": reference.get("kind"),
            "url": url,
            "metadata_only": True,
            "status": "RETRIEVED",
            "bytes": len(payload),
            "sha256": _sha256(payload),
        },
        text,
    )


def _read_local_evidence(
    reference: Mapping[str, Any],
    *,
    repository_root: Path,
    max_bytes: int,
) -> tuple[dict[str, Any], str]:
    relative_path = str(reference.get("path", ""))
    suffix = Path(relative_path).suffix.lower()
    if suffix in _FORBIDDEN_SUFFIXES:
        return _evidence_error(reference, "forbidden_payload"), ""
    if reference.get("allow_signal_samples") is not False:
        return _evidence_error(reference, "metadata_only_boundary_not_declared"), ""
    if suffix not in _ALLOWED_SUFFIXES:
        return _evidence_error(reference, "suffix_not_allowlisted"), ""
    path = (repository_root / relative_path).resolve()
    try:
        path.relative_to(repository_root.resolve())
    except ValueError:
        return _evidence_error(reference, "path_outside_repository"), ""
    if not path.is_file():
        return _evidence_error(reference, "local_metadata_missing"), ""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        return (
            {
                **_evidence_error(reference, "local_metadata_read_failed"),
                "error_type": type(exc).__name__,
            },
            "",
        )
    if len(payload) > max_bytes:
        return _evidence_error(reference, "metadata_payload_too_large"), ""
    return (
        {
            "kind": reference.get("kind"),
            "path": relative_path,
            "metadata_only": True,
            "status": "RETRIEVED",
            "bytes": len(payload),
            "sha256": _sha256(payload),
        },
        payload.decode("utf-8", errors="replace"),
    )


def _parse_wfdb_header(text: str) -> dict[str, Any]:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        return {"status": "UNRESOLVED", "reason": "HEADER_EMPTY"}
    record_tokens = lines[0].split()
    if len(record_tokens) < 4:
        return {"status": "UNRESOLVED", "reason": "HEADER_RECORD_LINE_INVALID"}
    try:
        channel_count = int(record_tokens[1])
        sampling_frequency = float(record_tokens[2])
        sample_count = int(record_tokens[3])
    except (TypeError, ValueError):
        return {"status": "UNRESOLVED", "reason": "HEADER_RECORD_FIELDS_INVALID"}

    signal_lines = lines[1:]
    formats: list[str] = []
    gains: list[float] = []
    baselines: list[float] = []
    units: list[str] = []
    channels: list[str] = []
    for line in signal_lines:
        tokens = line.split()
        if len(tokens) < 9:
            continue
        formats.append(tokens[1])
        gain_match = re.match(
            r"(?P<gain>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
            r"(?:\((?P<baseline>[-+]?\d+(?:\.\d*)?)\))?/"
            r"(?P<unit>\S+)",
            tokens[2],
        )
        if gain_match:
            gains.append(float(gain_match.group("gain")))
            baseline = gain_match.group("baseline")
            if baseline is not None:
                baselines.append(float(baseline))
            units.append(gain_match.group("unit"))
        channels.append(tokens[-1])
    return {
        "status": "PASS",
        "record_name": record_tokens[0],
        "channel_count": channel_count,
        "sampling_frequency": sampling_frequency,
        "sample_count": sample_count,
        "signal_line_count": len(signal_lines),
        "formats": sorted(set(formats)),
        "gain_count": len(gains),
        "baseline_count": len(baselines),
        "units": sorted(set(units)),
        "channel_names": channels,
        "start_time": next(
            (
                line.split(":", 1)[1].strip()
                for line in text.splitlines()
                if line.lower().startswith("#start time:")
            ),
            None,
        ),
        "end_time": next(
            (
                line.split(":", 1)[1].strip()
                for line in text.splitlines()
                if line.lower().startswith("#end time:")
            ),
            None,
        ),
        "format_16_present": any(value.startswith("16") for value in formats),
        "units_nu_present": "nu" in {value.lower() for value in units},
    }


def _export_evidence_summary(evidence_text: list[tuple[dict[str, Any], str]]) -> dict[str, Any]:
    combined = "\n".join(text for _, text in evidence_text).lower()
    normalized = re.sub(r"\s+", "", combined)
    formula_present = (
        "(val(m,:)-siginfo(m).baseline)/siginfo(m).gain" in normalized
        or "(digital-baseline)/gain" in normalized
        or "digital-baseline/adc_gain" in normalized
    )
    sentinel_present = (
        "wfdbnan=-32768" in normalized
        or "format16" in normalized and "32768" in normalized
    )
    format16_present = (
        "16-bit two" in combined
        or "format 16" in combined
        or "16+24" in combined
    )
    gain_baseline_present = "gain" in combined and "baseline" in combined
    return {
        "formula_present": formula_present,
        "format16_present": format16_present,
        "gain_baseline_present": gain_baseline_present,
        "format16_missing_sentinel_present": sentinel_present,
    }


def _candidate_rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    ranking = candidate.get("ranking", {})
    return (
        -int(ranking.get("native_representation", 0)),
        -int(ranking.get("export_transform", 0)),
        -int(ranking.get("independent_truth_surface", 0)),
        -int(ranking.get("reproducibility", 0)),
        int(ranking.get("data_burden", 0)),
        str(candidate.get("source_id", "")),
    )


def _audit_candidate(
    candidate: Mapping[str, Any],
    *,
    repository_root: Path,
    fetch_remote: bool,
) -> dict[str, Any]:
    evidence_results: list[dict[str, Any]] = []
    evidence_text: list[tuple[dict[str, Any], str]] = []
    for reference in candidate.get("evidence", []):
        if "url" in reference:
            if not fetch_remote:
                evidence_results.append(
                    _evidence_error(reference, "remote_fetch_disabled")
                )
                continue
            result, text = _fetch_text_evidence(
                reference,
                timeout_seconds=20.0,
                max_bytes=250_000,
            )
        else:
            result, text = _read_local_evidence(
                reference,
                repository_root=repository_root,
                max_bytes=250_000,
            )
        evidence_results.append(result)
        if text:
            evidence_text.append((result, text))

    summary: dict[str, Any] = {
        "source_id": candidate.get("source_id"),
        "reference_id": candidate.get("reference_id"),
        "dataset_version": candidate.get("dataset_version"),
        "independent_of_ds003670": bool(
            candidate.get("independent_of_ds003670")
        ),
        "public_access": bool(candidate.get("public_access")),
        "evidence": evidence_results,
        "independent_truth_surface": candidate.get(
            "independent_truth_surface",
            {},
        ),
        "manifest_declared_status": candidate.get(
            "declared_status",
            "UNDECLARED",
        ),
    }

    if candidate.get("source_id") != "physionet_icare_2_1":
        summary["status"] = (
            REFERENCE_PARTIAL
            if candidate.get("native_representation", {}).get("status")
            == REFERENCE_PARTIAL
            else SOURCE_UNCERTAIN
        )
        summary["reason"] = "candidate_not_selected_after_frozen_ranking"
        return summary

    header_summaries = [
        _parse_wfdb_header(text)
        for result, text in evidence_text
        if str(result.get("url", "")).lower().endswith(".hea")
    ]
    export_summary = _export_evidence_summary(evidence_text)
    summary["header_summaries"] = header_summaries
    summary["export_evidence"] = export_summary
    retrieved = [
        result for result in evidence_results
        if result.get("status") == "RETRIEVED"
    ]
    critical_urls = {
        str(reference.get("url"))
        for reference in candidate.get("evidence", [])
        if reference.get("url")
    }
    retrieved_urls = {
        str(result.get("url"))
        for result in retrieved
    }
    header_ok = any(
        header.get("status") == "PASS"
        and header.get("format_16_present")
        and header.get("sampling_frequency", 0) > 0
        and header.get("channel_count", 0) > 0
        and header.get("sample_count", 0) > 0
        and header.get("signal_line_count")
        == header.get("channel_count")
        and header.get("gain_count")
        == header.get("channel_count")
        and header.get("baseline_count")
        == header.get("channel_count")
        and header.get("units_nu_present")
        for header in header_summaries
    )
    landing_ok = any(
        str(result.get("url", "")).rstrip("/").endswith(
            "physionet.org/content/i-care/2.1"
        )
        and result.get("status") == "RETRIEVED"
        for result in evidence_results
    )
    format_ok = export_summary["format16_present"]
    transform_ok = export_summary["formula_present"]
    sentinel_ok = export_summary["format16_missing_sentinel_present"]
    license_ok = bool(candidate.get("license")) and bool(
        candidate.get("public_access")
    )
    meaningful_truth = {
        key: value
        for key, value in candidate.get("independent_truth_surface", {}).items()
        if value == "TRUTH_KNOWN"
    }
    critical_evidence_complete = critical_urls.issubset(retrieved_urls)
    summary["critical_evidence_complete"] = critical_evidence_complete
    if (
        summary["independent_of_ds003670"]
        and header_ok
        and landing_ok
        and format_ok
        and transform_ok
        and sentinel_ok
        and license_ok
        and {"R5", "R8"}.issubset(meaningful_truth)
        and critical_evidence_complete
    ):
        summary["status"] = REFERENCE_SEMANTICS_PASS
        summary["reason"] = "all_m4a_pass_criteria_met"
    elif header_ok or transform_ok:
        summary["status"] = REFERENCE_PARTIAL
        summary["reason"] = "useful_evidence_but_critical_layer_unresolved"
    else:
        summary["status"] = SOURCE_UNCERTAIN
        summary["reason"] = "native_or_export_semantics_not_bound"
    return summary


def audit_m4a(
    *,
    candidate_manifest_path: Path | None = None,
    fetch_remote: bool = True,
) -> dict[str, Any]:
    """Run the documentation-only M4A audit."""
    repository_root = _repository_root()
    manifest_path = candidate_manifest_path or (
        repository_root / CANDIDATE_MANIFEST_RELATIVE
    )
    payload = manifest_path.read_bytes()
    manifest_hash = _sha256(payload)
    manifest = json.loads(payload.decode("utf-8"))
    if manifest_hash != CANDIDATE_MANIFEST_SHA256:
        raise ValueError("M4A candidate manifest hash mismatch")
    if manifest.get("rule_manifest_sha256") != RULE_MANIFEST_SHA256:
        raise ValueError("M4A rule manifest binding mismatch")

    candidates = list(manifest.get("candidates", []))
    ordered = sorted(candidates, key=_candidate_rank)
    candidate_audits = [
        _audit_candidate(
            candidate,
            repository_root=repository_root,
            fetch_remote=fetch_remote,
        )
        for candidate in ordered
    ]
    selected = candidate_audits[0] if candidate_audits else None
    global_status = selected["status"] if selected else NOT_TESTABLE
    result = {
        "protocol_id": PROTOCOL_ID,
        "gate_id": GATE_ID,
        "scope": "documentation_and_metadata_only",
        "rule_manifest_sha256": RULE_MANIFEST_SHA256,
        "candidate_manifest_path": str(
            manifest_path.relative_to(repository_root)
        ),
        "candidate_manifest_sha256": manifest_hash,
        "signal_samples_opened": False,
        "native_binary_payloads_opened": False,
        "nmd_qc_executed_on_real_data": False,
        "ds003670_accessed": False,
        "selected_candidate": (
            selected.get("source_id") if selected else None
        ),
        "selected_reference_id": (
            selected.get("reference_id") if selected else None
        ),
        "global_status": global_status,
        "m4b_authorized": False,
        "candidates": candidate_audits,
        "closures": {
            "M4B": "NOT_AUTHORIZED",
            "ds003670": "FORBIDDEN",
            "FAR-EXT-002B": "PERMANENT_NOT_TESTABLE",
            "FAR-EXT-002D": "NOT_AUTHORIZED",
            "MNPS": "NOT_AUTHORIZED",
            "FAR": "NOT_AUTHORIZED",
            "FAR-003B": "NOT_AUTHORIZED",
            "reconciliation": "NOT_AUTHORIZED",
        },
    }
    result["audit_hash"] = _sha256(_canonical_json(result))
    return result
