"""Source-only FAR-EXT-002E post-ramp interpretation bridge.

This module binds the frozen FAR-EXT-002A/002D certificates and audits public
documentation and source code for the F30 post-stimulation interval.  It never
opens EEG, FDT, CNT, NMD, or outcome payloads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import copy
import hashlib
import html
import json
from pathlib import Path
import re
from typing import Any, Mapping
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET


PROTOCOL_ID = "FAR-EXT-002E"
DATASET_ID = "ds003670"
F30_FAMILY_ID = "frontal_sinusoidal_30hz"
F30_PUBLISHED_POST_SEC = 30.0
F30_TECHNICAL_HORIZON_SEC = 60.0
DIRECT_TRANSFER_RHO = 1.0
UNPROVEN_TRANSFER_RHO = 0.5

POST_RAMP_INTERPRETABILITY_BRIDGE_PASS = (
    "POST_RAMP_INTERPRETABILITY_BRIDGE_PASS"
)
POST_RAMP_METHOD_LIMITED = "POST_RAMP_METHOD_LIMITED"
POST_RAMP_SOURCE_MISMATCH = "POST_RAMP_SOURCE_MISMATCH"
POST_RAMP_INTERPRETABILITY_UNRESOLVED = (
    "POST_RAMP_INTERPRETABILITY_UNRESOLVED"
)

_MAX_SOURCE_BYTES = 16 * 1024 * 1024
_FORBIDDEN_SUFFIXES = {
    ".bdf",
    ".cnt",
    ".dat",
    ".edf",
    ".fdt",
    ".fif",
    ".mat",
    ".nwb",
    ".set",
}


SOURCE_DOCUMENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "gx_dataset_descriptor",
        "kind": "published_dataset",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8551279/",
        "allow_signal_samples": False,
        "evidence_targets": [
            "same GX acquisition supports Experiment 1 and Experiment 2",
            "30-second stimulation plus 5-second ramp-down",
            "Experiment 2 received 1 mA",
            "trigger 16 and trigger 32 semantics",
        ],
        "required_markers": [
            "experiment 1 and 2",
            "30-sec stimulation period",
            "5-sec ramp down period",
            "all participants in experiment 2 received 1 ma",
            "stimulation start trigger (code:16)",
            "stimulation stop trigger (code:32)",
        ],
        "role": "acquisition_and_trigger_provenance",
    },
    {
        "id": "f30_followup_paper",
        "kind": "published_followup",
        "url": "https://doi.org/10.1016/j.brs.2024.05.009",
        "allow_signal_samples": False,
        "evidence_targets": [
            "F30 Experiment 2 analysis",
            "immediately post 30-second EEG interval",
            "during-stimulation exclusion because of nonlinear artifacts",
            "source-defined EEG preprocessing",
        ],
        "required_markers": [
            "immediately post",
            "30 sec",
            "not examined",
            "nonlinear artifacts",
            "0.35-40 hz",
            "bandpass filtered",
        ],
        "role": "published_f30_post_ramp_precedent",
    },
    {
        "id": "gx_trial_extraction_code",
        "kind": "source_code",
        "url": (
            "https://raw.githubusercontent.com/ngebodh/"
            "GX_tES_EEG_Physio_Behavior/"
            "358804f5b8b47f4753a51fc9990f60d8f37a5b27/"
            "GX_PullingDataIntoTrials_PlottingTopoplots.m"
        ),
        "revision": "358804f5b8b47f4753a51fc9990f60d8f37a5b27",
        "allow_signal_samples": False,
        "evidence_targets": [
            "F30 post-trial extraction",
            "source-defined delay after ramp-down",
            "30-second post interval",
            "native EEG channel subset",
        ],
        "required_markers": [
            "eegout.poststim",
            "41.5*fs",
            "41.5+30",
            "spectral lekage",
            "1:32",
        ],
        "role": "published_trial_extraction_implementation",
    },
    {
        "id": "f30_followup_code_revision",
        "kind": "source_code_revision",
        "url": (
            "https://github.com/ngebodh/"
            "GX_tES_EEG_Physio_Behavior/commit/"
            "7bee67e915da44143773e715ee6faa918c3ce766"
        ),
        "revision": "7bee67e915da44143773e715ee6faa918c3ce766",
        "allow_signal_samples": False,
        "evidence_targets": [
            "2024 F30/M30 analysis provenance",
            "Experiment 2 analysis code and processed-output commit identity",
        ],
        "required_markers": [
            "7bee67e915da44143773e715ee6faa918c3ce766",
            "f30",
            "experiment 2",
        ],
        "role": "followup_code_provenance",
    },
    {
        "id": "tes_artifact_context",
        "kind": "artifact_context",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6289749/",
        "allow_signal_samples": False,
        "evidence_targets": [
            "general tES-related EEG artifact mechanisms",
        ],
        "required_markers": [
            "artifact",
            "tdcs",
            "eeg",
        ],
        "role": "context_only_not_an_eligibility_rule",
    },
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _load_json_object(path: Path, reason: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(reason) from exc
    if not isinstance(value, dict):
        raise RuntimeError(reason)
    return value


def load_scope(
    path: Path,
    *,
    expected_scope_version: str | None = None,
) -> tuple[dict[str, Any], str]:
    scope = _load_json_object(path, "far_ext_002e_scope_unreadable")
    if scope.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("far_ext_002e_scope_protocol_mismatch")
    scope_version = str(scope.get("scope_version", "0.1"))
    if expected_scope_version is not None and scope_version != (
        expected_scope_version
    ):
        raise ValueError("far_ext_002e_scope_version_mismatch")
    if scope.get("execution_status") != "PREREGISTRATION_FROZEN":
        raise ValueError("far_ext_002e_scope_not_frozen")
    source_documents = scope.get(
        "source_objects" if scope_version == "0.2" else "source_documents"
    )
    if not isinstance(source_documents, list) or not source_documents:
        raise ValueError("far_ext_002e_source_documents_missing")
    return scope, sha256_file(path)


def canonical_f30_eligible_subset(
    result: Mapping[str, Any],
    *,
    rho: float | None = None,
) -> tuple[list[dict[str, Any]], str]:
    table = result.get("exported_signal_technical_status", {}).get(
        "event_horizon_status_table",
        [],
    )
    subset = [
        {
            "event_key": row["event_key"],
            "rho": float(row["rho"]),
            "horizon_sec": float(row["horizon_sec"]),
            "eligible": bool(row["eligible"]),
            "technical_eligible": bool(row["technical_eligible"]),
            "reason": str(row["reason"]),
        }
        for row in table
        if isinstance(row, Mapping)
        and row.get("family_id") == F30_FAMILY_ID
        and float(row.get("horizon_sec", -1.0))
        == F30_TECHNICAL_HORIZON_SEC
        and bool(row.get("eligible"))
        and (
            rho is None
            or float(row.get("rho", float("nan"))) == float(rho)
        )
    ]
    return subset, _json_hash(subset)


def bind_002d(
    *,
    result_path: Path,
    expected_sha256: str,
    expected_subset_sha256: str,
    expected_subset_count: int,
    expected_direct_subset_sha256: str | None = None,
    expected_direct_subset_count: int | None = None,
    expected_status: str = "SIGNAL_TIMEBASE_PASS",
) -> dict[str, Any]:
    if not result_path.is_file():
        return {
            "status": POST_RAMP_SOURCE_MISMATCH,
            "reason": "far_ext_002d_result_missing",
            "path": str(result_path),
        }
    actual_sha256 = sha256_file(result_path)
    try:
        result = _load_json_object(
            result_path,
            "far_ext_002d_result_unreadable",
        )
    except RuntimeError as exc:
        return {
            "status": POST_RAMP_SOURCE_MISMATCH,
            "reason": str(exc),
            "path": str(result_path),
            "sha256": actual_sha256,
        }
    subset, subset_sha256 = canonical_f30_eligible_subset(result)
    direct_subset, direct_subset_sha256 = canonical_f30_eligible_subset(
        result,
        rho=DIRECT_TRANSFER_RHO,
    )
    checks = {
        "protocol_id": result.get("protocol_id") == "FAR-EXT-002D",
        "dataset_id": result.get("dataset_id") == DATASET_ID,
        "global_status": result.get("global_status") == expected_status,
        "primary_family_id": result.get("primary_family_id")
        == F30_FAMILY_ID,
        "result_sha256": actual_sha256 == expected_sha256,
        "f30_subset_sha256": subset_sha256 == expected_subset_sha256,
        "f30_subset_count": len(subset) == expected_subset_count,
    }
    if expected_direct_subset_sha256 is not None:
        checks["f30_direct_subset_sha256"] = (
            direct_subset_sha256 == expected_direct_subset_sha256
        )
    if expected_direct_subset_count is not None:
        checks["f30_direct_subset_count"] = (
            len(direct_subset) == expected_direct_subset_count
        )
    return {
        "status": "PASS" if all(checks.values()) else POST_RAMP_SOURCE_MISMATCH,
        "reason": (
            "frozen_002d_and_f30_subset_bound"
            if all(checks.values())
            else "far_ext_002d_binding_mismatch"
        ),
        "path": str(result_path),
        "sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "f30_subset_sha256": subset_sha256,
        "expected_f30_subset_sha256": expected_subset_sha256,
        "f30_subset_count": len(subset),
        "expected_f30_subset_count": expected_subset_count,
        "f30_direct_subset_sha256": direct_subset_sha256,
        "expected_f30_direct_subset_sha256": expected_direct_subset_sha256,
        "f30_direct_subset_count": len(direct_subset),
        "expected_f30_direct_subset_count": expected_direct_subset_count,
        "checks": checks,
        "f30_subset": subset,
        "f30_direct_subset": direct_subset,
    }


def bind_inherited_contract(
    *,
    scope: Mapping[str, Any],
    repo_root: Path,
    result_path: Path,
) -> dict[str, Any]:
    """Bind every inherited artifact named by the frozen 002E scope."""
    inherited_002a = scope["inherited_002a"]
    inherited_002d = scope["inherited_002d"]
    result_binding = bind_002d(
        result_path=result_path,
        expected_sha256=str(inherited_002d["result_sha256"]),
        expected_subset_sha256=str(inherited_002d["f30_subset_sha256"]),
        expected_subset_count=int(inherited_002d["f30_subset_count"]),
        expected_direct_subset_sha256=str(
            inherited_002d["f30_direct_subset_sha256"]
        ),
        expected_direct_subset_count=int(
            inherited_002d["f30_direct_subset_count"]
        ),
    )

    certificate_path = repo_root / Path(
        str(inherited_002a["certificate_path"])
    )
    certificate_checks = {
        "exists": certificate_path.is_file(),
        "sha256": False,
        "protocol_id": False,
        "ledger_sha256": False,
        "promoted_event_count": False,
    }
    certificate_sha256: str | None = None
    ledger_sha256: str | None = None
    try:
        if certificate_path.is_file():
            certificate_sha256 = sha256_file(certificate_path)
            certificate = _load_json_object(
                certificate_path,
                "far_ext_002a_certificate_unreadable",
            )
            from mndm.dynamical_families.far_ext_002d_ds003670_exported_signal import (  # noqa: PLC0415
                canonical_002a_promoted_events,
            )

            promoted, ledger_sha256 = canonical_002a_promoted_events(certificate)
            certificate_checks.update(
                {
                    "sha256": certificate_sha256
                    == str(inherited_002a["certificate_sha256"]),
                    "protocol_id": certificate.get("protocol_id")
                    == "FAR-EXT-002A",
                    "ledger_sha256": ledger_sha256
                    == str(inherited_002a["promoted_event_ledger_sha256"]),
                    "promoted_event_count": len(promoted)
                    == int(inherited_002a["promoted_event_count"]),
                }
            )
    except (ImportError, OSError, RuntimeError, ValueError, TypeError):
        pass

    scope_path = repo_root / Path(str(inherited_002d["result_scope_path"]))
    scope_checks = {
        "exists": scope_path.is_file(),
        "sha256": False,
        "protocol_id": False,
        "execution_status": False,
    }
    scope_sha256: str | None = None
    try:
        if scope_path.is_file():
            scope_sha256 = sha256_file(scope_path)
            bound_scope = _load_json_object(
                scope_path,
                "far_ext_002d_scope_unreadable",
            )
            scope_checks.update(
                {
                    "sha256": scope_sha256
                    == str(inherited_002d["result_scope_sha256"]),
                    "protocol_id": bound_scope.get("protocol_id")
                    == "FAR-EXT-002D",
                    "execution_status": bound_scope.get("execution_status")
                    == "PREREGISTRATION_FROZEN",
                }
            )
    except (OSError, RuntimeError, ValueError, TypeError):
        pass

    protocol_path = repo_root / Path(
        str(inherited_002d["result_protocol_path"])
    )
    protocol_checks = {
        "exists": protocol_path.is_file(),
        "sha256": False,
        "protocol_id": False,
    }
    protocol_sha256: str | None = None
    try:
        if protocol_path.is_file():
            protocol_sha256 = sha256_file(protocol_path)
            protocol_text = protocol_path.read_text(encoding="utf-8")
            protocol_checks.update(
                {
                    "sha256": protocol_sha256
                    == str(inherited_002d["result_protocol_sha256"]),
                    "protocol_id": PROTOCOL_ID.replace("002E", "002D")
                    in protocol_text,
                }
            )
    except (OSError, UnicodeError, ValueError):
        pass

    qc_path = repo_root / Path(str(inherited_002d["qc_rules_path"]))
    qc_checks = {
        "exists": qc_path.is_file(),
        "sha256": False,
    }
    qc_sha256: str | None = None
    if qc_path.is_file():
        try:
            qc_sha256 = sha256_file(qc_path)
        except OSError:
            qc_sha256 = None
        qc_checks["sha256"] = qc_sha256 == str(
            inherited_002d["qc_rules_sha256"]
        )

    checks: dict[str, bool] = {
        "result": result_binding["status"] == "PASS",
        **{
            f"002a_{key}": bool(value)
            for key, value in certificate_checks.items()
        },
        **{
            f"002d_scope_{key}": bool(value)
            for key, value in scope_checks.items()
        },
        **{
            f"002d_protocol_{key}": bool(value)
            for key, value in protocol_checks.items()
        },
        **{f"qc_rules_{key}": bool(value) for key, value in qc_checks.items()},
    }
    return {
        "status": "PASS" if all(checks.values()) else POST_RAMP_SOURCE_MISMATCH,
        "reason": (
            "all_inherited_artifacts_bound"
            if all(checks.values())
            else "inherited_artifact_binding_mismatch"
        ),
        "checks": checks,
        "result": result_binding,
        "002a": {
            **certificate_checks,
            "path": str(certificate_path),
            "sha256": certificate_sha256,
            "ledger_sha256": ledger_sha256,
        },
        "002d_scope": {
            **scope_checks,
            "path": str(scope_path),
            "sha256": scope_sha256,
        },
        "002d_protocol": {
            **protocol_checks,
            "path": str(protocol_path),
            "sha256": protocol_sha256,
        },
        "qc_rules": {
            **qc_checks,
            "path": str(qc_path),
            "sha256": qc_sha256,
        },
    }


def _source_suffix(url: str) -> str:
    path = urlparse(url).path
    return Path(path).suffix.lower()


def _fetch_source_document(
    document: Mapping[str, Any],
    *,
    timeout_seconds: float,
    max_bytes: int,
    fetcher: Callable[[str, float, int], tuple[bytes, str, str]] | None,
) -> dict[str, Any]:
    url = str(document.get("url", ""))
    suffix = _source_suffix(url)
    base = {
        "id": document.get("id"),
        "kind": document.get("kind"),
        "url": url,
        "revision": document.get("revision"),
        "role": document.get("role"),
        "evidence_targets": list(document.get("evidence_targets", [])),
        "metadata_only": True,
        "allow_signal_samples": document.get("allow_signal_samples"),
    }
    if suffix in _FORBIDDEN_SUFFIXES:
        return {
            **base,
            "status": "REJECTED",
            "reason": "forbidden_payload_suffix",
        }
    if urlparse(url).scheme.lower() != "https":
        return {
            **base,
            "status": "REJECTED",
            "reason": "https_required",
        }
    if document.get("allow_signal_samples") is not False:
        return {
            **base,
            "status": "REJECTED",
            "reason": "metadata_only_boundary_not_declared",
        }
    try:
        if fetcher is None:
            request = Request(
                url,
                headers={"User-Agent": "NeuralManifoldDynamics-FAR-EXT-002E"},
            )
            with urlopen(request, timeout=timeout_seconds) as response:
                body = response.read(max_bytes + 1)
                final_url = response.geturl()
                content_type = response.headers.get("Content-Type", "")
        else:
            body, final_url, content_type = fetcher(
                url,
                timeout_seconds,
                max_bytes,
            )
        if urlparse(final_url).scheme.lower() != "https":
            return {
                **base,
                "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
                "reason": "unsafe_final_url_scheme",
                "final_url": final_url,
            }
        if _source_suffix(final_url) in _FORBIDDEN_SUFFIXES:
            return {
                **base,
                "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
                "reason": "unsafe_final_url_suffix",
                "final_url": final_url,
            }
        if len(body) > max_bytes:
            return {
                **base,
                "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
                "reason": "source_document_too_large",
            }
        text = body.decode("utf-8", errors="replace")
        lowered = text.lower()
        markers = [
            str(marker).lower()
            for marker in document.get("required_markers", [])
        ]
        marker_presence = {
            marker: marker in lowered
            for marker in markers
        }
        marker_pass = all(marker_presence.values())
        return {
            **base,
            "status": "PASS"
            if marker_pass
            else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": (
                "source_markers_present"
                if marker_pass
                else "required_source_marker_missing"
            ),
            "final_url": final_url,
            "content_type": content_type,
            "content_bytes": len(body),
            "content_sha256": hashlib.sha256(body).hexdigest(),
            "required_markers": markers,
            "marker_presence": marker_presence,
        }
    except (HTTPError, URLError, OSError, TimeoutError, ValueError) as exc:
        return {
            **base,
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": f"source_document_fetch_failed:{type(exc).__name__}",
        }


def _source_semantics(
    documents: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {str(document.get("id")): document for document in documents}
    dataset = by_id.get("gx_dataset_descriptor", {})
    paper = by_id.get("f30_followup_paper", {})
    code = by_id.get("gx_trial_extraction_code", {})
    revision = by_id.get("f30_followup_code_revision", {})
    context = by_id.get("tes_artifact_context", {})
    all_pass = all(
        document.get("status") == "PASS"
        for document in (dataset, paper, code, revision, context)
    )
    dataset_pass = dataset.get("status") == "PASS"
    paper_pass = paper.get("status") == "PASS"
    code_pass = code.get("status") == "PASS"
    revision_pass = revision.get("status") == "PASS"
    return {
        "source_documents_complete": all_pass,
        "acquisition_transfer": {
            "status": "PASS"
            if dataset_pass and revision_pass
            else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "basis": [
                "same GX dataset Experiment 1 and Experiment 2 source",
                "fixed follow-up analysis revision",
            ],
        },
        "published_post_interval": {
            "status": "PASS"
            if paper_pass and code_pass
            else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "duration_sec": F30_PUBLISHED_POST_SEC,
            "anchor": "source code Evnt_Stimstrt",
            "source_start_expression": "Evnt_Stimstrt + 41.5*fs",
            "source_defined_offset_after_ramp_down_sec": 1.5,
            "directly_after_ramp_down": False,
            "source_defined_delay_is_not_invented_by_002e": True,
        },
        "during_stimulation": {
            "status": "PASS" if paper_pass else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "excluded": True,
            "reason": "published_analysis_excludes_during_stimulation_due_to_nonlinear_artifacts",
        },
        "source_preprocessing": {
            "status": "PASS" if paper_pass else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "baseline_correction": "source_defined",
            "bandpass_hz": [0.35, 40.0],
            "introduced_by_002e": False,
        },
        "amplitude_transfer": {
            "1.0_mA": {
                "status": POST_RAMP_INTERPRETABILITY_BRIDGE_PASS
                if dataset_pass and paper_pass
                else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
                "basis": "published Experiment-2 F30 precedent",
            },
            "0.5_mA": {
                "status": POST_RAMP_METHOD_LIMITED,
                "basis": "published Experiment-2 F30 precedent is 1 mA",
                "silent_transfer": False,
            },
        },
        "artifact_context": {
            "status": "PASS"
            if context.get("status") == "PASS"
            else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "role": "context_only_not_an_eligibility_rule",
        },
    }


def normalize_source_text(
    text: str,
    *,
    strip_tags: bool = True,
) -> str:
    """Normalize a frozen source text layer without changing its assertions."""
    normalized = html.unescape(text)
    if strip_tags:
        normalized = re.sub(r"<[^>]+>", " ", normalized)
    normalized = unicodedata.normalize("NFKC", normalized)
    normalized = normalized.translate(
        str.maketrans(
            {
                "\u2010": "-",
                "\u2011": "-",
                "\u2012": "-",
                "\u2013": "-",
                "\u2014": "-",
                "\u2212": "-",
            }
        )
    )
    return " ".join(normalized.lower().split())


def _normalized_assertion_terms(
    values: object,
) -> list[str]:
    if not isinstance(values, list):
        return []
    return [
        normalize_source_text(str(value))
        for value in values
        if normalize_source_text(str(value))
    ]


def evaluate_source_assertions(
    text: str,
    assertions: object,
    *,
    strip_tags: bool = True,
) -> dict[str, Any]:
    normalized = normalize_source_text(text, strip_tags=strip_tags)
    rows: list[dict[str, Any]] = []
    if not isinstance(assertions, list):
        return {
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "required_assertions_pass": False,
            "assertions": rows,
        }
    for assertion in assertions:
        if not isinstance(assertion, Mapping):
            rows.append(
                {
                    "id": None,
                    "required": True,
                    "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
                    "reason": "assertion_row_invalid",
                }
            )
            continue
        assertion_id = str(assertion.get("id", ""))
        all_of = _normalized_assertion_terms(assertion.get("all_of", []))
        any_of = _normalized_assertion_terms(assertion.get("any_of", []))
        none_of = _normalized_assertion_terms(assertion.get("none_of", []))
        all_presence = {term: term in normalized for term in all_of}
        any_presence = {term: term in normalized for term in any_of}
        none_presence = {term: term in normalized for term in none_of}
        passed = (
            all(all_presence.values())
            and (not any_of or any(any_presence.values()))
            and not any(none_presence.values())
        )
        rows.append(
            {
                "id": assertion_id,
                "required": bool(assertion.get("required", True)),
                "status": "PASS" if passed else "UNRESOLVED",
                "all_of": all_presence,
                "any_of": any_presence,
                "none_of": none_presence,
            }
        )
    required_pass = all(
        row["status"] == "PASS"
        for row in rows
        if row.get("required", True)
    )
    return {
        "status": "PASS"
        if required_pass
        else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
        "required_assertions_pass": required_pass,
        "assertions": rows,
    }


def _confined_snapshot_path(
    repository_root: Path,
    relative_path: str,
) -> Path:
    root = repository_root.resolve()
    path = (root / Path(relative_path)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("source_snapshot_outside_repository") from exc
    return path


def read_frozen_source_object(
    document: Mapping[str, Any],
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Read and verify one pre-deposited source object."""
    retrieval = document.get("retrieval")
    base = {
        "id": document.get("id"),
        "evidence_id": document.get("evidence_id"),
        "role": document.get("role"),
        "kind": document.get("kind"),
        "metadata_only": True,
        "allow_signal_samples": document.get("allow_signal_samples"),
        "retrieval_url": retrieval.get("declared_url")
        if isinstance(retrieval, Mapping)
        else None,
        "snapshot_path": retrieval.get("snapshot_path")
        if isinstance(retrieval, Mapping)
        else None,
        "text_path": retrieval.get("text_path")
        if isinstance(retrieval, Mapping)
        else None,
    }
    if document.get("allow_signal_samples") is not False:
        return {
            **base,
            "status": "REJECTED",
            "reason": "metadata_only_boundary_not_declared",
        }
    if not isinstance(retrieval, Mapping):
        return {
            **base,
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": "snapshot_retrieval_missing",
        }
    if retrieval.get("mode") != "frozen_snapshot":
        return {
            **base,
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": "frozen_snapshot_mode_required",
        }
    snapshot_relative = str(retrieval.get("snapshot_path", ""))
    text_relative = str(retrieval.get("text_path", ""))
    for candidate in (snapshot_relative, text_relative):
        if _source_suffix(candidate) in _FORBIDDEN_SUFFIXES:
            return {
                **base,
                "status": "REJECTED",
                "reason": "forbidden_snapshot_suffix",
                "path": candidate,
            }
    try:
        snapshot_path = _confined_snapshot_path(
            repository_root,
            snapshot_relative,
        )
        text_path = _confined_snapshot_path(repository_root, text_relative)
        snapshot_bytes = snapshot_path.read_bytes()
        text_bytes = text_path.read_bytes()
    except (OSError, ValueError) as exc:
        return {
            **base,
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": str(exc)
            if isinstance(exc, ValueError)
            else "source_snapshot_missing",
        }
    snapshot_sha256 = hashlib.sha256(snapshot_bytes).hexdigest()
    text_sha256 = hashlib.sha256(text_bytes).hexdigest()
    max_bytes = int(retrieval.get("max_bytes", _MAX_SOURCE_BYTES))
    min_bytes = int(retrieval.get("min_content_bytes", 1))
    if len(snapshot_bytes) > max_bytes or len(text_bytes) > max_bytes:
        return {
            **base,
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": "source_snapshot_too_large",
            "snapshot_sha256": snapshot_sha256,
            "text_sha256": text_sha256,
        }
    snapshot_text = snapshot_bytes.decode("utf-8", errors="replace")
    required_snapshot_markers = _normalized_assertion_terms(
        retrieval.get("required_snapshot_markers", [])
    )
    snapshot_body_checks = {
        marker: marker in normalize_source_text(
            snapshot_text,
            strip_tags=False,
        )
        for marker in required_snapshot_markers
    }
    if required_snapshot_markers and not any(snapshot_body_checks.values()):
        return {
            **base,
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": "source_snapshot_body_missing",
            "snapshot_bytes": len(snapshot_bytes),
            "text_bytes": len(text_bytes),
            "snapshot_sha256": snapshot_sha256,
            "text_sha256": text_sha256,
            "snapshot_body_checks": snapshot_body_checks,
        }
    hash_checks = {
        "snapshot_sha256": snapshot_sha256
        == str(retrieval.get("expected_content_sha256")),
        "text_sha256": text_sha256
        == str(retrieval.get("expected_text_sha256")),
        "snapshot_min_bytes": len(snapshot_bytes) >= min_bytes,
        "text_min_bytes": len(text_bytes) >= min_bytes,
    }
    expected_normalized_sha256 = retrieval.get(
        "expected_normalized_text_sha256"
    )
    if expected_normalized_sha256 is not None:
        hash_checks["normalized_text_sha256"] = False
    try:
        text = text_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return {
            **base,
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": "source_text_not_utf8",
            "snapshot_sha256": snapshot_sha256,
            "text_sha256": text_sha256,
            "hash_checks": hash_checks,
        }
    text_layer_derivation = retrieval.get("text_layer_derivation")
    text_derivation_check = True
    if text_layer_derivation == "identity_v1":
        text_derivation_check = text_bytes == snapshot_bytes
    elif text_layer_derivation == "xml_itertext_v1":
        try:
            derived = "\n".join(
                ET.fromstring(snapshot_bytes).itertext()
            )
            text_derivation_check = (
                normalize_source_text(
                    derived,
                    strip_tags=False,
                )
                == normalize_source_text(
                    text,
                    strip_tags=False,
                )
            )
        except ET.ParseError:
            text_derivation_check = False
    else:
        text_derivation_check = False
    if not text_derivation_check:
        return {
            **base,
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": "source_text_not_derived_from_snapshot",
            "snapshot_bytes": len(snapshot_bytes),
            "text_bytes": len(text_bytes),
            "snapshot_sha256": snapshot_sha256,
            "text_sha256": text_sha256,
            "hash_checks": hash_checks,
            "text_derivation_check": False,
        }
    normalization = document.get("normalization")
    strip_tags = (
        bool(normalization.get("strip_tags", True))
        if isinstance(normalization, Mapping)
        else True
    )
    normalized = normalize_source_text(text, strip_tags=strip_tags)
    normalized_sha256 = hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()
    if expected_normalized_sha256 is not None:
        hash_checks["normalized_text_sha256"] = (
            normalized_sha256 == str(expected_normalized_sha256)
        )
    assertion_result = evaluate_source_assertions(
        normalized,
        document.get("assertions"),
        strip_tags=False,
    )
    passed = all(hash_checks.values()) and (
        assertion_result["status"] == "PASS"
    )
    return {
        **base,
        "status": "PASS"
        if passed
        else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
        "reason": "frozen_source_assertions_pass"
        if passed
        else "frozen_source_hash_or_assertion_failure",
        "snapshot_bytes": len(snapshot_bytes),
        "text_bytes": len(text_bytes),
        "snapshot_sha256": snapshot_sha256,
        "text_sha256": text_sha256,
        "normalized_text_sha256": normalized_sha256,
        "hash_checks": hash_checks,
        "snapshot_body_checks": snapshot_body_checks,
        "text_derivation_check": text_derivation_check,
        "assertion_result": assertion_result,
    }


def _source_semantics_v2(
    documents: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    by_id = {str(document.get("id")): document for document in documents}
    required_ids = {"E1", "E2", "E3", "E4"}
    required = {
        evidence_id: by_id.get(evidence_id, {})
        for evidence_id in required_ids
    }
    required_pass = all(
        document.get("status") == "PASS"
        for document in required.values()
    )
    assertion_rows = {
        evidence_id: required[evidence_id].get("assertion_result", {})
        for evidence_id in required_ids
    }
    assertion_sources = {
        "A1": "E1",
        "A2": "E1",
        "A3": "E1",
        "A4": "E2",
        "A5": "E3",
        "A6": "E4",
        "A7": "E4",
    }
    assertion_pass = {
        assertion_id: any(
            row.get("id") == assertion_id
            and row.get("status") == "PASS"
            for row in assertion_rows[evidence_id].get("assertions", [])
        )
        for assertion_id, evidence_id in assertion_sources.items()
    }
    return {
        "required_source_objects_complete": required_pass,
        "assertion_pass": assertion_pass,
        "published_post_interval": {
            "status": "PASS"
            if assertion_pass["A1"] and assertion_pass["A5"]
            else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "duration_sec": F30_PUBLISHED_POST_SEC,
            "source_defined_offset_after_ramp_down_sec": 1.5,
            "directly_after_ramp_down": False,
            "technical_60s_horizon_interpretability_transfer": False,
        },
        "during_stimulation": {
            "status": "PASS"
            if assertion_pass["A2"]
            else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "excluded": True,
            "reason": "published_analysis_excludes_during_stimulation_due_to_nonlinear_artifacts",
        },
        "source_preprocessing": {
            "status": "PASS"
            if assertion_pass["A3"]
            else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "baseline_correction": "source_defined",
            "bandpass_hz": [0.35, 40.0],
            "introduced_by_002e": False,
        },
        "acquisition_transfer": {
            "status": "PASS"
            if assertion_pass["A6"]
            else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "basis": "E4 normalized source assertions",
        },
        "amplitude_transfer": {
            "1.0_mA": {
                "status": POST_RAMP_INTERPRETABILITY_BRIDGE_PASS
                if assertion_pass["A4"] and assertion_pass["A7"]
                else POST_RAMP_INTERPRETABILITY_UNRESOLVED,
                "basis": "E2 and E4 normalized source assertions",
            },
            "0.5_mA": {
                "status": POST_RAMP_METHOD_LIMITED,
                "basis": "no silent transfer from 1.0 mA precedent",
                "silent_transfer": False,
            },
        },
        "artifact_context": {
            "status": by_id.get("C0", {}).get("status", "NOT_REQUIRED"),
            "role": "context_only_not_an_eligibility_rule",
        },
    }


def _audit_far_ext_002e_v2(
    *,
    scope: Mapping[str, Any],
    scope_hash: str,
    scope_path: Path,
    repo_root: Path,
    result_path: Path | None,
    fetch_documents: bool,
) -> dict[str, Any]:
    inherited = scope["inherited_002d"]
    bound_result_path = (
        result_path
        if result_path is not None
        else repo_root / Path(str(inherited["result_path"]))
    )
    inherited_binding = bind_inherited_contract(
        scope=scope,
        repo_root=repo_root,
        result_path=bound_result_path,
    )
    prior = scope["prior_002e_0_1"]
    prior_path = repo_root / Path(str(prior["result_path"]))
    prior_binding = {
        "path": str(prior_path),
        "exists": prior_path.is_file(),
        "sha256": None,
        "expected_sha256": str(prior["result_sha256"]),
        "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
        "checks": {
            "exists": prior_path.is_file(),
            "sha256": False,
            "protocol_id": False,
            "global_status": False,
        },
    }
    if prior_path.is_file():
        prior_binding["sha256"] = sha256_file(prior_path)
        try:
            prior_result = _load_json_object(
                prior_path,
                "prior_far_ext_002e_result_unreadable",
            )
        except RuntimeError:
            prior_result = {}
        prior_binding["checks"].update(
            {
                "sha256": prior_binding["sha256"]
                == prior_binding["expected_sha256"],
                "protocol_id": prior_result.get("protocol_id") == PROTOCOL_ID,
                "global_status": prior_result.get("global_status")
                == str(prior["result_global_status"]),
            }
        )
        prior_binding["status"] = (
            "PASS"
            if all(prior_binding["checks"].values())
            else POST_RAMP_SOURCE_MISMATCH
        )
    source_materialization = scope["source_materialization"]
    manifest_relative = str(source_materialization["manifest_path"])
    manifest_binding = {
        "path": manifest_relative,
        "expected_sha256": str(source_materialization["manifest_sha256"]),
        "sha256": None,
        "status": POST_RAMP_SOURCE_MISMATCH,
        "checks": {
            "exists": False,
            "sha256": False,
            "object_ids": False,
        },
    }
    try:
        manifest_path = _confined_snapshot_path(
            repo_root,
            manifest_relative,
        )
        manifest_binding["checks"]["exists"] = manifest_path.is_file()
        if manifest_path.is_file():
            manifest_binding["sha256"] = sha256_file(manifest_path)
            manifest = _load_json_object(
                manifest_path,
                "source_snapshot_manifest_unreadable",
            )
            manifest_binding["checks"]["sha256"] = (
                manifest_binding["sha256"]
                == manifest_binding["expected_sha256"]
            )
            manifest_rows = {
                str(row.get("evidence_id")): row
                for row in manifest.get("objects", [])
                if isinstance(row, Mapping)
            }
            scope_rows = {
                str(row.get("evidence_id")): row
                for row in scope["source_objects"]
            }
            manifest_ids = set(manifest_rows)
            manifest_binding["checks"]["object_ids"] = manifest_ids == {
                str(row.get("evidence_id"))
                for row in scope["source_objects"]
            }
            manifest_binding["checks"]["object_hashes"] = (
                manifest_ids == set(scope_rows)
                and all(
                    manifest_rows[evidence_id].get("content_sha256")
                    == scope_rows[evidence_id]["retrieval"].get(
                        "expected_content_sha256"
                    )
                    and manifest_rows[evidence_id].get("text_sha256")
                    == scope_rows[evidence_id]["retrieval"].get(
                        "expected_text_sha256"
                    )
                    and manifest_rows[evidence_id].get(
                        "normalized_text_sha256"
                    )
                    == scope_rows[evidence_id]["retrieval"].get(
                        "expected_normalized_text_sha256"
                    )
                    for evidence_id in scope_rows
                )
            )
    except (OSError, RuntimeError, ValueError, TypeError):
        pass
    manifest_binding["status"] = (
        "PASS"
        if all(manifest_binding["checks"].values())
        else POST_RAMP_SOURCE_MISMATCH
    )
    source_documents = [
        (
            read_frozen_source_object(document, repository_root=repo_root)
            if fetch_documents
            else {
                "id": document.get("id"),
                "evidence_id": document.get("evidence_id"),
                "kind": document.get("kind"),
                "role": document.get("role"),
                "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
                "reason": "source_fetch_disabled",
                "metadata_only": True,
                "allow_signal_samples": False,
            }
        )
        for document in scope["source_objects"]
    ]
    required_source_ids = {"E1", "E2", "E3", "E4"}
    required_source_ok = all(
        next(
            (
                document
                for document in source_documents
                if document.get("evidence_id") == evidence_id
            ),
            {},
        ).get("status")
        == "PASS"
        for evidence_id in required_source_ids
    )
    semantics = _source_semantics_v2(source_documents)
    binding_ok = inherited_binding["status"] == "PASS"
    prior_ok = prior_binding["status"] == "PASS"
    manifest_ok = manifest_binding["status"] == "PASS"
    if not binding_ok or not prior_ok or not manifest_ok:
        global_status = POST_RAMP_SOURCE_MISMATCH
        global_reason = "inherited_002e_binding_failed"
    elif not required_source_ok:
        global_status = POST_RAMP_INTERPRETABILITY_UNRESOLVED
        global_reason = "required_frozen_source_evidence_unresolved"
    elif not all(semantics["assertion_pass"].values()):
        global_status = POST_RAMP_INTERPRETABILITY_UNRESOLVED
        global_reason = "required_normalized_assertion_unresolved"
    else:
        global_status = POST_RAMP_INTERPRETABILITY_BRIDGE_PASS
        global_reason = "scoped_30s_1mA_source_bridge"
    allowed_later_scope = (
        {
            "rho_mA": [DIRECT_TRANSFER_RHO],
            "post_duration_sec": [F30_PUBLISHED_POST_SEC],
            "source_defined_offset_after_ramp_down_sec": [1.5],
            "technical_60s_horizon_interpretability_transfer": False,
        }
        if global_status == POST_RAMP_INTERPRETABILITY_BRIDGE_PASS
        else {
            "rho_mA": [],
            "post_duration_sec": [],
            "source_defined_offset_after_ramp_down_sec": [],
            "technical_60s_horizon_interpretability_transfer": False,
        }
    )
    return {
        "schema": "mndm.far_ext_002e_ds003670_post_ramp_interpretation.v2",
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "global_status": global_status,
        "global_reason": global_reason,
        "scope_path": str(scope_path),
        "scope_sha256": scope_hash,
        "entry_far_ext_002e_0_1": prior_binding,
        "entry_far_ext_002d": inherited_binding,
        "source_manifest": manifest_binding,
        "source_objects": source_documents,
        "source_semantics": semantics,
        "f30_family_id": F30_FAMILY_ID,
        "f30_published_post_sec": F30_PUBLISHED_POST_SEC,
        "f30_technical_horizon_sec": F30_TECHNICAL_HORIZON_SEC,
        "post_stimulation_biological_interpretability": (
            "BRIDGE_SCOPED_NOT_ARTIFACT_FREE"
            if global_status == POST_RAMP_INTERPRETABILITY_BRIDGE_PASS
            else "NOT_ESTABLISHED"
        ),
        "allowed_later_scope": allowed_later_scope,
        "tolerability_confound": True,
        "far_ext_003b_authorized": False,
        "mnps_calculated": False,
        "far_calculated": False,
        "audit_scope": {
            "signal_payloads_opened": False,
            "outcome_tables_opened": False,
            "nmd_outputs_opened": False,
            "mnps_calculated": False,
            "far_calculated": False,
        },
        "claim_boundary": (
            "This source-rematerialization bridge can establish only a "
            "scoped source precedent for a 30-second F30 post-ramp "
            "observation at 1.0 mA. It never establishes artifact-free EEG, "
            "biological truth, 0.5 mA transfer, a 60-second interpretability "
            "license, MNPS, FAR, home/away, or causal amplitude effects."
        ),
    }


def _global_status(
    *,
    binding: Mapping[str, Any],
    semantics: Mapping[str, Any],
) -> tuple[str, str]:
    if binding.get("status") != "PASS":
        return POST_RAMP_SOURCE_MISMATCH, "002d_binding_failed"
    if not semantics.get("source_documents_complete"):
        return (
            POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "required_source_evidence_unresolved",
        )
    post = semantics["published_post_interval"]
    if (
        post.get("duration_sec") != F30_PUBLISHED_POST_SEC
        or post.get("directly_after_ramp_down") is True
    ):
        return POST_RAMP_SOURCE_MISMATCH, "post_interval_semantics_mismatch"
    if semantics["amplitude_transfer"]["1.0_mA"]["status"] != (
        POST_RAMP_INTERPRETABILITY_BRIDGE_PASS
    ):
        return (
            POST_RAMP_METHOD_LIMITED,
            "direct_1mA_precedent_not_bound",
        )
    return (
        POST_RAMP_INTERPRETABILITY_BRIDGE_PASS,
        "scoped_30s_1mA_source_bridge",
    )


def audit_far_ext_002e(
    *,
    scope_path: Path,
    repo_root: Path,
    result_path: Path | None = None,
    fetch_documents: bool = True,
    fetcher: Callable[[str, float, int], tuple[bytes, str, str]] | None = None,
) -> dict[str, Any]:
    """Run the frozen source-only bridge without touching signal payloads."""
    try:
        scope, scope_hash = load_scope(scope_path)
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "schema": "mndm.far_ext_002e_ds003670_post_ramp_interpretation.v1",
            "protocol_id": PROTOCOL_ID,
            "dataset_id": DATASET_ID,
            "global_status": POST_RAMP_SOURCE_MISMATCH,
            "global_reason": f"scope_binding_failed:{type(exc).__name__}",
            "audit_scope": {
                "signal_payloads_opened": False,
                "outcome_tables_opened": False,
                "nmd_outputs_opened": False,
                "mnps_calculated": False,
                "far_calculated": False,
            },
        }

    if str(scope.get("scope_version", "0.1")) == "0.2":
        return _audit_far_ext_002e_v2(
            scope=scope,
            scope_hash=scope_hash,
            scope_path=scope_path,
            repo_root=repo_root,
            result_path=result_path,
            fetch_documents=fetch_documents,
        )

    inherited = scope["inherited_002d"]
    bound_result_path = (
        result_path
        if result_path is not None
        else repo_root / Path(str(inherited["result_path"]))
    )
    binding = bind_inherited_contract(
        scope=scope,
        repo_root=repo_root,
        result_path=bound_result_path,
    )
    if binding["status"] != "PASS":
        return {
            "schema": "mndm.far_ext_002e_ds003670_post_ramp_interpretation.v1",
            "protocol_id": PROTOCOL_ID,
            "dataset_id": DATASET_ID,
            "scope_sha256": scope_hash,
            "global_status": POST_RAMP_SOURCE_MISMATCH,
            "global_reason": "002d_binding_failed",
            "entry_far_ext_002d": binding,
            "source_documents": [],
            "audit_scope": {
                "signal_payloads_opened": False,
                "outcome_tables_opened": False,
                "nmd_outputs_opened": False,
                "mnps_calculated": False,
                "far_calculated": False,
            },
        }

    documents = [
        _fetch_source_document(
            document,
            timeout_seconds=120.0,
            max_bytes=_MAX_SOURCE_BYTES,
            fetcher=fetcher,
        )
        if fetch_documents
        else {
            "id": document["id"],
            "kind": document["kind"],
            "url": document["url"],
            "revision": document.get("revision"),
            "role": document["role"],
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": "source_fetch_disabled",
            "metadata_only": True,
            "allow_signal_samples": False,
        }
        for document in scope["source_documents"]
    ]
    semantics = _source_semantics(documents)
    global_status, global_reason = _global_status(
        binding=binding,
        semantics=semantics,
    )
    return {
        "schema": "mndm.far_ext_002e_ds003670_post_ramp_interpretation.v1",
        "protocol_id": PROTOCOL_ID,
        "dataset_id": DATASET_ID,
        "global_status": global_status,
        "global_reason": global_reason,
        "scope_path": str(scope_path),
        "scope_sha256": scope_hash,
        "entry_far_ext_002d": binding,
        "source_documents": documents,
        "source_semantics": semantics,
        "f30_family_id": F30_FAMILY_ID,
        "f30_published_post_sec": F30_PUBLISHED_POST_SEC,
        "f30_technical_horizon_sec": F30_TECHNICAL_HORIZON_SEC,
        "post_stimulation_biological_interpretability": (
            "BRIDGE_SCOPED_NOT_ARTIFACT_FREE"
            if global_status == POST_RAMP_INTERPRETABILITY_BRIDGE_PASS
            else "NOT_ESTABLISHED"
        ),
        "allowed_later_scope": {
            "rho_mA": [DIRECT_TRANSFER_RHO],
            "post_duration_sec": [F30_PUBLISHED_POST_SEC],
            "source_defined_offset_after_ramp_down_sec": [1.5],
            "technical_60s_horizon_interpretability_transfer": False,
        },
        "tolerability_confound": True,
        "far_ext_003b_authorized": False,
        "mnps_calculated": False,
        "far_calculated": False,
        "audit_scope": {
            "signal_payloads_opened": False,
            "outcome_tables_opened": False,
            "nmd_outputs_opened": False,
            "mnps_calculated": False,
            "far_calculated": False,
        },
        "claim_boundary": (
            "This source-only bridge can establish only a scoped source "
            "precedent for a 30-second F30 post-ramp observation at 1.0 mA. "
            "It never establishes artifact-free EEG, biological truth, "
            "0.5 mA transfer, a 60-second interpretability license, MNPS, "
            "FAR, home/away, or causal amplitude effects."
        ),
    }


ALLOWED_COMPLETE_SOURCE_CLASSES = frozenset(
    {
        "PUBLISHER_VOR",
        "COMPLETE_OA_SOURCE_OBJECT",
    }
)
E1_IDENTITY_MARKERS = (
    "Frontal HD-tACS enhances behavioral and EEG biomarkers",
    "10.1016/j.brs.2024.05.009",
)
E2_IDENTITY_MARKERS = (
    "Response regarding",
    "10.1016/j.brs.2024.11.004",
)


def derive_xml_itertext(snapshot_bytes: bytes) -> str:
    """Deterministic XML text layer used by ``xml_itertext_v1``."""
    return "\n".join(ET.fromstring(snapshot_bytes).itertext())


def assess_complete_xml_candidate(
    snapshot_bytes: bytes,
    *,
    assertions: Sequence[Mapping[str, Any]],
    identity_markers: Sequence[str],
    required_snapshot_markers: Sequence[str],
    min_content_bytes: int,
) -> dict[str, Any]:
    """Fail-closed completeness check for an E1/E2 XML candidate.

    Does not write files, freeze the scope, or run the 0.2 gate.
    """
    snapshot_text = snapshot_bytes.decode("utf-8", errors="replace")
    markers = _normalized_assertion_terms(list(required_snapshot_markers))
    body_checks = {
        marker: marker in normalize_source_text(snapshot_text, strip_tags=False)
        for marker in markers
    }
    if markers and not any(body_checks.values()):
        return {
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": "source_snapshot_body_missing",
            "snapshot_bytes": len(snapshot_bytes),
            "snapshot_body_checks": body_checks,
        }
    if len(snapshot_bytes) < int(min_content_bytes):
        return {
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": "source_snapshot_below_min_content_bytes",
            "snapshot_bytes": len(snapshot_bytes),
            "snapshot_body_checks": body_checks,
        }
    try:
        derived = derive_xml_itertext(snapshot_bytes)
    except ET.ParseError:
        return {
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": "source_text_not_derived_from_snapshot",
            "snapshot_bytes": len(snapshot_bytes),
            "snapshot_body_checks": body_checks,
        }
    derived_bytes = derived.encode("utf-8")
    if len(derived_bytes) < int(min_content_bytes):
        return {
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": "source_text_below_min_content_bytes",
            "snapshot_bytes": len(snapshot_bytes),
            "text_bytes": len(derived_bytes),
            "snapshot_body_checks": body_checks,
        }
    identity = {
        marker: normalize_source_text(marker) in normalize_source_text(derived)
        for marker in identity_markers
    }
    if identity_markers and not all(identity.values()):
        return {
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": "source_identity_markers_missing",
            "snapshot_bytes": len(snapshot_bytes),
            "text_bytes": len(derived_bytes),
            "identity_checks": identity,
            "snapshot_body_checks": body_checks,
        }
    assertion_result = evaluate_source_assertions(
        derived,
        list(assertions),
        strip_tags=True,
    )
    if assertion_result["status"] != "PASS":
        return {
            "status": POST_RAMP_INTERPRETABILITY_UNRESOLVED,
            "reason": "required_normalized_assertion_unresolved",
            "snapshot_bytes": len(snapshot_bytes),
            "text_bytes": len(derived_bytes),
            "identity_checks": identity,
            "assertion_result": assertion_result,
            "snapshot_body_checks": body_checks,
        }
    normalized = normalize_source_text(derived, strip_tags=True)
    return {
        "status": "PASS",
        "reason": "complete_xml_candidate",
        "snapshot_bytes": len(snapshot_bytes),
        "text_bytes": len(derived_bytes),
        "derived_text": derived,
        "content_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
        "text_sha256": hashlib.sha256(derived_bytes).hexdigest(),
        "normalized_text_sha256": hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest(),
        "identity_checks": identity,
        "assertion_result": assertion_result,
        "snapshot_body_checks": body_checks,
    }


def bind_complete_e1_e2_xml(
    *,
    repo_root: Path,
    e1_xml: Path,
    e2_xml: Path,
    source_object_class: str,
    apply: bool = False,
) -> dict[str, Any]:
    """Assess or bind complete E1/E2 XML without freezing or running 0.2."""
    if source_object_class not in ALLOWED_COMPLETE_SOURCE_CLASSES:
        raise ValueError("far_ext_002e_unknown_complete_source_class")
    root = Path(repo_root)
    far_root = (
        root
        / "project"
        / "orthagonal_axis"
        / "orthagonal_dynamics"
        / "finite-amplitude_resilience"
    )
    scope_path = far_root / "FAR-EXT-002E-SCOPE-0.2.json"
    prereg_path = (
        far_root / "014_far_ext_002e_scope_0.2_source_rematerialization_prereg.md"
    )
    manifest_path = (
        far_root
        / "source_snapshots"
        / "far_ext_002e_scope_0.2"
        / "manifest.json"
    )
    result_dir = far_root / "results" / "far_ext_002e_scope_0.2"
    scope = copy.deepcopy(
        _load_json_object(scope_path, "far_ext_002e_scope_unreadable")
    )
    if scope.get("execution_status") == "PREREGISTRATION_FROZEN":
        raise ValueError("far_ext_002e_refusing_to_mutate_frozen_scope")
    if bool(scope.get("source_materialization", {}).get("execution_authorized")):
        raise ValueError("far_ext_002e_refusing_to_mutate_authorized_scope")
    if result_dir.exists():
        raise FileExistsError("refusing_to_mutate_after_far_ext_002e_scope_0_2_archive")
    by_id = {
        str(row.get("id")): row
        for row in scope.get("source_objects", [])
        if isinstance(row, Mapping)
    }
    assessments = {
        "E1": assess_complete_xml_candidate(
            Path(e1_xml).read_bytes(),
            assertions=by_id["E1"]["assertions"],
            identity_markers=E1_IDENTITY_MARKERS,
            required_snapshot_markers=by_id["E1"]["retrieval"][
                "required_snapshot_markers"
            ],
            min_content_bytes=int(
                by_id["E1"]["retrieval"]["min_content_bytes"]
            ),
        ),
        "E2": assess_complete_xml_candidate(
            Path(e2_xml).read_bytes(),
            assertions=by_id["E2"]["assertions"],
            identity_markers=E2_IDENTITY_MARKERS,
            required_snapshot_markers=by_id["E2"]["retrieval"][
                "required_snapshot_markers"
            ],
            min_content_bytes=int(
                by_id["E2"]["retrieval"]["min_content_bytes"]
            ),
        ),
    }
    complete = all(row["status"] == "PASS" for row in assessments.values())
    payload = {
        "protocol_id": PROTOCOL_ID,
        "scope_version": "0.2",
        "apply": bool(apply),
        "source_object_class": source_object_class,
        "execution_status": scope.get("execution_status"),
        "execution_authorized": False,
        "current_candidate_status": (
            "E1_E2_COMPLETE_PENDING_FREEZE"
            if complete
            else "E1_E2_INCOMPLETE"
        ),
        "assessments": {
            evidence_id: {
                key: value
                for key, value in row.items()
                if key != "derived_text"
            }
            for evidence_id, row in assessments.items()
        },
        "live_0_2_run": False,
    }
    if not complete or not apply:
        return payload
    snapshot_root = far_root / "source_snapshots" / "far_ext_002e_scope_0.2"
    written = {}
    for evidence_id, xml_name, txt_name in (
        ("E1", "E1_f30_followup.xml", "E1_f30_followup.txt"),
        ("E2", "E2_f30_response.xml", "E2_f30_response.txt"),
    ):
        xml_bytes = Path(e1_xml if evidence_id == "E1" else e2_xml).read_bytes()
        derived = assessments[evidence_id]["derived_text"]
        xml_path = snapshot_root / xml_name
        txt_path = snapshot_root / txt_name
        xml_path.write_bytes(xml_bytes)
        txt_path.write_text(derived, encoding="utf-8", newline="\n")
        retrieval = by_id[evidence_id]["retrieval"]
        retrieval["expected_content_sha256"] = assessments[evidence_id][
            "content_sha256"
        ]
        retrieval["expected_text_sha256"] = assessments[evidence_id][
            "text_sha256"
        ]
        retrieval["expected_normalized_text_sha256"] = assessments[evidence_id][
            "normalized_text_sha256"
        ]
        retrieval["text_layer_kind"] = "xml_itertext_from_complete_snapshot"
        retrieval["capture_status"] = "complete_pending_freeze"
        retrieval["source_object_class"] = source_object_class
        written[evidence_id] = {
            "snapshot_path": str(xml_path),
            "text_path": str(txt_path),
        }
    scope["source_materialization"]["current_candidate_status"] = (
        "E1_E2_COMPLETE_PENDING_FREEZE"
    )
    scope["source_materialization"]["execution_authorized"] = False
    scope["execution_status"] = "DESIGN_SCOPE_ONLY"
    manifest = _load_json_object(manifest_path, "far_ext_002e_manifest_unreadable")
    for row in manifest.get("objects", []):
        evidence_id = str(row.get("evidence_id"))
        if evidence_id not in assessments or assessments[evidence_id]["status"] != "PASS":
            continue
        row["content_sha256"] = assessments[evidence_id]["content_sha256"]
        row["text_sha256"] = assessments[evidence_id]["text_sha256"]
        row["normalized_text_sha256"] = assessments[evidence_id][
            "normalized_text_sha256"
        ]
        row["text_layer_provenance"] = source_object_class
        row["capture_status"] = "complete_pending_freeze"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    scope["source_materialization"]["manifest_sha256"] = sha256_file(manifest_path)
    scope_path.write_text(
        json.dumps(scope, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    scope_sha256 = sha256_file(scope_path)
    prereg_text = prereg_path.read_text(encoding="utf-8")
    prereg_path.write_text(
        re.sub(
            r"(Scope manifest SHA256:\s*\n\s*`?)[0-9a-f]{64}(`?)",
            rf"\g<1>{scope_sha256}\2",
            prereg_text,
            count=1,
            flags=re.IGNORECASE,
        ),
        encoding="utf-8",
    )
    payload["written"] = written
    payload["scope_sha256"] = scope_sha256
    payload["manifest_sha256"] = scope["source_materialization"]["manifest_sha256"]
    return payload
