from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families import (  # noqa: E402
    far_ext_002e_ds003670_post_ramp_interpretation as far_ext,
)

_ROOT = Path(__file__).resolve().parents[2]
_FAR_ROOT = (
    _ROOT
    / "project"
    / "orthagonal_axis"
    / "orthagonal_dynamics"
    / "finite-amplitude_resilience"
)


def _result() -> dict[str, object]:
    return {
        "protocol_id": "FAR-EXT-002D",
        "dataset_id": "ds003670",
        "global_status": "SIGNAL_TIMEBASE_PASS",
        "primary_family_id": far_ext.F30_FAMILY_ID,
        "exported_signal_technical_status": {
            "event_horizon_status_table": [
                {
                    "family_id": far_ext.F30_FAMILY_ID,
                    "event_key": "event-1",
                    "rho": 1.0,
                    "horizon_sec": 60.0,
                    "eligible": True,
                    "technical_eligible": True,
                    "reason": "eligible",
                },
                {
                    "family_id": far_ext.F30_FAMILY_ID,
                    "event_key": "event-2",
                    "rho": 0.5,
                    "horizon_sec": 60.0,
                    "eligible": False,
                    "technical_eligible": False,
                    "reason": "fixture_invalid",
                },
                {
                    "family_id": "motor_sinusoidal_30hz",
                    "event_key": "other",
                    "rho": 1.0,
                    "horizon_sec": 60.0,
                    "eligible": True,
                    "technical_eligible": True,
                    "reason": "eligible",
                },
            ]
        },
    }


def _scope(
    *,
    result_sha256: str,
    subset_sha256: str,
    subset_count: int,
) -> dict[str, object]:
    return {
        "protocol_id": far_ext.PROTOCOL_ID,
        "execution_status": "PREREGISTRATION_FROZEN",
        "source_documents": [
            dict(document) for document in far_ext.SOURCE_DOCUMENTS
        ],
        "inherited_002a": {
            "certificate_path": str(
                _FAR_ROOT / "results" / "far_ext_002a" / "far_ext_002a.json"
            ),
            "certificate_sha256": "b151976a5a9303ce0a1741331b16cf3ee6cbf4ab77db5a711e3a85d61d3fbd1b",
            "promoted_event_ledger_sha256": "bb14b77523ea9e3a7353a31bf4ad1a7dbad4e79e8e2c036d00f056dc8f3694cd",
            "promoted_event_count": 323,
        },
        "inherited_002d": {
            "result_sha256": result_sha256,
            "f30_subset_sha256": subset_sha256,
            "f30_subset_count": subset_count,
            "f30_direct_subset_sha256": far_ext.canonical_f30_eligible_subset(
                _result(),
                rho=1.0,
            )[1],
            "f30_direct_subset_count": len(
                far_ext.canonical_f30_eligible_subset(_result(), rho=1.0)[0]
            ),
            "result_scope_path": str(
                _FAR_ROOT / "FAR-EXT-002D-SCOPE-0.1.json"
            ),
            "result_scope_sha256": "5698cf6ed821a675ae8b07686e21d006c4785fe3424ccc5516aacee14033b2a7",
            "result_protocol_path": str(
                _FAR_ROOT
                / "012_far_ext_002d_ds003670_exported_signal_eligibility_prereg.md"
            ),
            "result_protocol_sha256": "cc8cf1526a4bb0735c10adcd1bc665a333dc0fe79f2667eb78fcb38583d35ece",
            "qc_rules_path": str(
                _ROOT
                / "project"
                / "orthagonal_axis"
                / "orthagonal_dynamics"
                / "nmd-qc-float"
                / "NMD-QC-FLOAT-RULES-0.3.json"
            ),
            "qc_rules_sha256": "054394322bad951a0a7e245be657fe6842ba4f3fc5807787c2e1e47f5592a6fa",
        },
    }


def _source_fetcher(
    url: str,
    _timeout_seconds: float,
    _max_bytes: int,
) -> tuple[bytes, str, str]:
    del url
    content = """
    Experiment 1 and 2 used the same GX acquisition. The dataset has
    multichannel EEG sampled at 2 kHz, a 30-sec stimulation period and a
    5-sec ramp down period. All participants in Experiment 2 received 1 mA.
    The stimulation start trigger (code:16) and stimulation stop trigger (code:32) define the trial.
    F30 was measured immediately post over 30 sec. EEG data was not examined
    during stimulation because of nonlinear artifacts. EEG was baseline
    corrected and bandpass filtered (0.35-40 Hz).
    EEGout.PostStim uses 41.5*fs and 41.5+30 with spectral lekage and 1:32.
    7bee67e915da44143773e715ee6faa918c3ce766 F30 Experiment 2.
    tDCS EEG artifact context.
    """
    return content.encode("utf-8"), "https://example.test/source", "text/html"


def test_f30_subset_is_canonical_and_excludes_other_families() -> None:
    subset, digest = far_ext.canonical_f30_eligible_subset(_result())
    assert len(subset) == 1
    assert subset[0]["event_key"] == "event-1"
    assert digest == far_ext._json_hash(subset)


def test_bind_002d_requires_result_and_subset_hashes(tmp_path: Path) -> None:
    result_path = tmp_path / "far_ext_002d.json"
    result_path.write_text(json.dumps(_result()), encoding="utf-8")
    result_sha = far_ext.sha256_file(result_path)
    subset, subset_sha = far_ext.canonical_f30_eligible_subset(_result())
    binding = far_ext.bind_002d(
        result_path=result_path,
        expected_sha256=result_sha,
        expected_subset_sha256=subset_sha,
        expected_subset_count=len(subset),
    )
    assert binding["status"] == "PASS"
    assert binding["checks"] == {
        "protocol_id": True,
        "dataset_id": True,
        "global_status": True,
        "primary_family_id": True,
        "result_sha256": True,
        "f30_subset_sha256": True,
        "f30_subset_count": True,
    }
    mismatch = far_ext.bind_002d(
        result_path=result_path,
        expected_sha256="0" * 64,
        expected_subset_sha256=subset_sha,
        expected_subset_count=len(subset),
    )
    assert mismatch["status"] == far_ext.POST_RAMP_SOURCE_MISMATCH


def test_source_only_audit_pass_is_scoped_to_30s_and_1mA(tmp_path: Path) -> None:
    result_path = tmp_path / "far_ext_002d.json"
    result_path.write_text(json.dumps(_result()), encoding="utf-8")
    subset, subset_sha = far_ext.canonical_f30_eligible_subset(_result())
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(
        json.dumps(
            _scope(
                result_sha256=far_ext.sha256_file(result_path),
                subset_sha256=subset_sha,
                subset_count=len(subset),
            )
        ),
        encoding="utf-8",
    )
    payload = far_ext.audit_far_ext_002e(
        scope_path=scope_path,
        repo_root=tmp_path,
        result_path=result_path,
        fetch_documents=True,
        fetcher=_source_fetcher,
    )
    assert payload["global_status"] == (
        far_ext.POST_RAMP_INTERPRETABILITY_BRIDGE_PASS
    )
    assert payload["allowed_later_scope"] == {
        "rho_mA": [1.0],
        "post_duration_sec": [30.0],
        "source_defined_offset_after_ramp_down_sec": [1.5],
        "technical_60s_horizon_interpretability_transfer": False,
    }
    assert payload["source_semantics"]["amplitude_transfer"]["0.5_mA"][
        "status"
    ] == far_ext.POST_RAMP_METHOD_LIMITED
    assert payload["source_semantics"]["amplitude_transfer"]["1.0_mA"][
        "status"
    ] == far_ext.POST_RAMP_INTERPRETABILITY_BRIDGE_PASS
    assert payload["source_semantics"]["published_post_interval"][
        "directly_after_ramp_down"
    ] is False
    assert payload["audit_scope"]["signal_payloads_opened"] is False
    assert payload["far_ext_003b_authorized"] is False
    assert payload["post_stimulation_biological_interpretability"] == (
        "BRIDGE_SCOPED_NOT_ARTIFACT_FREE"
    )
    assert "ARTIFACT_FREE_PASS" not in json.dumps(payload)
    assert payload["mnps_calculated"] is False
    assert payload["far_calculated"] is False


def test_source_marker_failure_is_unresolved(tmp_path: Path) -> None:
    result_path = tmp_path / "far_ext_002d.json"
    result_path.write_text(json.dumps(_result()), encoding="utf-8")
    subset, subset_sha = far_ext.canonical_f30_eligible_subset(_result())
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(
        json.dumps(
            _scope(
                result_sha256=far_ext.sha256_file(result_path),
                subset_sha256=subset_sha,
                subset_count=len(subset),
            )
        ),
        encoding="utf-8",
    )

    def missing_marker_fetcher(
        url: str,
        timeout_seconds: float,
        max_bytes: int,
    ) -> tuple[bytes, str, str]:
        del url, timeout_seconds, max_bytes
        return b"only unrelated text", "https://example.test/source", "text/html"

    payload = far_ext.audit_far_ext_002e(
        scope_path=scope_path,
        repo_root=tmp_path,
        result_path=result_path,
        fetch_documents=True,
        fetcher=missing_marker_fetcher,
    )
    assert payload["global_status"] == (
        far_ext.POST_RAMP_INTERPRETABILITY_UNRESOLVED
    )


def test_inherited_contract_binds_all_declared_artifacts(tmp_path: Path) -> None:
    result_path = tmp_path / "far_ext_002d.json"
    result_path.write_text(json.dumps(_result()), encoding="utf-8")
    subset, subset_sha = far_ext.canonical_f30_eligible_subset(_result())
    scope = _scope(
        result_sha256=far_ext.sha256_file(result_path),
        subset_sha256=subset_sha,
        subset_count=len(subset),
    )
    binding = far_ext.bind_inherited_contract(
        scope=scope,
        repo_root=tmp_path,
        result_path=result_path,
    )
    assert binding["status"] == "PASS"
    assert all(binding["checks"].values())
    assert binding["002a"]["ledger_sha256"] == (
        "bb14b77523ea9e3a7353a31bf4ad1a7dbad4e79e8e2c036d00f056dc8f3694cd"
    )
    assert binding["qc_rules"]["sha256"] == (
        "054394322bad951a0a7e245be657fe6842ba4f3fc5807787c2e1e47f5592a6fa"
    )


def test_inherited_contract_rejects_declared_hash_mismatch(tmp_path: Path) -> None:
    result_path = tmp_path / "far_ext_002d.json"
    result_path.write_text(json.dumps(_result()), encoding="utf-8")
    subset, subset_sha = far_ext.canonical_f30_eligible_subset(_result())
    scope = _scope(
        result_sha256=far_ext.sha256_file(result_path),
        subset_sha256=subset_sha,
        subset_count=len(subset),
    )
    broken_scope = copy.deepcopy(scope)
    broken_scope["inherited_002d"]["qc_rules_sha256"] = "0" * 64
    binding = far_ext.bind_inherited_contract(
        scope=broken_scope,
        repo_root=tmp_path,
        result_path=result_path,
    )
    assert binding["status"] == far_ext.POST_RAMP_SOURCE_MISMATCH
    assert binding["checks"]["qc_rules_sha256"] is False


def test_scope_must_be_frozen(tmp_path: Path) -> None:
    scope = {
        "protocol_id": far_ext.PROTOCOL_ID,
        "execution_status": "DESIGN_SCOPE_ONLY",
        "source_documents": [
            dict(document) for document in far_ext.SOURCE_DOCUMENTS
        ],
    }
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    with pytest.raises(ValueError, match="far_ext_002e_scope_not_frozen"):
        far_ext.load_scope(scope_path)


def test_forbidden_source_payload_is_rejected() -> None:
    document = {
        "id": "payload",
        "kind": "signal",
        "url": "https://example.test/file.fdt",
        "allow_signal_samples": False,
        "required_markers": [],
    }
    result = far_ext._fetch_source_document(
        document,
        timeout_seconds=1.0,
        max_bytes=100,
        fetcher=_source_fetcher,
    )
    assert result["status"] == "REJECTED"
    assert result["reason"] == "forbidden_payload_suffix"


def test_source_fetch_rejects_unsafe_final_url() -> None:
    document = {
        "id": "metadata",
        "kind": "source",
        "url": "https://example.test/source",
        "allow_signal_samples": False,
        "required_markers": [],
    }

    def unsafe_fetcher(
        url: str,
        timeout_seconds: float,
        max_bytes: int,
    ) -> tuple[bytes, str, str]:
        del url, timeout_seconds, max_bytes
        return b"metadata", "file:///K:/payload.fdt", "application/octet-stream"

    result = far_ext._fetch_source_document(
        document,
        timeout_seconds=1.0,
        max_bytes=100,
        fetcher=unsafe_fetcher,
    )
    assert result["status"] == far_ext.POST_RAMP_INTERPRETABILITY_UNRESOLVED
    assert result["reason"] == "unsafe_final_url_scheme"


def test_disabled_source_fetch_cannot_pass(tmp_path: Path) -> None:
    result_path = tmp_path / "far_ext_002d.json"
    result_path.write_text(json.dumps(_result()), encoding="utf-8")
    subset, subset_sha = far_ext.canonical_f30_eligible_subset(_result())
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(
        json.dumps(
            _scope(
                result_sha256=far_ext.sha256_file(result_path),
                subset_sha256=subset_sha,
                subset_count=len(subset),
            )
        ),
        encoding="utf-8",
    )
    payload = far_ext.audit_far_ext_002e(
        scope_path=scope_path,
        repo_root=tmp_path,
        result_path=result_path,
        fetch_documents=False,
    )
    assert payload["global_status"] == (
        far_ext.POST_RAMP_INTERPRETABILITY_UNRESOLVED
    )
