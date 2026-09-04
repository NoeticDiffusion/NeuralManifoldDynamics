from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mndm.dynamical_families import (  # noqa: E402
    far_ext_002e_ds003670_post_ramp_interpretation as far_ext,
)
from od_far_ext_002e_scope_0_2_source_rematerialization_gate import (  # noqa: E402
    DEFAULT_OUTPUT,
    DEFAULT_REPORT,
    PREREG_PATH,
    SCOPE_PATH,
    run_gate,
)


_ROOT = Path(__file__).resolve().parents[2]
_FAR_ROOT = (
    _ROOT
    / "project"
    / "orthagonal_axis"
    / "orthagonal_dynamics"
    / "finite-amplitude_resilience"
)
_SNAPSHOT_ROOT = _FAR_ROOT / "source_snapshots" / "far_ext_002e_scope_0.2"


def test_normalizer_handles_entities_tags_nbsp_and_dashes() -> None:
    normalized = far_ext.normalize_source_text(
        "<span>Intensity&nbsp;</span> 1\u2013mA &amp; 30&nbsp;Hz"
    )
    assert normalized == "intensity 1-ma & 30 hz"


def test_code_normalization_does_not_strip_comparison_tokens() -> None:
    source = "EEGout.PostStim{mm,1} = x(41.5*fs):y((41.5+30)*fs);"
    result = far_ext.evaluate_source_assertions(
        source,
        [
            {
                "id": "A5",
                "required": True,
                "all_of": ["EEGout.PostStim", "41.5*fs", "41.5+30"],
            }
        ],
        strip_tags=False,
    )
    assert result["status"] == "PASS"


def test_scope_02_stays_design_only_with_incomplete_sources() -> None:
    payload = far_ext.audit_far_ext_002e(
        scope_path=SCOPE_PATH,
        repo_root=_ROOT,
        result_path=_FAR_ROOT / "results" / "far_ext_002d" / "far_ext_002d.json",
        fetch_documents=True,
    )
    assert payload["global_status"] == far_ext.POST_RAMP_SOURCE_MISMATCH
    assert payload["global_reason"] == (
        "scope_binding_failed:ValueError"
    )
    assert "source_objects" not in payload
    assert payload["audit_scope"]["signal_payloads_opened"] is False
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    results = [
        far_ext.read_frozen_source_object(
            document,
            repository_root=_ROOT,
        )
        for document in scope["source_objects"][:2]
    ]
    assert [result["reason"] for result in results] == [
        "source_snapshot_body_missing",
        "source_snapshot_body_missing",
    ]


def test_frozen_candidate_audit_is_unresolved_on_incomplete_sources(
    tmp_path: Path,
) -> None:
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    scope["execution_status"] = "PREREGISTRATION_FROZEN"
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    payload = far_ext.audit_far_ext_002e(
        scope_path=scope_path,
        repo_root=_ROOT,
        result_path=_FAR_ROOT / "results" / "far_ext_002d" / "far_ext_002d.json",
        fetch_documents=True,
    )
    assert payload["global_status"] == (
        far_ext.POST_RAMP_INTERPRETABILITY_UNRESOLVED
    )
    assert payload["global_reason"] == (
        "required_frozen_source_evidence_unresolved"
    )
    assert payload["source_manifest"]["status"] == "PASS"
    assert [document["reason"] for document in payload["source_objects"][:2]] == [
        "source_snapshot_body_missing",
        "source_snapshot_body_missing",
    ]


def test_disabled_source_reads_cannot_pass(tmp_path: Path) -> None:
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    scope["execution_status"] = "PREREGISTRATION_FROZEN"
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(json.dumps(scope), encoding="utf-8")
    payload = far_ext.audit_far_ext_002e(
        scope_path=scope_path,
        repo_root=_ROOT,
        result_path=_FAR_ROOT / "results" / "far_ext_002d" / "far_ext_002d.json",
        fetch_documents=False,
    )
    assert payload["global_status"] == (
        far_ext.POST_RAMP_INTERPRETABILITY_UNRESOLVED
    )
    assert payload["global_reason"] == (
        "required_frozen_source_evidence_unresolved"
    )


def test_frozen_source_hash_mismatch_is_unresolved(tmp_path: Path) -> None:
    snapshot = tmp_path / "source.txt"
    text = tmp_path / "source-extract.txt"
    snapshot.write_bytes(b"source")
    text.write_text("required phrase", encoding="utf-8")
    document = {
        "id": "E1",
        "evidence_id": "E1",
        "kind": "source",
        "role": "test",
        "allow_signal_samples": False,
        "normalization": {"strip_tags": True},
        "retrieval": {
            "mode": "frozen_snapshot",
            "snapshot_path": "source.txt",
            "text_path": "source-extract.txt",
            "expected_content_sha256": "0" * 64,
            "expected_text_sha256": far_ext.sha256_file(text),
            "min_content_bytes": 1,
            "max_bytes": 1024,
        },
        "assertions": [],
    }
    result = far_ext.read_frozen_source_object(
        document,
        repository_root=tmp_path,
    )
    assert result["status"] == (
        far_ext.POST_RAMP_INTERPRETABILITY_UNRESOLVED
    )
    assert result["hash_checks"]["snapshot_sha256"] is False


def test_metadata_only_snapshot_cannot_satisfy_body_requirement(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "source.xml"
    text = tmp_path / "source.txt"
    snapshot.write_text("<coredata>metadata only</coredata>", encoding="utf-8")
    text.write_text("A1 required source phrase", encoding="utf-8")
    document = {
        "id": "E1",
        "evidence_id": "E1",
        "kind": "source",
        "role": "test",
        "allow_signal_samples": False,
        "normalization": {"strip_tags": True},
        "retrieval": {
            "mode": "frozen_snapshot",
            "snapshot_path": "source.xml",
            "text_path": "source.txt",
            "expected_content_sha256": far_ext.sha256_file(snapshot),
            "expected_text_sha256": far_ext.sha256_file(text),
            "expected_normalized_text_sha256": far_ext.sha256_file(text),
            "required_snapshot_markers": ["<body", "<originalText"],
            "min_content_bytes": 1,
            "max_bytes": 1024,
        },
        "assertions": [],
    }
    result = far_ext.read_frozen_source_object(
        document,
        repository_root=tmp_path,
    )
    assert result["status"] == (
        far_ext.POST_RAMP_INTERPRETABILITY_UNRESOLVED
    )
    assert result["reason"] == "source_snapshot_body_missing"


def test_normalized_text_hash_is_bound(tmp_path: Path) -> None:
    snapshot = tmp_path / "source.xml"
    text = tmp_path / "source.txt"
    snapshot.write_text("<body>source</body>", encoding="utf-8")
    text.write_text("source", encoding="utf-8")
    document = {
        "id": "E1",
        "evidence_id": "E1",
        "kind": "source",
        "role": "test",
        "allow_signal_samples": False,
        "normalization": {"strip_tags": True},
        "retrieval": {
            "mode": "frozen_snapshot",
            "snapshot_path": "source.xml",
            "text_path": "source.txt",
            "expected_content_sha256": far_ext.sha256_file(snapshot),
            "expected_text_sha256": far_ext.sha256_file(text),
            "expected_normalized_text_sha256": "0" * 64,
            "required_snapshot_markers": ["<body"],
            "min_content_bytes": 1,
            "max_bytes": 1024,
        },
        "assertions": [],
    }
    result = far_ext.read_frozen_source_object(
        document,
        repository_root=tmp_path,
    )
    assert result["status"] == (
        far_ext.POST_RAMP_INTERPRETABILITY_UNRESOLVED
    )
    assert result["hash_checks"]["normalized_text_sha256"] is False


def test_snapshot_path_and_suffix_are_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.fdt"
    outside.write_bytes(b"not a source")
    document = {
        "id": "E1",
        "evidence_id": "E1",
        "kind": "source",
        "role": "test",
        "allow_signal_samples": False,
        "retrieval": {
            "mode": "frozen_snapshot",
            "snapshot_path": "../outside.fdt",
            "text_path": "../outside.fdt",
        },
        "assertions": [],
    }
    result = far_ext.read_frozen_source_object(
        document,
        repository_root=tmp_path,
    )
    assert result["status"] == "REJECTED"
    assert result["reason"] == "forbidden_snapshot_suffix"


def test_gate_refuses_unfrozen_scope_before_writing(tmp_path: Path) -> None:
    output_json = tmp_path / "far_ext_002e.json"
    output_report = tmp_path / "far_ext_002e.md"
    with pytest.raises(ValueError, match="far_ext_002e_scope_not_frozen"):
        run_gate(
            repo_root=_ROOT,
            scope_path=SCOPE_PATH,
            prereg_path=PREREG_PATH,
            output_json=output_json,
            output_report=output_report,
            fetch_documents=True,
        )
    assert not output_json.exists()
    assert not output_report.exists()
    assert DEFAULT_OUTPUT != output_json
    assert DEFAULT_REPORT != output_report
    assert (
        _FAR_ROOT / "results" / "far_ext_002e" / "far_ext_002e.json"
    ).is_file()


def test_gate_refuses_existing_0_2_archive(tmp_path: Path) -> None:
    output_json = tmp_path / "far_ext_002e.json"
    output_report = tmp_path / "far_ext_002e.md"
    output_json.write_text("write once", encoding="utf-8")
    with pytest.raises(
        FileExistsError,
        match="refusing_to_overwrite_far_ext_002e_scope_0_2_archive",
    ):
        run_gate(
            repo_root=_ROOT,
            scope_path=SCOPE_PATH,
            prereg_path=PREREG_PATH,
            output_json=output_json,
            output_report=output_report,
        )


def _padded_complete_xml(body: str, *, min_bytes: int = 10000) -> bytes:
    pad = " padding" * ((min_bytes // 8) + 1)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<article><body><ce:para xmlns:ce=\"http://www.elsevier.com/xml/common/dtd\">"
        f"{body}{pad}</ce:para></body></article>"
    )
    encoded = xml.encode("utf-8")
    assert len(encoded) >= min_bytes
    return encoded


def test_current_e1_e2_candidates_fail_complete_xml_assessment() -> None:
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in scope["source_objects"]}
    e1 = far_ext.assess_complete_xml_candidate(
        (_SNAPSHOT_ROOT / "E1_f30_followup.xml").read_bytes(),
        assertions=by_id["E1"]["assertions"],
        identity_markers=far_ext.E1_IDENTITY_MARKERS,
        required_snapshot_markers=by_id["E1"]["retrieval"][
            "required_snapshot_markers"
        ],
        min_content_bytes=int(by_id["E1"]["retrieval"]["min_content_bytes"]),
    )
    e2 = far_ext.assess_complete_xml_candidate(
        (_SNAPSHOT_ROOT / "E2_f30_response.xml").read_bytes(),
        assertions=by_id["E2"]["assertions"],
        identity_markers=far_ext.E2_IDENTITY_MARKERS,
        required_snapshot_markers=by_id["E2"]["retrieval"][
            "required_snapshot_markers"
        ],
        min_content_bytes=int(by_id["E2"]["retrieval"]["min_content_bytes"]),
    )
    assert e1["reason"] == "source_snapshot_body_missing"
    assert e2["reason"] == "source_snapshot_body_missing"
    dry_run = far_ext.bind_complete_e1_e2_xml(
        repo_root=_ROOT,
        e1_xml=_SNAPSHOT_ROOT / "E1_f30_followup.xml",
        e2_xml=_SNAPSHOT_ROOT / "E2_f30_response.xml",
        source_object_class="COMPLETE_OA_SOURCE_OBJECT",
        apply=False,
    )
    assert dry_run["current_candidate_status"] == "E1_E2_INCOMPLETE"
    assert dry_run["execution_status"] == "DESIGN_SCOPE_ONLY"
    assert dry_run["live_0_2_run"] is False
    assert not (_FAR_ROOT / "results" / "far_ext_002e_scope_0.2").exists()


def test_bind_complete_xml_dry_run_does_not_write_live_tree(
    tmp_path: Path,
) -> None:
    e1_body = (
        "Frontal HD-tACS enhances behavioral and EEG biomarkers of vigilance "
        "in continuous attention task. DOI 10.1016/j.brs.2024.05.009. "
        "EEG outcomes were calculated immediately post over 30 sec. "
        "EEG data during stimulation was not examined in order to avoid "
        "nonlinear artifacts. EEG data were baseline corrected and "
        "bandpass filtered (0.35-40 Hz)."
    )
    e2_body = (
        "Response regarding: Frontal HD-tACS enhances behavioral and EEG "
        "biomarkers of vigilance in continuous attention task. "
        "DOI 10.1016/j.brs.2024.11.004. Measures included "
        "intensity/frequency matching (1 mA, 30 Hz)."
    )
    e1_xml = tmp_path / "e1.xml"
    e2_xml = tmp_path / "e2.xml"
    e1_xml.write_bytes(_padded_complete_xml(e1_body))
    e2_xml.write_bytes(_padded_complete_xml(e2_body))
    payload = far_ext.bind_complete_e1_e2_xml(
        repo_root=_ROOT,
        e1_xml=e1_xml,
        e2_xml=e2_xml,
        source_object_class="COMPLETE_OA_SOURCE_OBJECT",
        apply=False,
    )
    assert payload["current_candidate_status"] == "E1_E2_COMPLETE_PENDING_FREEZE"
    assert payload["assessments"]["E1"]["status"] == "PASS"
    assert payload["assessments"]["E2"]["status"] == "PASS"
    assert payload["apply"] is False
    assert "written" not in payload
    live_scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    assert live_scope["execution_status"] == "DESIGN_SCOPE_ONLY"
    assert live_scope["source_materialization"]["current_candidate_status"] == (
        "E1_E2_INCOMPLETE"
    )
    assert not (_FAR_ROOT / "results" / "far_ext_002e_scope_0.2").exists()


def test_bind_apply_writes_only_inside_isolated_tree(tmp_path: Path) -> None:
    live_e1 = (_SNAPSHOT_ROOT / "E1_f30_followup.xml").read_bytes()
    live_e2 = (_SNAPSHOT_ROOT / "E2_f30_response.xml").read_bytes()
    repo = tmp_path / "repo"
    far_root = (
        repo
        / "project"
        / "orthagonal_axis"
        / "orthagonal_dynamics"
        / "finite-amplitude_resilience"
    )
    snap = far_root / "source_snapshots" / "far_ext_002e_scope_0.2"
    snap.mkdir(parents=True)
    shutil.copy(SCOPE_PATH, far_root / "FAR-EXT-002E-SCOPE-0.2.json")
    shutil.copy(
        PREREG_PATH,
        far_root / "014_far_ext_002e_scope_0.2_source_rematerialization_prereg.md",
    )
    for path in _SNAPSHOT_ROOT.iterdir():
        shutil.copy(path, snap / path.name)
    e1_xml = tmp_path / "e1.xml"
    e2_xml = tmp_path / "e2.xml"
    e1_xml.write_bytes(
        _padded_complete_xml(
            "Frontal HD-tACS enhances behavioral and EEG biomarkers of vigilance "
            "in continuous attention task. DOI 10.1016/j.brs.2024.05.009. "
            "EEG outcomes were calculated immediately post over 30 sec. "
            "EEG data during stimulation was not examined in order to avoid "
            "nonlinear artifacts. EEG data were baseline corrected and "
            "bandpass filtered (0.35-40 Hz)."
        )
    )
    e2_xml.write_bytes(
        _padded_complete_xml(
            "Response regarding: Frontal HD-tACS enhances behavioral and EEG "
            "biomarkers of vigilance in continuous attention task. "
            "DOI 10.1016/j.brs.2024.11.004. Measures included "
            "intensity/frequency matching (1 mA, 30 Hz)."
        )
    )
    payload = far_ext.bind_complete_e1_e2_xml(
        repo_root=repo,
        e1_xml=e1_xml,
        e2_xml=e2_xml,
        source_object_class="COMPLETE_OA_SOURCE_OBJECT",
        apply=True,
    )
    isolated_scope = json.loads(
        (far_root / "FAR-EXT-002E-SCOPE-0.2.json").read_text(encoding="utf-8")
    )
    live_scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    assert payload["apply"] is True
    assert payload["current_candidate_status"] == "E1_E2_COMPLETE_PENDING_FREEZE"
    assert isolated_scope["execution_status"] == "DESIGN_SCOPE_ONLY"
    assert isolated_scope["source_materialization"]["execution_authorized"] is False
    assert isolated_scope["source_materialization"]["current_candidate_status"] == (
        "E1_E2_COMPLETE_PENDING_FREEZE"
    )
    assert (snap / "E1_f30_followup.xml").read_bytes() != live_e1
    assert (_SNAPSHOT_ROOT / "E1_f30_followup.xml").read_bytes() == live_e1
    assert (_SNAPSHOT_ROOT / "E2_f30_response.xml").read_bytes() == live_e2
    assert live_scope["source_materialization"]["current_candidate_status"] == (
        "E1_E2_INCOMPLETE"
    )
    assert live_scope["execution_status"] == "DESIGN_SCOPE_ONLY"
    assert not (_FAR_ROOT / "results" / "far_ext_002e_scope_0.2").exists()


def test_bind_refuses_authorized_scope_and_existing_archive(tmp_path: Path) -> None:
    def _far_tree(name: str) -> Path:
        repo = tmp_path / name
        far_root = (
            repo
            / "project"
            / "orthagonal_axis"
            / "orthagonal_dynamics"
            / "finite-amplitude_resilience"
        )
        snap = far_root / "source_snapshots" / "far_ext_002e_scope_0.2"
        snap.mkdir(parents=True)
        scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
        if name == "authorized":
            scope["source_materialization"]["execution_authorized"] = True
        (far_root / "FAR-EXT-002E-SCOPE-0.2.json").write_text(
            json.dumps(scope), encoding="utf-8"
        )
        (far_root / "014_far_ext_002e_scope_0.2_source_rematerialization_prereg.md").write_text(
            "Scope manifest SHA256:\n`abc`\n", encoding="utf-8"
        )
        (snap / "manifest.json").write_text("{}", encoding="utf-8")
        if name == "archive":
            out = far_root / "results" / "far_ext_002e_scope_0.2"
            out.mkdir(parents=True)
            (out / "far_ext_002e.json").write_text("{}", encoding="utf-8")
        return repo

    with pytest.raises(ValueError, match="refusing_to_mutate_authorized_scope"):
        far_ext.bind_complete_e1_e2_xml(
            repo_root=_far_tree("authorized"),
            e1_xml=tmp_path / "missing.xml",
            e2_xml=tmp_path / "missing.xml",
            source_object_class="COMPLETE_OA_SOURCE_OBJECT",
            apply=True,
        )
    with pytest.raises(
        FileExistsError,
        match="refusing_to_mutate_after_far_ext_002e_scope_0_2_archive",
    ):
        far_ext.bind_complete_e1_e2_xml(
            repo_root=_far_tree("archive"),
            e1_xml=tmp_path / "missing.xml",
            e2_xml=tmp_path / "missing.xml",
            source_object_class="COMPLETE_OA_SOURCE_OBJECT",
            apply=True,
        )


def test_bind_refuses_to_mutate_frozen_scope(tmp_path: Path) -> None:
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    scope["execution_status"] = "PREREGISTRATION_FROZEN"
    repo = tmp_path / "repo"
    far_root = (
        repo
        / "project"
        / "orthagonal_axis"
        / "orthagonal_dynamics"
        / "finite-amplitude_resilience"
    )
    snap = far_root / "source_snapshots" / "far_ext_002e_scope_0.2"
    snap.mkdir(parents=True)
    (far_root / "FAR-EXT-002E-SCOPE-0.2.json").write_text(
        json.dumps(scope), encoding="utf-8"
    )
    (far_root / "014_far_ext_002e_scope_0.2_source_rematerialization_prereg.md").write_text(
        "Scope manifest SHA256:\n`abc`\n", encoding="utf-8"
    )
    (snap / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="refusing_to_mutate_frozen_scope"):
        far_ext.bind_complete_e1_e2_xml(
            repo_root=repo,
            e1_xml=tmp_path / "missing.xml",
            e2_xml=tmp_path / "missing.xml",
            source_object_class="COMPLETE_OA_SOURCE_OBJECT",
            apply=True,
        )


def test_scope_has_four_required_source_objects() -> None:
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    assert scope["scope_version"] == "0.2"
    assert {row["evidence_id"] for row in scope["source_objects"]} == {
        "E1",
        "E2",
        "E3",
        "E4",
    }
    assert all(
        (_ROOT / row["retrieval"]["snapshot_path"]).is_file()
        for row in scope["source_objects"]
    )
    assert _SNAPSHOT_ROOT.is_dir()


def test_scope_loader_rejects_wrong_version(tmp_path: Path) -> None:
    scope = json.loads(SCOPE_PATH.read_text(encoding="utf-8"))
    scope["scope_version"] = "0.1"
    scope["execution_status"] = "PREREGISTRATION_FROZEN"
    path = tmp_path / "scope.json"
    path.write_text(json.dumps(scope), encoding="utf-8")
    with pytest.raises(ValueError, match="far_ext_002e_scope_version_mismatch"):
        far_ext.load_scope(path, expected_scope_version="0.2")
