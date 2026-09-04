"""FAR-SCOUT-003 local archive perturbation sweep.

The sweep reads local archive documentation and structured metadata artifacts
only.  It performs discovery first and source-specific promotion checks second.
Raw neural payloads, NMD outputs, downloads, and biological outcomes are never
opened.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


PROTOCOL_ID = "FAR-SCOUT-003"
NO_PERTURBATION = "NO_PERTURBATION"
DIRECTION_ONLY = "DIRECTION_ONLY"
RHO_CANDIDATE = "RHO_CANDIDATE"
TIMING_CANDIDATE = "TIMING_CANDIDATE"
FULL_METADATA_CANDIDATE = "FULL_METADATA_CANDIDATE"
SOURCE_UNCERTAIN = "SOURCE_UNCERTAIN"

PROMOTION_PASS = "PROMOTION_PASS"
NMD_TIMEBASE_INCOMPATIBLE = "NMD_TIMEBASE_INCOMPATIBLE"
PERTURBATION_VALID_NMD_TIMEBASE_INCOMPATIBLE = (
    "PERTURBATION_VALID_NMD_TIMEBASE_INCOMPATIBLE"
)
SIGNAL_CONTRACT_UNRESOLVED = "SIGNAL_CONTRACT_UNRESOLVED"
INCONCLUSIVE = "INCONCLUSIVE"

ALLOWED_METADATA_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".tsv",
    ".txt",
    ".yaml",
    ".yml",
}
EXCLUDED_DIRECTORIES = {
    ".git",
    ".datalad",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
}
TERM_PATTERNS = {
    "physical_magnitude": re.compile(
        r"\b(amplitude|contrast|current|dose|intensity|luminance|power|"
        r"irradiance|laser[_ -]?power|stim[_ -]?power|voltage)\b",
        re.I,
    ),
    "physical_units": re.compile(
        r"(?:\b\d+(?:\.\d+)?\s*%|\bmW\b|\b(?:u|μ|µ)A\b|\bmg/?kg\b|"
        r"\b(?:u|μ|µ)m\b|\bV\b)",
        re.I,
    ),
    "timing": re.compile(
        r"\b(start[_ -]?time|stop[_ -]?time|onset|offset|stimulus time|"
        r"event time|timestamp|ttl|sync|inter[- ]stimulus|ITI)\b",
        re.I,
    ),
    "continuous_signal": re.compile(
        r"\b(continuous|eeg|lfp|ecephys|neuropixels|spike|calcium|"
        r"widefield|ecog)\b",
        re.I,
    ),
    "repetition": re.compile(
        r"\b(repeated|repetition|trial|session|block|replication|mouse|"
        r"subject|animal)\b",
        re.I,
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_text(path: Path, limit: int = 512_000) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.read(limit)


def _walk_metadata_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    files: list[Path] = []
    for directory, subdirectories, filenames in os.walk(root):
        subdirectories[:] = [
            name for name in subdirectories if name not in EXCLUDED_DIRECTORIES
        ]
        for filename in filenames:
            path = Path(directory) / filename
            if path.suffix.lower() in ALLOWED_METADATA_SUFFIXES:
                files.append(path)
    return sorted(files)


def _term_inventory(files: list[Path]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    suffixes: Counter[str] = Counter()
    bytes_read = 0
    for path in files:
        suffixes[path.suffix.lower()] += 1
        try:
            text = _read_text(path).lower()
        except OSError:
            continue
        bytes_read += len(text.encode("utf-8", errors="ignore"))
        for name, pattern in TERM_PATTERNS.items():
            if pattern.search(text):
                counts[name] += 1
    return {
        "metadata_file_count": len(files),
        "metadata_suffix_counts": dict(sorted(suffixes.items())),
        "metadata_bytes_scanned": bytes_read,
        "term_file_counts": dict(sorted(counts.items())),
    }


def _relative_records(files: list[tuple[Path, Path]]) -> list[dict[str, Any]]:
    return [
        {
            "source_root": str(root),
            "relative_path": str(path.relative_to(root)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
        }
        for root, path in files[:200]
    ]


def _archive_specs(
    repo_root: Path,
    source_roots: Mapping[str, Path] | None = None,
    subsystems_root: Path | None = None,
) -> list[dict[str, Any]]:
    subsystems_root = subsystems_root or Path(
        os.environ.get("NMD_SUBSYSTEMS_ROOT") or r"J:\repos\ndt-subsystems"
    )
    specs = [
        {
            "dataset_id": "dandi_000458",
            "source_family": "dandi",
            "roots": [
                repo_root
                / "project/orthagonal_axis/orthagonal_dynamics/"
                "finite-amplitude_resilience/results/far_scout_002"
            ],
            "profile": "dandi_000458_carried_forward",
            "nmd_window_sec": 4.0,
        },
        {
            "dataset_id": "dandi_000690",
            "source_family": "dandi",
            "source_institution": "Allen Institute / OpenScope",
            "roots": [
                subsystems_root / "project/HELS",
            ],
            "profile": "dandi_000690_contrast_uncertain",
            "nmd_window_sec": 4.0,
            "evidence_relative_paths": [
                "dandi000690_stage0_gate_report.md",
                "dandi000690_stage0_summary.json",
                "dandi000690_assets.csv",
                "dandi000690_source_clarification_request.md",
            ],
        },
        {
            "dataset_id": "ibl_bwm_000409",
            "source_family": "ibl",
            "roots": [
                subsystems_root / "articles/IBL_BWM_CCTG",
                subsystems_root / "project/SDSR",
            ],
            "profile": "ibl_contrast_timing_uncertain",
            "nmd_window_sec": 4.0,
            "evidence_relative_paths_by_root": [
                ["README.md", "results/coverage_summary.json"],
                ["IBL.md"],
            ],
        },
        {
            "dataset_id": "dryad_vpm_s1_dbrv15f3n",
            "source_family": "dryad",
            "roots": [
                subsystems_root / "articles/DataDryad_Garett",
            ],
            "profile": "dryad_power_no_continuous_signal",
            "nmd_window_sec": 4.0,
            "evidence_relative_paths": [
                "README.md",
                "data/MANIFEST.md",
            ],
        },
        {
            "dataset_id": "dryad_cecchetto_dbrv15f23",
            "source_family": "dryad",
            "roots": [
                Path(r"J:\repos\m3-bridge\project\diary"),
            ],
            "profile": "dryad_peri_event_only",
            "nmd_window_sec": 4.0,
            "evidence_relative_paths": [
                "044_20260724_cecchetto_dryad_E001_audit.md",
                "062_20260803_cecchetto_dryad_E001_micro_meso_macro_audit.md",
            ],
        },
        {
            "dataset_id": "dandi_000574",
            "source_family": "dandi",
            "roots": [
                Path(r"J:\repos\m3-bridge\project\diary"),
                Path(r"J:\repos\m3-bridge\articles\DANDI\000574"),
            ],
            "profile": "dandi_000574_no_replacement_qualification",
            "nmd_window_sec": 4.0,
            "evidence_relative_paths": [
                "S1_feature_contract.md",
                "038_20260719_dandi_000574_E000_pass_signal_architecture.md",
                "039_20260719_dandi_000574_E001_pass_true_sampling_rates.md",
            ],
        },
    ]
    if source_roots:
        for spec in specs:
            override = source_roots.get(str(spec["dataset_id"]))
            if override is not None:
                spec["roots"] = [Path(override)]
    return specs


def _profile_semantics(profile: str) -> dict[str, Any]:
    common = {
        "modality": None,
        "candidate_rho_field": None,
        "units": None,
        "candidate_v": None,
        "onset_field": None,
        "offset_field": None,
        "n_levels_known": None,
        "repetition_evidence": None,
        "continuous_signal_evidence": None,
        "isolated_post_horizon_sec": None,
        "local_raw_signal_available": False,
    }
    if profile == "dandi_000458_carried_forward":
        common.update(
            {
                "modality": "surface EEG with mesoscopic electrophysiology",
                "candidate_rho_field": "estim_current",
                "units": "μA",
                "candidate_v": "electrical stimulation site/depth",
                "onset_field": "start_time",
                "offset_field": "stop_time",
                "n_levels_known": 11,
                "repetition_evidence": "FAR-002/FAR-003A frozen ledger",
                "continuous_signal_evidence": (
                    "native EEG established by prior FAR-003A certificate; "
                    "not reopened by this scout"
                ),
                "isolated_post_horizon_sec": 2.0,
            }
        )
    elif profile == "dandi_000690_contrast_uncertain":
        common.update(
            {
                "modality": "Neuropixels ecephys with reported LFP",
                "candidate_rho_field": "visual contrast token in table names",
                "candidate_v": "visual stimulus family/content",
                "onset_field": "session stimulus tables, not frozen",
                "offset_field": "session stimulus tables, not frozen",
                "n_levels_known": "Cntst0/Cntst1 tokens only",
                "repetition_evidence": "25 sessions and repeated stimulus tables",
                "continuous_signal_evidence": "Neuropixels/LFP reported",
            }
        )
    elif profile == "ibl_contrast_timing_uncertain":
        common.update(
            {
                "modality": "Neuropixels ecephys with reported LFP",
                "candidate_rho_field": "signed visual contrast",
                "units": "fraction/% documented in local archive",
                "candidate_v": "stimulus side",
                "onset_field": "IBL trial stimulus times",
                "offset_field": "not frozen in local archive",
                "n_levels_known": "zero and graded contrast levels",
                "repetition_evidence": "149 local sessions, 139 mice in release description",
                "continuous_signal_evidence": "Neuropixels/LFP",
            }
        )
    elif profile == "dryad_power_no_continuous_signal":
        common.update(
            {
                "modality": "VPm/S1 electrophysiology with optical modulation",
                "candidate_rho_field": "optogenetic modulation parameter",
                "candidate_v": "laser-on/off and tone condition",
                "onset_field": "not frozen in outcome-free metadata",
                "offset_field": "not frozen in outcome-free metadata",
                "n_levels_known": "not frozen in outcome-free metadata",
                "repetition_evidence": "VPm/S1 multi-session deposit described",
                "continuous_signal_evidence": (
                    "extracellular and broadband assets documented; continuous "
                    "NMD contract not frozen"
                ),
            }
        )
    elif profile == "dryad_peri_event_only":
        common.update(
            {
                "modality": "peri-event optical/LFP/EEG summaries",
                "candidate_rho_field": "air-puff/stimulus parameter not established",
                "candidate_v": "depth/state branch",
                "onset_field": "peri-event window only",
                "offset_field": "peri-event window only",
                "n_levels_known": None,
                "repetition_evidence": "67 files, averaged branches",
                "continuous_signal_evidence": "no continuous raw stream",
            }
        )
    elif profile == "dandi_000574_no_replacement_qualification":
        common.update(
            {
                "modality": "scalp EEG with MTL LFP and units",
                "candidate_rho_field": "working-memory set size",
                "candidate_v": "task condition / set-size class",
                "onset_field": "concatenated trial-window position",
                "offset_field": "concatenated trial-window position",
                "n_levels_known": "task levels only; not physical amplitude",
                "repetition_evidence": "37 sessions with repeated trial structure",
                "continuous_signal_evidence": "concatenated fixed trial windows",
            }
        )
    return common


def _profile_decision(profile: str) -> dict[str, Any]:
    if profile == "dandi_000458_carried_forward":
        return {
            "pass1_class": FULL_METADATA_CANDIDATE,
            "stage2_status": NMD_TIMEBASE_INCOMPATIBLE,
            "classification": PERTURBATION_VALID_NMD_TIMEBASE_INCOMPATIBLE,
            "reason": (
                "physical amplitude and native timing are validated, but the "
                "frozen 2 s support horizon is below the current 4 s NMD window"
            ),
            "scores": {"P": 4, "A": 4, "T": 1, "R": 4, "M": 4},
        }
    if profile == "dryad_power_no_continuous_signal":
        return {
            "pass1_class": RHO_CANDIDATE,
            "stage2_status": INCONCLUSIVE,
            "classification": SOURCE_UNCERTAIN,
            "reason": (
                "optogenetic modulation is documented, but the outcome-free "
                "archive metadata does not freeze physical power, event timing, "
                "or a continuous NMD-compatible signal"
            ),
            "scores": {"P": 1, "A": 0, "T": 0, "R": 2, "M": 0},
        }
    if profile == "ibl_contrast_timing_uncertain":
        return {
            "pass1_class": FULL_METADATA_CANDIDATE,
            "stage2_status": INCONCLUSIVE,
            "classification": SOURCE_UNCERTAIN,
            "reason": (
                "contrast and trial structure are documented, but the local "
                "archive does not freeze event offsets or an isolated post "
                "horizon against the NMD window"
            ),
            "scores": {"P": 4, "A": 4, "T": 0, "R": 4, "M": 2},
        }
    if profile == "dandi_000690_contrast_uncertain":
        return {
            "pass1_class": RHO_CANDIDATE,
            "stage2_status": INCONCLUSIVE,
            "classification": SOURCE_UNCERTAIN,
            "reason": (
                "contrast-like table tokens are present, but no source-frozen "
                "physical contrast scale/unit or isolated timing contract is "
                "available in the local review archive"
            ),
            "scores": {"P": 2, "A": 1, "T": 0, "R": 3, "M": 2},
        }
    if profile == "dryad_peri_event_only":
        return {
            "pass1_class": SOURCE_UNCERTAIN,
            "stage2_status": "NOT_ENTERED",
            "classification": SOURCE_UNCERTAIN,
            "reason": (
                "the local release is pre-averaged peri-event data without "
                "continuous timestamps or paired trial-level modalities"
            ),
            "scores": {"P": 0, "A": 0, "T": 0, "R": 1, "M": 0},
        }
    if profile == "dandi_000574_no_replacement_qualification":
        return {
            "pass1_class": DIRECTION_ONLY,
            "stage2_status": "NOT_ENTERED",
            "classification": DIRECTION_ONLY,
            "reason": (
                "working-memory set size is a task parameter rather than a "
                "source-defined physical perturbation; the local release also "
                "contains concatenated fixed trial windows rather than a "
                "continuous FAR event stream"
            ),
            "scores": {"P": 0, "A": 0, "T": 0, "R": 2, "M": 1},
        }
    return {
        "pass1_class": SOURCE_UNCERTAIN,
        "stage2_status": INCONCLUSIVE,
        "classification": SOURCE_UNCERTAIN,
        "reason": "no local source-qualification evidence establishes a physical amplitude and timing contract",
        "scores": {"P": 0, "A": 0, "T": 0, "R": 0, "M": 0},
    }


def _timebase_rank(
    isolated_post_horizon_sec: float | None,
    nmd_window_sec: float,
) -> str:
    if isolated_post_horizon_sec is None:
        return "TIMEBASE_UNRESOLVED"
    if isolated_post_horizon_sec <= nmd_window_sec:
        return "TIMEBASE_INCOMPATIBLE"
    if isolated_post_horizon_sec >= 8.0:
        return "TIMEBASE_PREFERRED"
    if isolated_post_horizon_sec >= 2.0 * nmd_window_sec:
        return "TIMEBASE_STRONG"
    return "TIMEBASE_MINIMAL"


def _passes_hard_block(
    stage2_status: str,
    scores: Mapping[str, int],
    timebase_rank: str | None = None,
) -> bool:
    if stage2_status != PROMOTION_PASS:
        return False
    if not all(scores.get(axis, 0) > 0 for axis in ("P", "A", "T")):
        return False
    return timebase_rank not in {
        "TIMEBASE_INCOMPATIBLE",
        "TIMEBASE_UNRESOLVED",
    }


def _audit_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    roots = [Path(root) for root in spec["roots"]]
    existing_roots = [candidate for candidate in roots if candidate.is_dir()]
    if not existing_roots:
        return {
            "dataset_id": spec["dataset_id"],
            "source_family": spec["source_family"],
            "source_status": "LOCAL_UNAVAILABLE",
            "classification": SOURCE_UNCERTAIN,
            "stage2_status": INCONCLUSIVE,
            "reason": "configured local archive root is unavailable",
            "candidate_roots": [str(candidate) for candidate in roots],
            "scores_P_A_T_R_M": {"P": 0, "A": 0, "T": 0, "R": 0, "M": 0},
            "timebase_rank": "TIMEBASE_UNRESOLVED",
            "hard_blocked": True,
            "timebase_blocked": True,
        }
    evidence_relative_paths_by_root = spec.get("evidence_relative_paths_by_root")
    if evidence_relative_paths_by_root and len(existing_roots) != len(
        evidence_relative_paths_by_root
    ):
        return {
            "dataset_id": spec["dataset_id"],
            "source_family": spec["source_family"],
            "source_institution": spec.get("source_institution"),
            "source_status": "LOCAL_METADATA_SCOPE_UNRESOLVED",
            "source_roots": [str(root) for root in existing_roots],
            "classification": SOURCE_UNCERTAIN,
            "stage2_status": "NOT_ENTERED",
            "reason": "root-bound evidence scope is missing one or more configured roots",
            "metadata_inventory": {
                "metadata_file_count": 0,
                "metadata_suffix_counts": {},
                "metadata_bytes_scanned": 0,
                "term_file_counts": {},
            },
            "metadata_file_records": [],
            "scores_P_A_T_R_M": {"P": 0, "A": 0, "T": 0, "R": 0, "M": 0},
            "timebase_rank": "TIMEBASE_UNRESOLVED",
            "hard_blocked": True,
            "timebase_blocked": True,
            "audit_scope": {
                "metadata_only": True,
                "signal_payloads_opened": False,
                "nmd_outputs_opened": False,
                "downloads_performed": False,
                "outcome_tables_opened": False,
            },
        }
    files_with_roots = [
        (root, path)
        for root in existing_roots
        for path in _walk_metadata_files(root)
    ]
    evidence_relative_paths = {
        str(relative).replace("\\", "/").lower()
        for relative in spec.get("evidence_relative_paths", [])
    }
    evidence_patterns = spec.get("evidence_patterns")
    if evidence_relative_paths_by_root:
        evidence_files = []
        root_indexes = {
            root: index for index, root in enumerate(existing_roots)
        }
        for root, path in files_with_roots:
            root_index = root_indexes[root]
            allowed = {
                str(relative).replace("\\", "/").lower()
                for relative in (
                    evidence_relative_paths_by_root[root_index]
                    if root_index < len(evidence_relative_paths_by_root)
                    else []
                )
            }
            if (
                str(path.relative_to(root)).replace("\\", "/").lower()
                in allowed
            ):
                evidence_files.append((root, path))
        files_with_roots = evidence_files
    elif evidence_relative_paths:
        evidence_files = [
            (root, path)
            for root, path in files_with_roots
            if str(path.relative_to(root)).replace("\\", "/").lower()
            in evidence_relative_paths
        ]
        files_with_roots = evidence_files
    elif evidence_patterns:
        evidence_files = [
            (root, path)
            for root, path in files_with_roots
            if any(
                path.match(pattern) or pattern.lower() in path.name.lower()
                for pattern in evidence_patterns
            )
        ]
        files_with_roots = evidence_files
    if (
        evidence_relative_paths_by_root
        or evidence_relative_paths
        or evidence_patterns
    ) and not files_with_roots:
        return {
            "dataset_id": spec["dataset_id"],
            "source_family": spec["source_family"],
            "source_institution": spec.get("source_institution"),
            "source_status": "LOCAL_METADATA_SCOPE_UNRESOLVED",
            "source_roots": [str(root) for root in existing_roots],
            "classification": SOURCE_UNCERTAIN,
            "stage2_status": "NOT_ENTERED",
            "reason": "allowlisted evidence scope matched no local metadata files",
            "metadata_inventory": {
                "metadata_file_count": 0,
                "metadata_suffix_counts": {},
                "metadata_bytes_scanned": 0,
                "term_file_counts": {},
            },
            "metadata_file_records": [],
            "scores_P_A_T_R_M": {"P": 0, "A": 0, "T": 0, "R": 0, "M": 0},
            "timebase_rank": "TIMEBASE_UNRESOLVED",
            "hard_blocked": True,
            "audit_scope": {
                "metadata_only": True,
                "signal_payloads_opened": False,
                "nmd_outputs_opened": False,
                "downloads_performed": False,
                "outcome_tables_opened": False,
            },
        }
    files = [path for _, path in files_with_roots]
    inventory = _term_inventory(files)
    decision = _profile_decision(str(spec["profile"]))
    semantics = _profile_semantics(str(spec["profile"]))
    timebase_rank = _timebase_rank(
        semantics["isolated_post_horizon_sec"],
        float(spec["nmd_window_sec"]),
    )
    hard_blocked = any(
        decision["scores"].get(axis, 0) == 0 for axis in ("P", "A", "T")
    )
    timebase_blocked = timebase_rank in {
        "TIMEBASE_INCOMPATIBLE",
        "TIMEBASE_UNRESOLVED",
    }
    return {
        "dataset_id": spec["dataset_id"],
        "source_family": spec["source_family"],
        "source_institution": spec.get("source_institution"),
        "source_status": "LOCAL_METADATA_ARCHIVE",
        "source_roots": [str(root) for root in existing_roots],
        "nmd_window_sec": spec["nmd_window_sec"],
        "profile": spec["profile"],
        "metadata_inventory": inventory,
        "metadata_file_records": _relative_records(files_with_roots),
        "pass1": {
            "classification": decision["pass1_class"],
            "modality": semantics["modality"],
            "candidate_rho_field": semantics["candidate_rho_field"],
            "units": semantics["units"],
            "candidate_v": semantics["candidate_v"],
            "onset_field": semantics["onset_field"],
            "offset_field": semantics["offset_field"],
            "n_levels_known": semantics["n_levels_known"],
            "repetition_evidence": semantics["repetition_evidence"],
            "source_status": "LOCAL_METADATA_ARCHIVE",
        },
        "source_semantics": semantics,
        "scores_P_A_T_R_M": decision["scores"],
        "timebase_rank": timebase_rank,
        "hard_blocked": hard_blocked,
        "timebase_blocked": timebase_blocked,
        "stage2_status": decision["stage2_status"],
        "classification": decision["classification"],
        "classification_reason": decision["reason"],
        "audit_scope": {
            "metadata_only": True,
            "signal_payloads_opened": False,
            "nmd_outputs_opened": False,
            "downloads_performed": False,
            "outcome_tables_opened": False,
        },
    }


def run_inventory(
    *,
    repo_root: Path,
    protocol_path: Path,
    source_roots: Mapping[str, Path] | None = None,
    subsystems_root: Path | None = None,
) -> dict[str, Any]:
    audits = [
        _audit_spec(spec)
        for spec in _archive_specs(
            repo_root,
            source_roots=source_roots,
            subsystems_root=subsystems_root,
        )
    ]
    classifications = [audit["classification"] for audit in audits]
    promotable = [
        audit
        for audit in audits
        if _passes_hard_block(
            audit["stage2_status"],
            audit.get("scores_P_A_T_R_M", {}),
            audit.get("timebase_rank"),
        )
    ]
    gate_status = "PASS" if promotable else "NOT_TESTABLE"
    unavailable = [
        audit
        for audit in audits
        if audit["source_status"]
        in {"LOCAL_UNAVAILABLE", "LOCAL_METADATA_SCOPE_UNRESOLVED"}
    ]
    gate_reason = (
        "full_metadata_candidate_with_time_compatible_contract"
        if promotable
        else (
            "local_archive_scope_incomplete_and_no_promotable_source"
            if unavailable
            else "no_locally_archived_source_cleared_physical_timing_signal_and_nmd_filters"
        )
    )
    return {
        "schema": "mndm.far_scout_003_local_archive.v1",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": sha256_file(protocol_path),
        "gate_status": gate_status,
        "gate_reason": gate_reason,
        "source_families": ["dandi", "dryad", "ibl"],
        "source_institutions": ["Allen Institute / OpenScope", "IBL"],
        "pass_design": {
            "pass_1": "metadata_discovery",
            "pass_2": "source_specific_audit_for_rho_and_timing_candidates",
            "signal_payloads_opened": False,
        },
        "datasets": audits,
        "scope_complete": not unavailable,
        "scope_incomplete_dataset_ids": [
            audit["dataset_id"] for audit in unavailable
        ],
        "classification_counts": {
            classification: classifications.count(classification)
            for classification in sorted(set(classifications))
        },
        "score_axes": {
            "P": "physical perturbation validity",
            "A": "amplitude quality",
            "T": "temporal isolation",
            "R": "repetitions and biological support",
            "M": "NMD modality compatibility",
        },
        "hard_block_rule": "P=0 or A=0 or T=0 implies not promotable",
        "fail_closed_assertions": {
            "signal_payloads_opened": False,
            "nmd_outputs_opened": False,
            "downloads_performed": False,
            "outcomes_opened": False,
            "inferred_amplitude": False,
            "frequency_promoted_to_rho": False,
            "condition_promoted_to_rho": False,
            "home_away_constructed": False,
            "far_calculated": False,
        },
        "claim_boundary": (
            "FAR-SCOUT-003 is a local metadata/source-semantics sweep. "
            "It does not establish a time-compatible FAR source, open neural "
            "payloads, inspect NMD outcomes, or authorize FAR-003B."
        ),
    }
