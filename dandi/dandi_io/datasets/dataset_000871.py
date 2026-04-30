from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from dandi_io.contracts import AssetRecord, DandiIngestionConfig, ProbeSummary, TriageResult


class Dataset000871Adapter:
    adapter_id = "dataset_000871"

    _PREFERRED_PATHS: tuple[str, ...] = (
        "sub-656228/sub-656228_ses-1245548523-acq-1245937736_image+ophys.nwb",
        "sub-644972/sub-644972_ses-1237338784-acq-1237809219_image+ophys.nwb",
        "sub-644972/sub-644972_ses-1237338784-acq-1237809217_image+ophys.nwb",
        "sub-656228/sub-656228_ses-1247233186-acq-1247385130_image+ophys.nwb",
        "sub-656228/sub-656228_ses-1247233186-acq-1247385128_image+ophys.nwb",
    )

    _PREFERRED_ROLES = {
        "sub-656228/sub-656228_ses-1245548523-acq-1245937736_image+ophys.nwb": "proof_of_access",
        "sub-644972/sub-644972_ses-1237338784-acq-1237809219_image+ophys.nwb": "pair_a",
        "sub-644972/sub-644972_ses-1237338784-acq-1237809217_image+ophys.nwb": "pair_a",
        "sub-656228/sub-656228_ses-1247233186-acq-1247385130_image+ophys.nwb": "pair_b",
        "sub-656228/sub-656228_ses-1247233186-acq-1247385128_image+ophys.nwb": "pair_b",
    }

    _PAIR_SESSION_HINTS: tuple[str, ...] = (
        "1237338784",
        "1247233186",
        "1245548523",
    )

    def select_assets(
        self,
        records: Sequence[AssetRecord],
        config: DandiIngestionConfig,
    ) -> list[AssetRecord]:
        candidate_records = [record for record in records if self._is_clean_nwb(record)]
        if not candidate_records:
            candidate_records = [record for record in records if record.path.lower().endswith(".nwb")]
        if not candidate_records:
            candidate_records = list(records)

        for record in candidate_records:
            score, reasons = self._score_record(record)
            record.metadata["triage_score"] = score
            record.metadata["triage_reasons"] = reasons
            record.metadata["triage_role"] = self._PREFERRED_ROLES.get(record.path, "fallback")

        limit = config.selection.asset_limit or 5
        if limit <= 0:
            return []

        selected: list[AssetRecord] = []
        by_path = {record.path: record for record in candidate_records}

        for path in self._PREFERRED_PATHS:
            record = by_path.get(path)
            if record is None or record in selected:
                continue
            selected.append(record)
            if len(selected) >= limit:
                return selected[:limit]

        ranked = sorted(
            candidate_records,
            key=lambda record: (
                int(record.metadata.get("triage_score", 0)),
                int(record.size or 0),
                record.path,
            ),
            reverse=True,
        )

        by_subject: dict[str, list[AssetRecord]] = defaultdict(list)
        for record in ranked:
            if record.subject_id:
                by_subject[record.subject_id].append(record)

        for subject_id in sorted(by_subject):
            candidate = by_subject[subject_id][0]
            if candidate not in selected:
                selected.append(candidate)
            if len(selected) >= limit:
                return selected[:limit]

        for record in ranked:
            if record in selected:
                continue
            selected.append(record)
            if len(selected) >= limit:
                break
        return selected[:limit]

    def build_triage(
        self,
        records: Sequence[AssetRecord],
        config: DandiIngestionConfig,
        *,
        probes: Sequence[ProbeSummary] | None = None,
    ) -> TriageResult:
        probe_by_path = {probe.path: probe for probe in probes or ()}
        notes = [
            "DANDI 000871 is treated as the exploratory dendritic-coupling dataset for the lock-side plausibility program.",
            "The first-pass selection favors canonical `image+ophys` NWB assets and excludes `raw-movies` and `denoised-movies` derivatives from the initial bundle.",
            "The candidate bundle is designed around one proof-of-access file plus two same-subject same-session pairs that are useful for early coupling comparisons across planes/depths.",
        ]
        metadata = {
            "selected_count": len(records),
            "probed_count": sum(1 for record in records if record.path in probe_by_path),
            "selection_goal": "metadata_first_coupling_triage",
        }
        return TriageResult(
            adapter_id=self.adapter_id,
            dandiset_id=config.dataset.dandiset_id,
            selected_assets=tuple(records),
            notes=tuple(notes),
            metadata=metadata,
        )

    def render_triage_markdown(self, triage: TriageResult) -> str:
        lines = [
            f"# Triage Summary: DANDI {triage.dandiset_id}",
            "",
            "## Rationale",
            "",
        ]
        for note in triage.notes:
            lines.append(f"- {note}")
        lines.extend(
            [
                "",
                "## Selected Assets",
                "",
                "| Path | Subject | Session | Size (bytes) | Role | Score | Reasons |",
                "| --- | --- | --- | ---: | --- | ---: | --- |",
            ]
        )
        for record in triage.selected_assets:
            reasons = ", ".join(record.metadata.get("triage_reasons", []))
            lines.append(
                "| "
                f"`{record.path}` | "
                f"`{record.subject_id or 'unknown'}` | "
                f"`{record.session_id or 'unknown'}` | "
                f"{record.size or 0} | "
                f"`{record.metadata.get('triage_role', 'fallback')}` | "
                f"{record.metadata.get('triage_score', 0)} | "
                f"{reasons or 'clean_nwb'} |"
            )
        lines.extend(
            [
                "",
                "## Next Step",
                "",
                "Probe the selected NWB bundle first, then choose one proof-of-access file and one same-session pair for the first coupling-oriented smoke tests.",
            ]
        )
        return "\n".join(lines) + "\n"

    def _is_clean_nwb(self, record: AssetRecord) -> bool:
        path = record.path.lower()
        return (
            path.endswith(".nwb")
            and "-raw-movies_" not in path
            and "-denoised-movies_" not in path
        )

    def _score_record(self, record: AssetRecord) -> tuple[int, list[str]]:
        score = 0
        reasons: list[str] = []
        path = record.path
        path_lower = path.lower()

        if self._is_clean_nwb(record):
            score += 10
            reasons.append("canonical_image_ophys")
        if path in self._PREFERRED_PATHS:
            score += 100
            reasons.append("preferred_open_scope_path")
        if any(session_hint in path for session_hint in self._PAIR_SESSION_HINTS):
            score += 15
            reasons.append("paired_session_candidate")
        if "image+ophys" in path_lower:
            score += 5
            reasons.append("image_ophys")
        if record.subject_id:
            score += 1
            reasons.append("subject_tag")
        if record.session_id:
            score += 1
            reasons.append("session_tag")
        return score, reasons
