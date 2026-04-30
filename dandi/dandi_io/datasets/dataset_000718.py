from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from dandi_io.contracts import AssetRecord, DandiIngestionConfig, ProbeSummary, TriageResult


class Dataset000718Adapter:
    adapter_id = "dataset_000718"

    _KEYWORD_WEIGHTS: tuple[tuple[str, int], ...] = (
        ("offline", 5),
        ("week", 5),
        ("sleep", 4),
        ("reactivation", 4),
        ("neutral", 4),
        ("aversive", 4),
        ("fear", 3),
        ("memory", 3),
        ("registration", 3),
        ("cell", 2),
        ("eeg", 2),
        ("emg", 2),
        ("ophys", 1),
        ("image", 1),
    )

    def select_assets(
        self,
        records: Sequence[AssetRecord],
        config: DandiIngestionConfig,
    ) -> list[AssetRecord]:
        nwb_records = [record for record in records if record.path.lower().endswith(".nwb")]
        if not nwb_records:
            nwb_records = list(records)
        for record in nwb_records:
            score, reasons = self._score_record(record)
            record.metadata["triage_score"] = score
            record.metadata["triage_reasons"] = reasons

        ranked = sorted(
            nwb_records,
            key=lambda record: (
                int(record.metadata.get("triage_score", 0)),
                int(record.size or 0),
                record.path,
            ),
            reverse=True,
        )
        limit = config.selection.asset_limit or 6
        if limit <= 0:
            return []

        selected: list[AssetRecord] = []
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
            "DANDI 000718 is treated as the primary linking dataset because it directly targets offline ensemble co-reactivation and memory integration across days.",
            "The first-pass selection favors NWB assets with offline/week/session hints, both subjects when available, and metadata useful for later linking analysis triage.",
            "This adapter stops at manifest and metadata triage; it does not perform ensemble extraction or biological hypothesis testing.",
        ]
        metadata = {
            "selected_count": len(records),
            "probed_count": sum(1 for record in records if record.path in probe_by_path),
            "selection_goal": "metadata_first_linking_triage",
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
                "| Path | Subject | Session | Size (bytes) | Score | Reasons |",
                "| --- | --- | --- | ---: | ---: | --- |",
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
                f"{record.metadata.get('triage_score', 0)} | "
                f"{reasons or 'generic_nwb_priority'} |"
            )
        lines.extend(
            [
                "",
                "## Next Step",
                "",
                "Use the selected asset bundle for metadata/header probing first, then decide which subset should be downloaded for the first offline-linking analysis pass.",
            ]
        )
        return "\n".join(lines) + "\n"

    def _score_record(self, record: AssetRecord) -> tuple[int, list[str]]:
        haystack = f"{record.path} {record.metadata}".lower()
        score = 0
        reasons: list[str] = []
        for keyword, weight in self._KEYWORD_WEIGHTS:
            if keyword in haystack:
                score += weight
                reasons.append(keyword)
        if record.subject_id:
            score += 1
            reasons.append("subject_tag")
        if record.session_id:
            score += 1
            reasons.append("session_tag")
        if record.path.lower().endswith(".nwb"):
            score += 1
            reasons.append("nwb")
        return score, reasons
