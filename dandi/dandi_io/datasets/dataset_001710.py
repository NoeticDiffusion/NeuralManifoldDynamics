from __future__ import annotations

from typing import Sequence

from dandi_io.contracts import AssetRecord, DandiIngestionConfig, ProbeSummary, TriageResult


class Dataset001710Adapter:
    """Triage adapter for DANDI 001710 (Y-maze context-sensitive remapping dataset).

    001710 contains multi-day calcium imaging sessions from mice performing a
    Y-maze virtual-reality task.  Sessions are structured as
    ``sub-<id>_ses-ymaze-day<N>-scan0-novel-arm-<M>_behavior+ophys.nwb``.

    The adapter selects one subject's full longitudinal day-series (day0–day5)
    so that cross-day remapping and context-sensitive retrieval analyses can be
    built without downloading redundant assets first.
    """

    adapter_id = "dataset_001710"

    _KEYWORD_WEIGHTS: tuple[tuple[str, int], ...] = (
        ("ymaze", 6),
        ("novel", 5),
        ("behavior+ophys", 5),
        ("day", 4),
        ("remapping", 4),
        ("place", 3),
        ("ophys", 2),
        ("behavior", 2),
        ("scan", 1),
    )

    def select_assets(
        self,
        records: Sequence[AssetRecord],
        config: DandiIngestionConfig,
    ) -> list[AssetRecord]:
        nwb_records = [r for r in records if r.path.lower().endswith(".nwb")]
        if not nwb_records:
            nwb_records = list(records)

        for record in nwb_records:
            score, reasons = self._score_record(record)
            record.metadata["triage_score"] = score
            record.metadata["triage_reasons"] = reasons
            record.metadata["triage_day"] = self._extract_day(record)

        # Sort: highest score first, then day order within a subject, then path
        ranked = sorted(
            nwb_records,
            key=lambda r: (
                -int(r.metadata.get("triage_score", 0)),
                self._extract_day(r),
                r.path,
            ),
        )

        limit = config.selection.asset_limit or 6
        if limit <= 0:
            return []

        # Prefer a full day-series from one subject
        from collections import defaultdict
        by_subject: dict[str, list[AssetRecord]] = defaultdict(list)
        for record in ranked:
            sid = record.subject_id or "unknown"
            by_subject[sid].append(record)

        selected: list[AssetRecord] = []
        for subject_id in sorted(by_subject):
            day_sorted = sorted(
                by_subject[subject_id],
                key=lambda r: self._extract_day(r),
            )
            for record in day_sorted:
                if record not in selected:
                    selected.append(record)
                if len(selected) >= limit:
                    return selected[:limit]

        for record in ranked:
            if record not in selected:
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
            "DANDI 001710 is the context-sensitive remapping dataset for the key-lock theory program.",
            "The first-pass selection favors `behavior+ophys` NWB assets from a full day-series (day0–day5) to support cross-day place-code and remapping analyses.",
            "Sessions are Y-maze virtual-reality with left/right arm trials and embedded place-cell annotations.",
            "This adapter stops at manifest and metadata triage; it does not perform biological analysis.",
        ]
        metadata = {
            "selected_count": len(records),
            "probed_count": sum(1 for r in records if r.path in probe_by_path),
            "selection_goal": "metadata_first_remapping_triage",
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
                "| Path | Subject | Session | Size (bytes) | Day | Score | Reasons |",
                "| --- | --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for record in triage.selected_assets:
            reasons = ", ".join(record.metadata.get("triage_reasons", []))
            day = record.metadata.get("triage_day", -1)
            lines.append(
                "| "
                f"`{record.path}` | "
                f"`{record.subject_id or 'unknown'}` | "
                f"`{record.session_id or 'unknown'}` | "
                f"{record.size or 0} | "
                f"{day} | "
                f"{record.metadata.get('triage_score', 0)} | "
                f"{reasons or 'generic_nwb_priority'} |"
            )
        lines.extend(
            [
                "",
                "## Next Step",
                "",
                "Probe selected assets for behavior channels, ophys interfaces, and annotation payload keys, "
                "then proceed to trial reconstruction and activity matrix extraction.",
            ]
        )
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_day(self, record: AssetRecord) -> int:
        """Return the integer day index from the asset path, or 999 if absent."""
        import re
        m = re.search(r"day(\d+)", record.path, re.IGNORECASE)
        return int(m.group(1)) if m else 999

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
