from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from dandi_ingest.contracts import AssetRecord, DandiIngestionConfig, ProbeSummary, TriageResult


class Dataset000458Adapter:
    """Triage adapter for DANDI 000458 mouse EEG/ecephys pilot assets."""

    adapter_id = "dataset_000458"

    _KEYWORD_WEIGHTS: tuple[tuple[str, int], ...] = (
        ("eeg", 8),
        ("surface", 6),
        ("isoflurane", 6),
        ("anesthesia", 6),
        ("awake", 5),
        ("recovery", 4),
        ("lfp", 4),
        ("ecephys", 4),
        ("electrical", 3),
        ("neuropixels", 3),
        ("stim", 2),
        ("speed", 2),
        ("epoch", 2),
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
            record.metadata["triage_role"] = self._role_for_record(record)

        ranked = sorted(
            nwb_records,
            key=lambda record: (
                int(record.metadata.get("triage_score", 0)),
                int(record.size or 0),
                record.path,
            ),
            reverse=True,
        )
        limit = config.selection.asset_limit or 8
        if limit <= 0:
            return []

        selected: list[AssetRecord] = []
        by_subject: dict[str, list[AssetRecord]] = defaultdict(list)
        for record in ranked:
            if record.subject_id:
                by_subject[record.subject_id].append(record)

        # Prefer breadth across animals for the first smoke bundle.
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
            "DANDI 000458 is treated as the first NeuralManifoldDynamics NWB pilot because it includes mouse surface EEG and structured awake/isoflurane/recovery state epochs.",
            "The first-pass selection favors NWB assets with EEG, ecephys/LFP, anesthesia, and epoch/state hints while keeping the MNDM path EEG-first.",
            "LFP, Neuropixels spikes, stimulation, and speed variables are useful probe targets for later multimodal expansion, but are not required for the first MNPS smoke run.",
        ]
        metadata = {
            "selected_count": len(records),
            "probed_count": sum(1 for record in records if record.path in probe_by_path),
            "selection_goal": "mouse_eeg_state_epoch_nwb_pilot",
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
                f"`{record.metadata.get('triage_role', 'pilot')}` | "
                f"{record.metadata.get('triage_score', 0)} | "
                f"{reasons or 'nwb'} |"
            )
        lines.extend(
            [
                "",
                "## Next Step",
                "",
                "Probe selected assets for surface EEG ElectricalSeries and state/epoch interval tables, then run the EEG-only MNDM overlay on one local NWB file.",
            ]
        )
        return "\n".join(lines) + "\n"

    def _role_for_record(self, record: AssetRecord) -> str:
        haystack = f"{record.path} {record.metadata}".lower()
        if "eeg" in haystack and ("isoflurane" in haystack or "anesthesia" in haystack):
            return "eeg_state_contrast"
        if "eeg" in haystack:
            return "eeg_pilot"
        if "lfp" in haystack or "ecephys" in haystack or "neuropixels" in haystack:
            return "ephys_probe"
        return "nwb_probe"

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
