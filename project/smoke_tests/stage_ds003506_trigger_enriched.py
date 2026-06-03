from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


INSTR_RE = re.compile(r'^Instr:\s*"(?P<mode>Choose|Match)"$', flags=re.IGNORECASE)
RESPONSE_RE = re.compile(r"^(?P<hand>Left|Right) Button Choice$", flags=re.IGNORECASE)
FEEDBACK_RE = re.compile(r'^FB:\s*(?P<outcome>\+1|0|"No Match"|"Too Slow")$', flags=re.IGNORECASE)
SKIP_DIRS = {".git", ".datalad", "__pycache__"}


def _default_source_root() -> Path:
    return Path("M:/datasets/received/openneuro/ds003506")


def _default_dest_root() -> Path:
    return Path("M:/datasets/received/openneuro/ds003506_trigger_enriched")


def _is_events_tsv(path: Path) -> bool:
    return path.name.endswith("_events.tsv")


def _is_events_json(path: Path) -> bool:
    return path.name.endswith("_events.json")


def _should_skip(rel_path: Path) -> bool:
    return any(part in SKIP_DIRS for part in rel_path.parts)


def _normalize_mode(value: str) -> str:
    return str(value).strip().lower()


def _normalize_hand(value: str) -> str:
    return str(value).strip().lower()


def _normalize_outcome(raw: str) -> str:
    val = str(raw).strip()
    return {
        "+1": "reward",
        "0": "punishment",
        '"No Match"': "error",
        '"Too Slow"': "timeout",
    }.get(val, val.lower().replace(" ", "_").replace('"', ""))


def _derive_event_fields(trial_type: object) -> Dict[str, object]:
    label = "" if trial_type is None else str(trial_type).strip()
    out: Dict[str, object] = {
        "event_role": "",
        "event_family": "",
        "event_subtype_role": "",
        "trial_mode": "",
        "response_hand": "",
        "feedback_outcome": "",
    }

    if label == "STATUS":
        out["event_role"] = "boundary"
        out["event_family"] = "boundary"
        return out

    match = INSTR_RE.match(label)
    if match:
        mode = _normalize_mode(match.group("mode"))
        out.update(
            {
                "event_role": "instruction",
                "event_family": "instruction",
                "event_subtype_role": f"{mode}_instruction",
                "trial_mode": mode,
            }
        )
        return out

    if label == "Imperative Stimulus":
        out.update(
            {
                "event_role": "stimulus",
                "event_family": "stimulus",
            }
        )
        return out

    match = RESPONSE_RE.match(label)
    if match:
        hand = _normalize_hand(match.group("hand"))
        out.update(
            {
                "event_role": "response",
                "event_family": "response",
                "response_hand": hand,
            }
        )
        return out

    match = FEEDBACK_RE.match(label)
    if match:
        outcome = _normalize_outcome(match.group("outcome"))
        out.update(
            {
                "event_role": "feedback",
                "event_family": "feedback",
                "feedback_outcome": outcome,
            }
        )
        return out

    out["event_family"] = "unknown"
    return out


def _apply_trial_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    trial_modes: List[object] = []
    trial_sequence: List[object] = []
    phase_trial_index: List[object] = []
    subtype_roles: List[str] = []

    counters = {"choose": 0, "match": 0}
    active_mode = ""
    active_global: object = pd.NA
    active_phase: object = pd.NA
    global_counter = 0

    for _, row in df.iterrows():
        role = str(row.get("event_role", "") or "")
        mode = str(row.get("trial_mode", "") or "")
        hand = str(row.get("response_hand", "") or "")
        outcome = str(row.get("feedback_outcome", "") or "")
        subtype = str(row.get("event_subtype_role", "") or "")

        if role == "instruction" and mode in counters:
            active_mode = mode
            counters[mode] += 1
            global_counter += 1
            active_global = global_counter
            active_phase = counters[mode]
            subtype = f"{mode}_instruction"
            trial_modes.append(active_mode)
            trial_sequence.append(active_global)
            phase_trial_index.append(active_phase)
            subtype_roles.append(subtype)
            continue

        if role == "stimulus" and active_mode:
            subtype = f"{active_mode}_stimulus"
        elif role == "response" and active_mode and hand:
            subtype = f"{active_mode}_response_{hand}"
        elif role == "feedback" and active_mode and outcome:
            subtype = f"{active_mode}_feedback_{outcome}"

        if role in {"stimulus", "response", "feedback"} and active_mode:
            trial_modes.append(active_mode)
            trial_sequence.append(active_global)
            phase_trial_index.append(active_phase)
        else:
            trial_modes.append(mode if mode else pd.NA)
            trial_sequence.append(pd.NA)
            phase_trial_index.append(pd.NA)

        subtype_roles.append(subtype)

    df["trial_mode"] = pd.array(trial_modes, dtype="string")
    df["trial_sequence_index"] = pd.array(trial_sequence, dtype="Int64")
    df["phase_trial_index"] = pd.array(phase_trial_index, dtype="Int64")
    df["event_subtype_role"] = subtype_roles
    return df


def _augment_events_df(events_df: pd.DataFrame) -> pd.DataFrame:
    derived = pd.DataFrame([_derive_event_fields(v) for v in events_df.get("trial_type", pd.Series("", index=events_df.index))])
    augmented = pd.concat([events_df.copy(), derived], axis=1)
    return _apply_trial_context(augmented)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _enriched_events_json(existing: Dict[str, object]) -> Dict[str, object]:
    payload = dict(existing)
    payload.update(
        {
            "event_role": {
                "Description": "Derived event role inferred from ds003506 trigger semantics.",
                "Levels": {
                    "instruction": "Choice-mode instruction event.",
                    "stimulus": "Imperative stimulus event.",
                    "response": "Button-response event.",
                    "feedback": "Feedback event.",
                    "boundary": "Boundary/status marker.",
                },
            },
            "event_family": {
                "Description": "Compact derived event family from ds003506 trigger semantics.",
            },
            "event_subtype_role": {
                "Description": "Compact derived label used as the staged raw event provenance surface for MNDM event mapping.",
            },
            "trial_mode": {
                "Description": "Derived trial mode propagated from the instruction cue.",
                "Levels": {
                    "choose": "Volitional/choose trial mode.",
                    "match": "Instructed/match trial mode.",
                },
            },
            "response_hand": {
                "Description": "Left/right response hand parsed from trigger text when available.",
            },
            "feedback_outcome": {
                "Description": "Derived feedback outcome parsed from trigger text.",
            },
            "trial_sequence_index": {
                "Description": "Derived sequential trial index within the run, anchored on instruction events.",
            },
            "phase_trial_index": {
                "Description": "Derived sequential trial index within the current trial mode, anchored on instruction events.",
            },
        }
    )
    return payload


def _ensure_link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return "existing"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def _process_events_tsv(src: Path, dst: Path) -> Dict[str, object]:
    df = pd.read_csv(src, sep="\t")
    augmented = _augment_events_df(df)
    augmented = augmented[augmented["event_family"] != "boundary"].reset_index(drop=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(dst, sep="\t", index=False)
    counts = augmented["event_subtype_role"].fillna("").astype(str)
    nonempty = counts[counts != ""]
    summary = {
        "rel_path": str(src),
        "rows": int(len(augmented)),
        "nonempty_event_subtype_role_rows": int((counts != "").sum()),
        "unique_event_subtype_roles": int(nonempty.nunique()),
        "instruction_rows": int((augmented["event_family"] == "instruction").sum()),
        "stimulus_rows": int((augmented["event_family"] == "stimulus").sum()),
        "response_rows": int((augmented["event_family"] == "response").sum()),
        "feedback_rows": int((augmented["event_family"] == "feedback").sum()),
    }
    return summary


def _process_events_json(src: Path, dst: Path) -> None:
    try:
        existing = json.loads(src.read_text(encoding="utf-8"))
    except Exception:
        existing = {}
    _write_json(dst, _enriched_events_json(existing))


def _iter_source_files(source_root: Path) -> Iterable[Path]:
    for path in source_root.rglob("*"):
        rel = path.relative_to(source_root)
        if _should_skip(rel):
            continue
        yield path


def stage_dataset(source_root: Path, dest_root: Path) -> pd.DataFrame:
    dest_root.mkdir(parents=True, exist_ok=True)
    audit_rows: List[Dict[str, object]] = []

    for path in _iter_source_files(source_root):
        rel = path.relative_to(source_root)
        dest = dest_root / rel

        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue

        if _is_events_tsv(path):
            summary = _process_events_tsv(path, dest)
            summary["rel_path"] = str(rel).replace("\\", "/")
            audit_rows.append(summary)
            continue

        if _is_events_json(path):
            _process_events_json(path, dest)
            continue

        _ensure_link_or_copy(path, dest)

    audit_df = pd.DataFrame(audit_rows).sort_values("rel_path").reset_index(drop=True)
    audit_path = dest_root / "trigger_event_enrichment_audit.tsv"
    audit_df.to_csv(audit_path, sep="\t", index=False)
    return audit_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a trigger-enriched staged root for ds003506.")
    parser.add_argument("--source-root", type=Path, default=_default_source_root())
    parser.add_argument("--dest-root", type=Path, default=_default_dest_root())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_df = stage_dataset(args.source_root, args.dest_root)
    print(f"STAGED_ROOT={args.dest_root}")
    print(f"EVENT_FILES={len(audit_df)}")
    print(f"AUDIT_PATH={args.dest_root / 'trigger_event_enrichment_audit.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
