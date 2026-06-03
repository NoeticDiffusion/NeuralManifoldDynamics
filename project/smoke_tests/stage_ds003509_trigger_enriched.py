from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


TRAIN_STIM_RE = re.compile(
    r"^Trn Stim:\s*(?P<color>blue|yellow)\s+(?P<congruency>congru|incongru)\s+(?P<stimulus>[ABCD])$",
    flags=re.IGNORECASE,
)
TRAIN_RESP_RE = re.compile(
    r"^Trn Resp:\s*(?P<hand>left|right),(?P<accuracy>correct|incorrect)$",
    flags=re.IGNORECASE,
)
TEST_STIM_RE = re.compile(r"^Test Stim:\s*(?P<pair>[ABCD]{2})$", flags=re.IGNORECASE)
TEST_RESP_RE = re.compile(
    r"^Test Resp:\s*(?P<hand>left|right),(?P<accuracy>correct|incorrect)$",
    flags=re.IGNORECASE,
)
FEEDBACK_RE = re.compile(r"^FB:\s*(?P<outcome>\+1|0|.+)$", flags=re.IGNORECASE)

RANK = {"A": 1, "B": 2, "C": 3, "D": 4}
TRAIN_REWARD_PROBABILITY = {"A": 1.0, "B": 0.5, "C": 0.5, "D": 0.0}
TRAIN_REWARD_PROFILE = {
    "A": "reward_100",
    "B": "reward_50_conflict_before_reward",
    "C": "reward_50_conflict_before_punishment",
    "D": "reward_0",
}
TEST_PAIR_ANALYSIS_FAMILY = {
    "AB": "relative_conflict",
    "AC": "relative_conflict",
    "AD": "easy",
    "BC": "direct_conflict",
    "BD": "relative_conflict",
    "CD": "relative_conflict",
}
SKIP_DIRS = {".git", ".datalad", "__pycache__"}


def _default_source_root() -> Path:
    return Path("M:/datasets/received/openneuro/ds003509")


def _default_dest_root() -> Path:
    return Path("M:/datasets/received/openneuro/ds003509_trigger_enriched")


def _is_events_tsv(path: Path) -> bool:
    return path.name.endswith("_events.tsv")


def _is_events_json(path: Path) -> bool:
    return path.name.endswith("_events.json")


def _should_skip(rel_path: Path) -> bool:
    return any(part in SKIP_DIRS for part in rel_path.parts)


def _normalize_hand(hand: str) -> str:
    return str(hand).strip().lower()


def _normalize_accuracy(acc: str) -> str:
    return str(acc).strip().lower()


def _derive_event_fields(trial_type: object) -> Dict[str, object]:
    label = "" if trial_type is None else str(trial_type).strip()
    out: Dict[str, object] = {
        "event_phase": "",
        "event_role": "",
        "event_family": "",
        "event_subtype_role": "",
        "stimulus_color": "",
        "stimulus_congruency": "",
        "stimulus_class": "",
        "stimulus_reward_probability": pd.NA,
        "stimulus_reward_profile": "",
        "response_hand": "",
        "response_accuracy": "",
        "feedback_outcome": "",
        "test_pair_code": "",
        "test_pair_sorted": "",
        "test_left_stimulus_class": "",
        "test_right_stimulus_class": "",
        "test_optimal_side": "",
        "test_pair_analysis_family": "",
    }

    if label == "STATUS":
        out["event_role"] = "boundary"
        out["event_family"] = "boundary"
        return out

    match = TRAIN_STIM_RE.match(label)
    if match:
        stimulus = match.group("stimulus").upper()
        congruency_raw = match.group("congruency").lower()
        congruency = "congruent" if congruency_raw == "congru" else "incongruent"
        out.update(
            {
                "event_phase": "training",
                "event_role": "stimulus",
                "event_family": "training_stimulus",
                "event_subtype_role": f"training_stimulus_{congruency}",
                "stimulus_color": match.group("color").lower(),
                "stimulus_congruency": congruency,
                "stimulus_class": stimulus,
                "stimulus_reward_probability": TRAIN_REWARD_PROBABILITY[stimulus],
                "stimulus_reward_profile": TRAIN_REWARD_PROFILE[stimulus],
            }
        )
        return out

    match = TRAIN_RESP_RE.match(label)
    if match:
        hand = _normalize_hand(match.group("hand"))
        accuracy = _normalize_accuracy(match.group("accuracy"))
        out.update(
            {
                "event_phase": "training",
                "event_role": "response",
                "event_family": "training_response",
                "event_subtype_role": f"training_response_{accuracy}",
                "response_hand": hand,
                "response_accuracy": accuracy,
            }
        )
        return out

    if label == "Trn No Response":
        out.update(
            {
                "event_phase": "training",
                "event_role": "response",
                "event_family": "training_no_response",
                "event_subtype_role": "training_no_response",
                "response_accuracy": "no_response",
            }
        )
        return out

    match = FEEDBACK_RE.match(label)
    if match:
        raw_outcome = match.group("outcome").strip()
        outcome = {
            "+1": "reward",
            "0": "punishment",
            '"Too Slow"': "timeout",
            '"No Match"': "error",
        }.get(raw_outcome, raw_outcome.lower().replace(" ", "_").replace('"', ""))
        out.update(
            {
                "event_phase": "training",
                "event_role": "feedback",
                "event_family": "training_feedback",
                "event_subtype_role": f"training_feedback_{outcome}" if outcome else "training_feedback",
                "feedback_outcome": outcome,
            }
        )
        return out

    match = TEST_STIM_RE.match(label)
    if match:
        pair = match.group("pair").upper()
        left = pair[0]
        right = pair[1]
        pair_sorted = "".join(sorted(pair))
        optimal_side = "left" if RANK[left] < RANK[right] else "right"
        analysis_family = TEST_PAIR_ANALYSIS_FAMILY.get(pair_sorted, "other")
        out.update(
            {
                "event_phase": "test",
                "event_role": "stimulus",
                "event_family": "test_stimulus",
                "event_subtype_role": f"test_stimulus_{analysis_family}",
                "test_pair_code": pair,
                "test_pair_sorted": pair_sorted,
                "test_left_stimulus_class": left,
                "test_right_stimulus_class": right,
                "test_optimal_side": optimal_side,
                "test_pair_analysis_family": analysis_family,
            }
        )
        return out

    match = TEST_RESP_RE.match(label)
    if match:
        hand = _normalize_hand(match.group("hand"))
        accuracy = _normalize_accuracy(match.group("accuracy"))
        out.update(
            {
                "event_phase": "test",
                "event_role": "response",
                "event_family": "test_response",
                "event_subtype_role": f"test_response_{accuracy}",
                "response_hand": hand,
                "response_accuracy": accuracy,
            }
        )
        return out

    if label == "Test No Response":
        out.update(
            {
                "event_phase": "test",
                "event_role": "response",
                "event_family": "test_no_response",
                "event_subtype_role": "test_no_response",
                "response_accuracy": "no_response",
            }
        )
        return out

    out["event_family"] = "unknown"
    return out


def _assign_trial_indices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    trial_sequence: List[object] = []
    phase_trial_index: List[object] = []

    global_counter = 0
    phase_counters = {"training": 0, "test": 0}
    active_global: object = pd.NA
    active_phase_index: object = pd.NA
    active_phase = ""

    for _, row in df.iterrows():
        phase = str(row.get("event_phase", "") or "")
        family = str(row.get("event_family", "") or "")

        if family in {"training_stimulus", "test_stimulus"}:
            global_counter += 1
            if phase in phase_counters:
                phase_counters[phase] += 1
                active_phase_index = phase_counters[phase]
            else:
                active_phase_index = pd.NA
            active_global = global_counter
            active_phase = phase
            trial_sequence.append(active_global)
            phase_trial_index.append(active_phase_index)
            continue

        if phase and phase == active_phase and family in {
            "training_response",
            "training_no_response",
            "training_feedback",
            "test_response",
            "test_no_response",
        }:
            trial_sequence.append(active_global)
            phase_trial_index.append(active_phase_index)
            continue

        trial_sequence.append(pd.NA)
        phase_trial_index.append(pd.NA)

    df["trial_sequence_index"] = pd.array(trial_sequence, dtype="Int64")
    df["phase_trial_index"] = pd.array(phase_trial_index, dtype="Int64")
    return df


def _augment_events_df(events_df: pd.DataFrame) -> pd.DataFrame:
    derived = pd.DataFrame([_derive_event_fields(v) for v in events_df.get("trial_type", pd.Series("", index=events_df.index))])
    augmented = pd.concat([events_df.copy(), derived], axis=1)
    return _assign_trial_indices(augmented)


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _enriched_events_json(existing: Dict[str, object]) -> Dict[str, object]:
    payload = dict(existing)
    payload.update(
        {
            "event_phase": {
                "Description": "Derived task phase inferred from ds003509 trigger semantics.",
                "Levels": {
                    "training": "Training phase event.",
                    "test": "Test phase event.",
                },
            },
            "event_role": {
                "Description": "Derived event role inferred from ds003509 trigger semantics.",
                "Levels": {
                    "stimulus": "Stimulus presentation event.",
                    "response": "Behavioral response or no-response marker.",
                    "feedback": "Feedback event.",
                    "boundary": "Boundary/status marker.",
                },
            },
            "event_family": {
                "Description": "Compact derived event family from ds003509 trigger semantics.",
            },
            "event_subtype_role": {
                "Description": "Compact derived label used as the staged raw event provenance surface for MNDM event mapping.",
            },
            "stimulus_color": {
                "Description": "Training stimulus color parsed from trigger text.",
            },
            "stimulus_congruency": {
                "Description": "Training stimulus congruency parsed from trigger text.",
                "Levels": {
                    "congruent": "Color and response mapping are congruent.",
                    "incongruent": "Color and response mapping are incongruent.",
                },
            },
            "stimulus_class": {
                "Description": "Stimulus class A-D parsed from trigger text.",
            },
            "stimulus_reward_probability": {
                "Description": "Reward probability derived from the ds003509 trigger code comments in CC_Triggers.m.",
            },
            "stimulus_reward_profile": {
                "Description": "Reward-profile shorthand derived from the ds003509 trigger code comments in CC_Triggers.m.",
            },
            "response_hand": {
                "Description": "Left/right response hand parsed from trigger text when available.",
            },
            "response_accuracy": {
                "Description": "Derived response correctness or no-response state parsed from trigger text.",
            },
            "feedback_outcome": {
                "Description": "Derived feedback outcome parsed from trigger text.",
            },
            "test_pair_code": {
                "Description": "Ordered left-right test pair code parsed from trigger text (for example AB, DB).",
            },
            "test_pair_sorted": {
                "Description": "Canonical unordered test pair code with letters sorted alphabetically.",
            },
            "test_left_stimulus_class": {
                "Description": "Left test stimulus class parsed from trigger text.",
            },
            "test_right_stimulus_class": {
                "Description": "Right test stimulus class parsed from trigger text.",
            },
            "test_optimal_side": {
                "Description": "Derived optimal side assuming A > B > C > D, following CC_Triggers.m comments.",
            },
            "test_pair_analysis_family": {
                "Description": "Coarse pair family derived from the ds003509 behavioral analysis script (easy, direct_conflict, relative_conflict).",
            },
            "trial_sequence_index": {
                "Description": "Derived sequential trial index within the run, anchored on stimulus events.",
            },
            "phase_trial_index": {
                "Description": "Derived sequential trial index within the current task phase, anchored on stimulus events.",
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
    # Drop BIDS/EEGLAB boundary status rows from the staged event stream so they
    # do not become meaningless raw_* label surfaces like raw_nan_onset_sec.
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
        "training_stimulus_rows": int((augmented["event_family"] == "training_stimulus").sum()),
        "test_stimulus_rows": int((augmented["event_family"] == "test_stimulus").sum()),
        "training_response_rows": int(
            augmented["event_family"].isin(["training_response", "training_no_response"]).sum()
        ),
        "test_response_rows": int(augmented["event_family"].isin(["test_response", "test_no_response"]).sum()),
        "feedback_rows": int((augmented["event_family"] == "training_feedback").sum()),
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
    parser = argparse.ArgumentParser(description="Create a trigger-enriched staged root for ds003509.")
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
