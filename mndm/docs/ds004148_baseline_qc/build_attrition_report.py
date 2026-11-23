"""Build a compact participant-level QC attrition report from frozen outputs.

This reads the completed ds004148 baseline QC/summary CSVs and writes only
small attrition artifacts beside this script. It does not re-read EEG or alter
the QC policy, feature data, H5 files, or cohort summaries.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(
    r"M:/repos/NeuralManifoldDynamics/project/mnps2_audit/outputs/"
    r"20261116_ds004148_v3_4_frozen_ec_baseline"
)
OUTPUT_DIR = Path(__file__).resolve().parent
TASKS = ("eyesclosed", "eyesopen")
DISPLAY = {"eyesclosed": "EC", "eyesopen": "EO"}
CORE_COORDINATES = (
    "delta_theta_level",
    "alpha_level",
    "beta_alpha_ratio",
    "gamma_level",
    "hjorth_mobility",
    "alpha_theta_ratio",
    "permutation_entropy",
    "alpha_centroid_hz",
)
ALPHA_CV = "alpha_envelope_cv"
MIN_WINDOWS = 30
MIN_UNIQUE_SECONDS = 120.0


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def support_reason(windows: int, seconds: float) -> str:
    reasons: list[str] = []
    if windows < MIN_WINDOWS:
        reasons.append(f"eligible_windows_below_{MIN_WINDOWS}")
    if seconds + 1e-9 < MIN_UNIQUE_SECONDS:
        reasons.append(f"unique_seconds_below_{MIN_UNIQUE_SECONDS:g}")
    return ";".join(reasons)


def main() -> None:
    conditions = read_csv("participant_condition_summaries.csv")
    paired = read_csv("participant_paired_differences.csv")
    complete_pairs = read_csv("participant_complete_9d_pairs.csv")
    qc_windows = read_csv("qc_windows.csv")

    condition_index = {
        (row["subject"], row["coordinate_layer"], row["coordinate"], row["task"]): row
        for row in conditions
    }
    paired_index = {
        (row["subject"], row["coordinate_layer"], row["coordinate"]): row
        for row in paired
        if row["coordinate_layer"] == "frozen_ec_anchored"
    }
    complete_index = {row["subject"]: row for row in complete_pairs}
    qc_by_subject_task: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in qc_windows:
        qc_by_subject_task.setdefault((row["subject"], row["task"]), []).append(row)

    subjects = sorted({row["subject"] for row in conditions})
    output_rows: list[dict[str, Any]] = []
    core_pair_subjects: set[str] = set()
    cv_pair_subjects: set[str] = set()
    complete_pair_subjects: set[str] = set()

    for subject in subjects:
        row: dict[str, Any] = {"subject": subject}
        core_pair_rows = [paired_index[(subject, "frozen_ec_anchored", name)] for name in CORE_COORDINATES]
        core_pair_reasons = sorted({
            item["pair_exclusion_reason"] for item in core_pair_rows
            if item["pair_exclusion_reason"]
        })
        row["core8_paired_coordinate_count"] = sum(
            item["pair_included"].lower() == "true" for item in core_pair_rows
        )
        row["core8_pair_included"] = row["core8_paired_coordinate_count"] == len(CORE_COORDINATES)
        row["core8_pair_exclusion_reason"] = ";".join(core_pair_reasons)

        cv_pair = paired_index[(subject, "frozen_ec_anchored", ALPHA_CV)]
        row["alpha_envelope_cv_pair_included"] = cv_pair["pair_included"].lower() == "true"
        row["alpha_envelope_cv_pair_exclusion_reason"] = cv_pair["pair_exclusion_reason"]

        complete_pair = complete_index[subject]
        row["complete_9d_pair_included"] = (
            complete_pair["both_conditions_minimum_support_met"].lower() == "true"
        )
        row["complete_9d_pair_exclusion_reason"] = complete_pair["pair_exclusion_reason"]

        if row["core8_pair_included"]:
            core_pair_subjects.add(subject)
        if row["alpha_envelope_cv_pair_included"]:
            cv_pair_subjects.add(subject)
        if row["complete_9d_pair_included"]:
            complete_pair_subjects.add(subject)

        for task in TASKS:
            label = DISPLAY[task]
            condition_prefix = "ec" if task == "eyesclosed" else "eo"
            core_rows = [
                condition_index[(subject, "frozen_ec_anchored", name, task)]
                for name in CORE_COORDINATES
            ]
            if len(core_rows) != len(CORE_COORDINATES):
                raise ValueError(f"Missing core-coordinate support rows for {subject} {task}")

            core_n = {int(float(item["qc_eligible_supported_windows"])) for item in core_rows}
            core_seconds = {
                round(float(item["qc_eligible_unique_seconds"]), 6) for item in core_rows
            }
            core_before_qc = {
                int(float(item["supported_finite_windows_before_external_qc"]))
                for item in core_rows
            }
            core_excluded = {
                int(float(item["qc_excluded_supported_windows"])) for item in core_rows
            }
            core_support = {item["minimum_support_met"].lower() == "true" for item in core_rows}
            core_consistent = all(
                len(values) == 1 for values in (core_n, core_seconds, core_before_qc, core_excluded, core_support)
            )
            if not core_consistent:
                raise ValueError(f"Core-8 support diverges by coordinate for {subject} {task}")

            n_windows = next(iter(core_n))
            seconds = next(iter(core_seconds))
            before_qc = next(iter(core_before_qc))
            excluded_supported = next(iter(core_excluded))
            condition_support = next(iter(core_support))
            qc_rows = qc_by_subject_task[(subject, task)]
            overlap_flagged = sum(
                float(item["muscle_zscore_exceedance_fraction"]) > 0 for item in qc_rows
            )
            source_nonfinite = sum(float(item["finite_fraction"]) != 1.0 for item in qc_rows)
            detector_invalid = sum(item["muscle_detector_valid"].lower() != "true" for item in qc_rows)

            cv_condition = condition_index[(subject, "frozen_ec_anchored", ALPHA_CV, task)]
            row.update({
                f"{label}_recorded_windows": len(qc_rows),
                f"{label}_core8_supported_finite_windows_before_qc": before_qc,
                f"{label}_core8_qc_eligible_windows": n_windows,
                f"{label}_core8_qc_excluded_supported_windows": excluded_supported,
                f"{label}_core8_unique_seconds": seconds,
                f"{label}_core8_minimum_support_met": condition_support,
                f"{label}_core8_support_reason": support_reason(n_windows, seconds),
                f"{label}_muscle_overlap_flagged_windows": overlap_flagged,
                f"{label}_source_nonfinite_windows": source_nonfinite,
                f"{label}_detector_invalid_windows": detector_invalid,
                f"{label}_alpha_envelope_cv_qc_eligible_windows": int(
                    float(cv_condition["qc_eligible_supported_windows"])
                ),
                f"{label}_alpha_envelope_cv_unique_seconds": float(
                    cv_condition["qc_eligible_unique_seconds"]
                ),
                f"{label}_alpha_envelope_cv_minimum_support_met": (
                    cv_condition["minimum_support_met"].lower() == "true"
                ),
                f"{label}_complete_9d_qc_eligible_windows": int(
                    complete_pair[f"{condition_prefix}_qc_eligible_complete_9d_windows"]
                ),
                f"{label}_complete_9d_unique_seconds": float(
                    complete_pair[f"{condition_prefix}_unique_seconds"]
                ),
                f"{label}_complete_9d_minimum_support_met": (
                    complete_pair[f"{condition_prefix}_minimum_support_met"].lower() == "true"
                ),
            })

        output_rows.append(row)

    write_csv(OUTPUT_DIR / "participant_qc_attrition.csv", output_rows)
    core_excluded = [row for row in output_rows if not row["core8_pair_included"]]
    cv_additional = [
        row for row in output_rows
        if row["core8_pair_included"] and not row["alpha_envelope_cv_pair_included"]
    ]
    summary = {
        "schema": "mndm.ds004148_qc_attrition.v1",
        "policy_version": "DS004148_BASELINE_EC_QC_POLICY_V1_1",
        "expected_subjects": len(subjects),
        "qc_recordings": len({(row["subject"], row["task"]) for row in qc_windows}),
        "qc_windows": len(qc_windows),
        "core8_coordinates": list(CORE_COORDINATES),
        "core8_paired_subjects": len(core_pair_subjects),
        "core8_excluded_subjects": [row["subject"] for row in core_excluded],
        "alpha_envelope_cv_paired_subjects": len(cv_pair_subjects),
        "alpha_envelope_cv_additional_loss_subjects": [row["subject"] for row in cv_additional],
        "complete_9d_paired_subjects": len(complete_pair_subjects),
        "condition_window_qc": {},
    }
    for task in TASKS:
        label = DISPLAY[task]
        task_rows = [row for row in qc_windows if row["task"] == task]
        summary["condition_window_qc"][label] = {
            "recordings": len({row["subject"] for row in task_rows}),
            "recorded_windows": len(task_rows),
            "muscle_overlap_flagged_windows": sum(
                float(row["muscle_zscore_exceedance_fraction"]) > 0 for row in task_rows
            ),
            "eligible_windows": sum(
                float(row["finite_fraction"]) == 1.0
                and row["muscle_detector_valid"].lower() == "true"
                and float(row["muscle_zscore_exceedance_fraction"]) == 0.0
                for row in task_rows
            ),
        }
    summary["core8_exclusion_reason_counts"] = {}
    for row in core_excluded:
        reason = row["core8_pair_exclusion_reason"] or "no_pair_included_reason_recorded"
        summary["core8_exclusion_reason_counts"][reason] = (
            summary["core8_exclusion_reason_counts"].get(reason, 0) + 1
        )
    (OUTPUT_DIR / "qc_attrition_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# ds004148 baseline QC attrition",
        "",
        "Participant attrition from the completed, frozen v1.1 QC comparison.",
        "Core-8 support means the same participant has at least 30 QC-eligible",
        "windows and at least 120 unique seconds in both EC and EO for each of",
        "the eight listed coordinates. No thresholds were changed after inspecting",
        "the cohort. Window-level measurements remain in the baseline output root.",
        "",
        f"- Core-8 paired participants: {len(core_pair_subjects)}/{len(subjects)}.",
        f"- Alpha-envelope-CV paired participants: {len(cv_pair_subjects)}/{len(subjects)}.",
        f"- Complete-9D paired participants: {len(complete_pair_subjects)}/{len(subjects)}.",
        "- Each EC/EO cell reports QC-eligible windows / unique supported seconds.",
        "",
        "## Participants without a Core-8 pair",
        "",
        "| Subject | EC retained windows / s | EC reason | EO retained windows / s | EO reason |",
        "|---|---:|---|---:|---|",
    ]
    for row in core_excluded:
        lines.append(
            f"| {row['subject']} | {row['EC_core8_qc_eligible_windows']} / "
            f"{row['EC_core8_unique_seconds']:.0f} | {row['EC_core8_support_reason'] or 'pass'} | "
            f"{row['EO_core8_qc_eligible_windows']} / {row['EO_core8_unique_seconds']:.0f} | "
            f"{row['EO_core8_support_reason'] or 'pass'} |"
        )
    lines.extend(["", "## Additional alpha-envelope-CV loss", ""])
    if cv_additional:
        for row in cv_additional:
            lines.append(
                f"{row['subject']}: EC {row['EC_alpha_envelope_cv_qc_eligible_windows']} windows / "
                f"{row['EC_alpha_envelope_cv_unique_seconds']:.0f} s; EO "
                f"{row['EO_alpha_envelope_cv_qc_eligible_windows']} windows / "
                f"{row['EO_alpha_envelope_cv_unique_seconds']:.0f} s. "
                f"Pair reason: {row['alpha_envelope_cv_pair_exclusion_reason']}."
            )
    else:
        lines.append("None beyond the Core-8 exclusions.")
    lines.extend([
        "",
        "## Per-participant condition data",
        "",
        "See `participant_qc_attrition.csv` for all 60 participants, both",
        "conditions, retained and excluded supported windows, unique seconds,",
        "muscle-overlap counts, detector/source validity, alpha-envelope-CV",
        "support, complete-9D support, and pair-exclusion reasons.",
        "",
    ])
    (OUTPUT_DIR / "qc_attrition_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
