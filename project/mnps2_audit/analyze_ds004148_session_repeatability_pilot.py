"""Analyze a bounded ds004148 MNPS 3.4 cross-session pilot."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import spearmanr

REPO = Path(r"M:/repos/NeuralManifoldDynamics")
ROOT = REPO / "project/mnps2_audit/outputs/20261118_ds004148_v3_4_session_repeatability"
BASE = REPO / "project/mnps2_audit/outputs/20261116_ds004148_v3_4_frozen_ec_baseline"
NEW_H5 = ROOT / "pilot_reextracted/processed/ds004148/neuralmanifolddynamics_ds004148_20261118_193222"
QC23 = ROOT / "pilot_cached/qc_sessions23"
SELECTION = ROOT / "selection/selection_manifest.json"
POLICY = REPO / "mndm/config/v.3.4/DS004148_SESSION_REPEATABILITY_V1.md"
ANCHOR = REPO / "mndm/config/v.3.4/anchors/mnps_v3_4_ds005385_ses1_pretask_EyesClosed_20261111_v1.json"
CONFIGS = [
    REPO / "mndm/config/v.3.4/config_ingest_ds004148_v34_baseline_ec_anchor.yml",
    REPO / "mndm/config/v.3.4/config_ingest_ds004148_v34_session23_reextracted_pilot.yml",
]
SHARED = REPO / "project/mnps2_audit/summarize_ds004148_baseline.py"
OUT = ROOT / "pilot_reextracted/analysis"
DOCS = REPO / "mndm/docs/ds004148_session_repeatability_pilot"
ANCHOR_ID = "ds005385_ses1_pretask_EyesClosed_20261111_v1"
ANCHOR_SHA = "79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab"
SUBJECTS = ("sub-01", "sub-02", "sub-03")
SESSIONS = ("session1", "session2", "session3")
TASKS = ("eyesclosed", "eyesopen")
TASK_LABEL = {"eyesclosed": "EC", "eyesopen": "EO"}
SESSION_PAIRS = (("session1", "session2"), ("session1", "session3"), ("session2", "session3"))
MIN_WINDOWS, MIN_SECONDS = 30, 120.0
FILE_RE = re.compile(r"^(sub-\d+)_ses-(session\d+)_(eyesclosed|eyesopen)\.h5$")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_shared():
    spec = importlib.util.spec_from_file_location("ds004148_baseline_summary", SHARED)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SHARED}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.SUBJECT_RE = re.compile(r"^(sub-\d+)_ses-(?:session1|session2|session3)_(eyesclosed|eyesopen)\.h5$")
    mod.ANCHOR_ID, mod.ANCHOR_SHA256 = ANCHOR_ID, ANCHOR_SHA
    return mod


def csv_rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        raise ValueError(f"Empty output: {path}")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def labels_from_h5(path: Path):
    with h5py.File(path, "r") as h5:
        a = np.asarray(h5["measurements/v3_4/provenance/target_measurement_identity/ordered_channel_labels_by_window"][()])
    if a.ndim != 2 or len({tuple(x) for x in a}) != 1:
        raise ValueError(f"Channel label/order varies in {path}")
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a[0]]


def load_record(path: Path, mod):
    m = FILE_RE.match(path.name)
    if not m:
        raise ValueError(f"Unexpected H5 name: {path}")
    subject, session, task = m.groups()
    r = mod.read_h5(path)
    r.update(subject=subject, session=session, task=task, labels=labels_from_h5(path))
    if len(r["starts"]) != 74 or not np.allclose(r["ends"] - r["starts"], 8.0, atol=1e-6):
        raise ValueError(f"Unexpected sample/window support: {path}")
    if not np.allclose(np.diff(r["starts"]), 4.0, atol=1e-6):
        raise ValueError(f"Unexpected window step: {path}")
    if r["pair_signature"]["feature_extraction_implementation_status"] != "single_consistent":
        raise ValueError(f"Unstable per-window implementation identity: {path}")
    for name in r["names"]:
        if r["coordinate_source_units"][name] != mod.EXPECTED_COORDINATE_SOURCE_UNITS[name]:
            raise ValueError(f"Unexpected source units for {name}: {path}")
    return r


def validated_qc(window_rows, recording_rows, session_override=None):
    recordings = {}
    for r in recording_rows:
        session = session_override or r.get("session", "ses-session1").removeprefix("ses-")
        key = (r["subject"], session, r["task"])
        finite = float(r["finite_sample_fraction"]) == 1.0
        score_finite = float(r["muscle_score_finite_fraction"]) == 1.0 and all(
            np.isfinite(float(r[k])) for k in ("muscle_score_p95", "muscle_score_max")
        )
        active = int(r["muscle_detector_active_channel_count"])
        channels = int(r["channel_count_brainvision"])
        if active < 0 or active > channels or key in recordings:
            raise ValueError(f"Invalid/duplicate QC recording {key}")
        recordings[key] = bool(
            finite and score_finite and active > 0 and r["muscle_detector_valid"].lower() == "true"
        )
    windows = {}
    for r in window_rows:
        session = session_override or r["session"].removeprefix("ses-")
        key = (r["subject"], session, r["task"], round(float(r["window_start_sec"]), 6), round(float(r["window_end_sec"]), 6))
        if key in windows or key[:3] not in recordings:
            raise ValueError(f"Duplicate/orphan QC window {key}")
        r["muscle_detector_valid"] = str(
            recordings[key[:3]]
            and r["muscle_detector_valid"].lower() == "true"
            and float(r["muscle_score_finite_fraction"]) == 1.0
        ).lower()
        windows[key] = r
    return windows, recordings


def unique_seconds(intervals):
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    left, right = ordered[0]
    total = 0.0
    for a, b in ordered[1:]:
        if a <= right + 1e-9:
            right = max(right, b)
        else:
            total += right - left
            left, right = a, b
    return total + right - left


def support_ok(n, seconds):
    return n >= MIN_WINDOWS and seconds + 1e-9 >= MIN_SECONDS


def signature_without_order(record):
    result = dict(record["pair_signature"])
    result.pop("channel_order_hashes", None)
    return result


def identity_mismatches(a, b):
    aa, bb = signature_without_order(a), signature_without_order(b)
    return sorted(k for k in set(aa) | set(bb) if aa.get(k) != bb.get(k))


def order_relation(a, b):
    x, y = a["labels"], b["labels"]
    if x == y:
        return "exact_order_match"
    if len(x) == len(y) and all(i.casefold() == j.casefold() for i, j in zip(x, y)):
        return "case_only_label_difference"
    return "ordered_label_difference"


def agreement(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    d = y - x
    bias = float(np.mean(d)) if n else math.nan
    sd = float(np.std(d, ddof=1)) if n > 1 else math.nan
    rho = float(spearmanr(x, y).statistic) if n > 1 and np.std(x) > 0 and np.std(y) > 0 else math.nan
    icc = math.nan
    if n >= 2:
        data = np.column_stack((x, y)); k = 2; grand = np.mean(data)
        subject_mean = np.mean(data, axis=1); session_mean = np.mean(data, axis=0)
        ms_subject = k * np.sum((subject_mean - grand) ** 2) / (n - 1)
        ms_session = n * np.sum((session_mean - grand) ** 2) / (k - 1)
        residual = data - subject_mean[:, None] - session_mean[None, :] + grand
        ms_error = np.sum(residual**2) / ((n - 1) * (k - 1))
        denom = ms_subject + (k - 1) * ms_error + k * (ms_session - ms_error) / n
        if np.isfinite(denom) and denom != 0:
            icc = float((ms_subject - ms_error) / denom)
    return {
        "n": n, "icc_a1": icc, "bias_later_minus_earlier": bias,
        "sd_difference_sample": sd,
        "bland_altman_95_lower": bias - 1.96 * sd if n > 1 else math.nan,
        "bland_altman_95_upper": bias + 1.96 * sd if n > 1 else math.nan,
        "median_absolute_difference": float(np.median(np.abs(d))) if n else math.nan,
        "spearman_rank_correlation": rho,
    }


def fmt(v):
    try:
        x = float(v)
    except (ValueError, TypeError):
        return str(v)
    return "—" if not np.isfinite(x) else f"{x:.3f}"


def main():
    OUT.mkdir(parents=True, exist_ok=True); DOCS.mkdir(parents=True, exist_ok=True)
    mod = load_shared()
    if sha(ANCHOR) != ANCHOR_SHA:
        raise ValueError("Frozen anchor file hash mismatch")
    if not json.loads(SELECTION.read_text(encoding="utf-8")).get("metadata_gate_pass"):
        raise ValueError("Frozen metadata-only selection gate failed")

    old_paths = sorted((BASE / "processed/ds004148").rglob("sub-0[1-3]_ses-session1_*.h5"))
    new_paths = sorted(NEW_H5.rglob("*.h5"))
    records = [load_record(p, mod) for p in old_paths + new_paths]
    idx = {(r["subject"], r["session"], r["task"]): r for r in records}
    expected = {(s, v, t) for s in SUBJECTS for v in SESSIONS for t in TASKS}
    if len(records) != 18 or set(idx) != expected:
        raise ValueError(f"Expected 18 selected H5s; got {len(records)}")

    q1, rec1 = validated_qc(csv_rows(BASE / "qc_windows.csv"), csv_rows(BASE / "qc_recordings.csv"))
    q23, rec23 = validated_qc(csv_rows(QC23 / "qc_windows.csv"), csv_rows(QC23 / "qc_recordings.csv"))
    q = {k: v for k, v in q1.items() if k[0] in SUBJECTS}
    q.update({k: v for k, v in q23.items() if k[0] in SUBJECTS})
    qrec = {k: v for k, v in rec1.items() if k[0] in SUBJECTS}
    qrec.update({k: v for k, v in rec23.items() if k[0] in SUBJECTS})
    expected_windows = {
        (r["subject"], r["session"], r["task"], round(float(a), 6), round(float(b), 6))
        for r in records for a, b in zip(r["starts"], r["ends"])
    }
    if set(q) != expected_windows:
        raise ValueError(f"QC/H5 windows mismatch: missing={len(expected_windows-set(q))}, extra={len(set(q)-expected_windows)}")

    # Scalar v3.4 estimators aggregate across channels. Require identical channel
    # compatibility set/surface/reference; preserve order/case differences as QC.
    compatibility, compat_rows = {}, []
    for s in SUBJECTS:
        for session in SESSIONS:
            ec, eo = idx[(s, session, "eyesclosed")], idx[(s, session, "eyesopen")]
            bad = identity_mismatches(ec, eo)
            ok = not bad
            compatibility[(s, session, "within")] = (ok, ";".join(bad))
            compat_rows.append({
                "subject": s, "session": session, "comparison": "EC_vs_EO",
                "compatible": ok, "identity_mismatches": ";".join(bad),
                "channel_order_relation": order_relation(ec, eo),
                "channel_compatibility_set_hash": ec["pair_signature"]["channel_compatibility_set_hash"],
                "source_surface_hash": ec["pair_signature"]["source_surface_hash"],
            })
        for a, b in SESSION_PAIRS:
            for task in TASKS:
                left, right = idx[(s, a, task)], idx[(s, b, task)]
                bad = identity_mismatches(left, right); ok = not bad
                compatibility[(s, f"{a}:{b}", task)] = (ok, ";".join(bad))
                compat_rows.append({
                    "subject": s, "session": f"{a}:{b}", "comparison": task,
                    "compatible": ok, "identity_mismatches": ";".join(bad),
                    "channel_order_relation": order_relation(left, right),
                    "channel_compatibility_set_hash": left["pair_signature"]["channel_compatibility_set_hash"],
                    "source_surface_hash": left["pair_signature"]["source_surface_hash"],
                })

    levels, complete = [], []
    for key, r in idx.items():
        s, session, task = key
        qs = [q[(s, session, task, round(float(a), 6), round(float(b), 6))] for a, b in zip(r["starts"], r["ends"])]
        qok = np.array([mod.eligible(row) for row in qs], dtype=bool)
        for ci, coord in enumerate(r["names"]):
            supported = r["raw_support"][:, ci] & r["anchored_valid"][:, ci] & np.isfinite(r["anchored"][:, ci])
            good = supported & qok
            ix = np.flatnonzero(good); vals = r["anchored"][ix, ci].astype(float)
            seconds = unique_seconds([(float(r["starts"][j]), float(r["ends"][j])) for j in ix])
            levels.append({
                "subject": s, "session": session, "condition": TASK_LABEL[task], "task": task,
                "coordinate": coord, "unit": "dimensionless anchored score",
                "eligible_qc_windows": int(qok.sum()), "supported_before_qc": int(supported.sum()),
                "qc_eligible_supported_windows": len(ix), "qc_excluded_supported_windows": int(supported.sum()-len(ix)),
                "unique_seconds": seconds, "minimum_support_met": support_ok(len(ix), seconds),
                "median": float(np.median(vals)) if len(vals) else math.nan,
                "mean": float(np.mean(vals)) if len(vals) else math.nan,
                "q1": float(np.quantile(vals, .25)) if len(vals) else math.nan,
                "q3": float(np.quantile(vals, .75)) if len(vals) else math.nan,
            })
        all9 = r["complete_9d"] & np.all(r["anchored_valid"], axis=1) & qok
        ix = np.flatnonzero(all9); sec = unique_seconds([(float(r["starts"][j]), float(r["ends"][j])) for j in ix])
        complete.append({
            "subject": s, "session": session, "condition": TASK_LABEL[task],
            "qc_eligible_complete_9d_windows": len(ix), "unique_seconds": sec,
            "minimum_support_met": support_ok(len(ix), sec),
        })

    li = {(r["subject"], r["session"], r["task"], r["coordinate"]): r for r in levels}
    contrasts, ci = [], {}
    for s in SUBJECTS:
        for session in SESSIONS:
            within_ok, within_reason = compatibility[(s, session, "within")]
            for coord in mod.RAW_UNITS:
                ec, eo = li[(s, session, "eyesclosed", coord)], li[(s, session, "eyesopen", coord)]
                estimable = within_ok and np.isfinite(ec["median"]) and np.isfinite(eo["median"])
                support = bool(estimable and ec["minimum_support_met"] and eo["minimum_support_met"])
                row = {
                    "subject": s, "session": session, "coordinate": coord,
                    "ec_median": ec["median"], "eo_median": eo["median"],
                    "ec_n": ec["qc_eligible_supported_windows"], "eo_n": eo["qc_eligible_supported_windows"],
                    "ec_seconds": ec["unique_seconds"], "eo_seconds": eo["unique_seconds"],
                    "within_session_compatible": within_ok, "compatibility_note": within_reason,
                    "eo_minus_ec": float(eo["median"]-ec["median"]) if estimable else math.nan,
                    "contrast_estimable": bool(estimable), "eligible_for_repeatability": support,
                    "exclusion_reason": "" if support else (
                        within_reason or ("EC_support_below_floor" if not ec["minimum_support_met"] else "")
                        or ("EO_support_below_floor" if not eo["minimum_support_met"] else "")
                    ),
                }
                contrasts.append(row); ci[(s, session, coord)] = row

    summary, elig = [], []
    for typ in ("level", "EO_minus_EC"):
        for a, b in SESSION_PAIRS:
            for coord in mod.RAW_UNITS:
                conditions = TASKS if typ == "level" else (None,)
                for task in conditions:
                    xs, ys, excluded = [], [], []
                    condition = TASK_LABEL[task] if task else "EO_minus_EC"
                    for s in SUBJECTS:
                        if task:
                            ra, rb = li[(s, a, task, coord)], li[(s, b, task, coord)]
                            ok, reason = compatibility[(s, f"{a}:{b}", task)]
                            xa, xb = ra["median"], rb["median"]
                            sa, sb = ra["minimum_support_met"], rb["minimum_support_met"]
                        else:
                            ra, rb = ci[(s, a, coord)], ci[(s, b, coord)]
                            ca, cb = compatibility[(s, f"{a}:{b}", "eyesclosed")], compatibility[(s, f"{a}:{b}", "eyesopen")]
                            ok = ca[0] and cb[0]; reason = ";".join(x[1] for x in (ca, cb) if x[1])
                            xa, xb = ra["eo_minus_ec"], rb["eo_minus_ec"]
                            sa, sb = ra["eligible_for_repeatability"], rb["eligible_for_repeatability"]
                        include = bool(ok and sa and sb and np.isfinite(xa) and np.isfinite(xb))
                        if include:
                            xs.append(float(xa)); ys.append(float(xb))
                        else:
                            excluded.append(f"{s}:" + (reason or "support_or_value_unavailable"))
                        elig.append({
                            "measure_type": typ, "condition": condition, "coordinate": coord,
                            "session_a": a, "session_b": b, "subject": s, "included": include,
                            "value_a": xa, "value_b": xb, "support_a": sa, "support_b": sb,
                            "exclusion_reason": "" if include else (reason or "support_or_value_unavailable"),
                        })
                    metrics = agreement(xs, ys)
                    summary.append({
                        "measure_type": typ, "condition": condition, "coordinate": coord,
                        "session_a": a, "session_b": b, "n_candidates": 3, "n_included": metrics["n"],
                        "excluded_subject_reasons": ";".join(excluded), **metrics,
                    })

    write_csv(OUT / "participant_session_levels.csv", levels)
    write_csv(OUT / "participant_session_eo_minus_ec.csv", contrasts)
    write_csv(OUT / "complete_9d_support.csv", complete)
    write_csv(OUT / "realization_compatibility.csv", compat_rows)
    write_csv(OUT / "repeatability_eligibility.csv", elig)
    write_csv(OUT / "repeatability_summary.csv", summary)

    notes = [r for r in compat_rows if r["channel_order_relation"] != "exact_order_match"]
    n_supported = sum(bool(r["minimum_support_met"]) for r in levels)
    n_complete = sum(bool(r["minimum_support_met"]) for r in complete)
    support_failures = [r for r in levels if not r["minimum_support_met"]]
    failed_groups = sorted({
        (r["subject"], r["session"], r["condition"], r["qc_eligible_supported_windows"], r["unique_seconds"])
        for r in support_failures
    })
    attrition_note = "; ".join(
        f"{subject} {session} {condition}: {n} eligible windows, {seconds:g} unique seconds"
        for subject, session, condition, n, seconds in failed_groups
    ) or "none"
    nrange = [min(r["n_included"] for r in summary), max(r["n_included"] for r in summary)]
    show = []
    for r in summary:
        show.append("| {condition} | {coordinate} | {session_a} to {session_b} | {n_included}/3 | {icc} | {bias} | {mad} |".format(
            **r, icc=fmt(r["icc_a1"]), bias=fmt(r["bias_later_minus_earlier"]),
            mad=fmt(r["median_absolute_difference"])
        ))
    table = "\n".join([
        "| Measure | Coordinate | Sessions | n | ICC(A,1) | Bias | Median |",
        "|---|---|---|---:|---:|---:|---:|", *show,
    ])
    report = f"""# ds004148 MNPS 3.4 session-repeatability pilot

**Descriptive three-participant pilot only.** One frozen ds005385 EyesClosed anchor was used throughout; no target-session refit. QC v1.1 and the support floor (at least {MIN_WINDOWS} eligible windows and {MIN_SECONDS:g} unique seconds) were fixed before examining sessions 2-3 coordinates.

## Processing and support

- Reused session-1 H5/QC for sub-01-sub-03. Re-extracted sessions 2-3 from source with current MNPS 3.4 code; the old cache was refused by the feature-implementation identity gate and was not used.
- Checked 18 H5s (3 participants x 3 sessions x EC/EO): same anchor, consistent feature implementation, expected raw units and identical 8 s windows/4 s steps. QC matched all {len(expected_windows)} H5 windows.
- {n_supported}/{len(levels)} participant x session x condition x coordinate summaries met the support floor. Complete 9D support met it in {n_complete}/{len(complete)} session-condition rows. Per-coordinate counts and support are retained in the CSV outputs.
- The support loss is concentrated in: {attrition_note}. This removes that participant from every EC level comparison involving session 1 and from session 1 EO-EC repeatability, leaving n=2 for those rows.
- Two participants (sub-02 and sub-03) have case-only channel-label change CPz/Cpz in session 3. Their ordered channel positions, normalized channel set, source surface and reference match. The v3.4 scalar estimators aggregate over channels and do not depend on channel position; the label difference is retained as QC metadata, and these comparisons are included.
- Included participant counts across the 81 repeatability rows range from {nrange[0]} to {nrange[1]} of 3. ICC(A,1), agreement differences and correlations are highly unstable at this sample size; they are descriptive pilot values, not dataset-level reliability estimates.

## Session comparisons

Bias is later session minus earlier session. Median is the median absolute difference. Full sample SD, 95% Bland-Altman limits and Spearman rank correlation are in artifacts/repeatability_summary.csv; Spearman is a rank-order description, not agreement.

{table}

All anchored coordinates are dimensionless (including alpha_centroid_hz). EO-EC is the within-participant/session difference between condition medians using the same anchor.

## Interpretation and scope

This completes the bounded repeatability workflow technically and preserves both condition levels and EO-EC changes for the three preselected pilot participants. It does not establish population-level repeatability, biological stability, artifact absence or clinical qualification. No p-values, Jacobian work, threshold exploration or anchor refitting were performed. The full 60-participant sessions 1-3 cohort was not processed; the full feature run was stopped after runtime extrapolation.

The frozen method is documented in mndm/config/v.3.4/DS004148_SESSION_REPEATABILITY_V1.md. The analysis implementation is project/mnps2_audit/analyze_ds004148_session_repeatability_pilot.py and reuses schema/anchor checks from project/mnps2_audit/summarize_ds004148_baseline.py. H5 and raw EEG remain in local pilot outputs; raw EEG binaries were not hashed.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    code_inputs = [
        POLICY, ANCHOR, *CONFIGS, SELECTION, SHARED, Path(__file__),
        REPO / "mndm/src/mndm/features/eeg.py",
        REPO / "mndm/src/mndm/mnps/v3_4/candidates.py",
        REPO / "mndm/src/mndm/mnps/v3_4/coordinates.py",
        REPO / "mndm/src/mndm/mnps/v3_4/external_anchor.py",
        REPO / "mndm/src/mndm/pipeline/summary.py",
        BASE / "qc_windows.csv", BASE / "qc_recordings.csv", BASE / "qc_assessment_manifest.json",
        QC23 / "qc_windows.csv", QC23 / "qc_recordings.csv", QC23 / "qc_assessment_manifest.json",
        next((BASE / "processed/ds004148").glob("neural*/run_manifest.json")),
        NEW_H5 / "run_manifest.json",
    ]
    manifest = {
        "schema": "mndm.ds004148_session_repeatability_analysis.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "selection": {"subjects": list(SUBJECTS), "sessions": list(SESSIONS), "tasks": list(TASKS), "basis": "metadata-only frozen pilot subset"},
        "anchor": {"id": ANCHOR_ID, "sha256": ANCHOR_SHA},
        "methods": {
            "qc": "fixed baseline policy v1.1; finite source/detector, valid detector, zero score>4 sample overlap",
            "support": {"eligible_windows": MIN_WINDOWS, "unique_seconds": MIN_SECONDS},
            "summary": "median of QC-eligible supported windows per coordinate, participant, session and condition",
            "contrast": "EO median minus EC median in the same participant/session",
            "comparability": "same anchor, source units, estimator identity, normalized channel set, source surface, reference and time grid; channel order/case changes retained as QC when the channel-aggregated estimator remains permutation invariant",
            "statistics": "ICC(A,1); bias, sample SD and 95% Bland-Altman limits; median absolute difference; Spearman rank correlation descriptive only; no p-values",
        },
        "counts": {"h5": len(records), "matched_qc_h5_windows": len(expected_windows), "supported_coordinate_rows": n_supported, "complete_9d_rows_meeting_support": n_complete, "repeatability_rows": len(summary), "included_n_range": nrange},
        "channel_order_qc_notes": notes,
        "input_hashes": [{"path": str(p), "sha256": sha(p)} for p in code_inputs],
        "h5_hashes": [{"path": str(r["path"]), "sha256": sha(r["path"])} for r in records],
        "raw_eeg_binary_hashes_computed": False,
        "limitations": ["Three-person pilot; repeatability statistics are unstable.", "QC does not establish artifact absence.", "No population, biological or clinical qualification claim.", "Full 60-participant repeatability cohort not processed."],
    }
    for name in ("participant_session_levels.csv", "participant_session_eo_minus_ec.csv", "complete_9d_support.csv", "realization_compatibility.csv", "repeatability_eligibility.csv", "repeatability_summary.csv", "report.md"):
        manifest.setdefault("output_hashes", {})[name] = sha(OUT / name)
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (DOCS / "artifacts").mkdir(parents=True, exist_ok=True)
    for name in ("participant_session_levels.csv", "participant_session_eo_minus_ec.csv", "complete_9d_support.csv", "realization_compatibility.csv", "repeatability_eligibility.csv", "repeatability_summary.csv", "analysis_manifest.json"):
        (DOCS / "artifacts" / name).write_bytes((OUT / name).read_bytes())
    (DOCS / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "docs": str(DOCS), "h5": len(records), "windows": len(expected_windows), "supported_coordinate_rows": n_supported, "complete_9d_support_rows": n_complete, "included_n_range": nrange, "channel_order_notes": notes}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
