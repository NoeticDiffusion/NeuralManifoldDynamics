"""Analyze the frozen ds004148 MNPS 3.4 full-cohort cross-session run."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.stats import spearmanr

REPO = Path(r"M:/repos/NeuralManifoldDynamics")
ROOT = REPO / "project/mnps2_audit/outputs/20261118_ds004148_v3_4_session_repeatability"
BASE = REPO / "project/mnps2_audit/outputs/20261116_ds004148_v3_4_frozen_ec_baseline"
NEW_H5 = ROOT / "full_reextracted/processed/ds004148/neuralmanifolddynamics_ds004148_20261118_205216"
QC23 = ROOT / "full_cohort/qc_sessions23"
SELECTION = ROOT / "selection/selection_manifest.json"
POLICY = REPO / "mndm/config/v.3.4/DS004148_SESSION_REPEATABILITY_V1.md"
ANCHOR = REPO / "mndm/config/v.3.4/anchors/mnps_v3_4_ds005385_ses1_pretask_EyesClosed_20261111_v1.json"
CONFIGS = [
    REPO / "mndm/config/v.3.4/config_ingest_ds004148_v34_baseline_ec_anchor.yml",
    REPO / "mndm/config/v.3.4/config_ingest_ds004148_v34_session23_fullcohort_repeatability.yml",
]
SHARED = REPO / "project/mnps2_audit/summarize_ds004148_baseline.py"
OUT = ROOT / "full_reextracted/analysis"
DOCS = REPO / "mndm/docs/ds004148_session_repeatability_fullcohort"
ANCHOR_ID = "ds005385_ses1_pretask_EyesClosed_20261111_v1"
ANCHOR_SHA = "79da8c208a6c9240e84cefee7abc9a26c3121d59c1c86bc3324d33172193d1ab"
SUBJECTS = tuple(f"sub-{i:02d}" for i in range(1, 61))
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

    old_paths = sorted((BASE / "processed/ds004148").rglob("sub-*_ses-session1_*.h5"))
    new_paths = sorted(NEW_H5.rglob("*.h5"))
    records = [load_record(p, mod) for p in old_paths + new_paths]
    idx = {(r["subject"], r["session"], r["task"]): r for r in records}
    expected = {(s, v, t) for s in SUBJECTS for v in SESSIONS for t in TASKS}
    if len(records) != 360 or set(idx) != expected:
        raise ValueError(f"Expected 360 selected H5s; got {len(records)}")

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
                        "session_a": a, "session_b": b, "n_candidates": len(SUBJECTS), "n_included": metrics["n"],
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
        (r["subject"], r["session"], r["condition"])
        for r in support_failures
    })
    failed_by_group = [
        f"{session} {condition}: {sum(1 for _, s, c in failed_groups if s == session and c == condition)}"
        for session in SESSIONS for condition in ("EC", "EO")
        if any(s == session and c == condition for _, s, c in failed_groups)
    ]
    attrition_note = (
        f"{len(support_failures)} coordinate summaries across {len(failed_groups)} participant-session-condition groups "
        f"({'; '.join(failed_by_group)})"
    ) if support_failures else "none"
    nrange = [min(r["n_included"] for r in summary), max(r["n_included"] for r in summary)]
    show = []
    for r in summary:
        show.append("| {condition} | {coordinate} | {session_a} to {session_b} | {n_included}/60 | {icc} | {bias} | {mad} |".format(
            **r, icc=fmt(r["icc_a1"]), bias=fmt(r["bias_later_minus_earlier"]),
            mad=fmt(r["median_absolute_difference"])
        ))
    table = "\n".join([
        "| Measure | Coordinate | Sessions | n | ICC(A,1) | Bias | Median |Δ| |",
        "|---|---|---|---:|---:|---:|---:|", *show,
    ])
    report = f"""# ds004148 MNPS 3.4 session repeatability: full cohort

**Descriptive full-cohort analysis.** The metadata-only selection fixed all 60 participants, sessions 1–3, and resting EC/EO recordings before coordinate inspection. Every recording used the same frozen ds005385 EyesClosed anchor; target sessions were not used to refit it. The source-level QC rule and support floor (at least {MIN_WINDOWS} eligible windows and {MIN_SECONDS:g} unique seconds) were fixed in advance.

## Processing and support

- Reused session-1 H5 and QC from the frozen baseline and re-extracted sessions 2–3 with the current MNPS 3.4 implementation. Session 1 was not reprocessed.
- Checked {len(records)} H5s (60 participants × 3 sessions × EC/EO): frozen anchor ID/hash, per-window implementation identity, expected source units, and the expected 8-second windows at 4-second steps. QC matched all {len(expected_windows)} H5 windows.
- {n_supported}/{len(levels)} participant × session × condition × coordinate summaries met the support floor. Complete 9D support met it in {n_complete}/{len(complete)} participant-session-condition rows. Coordinate-specific support is preserved in the tables.
- Support-floor failures: {attrition_note}. Per-session/condition counts are groups with at least one coordinate below the support floor; detailed coordinate-level counts and exclusion reasons are in `repeatability_eligibility.csv` and `participant_session_levels.csv`.
- {len(notes)} realization-comparison rows have a channel-order or case-only label difference. These are reported as QC; comparisons remain eligible only when the normalized channel set, source surface, reference, preprocessing, estimator identity, time grid and units match.
- Included participant counts across the 81 repeatability rows range from {nrange[0]} to {nrange[1]} of 60.

## Session comparisons

Bias is later session minus earlier session. Median is the median absolute difference. Sample SD, 95% Bland–Altman limits and Spearman rank correlation are in `repeatability_summary.csv`; Spearman describes rank order, not agreement. Each row uses participants who pass the predeclared support and compatibility rules in both sessions.

{table}

All anchored coordinates are dimensionless (including `alpha_centroid_hz`). EO−EC is the within-participant/session difference between the EC and EO coordinate medians using the same anchor. The comparison is descriptive and does not use p-values.

## Interpretation and scope

This completes the frozen-anchor session-repeatability analysis for the selected 60-participant ds004148 cohort. It estimates repeatability of this measurement procedure in these recordings; it does not establish biological stability, artifact absence, cross-dataset equivalence or clinical qualification. QC eligibility is an operational screen, not evidence that a window is artifact-free.

The analysis uses only MNPS 3.4 coordinate values and their declared support. The run disabled v3.1 experimental measurements and the v3.4 candidate Jacobian. Standard summarize still emitted its primary 3D and 9D Jacobian datasets; they are not analyzed or used in these summaries. No anchor refitting, threshold search or inferential p-values were used.

The frozen method is documented in `mndm/config/v.3.4/DS004148_SESSION_REPEATABILITY_V1.md`. The reproducibility package contains the selection, QC and full-cohort analysis manifests, scripts, and summary tables. EEG source files and generated H5 files remain in the local output tree; source EEG binaries were not hashed.
"""
    (OUT / "report.md").write_text(report, encoding="utf-8")

    code_inputs = [
        POLICY, ANCHOR, *CONFIGS, SELECTION, SHARED, Path(__file__),
        REPO / "project/mnps2_audit/run_ds004148_session23_fullcohort_qc.py",
        REPO / "project/mnps2_audit/run_ds004148_v34_baseline_qc.py",
        REPO / "project/mnps2_audit/prepare_ds004148_session_repeatability_selection.py",
        REPO / "mndm/config/v.3.4/config_ingest_ds004148_v34_session23_frozen_ec_anchor.yml",
        REPO / "mndm/src/mndm/features/eeg.py",
        REPO / "mndm/src/mndm/preprocess.py",
        REPO / "mndm/src/mndm/parallel.py",
        REPO / "core/src/core/io/h5_writer.py",
        REPO / "mndm/src/mndm/mnps/v3_4/__init__.py",
        REPO / "mndm/src/mndm/mnps/v3_4/candidates.py",
        REPO / "mndm/src/mndm/mnps/v3_4/coordinates.py",
        REPO / "mndm/src/mndm/mnps/v3_4/external_anchor.py",
        REPO / "mndm/src/mndm/pipeline/summary.py",
        BASE / "qc_windows.csv", BASE / "qc_recordings.csv", BASE / "qc_assessment_manifest.json",
        QC23 / "qc_windows.csv", QC23 / "qc_recordings.csv", QC23 / "qc_assessment_manifest.json",
        next((BASE / "processed/ds004148").glob("neural*/run_manifest.json")),
        NEW_H5 / "run_manifest.json",
    ]
    full_qc_manifest = json.loads((QC23 / "qc_assessment_manifest.json").read_text(encoding="utf-8"))
    feature_run_manifest = json.loads((NEW_H5 / "run_manifest.json").read_text(encoding="utf-8"))
    package_names = ("numpy", "scipy", "pandas", "h5py", "mne", "PyYAML", "pyarrow")
    package_versions = {}
    for package in package_names:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    manifest = {
        "schema": "mndm.ds004148_session_repeatability_analysis.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "analysis_python": sys.version,
            "analysis_platform": platform.platform(),
            "package_versions": package_versions,
            "qc_processing_runtime": full_qc_manifest.get("runtime", {}),
            "feature_processing_runtime": feature_run_manifest.get("environment", {}),
            "feature_processing_flags": {"MNE_USE_NUMBA": "false", "NUMBA_DISABLE_JIT": "1"},
        },
        "selection": {"subjects": list(SUBJECTS), "sessions": list(SESSIONS), "tasks": list(TASKS), "basis": "metadata-only frozen full cohort"},
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
        "limitations": ["Descriptive dataset-specific repeatability; not biological stability or clinical qualification.", "QC does not establish artifact absence.", "Jacobians are present in standard H5 exports but are not analyzed.", "Source EEG binaries were not hashed."],
    }
    for name in ("participant_session_levels.csv", "participant_session_eo_minus_ec.csv", "complete_9d_support.csv", "realization_compatibility.csv", "repeatability_eligibility.csv", "repeatability_summary.csv", "report.md"):
        manifest.setdefault("output_hashes", {})[name] = sha(OUT / name)
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (DOCS / "artifacts").mkdir(parents=True, exist_ok=True)
    for name in ("participant_session_levels.csv", "participant_session_eo_minus_ec.csv", "complete_9d_support.csv", "realization_compatibility.csv", "repeatability_eligibility.csv", "repeatability_summary.csv", "analysis_manifest.json"):
        (DOCS / "artifacts" / name).write_bytes((OUT / name).read_bytes())
    provenance_files = [
        (SELECTION, DOCS / "artifacts/selection_manifest.json"),
        (BASE / "qc_assessment_manifest.json", DOCS / "artifacts/session1_qc_manifest.json"),
        (QC23 / "qc_assessment_manifest.json", DOCS / "artifacts/sessions23_qc_manifest.json"),
        (NEW_H5 / "run_manifest.json", DOCS / "artifacts/session23_feature_run_manifest.json"),
    ]
    for source, destination in provenance_files:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    reproducibility_files = [
        (Path(__file__), "scripts/analyze_ds004148_fullcohort_session_repeatability.py"),
        (REPO / "project/mnps2_audit/run_ds004148_session23_fullcohort_qc.py", "scripts/run_ds004148_session23_fullcohort_qc.py"),
        (REPO / "project/mnps2_audit/run_ds004148_v34_baseline_qc.py", "scripts/run_ds004148_v34_baseline_qc.py"),
        (REPO / "project/mnps2_audit/prepare_ds004148_session_repeatability_selection.py", "scripts/prepare_ds004148_session_repeatability_selection.py"),
        (SHARED, "scripts/summarize_ds004148_baseline.py"),
        (REPO / "mndm/config/v.3.4/config_ingest_ds004148_v34_baseline_ec_anchor.yml", "config/config_ingest_ds004148_v34_baseline_ec_anchor.yml"),
        (REPO / "mndm/config/v.3.4/config_ingest_ds004148_v34_session23_fullcohort_repeatability.yml", "config/config_ingest_ds004148_v34_session23_fullcohort_repeatability.yml"),
        (REPO / "mndm/config/v.3.4/config_ingest_ds004148_v34_session23_frozen_ec_anchor.yml", "config/config_ingest_ds004148_v34_session23_frozen_ec_anchor.yml"),
        (REPO / "mndm/config/v.3.4/config_ingest_common_eeg_9d.yml", "config/config_ingest_common_eeg_9d.yml"),
        (POLICY, "config/DS004148_SESSION_REPEATABILITY_V1.md"),
        (REPO / "mndm/config/v.3.4/DS004148_BASELINE_EC_QC_POLICY_V1_1.md", "config/DS004148_BASELINE_EC_QC_POLICY_V1_1.md"),
        (ANCHOR, "config/mnps_v3_4_ds005385_ses1_pretask_EyesClosed_20261111_v1.json"),
    ]
    for source, relative in reproducibility_files:
        destination = DOCS / "reproduction" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    (DOCS / "README.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "docs": str(DOCS), "h5": len(records), "windows": len(expected_windows), "supported_coordinate_rows": n_supported, "complete_9d_support_rows": n_complete, "included_n_range": nrange, "channel_order_notes": len(notes)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
