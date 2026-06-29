"""
pr03_shortwindow_robustness.py
==============================
Peer-review follow-up Task 3: 4 s window robustness check (F2/F3 purity).

This script is run AFTER a new pipeline run with window_sec=4, step_sec=2.
It:
  1. Re-runs the 04b encoding-phase analysis on the 4s run directory.
  2. Re-runs the 04c purity audit on the 4s run directory.
  3. Reads the 8s comparison data from the NoeticDiffusion handoff package.
  4. Produces a combined filter comparison table:
       filter | window_8s_p_m | window_4s_p_m | window_8s_p_d | window_4s_p_d
  5. Reports whether F2/F3 are now achievable for Fast/Simultaneous.

Usage:
  python project/scripts/pr03_shortwindow_robustness.py \\
      --run-dir-4s J:/processed/openneuro/ds006848/<new_4s_run_dir> \\
      --run-dir-8s J:/repos/NoeticDiffusion/data/raw/neuralmanifolddynamics_ds006848_20260626_114620 \\
      --bids-dir K:/ExternalReceivedDatasets/openneuro/received/ds006848 \\
      --out-dir J:/repos/NoeticDiffusion/articles/embodied_anchoring_follow_up/results/peer_review_followup
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import warnings
from datetime import date
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")

SCRIPTS_DIR = Path(__file__).parent
BIDS_DIR_DEFAULT = Path("K:/ExternalReceivedDatasets/openneuro/received/ds006848")
RUN_8S_DEFAULT = Path(
    "J:/repos/NoeticDiffusion/data/raw/neuralmanifolddynamics_ds006848_20260626_114620"
)
OUT_DIR_DEFAULT = Path(
    "J:/repos/NoeticDiffusion/articles/embodied_anchoring_follow_up/results/peer_review_followup"
)

FILTER_NAMES = ["F0_any25", "F1_centre_in", "F2_enc50", "F3_enc70", "F4_weighted"]


# ── Helper: extract filter p-values from a 04c audit_summary.txt ─────────────
def parse_audit_summary(summary_txt: Path) -> Dict[str, Dict[str, float]]:
    """
    Returns {filter: {metric: p_value}} from audit_summary.txt.
    Also captures condition medians for the pattern check.
    """
    if not summary_txt.exists():
        return {}
    text = summary_txt.read_text(encoding="utf-8")
    results: Dict[str, Dict[str, float]] = {}
    lines = text.splitlines()
    current_filter = None
    for line in lines:
        # Detect filter section headers like "F0", "F1" etc.
        stripped = line.strip()
        for fn in FILTER_NAMES:
            if stripped.startswith(fn + ":") or stripped.startswith(fn + " "):
                current_filter = fn
                break
    return results


def parse_04c_friedman_csv(friedman_csv: Path) -> pd.DataFrame:
    """Load friedman_results.csv from 04c purity audit."""
    if not friedman_csv.exists():
        return pd.DataFrame()
    return pd.read_csv(friedman_csv)


# ── Run 04b and 04c on the 4s run dir ────────────────────────────────────────
def run_04b(run_dir: Path, bids_dir: Path, out_dir: Path) -> Path:
    """Run 04b_encoding_phase_analysis.py and return its output directory."""
    script = SCRIPTS_DIR / "04b_encoding_phase_analysis.py"
    out_04b = out_dir / "04b_4s"
    out_04b.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--run-dir",
        str(run_dir),
        "--bids-dir",
        str(bids_dir),
        "--out-dir",
        str(out_04b),
    ]
    print(f"[pr03] Running 04b on 4s data: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[pr03] WARNING: 04b exited with code {result.returncode}")
    return out_04b


def run_04c(run_dir: Path, bids_dir: Path, out_dir: Path) -> Path:
    """Run 04c_window_overlap_purity_audit.py and return its output directory."""
    script = SCRIPTS_DIR / "04c_window_overlap_purity_audit.py"
    out_04c = out_dir / "04c_4s"
    out_04c.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(script),
        "--run-dir",
        str(run_dir),
        "--bids-dir",
        str(bids_dir),
        "--out-dir",
        str(out_04c),
    ]
    print(f"[pr03] Running 04c on 4s data: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"[pr03] WARNING: 04c exited with code {result.returncode}")
    return out_04c


# ── Load 04c Friedman results from a run ────────────────────────────────────
def load_04c_filter_results(out_04c: Path) -> pd.DataFrame:
    """
    Reads the friedman_by_filter.csv from a 04c run.
    Expected columns: filter, test, metric, chi2, p_friedman, n_subjects
    """
    csv = out_04c / "friedman_by_filter.csv"
    if not csv.exists():
        print(f"[pr03] WARNING: no Friedman CSV found in {out_04c}")
        return pd.DataFrame()
    df = pd.read_csv(csv)
    if "p_friedman" in df.columns and "p" not in df.columns:
        df = df.rename(columns={"p_friedman": "p"})
    return df


def load_04c_condition_medians(out_04c: Path) -> pd.DataFrame:
    """Load condition medians per filter from 04c."""
    csv = out_04c / "filter_comparison.csv"
    if not csv.exists():
        return pd.DataFrame()
    return pd.read_csv(csv)


# ── Assess F2/F3 achievability (geometric check) ─────────────────────────────
def assess_filter_achievability(out_04c: Path) -> Dict[str, Dict[str, int]]:
    """
    Count windows per condition passing F2 (>=50%) and F3 (>=70%) encoding overlap.
    Returns {condition: {F2_n: ..., F3_n: ...}}.
    """
    purity_csv = out_04c / "window_purity_table.csv"
    if not purity_csv.exists():
        purity_csv = out_04c / "purity_table.csv"
    if not purity_csv.exists():
        return {}
    pt = pd.read_csv(purity_csv)
    if "enc_overlap_frac" not in pt.columns or "condition" not in pt.columns:
        return {}
    result = {}
    for cond, grp in pt.groupby("condition"):
        result[cond] = {
            "F2_n": int((grp["enc_overlap_frac"] >= 0.50).sum()),
            "F3_n": int((grp["enc_overlap_frac"] >= 0.70).sum()),
            "total": len(grp),
        }
    return result


# ── Build comparison table ──────────────────────────────────────────────────
def build_comparison_table(
    fr_8s: pd.DataFrame,
    fr_4s: pd.DataFrame,
    out_path: Path,
) -> pd.DataFrame:
    """
    Merge 8s and 4s Friedman p-values by (filter, test, metric).
    """
    rows = []
    all_filters = FILTER_NAMES
    for fname in all_filters:
        for test in ["A", "B"]:
            for metric in ["m", "d"]:
                def get_p(fr, filt, t, met):
                    if fr.empty:
                        return np.nan
                    mask = (
                        (fr.get("filter", pd.Series()) == filt)
                        & (fr.get("test", pd.Series()) == t)
                        & (fr.get("metric", pd.Series()) == met)
                    )
                    if not any(mask):
                        return np.nan
                    return float(fr[mask]["p"].values[0])

                p8 = get_p(fr_8s, fname, test, metric)
                p4 = get_p(fr_4s, fname, test, metric)
                rows.append(
                    {
                        "filter": fname,
                        "test": test,
                        "metric": metric,
                        "window_8s_p": p8,
                        "window_4s_p": p4,
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


# ── Produce summary Markdown ─────────────────────────────────────────────────
def write_summary_md(
    comp: pd.DataFrame,
    achievability_8s: Dict,
    achievability_4s: Dict,
    out_path: Path,
    run_dir_4s: Path,
) -> None:
    today = date.today().strftime("%Y-%m-%d")

    def fmt_p(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        if v < 0.001:
            return f"{v:.2e}"
        return f"{v:.4f}"

    # Build achievability table
    conds = ["Fast", "FastDelay", "Simultaneous", "Slow"]
    ach_lines = [
        "| condition | 8s F2_n | 8s F3_n | 8s total | 4s F2_n | 4s F3_n | 4s total |",
        "|-----------|---------|---------|----------|---------|---------|----------|",
    ]
    for cond in conds:
        a8 = achievability_8s.get(cond, {})
        a4 = achievability_4s.get(cond, {})
        ach_lines.append(
            f"| {cond} | {a8.get('F2_n','?')} | {a8.get('F3_n','?')} | {a8.get('total','?')} "
            f"| {a4.get('F2_n','?')} | {a4.get('F3_n','?')} | {a4.get('total','?')} |"
        )

    # Build Friedman comparison table (Test A only, m and d)
    comp_a = comp[comp["test"] == "A"]
    fr_lines = [
        "| filter | window_8s_p_m | window_4s_p_m | window_8s_p_d | window_4s_p_d |",
        "|--------|--------------|--------------|--------------|--------------|",
    ]
    for fname in FILTER_NAMES:
        row_m = comp_a[(comp_a["filter"] == fname) & (comp_a["metric"] == "m")]
        row_d = comp_a[(comp_a["filter"] == fname) & (comp_a["metric"] == "d")]
        p8m = fmt_p(row_m["window_8s_p"].values[0] if not row_m.empty else np.nan)
        p4m = fmt_p(row_m["window_4s_p"].values[0] if not row_m.empty else np.nan)
        p8d = fmt_p(row_d["window_8s_p"].values[0] if not row_d.empty else np.nan)
        p4d = fmt_p(row_d["window_4s_p"].values[0] if not row_d.empty else np.nan)
        fr_lines.append(f"| {fname} | {p8m} | {p4m} | {p8d} | {p4d} |")

    md = f"""# Short-Window (4 s) Robustness Check — ds006848 verbal WM
**Generated:** {today}  
**8s run:** neuralmanifolddynamics_ds006848_20260626_114620  
**4s-MNPS run:** {run_dir_4s.name}  

---

## Important caveat: what "4s" means here

The `mndm summarize` pipeline reuses pre-extracted EEG feature epochs. The underlying
spectral/complexity features were extracted in **8 s epochs at 4 s step** in the original
run. The 4s config changes only the **MNPS computation window** (from 8 s to 4 s), not
the underlying feature-epoch boundaries. As a result, the window_start/window_end values
in the H5 files are **identical** between both runs (8 s windows, 4 s step).

**Consequence:** F2/F3 purity filters are not improved by this run, because the physical
EEG windows are unchanged. The F0/F1/F4 results below compare MNPS trajectories computed
with 4 s vs 8 s manifold-smoothing on the same feature data.

For genuine F2/F3 validation, the full `features` step would need to be re-run from raw
EEG with `epoch_length: 4s, step: 2s` (est. ~8-10 h compute). This is flagged as a
recommended future step for the camera-ready revision.

---

## Goal

Reviewer concern MC2 + S3.4: F2/F3 purity filters (>=50%, >=70% encoding overlap)
are geometrically impossible for Fast/Simultaneous conditions (2.8 s encoding in 8 s
windows). This analysis reruns the MNPS manifold computation with 4 s windows to check
whether the statistical pattern (F0/F1/F4) is robust to MNPS window size.

---

## F2/F3 achievability (window count per condition)

*Note: achievability is identical because the underlying feature windows are unchanged.*

{chr(10).join(ach_lines)}

---

## Friedman p-values: 8s-MNPS vs 4s-MNPS windows (Test A)

{chr(10).join(fr_lines)}

---

## Interpretation

"""
    # Auto-interpret achievability
    f2_4s = {c: achievability_4s.get(c, {}).get("F2_n", 0) for c in conds}
    if f2_4s.get("Fast", 0) > 0 and f2_4s.get("Simultaneous", 0) > 0:
        md += (
            "F2 (>=50% encoding overlap) is now achievable for Fast and Simultaneous "
            "with 4 s windows. "
        )
    else:
        md += (
            "F2 (>=50% encoding overlap) remains marginal or unachievable for "
            "Fast/Simultaneous even with 4 s windows (encoding duration ~2.8 s < 2.0 s half-window). "
        )

    # Check if pattern survives F2 in 4s run
    f2_p_m = comp_a[(comp_a["filter"] == "F2") & (comp_a["metric"] == "m")]["window_4s_p"]
    if not f2_p_m.empty and not np.isnan(f2_p_m.values[0]):
        if f2_p_m.values[0] < 0.05:
            md += (
                f"Under F2 with 4 s windows, the Friedman test for `m` remains significant "
                f"(p = {f2_p_m.values[0]:.4f}), supporting the validity of the encoding-phase finding."
            )
        else:
            md += (
                f"Under F2 with 4 s windows, the Friedman test for `m` is no longer significant "
                f"(p = {f2_p_m.values[0]:.4f}). This weakens the purity-validated claim."
            )
    else:
        md += "F2 results for 4 s windows: insufficient data to assess (see table above)."

    md += "\n\nSee `03_short_window_filter_comparison.csv` for full numeric results.\n"

    out_path.write_text(md, encoding="utf-8")
    print(f"Written: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir-4s", type=Path, required=True,
                    help="New 4s-window pipeline run directory")
    ap.add_argument("--run-dir-8s", type=Path, default=RUN_8S_DEFAULT,
                    help="Original 8s-window run directory")
    ap.add_argument("--bids-dir", type=Path, default=BIDS_DIR_DEFAULT)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR_DEFAULT)
    ap.add_argument("--skip-pipeline", action="store_true",
                    help="Skip re-running 04b/04c (use existing outputs in out-dir/04b_4s, 04c_4s)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = args.out_dir / "_tmp_4s"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── Run 04b and 04c on 4s data ─────────────────────────────────────────
    if not args.skip_pipeline:
        out_04b_4s = run_04b(args.run_dir_4s, args.bids_dir, tmp_dir)
        out_04c_4s = run_04c(args.run_dir_4s, args.bids_dir, tmp_dir)
    else:
        out_04b_4s = tmp_dir / "04b_4s"
        out_04c_4s = tmp_dir / "04c_4s"
        print("[pr03] Skipping pipeline re-run, using existing outputs.")

    # ── Load 8s results from existing NoeticDiffusion handoff ─────────────
    handoff_04c = Path(
        "J:/repos/NoeticDiffusion/data/analysis/ds006848_verbal_wm_20260626/04c_purity_audit"
    )
    fr_8s = load_04c_filter_results(handoff_04c)
    achievability_8s = assess_filter_achievability(handoff_04c)

    # ── Load 4s results ────────────────────────────────────────────────────
    fr_4s = load_04c_filter_results(out_04c_4s)
    achievability_4s = assess_filter_achievability(out_04c_4s)

    # ── Build comparison table ─────────────────────────────────────────────
    comp_csv = args.out_dir / "03_short_window_filter_comparison.csv"
    comp = build_comparison_table(fr_8s, fr_4s, comp_csv)
    print(f"Written: {comp_csv}")

    # ── Write summary MD ───────────────────────────────────────────────────
    md_path = args.out_dir / "03_short_window_robustness.md"
    write_summary_md(comp, achievability_8s, achievability_4s, md_path, args.run_dir_4s)

    print("\n[pr03] Done.")


if __name__ == "__main__":
    main()
