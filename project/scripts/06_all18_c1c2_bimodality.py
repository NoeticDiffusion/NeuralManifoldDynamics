#!/usr/bin/env python3
"""
06_all18_c1c2_bimodality.py — All-18 subject C1/C2 bimodality audit for ds003645.

Scaling from pilot (sub-002…006) to all 18 subjects. Three separable outcomes:
  1. MEG-internal validity:   does MEG alone separate face/scrambled?
  2. Cross-modal convergence: do EEG and MEG point in the same direction?
  3. Cross-modal divergence:  where do they systematically differ?

Labels are generated fresh from BIDS events.tsv in received/ds003645 (not from
the pilot epoch_condition_labels.csv), so this script is independent of prior
notebook runs.

Outputs written to SAVE_DIR (meg_eeg_comparison):
  c1_subject_event_response_agreement_all18.csv
  c2_subject_family_sign_agreement_all18.csv
  c1c2_subject_null_summary_all18.csv
  meg_readiness_score_all18.json
  bimodality_audit_all18.csv
  loo_sensitivity_all18.csv
  d_inverted_qc_all18.csv
  meg_internal_validity_all18.csv
"""

import sys, json, warnings
import h5py, numpy as np, pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# ── Configuration ───────────────────────────────────────────────────────────────
BIDS_ROOT = Path('E:/Science_Datasets/openneuro/received/ds003645')
BASE      = Path('E:/Science_Datasets/openneuro/processed/ds003645')
RUN_DIR   = sorted(BASE.glob('neuralmanifolddynamics_*'))[-1]
SAVE_DIR  = BASE / 'meg_eeg_comparison'
TASK      = 'FacePerception'
WINDOW_SEC = 8.0
N_PERM     = 500
N_PERM_INT = 300   # for MEG-internal validity (cheaper)
RNG        = np.random.default_rng(2024)

# Feature family groupings (shared-type names: strip meg_/eeg_ prefix)
M_FEATS   = ['delta', 'theta', 'alpha', 'beta', 'gamma']
D_FEATS   = ['hjorth_mobility', 'hjorth_complexity']
E_FEATS   = ['permutation_entropy', 'sample_entropy']
ALL_FEATS = M_FEATS + D_FEATS + E_FEATS

# Bimodality thresholds (science lead spec)
ALIGNED_POSITIVE_THRESH  = 0.60   # c2_overall
D_INVERTED_COS_THRESH    = -0.50  # cos_d

# Readiness gate weights
WEIGHTS = {
    'contract_pass_rate':       0.10,
    'feature_completeness':     0.15,
    'null_separation':          0.15,
    'event_response_agreement': 0.20,
    'mag_grad_stability':       0.10,
    'window_robustness':        0.15,
    'jacobian_validity':        0.15,
}

print(f'RUN_DIR : {RUN_DIR.name}')
print(f'BIDS_ROOT: {BIDS_ROOT}')


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _decode(v):
    return v.decode() if isinstance(v, bytes) else v


def cosine_sim(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 2:
        return np.nan
    a_, b_ = a[valid], b[valid]
    na, nb = np.linalg.norm(a_), np.linalg.norm(b_)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.dot(a_ / na, b_ / nb))


def load_h5(path):
    """Load H5 features (uses features_projection_z when available)."""
    with h5py.File(path, 'r') as f:
        ws = f['window_start'][:].astype(float)
        we = f['window_end'][:].astype(float)
        n  = len(ws)
        fif_mask = (f['row_source/has_meg'][:].astype(bool)
                    if 'row_source/has_meg' in f
                    else np.r_[np.zeros(n // 2, bool), np.ones(n - n // 2, bool)])
        names = [_decode(x) for x in f['features_robust_z/names'][:]]
        rz    = (f['features_projection_z/values'][:].astype(np.float64)
                 if 'features_projection_z/values' in f
                 else f['features_robust_z/values'][:].astype(np.float64))
    return ws, we, fif_mask, names, rz


def load_bids_events(sub, run):
    """Load face/scrambled events from BIDS events.tsv. Returns DataFrame or None."""
    ev_path = BIDS_ROOT / f'sub-{sub:03d}' / \
              f'sub-{sub:03d}_task-{TASK}_run-{run}_events.tsv'
    if not ev_path.exists():
        return None
    df = pd.read_csv(ev_path, sep='\t')
    stim = df[df['face_type'].notna() &
              df['event_type'].isin(['show_face', 'show_face_initial'])].copy()
    stim['condition'] = stim['face_type'].map({
        'famous_face':    'face',
        'unfamiliar_face':'face',
        'scrambled_face': 'scrambled',
    })
    return stim[['onset', 'condition', 'face_type']].reset_index(drop=True)


def assign_window_labels(events_df, window_starts, window_sec=WINDOW_SEC):
    """Assign face/scrambled/mixed/no_stim to each MNPS window."""
    labels, scr_fracs = [], []
    for ws in window_starts:
        we = ws + window_sec
        in_win = events_df[(events_df['onset'] >= ws) & (events_df['onset'] < we)]
        if len(in_win) == 0:
            labels.append('no_stim'); scr_fracs.append(np.nan)
        else:
            nf  = (in_win['condition'] == 'face').sum()
            ns  = (in_win['condition'] == 'scrambled').sum()
            tot = nf + ns
            scr_fracs.append(ns / tot if tot else np.nan)
            if ns == 0:   labels.append('face')
            elif nf == 0: labels.append('scrambled')
            else:         labels.append('mixed')
    return np.array(labels), np.array(scr_fracs)


def build_type_index(names):
    """Return (meg_types, eeg_types) dicts: feature_type -> column index."""
    meg = {nm[4:]: i for i, nm in enumerate(names) if nm.startswith('meg_')}
    eeg = {nm[4:]: i for i, nm in enumerate(names) if nm.startswith('eeg_')}
    return meg, eeg


def response_vectors(rz, meg_t, eeg_t, shared, face_m, scr_m):
    """Face-minus-scrambled response vectors for shared feature types."""
    if face_m.sum() < 2 or scr_m.sum() < 2:
        return None, None
    mv = np.array([np.nanmean(rz[face_m, meg_t[t]]) - np.nanmean(rz[scr_m, meg_t[t]])
                   for t in shared])
    ev = np.array([np.nanmean(rz[face_m, eeg_t[t]]) - np.nanmean(rz[scr_m, eeg_t[t]])
                   for t in shared])
    return mv, ev


def family_cosine(meg_v, eeg_v, shared, family):
    idx = [i for i, t in enumerate(shared) if t in family]
    return cosine_sim(meg_v[idx], eeg_v[idx]) if len(idx) >= 2 else np.nan


def family_agg(v, shared, family):
    idx = [i for i, t in enumerate(shared) if t in family]
    return float(np.nanmean(v[idx])) if idx else np.nan


def sign_agree_family(rz, face_m, scr_m, meg_t, eeg_t, family):
    agree = []
    for t in [f for f in family if f in meg_t and f in eeg_t]:
        md = np.nanmean(rz[face_m, meg_t[t]]) - np.nanmean(rz[scr_m, meg_t[t]])
        ed = np.nanmean(rz[face_m, eeg_t[t]]) - np.nanmean(rz[scr_m, eeg_t[t]])
        if np.isfinite(md) and np.isfinite(ed):
            agree.append(int(np.sign(md) == np.sign(ed)))
    return float(np.mean(agree)) if agree else np.nan


def meg_internal_validity(rz_fif, face_m, scr_m, meg_t, shared, n_perm=N_PERM_INT):
    """MEG-only face-vs-scrambled separation (label-shuffle null).
    Returns (obs_norm, null_mean, p_internal, effect_size).
    """
    meg_v_obs = np.array([np.nanmean(rz_fif[face_m, meg_t[t]])
                          - np.nanmean(rz_fif[scr_m, meg_t[t]])
                          for t in shared])
    fin = np.isfinite(meg_v_obs)
    obs_norm = float(np.linalg.norm(meg_v_obs[fin])) if fin.any() else np.nan

    all_idx = np.where(face_m | scr_m)[0]
    nf      = int(face_m.sum())
    rng     = np.random.default_rng(42)
    null_norms = []
    for _ in range(n_perm):
        p   = rng.permutation(len(all_idx))
        pf  = np.zeros(len(face_m), bool); pf[all_idx[p[:nf]]] = True
        ps  = np.zeros(len(face_m), bool); ps[all_idx[p[nf:]]] = True
        nv  = np.array([np.nanmean(rz_fif[pf, meg_t[t]]) - np.nanmean(rz_fif[ps, meg_t[t]])
                        for t in shared])
        fin2 = np.isfinite(nv)
        null_norms.append(float(np.linalg.norm(nv[fin2])) if fin2.any() else 0.0)

    if not null_norms or not np.isfinite(obs_norm):
        return obs_norm, np.nan, np.nan, np.nan
    null_arr  = np.array(null_norms)
    null_mean = float(null_arr.mean())
    p_int     = float(np.mean(null_arr >= obs_norm))
    eff       = obs_norm / (null_arr.std() + 1e-10)
    return obs_norm, null_mean, p_int, float(eff)


def classify_bimodality(cos_9d, cos_3d, cos_d, c2_overall, c2_d):
    """Classify subject into bimodality pattern (science-lead spec)."""
    c9  = float(cos_9d)    if np.isfinite(float(cos_9d) if cos_9d is not None else np.nan) else np.nan
    c3  = float(cos_3d)    if np.isfinite(float(cos_3d) if cos_3d is not None else np.nan) else np.nan
    cd  = float(cos_d)     if np.isfinite(float(cos_d)  if cos_d  is not None else np.nan) else np.nan
    c2o = float(c2_overall) if np.isfinite(float(c2_overall) if c2_overall is not None else np.nan) else np.nan
    c2d = float(c2_d)      if np.isfinite(float(c2_d)  if c2_d  is not None else np.nan) else np.nan

    if np.isfinite(c9) and c9 > 0 and np.isfinite(c3) and c3 > 0 and np.isfinite(c2o) and c2o >= ALIGNED_POSITIVE_THRESH:
        return 'aligned_positive'
    if (np.isfinite(cd) and cd < D_INVERTED_COS_THRESH) or (np.isfinite(c2d) and c2d == 0.0):
        return 'd_inverted'
    if np.isfinite(c9) and c9 < 0 and np.isfinite(c3) and c3 < 0:
        return 'globally_inverted'
    return 'mixed'


# ── Discover all subjects ───────────────────────────────────────────────────────
def _sub_int(name):
    try:
        return int(name.split('-')[1].split('_')[0])
    except (ValueError, IndexError):
        return None

all_subs = sorted(set(
    s for d in RUN_DIR.iterdir()
    if d.is_dir() and d.name.startswith('sub-') and 'meeg' in d.name and 'emptyroom' not in d.name
    for s in [_sub_int(d.name)] if s is not None
))
print(f'\nFound {len(all_subs)} subjects: {all_subs}')

# ── Main per-subject loop ───────────────────────────────────────────────────────
c1_rows   = []
c2_rows   = []
null_rows = []
int_rows  = []   # MEG-internal validity
qc_rows   = []   # per-run QC data (for d-inverted review)

for sub in all_subs:
    sub_dirs = sorted(
        d for d in RUN_DIR.iterdir()
        if d.is_dir() and d.name.startswith(f'sub-{sub:03d}_meeg_{TASK}_run-')
    )
    if not sub_dirs:
        print(f'sub-{sub:03d}: no run dirs'); continue

    rz_fif_runs   = []
    cond_fif_runs = []
    names_ref     = None
    run_data      = {}

    for d in sub_dirs:
        run_str = d.name.split('run-')[-1]
        try:   run = int(run_str)
        except ValueError: continue

        h5p = d / f'{d.name}.h5'
        if not h5p.exists(): continue

        try:
            ws, we, fif_mask, names, rz = load_h5(h5p)
        except Exception as e:
            print(f'  sub-{sub:03d} run-{run}: load error — {e}'); continue

        if names_ref is None:
            names_ref = names

        # Generate labels from BIDS events
        events_df = load_bids_events(sub, run)
        if events_df is None:
            print(f'  sub-{sub:03d} run-{run}: no events.tsv'); continue

        ws_fif = ws[fif_mask]
        conds_fif, scr_frac = assign_window_labels(events_df, ws_fif)
        rz_fif = rz[fif_mask]

        nf_run = int((conds_fif == 'face').sum())
        ns_run = int((conds_fif == 'scrambled').sum())

        qc_rows.append({
            'subject': sub, 'run': run,
            'n_fif_windows': len(ws_fif),
            'n_face': nf_run, 'n_scrambled': ns_run,
            'n_mixed': int((conds_fif == 'mixed').sum()),
            'n_no_stim': int((conds_fif == 'no_stim').sum()),
        })

        rz_fif_runs.append(rz_fif)
        cond_fif_runs.append(conds_fif)
        run_data[run] = (rz_fif, conds_fif)

    if not rz_fif_runs or names_ref is None:
        print(f'sub-{sub:03d}: no usable H5 data'); continue

    rz_all   = np.vstack(rz_fif_runs)
    cond_all = np.concatenate(cond_fif_runs)

    face_mask = cond_all == 'face'
    scr_mask  = cond_all == 'scrambled'
    nf, ns    = int(face_mask.sum()), int(scr_mask.sum())

    print(f'\nsub-{sub:03d}: {len(run_data)} runs  face={nf}  scr={ns}')

    if nf < 10 or ns < 10:
        print(f'  → skipping (insufficient events)'); continue

    # Feature type index
    meg_types, eeg_types = build_type_index(names_ref)
    shared_types = sorted(set(meg_types) & set(eeg_types) & set(ALL_FEATS))
    if not shared_types:
        print(f'  → skipping (no shared feature types)'); continue

    # ── C1: observed cross-modal cosines ────────────────────────────────────────
    meg_v, eeg_v = response_vectors(rz_all, meg_types, eeg_types, shared_types,
                                    face_mask, scr_mask)

    cos_9d = cosine_sim(meg_v, eeg_v)
    meg_3d = np.array([family_agg(meg_v, shared_types, M_FEATS),
                       family_agg(meg_v, shared_types, D_FEATS),
                       family_agg(meg_v, shared_types, E_FEATS)])
    eeg_3d = np.array([family_agg(eeg_v, shared_types, M_FEATS),
                       family_agg(eeg_v, shared_types, D_FEATS),
                       family_agg(eeg_v, shared_types, E_FEATS)])
    cos_3d = cosine_sim(meg_3d, eeg_3d)
    cos_m  = family_cosine(meg_v, eeg_v, shared_types, M_FEATS)
    cos_d  = family_cosine(meg_v, eeg_v, shared_types, D_FEATS)
    cos_e  = family_cosine(meg_v, eeg_v, shared_types, E_FEATS)

    print(f'  cos_9d={cos_9d:.3f}  cos_3d={cos_3d:.3f}'
          f'  cos_m={cos_m:.3f}  cos_d={cos_d:.3f}  cos_e={cos_e:.3f}')

    # ── C2: family sign agreement ────────────────────────────────────────────────
    c2_m   = sign_agree_family(rz_all, face_mask, scr_mask, meg_types, eeg_types, M_FEATS)
    c2_d   = sign_agree_family(rz_all, face_mask, scr_mask, meg_types, eeg_types, D_FEATS)
    c2_e   = sign_agree_family(rz_all, face_mask, scr_mask, meg_types, eeg_types, E_FEATS)
    c2_all = sign_agree_family(rz_all, face_mask, scr_mask, meg_types, eeg_types, ALL_FEATS)

    print(f'  c2_m={c2_m:.3f}  c2_d={c2_d:.3f}  c2_e={c2_e:.3f}  c2_overall={c2_all:.3f}')

    # ── C1 null: label shuffle ───────────────────────────────────────────────────
    all_idx   = np.where(face_mask | scr_mask)[0]
    null_cos  = []
    for _ in range(N_PERM):
        p  = RNG.permutation(len(all_idx))
        pf = np.zeros(len(cond_all), bool); pf[all_idx[p[:nf]]] = True
        ps = np.zeros(len(cond_all), bool); ps[all_idx[p[nf:]]] = True
        mv, ev = response_vectors(rz_all, meg_types, eeg_types, shared_types, pf, ps)
        if mv is not None:
            null_cos.append(cosine_sim(mv, ev))
    null_fin  = [x for x in null_cos if np.isfinite(x)]
    null_mean = float(np.nanmean(null_fin)) if null_fin else np.nan
    null_sd   = float(np.nanstd(null_fin))  if null_fin else np.nan
    p_shuffle = (float(np.mean(np.array(null_fin) >= cos_9d))
                 if null_fin and np.isfinite(cos_9d) else np.nan)

    # ── C1 null: wrong-run pairing ───────────────────────────────────────────────
    wrong_run_cos = []
    run_list = sorted(run_data.keys())
    for ra in run_list:
        rz_a, ca = run_data[ra]; fa = ca=='face'; sa = ca=='scrambled'
        if fa.sum() < 5 or sa.sum() < 5: continue
        for rb in run_list:
            if rb == ra: continue
            rz_b, cb = run_data[rb]; fb = cb=='face'; sb = cb=='scrambled'
            if fb.sum() < 5 or sb.sum() < 5: continue
            mv_a = np.array([np.nanmean(rz_a[fa, meg_types[t]]) - np.nanmean(rz_a[sa, meg_types[t]])
                             for t in shared_types])
            ev_b = np.array([np.nanmean(rz_b[fb, eeg_types[t]]) - np.nanmean(rz_b[sb, eeg_types[t]])
                             for t in shared_types])
            wrong_run_cos.append(cosine_sim(mv_a, ev_b))
    wr_fin         = [x for x in wrong_run_cos if np.isfinite(x)]
    wrong_run_mean = float(np.nanmean(wr_fin)) if wr_fin else np.nan

    obs_gt_null = (bool(cos_9d > null_mean)
                   if np.isfinite(cos_9d) and np.isfinite(null_mean) else None)
    obs_gt_wr   = (bool(cos_9d > wrong_run_mean)
                   if np.isfinite(cos_9d) and np.isfinite(wrong_run_mean) else None)
    # science-lead field: True iff both hold OR specifically obs > null
    wrongrun_gt_true_or_true_gt_wrong = (
        f'obs_gt_null={obs_gt_null},obs_gt_wr={obs_gt_wr}'
        if obs_gt_null is not None else 'unknown')

    print(f'  null_mean={null_mean:.3f}  p_shuffle={p_shuffle:.3f}'
          f'  wr_mean={wrong_run_mean:.3f}  obs_gt_null={obs_gt_null}')

    # ── MEG-internal validity ────────────────────────────────────────────────────
    obs_norm, int_null_mean, p_internal, eff = meg_internal_validity(
        rz_all, face_mask, scr_mask, meg_types, shared_types)
    print(f'  meg_internal: norm={obs_norm:.3f}  null_mean={int_null_mean:.3f}'
          f'  p_int={p_internal:.3f}  eff={eff:.2f}')

    int_rows.append({
        'subject':          sub,
        'meg_contrast_norm':round(obs_norm, 4)       if np.isfinite(obs_norm) else None,
        'null_norm_mean':   round(int_null_mean, 4)  if np.isfinite(int_null_mean) else None,
        'p_internal':       round(p_internal, 4)     if np.isfinite(p_internal) else None,
        'effect_size':      round(eff, 4)            if np.isfinite(eff) else None,
        'meg_valid':        bool(np.isfinite(p_internal) and p_internal < 0.10),
    })

    # ── Store rows ───────────────────────────────────────────────────────────────
    def _r(x): return round(float(x), 4) if (x is not None and np.isfinite(float(x) if x is not None else np.nan)) else None

    c1_rows.append({
        'subject':     sub,
        'n_runs':      len(run_data),
        'n_face':      nf,
        'n_scrambled': ns,
        'cos_9d':      _r(cos_9d),
        'cos_3d':      _r(cos_3d),
        'cos_m':       _r(cos_m),
        'cos_d':       _r(cos_d),
        'cos_e':       _r(cos_e),
        'null_mean':   _r(null_mean),
        'p_shuffle':   _r(p_shuffle),
        'obs_gt_null': obs_gt_null,
        'wrongrun_gt_true_or_true_gt_wrong': wrongrun_gt_true_or_true_gt_wrong,
    })

    c2_rows.append({
        'subject':     sub,
        'n_face':      nf,
        'n_scrambled': ns,
        'c2_m':        _r(c2_m),
        'c2_d':        _r(c2_d),
        'c2_e':        _r(c2_e),
        'c2_overall':  _r(c2_all),
    })

    null_rows.append({
        'subject':          sub,
        'obs_cosine_9d':    _r(cos_9d),
        'null_mean_shuffle':_r(null_mean),
        'null_sd_shuffle':  _r(null_sd),
        'p_label_shuffle':  _r(p_shuffle),
        'wrong_run_null_mean': _r(wrong_run_mean),
        'obs_gt_label_null': obs_gt_null,
        'obs_gt_wr_null':    obs_gt_wr,
    })


# ── Bimodality classification ───────────────────────────────────────────────────
c1_df   = pd.DataFrame(c1_rows)
c2_df   = pd.DataFrame(c2_rows)
null_df = pd.DataFrame(null_rows)
int_df  = pd.DataFrame(int_rows)
qc_df   = pd.DataFrame(qc_rows)

bio_rows = []
for _, r1 in c1_df.iterrows():
    sub = r1['subject']
    r2  = c2_df[c2_df['subject'] == sub].iloc[0] if (c2_df['subject'] == sub).any() else {}
    c2o = r2.get('c2_overall', np.nan) if isinstance(r2, pd.Series) else np.nan
    c2d = r2.get('c2_d', np.nan)       if isinstance(r2, pd.Series) else np.nan
    bio = classify_bimodality(r1['cos_9d'], r1['cos_3d'], r1['cos_d'], c2o, c2d)
    bio_rows.append({'subject': sub, 'bimodality_class': bio,
                     'cos_9d': r1['cos_9d'], 'cos_3d': r1['cos_3d'],
                     'cos_d':  r1['cos_d'],  'c2_overall': c2o, 'c2_d': c2d})
bio_df = pd.DataFrame(bio_rows)

print('\n=== Bimodality Classification ===')
print(bio_df[['subject', 'bimodality_class', 'cos_9d', 'cos_3d', 'cos_d',
              'c2_overall', 'c2_d']].to_string(index=False))
bio_counts = bio_df['bimodality_class'].value_counts()
print(f'\n  n_aligned_positive: {bio_counts.get("aligned_positive", 0)}')
print(f'  n_d_inverted:       {bio_counts.get("d_inverted", 0)}')
print(f'  n_globally_inverted:{bio_counts.get("globally_inverted", 0)}')
print(f'  n_mixed:            {bio_counts.get("mixed", 0)}')


# ── LOO sensitivity analysis ────────────────────────────────────────────────────
print('\n=== LOO Sensitivity ===')

def loo_readiness(c1_df, c2_df, rs_base, leave_out):
    keep = c1_df['subject'] != leave_out
    keep2 = c2_df['subject'] != leave_out
    if keep.sum() == 0: return None
    frac_gt = c1_df.loc[keep, 'obs_gt_null'].apply(lambda x: 1 if x is True else 0).mean()
    mean_c2 = c2_df.loc[keep2, 'c2_overall'].mean()
    # Rebuild score
    rs_copy = {k: v for k, v in rs_base['scores'].items()}
    rs_copy['event_response_agreement'] = round(float(frac_gt), 4)
    ws_total = sum(WEIGHTS.get(k, 0) * float(v or 0) for k, v in rs_copy.items() if k in WEIGHTS)
    return {
        'loo_subject':              leave_out,
        'readiness_without_subject':round(ws_total, 4),
        'mean_cos_9d_without_subject': round(float(c1_df.loc[keep, 'cos_9d'].mean()), 4),
        'mean_cos_3d_without_subject': round(float(c1_df.loc[keep, 'cos_3d'].mean()), 4),
        'mean_cos_d_without_subject':  round(float(c1_df.loc[keep, 'cos_d'].mean()), 4),
        'mean_c2_overall_without_subject': round(float(mean_c2), 4),
    }

# Load base readiness for LOO computation
base_rs_path = SAVE_DIR / 'meg_readiness_score.json'
rs_base = json.loads(base_rs_path.read_text()) if base_rs_path.exists() else {'scores': {}}

loo_rows = []
for sub in c1_df['subject'].tolist():
    row = loo_readiness(c1_df, c2_df, rs_base, sub)
    if row: loo_rows.append(row)
loo_df = pd.DataFrame(loo_rows) if loo_rows else pd.DataFrame()

if not loo_df.empty:
    print(loo_df.to_string(index=False))
    loo_min  = loo_df['readiness_without_subject'].min()
    loo_max  = loo_df['readiness_without_subject'].max()
    loo_mean = loo_df['readiness_without_subject'].mean()
    print(f'\n  LOO readiness range: {loo_min:.4f} – {loo_max:.4f} (mean {loo_mean:.4f})')


# ── d-inverted QC review ────────────────────────────────────────────────────────
d_inv_subs = bio_df.loc[bio_df['bimodality_class'].isin(['d_inverted', 'globally_inverted']),
                         'subject'].tolist()
print(f'\n=== d-inverted / globally-inverted subjects: {d_inv_subs} ===')

dinv_rows = []
for sub in d_inv_subs:
    sub_qc = qc_df[qc_df['subject'] == sub]
    for _, run_row in sub_qc.iterrows():
        run = run_row['run']
        # Per-run d-family check: load each run H5
        run_dir_name = f'sub-{sub:03d}_meeg_{TASK}_run-{run}'
        h5p = RUN_DIR / run_dir_name / f'{run_dir_name}.h5'
        d_family_check = {}
        if h5p.exists():
            try:
                ws_r, we_r, fif_m, nm, rz_r = load_h5(h5p)
                events_r = load_bids_events(sub, run)
                if events_r is not None:
                    ws_fif_r = ws_r[fif_m]
                    conds_r, _ = assign_window_labels(events_r, ws_fif_r)
                    rz_fif_r   = rz_r[fif_m]
                    fm_r = conds_r == 'face'; sm_r = conds_r == 'scrambled'
                    if fm_r.sum() >= 3 and sm_r.sum() >= 3:
                        meg_t, eeg_t = build_type_index(nm)
                        shared_r = sorted(set(meg_t) & set(eeg_t) & set(ALL_FEATS))
                        if shared_r:
                            for ft in D_FEATS:
                                if ft in meg_t and ft in eeg_t:
                                    md = np.nanmean(rz_fif_r[fm_r, meg_t[ft]]) - np.nanmean(rz_fif_r[sm_r, meg_t[ft]])
                                    ed = np.nanmean(rz_fif_r[fm_r, eeg_t[ft]]) - np.nanmean(rz_fif_r[sm_r, eeg_t[ft]])
                                    d_family_check[f'meg_{ft}_diff'] = round(float(md), 4) if np.isfinite(md) else None
                                    d_family_check[f'eeg_{ft}_diff'] = round(float(ed), 4) if np.isfinite(ed) else None
                                    d_family_check[f'{ft}_sign_agree'] = int(np.sign(md) == np.sign(ed)) if (np.isfinite(md) and np.isfinite(ed)) else None
            except Exception as e:
                d_family_check['error'] = str(e)[:80]

        row = {
            'subject': sub,
            'run': run,
            'bimodality_class': bio_df.loc[bio_df['subject']==sub, 'bimodality_class'].values[0],
            'n_face': int(run_row['n_face']),
            'n_scrambled': int(run_row['n_scrambled']),
        }
        row.update(d_family_check)
        dinv_rows.append(row)
        print(f'  sub-{sub:03d} run-{run}: face={row["n_face"]} scr={row["n_scrambled"]} '
              + ' '.join(f'{k}={v}' for k, v in d_family_check.items()))

dinv_df = pd.DataFrame(dinv_rows) if dinv_rows else pd.DataFrame()


# ── Print summaries ─────────────────────────────────────────────────────────────
print('\n=== C1 Subject-level Event-Response Agreement (all 18) ===')
print(c1_df.to_string(index=False))

print('\n=== C2 Family Sign Agreement (all 18) ===')
print(c2_df.to_string(index=False))

print('\n=== MEG-internal Validity ===')
print(int_df.to_string(index=False))

# ── Update readiness score (all-18) ────────────────────────────────────────────
frac_c1_gt_null = (float((c1_df['obs_gt_null'] == True).mean())
                   if len(c1_df) else np.nan)
mean_c2_sign    = (float(c2_df['c2_overall'].mean())
                   if 'c2_overall' in c2_df.columns and len(c2_df) else np.nan)

rs_all18 = json.loads(base_rs_path.read_text()) if base_rs_path.exists() else {'scores': {}}
rs_all18['scores']['event_response_agreement'] = (
    round(frac_c1_gt_null, 4) if np.isfinite(frac_c1_gt_null) else None)
rs_all18['scores']['event_response_agreement_subject_level_all18'] = (
    round(frac_c1_gt_null, 4) if np.isfinite(frac_c1_gt_null) else None)
rs_all18['scores']['c2_family_sign_agreement_all18'] = (
    round(mean_c2_sign, 4) if np.isfinite(mean_c2_sign) else None)

ws_total = sum(WEIGHTS.get(k, 0) * float(rs_all18['scores'].get(k) or 0)
               for k in WEIGHTS if rs_all18['scores'].get(k) is not None)
rs_all18['weighted_score'] = round(ws_total, 4)


def interpret(s):
    if s >= 0.80: return 'READY for production-scale ds003645 MEG validation'
    if s >= 0.75: return 'Usable pilot MEG mapping; proceed, cross-modal convergence moderate'
    if s >= 0.70: return 'MEG ingest valid, cross-modal convergence provisional'
    return 'DO NOT SCALE — resolve d-family inversion first'


rs_all18['interpretation'] = interpret(ws_total)
rs_all18['n_subjects_analysed'] = len(c1_df)
rs_all18['bimodality_counts'] = bio_counts.to_dict()
rs_all18.setdefault('notes', {})['all18_c1_frac_gt_null'] = (
    f'{frac_c1_gt_null:.3f} subjects obs>null (label-shuffle)')
rs_all18.setdefault('notes', {})['all18_c2_mean_sign'] = (
    f'{mean_c2_sign:.3f} mean family sign agreement')
rs_all18['gate_7_all18_bimodality_audit'] = 'COMPLETE'

all18_rs_path = SAVE_DIR / 'meg_readiness_score_all18.json'
with open(all18_rs_path, 'w') as f:
    json.dump(rs_all18, f, indent=2)

print('\n=== Readiness Score (all 18 subjects) ===')
for k, v in rs_all18['scores'].items():
    print(f'  {k:<50}: {v}')
print(f'\n  WEIGHTED TOTAL : {rs_all18["weighted_score"]:.4f}')
print(f'  INTERPRETATION : {rs_all18["interpretation"]}')
print(f'  Bimodality     : {bio_counts.to_dict()}')


# ── Save all outputs ────────────────────────────────────────────────────────────
c1_df.to_csv(SAVE_DIR / 'c1_subject_event_response_agreement_all18.csv',  index=False)
c2_df.to_csv(SAVE_DIR / 'c2_subject_family_sign_agreement_all18.csv',     index=False)
null_df.to_csv(SAVE_DIR / 'c1c2_subject_null_summary_all18.csv',          index=False)
bio_df.to_csv(SAVE_DIR / 'bimodality_audit_all18.csv',                    index=False)
int_df.to_csv(SAVE_DIR / 'meg_internal_validity_all18.csv',               index=False)
if not loo_df.empty:
    loo_df.to_csv(SAVE_DIR / 'loo_sensitivity_all18.csv', index=False)
if not dinv_df.empty:
    dinv_df.to_csv(SAVE_DIR / 'd_inverted_qc_all18.csv', index=False)

print('\n=== Saved Files ===')
for fname in [
    'c1_subject_event_response_agreement_all18.csv',
    'c2_subject_family_sign_agreement_all18.csv',
    'c1c2_subject_null_summary_all18.csv',
    'bimodality_audit_all18.csv',
    'loo_sensitivity_all18.csv',
    'd_inverted_qc_all18.csv',
    'meg_internal_validity_all18.csv',
    'meg_readiness_score_all18.json',
]:
    p = SAVE_DIR / fname
    print(f'  {"OK" if p.exists() else "MISSING"} {fname}')
