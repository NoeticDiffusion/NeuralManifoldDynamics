#!/usr/bin/env python3
"""
05_subject_level_c1c2.py — Subject-level C1/C2 aggregation for ds003645 MEG validation.

For each subject: pool all run-level face/scrambled FIF windows, compute
MEG-vs-EEG response-vector cosines (3D family-aggregated, 9D shared-feature,
per-family), run label-shuffle and wrong-run null controls, then update the
readiness score.

Outputs (all written to SAVE_DIR):
  c1_subject_event_response_agreement.csv
  c2_subject_family_sign_agreement.csv
  c1c2_subject_null_summary.csv
  meg_readiness_score.json  (updated)
"""

import sys, json, warnings
import h5py, numpy as np, pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────────────
BASE     = Path('E:/Science_Datasets/openneuro/processed/ds003645')
RUN_DIR  = sorted(BASE.glob('neuralmanifolddynamics_*'))[-1]
SAVE_DIR = BASE / 'meg_eeg_comparison'
TASK     = 'FacePerception'
N_PERM   = 500
RNG      = np.random.default_rng(2024)

# Feature family groupings (shared-type names, i.e. strip meg_/eeg_ prefix)
M_FEATS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
D_FEATS = ['hjorth_mobility', 'hjorth_complexity']
E_FEATS = ['permutation_entropy', 'sample_entropy']
ALL_FEATS = M_FEATS + D_FEATS + E_FEATS

# ── Helpers ────────────────────────────────────────────────────────────────────

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
    """Return (ws, fif_mask, names, rz_values).

    Uses features_projection_z when available (transform-aware: log10 applied
    to MEG spectral features before robust-z, fixing the physical-unit collapse
    in features_robust_z for values around 1e-25 W).
    Falls back to features_robust_z if projection_z is absent.
    """
    with h5py.File(path, 'r') as f:
        ws = f['window_start'][:].astype(float)
        n  = len(ws)
        if 'row_source/has_meg' in f:
            fif_mask = f['row_source/has_meg'][:].astype(bool)
        else:
            fif_mask = np.zeros(n, bool)
            fif_mask[n // 2:] = True
        names = [_decode(x) for x in f['features_robust_z/names'][:]]
        if 'features_projection_z/values' in f:
            rz = f['features_projection_z/values'][:].astype(np.float64)
        else:
            rz = f['features_robust_z/values'][:].astype(np.float64)
    return ws, fif_mask, names, rz


def assign_labels(labels_df, sub, run, ws):
    """Assign 'face'/'scrambled'/'mixed'/'no_stim'/'unknown' to each window."""
    lsub = (labels_df[(labels_df['sub'] == int(sub)) & (labels_df['run'] == int(run))]
            .sort_values('window_start'))
    if len(lsub) == 0:
        return np.full(len(ws), 'unknown')
    ws8   = lsub['window_start'].values
    conds = lsub['condition'].values
    out = []
    for w in ws:
        idx = int(np.searchsorted(ws8, w, side='right')) - 1
        out.append(conds[idx] if 0 <= idx < len(conds) else 'unknown')
    return np.array(out)


def build_type_index(names):
    """Return (meg_types, eeg_types) dicts: feature_type -> column index."""
    meg = {nm[4:]: i for i, nm in enumerate(names) if nm.startswith('meg_')}
    eeg = {nm[4:]: i for i, nm in enumerate(names) if nm.startswith('eeg_')}
    return meg, eeg


def response_vectors(rz, meg_types, eeg_types, shared_types, face_mask, scr_mask):
    """Face-minus-scrambled response vectors for MEG and EEG on shared feature types."""
    if face_mask.sum() < 2 or scr_mask.sum() < 2:
        return None, None
    meg_v = np.array([np.nanmean(rz[face_mask, meg_types[t]])
                      - np.nanmean(rz[scr_mask,  meg_types[t]]) for t in shared_types])
    eeg_v = np.array([np.nanmean(rz[face_mask, eeg_types[t]])
                      - np.nanmean(rz[scr_mask,  eeg_types[t]]) for t in shared_types])
    return meg_v, eeg_v


def family_cosine(meg_v, eeg_v, shared_types, family_feats):
    idx = [i for i, t in enumerate(shared_types) if t in family_feats]
    if len(idx) < 2:
        return np.nan
    return cosine_sim(meg_v[idx], eeg_v[idx])


def family_agg(v, shared_types, family_feats):
    idx = [i for i, t in enumerate(shared_types) if t in family_feats]
    if not idx:
        return np.nan
    return float(np.nanmean(v[idx]))


# ── Load labels ────────────────────────────────────────────────────────────────
labels_df = pd.read_csv(SAVE_DIR / 'epoch_condition_labels.csv')
pilot_subs = sorted(labels_df['sub'].unique().tolist())
print(f'Pilot subjects with labels: {pilot_subs}')
print(f'RUN_DIR: {RUN_DIR.name}')

# ── Main loop: pool per subject ────────────────────────────────────────────────
c1_rows   = []
c2_rows   = []
null_rows = []

for sub in pilot_subs:
    sub_dirs = sorted(
        d for d in RUN_DIR.iterdir()
        if d.is_dir() and d.name.startswith(f'sub-{sub:03d}_meeg_{TASK}_run-')
    )
    if not sub_dirs:
        print(f'sub-{sub:03d}: no run directories found')
        continue

    rz_fif_runs  = []
    cond_fif_runs = []
    names_ref    = None
    run_data     = {}   # run -> (rz_fif, cond_fif) for wrong-run null

    for d in sub_dirs:
        run_str = d.name.split('run-')[-1]
        try:
            run = int(run_str)
        except ValueError:
            continue
        h5p = d / f'{d.name}.h5'
        if not h5p.exists():
            continue
        try:
            ws, fif_mask, names, rz = load_h5(h5p)
        except Exception as e:
            print(f'  sub-{sub:03d} run-{run}: load error — {e}')
            continue

        if names_ref is None:
            names_ref = names

        conds_fif = assign_labels(labels_df, sub, run, ws[fif_mask])
        rz_fif = rz[fif_mask]

        rz_fif_runs.append(rz_fif)
        cond_fif_runs.append(conds_fif)
        run_data[run] = (rz_fif, conds_fif)

    if not rz_fif_runs or names_ref is None:
        print(f'sub-{sub:03d}: no usable H5 data')
        continue

    # Pool all FIF rows across runs
    rz_all   = np.vstack(rz_fif_runs)
    cond_all = np.concatenate(cond_fif_runs)

    face_mask = cond_all == 'face'
    scr_mask  = cond_all == 'scrambled'
    nf, ns    = int(face_mask.sum()), int(scr_mask.sum())

    print(f'sub-{sub:03d}: {len(run_data)} runs  |  face={nf}  scr={ns}')

    if nf < 10 or ns < 10:
        print(f'  → skipping: insufficient events')
        continue

    # Feature type indices
    meg_types, eeg_types = build_type_index(names_ref)
    shared_types = sorted(set(meg_types) & set(eeg_types) & set(ALL_FEATS))
    if not shared_types:
        print(f'  → skipping: no shared feature types')
        continue

    # ── C1: observed cosines ───────────────────────────────────────────────────
    meg_v, eeg_v = response_vectors(rz_all, meg_types, eeg_types, shared_types, face_mask, scr_mask)

    cos_9d = cosine_sim(meg_v, eeg_v)

    # 3D: project to (m_agg, d_agg, e_agg) per modality
    meg_3d = np.array([family_agg(meg_v, shared_types, M_FEATS),
                       family_agg(meg_v, shared_types, D_FEATS),
                       family_agg(meg_v, shared_types, E_FEATS)])
    eeg_3d = np.array([family_agg(eeg_v, shared_types, M_FEATS),
                       family_agg(eeg_v, shared_types, D_FEATS),
                       family_agg(eeg_v, shared_types, E_FEATS)])
    cos_3d = cosine_sim(meg_3d, eeg_3d)

    cos_m = family_cosine(meg_v, eeg_v, shared_types, M_FEATS)
    cos_d = family_cosine(meg_v, eeg_v, shared_types, D_FEATS)
    cos_e = family_cosine(meg_v, eeg_v, shared_types, E_FEATS)

    print(f'  cos_9d={cos_9d:.4f}  cos_3d={cos_3d:.4f}'
          f'  cos_m={cos_m:.4f}  cos_d={cos_d:.4f}  cos_e={cos_e:.4f}')

    # ── C1 null: label shuffle ─────────────────────────────────────────────────
    all_idx   = np.where(face_mask | scr_mask)[0]
    null_cos  = []
    for _ in range(N_PERM):
        p    = RNG.permutation(len(all_idx))
        pf   = np.zeros(len(cond_all), bool); pf[all_idx[p[:nf]]] = True
        ps   = np.zeros(len(cond_all), bool); ps[all_idx[p[nf:]]] = True
        mv, ev = response_vectors(rz_all, meg_types, eeg_types, shared_types, pf, ps)
        if mv is not None:
            null_cos.append(cosine_sim(mv, ev))
    null_cos_fin = [x for x in null_cos if np.isfinite(x)]
    null_mean = float(np.nanmean(null_cos_fin)) if null_cos_fin else np.nan
    null_sd   = float(np.nanstd(null_cos_fin))  if null_cos_fin else np.nan
    p_shuffle = (float(np.mean(np.array(null_cos_fin) >= cos_9d))
                 if null_cos_fin and np.isfinite(cos_9d) else np.nan)
    print(f'  null_mean={null_mean:.4f}  p_shuffle={p_shuffle:.4f}')

    # ── C1 null: wrong-run pairing ─────────────────────────────────────────────
    wrong_run_cos = []
    run_list = sorted(run_data.keys())
    for ra in run_list:
        rz_a, cond_a = run_data[ra]
        fa = cond_a == 'face'; sa = cond_a == 'scrambled'
        if fa.sum() < 5 or sa.sum() < 5:
            continue
        for rb in run_list:
            if rb == ra:
                continue
            rz_b, cond_b = run_data[rb]
            fb = cond_b == 'face'; sb = cond_b == 'scrambled'
            if fb.sum() < 5 or sb.sum() < 5:
                continue
            # MEG response from run A, EEG response from run B
            mv_a = np.array([np.nanmean(rz_a[fa, meg_types[t]]) - np.nanmean(rz_a[sa, meg_types[t]])
                             for t in shared_types])
            ev_b = np.array([np.nanmean(rz_b[fb, eeg_types[t]]) - np.nanmean(rz_b[sb, eeg_types[t]])
                             for t in shared_types])
            wrong_run_cos.append(cosine_sim(mv_a, ev_b))
    wr_fin = [x for x in wrong_run_cos if np.isfinite(x)]
    wrong_run_mean = float(np.nanmean(wr_fin)) if wr_fin else np.nan

    # obs > wrong-run null?
    obs_gt_null    = bool(cos_9d > null_mean)         if np.isfinite(cos_9d) and np.isfinite(null_mean) else None
    obs_gt_wr_null = bool(cos_9d > wrong_run_mean)    if np.isfinite(cos_9d) and np.isfinite(wrong_run_mean) else None

    c1_rows.append({
        'subject':              sub,
        'window_sec':           8,
        'n_runs':               len(run_data),
        'n_face':               nf,
        'n_scrambled':          ns,
        'cosine_9d':            round(float(cos_9d), 4) if np.isfinite(cos_9d) else None,
        'cosine_3d':            round(float(cos_3d), 4) if np.isfinite(cos_3d) else None,
        'cosine_m_family':      round(float(cos_m), 4) if np.isfinite(cos_m) else None,
        'cosine_d_family':      round(float(cos_d), 4) if np.isfinite(cos_d) else None,
        'cosine_e_family':      round(float(cos_e), 4) if np.isfinite(cos_e) else None,
        'null_mean':            round(null_mean, 4)     if np.isfinite(null_mean) else None,
        'null_sd':              round(null_sd, 4)       if np.isfinite(null_sd) else None,
        'p_value':              round(p_shuffle, 4)     if np.isfinite(p_shuffle) else None,
        'obs_gt_null':          obs_gt_null,
        'wrong_run_null_mean':  round(wrong_run_mean, 4) if np.isfinite(wrong_run_mean) else None,
        'obs_gt_wr_null':       obs_gt_wr_null,
    })

    # ── C2: family sign agreement ──────────────────────────────────────────────
    def sign_agree_family(family_feats):
        agree = []
        for t in [f for f in family_feats if f in meg_types and f in eeg_types]:
            meg_diff = (np.nanmean(rz_all[face_mask, meg_types[t]])
                        - np.nanmean(rz_all[scr_mask,  meg_types[t]]))
            eeg_diff = (np.nanmean(rz_all[face_mask, eeg_types[t]])
                        - np.nanmean(rz_all[scr_mask,  eeg_types[t]]))
            if np.isfinite(meg_diff) and np.isfinite(eeg_diff):
                agree.append(int(np.sign(meg_diff) == np.sign(eeg_diff)))
        return float(np.mean(agree)) if agree else np.nan

    c2_rows.append({
        'subject':              sub,
        'window_sec':           8,
        'n_face':               nf,
        'n_scrambled':          ns,
        'sign_agree_m_family':  sign_agree_family(M_FEATS),
        'sign_agree_d_family':  sign_agree_family(D_FEATS),
        'sign_agree_e_family':  sign_agree_family(E_FEATS),
        'sign_agree_overall':   sign_agree_family(ALL_FEATS),
    })

    null_rows.append({
        'subject':              sub,
        'window_sec':           8,
        'obs_cosine_9d':        cos_9d,
        'null_mean_shuffle':    null_mean,
        'null_sd_shuffle':      null_sd,
        'p_label_shuffle':      p_shuffle,
        'wrong_run_null_mean':  wrong_run_mean,
        'obs_gt_label_null':    obs_gt_null,
        'obs_gt_wr_null':       obs_gt_wr_null,
    })

# ── Save CSVs ──────────────────────────────────────────────────────────────────
c1_df   = pd.DataFrame(c1_rows)
c2_df   = pd.DataFrame(c2_rows)
null_df = pd.DataFrame(null_rows)

c1_df.to_csv(SAVE_DIR / 'c1_subject_event_response_agreement.csv',  index=False)
c2_df.to_csv(SAVE_DIR / 'c2_subject_family_sign_agreement.csv',     index=False)
null_df.to_csv(SAVE_DIR / 'c1c2_subject_null_summary.csv',          index=False)

print('\n=== C1 Subject-level Event-Response Agreement ===')
show_cols = ['subject', 'n_runs', 'n_face', 'n_scrambled',
             'cosine_9d', 'cosine_3d', 'cosine_m_family', 'cosine_d_family',
             'cosine_e_family', 'null_mean', 'p_value', 'obs_gt_null']
print(c1_df[[c for c in show_cols if c in c1_df.columns]].to_string(index=False))

print('\n=== C2 Family Sign Agreement ===')
print(c2_df[['subject', 'sign_agree_m_family', 'sign_agree_d_family',
             'sign_agree_e_family', 'sign_agree_overall']].to_string(index=False))

print('\n=== Null summary (C1 label-shuffle + wrong-run) ===')
print(null_df[['subject', 'obs_cosine_9d', 'null_mean_shuffle',
               'p_label_shuffle', 'wrong_run_null_mean',
               'obs_gt_label_null', 'obs_gt_wr_null']].to_string(index=False))

# ── Update readiness score ─────────────────────────────────────────────────────
frac_c1_gt_null = (float((c1_df['obs_gt_null'] == True).mean())
                   if 'obs_gt_null' in c1_df.columns and len(c1_df) else np.nan)
mean_c2_sign    = (float(c2_df['sign_agree_overall'].mean())
                   if 'sign_agree_overall' in c2_df.columns and len(c2_df) else np.nan)

rs_path = SAVE_DIR / 'meg_readiness_score.json'
with open(rs_path) as f:
    rs = json.load(f)

rs['scores']['event_response_agreement_subject_level'] = (
    round(frac_c1_gt_null, 4) if np.isfinite(frac_c1_gt_null) else None)
rs['scores']['c2_family_sign_agreement_subject_level'] = (
    round(mean_c2_sign, 4) if np.isfinite(mean_c2_sign) else None)

# Use subject-level C1 as the primary event_response_agreement gate
if np.isfinite(frac_c1_gt_null):
    rs['scores']['event_response_agreement'] = round(frac_c1_gt_null, 4)

# Recompute weighted total
WEIGHTS = {
    'contract_pass_rate':       0.10,
    'feature_completeness':     0.15,
    'null_separation':          0.15,
    'event_response_agreement': 0.20,
    'mag_grad_stability':       0.10,
    'window_robustness':        0.15,
    'jacobian_validity':        0.15,
}
ws_total = sum(WEIGHTS[k] * rs['scores'].get(k, 0.0)
               for k in WEIGHTS if rs['scores'].get(k) is not None)
rs['weighted_score'] = round(ws_total, 4)


def interpret(s):
    if s >= 0.85: return 'READY for production'
    if s >= 0.80: return 'READY — proceed to pilot expansion'
    if s >= 0.70: return 'USABLE - minor fixes before scaling'
    if s >= 0.60: return 'USABLE but needs targeted fixes'
    return 'DO NOT SCALE'


rs['interpretation'] = interpret(ws_total)
rs.setdefault('notes', {})['C1_subject_level'] = (
    f'{frac_c1_gt_null:.3f} subjects obs>null (label-shuffle)')
rs.setdefault('notes', {})['C2_subject_level'] = (
    f'{mean_c2_sign:.3f} mean family sign agreement')
rs['gate_6_subject_level_c1c2'] = 'COMPLETE'

with open(rs_path, 'w') as f:
    json.dump(rs, f, indent=2)

print('\n=== Updated Readiness Score ===')
for k, v in rs['scores'].items():
    print(f'  {k:<45}: {v}')
print(f'\n  WEIGHTED TOTAL:  {rs["weighted_score"]:.4f}')
print(f'  INTERPRETATION:  {rs["interpretation"]}')
print(f'\nSaved: {rs_path}')
