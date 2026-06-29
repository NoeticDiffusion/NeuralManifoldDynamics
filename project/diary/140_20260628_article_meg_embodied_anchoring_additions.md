# 140 — 2026-06-28 — Article: MEG exploratory extension + embodied anchoring additions

## Session purpose

Added MEG as an exploratory/beta extension and the concrete embodied anchoring
(HRV, vagal index, pupillometry) material to the NeuralManifoldDynamics article.

## Changes to article

File: `project/articles/NeuralManifoldDynamics/NeuralManifoldDynamics A Versioned Measurement Contract for Low-Dimensional Neural-Manifold Trajectories.typ`

Line count: 854 → 919 (+65 lines).

### New subsection: "Embodied Anchoring: HRV, Vagal Index, and Pupillometry"
Position: within "Key Methodological Advances" (before "Formal 3D, 9D, and anchoring separation").

Content:
- ECG/HRV surface: `ecg_hrv_*` columns, `vagal_index`/`sympathetic_index`/`anchor_index`/`vascular_index` priority hierarchy, HRV v0.1 definition, ECG polarity correction (ds006848: 92.7% inverted QRS fixed).
- ds003838 validation table (Friedman results, n=62): vagal_index χ²=88.0 p=3.4×10⁻¹⁸, listen–mem13 Cohen's d=1.995.
- HRV superwindow contamination gating: ds006848 WM-phase 87.7% contaminated, rest-phase clean.
- PPG surface: vascular_index, quality flags.
- Pupillometry surface: mean diameter, volatility, blink proxy, quality — implemented but not yet dataset-scale validated.

### New subsection: "MEG Ingest — Exploratory Pilot Extension"
Position: within "Modality Coverage" (after "PhysioNet / WFDB ingest path", before "Axis Construction").

Content:
- FIF ingest via MNE-Python: `meg_mag_*`, `meg_grad_*`, `meg_*` feature columns.
- Shadow mapping: MEG feature types route through existing 9D contract.
- Row provenance (`row_source/`, `mndm.row_source.v1`) for simultaneous MEEG recordings.
- `features_projection_z` for transform-aware spectral export (log10 → robust-z → clip).
- Pilot readiness table (5 subjects, ds003645): readiness score 0.7879.
- Labeled EXPLORATORY — not yet production-scale.

### Updated: Abstract, Author Summary
- Abstract: detailed MEG and embodied anchor additions.
- Author Summary: MEG exploratory extension and concrete embodied anchor modalities.

### Updated: Conclusions (item 4 and 7)
- Item 4: concrete anchor modalities listed; ds003838 Cohen's d cited.
- Item 7: MEG exploratory label, readiness score, row-source provenance, features_projection_z.

### Updated: Limitations
- Added MEG-specific limitations: features_robust_z spectral collapse, bimodal subject structure.
- Added embodied anchoring limitations: ds003838 not yet cross-replicated; ds006848 HRV gated; pupillometry not yet validated at dataset scale.

### Updated: Technical Terms
- Added: vagal_index, sympathetic_index, HRV v0.1, row_source provenance, features_projection_z, MEG shadow mapping.

## Sources used

- Diary 110 (HRV v0.1 implementation)
- Diary 111 (block-native HRV export, ds003838)
- Diary 112 (ds003838 full HRV block-native validation)
- Diary 113 (noetic anchor article handover)
- Diary 127 (full stats package ds003838)
- Diary 134 (generic ECG BIDS infrastructure)
- Diary 139 (subject-level C1/C2, MEG readiness)
- claims/ds006848_verbal_wm_claims.md (HRV contamination gate)
- features/pupil.py, features/ppg.py (implemented surfaces)
