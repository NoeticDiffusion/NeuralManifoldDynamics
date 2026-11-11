MNPS v3.2: VECTOR-FEASIBILITY-002 results
==========================================

Status
------

**Engineering PASS; scientific HOLD.** The frozen twelve-participant cohort
completed the common audited E1-G2 grid. The experimental ED, OI, and NIE
estimators each produced 168 valid measurements across 12 participants and 14
supported windows per participant. The first nominal window remains explicitly
unsupported for every participant. All records remain exploratory and
unqualified.

This bounded study establishes cross-recording measurement feasibility and
descriptive participant-level associations. It does not establish a qualified
vector, biological interpretation, prediction, or an MNPS state-space model.

Frozen inputs and output
-----------------------

The outcome-blind sample was drawn from 21 locally available, fixed-83-compatible
sources after excluding the earlier EPCTL01 development recording and five
fixed-surface label mismatches. The twelve selected IDs, draw order, metadata
roster, and selection digest are recorded in
``mndm/config/mnps_v3_2_vector_feasibility_002_selection_20261013.json``.
There were no replacements. The frozen method and review rules are in
:doc:`mnps_v3_2_vector_feasibility_002_protocol`.

The final verified run is
``outputs/VECTOR-FEASIBILITY-002/run_20261013T_sl006_vectorfeasibility002_final/``.
This exact run is authoritative for the status below; earlier implementation
attempts are retained as local history.
Its ``run_manifest.json`` pins the selection, protocol, base overlay, position
file, code components, runtime, and output hashes. The selected source EDFs
have full SHA-256 hashes recorded before loader extraction and stable matching
hashes after extraction. The loader's per-window source, overlay, and position
identities all passed. Each computed measurement records its matrix hash and
the ordered fixed-83 realization.
The composite code SHA-256 for this run is
``cac9cb8befb90644f6dd198ddee523f3c393610b5c3bedc6efb28c72be2c1c1c``.
The run-manifest SHA-256 is
``b10f582844c49614d0238f6303947610345fea34f25ec79dc50f7a251a88d702``.

The public source is ANPHY-Sleep (`OSF record <https://doi.org/10.17605/OSF.IO/R26FH>`_;
`dataset paper <https://doi.org/10.1038/s41597-024-03722-1>`_). The exact local archive
revision is unverified. The paper describes EDF+D while the local files report
EDF+C; the
results identify the processed local snapshot by per-file hashes and do not
claim byte identity with a particular OSF archive release. The broader local
inventory has 29 demographic IDs but only 27 EDF files.

Window and measurement accounting
---------------------------------

* All twelve selected source grids completed and retained all 15 nominal
  half-open 20-second windows in the first 300 seconds: 180 rows total.
* The ``[0,20)`` window is unsupported in all twelve recordings because the
  causal preprocessing dependency reaches before the source recording start.
  Those rows remain present with null values and explicit support reasons.
* The remaining 14 windows per participant passed support checks. ED, OI, and
  NIE each have 168 valid values and 12 unsupported values; none is qualified.
* The realization is the named ordered fixed-83 surface with the frozen
  average-reference set. No channels were imputed or substituted.

The raw covariance ED reconstruction agreed with the primary ED kernel to a
maximum absolute residual of ``6.22e-15`` across the 168 supported windows. All
168 common-time-index permutation controls passed the frozen covariance,
correlation, ED, and NIE invariance checks. OI changes under this shared
permutation, as allowed by the protocol; the control is not a biological
utility test.

Participant-level descriptive results
--------------------------------------

The primary association matrix uses one median across supported windows per
participant, with ``n=12`` for each primary pair. Pearson correlations were:

.. list-table:: Participant-median associations
   :header-rows: 1
   :widths: 24 18 58

   * - Pair
     - Pearson ``r``
     - Descriptive reading
   * - ED / OI
     - ``+0.098``
     - Little linear association in this bounded sample.
   * - ED / NIE
     - ``-0.890``
     - Strong negative association, below the preregistered absolute ``0.90`` review trigger.
   * - OI / NIE
     - ``-0.138``
     - Little linear association in this bounded sample.

No preregistered review trigger fired. In particular, no primary pair met the
``|r| >= 0.90`` rule with at least ten pairwise-valid participant medians; the
ED-versus-variance and NIE path-increment zero checks did not fire, and OI
changed under the shared time permutation in every participant. No coordinate
is automatically removed or promoted.

As a separate structural diagnostic, NIE and mean absolute off-diagonal
correlation had participant-level ``r=+0.999``. These quantities share the
absolute Pearson edge realization; this association is not an independent
candidate comparison or evidence that either is a qualified neural feature.
The participant-median NIE-minus-mean-|r| path increment remained nonzero in
all twelve participants (approximately ``0.052`` to ``0.070``).

All correlations are descriptive only. The sample is small, adjacent window
source dependencies overlap, and no windows were pooled as independent
observations. The study reports no p-values or confidence claims.

Scientific limits and follow-up
-------------------------------

The result supports computing these three unchanged estimators on this named
fixed-83 realization for the selected local ANPHY-Sleep recordings. It does not
resolve the EDF+D versus EDF+C discrepancy or the unknown archive revision; it
does not generalize to the five recordings with different channel labels, other
sensor realizations, other time ranges, or other cohorts. It also does not
establish cross-dataset equivalence, predictive value, biological
complementarity, or qualification. The strong descriptive ED-NIE association
and the nearly identical NIE / mean-|r| diagnostic warrant scientific review
before proposing a vector ablation, but do not authorize feature removal.

The raw long-form window outputs, per-participant provenance, summary matrix,
and hashes remain under the ignored run output root above. Scientific status
remains HOLD pending a separately stated research question and evidence.
