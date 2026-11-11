MNPS v3.2: NIE-EDGE-CONTRACT-001
================================

Status: frozen narrow experimental contract
--------------------------------------------

This record freezes one repository-backed EEG connectivity baseline for a
future NIE study. It authorizes no cohort run, empirical graph comparison,
condition contrast, or scientific qualification. It does not integrate the
matrix-level NIE API into pipeline/export.

The full versioned protocol is
``project/new_mnps_v.3.2/protocols/NIE-EDGE-CONTRACT-001.md``.

Frozen baseline
---------------

The input is the native effective EEG channel realization from the documented
``ds004148-1to45hz-250hz-causal-polyphase-minphase-v1`` branch: resampled to
250 Hz, minimum-phase FIR bandpass 1–45 Hz, then instantaneous average
reference over the exact effective surface. No further preprocessing occurs in
the edge estimator. The declared window is 20 seconds (5,000 samples),
half-open and nonoverlapping; every supported full window is retained and an
incomplete tail is reported. This is one named realization, not a universal
minimum or the full 4/8/12/20-second grid.

For each window, ``R[i,j] = corr(X[i,:], X[j,:])`` and
``W[i,j] = abs(R[i,j])`` off diagonal, with ``W[i,i] = 0``. The edge estimator
ID is ``absolute_zero_lag_Pearson_r`` (the existing v3.1
window-helper construction label, adopted here as the edge ID). The
predeclared edge rule is ``no_threshold``: positive weights are kept, exact
zeros are absent. Beyond
taking the absolute value, no threshold, clipping, shrinkage, or graph-density
operation is used. NIE uses
``D1.sensor_weighted_global_efficiency_invweight.v0``: positive weights map
to lengths ``1/W``, shortest paths are computed, disconnected pairs contribute
zero, and efficiency is averaged over all ordered distinct node pairs.

Actual ordered labels, order/set hashes, reference-set hash, preprocessing ID,
source dependency bounds, temporal support, QC, and code/config/protocol/source
provenance must accompany each value. A different channel set/reference is a
distinct realization. Invalid inputs remain candidate-local nulls; no channel
or edge is silently dropped or repaired. Outputs remain ``qualified=false``.

Unresolved scope boundary
-------------------------

The codebase also has a dWPLI² estimator, but its finite estimates can be
negative and the NIE kernel accepts weights only in ``[0,1]``. The repository
does not justify clipping negatives, taking absolute values, or rejecting
negative edges as an NIE construction. Any dWPLI²-based NIE must therefore be
a separate versioned candidate with a justified edge mapping. The frozen
absolute-Pearson baseline remains reference-sensitive and is not a
volume-conduction correction or a biological connectivity claim.

No signal was processed and no tests of empirical graphs were run for this
contract.
