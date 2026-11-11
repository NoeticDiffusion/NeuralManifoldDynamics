MNPS v3.2: SL-007 bounded execution results
===========================================

Status
------

SL-007 authorizes two bounded, outcome-blind ANPHY-Sleep candidate-value
analyses and a separate I-CARE feasibility preflight. The NIE analysis is
complete. The OI analysis is running under its frozen protocol; its result
section will be updated from the completed run manifest. SL-006's engineering
PASS and scientific HOLD remain the governing qualification state. No
biological or CPC claim is made here.

NIE-INCREMENTAL-001
-------------------

Run
~~~

Final run:
``outputs/NIE-INCREMENTAL-001/run_20261013T_sl007_nie_incremental001_final``.
It used all 12 selected SL-006 participants and retained all 180 nominal rows:
168 valid NIE windows and 12 source-unsupported first windows with explicit
null values and reasons. The SL-006 manifest and all 197 artifacts it names
were verified before use. All 168 edge matrices were recalculated against the
stored NIE values and mean absolute weights. The output contains 8,232
deterministic controls (49 per supported window), and each preserves the exact
3,403-edge upper-triangle multiset including zeros.

Descriptive result
~~~~~~~~~~~~~~~~~~

The median ``Delta_path`` across supported windows was 0.062925; the median of
the per-window control medians was 0.106187. The observed value was below the
control 2.5th percentile in 168/168 windows. The median fraction of controls
no greater than the observed value was 0. This fraction is descriptive and is
not a p-value.

The control randomly assigns the same edge-weight multiset to graph positions
and does not preserve sensor geometry or produce a realizable EEG correlation
matrix. These results describe this graph-position contrast only. They do not
establish biological or population-level incremental value and do not qualify
NIE for a downstream scientific use.

OI-TEMPORAL-SPECIFICITY-001
--------------------------

The final frozen run is in progress. Its completion record will identify the
run ID, reconstructed-window hash audit, valid and unsupported row counts,
surrogate spectrum error, and participant-level pairwise ``n`` and IDs for the
OI, Welch power, spectral entropy and autocorrelation comparisons. OI uses the
unchanged order-5, delay-1 estimator; its two per-channel IAAFT replicates use
50 iterations and preserve each channel's amplitude multiset. No parameter
selection or inference threshold is applied.

COMA-VECTOR-PREFLIGHT-001
------------------------

The outcome-blind I-CARE preflight protocol is frozen, but source selection and
measurement extraction have not run. The local workspace contains the I-CARE
ingest configurations and metadata, but no located raw WFDB training files.
Therefore there is no verified release manifest, patient roster, target-period
support audit, or ED/OI/NIE value. The protocol retains selected patients with
no target-period support as zero-window feasibility outcomes and prohibits
opening CPC/outcome data during preflight.

Contracts and validation
------------------------

The AR low-level formula kernel returns ordered adjacent ``1 - Pearson r``
components and keeps undefined pairs local. The PCV low-level kernel returns
the population standard deviation of the phase order parameter. Neither kernel
chooses EEG preprocessing, channel realization, interval construction, or
qualification rules; both remain experimental and unqualified.

Focused tests for OI, NIE, PCV/AR, and the SL-007 runner pass (45 tests). Runner
manifests record the protocol hash, runner hash, imported kernel/helper/loader
and preprocessing source hashes, dependency versions, and output hashes.
Independent review defects around artifact verification, explicit null rows,
IAAFT power-spectrum error, and pairwise sample IDs/counts were addressed. The
OI summary field mapping was also corrected so its conventional comparisons
use participant-level values with the matching participant IDs.

This is a bounded method-development result, not an outcome analysis or
research qualification.
