MNPS v3.2: PCV-OI-AR-CONTRACT-003
=================================

Status: partial method freeze with low-level formula kernels for PCV and AR;
no end-to-end matched EEG realization is selected for PCV, OI, or AR. This
record captures the science-lead decisions without inventing unresolved
preprocessing, reference, sensor-set, or support choices.

OI — matrix kernel frozen
-------------------------

The implemented v3.2 kernel
``OI.matrix_input_channel_median_normalized_PE5_delay1.v1`` computes normalized
order-5 permutation entropy with delay 1 sample. For ``T`` samples it counts
all ``T-4`` consecutive patterns per supplied row, sorts ties with stable
mergesort in original sample order, uses Shannon entropy with natural
logarithms divided by ``ln(5!)``, and returns the median over all supplied
matrix rows. It rejects complex/non-finite input and ``T < 5``. The five-sample
minimum is a mathematical computability boundary, not a declared EEG-window
support criterion.

This is a distinct matrix-input ID. It does not relabel fixed-set v3.1 E1 or
the v3.1 native-window pipeline path. The full EEG realization remains open:
fixed-set versus native channels, exact preprocessing/reference, source sample
grid, dependency/gap support, and valid-window behavior must be audited before
an empirical support-matched comparison. The result contract leaves support
unknown, realization incomplete/unresolved and ``qualified=false``.

PCV — preserve the versioned alternatives
------------------------------------------

The v3.2 formula kernel
``PCV.sensor_phase_order_parameter_population_sd.v1`` computes the population
standard deviation (``ddof=0``) over time of
``abs(mean_c(exp(i*phi[c,t])))`` for a supplied finite real phase matrix. It
records optional phase-estimator/band provenance and actual row labels when
provided; absent provenance remains QC and does not suppress a computable
formula value. This low-level kernel does not estimate phase or infer its
band, sensor realization, reference, temporal support, or validity in
low-amplitude data.

The repository also contains a fixed-83-channel contextual 8–13 Hz, 250 Hz,
20 s M1-G2 path and a separate native-window pipeline path. They have
different estimator/realization identities and must not be conflated. No v3.2
choice is made between them. Band, phase/filter method, reference, sensor set,
source dependency and support for each proposed window length remain
unresolved.

AR — preserve the ordered vector
--------------------------------

For caller-constructed activity vectors ``v_b``, the v3.2 low-level kernel
``AR.sensor_activity_adjacent_pearson_change_vector.v1`` computes
``1 - corr_sensor(v_b, v_(b+1))`` and returns all ``B-1`` components in order.
It requires positive, equal-duration, contiguous block intervals and explicit
vector-construction and scaling IDs. A constant-across-sensor vector makes
only the corresponding component undefined; component-local nulls and reasons
are retained. The connectivity-vector variant has a separate identity and is
not implemented by this kernel. No component aggregation is defined. The v3.1
20 s / 2 s example has nine ordered values; that shape does not automatically
apply to other windows. Float-second interval comparisons allow at most 1 ns
absolute rounding, independent of the time origin; sample-index boundaries are
preferred when available.

Comparison consequence
----------------------

The OI matrix formula and PCV and AR low-level formula kernels are
implemented. They do not receive end-to-end EEG defaults here. None of the
three is yet an end-to-end v3.2 EEG candidate ready for a common vector ablation. Read
the current readiness decision in
:doc:`mnps_v3_2_vector_ablation_readiness_001` and the supporting audit in
:doc:`mnps_v3_2_vector_contract_002`. All measurements remain experimental and
unqualified; pipeline and export integration are outside this decision.

The focused OI and common-contract tests are recorded in
:doc:`mnps_v3_2_oi` and :doc:`mnps_v3_2_contracts`.
