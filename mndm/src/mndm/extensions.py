"""Extended NDT / MNPS coordinate utilities (EEG-first, modality-agnostic).

This module implements the mathematical building blocks sketched in
``openneuro/docs/todo_extensions_mnps_20251117.md``:

- Energetic curvature (E-Kappa) for a scalar energy series E(t).
- Resonant phase modes (RFM) for multivariate time series.
- Organisational coherence (O-Koh) via simple graph Betti numbers.
- Temporal integrity grade (TIG) from MNPS / Stratified trajectories.

The functions here are intentionally low-level and IO-free. They operate on
NumPy arrays and small Python dicts so they can be reused from both EEG and
future fMRI pipelines, as well as from downstream analysis code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def _bandpass_fft(data: np.ndarray, sfreq: float, l_freq: float, h_freq: float) -> np.ndarray:
    """FFT-based zero-phase bandpass filter (SciPy-free fallback).

    This is an "ideal" rectangular bandpass in the frequency domain. It's not a
    Butterworth filter, but it is stable, dependency-free, and sufficient for
    the lightweight extension modules (RFM/phase proxies).
    """
    data = np.asarray(data, dtype=float)
    sfreq = float(sfreq)
    l_freq = float(l_freq)
    h_freq = float(h_freq)
    if sfreq <= 0:
        raise ValueError("sfreq must be positive")
    if not (0.0 <= l_freq < h_freq <= 0.5 * sfreq):
        raise ValueError(f"Invalid band [{l_freq}, {h_freq}] for sfreq={sfreq}")

    n = data.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    X = np.fft.rfft(data, axis=-1)
    mask = (freqs >= l_freq) & (freqs <= h_freq)
    X *= mask.astype(X.dtype, copy=False)
    return np.fft.irfft(X, n=n, axis=-1)


def _hilbert_analytic(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute analytic signal via FFT (SciPy-free Hilbert transform)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    Xf = np.fft.fft(x, n=n, axis=axis)

    h = np.zeros(n, dtype=float)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0

    shape = [1] * x.ndim
    shape[axis] = n
    h = h.reshape(shape)
    return np.fft.ifft(Xf * h, n=n, axis=axis)


def bandpass_filter(
    data: np.ndarray,
    sfreq: float,
    l_freq: float,
    h_freq: float,
    order: int = 4,
) -> np.ndarray:
    """Simple zero-phase bandpass in the Fourier domain.

    Args:
        data: Array with shape ``(..., n_times)``.
        sfreq: Sampling frequency in Hz.
        l_freq: Low cut-off frequency in Hz.
        h_freq: High cut-off frequency in Hz.
        order: Kept for API compatibility; implementation uses a rectangular
            FFT bandpass (not a Butterworth of this order).

    Returns:
        Filtered array with the same shape as ``data``.
    """
    # Note: `order` is kept for API compatibility, but the current implementation
    # uses an FFT-domain rectangular bandpass to avoid SciPy binary dependencies.
    _ = order
    return _bandpass_fft(data, sfreq=float(sfreq), l_freq=float(l_freq), h_freq=float(h_freq))


def compute_energy_from_regions(
    reg_signals: np.ndarray,
    weights: Optional[np.ndarray] = None,
    window_size: Optional[int] = None,
) -> np.ndarray:
    """Compute a global energy series from regional signals.

    Args:
        reg_signals: Array with shape ``(n_regions, n_times)``.
        weights: Optional weights per region, shape ``(n_regions,)``. If None,
            uses uniform weights.
        window_size: Optional window length in samples for smoothing squared
            amplitude. If None or <= 1, uses instantaneous squared amplitude.

    Returns:
        1-D array ``E`` with shape ``(n_times,)``.
    """

    reg_signals = np.asarray(reg_signals, dtype=float)
    if reg_signals.ndim != 2:
        raise ValueError("reg_signals must have shape (n_regions, n_times)")
    n_regions, _ = reg_signals.shape

    if weights is None:
        weights = np.ones(n_regions, dtype=float) / float(max(n_regions, 1))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_regions,):
        raise ValueError("weights must have shape (n_regions,)")
    weights = weights / (weights.sum() or 1.0)

    if window_size is None or window_size <= 1:
        local_power = reg_signals ** 2  # (n_regions, n_times)
    else:
        # Simple moving average of squared amplitude
        sq = reg_signals ** 2
        kernel = np.ones(int(window_size), dtype=float) / float(window_size)
        local_power = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="same"),
            axis=1,
            arr=sq,
        )

    # Weighted sum over regions
    E = np.tensordot(weights, local_power, axes=(0, 0))
    return np.asarray(E, dtype=float)


def compute_kappa_E_1d(E: np.ndarray, dt: float) -> np.ndarray:
    """Compute 1D energetic curvature :math:`\\kappa_E` from scalar E(t).

    This uses the standard curvature of the curve (t, E(t)) in R^2:

    .. math::

        \\kappa_E(t) = \\frac{E''(t)}{\\big(1 + (E'(t))^2\\big)^{3/2}}.

    Args:
        E: 1-D array, shape ``(n_times,)``.
        dt: Sampling interval in seconds.

    Returns:
        1-D curvature series; first and last samples are set to NaN.
    """

    E = np.asarray(E, dtype=float)
    if E.ndim != 1:
        raise ValueError("E must be 1-D array (n_times,)")
    if dt <= 0:
        raise ValueError("dt must be positive")

    dE = (np.roll(E, -1) - np.roll(E, 1)) / (2.0 * dt)
    d2E = (np.roll(E, -1) - 2.0 * E + np.roll(E, 1)) / (dt ** 2)

    kappa = d2E / np.power(1.0 + dE ** 2, 1.5)
    # Boundary handling: curvature undefined at the sequence edges
    if kappa.size >= 1:
        kappa[0] = np.nan
        kappa[-1] = np.nan
    return kappa.astype(float)


@dataclass
class RFMResult:
    """Container for Resonant Phase Modes."""

    times: np.ndarray           # [n_windows]
    eigvals: np.ndarray         # [n_windows, n_channels]
    eigvecs: np.ndarray         # [n_windows, n_modes, n_channels]
    dominance: np.ndarray       # [n_windows]


def compute_rfm(
    x: np.ndarray,
    sfreq: float,
    window_sec: float,
    step_sec: float,
    n_modes: int = 3,
    band: Optional[Tuple[float, float]] = None,
) -> RFMResult:
    """Compute Resonant Phase Modes (RFM) from multichannel time series.

    Args:
        x: Array with shape ``(n_channels, n_times)``.
        sfreq: Sampling frequency in Hz.
        window_sec: Window length in seconds for phase-coherence estimation.
        step_sec: Step between successive windows in seconds.
        n_modes: Number of leading modes to retain.
        band: Optional ``(l_freq, h_freq)`` in Hz for bandpass filtering before
            the Hilbert transform; if None, no extra filtering.

    Returns:
        :class:`RFMResult` with window centers, eigenvalues/vectors, and dominance.
    """

    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("x must have shape (n_channels, n_times)")
    n_channels, n_times = x.shape
    if n_channels == 0 or n_times == 0:
        raise ValueError("x must be non-empty")

    if sfreq <= 0:
        raise ValueError("sfreq must be positive")

    n_modes = int(n_modes)
    if n_modes <= 0:
        raise ValueError("n_modes must be a positive integer")

    if band is not None:
        l_freq, h_freq = band
        x = bandpass_filter(x, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, order=4)

    # Analytic signal & phase
    analytic = _hilbert_analytic(x, axis=1)
    phases = np.angle(analytic)  # (n_channels, n_times)

    win = int(round(window_sec * sfreq))
    step = int(round(step_sec * sfreq))
    if win <= 1 or win > n_times:
        raise ValueError("Invalid RFM window size for given data length")
    if step <= 0:
        raise ValueError("step_sec must be positive")

    n_windows = 1 + (n_times - win) // step
    if n_windows <= 0:
        raise ValueError("Window/step too large for data length")

    eigvals = np.zeros((n_windows, n_channels), dtype=float)
    # If n_modes > n_channels, keep the requested shape but fill missing modes with NaN.
    eigvecs = np.full((n_windows, n_modes, n_channels), np.nan, dtype=float)
    dominance = np.zeros(n_windows, dtype=float)
    times = np.zeros(n_windows, dtype=float)

    for i in range(n_windows):
        start = i * step
        stop = start + win
        phi_win = phases[:, start:stop]  # (n_channels, win)

        # Phase coherence matrix Φ(t)
        phase_diff = phi_win[:, None, :] - phi_win[None, :, :]  # (ch, ch, win)
        plv = np.abs(np.exp(1j * phase_diff).mean(axis=-1))     # (ch, ch)
        # Robustify: enforce symmetry, finite values, and a well-conditioned diagonal.
        plv = np.nan_to_num(plv, nan=0.0, posinf=0.0, neginf=0.0)
        plv = (plv + plv.T) / 2.0                               # enforce symmetry
        # PLV diagonal should be 1.0 (self-coherence); enforce to avoid numerical drift.
        np.fill_diagonal(plv, 1.0)
        # Tiny diagonal jitter improves eigen solver stability for near-singular matrices.
        plv = plv + (1e-8 * np.eye(n_channels, dtype=float))

        # Eigen-decomposition (symmetric)
        try:
            w, v = np.linalg.eigh(plv)
        except np.linalg.LinAlgError:
            # Fallback: SVD is typically more robust than eigendecomposition in the
            # presence of numerical issues. For symmetric PSD-like PLV matrices,
            # singular vectors are a good proxy for eigenvectors and singular values
            # approximate eigenvalues.
            u, s, _vt = np.linalg.svd(plv, full_matrices=False)
            w = s
            v = u
        idx = np.argsort(w)[::-1]
        w = w[idx]
        v = v[:, idx]

        eigvals[i] = w
        m = min(n_modes, int(v.shape[1]), int(n_channels))
        if m > 0:
            eigvecs[i, :m, :] = v[:, :m].T
        dominance[i] = w[0] / (w.sum() + 1e-12)
        times[i] = (start + stop) / 2.0 / sfreq

    return RFMResult(
        times=times,
        eigvals=eigvals,
        eigvecs=eigvecs,
        dominance=dominance,
    )


def compute_OKoh(
    C: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute 0th and 1st order Organisational Coherence from a FC matrix.

    We approximate Betti numbers in a graph-theoretic way:

    - :math:`\\beta_0`: number of connected components.
    - :math:`\\beta_1`: cycle rank (edges minus vertices plus :math:`\\beta_0`).

    Args:
        C: Functional connectivity matrix, shape ``(n_nodes, n_nodes)``.
        thresholds: Thresholds on ``abs(C_ij)`` used to build graphs
            :math:`G(\\varepsilon)`. If None, thresholds are chosen from the
            10th to 90th percentile of off-diagonal ``abs(C_ij)`` (upper triangle).
        weights: Optional weights for integration over thresholds. If None,
            uniform.

    Returns:
        Dictionary with keys ``thresholds``, ``beta0``, ``beta1`` (each
        ``(n_thr,)`` arrays), ``OKoh0``, and ``OKoh1`` (floats).
    """

    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a square matrix (n_nodes, n_nodes)")

    n = C.shape[0]
    if n == 0:
        raise ValueError("C must have at least one node")

    upper = np.abs(C[np.triu_indices(n, k=1)])
    if upper.size == 0:
        raise ValueError("Connectivity matrix too small (no off-diagonal entries)")

    if thresholds is None:
        low = np.percentile(upper, 10)
        high = np.percentile(upper, 90)
        thresholds = np.linspace(low, high, 20, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    if weights is None:
        weights = np.ones_like(thresholds, dtype=float) / float(len(thresholds))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != thresholds.shape:
        raise ValueError("weights must have the same shape as thresholds")

    beta0 = np.zeros_like(thresholds, dtype=float)
    beta1 = np.zeros_like(thresholds, dtype=float)

    def _num_components(adj: np.ndarray) -> int:
        """Return number of connected components in an undirected graph."""

        visited = np.zeros(adj.shape[0], dtype=bool)
        n_comp = 0
        for start in range(adj.shape[0]):
            if visited[start]:
                continue
            n_comp += 1
            stack = [start]
            visited[start] = True
            while stack:
                node = stack.pop()
                neighbours = np.nonzero(adj[node])[0]
                for nb in neighbours:
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
        return n_comp

    for i, thr in enumerate(thresholds):
        adj = (np.abs(C) >= thr).astype(int)
        np.fill_diagonal(adj, 0)

        n_nodes = adj.shape[0]
        # Each undirected edge appears twice in adj (i,j) and (j,i).
        n_edges = int(np.count_nonzero(np.triu(adj, k=1)))
        n_comp = _num_components(adj)

        beta0[i] = float(n_comp)
        beta1[i] = float(n_edges - n_nodes + n_comp)

    OKoh0 = float(np.sum(beta0 * weights))
    OKoh1 = float(np.sum(beta1 * weights))

    return {
        "thresholds": thresholds,
        "beta0": beta0,
        "beta1": beta1,
        "OKoh0": OKoh0,
        "OKoh1": OKoh1,
    }


def compute_TIG_autocorr(
    s: np.ndarray,
    dt: float,
    max_lag_sec: float,
    n_lags: int = 20,
    T_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute Temporal Integrity Grade (TIG) from a state trajectory.

    TIG is defined from the decay of the normalized autocorrelation:

    .. math::

        C(\\Delta) = \\frac{\\mathbb{E}_t[ \\langle s(t), s(t+\\Delta)\\rangle ]}
                           {\\mathbb{E}_t[ \\|s(t)\\|^2 ]}

    assuming an exponential form :math:`C(\\Delta) \\approx e^{-\\Delta/\\tau}`.
    We estimate :math:`\\tau` from a log-linear fit and define:

    .. math::

        \\mathrm{TIG} = \\tau / T_\\text{max}.

    Args:
        s: Array with shape ``(n_times, n_dims)`` or ``(n_times,)``.
        dt: Sampling interval in seconds.
        max_lag_sec: Maximum lag in seconds for autocorrelation.
        n_lags: Number of lag samples between 0 and ``max_lag_sec``.
        T_max: Normalisation constant for TIG; if None, uses ``max_lag_sec``.

    Returns:
        Dict with keys such as ``lags_sec``, ``autocorr``, ``tau``, ``TIG``.
    """

    s = np.asarray(s, dtype=float)
    if s.ndim == 1:
        s = s[:, None]
    if s.ndim != 2:
        raise ValueError("s must have shape (n_times,) or (n_times, n_dims)")

    if dt <= 0:
        raise ValueError("dt must be positive")
    if max_lag_sec <= 0:
        raise ValueError("max_lag_sec must be positive")

    n_times, _ = s.shape
    max_lag = int(round(max_lag_sec / dt))
    if max_lag < 1:
        raise ValueError("max_lag_sec too small for given dt")

    lags = np.linspace(0, max_lag, n_lags, dtype=int)
    lags = np.unique(lags)

    norms = np.sum(s ** 2, axis=1)
    denom = norms.mean() + 1e-12

    C_vals = []
    real_lags = []
    for lag in lags:
        if lag == 0:
            c = 1.0
        else:
            if lag >= n_times:
                continue
            prod = np.sum(s[:-lag] * s[lag:], axis=1)  # (n_times - lag,)
            c = float(prod.mean() / denom)
        if c > 0:
            C_vals.append(c)
            real_lags.append(lag * dt)

    C_vals_arr = np.asarray(C_vals, dtype=float)
    real_lags_arr = np.asarray(real_lags, dtype=float)

    if C_vals_arr.size < 3:
        return {
            "lags_sec": real_lags_arr,
            "autocorr": C_vals_arr,
            "tau": np.nan,
            "TIG": np.nan,
            "provisional": True,
        }

    # Ignore zero lag for fitting (C(0) = 1)
    mask = real_lags_arr > 0
    if np.sum(mask) < 2:
        return {
            "lags_sec": real_lags_arr,
            "autocorr": C_vals_arr,
            "tau": np.nan,
            "TIG": np.nan,
            "provisional": True,
        }

    x = real_lags_arr[mask]
    y = np.log(C_vals_arr[mask] + 1e-12)
    slope, intercept = np.polyfit(x, y, 1)

    # Raw tau estimate from exponential fit; may be very large or infinite.
    if slope >= 0:
        tau_raw = np.inf
    else:
        tau_raw = -1.0 / slope

    if T_max is None:
        T_max = max_lag_sec

    provisional = False
    if not np.isfinite(tau_raw) or tau_raw > T_max:
        # Non-decaying or extremely slow decay: treat as "maximal" integrity on the
        # requested horizon, but mark as provisional (fit cannot identify tau well).
        tau_eff = float(T_max)
        tig = 1.0
        provisional = True
    elif tau_raw <= 0:
        # Degenerate negative/zero tau; treat as no integrity.
        tau_eff = 0.0
        tig = 0.0
        provisional = True
    else:
        tau_eff = float(tau_raw)
        tig = float(tau_eff / T_max)

    return {
        "lags_sec": real_lags_arr,
        "autocorr": C_vals_arr,
        "tau": tau_eff,
        "TIG": tig,
        "provisional": provisional,
    }



