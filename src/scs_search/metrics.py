"""EMG restoration metrics and seed-level aggregation helpers."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def emg_envelope(emg_signal: np.ndarray, window_ms: int = 25) -> np.ndarray:
    """Compute a simple full-wave-rectified moving-average EMG envelope."""

    signal = np.abs(np.asarray(emg_signal, dtype=float))
    window = max(1, int(window_ms))
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(signal, kernel, mode="same")


def _aligned_segments(reference: np.ndarray, candidate: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Return aligned signal segments under an integer lag."""

    if lag == 0:
        length = min(reference.size, candidate.size)
        return reference[:length], candidate[:length]
    if lag > 0:
        length = min(reference.size - lag, candidate.size)
        return reference[lag : lag + length], candidate[:length]
    shift = abs(lag)
    length = min(reference.size, candidate.size - shift)
    return reference[:length], candidate[shift : shift + length]


def pearson_correlation(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Compute a stable Pearson correlation for potentially flat traces."""

    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.size == 0 or cand.size == 0:
        return 0.0
    if np.allclose(ref, ref[0]) or np.allclose(cand, cand[0]):
        return 1.0 if np.allclose(ref, cand) else 0.0
    return float(np.corrcoef(ref, cand)[0, 1])


def _prepare_metric_signal(
    emg_signal: np.ndarray,
    *,
    envelope_window_ms: int,
    use_envelope: bool,
) -> np.ndarray:
    """Apply the configured EMG preprocessing used by downstream metrics."""

    signal = np.asarray(emg_signal, dtype=float)
    if use_envelope:
        return emg_envelope(signal, window_ms=envelope_window_ms)
    return signal


def compute_emg_similarity(
    reference_emg: np.ndarray,
    candidate_emg: np.ndarray,
    envelope_window_ms: int = 25,
    max_lag_ms: int = 0,
    use_envelope: bool = True,
) -> float:
    """Return the best aligned correlation between reference and candidate EMG."""

    reference = _prepare_metric_signal(
        reference_emg,
        envelope_window_ms=envelope_window_ms,
        use_envelope=use_envelope,
    )
    candidate = _prepare_metric_signal(
        candidate_emg,
        envelope_window_ms=envelope_window_ms,
        use_envelope=use_envelope,
    )
    best = -1.0
    for lag in range(-int(max_lag_ms), int(max_lag_ms) + 1):
        ref_seg, cand_seg = _aligned_segments(reference, candidate, lag)
        if ref_seg.size < 2 or cand_seg.size < 2:
            continue
        best = max(best, pearson_correlation(ref_seg, cand_seg))
    return float(best)


def relative_envelope_rmse(
    reference_emg: np.ndarray,
    candidate_emg: np.ndarray,
    envelope_window_ms: int = 25,
    max_lag_ms: int = 0,
    use_envelope: bool = True,
) -> float:
    """Return the best aligned relative RMSE on the configured EMG representation."""

    reference = _prepare_metric_signal(
        reference_emg,
        envelope_window_ms=envelope_window_ms,
        use_envelope=use_envelope,
    )
    candidate = _prepare_metric_signal(
        candidate_emg,
        envelope_window_ms=envelope_window_ms,
        use_envelope=use_envelope,
    )

    best_rrmse = float("inf")
    for lag in range(-int(max_lag_ms), int(max_lag_ms) + 1):
        ref_seg, cand_seg = _aligned_segments(reference, candidate, lag)
        if ref_seg.size == 0 or cand_seg.size == 0:
            continue
        denominator = float(np.linalg.norm(ref_seg))
        numerator = float(np.linalg.norm(ref_seg - cand_seg))
        if denominator <= np.finfo(float).eps:
            rrmse = 0.0 if numerator <= np.finfo(float).eps else 1.0
        else:
            rrmse = numerator / denominator
        best_rrmse = min(best_rrmse, float(rrmse))
    if not np.isfinite(best_rrmse):
        return 0.0
    return float(best_rrmse)


def mean_and_std_over_seeds(metric_values: Iterable[float]) -> tuple[float, float]:
    """Compute mean and standard deviation across repeated seeds."""

    values = np.asarray(list(metric_values), dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values, ddof=0))
