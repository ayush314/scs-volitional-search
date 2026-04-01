"""EMG restoration metrics and seed-level aggregation helpers."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

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


def compute_emg_similarity(
    reference_emg: np.ndarray,
    candidate_emg: np.ndarray,
    envelope_window_ms: int = 25,
    max_lag_ms: int = 0,
    use_envelope: bool = True,
) -> float:
    """Return the best aligned correlation between reference and candidate EMG."""

    reference = np.asarray(reference_emg, dtype=float)
    candidate = np.asarray(candidate_emg, dtype=float)
    if use_envelope:
        reference = emg_envelope(reference, window_ms=envelope_window_ms)
        candidate = emg_envelope(candidate, window_ms=envelope_window_ms)
    best = -1.0
    for lag in range(-int(max_lag_ms), int(max_lag_ms) + 1):
        ref_seg, cand_seg = _aligned_segments(reference, candidate, lag)
        if ref_seg.size < 2 or cand_seg.size < 2:
            continue
        best = max(best, pearson_correlation(ref_seg, cand_seg))
    return float(best)


def mean_and_std_over_seeds(metric_values: Iterable[float]) -> tuple[float, float]:
    """Compute mean and standard deviation across repeated seeds."""

    values = np.asarray(list(metric_values), dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.std(values, ddof=0))


def best_feasible_under_budget(records: Iterable[Mapping[str, Any]], budget_norm: float, score_key: str = "mean_corr") -> Mapping[str, Any] | None:
    """Return the best feasible record from a sweep or optimizer history."""

    feasible = [record for record in records if float(record["device_cost"]) <= float(budget_norm)]
    if not feasible:
        return None
    return max(feasible, key=lambda record: float(record.get(score_key, -np.inf)))
