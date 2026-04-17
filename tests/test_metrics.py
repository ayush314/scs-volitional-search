"""Tests for EMG similarity metrics."""

from __future__ import annotations

import numpy as np

from scs_search.metrics import compute_emg_similarity, relative_envelope_rmse


def test_identical_emg_has_unit_correlation() -> None:
    signal = np.sin(np.linspace(0.0, 4.0 * np.pi, 200))
    score = compute_emg_similarity(signal, signal, use_envelope=False)
    assert np.isclose(score, 1.0)


def test_degraded_signal_has_lower_correlation() -> None:
    reference = np.sin(np.linspace(0.0, 4.0 * np.pi, 200))
    candidate = np.sin(np.linspace(0.0, 4.0 * np.pi, 200) + 0.8)
    score = compute_emg_similarity(reference, candidate, use_envelope=False)
    assert score < 0.95


def test_lag_search_recovers_shifted_signal() -> None:
    reference = np.sin(np.linspace(0.0, 2.0 * np.pi, 100))
    candidate = np.roll(reference, 3)
    no_lag = compute_emg_similarity(reference, candidate, use_envelope=False, max_lag_ms=0)
    lag = compute_emg_similarity(reference, candidate, use_envelope=False, max_lag_ms=3)
    assert lag >= no_lag


def test_relative_envelope_rmse_is_zero_for_identical_signal() -> None:
    signal = np.sin(np.linspace(0.0, 4.0 * np.pi, 200))
    score = relative_envelope_rmse(signal, signal, use_envelope=False)
    assert np.isclose(score, 0.0)


def test_relative_envelope_rmse_penalizes_degraded_signal() -> None:
    reference = np.sin(np.linspace(0.0, 4.0 * np.pi, 200))
    candidate = 0.5 * reference
    score = relative_envelope_rmse(reference, candidate, use_envelope=False)
    assert score > 0.0
