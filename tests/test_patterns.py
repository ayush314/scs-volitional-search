"""Tests for stimulation pattern generation."""

from __future__ import annotations

import numpy as np

from scs_search.config import PatternParameters
from scs_search.patterns import generate_duty_cycled_constant_pattern, generate_stim_pattern, generate_tonic_pattern


def test_tonic_pattern_pulse_count_matches_frequency() -> None:
    pattern = generate_tonic_pattern(freq_hz=20.0, alpha=0.5, t_end_ms=1000)
    assert len(pattern.pulse_times_ms) == 20
    assert np.allclose(pattern.pulse_alpha, 0.5)


def test_duty_cycle_turns_stimulation_off_between_windows() -> None:
    pattern = generate_duty_cycled_constant_pattern(
        freq_hz=20.0,
        alpha=0.5,
        duty_cycle=0.5,
        t_end_ms=1000,
        cycle_ms=200.0,
    )
    assert np.all(pattern.alpha_t[100:200] == 0.0)
    assert np.all(pattern.alpha_t[:100] == 0.5)


def test_fourier_pattern_clips_envelope() -> None:
    theta = PatternParameters(
        f=50.0,
        T_on=100.0,
        T_off=100.0,
        alpha0=0.8,
        alpha1=0.5,
        phi1=0.0,
        alpha2=0.5,
        phi2=0.0,
    )
    pattern = generate_stim_pattern(theta, t_end_ms=400)
    assert np.min(pattern.alpha_t) >= 0.0
    assert np.max(pattern.alpha_t) <= 1.0


def test_pulse_phase_resets_each_on_window() -> None:
    theta = PatternParameters(
        f=10.0,
        T_on=100.0,
        T_off=100.0,
        alpha0=0.5,
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )
    pattern = generate_stim_pattern(theta, t_end_ms=450)
    assert np.allclose(pattern.pulse_times_ms, np.array([0.0, 200.0, 400.0]))
