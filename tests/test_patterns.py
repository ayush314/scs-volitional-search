"""Tests for stimulation pattern generation."""

from __future__ import annotations

import numpy as np

from scs_search.config import DeviceConfig, PatternParameters, default_theta_bounds
from scs_search.patterns import (
    generate_duty_cycled_constant_pattern,
    generate_stim_pattern,
    generate_tonic_pattern,
    invalid_theta_reason,
)


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
        pw_us=210.0,
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
        pw_us=210.0,
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


def test_pulse_width_and_current_arrays_come_from_theta() -> None:
    device = DeviceConfig()
    theta = PatternParameters(
        f=50.0,
        pw_us=500.0,
        T_on=100.0,
        T_off=0.0,
        alpha0=0.5,
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )
    pattern = generate_stim_pattern(theta, t_end_ms=200, device_config=device)
    assert pattern.metadata["pulse_width_us"] == theta.pw_us
    assert np.allclose(pattern.pulse_widths_us, theta.pw_us)
    assert np.allclose(pattern.pulse_current_ma, pattern.pulse_alpha * device.max_total_current_ma)


def test_pulse_scheduling_requires_full_pulse_to_fit_inside_on_window() -> None:
    theta = PatternParameters(
        f=1000.0,
        pw_us=800.0,
        T_on=2.5,
        T_off=2.5,
        alpha0=0.5,
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )
    pattern = generate_stim_pattern(theta, t_end_ms=5)
    assert np.allclose(pattern.pulse_times_ms, np.array([0.0, 1.0]))


def test_overlap_invalid_combination_is_rejected() -> None:
    theta = PatternParameters(
        f=1200.0,
        pw_us=1000.0,
        T_on=100.0,
        T_off=0.0,
        alpha0=0.5,
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )
    device = DeviceConfig(
        max_total_current_ma=20.0,
        min_pulse_width_us=60.0,
        max_pulse_width_us=1000.0,
        pulse_width_step_us=10.0,
        max_master_rate_hz=1200.0,
        default_pulse_width_us=210.0,
    )
    assert invalid_theta_reason(theta, device) == "pulse_overlap"


def test_pulse_width_round_trips_through_bounds_encoding() -> None:
    bounds = default_theta_bounds()
    theta = PatternParameters(
        f=321.0,
        pw_us=540.0,
        T_on=180.0,
        T_off=75.0,
        alpha0=0.4,
        alpha1=0.2,
        phi1=1.0,
        alpha2=0.1,
        phi2=2.0,
    )
    encoded = bounds.encode_unit(theta)
    decoded = bounds.decode_unit(encoded)
    assert np.isclose(decoded.pw_us, theta.pw_us)
    assert decoded.to_dict()["pw_us"] == theta.pw_us
