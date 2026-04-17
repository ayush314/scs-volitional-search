"""Tests for physical-modulation pattern generation."""

from __future__ import annotations

import numpy as np

from scs_search.config import DeviceConfig, PhysicalModulationParameters, SimulationConfig
from scs_search.stimulation.patterns import generate_stim_pattern, generate_tonic_pattern, invalid_theta_reason


def test_tonic_pattern_pulse_count_matches_frequency() -> None:
    pattern = generate_tonic_pattern(freq_hz=20.0, alpha=0.5, t_end_ms=1000)

    assert len(pattern.pulse_times_ms) == 20
    assert np.allclose(pattern.pulse_alpha, 0.5)


def test_physical_pattern_generates_variable_spacing_height_and_width() -> None:
    theta = PhysicalModulationParameters(
        I0_ma=8.0,
        I1_ma=2.0,
        f0_hz=60.0,
        f1_hz=20.0,
        PW1_us=60.0,
        T_ms=250.0,
    )
    pattern = generate_stim_pattern(theta, t_end_ms=400)

    assert pattern.pulse_times_ms.size > 3
    assert np.ptp(np.diff(pattern.pulse_times_ms)) > 0.0
    assert np.ptp(pattern.pulse_current_ma) > 0.0
    assert np.ptp(pattern.pulse_widths_us) > 0.0


def test_invalid_theta_reports_pulse_width_modulation_cap_violation() -> None:
    device = DeviceConfig(
        max_total_current_ma=20.0,
        min_pulse_width_us=60.0,
        max_pulse_width_us=1000.0,
        pulse_width_step_us=10.0,
        max_master_rate_hz=1200.0,
        default_pulse_width_us=210.0,
    )
    theta = PhysicalModulationParameters(
        I0_ma=10.0,
        I1_ma=0.0,
        f0_hz=1200.0,
        f1_hz=0.0,
        PW1_us=200.0,
        T_ms=200.0,
    )

    assert invalid_theta_reason(theta, device) == "pulse_width_modulation_out_of_range"


def test_theta_round_trips_through_bounds_encoding() -> None:
    config = SimulationConfig()
    theta = PhysicalModulationParameters(
        I0_ma=8.0,
        I1_ma=4.0,
        f0_hz=150.0,
        f1_hz=80.0,
        PW1_us=90.0,
        T_ms=420.0,
    )

    encoded = config.theta_bounds.encode_unit(theta, device_config=config.device_config)
    decoded = config.theta_bounds.decode_unit(encoded, device_config=config.device_config)

    assert np.isclose(decoded.I0_ma, theta.I0_ma)
    assert np.isclose(decoded.I1_ma, theta.I1_ma)
    assert np.isclose(decoded.f0_hz, theta.f0_hz)
    assert np.isclose(decoded.f1_hz, theta.f1_hz)
    assert np.isclose(decoded.PW1_us, theta.PW1_us)
    assert np.isclose(decoded.T_ms, theta.T_ms)
