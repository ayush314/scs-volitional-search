"""Tests for the physical shared-wave modulation search space."""

from __future__ import annotations

import numpy as np

from scs_search.config import PhysicalModulationBounds, PhysicalModulationParameters
from scs_search.search.sweep import (
    make_physical_modulation_simulation_config,
    physical_modulation_sweep_values,
)
from scs_search.simulation.evaluator import evaluate_pattern
from scs_search.stimulation.patterns import generate_stim_pattern


def test_physical_modulation_bounds_round_trip_and_enforce_caps() -> None:
    config = make_physical_modulation_simulation_config()
    bounds = config.theta_bounds

    theta = bounds.clip(
        {
            "I0_ma": 8.0,
            "I1_ma": 20.0,
            "f0_hz": 120.0,
            "f1_hz": 400.0,
            "PW1_us": 200.0,
            "T_ms": 500.0,
        },
        device_config=config.device_config,
    )
    encoded = bounds.encode_unit(theta, device_config=config.device_config)
    decoded = bounds.decode_unit(encoded, device_config=config.device_config)

    assert tuple(bounds.names) == tuple(PhysicalModulationBounds().names)
    assert np.isclose(theta.I1_ma, 8.0)
    assert np.isclose(theta.f1_hz, 110.0)
    assert np.isclose(theta.PW1_us, 150.0)
    assert np.isclose(decoded.I0_ma, theta.I0_ma)
    assert np.isclose(decoded.I1_ma, theta.I1_ma)
    assert np.isclose(decoded.f0_hz, theta.f0_hz)
    assert np.isclose(decoded.f1_hz, theta.f1_hz)
    assert np.isclose(decoded.PW1_us, theta.PW1_us)
    assert np.isclose(decoded.T_ms, theta.T_ms)


def test_constant_physical_controls_reduce_to_regular_pulse_train() -> None:
    theta = PhysicalModulationParameters(
        I0_ma=8.0,
        I1_ma=0.0,
        f0_hz=100.0,
        f1_hz=0.0,
        PW1_us=0.0,
        T_ms=500.0,
    )

    pattern = generate_stim_pattern(theta, t_end_ms=35)

    assert np.allclose(pattern.pulse_times_ms, np.array([0.0, 10.0, 20.0, 30.0]))
    assert np.allclose(pattern.pulse_current_ma, 8.0)
    assert np.allclose(pattern.pulse_widths_us, 210.0)
    assert np.allclose(pattern.pulse_alpha, 8.0 / 20.0)


def test_physical_modulation_varies_realized_pulse_widths() -> None:
    theta = PhysicalModulationParameters(
        I0_ma=8.0,
        I1_ma=2.0,
        f0_hz=60.0,
        f1_hz=20.0,
        PW1_us=100.0,
        T_ms=200.0,
    )

    pattern = generate_stim_pattern(theta, t_end_ms=250)

    assert pattern.pulse_widths_us.size > 1
    assert np.max(pattern.pulse_widths_us) > np.min(pattern.pulse_widths_us)
    assert np.max(pattern.pulse_current_ma) > np.min(pattern.pulse_current_ma)


def test_out_of_range_physical_candidate_is_clipped_to_valid_caps() -> None:
    config = make_physical_modulation_simulation_config()
    theta = PhysicalModulationParameters(
        I0_ma=4.0,
        I1_ma=6.0,
        f0_hz=60.0,
        f1_hz=10.0,
        PW1_us=50.0,
        T_ms=300.0,
    )

    summary = evaluate_pattern(theta, seeds=(101, 202), config=config)

    assert summary.valid is True
    assert summary.invalid_reason is None
    assert np.isclose(summary.theta.I1_ma, 4.0)


def test_physical_modulation_sweep_budget_matches_candidate_count() -> None:
    preset = physical_modulation_sweep_values(seed_trial_budget=1000, seed_count=3)

    tonic_count = len(preset["tonic_f0_hz"]) * len(preset["tonic_I0_ma"])
    lhs_count = int(preset["lhs_count"])

    assert tonic_count == 30
    assert tonic_count + lhs_count == 333
