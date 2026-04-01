"""Tests for hardware-aware stimulation cost."""

from __future__ import annotations

import numpy as np

from scs_search.config import DeviceConfig, DoseConfig
from scs_search.dose import combined_objective, compute_pattern_dose, raw_dose_from_pulse_alpha
from scs_search.patterns import generate_tonic_pattern


def test_raw_dose_is_sum_of_pulse_alpha() -> None:
    pulse_alpha = np.array([0.2, 0.4, 0.6])
    assert np.isclose(raw_dose_from_pulse_alpha(pulse_alpha), 1.2)


def test_device_cost_matches_full_current_rate_reference() -> None:
    dose_config = DoseConfig(max_frequency_hz=1200.0)
    device_config = DeviceConfig(
        max_total_current_ma=50.0,
        min_pulse_width_us=60.0,
        max_pulse_width_us=1000.0,
        pulse_width_step_us=10.0,
        max_master_rate_hz=1200.0,
        fixed_pulse_width_us=1000.0,
    )
    pattern = generate_tonic_pattern(
        freq_hz=1200.0,
        alpha=1.0,
        t_end_ms=1000,
        device_config=device_config,
    )
    metrics = compute_pattern_dose(pattern, dose_config, device_config)
    assert np.isclose(metrics["raw_recruitment_dose"], 1200.0)
    assert np.isclose(metrics["device_cost"], 1.0)
    assert np.isclose(metrics["mean_total_current_ma"], 50.0)


def test_device_cost_does_not_change_with_fixed_pulse_width_choice() -> None:
    dose_config = DoseConfig(max_frequency_hz=1200.0)
    device_config = DeviceConfig(fixed_pulse_width_us=500.0)
    pattern = generate_tonic_pattern(
        freq_hz=1200.0,
        alpha=1.0,
        t_end_ms=1000,
        device_config=device_config,
    )
    metrics = compute_pattern_dose(pattern, dose_config, device_config)
    assert np.isclose(metrics["device_cost"], 1.0)
    assert np.isclose(metrics["mean_charge_per_pulse_uc"], 25.0)


def test_combined_objective_penalizes_budget_violation() -> None:
    robust_score, penalized = combined_objective(
        mean_corr=0.8,
        std_corr=0.1,
        device_cost=1.2,
        budget_norm=1.0,
        dose_config=DoseConfig(objective_penalty_weight=10.0),
        robust=False,
    )
    assert np.isclose(robust_score, 0.8)
    assert penalized < robust_score
