"""Tests for hardware-aware stimulation cost."""

from __future__ import annotations

import numpy as np

from scs_search.config import DeviceConfig, DoseConfig
from scs_search.dose import combined_objective, compute_pattern_dose, raw_dose_from_pulse_recruitment
from scs_search.stimulation.patterns import generate_tonic_pattern


def test_raw_dose_is_sum_of_realized_recruitment() -> None:
    pulse_recruitment_fraction = np.array([0.2, 0.4, 0.6])
    assert np.isclose(raw_dose_from_pulse_recruitment(pulse_recruitment_fraction), 1.2)


def test_device_cost_reaches_one_at_max_valid_charge_rate_corner() -> None:
    dose_config = DoseConfig(max_frequency_hz=1200.0)
    device_config = DeviceConfig(
        max_total_current_ma=100.0,
        min_pulse_width_us=60.0,
        max_pulse_width_us=1000.0,
        pulse_width_step_us=10.0,
        max_master_rate_hz=1200.0,
        default_pulse_width_us=1000.0,
    )
    pattern = generate_tonic_pattern(
        freq_hz=1000.0,
        alpha=1.0,
        pw_us=1000.0,
        t_end_ms=1000,
        device_config=device_config,
    )
    metrics = compute_pattern_dose(
        pattern,
        pulse_recruitment_fraction=np.ones_like(pattern.pulse_alpha),
        dose_config=dose_config,
        device_config=device_config,
    )
    assert np.isclose(metrics["raw_recruitment_dose"], 1000.0)
    assert np.isclose(metrics["device_cost"], 1.0)
    assert np.isclose(metrics["mean_total_current_ma"], 100.0)
    assert np.isclose(metrics["mean_charge_per_pulse_uc"], 100.0)


def test_charge_metrics_scale_with_pulse_width() -> None:
    dose_config = DoseConfig(max_frequency_hz=1200.0)
    device_config = DeviceConfig()
    short_pattern = generate_tonic_pattern(
        freq_hz=200.0,
        alpha=0.5,
        pw_us=100.0,
        t_end_ms=1000,
        device_config=device_config,
    )
    long_pattern = generate_tonic_pattern(
        freq_hz=200.0,
        alpha=0.5,
        pw_us=500.0,
        t_end_ms=1000,
        device_config=device_config,
    )

    short_metrics = compute_pattern_dose(
        short_pattern,
        pulse_recruitment_fraction=np.zeros_like(short_pattern.pulse_alpha),
        dose_config=dose_config,
        device_config=device_config,
    )
    long_metrics = compute_pattern_dose(
        long_pattern,
        pulse_recruitment_fraction=np.zeros_like(long_pattern.pulse_alpha),
        dose_config=dose_config,
        device_config=device_config,
    )

    assert np.isclose(long_metrics["mean_charge_per_pulse_uc"], 5.0 * short_metrics["mean_charge_per_pulse_uc"])
    assert long_metrics["charge_rate_uc_per_s"] > short_metrics["charge_rate_uc_per_s"]
    assert long_metrics["device_cost"] > short_metrics["device_cost"]
    assert np.isclose(long_metrics["current_rate_usage"], short_metrics["current_rate_usage"])


def test_recruitment_dose_uses_realized_recruitment_not_pulse_alpha() -> None:
    dose_config = DoseConfig()
    device_config = DeviceConfig()
    pattern = generate_tonic_pattern(freq_hz=100.0, alpha=1.0, pw_us=200.0, t_end_ms=1000, device_config=device_config)

    zero_metrics = compute_pattern_dose(
        pattern,
        pulse_recruitment_fraction=np.zeros_like(pattern.pulse_alpha),
        dose_config=dose_config,
        device_config=device_config,
    )
    full_metrics = compute_pattern_dose(
        pattern,
        pulse_recruitment_fraction=np.ones_like(pattern.pulse_alpha),
        dose_config=dose_config,
        device_config=device_config,
    )

    assert np.isclose(zero_metrics["raw_recruitment_dose"], 0.0)
    assert np.isclose(full_metrics["raw_recruitment_dose"], float(len(pattern.pulse_alpha)))


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
