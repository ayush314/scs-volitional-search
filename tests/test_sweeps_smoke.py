"""Smoke coverage for the sweep pipeline with hardware-aware cost."""

from __future__ import annotations

import numpy as np

from scs_search.config import EvaluationSummary, PatternParameters, SimulationConfig
from scs_search.plotting import plot_frontier
from scs_search.sweeps import run_sweep_suite, sweep_grid_values


def _fake_summary(theta: PatternParameters, seeds: tuple[int, ...]) -> EvaluationSummary:
    pulse_width_us = 210.0
    device_cost = min(1.0, float(theta.alpha0) * float(theta.f) / 1200.0)
    corr = 1.0 - 0.5 * device_cost
    return EvaluationSummary(
        theta=theta,
        family="fake",
        seeds=seeds,
        per_seed_records=[],
        mean_corr=corr,
        std_corr=0.0,
        mean_raw_dose=0.0,
        std_raw_dose=0.0,
        mean_norm_dose=0.0,
        std_norm_dose=0.0,
        mean_device_cost=device_cost,
        std_device_cost=0.0,
        mean_total_current_ma=50.0 * float(theta.alpha0),
        std_total_current_ma=0.0,
        mean_charge_per_pulse_uc=float(theta.alpha0) * pulse_width_us / 10.0,
        std_charge_per_pulse_uc=0.0,
        mean_charge_rate_uc_per_s=device_cost * 120000.0,
        std_charge_rate_uc_per_s=0.0,
        penalized_objective=corr,
        robust_objective=corr,
        metadata={"pulse_width_us": pulse_width_us},
    )


def test_run_sweep_suite_emits_device_cost_records(tmp_path, monkeypatch) -> None:
    config = SimulationConfig()

    monkeypatch.setattr(
        "scs_search.sweeps.sweep_grid_values",
        lambda seed_trial_budget=100, seed_count=1: {
            "tonic_freqs": np.asarray([10.0, 20.0]),
            "tonic_alpha": np.asarray([0.2, 0.4]),
            "duty_freqs": np.asarray([15.0]),
            "duty_cycle": np.asarray([0.5]),
            "full_theta_samples": np.asarray([2]),
        },
    )
    monkeypatch.setattr(
        "scs_search.sweeps.evaluate_pattern",
        lambda theta, seeds, config, **_: _fake_summary(theta, tuple(int(seed) for seed in seeds)),
    )

    results = run_sweep_suite(config, seeds=(101,), seed_trial_budget=7, reference_emg_by_seed={101: np.zeros(8)})

    assert results["all"]
    assert all("device_cost" in record for record in results["all"])
    assert all("theta_T_on" in record for record in results["all"])
    assert results["preset"]["seed_trial_budget"] == 7

    output_path = tmp_path / "frontier.png"
    plot_frontier(results["all"], results["frontier"], output_path)
    assert output_path.exists()


def test_sweep_grid_values_allocates_from_seed_trial_budget() -> None:
    preset = sweep_grid_values(seed_trial_budget=100, seed_count=1)

    assert len(preset["tonic_freqs"]) * len(preset["tonic_alpha"]) == 20
    assert len(preset["duty_freqs"]) * len(preset["duty_cycle"]) == 20
    assert int(preset["full_theta_samples"][0]) == 60


def test_sweep_grid_values_stays_balanced_for_three_seed_budget() -> None:
    preset = sweep_grid_values(seed_trial_budget=700, seed_count=3)

    assert len(preset["tonic_freqs"]) * len(preset["tonic_alpha"]) == 48
    assert len(preset["duty_freqs"]) * len(preset["duty_cycle"]) == 48
    assert int(preset["full_theta_samples"][0]) == 137
