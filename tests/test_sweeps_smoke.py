"""Smoke coverage for the sweep pipeline with hardware-aware cost."""

from __future__ import annotations

import numpy as np

from scs_search.config import EvaluationSummary, PatternParameters, SimulationConfig
from scs_search.plotting import plot_frontier
from scs_search.sweeps import run_sweep_suite, sweep_grid_values


def _fake_summary(theta: PatternParameters, seeds: tuple[int, ...]) -> EvaluationSummary:
    device_cost = min(1.0, float(theta.alpha0) * float(theta.f) * float(theta.pw_us) / (400.0 * 600.0))
    current_rate_usage = min(1.0, float(theta.alpha0) * float(theta.f) / 400.0)
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
        mean_current_rate_usage=current_rate_usage,
        std_current_rate_usage=0.0,
        mean_total_current_ma=20.0 * float(theta.alpha0),
        std_total_current_ma=0.0,
        mean_charge_per_pulse_uc=20.0 * float(theta.alpha0) * float(theta.pw_us) / 1000.0,
        std_charge_per_pulse_uc=0.0,
        mean_charge_rate_uc_per_s=device_cost * 4800.0,
        std_charge_rate_uc_per_s=0.0,
        penalized_objective=corr,
        robust_objective=corr,
        valid=True,
        invalid_reason=None,
        metadata={"pulse_width_us": float(theta.pw_us), "usage_metric": "normalized_charge_rate_usage"},
    )


def test_run_sweep_suite_emits_device_cost_records(tmp_path, monkeypatch) -> None:
    config = SimulationConfig()

    monkeypatch.setattr(
        "scs_search.sweeps.sweep_grid_values",
        lambda seed_trial_budget=100, seed_count=1: {
            "tonic_freqs": np.asarray([10.0, 20.0]),
            "tonic_alpha": np.asarray([0.2, 0.4]),
            "tonic_pw_us": np.asarray([60.0, 210.0]),
            "duty_freqs": np.asarray([15.0]),
            "duty_cycle": np.asarray([0.5]),
            "duty_pw_us": np.asarray([60.0, 600.0]),
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
    assert all("theta_pw_us" in record for record in results["all"])
    assert results["preset"]["seed_trial_budget"] == 7

    output_path = tmp_path / "frontier.png"
    plot_frontier(results["all"], results["frontier"], output_path)
    assert output_path.exists()


def test_sweep_grid_values_allocates_from_seed_trial_budget() -> None:
    preset = sweep_grid_values(seed_trial_budget=100, seed_count=1)

    assert len(preset["tonic_freqs"]) * len(preset["tonic_alpha"]) * len(preset["tonic_pw_us"]) == 18
    assert len(preset["duty_freqs"]) * len(preset["duty_cycle"]) * len(preset["duty_pw_us"]) == 18
    assert int(preset["full_theta_samples"][0]) == 64


def test_sweep_grid_values_stays_balanced_for_three_seed_budget() -> None:
    preset = sweep_grid_values(seed_trial_budget=700, seed_count=3)

    assert len(preset["tonic_freqs"]) * len(preset["tonic_alpha"]) * len(preset["tonic_pw_us"]) == 45
    assert len(preset["duty_freqs"]) * len(preset["duty_cycle"]) * len(preset["duty_pw_us"]) == 45
    assert int(preset["full_theta_samples"][0]) == 143
