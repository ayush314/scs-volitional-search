"""Smoke coverage for the physical-modulation sweep suite."""

from __future__ import annotations

from scs_search.config import EvaluationSummary, PhysicalModulationParameters
from scs_search.reporting.plotting import plot_frontier
from scs_search.search.sweep import make_physical_modulation_simulation_config, run_physical_modulation_sweep_suite


def _fake_summary(theta: PhysicalModulationParameters, seeds: tuple[int, ...]) -> EvaluationSummary:
    device_cost = min(1.0, float(theta.I0_ma) * float(theta.f0_hz) * 210.0 / (20.0 * 400.0 * 600.0))
    corr = 1.0 - device_cost
    return EvaluationSummary(
        theta=theta,
        family="physical_modulation",
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
        mean_current_rate_usage=device_cost,
        std_current_rate_usage=0.0,
        mean_total_current_ma=float(theta.I0_ma),
        std_total_current_ma=0.0,
        mean_charge_per_pulse_uc=float(theta.I0_ma) * 0.21,
        std_charge_per_pulse_uc=0.0,
        mean_charge_rate_uc_per_s=device_cost * 4800.0,
        std_charge_rate_uc_per_s=0.0,
        mean_relative_envelope_rmse=1.0 - corr,
        std_relative_envelope_rmse=0.0,
        penalized_objective=corr,
        robust_objective=corr,
        valid=True,
        invalid_reason=None,
        metadata={},
    )


def test_physical_sweep_suite_uses_canonical_theta_columns(tmp_path, monkeypatch) -> None:
    config = make_physical_modulation_simulation_config()

    monkeypatch.setattr(
        "scs_search.search.sweep.evaluate_pattern",
        lambda theta, seeds, config, budget_norm=None, reference_emg_by_seed=None: _fake_summary(theta, tuple(seeds)),
    )

    results = run_physical_modulation_sweep_suite(config, config.seed_config.train_seeds, seed_trial_budget=1000)

    assert results["all"]
    assert all("theta_I0_ma" in record for record in results["all"])
    assert all("theta_T_ms" in record for record in results["all"])

    output_path = tmp_path / "frontier.png"
    plot_frontier(results["all"], results["frontier"], output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0
