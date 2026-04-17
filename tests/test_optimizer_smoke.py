"""Smoke coverage for optimizer output files with the canonical theta schema."""

from __future__ import annotations

from scs_search.config import EvaluationSummary, PhysicalModulationParameters
from scs_search.search.optimizer_history import history_entry


def _fake_summary(theta: PhysicalModulationParameters, seeds: tuple[int, ...]) -> EvaluationSummary:
    device_cost = min(1.0, float(theta.I0_ma) * float(theta.f0_hz) * 210.0 / (20.0 * 400.0 * 600.0))
    current_rate_usage = min(1.0, float(theta.I0_ma) * float(theta.f0_hz) / (20.0 * 400.0))
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
        mean_current_rate_usage=current_rate_usage,
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
        metadata={"pulse_width_us": 210.0, "usage_metric": "normalized_charge_rate_usage"},
    )


def test_history_entry_uses_canonical_theta_columns() -> None:
    theta = PhysicalModulationParameters(I0_ma=8.0, I1_ma=0.0, f0_hz=60.0, f1_hz=0.0, PW1_us=0.0, T_ms=500.0)
    summary = _fake_summary(theta, (101, 202, 303))
    row = history_entry(summary, algorithm="bohb", eval_index=1, seed_trials_used=3)

    assert "device_cost" in row
    assert "theta_I0_ma" in row
    assert "theta_T_ms" in row
    assert "theta_PW1_us" in row
