"""Smoke coverage for optimizer resume plumbing."""

from __future__ import annotations

import json
from types import SimpleNamespace

import scs_search.search.optimizer_cli as optimizer_common
from scs_search.config import EvaluationSummary, OptimizerRunResult, PhysicalModulationParameters


def _fake_physical_summary(theta: PhysicalModulationParameters, seeds: tuple[int, ...]) -> EvaluationSummary:
    return EvaluationSummary(
        theta=theta,
        family="physical_modulation",
        seeds=seeds,
        per_seed_records=[],
        mean_corr=0.75,
        std_corr=0.01,
        mean_raw_dose=1.0,
        std_raw_dose=0.0,
        mean_norm_dose=0.1,
        std_norm_dose=0.0,
        mean_device_cost=0.05,
        std_device_cost=0.0,
        mean_current_rate_usage=0.2,
        std_current_rate_usage=0.0,
        mean_total_current_ma=theta.I0_ma,
        std_total_current_ma=0.0,
        mean_charge_per_pulse_uc=theta.I0_ma * 0.21,
        std_charge_per_pulse_uc=0.0,
        mean_charge_rate_uc_per_s=100.0,
        std_charge_rate_uc_per_s=0.0,
        mean_relative_envelope_rmse=0.4,
        std_relative_envelope_rmse=0.0,
        penalized_objective=0.75,
        robust_objective=0.75,
        valid=True,
        invalid_reason=None,
        metadata={},
    )


def test_optimizer_cli_resume_uses_existing_seed_trials(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "cmaes"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_row = {
        "algorithm": "cmaes",
        "device_cost": 0.04,
        "eval_index": 12,
        "family": "physical_modulation",
        "mean_corr": 0.7,
        "penalized_objective": 0.7,
        "relative_envelope_rmse": 0.5,
        "seed_trials_used": 500,
        "theta_I0_ma": 5.0,
        "theta_I1_ma": 0.0,
        "theta_f0_hz": 80.0,
        "theta_f1_hz": 0.0,
        "theta_PW1_us": 0.0,
        "theta_T_ms": 500.0,
        "valid": True,
    }
    (output_dir / "history.jsonl").write_text(json.dumps(history_row) + "\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def fake_run_optimizer(run_config, output_dir_arg: str, *, resume: bool = False):
        captured["seed_trial_budget"] = run_config["optimizer"].seed_trial_budget
        captured["resume"] = resume
        theta = PhysicalModulationParameters(I0_ma=6.0, I1_ma=0.0, f0_hz=100.0, f1_hz=0.0, PW1_us=0.0, T_ms=500.0)
        summary = _fake_physical_summary(theta, (101, 202, 303))
        return OptimizerRunResult(
            algorithm="cmaes",
            output_dir=output_dir_arg,
            incumbent_theta=theta,
            incumbent_summary=summary,
            history=[history_row],
            metadata={"seed_trial_budget": run_config["optimizer"].seed_trial_budget},
        )

    monkeypatch.setitem(optimizer_common.RUNNERS, "cmaes", fake_run_optimizer)
    monkeypatch.setattr(
        optimizer_common,
        "parse_args",
        lambda _algorithm: SimpleNamespace(
            output_dir=str(output_dir),
            seed_trial_budget=700,
            resume=True,
            additional_seed_trial_budget=1000,
            supraspinal_drive_mode="aperiodic_envelope",
        ),
    )
    monkeypatch.setattr(optimizer_common, "reference_baseline_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer_common, "write_best_emg_panel", lambda **kwargs: None)
    monkeypatch.setattr(optimizer_common, "plot_best_so_far", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer_common, "plot_frontier", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer_common, "plot_frontier_overlay", lambda *args, **kwargs: None)

    optimizer_common.main("cmaes")

    assert captured["resume"] is True
    assert captured["seed_trial_budget"] == 1500
    summary_payload = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["metadata"]["seed_trial_budget"] == 1500


def test_optimizer_cli_bohb_progress_uses_full_fidelity_history(tmp_path, monkeypatch) -> None:
    output_dir = tmp_path / "bohb"
    output_dir.mkdir(parents=True, exist_ok=True)
    history = [
        {
            "algorithm": "bohb",
            "device_cost": 0.02,
            "eval_index": 1,
            "family": "physical_modulation",
            "mean_corr": 0.4,
            "penalized_objective": 0.4,
            "relative_envelope_rmse": 0.6,
            "seed_budget": 1,
            "seed_trials_used": 1,
            "theta_I0_ma": 4.0,
            "theta_I1_ma": 0.0,
            "theta_f0_hz": 80.0,
            "theta_f1_hz": 0.0,
            "theta_PW1_us": 0.0,
            "theta_T_ms": 500.0,
            "valid": True,
        },
        {
            "algorithm": "bohb",
            "device_cost": 0.05,
            "eval_index": 2,
            "family": "physical_modulation",
            "mean_corr": 0.7,
            "penalized_objective": 0.7,
            "relative_envelope_rmse": 0.4,
            "seed_budget": 3,
            "seed_trials_used": 4,
            "theta_I0_ma": 5.0,
            "theta_I1_ma": 0.0,
            "theta_f0_hz": 120.0,
            "theta_f1_hz": 0.0,
            "theta_PW1_us": 0.0,
            "theta_T_ms": 500.0,
            "valid": True,
        },
    ]

    captured: dict[str, object] = {}

    def fake_run_optimizer(_run_config, output_dir_arg: str, *, resume: bool = False):
        theta = PhysicalModulationParameters(I0_ma=5.0, I1_ma=0.0, f0_hz=120.0, f1_hz=0.0, PW1_us=0.0, T_ms=500.0)
        summary = _fake_physical_summary(theta, (101, 202, 303))
        return OptimizerRunResult(
            algorithm="bohb",
            output_dir=output_dir_arg,
            incumbent_theta=theta,
            incumbent_summary=summary,
            history=history,
            metadata={"seed_trial_budget": 4},
        )

    def fake_plot_best_so_far(trace_map, *_args, **_kwargs):
        captured["trace_map"] = trace_map

    monkeypatch.setitem(optimizer_common.RUNNERS, "bohb", fake_run_optimizer)
    monkeypatch.setattr(
        optimizer_common,
        "parse_args",
        lambda _algorithm: SimpleNamespace(
            output_dir=str(output_dir),
            seed_trial_budget=4,
            resume=False,
            additional_seed_trial_budget=None,
            supraspinal_drive_mode="aperiodic_envelope",
        ),
    )
    monkeypatch.setattr(optimizer_common, "reference_baseline_stats", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer_common, "write_best_emg_panel", lambda **kwargs: None)
    monkeypatch.setattr(optimizer_common, "plot_best_so_far", fake_plot_best_so_far)
    monkeypatch.setattr(optimizer_common, "plot_frontier", lambda *args, **kwargs: None)
    monkeypatch.setattr(optimizer_common, "plot_frontier_overlay", lambda *args, **kwargs: None)

    optimizer_common.main("bohb")

    trace_map = captured["trace_map"]
    trace = next(iter(trace_map.values()))
    assert len(trace) == 1
    assert trace[0]["seed_trials_used"] == 4.0
