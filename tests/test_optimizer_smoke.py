"""Smoke coverage for optimizer output files with hardware-aware cost."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from scs_search.config import EvaluationSummary, OptimizerRunResult, PatternParameters
from scs_search.optimizers import history_entry


_RUN_BOHB_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_bohb.py"
_RUN_BOHB_SPEC = importlib.util.spec_from_file_location("run_bohb_script", _RUN_BOHB_PATH)
assert _RUN_BOHB_SPEC is not None and _RUN_BOHB_SPEC.loader is not None
run_bohb = importlib.util.module_from_spec(_RUN_BOHB_SPEC)
_RUN_BOHB_SPEC.loader.exec_module(run_bohb)


def _fake_summary(theta: PatternParameters, seeds: tuple[int, ...]) -> EvaluationSummary:
    device_cost = min(1.0, float(theta.alpha0) * float(theta.f) * float(theta.pulse_width_us) / (1200.0 * 1000.0))
    corr = 1.0 - device_cost
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
        mean_total_current_ma=100.0 * float(theta.alpha0),
        std_total_current_ma=0.0,
        mean_charge_per_pulse_uc=float(theta.alpha0) * float(theta.pulse_width_us) / 10.0,
        std_charge_per_pulse_uc=0.0,
        mean_charge_rate_uc_per_s=device_cost * 120000.0,
        std_charge_rate_uc_per_s=0.0,
        penalized_objective=corr,
        robust_objective=corr,
        feasible_by_budget={"1.0": True},
        metadata={"pulse_width_us": float(theta.pulse_width_us)},
    )


def test_run_bohb_writes_device_cost_outputs(tmp_path, monkeypatch) -> None:
    theta = PatternParameters(
        f=40.0,
        pulse_width_us=300.0,
        T_on=100.0,
        T_off=50.0,
        alpha0=0.5,
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )
    summary = _fake_summary(theta, (1001, 1002, 1003))
    history = [history_entry(summary, algorithm="bohb", eval_index=1, seed_trials_used=3)]

    def fake_run_optimizer(_run_config, output_dir: str) -> OptimizerRunResult:
        return OptimizerRunResult(
            algorithm="bohb",
            output_dir=output_dir,
            incumbent_theta=theta,
            incumbent_summary=summary,
            history=history,
            metadata={"seed_trial_budget": 3},
        )

    output_dir = tmp_path / "bohb"
    monkeypatch.setattr(run_bohb, "parse_args", lambda: SimpleNamespace(output_dir=str(output_dir), seed_trial_budget=3))
    monkeypatch.setattr(run_bohb, "run_optimizer", fake_run_optimizer)
    monkeypatch.setattr(run_bohb, "reference_baseline_stats", lambda *args, **kwargs: None)

    run_bohb.main()

    with (output_dir / "metrics.jsonl").open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    assert rows
    assert "device_cost" in rows[0]
    assert "theta_pulse_width_us" in rows[0]
    assert (output_dir / "device_budget_vs_corr.png").exists()
