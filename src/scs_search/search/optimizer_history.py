"""Shared optimizer history helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..config import (
    EvaluationSummary,
    OptimizerConfig,
    OptimizerRunResult,
    PhysicalModulationParameters,
    SimulationConfig,
    THETA_NAMES,
)
from ..reporting.analysis import summary_to_record
from ..utils import read_jsonl


def history_entry(
    summary: EvaluationSummary,
    *,
    algorithm: str,
    eval_index: int,
    seed_trials_used: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert one evaluation summary into a history row."""

    record = summary_to_record(summary)
    record.update(
        {
            "algorithm": algorithm,
            "eval_index": int(eval_index),
            "seed_trials_used": int(seed_trials_used),
        }
    )
    if extra:
        record.update(dict(extra))
    return record


def final_run_result(
    *,
    algorithm: str,
    output_dir: str,
    incumbent_theta: Any,
    incumbent_summary: EvaluationSummary,
    history: list[dict[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> OptimizerRunResult:
    """Build a consistent optimizer result object."""

    return OptimizerRunResult(
        algorithm=algorithm,
        output_dir=output_dir,
        incumbent_theta=incumbent_theta,
        incumbent_summary=incumbent_summary,
        history=history,
        metadata=dict(metadata or {}),
    )


def theta_from_history_record(record: Mapping[str, Any]) -> PhysicalModulationParameters:
    """Reconstruct one theta dataclass from a flat optimizer history row."""

    theta_values = {key.removeprefix("theta_"): value for key, value in record.items() if str(key).startswith("theta_")}
    if not all(name in theta_values for name in THETA_NAMES):
        raise KeyError("History record does not contain a complete theta representation.")
    return PhysicalModulationParameters.from_any(theta_values)


def summary_from_history_record(
    record: Mapping[str, Any],
    *,
    theta: PhysicalModulationParameters,
    seeds: tuple[int, ...],
) -> EvaluationSummary:
    """Reconstruct one aggregate evaluation summary from a flat history row."""

    return EvaluationSummary(
        theta=theta,
        family=str(record.get("family", "unknown")),
        seeds=tuple(int(seed) for seed in seeds),
        per_seed_records=[],
        mean_corr=float(record.get("mean_corr", 0.0)),
        std_corr=float(record.get("std_corr", 0.0)),
        mean_raw_dose=float(record.get("recruitment_raw_dose", 0.0)),
        std_raw_dose=float(record.get("std_recruitment_raw_dose", 0.0)),
        mean_norm_dose=float(record.get("recruitment_norm_dose", 0.0)),
        std_norm_dose=float(record.get("std_recruitment_norm_dose", 0.0)),
        mean_device_cost=float(record.get("device_cost", 0.0)),
        std_device_cost=float(record.get("std_device_cost", 0.0)),
        mean_current_rate_usage=float(record.get("current_rate_usage", 0.0)),
        std_current_rate_usage=float(record.get("std_current_rate_usage", 0.0)),
        mean_total_current_ma=float(record.get("total_current_ma", 0.0)),
        std_total_current_ma=float(record.get("std_total_current_ma", 0.0)),
        mean_charge_per_pulse_uc=float(record.get("charge_per_pulse_uc", 0.0)),
        std_charge_per_pulse_uc=float(record.get("std_charge_per_pulse_uc", 0.0)),
        mean_charge_rate_uc_per_s=float(record.get("charge_rate_uc_per_s", 0.0)),
        std_charge_rate_uc_per_s=float(record.get("std_charge_rate_uc_per_s", 0.0)),
        mean_relative_envelope_rmse=float(record.get("relative_envelope_rmse", 0.0)),
        std_relative_envelope_rmse=float(record.get("std_relative_envelope_rmse", 0.0)),
        penalized_objective=float(record.get("penalized_objective", record.get("mean_corr", 0.0))),
        robust_objective=float(record.get("robust_objective", record.get("mean_corr", 0.0))),
        valid=bool(record.get("valid", True)),
        invalid_reason=record.get("invalid_reason"),
        metadata={},
    )


def load_optimizer_history(output_dir: str | Path) -> list[dict[str, Any]]:
    """Load one optimizer history when it already exists on disk."""

    history_path = Path(output_dir) / "history.jsonl"
    if not history_path.exists():
        return []
    return [dict(row) for row in read_jsonl(history_path)]


def history_eval_index(history: list[Mapping[str, Any]]) -> int:
    """Return the last evaluation index stored in a history."""

    if not history:
        return 0
    return int(history[-1].get("eval_index", len(history)))


def history_seed_trials(history: list[Mapping[str, Any]]) -> int:
    """Return the cumulative seed-trial count stored in a history."""

    if not history:
        return 0
    return int(history[-1].get("seed_trials_used", 0))


def best_history_record(history: list[dict[str, Any]], *, score_key: str = "penalized_objective") -> dict[str, Any]:
    """Return the highest-scoring search record from an optimizer history."""

    if not history:
        raise ValueError("Cannot choose a best history record from an empty history.")
    return max((dict(record) for record in history), key=lambda record: float(record[score_key]))


def optimizer_summary_payload(
    *,
    result: OptimizerRunResult,
    config_bundle: Mapping[str, Any],
    history_for_best_pattern: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the compact on-disk optimizer summary."""

    selected_history = history_for_best_pattern if history_for_best_pattern is not None else result.history
    best_record = best_history_record(selected_history)
    return {
        "algorithm": result.algorithm,
        "output_dir": result.output_dir,
        "config": dict(config_bundle),
        "cost_metric": "device_cost",
        "cost_metric_label": "normalized_charge_rate_usage",
        "metadata": dict(result.metadata),
        "best_pattern": {
            "theta": theta_from_history_record(best_record),
            "record": best_record,
        },
    }


def unpack_run_config(run_config: Mapping[str, Any]) -> tuple[SimulationConfig, OptimizerConfig]:
    """Normalize the shared runner input contract used by the CLI scripts."""

    simulation = run_config.get("simulation") or run_config.get("simulation_config")
    optimizer = run_config.get("optimizer") or run_config.get("optimizer_config")
    if not isinstance(simulation, SimulationConfig) or not isinstance(optimizer, OptimizerConfig):
        raise TypeError("run_config must contain `simulation` and `optimizer` dataclass entries.")
    return simulation, optimizer
