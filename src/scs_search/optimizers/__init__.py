"""Optimizer runners for patterned SCS search."""

from __future__ import annotations

from typing import Any, Mapping

from ..analysis import summary_to_record
from ..config import EvaluationSummary, OptimizerConfig, OptimizerRunResult, PatternParameters, SimulationConfig


def history_entry(
    summary: EvaluationSummary,
    *,
    algorithm: str,
    eval_index: int,
    seed_trials_used: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an evaluation summary into a history row."""

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
    incumbent_theta: PatternParameters,
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


def best_history_record(history: list[dict[str, Any]], *, score_key: str = "penalized_objective") -> dict[str, Any]:
    """Return the highest-scoring search record from an optimizer history."""

    if not history:
        raise ValueError("Cannot choose a best history record from an empty history.")
    return max((dict(record) for record in history), key=lambda record: float(record[score_key]))


def optimizer_summary_payload(
    *,
    result: OptimizerRunResult,
    config_bundle: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the compact on-disk optimizer summary."""

    return {
        "algorithm": result.algorithm,
        "output_dir": result.output_dir,
        "config": dict(config_bundle),
        "cost_metric": "device_cost",
        "cost_metric_label": "normalized_charge_rate_usage",
        "metadata": dict(result.metadata),
        "best_pattern": {
            "theta": result.incumbent_theta,
            "record": best_history_record(result.history),
        },
    }


def unpack_run_config(run_config: Mapping[str, Any]) -> tuple[SimulationConfig, OptimizerConfig]:
    """Normalize the shared runner input contract used by the CLI scripts."""

    simulation = run_config.get("simulation") or run_config.get("simulation_config")
    optimizer = run_config.get("optimizer") or run_config.get("optimizer_config")
    if not isinstance(simulation, SimulationConfig) or not isinstance(optimizer, OptimizerConfig):
        raise TypeError("run_config must contain `simulation` and `optimizer` dataclass entries.")
    return simulation, optimizer
