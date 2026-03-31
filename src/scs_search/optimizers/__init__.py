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


def unpack_run_config(run_config: Mapping[str, Any]) -> tuple[SimulationConfig, OptimizerConfig]:
    """Normalize the shared runner input contract used by the CLI scripts."""

    simulation = run_config.get("simulation") or run_config.get("simulation_config")
    optimizer = run_config.get("optimizer") or run_config.get("optimizer_config")
    if not isinstance(simulation, SimulationConfig) or not isinstance(optimizer, OptimizerConfig):
        raise TypeError("run_config must contain `simulation` and `optimizer` dataclass entries.")
    return simulation, optimizer
