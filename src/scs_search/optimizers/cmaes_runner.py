"""CMA-ES runner over the normalized SCS parameter box."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..simulator_adapter import evaluate_pattern, resolve_reference_emg_cache
from ..utils import progress
from . import final_run_result, history_entry, unpack_run_config


def run_optimizer(run_config: Mapping[str, Any], output_dir: str) -> Any:
    """Run CMA-ES and return a structured optimizer result."""

    try:
        import cma
    except ImportError as exc:
        raise RuntimeError("CMA-ES requires the `cma` package. Install the optimizer dependencies first.") from exc

    simulation_config, optimizer_config = unpack_run_config(run_config)
    train_seeds = simulation_config.seed_config.train_seeds
    report_seeds = simulation_config.seed_config.report_seeds
    reference_dir = Path(output_dir).resolve().parent / "reference"
    train_reference = resolve_reference_emg_cache(train_seeds, simulation_config, reference_dir=reference_dir)
    target_seed_trials = optimizer_config.seed_trial_budget
    target_evaluations = max(1, target_seed_trials // len(train_seeds))
    population_size = min(optimizer_config.cmaes_population_size, target_evaluations)

    rng = np.random.RandomState(simulation_config.structural_seed)
    x0 = rng.uniform(0.25, 0.75, size=len(simulation_config.theta_bounds.names))
    es = cma.CMAEvolutionStrategy(
        x0.tolist(),
        0.2,
        {
            "bounds": [0.0, 1.0],
            "popsize": population_size,
            "CMA_mirrors": 0,
            "verbose": -9,
        },
    )

    used_seed_evaluations = 0
    eval_index = 0
    history: list[dict[str, Any]] = []
    best_summary = None
    best_theta = None

    with progress(total=target_evaluations * len(train_seeds), desc="CMA-ES search", unit="seed") as bar:
        while eval_index < target_evaluations and not es.stop():
            remaining_evaluations = target_evaluations - eval_index
            if remaining_evaluations < population_size:
                break
            candidates = es.ask()
            losses = []
            evaluated_candidates = []
            for candidate in candidates:
                if eval_index >= target_evaluations:
                    break
                theta = simulation_config.theta_bounds.decode_unit(candidate)
                summary = evaluate_pattern(
                    theta=theta,
                    seeds=train_seeds,
                    config=simulation_config,
                    budget_norm=optimizer_config.budget_norm,
                    reference_emg_by_seed=train_reference,
                    robust_objective=optimizer_config.robust_objective,
                )
                eval_index += 1
                used_seed_evaluations += len(train_seeds)
                history.append(
                    history_entry(
                        summary,
                        algorithm="cmaes",
                        eval_index=eval_index,
                        seed_trials_used=used_seed_evaluations,
                    )
                )
                bar.update(len(train_seeds))
                losses.append(-summary.penalized_objective)
                evaluated_candidates.append(candidate)
                if best_summary is None or summary.penalized_objective > best_summary.penalized_objective:
                    best_summary = summary
                    best_theta = theta
            if evaluated_candidates:
                es.tell(evaluated_candidates, losses)
            else:
                break

    if best_theta is None or best_summary is None:
        raise RuntimeError("CMA-ES did not evaluate any candidate.")

    return final_run_result(
        algorithm="cmaes",
        output_dir=output_dir,
        incumbent_theta=best_theta,
        incumbent_summary=best_summary,
        history=history,
        metadata={
            "seed_trial_budget": target_seed_trials,
            "search_candidates_evaluated": eval_index,
            "search_seed_trials": used_seed_evaluations,
            "train_seed_count": len(train_seeds),
            "report_seed_count": len(report_seeds),
        },
    )
