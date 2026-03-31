"""BOHB multi-fidelity runner built on ConfigSpace."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..config import PatternParameters
from ..simulator_adapter import evaluate_pattern, resolve_reference_emg_cache
from ..utils import progress
from . import final_run_result, history_entry, unpack_run_config


def _build_configspace(bounds: Any, seed: int) -> Any:
    """Create a ConfigSpace object for the theta box."""

    from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

    configspace = ConfigurationSpace(seed=seed)
    for name, lower, upper in zip(bounds.names, bounds.lower, bounds.upper):
        configspace.add_hyperparameter(UniformFloatHyperparameter(name, lower=lower, upper=upper))
    return configspace


def _sample_theta(configspace: Any, bounds: Any, rng: np.random.RandomState, elites: list[PatternParameters]) -> PatternParameters:
    """Sample a random or elite-perturbed configuration."""

    if elites and rng.rand() < 0.5:
        base = bounds.encode_unit(rng.choice(elites))
        proposal = np.clip(base + rng.normal(scale=0.08, size=base.shape), 0.0, 1.0)
        return bounds.decode_unit(proposal)
    sampled = configspace.sample_configuration()
    return PatternParameters.from_any(sampled.get_dictionary())


def run_optimizer(run_config: Mapping[str, Any], output_dir: str) -> Any:
    """Run a BOHB successive-halving loop with seed fidelity 1 -> 2 -> 3."""

    try:
        import ConfigSpace  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("BOHB requires `ConfigSpace`. Install the optimizer dependencies first.") from exc

    simulation_config, optimizer_config = unpack_run_config(run_config)
    train_seeds = simulation_config.seed_config.train_seeds
    report_seeds = simulation_config.seed_config.report_seeds
    reference_dir = Path(output_dir).resolve().parent / "reference"
    train_reference = resolve_reference_emg_cache(train_seeds, simulation_config, reference_dir=reference_dir)
    report_reference = resolve_reference_emg_cache(report_seeds, simulation_config, reference_dir=reference_dir)
    configspace = _build_configspace(simulation_config.theta_bounds, seed=simulation_config.structural_seed)

    rng = np.random.RandomState(simulation_config.structural_seed)
    seed_budgets = [1, 2, 3]
    target_seed_trials = optimizer_config.seed_trial_budget
    bracket_sizes = [4, 2, 1]
    bracket_cost = sum(size * budget for size, budget in zip(bracket_sizes, seed_budgets))

    history: list[dict[str, Any]] = []
    elites: list[PatternParameters] = []
    used_seed_evaluations = 0
    eval_index = 0
    best_theta = None
    best_summary = None

    with progress(total=target_seed_trials, desc="BOHB search", unit="seed") as bar:
        while used_seed_evaluations + bracket_cost <= target_seed_trials:
            stage_candidates = [_sample_theta(configspace, simulation_config.theta_bounds, rng, elites) for _ in range(bracket_sizes[0])]
            for stage, (count, seed_budget) in enumerate(zip(bracket_sizes, seed_budgets)):
                seeds = train_seeds[:seed_budget]
                stage_summaries = []
                for theta in stage_candidates[:count]:
                    summary = evaluate_pattern(
                        theta=theta,
                        seeds=seeds,
                        config=simulation_config,
                        budget_norm=optimizer_config.budget_norm,
                        reference_emg_by_seed={seed: train_reference[seed] for seed in seeds},
                        robust_objective=optimizer_config.robust_objective,
                    )
                    eval_index += 1
                    used_seed_evaluations += seed_budget
                    history.append(
                        history_entry(
                            summary,
                            algorithm="bohb",
                            eval_index=eval_index,
                            seed_trials_used=used_seed_evaluations,
                            extra={"seed_budget": seed_budget, "stage": stage},
                        )
                    )
                    bar.update(seed_budget)
                    stage_summaries.append(summary)
                    if best_summary is None or summary.penalized_objective > best_summary.penalized_objective:
                        best_summary = summary
                        best_theta = summary.theta
                stage_summaries.sort(key=lambda summary: summary.penalized_objective, reverse=True)
                stage_candidates = [summary.theta for summary in stage_summaries[: max(1, len(stage_summaries) // 2)]]
            if stage_candidates:
                elites.extend(stage_candidates[:1])
                elites = elites[-8:]

        while used_seed_evaluations < target_seed_trials:
            remaining = target_seed_trials - used_seed_evaluations
            seed_budget = max(budget for budget in seed_budgets if budget <= remaining)
            seeds = train_seeds[:seed_budget]
            theta = _sample_theta(configspace, simulation_config.theta_bounds, rng, elites)
            summary = evaluate_pattern(
                theta=theta,
                seeds=seeds,
                config=simulation_config,
                budget_norm=optimizer_config.budget_norm,
                reference_emg_by_seed={seed: train_reference[seed] for seed in seeds},
                robust_objective=optimizer_config.robust_objective,
            )
            eval_index += 1
            used_seed_evaluations += seed_budget
            history.append(
                history_entry(
                    summary,
                    algorithm="bohb",
                    eval_index=eval_index,
                    seed_trials_used=used_seed_evaluations,
                    extra={"seed_budget": seed_budget, "stage": "fill"},
                )
            )
            bar.update(seed_budget)
            if best_summary is None or summary.penalized_objective > best_summary.penalized_objective:
                best_summary = summary
                best_theta = summary.theta

    if best_theta is None or best_summary is None:
        raise RuntimeError("BOHB did not evaluate any candidate.")

    incumbent_summary = evaluate_pattern(
        theta=best_theta,
        seeds=report_seeds,
        config=simulation_config,
        budget_norm=optimizer_config.budget_norm,
        reference_emg_by_seed=report_reference,
        robust_objective=optimizer_config.robust_objective,
    )
    return final_run_result(
        algorithm="bohb",
        output_dir=output_dir,
        incumbent_theta=best_theta,
        incumbent_summary=incumbent_summary,
        history=history,
        metadata={
            "seed_trial_budget": target_seed_trials,
            "search_candidates_evaluated": eval_index,
            "search_seed_trials": used_seed_evaluations,
            "train_seed_count": len(train_seeds),
            "report_seed_count": len(report_seeds),
            "controller": "bohb",
        },
    )
