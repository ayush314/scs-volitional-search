"""BOHB multi-fidelity runner built on ConfigSpace."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...config import PhysicalModulationParameters
from ...simulation.evaluator import evaluate_pattern, resolve_reference_emg_cache
from ...utils import progress, read_pickle, write_pickle
from ..optimizer_history import (
    best_history_record,
    final_run_result,
    history_entry,
    history_eval_index,
    history_seed_trials,
    load_optimizer_history,
    summary_from_history_record,
    theta_from_history_record,
    unpack_run_config,
)


def _build_configspace(bounds: Any, seed: int) -> Any:
    """Create a ConfigSpace object for the theta box."""

    from ConfigSpace import ConfigurationSpace, UniformFloatHyperparameter

    configspace = ConfigurationSpace(seed=seed)
    for name, lower, upper in zip(bounds.names, bounds.lower, bounds.upper):
        configspace.add_hyperparameter(UniformFloatHyperparameter(name, lower=lower, upper=upper))
    return configspace


def _sample_theta(
    configspace: Any,
    bounds: Any,
    device_config: Any,
    rng: np.random.RandomState,
    elites: list[PhysicalModulationParameters],
) -> PhysicalModulationParameters:
    """Sample a random or elite-perturbed configuration."""

    if elites and rng.rand() < 0.5:
        base = bounds.encode_unit(rng.choice(elites), device_config=device_config)
        proposal = np.clip(base + rng.normal(scale=0.08, size=base.shape), 0.0, 1.0)
        return bounds.decode_unit(proposal, device_config=device_config)
    sampled = configspace.sample_configuration()
    return bounds.clip(sampled.get_dictionary(), device_config=device_config)


def _state_path(output_dir: str) -> Path:
    """Return the on-disk BOHB checkpoint path."""

    return Path(output_dir) / "resume_state.pkl"


def _reconstruct_elites(history: list[dict[str, Any]], full_seed_budget: int) -> list[PhysicalModulationParameters]:
    """Warm-reconstruct BOHB elites from previous history rows."""

    candidate_rows = [row for row in history if int(row.get("seed_budget", full_seed_budget)) == int(full_seed_budget)]
    if not candidate_rows:
        candidate_rows = list(history)
    candidate_rows = [row for row in candidate_rows if bool(row.get("valid", True))]
    candidate_rows.sort(key=lambda row: float(row.get("penalized_objective", row.get("mean_corr", 0.0))), reverse=True)
    elites: list[PhysicalModulationParameters] = []
    for row in candidate_rows:
        theta = theta_from_history_record(row)
        if any(existing == theta for existing in elites):
            continue
        elites.append(theta)
        if len(elites) >= 8:
            break
    return elites


def run_optimizer(run_config: Mapping[str, Any], output_dir: str, *, resume: bool = False) -> Any:
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
    configspace = _build_configspace(simulation_config.theta_bounds, seed=simulation_config.structural_seed)

    rng = np.random.RandomState(simulation_config.structural_seed)
    seed_budgets = [1, 2, 3]
    target_seed_trials = optimizer_config.seed_trial_budget
    bracket_sizes = [4, 2, 1]
    bracket_cost = sum(size * budget for size, budget in zip(bracket_sizes, seed_budgets))
    checkpoint_path = _state_path(output_dir)
    history = load_optimizer_history(output_dir) if resume else []
    used_seed_evaluations = history_seed_trials(history)
    eval_index = history_eval_index(history)
    previous_seed_trials = used_seed_evaluations
    resume_mode = "fresh"

    if resume and checkpoint_path.exists():
        state = read_pickle(checkpoint_path)
        rng = np.random.RandomState()
        rng.set_state(state["rng_state"])
        elites = list(state["elites"])
        best_theta = state["best_theta"]
        best_summary = state["best_summary"]
        resume_mode = "state"
    else:
        if history:
            best_record = best_history_record(history)
            best_theta = theta_from_history_record(best_record)
            best_summary = summary_from_history_record(best_record, theta=best_theta, seeds=train_seeds)
            elites = _reconstruct_elites(history, len(train_seeds))
            rng = np.random.RandomState(simulation_config.structural_seed + eval_index)
            resume_mode = "history"
        else:
            rng = np.random.RandomState(simulation_config.structural_seed)
            elites = []
            best_theta = None
            best_summary = None

    if used_seed_evaluations >= target_seed_trials:
        if best_theta is None or best_summary is None:
            raise RuntimeError("Resume requested, but no incumbent could be reconstructed from the existing run.")
        write_pickle(
            checkpoint_path,
            {
                "algorithm": "bohb",
                "rng_state": rng.get_state(),
                "elites": elites,
                "best_theta": best_theta,
                "best_summary": best_summary,
            },
        )
        return final_run_result(
            algorithm="bohb",
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
                "controller": "bohb",
                "resume_mode": resume_mode,
                "previous_seed_trials": previous_seed_trials,
            },
        )

    with progress(total=target_seed_trials - used_seed_evaluations, desc="BOHB search", unit="seed") as bar:
        while used_seed_evaluations + bracket_cost <= target_seed_trials:
            stage_candidates = [
                _sample_theta(configspace, simulation_config.theta_bounds, simulation_config.device_config, rng, elites)
                for _ in range(bracket_sizes[0])
            ]
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
            theta = _sample_theta(
                configspace,
                simulation_config.theta_bounds,
                simulation_config.device_config,
                rng,
                elites,
            )
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

    write_pickle(
        checkpoint_path,
        {
            "algorithm": "bohb",
            "rng_state": rng.get_state(),
            "elites": elites,
            "best_theta": best_theta,
            "best_summary": best_summary,
        },
    )

    return final_run_result(
        algorithm="bohb",
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
            "controller": "bohb",
            "resume_mode": resume_mode,
            "previous_seed_trials": previous_seed_trials,
        },
    )
