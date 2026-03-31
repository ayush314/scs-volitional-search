"""Minimal TuRBO-1 runner using BoTorch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..simulator_adapter import evaluate_pattern, resolve_reference_emg_cache
from ..utils import latin_hypercube_samples, progress
from . import final_run_result, history_entry, unpack_run_config


def _fit_model(train_x: np.ndarray, train_y: np.ndarray) -> Any:
    """Fit a local GP surrogate in normalized space."""

    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood

    x = torch.tensor(train_x, dtype=torch.double)
    y = torch.tensor(train_y[:, None], dtype=torch.double)
    model = SingleTaskGP(x, y, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    return model


def _next_candidate(model: Any, center: np.ndarray, length: float) -> np.ndarray:
    """Optimize analytic EI inside the current trust region."""

    import torch
    from botorch.acquisition.analytic import ExpectedImprovement
    from botorch.optim import optimize_acqf

    lower = np.clip(center - length / 2.0, 0.0, 1.0)
    upper = np.clip(center + length / 2.0, 0.0, 1.0)
    bounds = torch.tensor(np.vstack([lower, upper]), dtype=torch.double)
    observed = model.train_targets.detach()
    acqf = ExpectedImprovement(model=model, best_f=observed.max())
    candidate, _ = optimize_acqf(acqf, bounds=bounds, q=1, num_restarts=8, raw_samples=64)
    return candidate.detach().cpu().numpy()[0]


def run_optimizer(run_config: Mapping[str, Any], output_dir: str) -> Any:
    """Run a TuRBO-1 loop and return a structured optimizer result."""

    try:
        import torch  # noqa: F401
        import botorch  # noqa: F401
        import gpytorch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "TuRBO requires `torch`, `botorch`, and `gpytorch`. Install the optimizer dependencies first."
        ) from exc

    simulation_config, optimizer_config = unpack_run_config(run_config)
    train_seeds = simulation_config.seed_config.train_seeds
    report_seeds = simulation_config.seed_config.report_seeds
    reference_dir = Path(output_dir).resolve().parent / "reference"
    train_reference = resolve_reference_emg_cache(train_seeds, simulation_config, reference_dir=reference_dir)
    report_reference = resolve_reference_emg_cache(report_seeds, simulation_config, reference_dir=reference_dir)

    dim = len(simulation_config.theta_bounds.names)
    seed = simulation_config.structural_seed
    target_seed_trials = optimizer_config.seed_trial_budget
    target_evaluations = max(1, target_seed_trials // len(train_seeds))
    initial_count = min(optimizer_config.turbo_initial_points, target_evaluations)
    x_observed = latin_hypercube_samples(dim=dim, n_samples=initial_count, seed=seed)
    y_observed: list[float] = []
    summaries = []
    history: list[dict[str, Any]] = []
    used_seed_evaluations = 0
    eval_index = 0

    with progress(total=target_evaluations * len(train_seeds), desc="TuRBO search", unit="seed") as bar:
        for point in x_observed:
            theta = simulation_config.theta_bounds.decode_unit(point)
            summary = evaluate_pattern(
                theta=theta,
                seeds=train_seeds,
                config=simulation_config,
                budget_norm=optimizer_config.budget_norm,
                reference_emg_by_seed=train_reference,
                robust_objective=optimizer_config.robust_objective,
            )
            summaries.append(summary)
            y_observed.append(summary.penalized_objective)
            used_seed_evaluations += len(train_seeds)
            eval_index += 1
            history.append(
                history_entry(
                    summary,
                    algorithm="turbo",
                    eval_index=eval_index,
                    seed_trials_used=used_seed_evaluations,
                    extra={"trust_region_length": optimizer_config.turbo_initial_length},
                )
            )
            bar.update(len(train_seeds))

        x_array = np.asarray(x_observed, dtype=float)
        y_array = np.asarray(y_observed, dtype=float)
        best_index = int(np.argmax(y_array))
        best_theta = simulation_config.theta_bounds.decode_unit(x_array[best_index])
        best_summary = summaries[best_index]
        length = optimizer_config.turbo_initial_length
        success_counter = 0
        failure_counter = 0

        while eval_index < target_evaluations:
            model = _fit_model(x_array, y_array)
            center = x_array[int(np.argmax(y_array))]
            candidate = _next_candidate(model, center=center, length=length)
            theta = simulation_config.theta_bounds.decode_unit(candidate)
            summary = evaluate_pattern(
                theta=theta,
                seeds=train_seeds,
                config=simulation_config,
                budget_norm=optimizer_config.budget_norm,
                reference_emg_by_seed=train_reference,
                robust_objective=optimizer_config.robust_objective,
            )
            score = summary.penalized_objective
            x_array = np.vstack([x_array, candidate])
            y_array = np.concatenate([y_array, np.asarray([score])])
            used_seed_evaluations += len(train_seeds)
            eval_index += 1
            bar.update(len(train_seeds))

            improved = score > float(np.max(y_array[:-1])) + 1e-8
            if improved:
                success_counter += 1
                failure_counter = 0
                if success_counter >= optimizer_config.turbo_success_tolerance:
                    length = min(length * 2.0, optimizer_config.turbo_max_length)
                    success_counter = 0
            else:
                failure_counter += 1
                success_counter = 0
                if failure_counter >= optimizer_config.turbo_failure_tolerance:
                    length = max(length / 2.0, optimizer_config.turbo_min_length)
                    failure_counter = 0

            history.append(
                history_entry(
                    summary,
                    algorithm="turbo",
                    eval_index=eval_index,
                    seed_trials_used=used_seed_evaluations,
                    extra={"trust_region_length": length},
                )
            )
            if score > best_summary.penalized_objective:
                best_summary = summary
                best_theta = theta

    incumbent_summary = evaluate_pattern(
        theta=best_theta,
        seeds=report_seeds,
        config=simulation_config,
        budget_norm=optimizer_config.budget_norm,
        reference_emg_by_seed=report_reference,
        robust_objective=optimizer_config.robust_objective,
    )
    return final_run_result(
        algorithm="turbo",
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
        },
    )
