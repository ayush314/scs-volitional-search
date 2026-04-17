"""Minimal TuRBO-1 runner using BoTorch."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ...simulation.evaluator import evaluate_pattern, resolve_reference_emg_cache
from ...utils import latin_hypercube_samples, progress, read_pickle, write_pickle
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


def _state_path(output_dir: str) -> Path:
    """Return the on-disk TuRBO checkpoint path."""

    return Path(output_dir) / "resume_state.pkl"


def _replay_turbo_length_state(
    scores: np.ndarray,
    *,
    initial_count: int,
    initial_length: float,
    success_tolerance: int,
    failure_tolerance: int,
    min_length: float,
    max_length: float,
) -> tuple[float, int, int]:
    """Reconstruct the trust-region state from prior objective values."""

    length = float(initial_length)
    success_counter = 0
    failure_counter = 0
    if scores.size <= initial_count:
        return length, success_counter, failure_counter
    for index in range(initial_count, scores.size):
        score = float(scores[index])
        best_before = float(np.max(scores[:index]))
        improved = score > best_before + 1e-8
        if improved:
            success_counter += 1
            failure_counter = 0
            if success_counter >= success_tolerance:
                length = min(length * 2.0, max_length)
                success_counter = 0
        else:
            failure_counter += 1
            success_counter = 0
            if failure_counter >= failure_tolerance:
                length = max(length / 2.0, min_length)
                failure_counter = 0
    return float(length), int(success_counter), int(failure_counter)


def run_optimizer(run_config: Mapping[str, Any], output_dir: str, *, resume: bool = False) -> Any:
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

    dim = len(simulation_config.theta_bounds.names)
    seed = simulation_config.structural_seed
    target_seed_trials = optimizer_config.seed_trial_budget
    target_evaluations = max(1, target_seed_trials // len(train_seeds))
    initial_count = min(optimizer_config.turbo_initial_points, target_evaluations)
    checkpoint_path = _state_path(output_dir)
    history = load_optimizer_history(output_dir) if resume else []
    used_seed_evaluations = history_seed_trials(history)
    eval_index = history_eval_index(history)
    previous_seed_trials = used_seed_evaluations
    resume_mode = "fresh"

    if resume and checkpoint_path.exists():
        state = read_pickle(checkpoint_path)
        x_array = np.asarray(state["x_array"], dtype=float)
        y_array = np.asarray(state["y_array"], dtype=float)
        length = float(state["length"])
        success_counter = int(state["success_counter"])
        failure_counter = int(state["failure_counter"])
        best_theta = state["best_theta"]
        best_summary = state["best_summary"]
        resume_mode = "state"
    elif history:
        x_array = np.asarray(
            [
                simulation_config.theta_bounds.encode_unit(
                    theta_from_history_record(record),
                    device_config=simulation_config.device_config,
                )
                for record in history
            ],
            dtype=float,
        )
        y_array = np.asarray([float(record["penalized_objective"]) for record in history], dtype=float)
        length, success_counter, failure_counter = _replay_turbo_length_state(
            y_array,
            initial_count=initial_count,
            initial_length=optimizer_config.turbo_initial_length,
            success_tolerance=optimizer_config.turbo_success_tolerance,
            failure_tolerance=optimizer_config.turbo_failure_tolerance,
            min_length=optimizer_config.turbo_min_length,
            max_length=optimizer_config.turbo_max_length,
        )
        best_record = best_history_record(history)
        best_theta = theta_from_history_record(best_record)
        best_summary = summary_from_history_record(best_record, theta=best_theta, seeds=train_seeds)
        resume_mode = "history"
    else:
        x_array = np.empty((0, dim), dtype=float)
        y_array = np.empty((0,), dtype=float)
        length = float(optimizer_config.turbo_initial_length)
        success_counter = 0
        failure_counter = 0
        best_theta = None
        best_summary = None

    if used_seed_evaluations >= target_seed_trials:
        if best_theta is None or best_summary is None:
            raise RuntimeError("Resume requested, but no incumbent could be reconstructed from the existing run.")
        write_pickle(
            checkpoint_path,
            {
                "algorithm": "turbo",
                "x_array": x_array,
                "y_array": y_array,
                "length": length,
                "success_counter": success_counter,
                "failure_counter": failure_counter,
                "best_theta": best_theta,
                "best_summary": best_summary,
            },
        )
        return final_run_result(
            algorithm="turbo",
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
                "resume_mode": resume_mode,
                "previous_seed_trials": previous_seed_trials,
            },
        )

    with progress(total=target_seed_trials - used_seed_evaluations, desc="TuRBO search", unit="seed") as bar:
        if x_array.shape[0] == 0:
            x_observed = latin_hypercube_samples(dim=dim, n_samples=initial_count, seed=seed)
            summaries = []
            y_observed: list[float] = []
            for point in x_observed:
                theta = simulation_config.theta_bounds.decode_unit(point, device_config=simulation_config.device_config)
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
            best_theta = simulation_config.theta_bounds.decode_unit(
                x_array[best_index],
                device_config=simulation_config.device_config,
            )
            best_summary = summaries[best_index]

        while eval_index < target_evaluations:
            model = _fit_model(x_array, y_array)
            center = x_array[int(np.argmax(y_array))]
            candidate = _next_candidate(model, center=center, length=length)
            theta = simulation_config.theta_bounds.decode_unit(candidate, device_config=simulation_config.device_config)
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

    if best_theta is None or best_summary is None:
        raise RuntimeError("TuRBO did not evaluate any candidate.")

    write_pickle(
        checkpoint_path,
        {
            "algorithm": "turbo",
            "x_array": x_array,
            "y_array": y_array,
            "length": length,
            "success_counter": success_counter,
            "failure_counter": failure_counter,
            "best_theta": best_theta,
            "best_summary": best_summary,
        },
    )

    return final_run_result(
        algorithm="turbo",
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
            "resume_mode": resume_mode,
            "previous_seed_trials": previous_seed_trials,
        },
    )
