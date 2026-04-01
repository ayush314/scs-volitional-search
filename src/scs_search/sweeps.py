"""Grid and Latin-hypercube sweep utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .analysis import build_best_under_limit_frontier, summary_to_record
from .config import DEFAULT_SWEEP_SEED_TRIALS, PatternParameters, SimulationConfig
from .patterns import theta_from_duty_cycle, theta_from_tonic
from .simulator_adapter import evaluate_pattern
from .utils import latin_hypercube_samples, progress


def _split_counts(total: int, proportions: tuple[float, ...]) -> list[int]:
    """Split an integer total according to proportions with exact summation."""

    if total <= 0:
        return [0 for _ in proportions]
    raw = [total * weight for weight in proportions]
    counts = [int(np.floor(value)) for value in raw]
    remainder = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda idx: raw[idx] - counts[idx], reverse=True)
    for idx in order[:remainder]:
        counts[idx] += 1
    return counts


def _grid_shape_for_budget(target: int, preferred_rows: int, preferred_cols: int) -> tuple[int, int]:
    """Choose a rectangular grid shape under a target count."""

    if target <= 1:
        return (1, max(1, target))

    preferred_ratio = preferred_rows / preferred_cols
    best_rows = 1
    best_cols = target
    best_points = 0
    best_ratio_error = float("inf")
    for rows in range(1, target + 1):
        cols = max(1, target // rows)
        points = rows * cols
        ratio_error = abs((rows / cols) - preferred_ratio)
        if points > best_points or (points == best_points and ratio_error < best_ratio_error):
            best_rows = rows
            best_cols = cols
            best_points = points
            best_ratio_error = ratio_error
    return best_rows, best_cols


def sweep_grid_values(
    seed_trial_budget: int = DEFAULT_SWEEP_SEED_TRIALS,
    seed_count: int = 1,
) -> dict[str, np.ndarray]:
    """Derive a sweep preset from the requested seed-trial budget."""

    candidate_budget = max(1, int(seed_trial_budget) // max(1, int(seed_count)))
    tonic_target, duty_target, lhs_target = _split_counts(candidate_budget, (0.2, 0.2, 0.6))
    tonic_rows, tonic_cols = _grid_shape_for_budget(tonic_target, preferred_rows=4, preferred_cols=5)
    duty_rows, duty_cols = _grid_shape_for_budget(duty_target, preferred_rows=4, preferred_cols=5)
    actual_tonic = tonic_rows * tonic_cols
    actual_duty = duty_rows * duty_cols
    lhs_count = max(1, candidate_budget - actual_tonic - actual_duty)

    return {
        "tonic_freqs": np.linspace(10.0, 1200.0, num=tonic_rows),
        "tonic_alpha": np.linspace(0.1, 0.9, num=tonic_cols),
        "duty_freqs": np.linspace(10.0, 1200.0, num=duty_rows),
        "duty_cycle": np.linspace(0.1, 0.9, num=duty_cols),
        "full_theta_samples": np.asarray([lhs_count]),
    }


def tonic_grid_points(
    freq_values: Iterable[float],
    alpha_values: Iterable[float],
    duration_ms: int,
) -> list[PatternParameters]:
    """Generate theta values for the tonic slice."""

    return [
        theta_from_tonic(
            freq_hz=float(freq_hz),
            alpha=float(alpha),
            t_end_ms=float(duration_ms),
        )
        for freq_hz in freq_values
        for alpha in alpha_values
    ]


def duty_cycle_grid_points(
    freq_values: Iterable[float],
    duty_cycle_values: Iterable[float],
    alpha: float,
    cycle_ms: float,
) -> list[PatternParameters]:
    """Generate theta values for the duty-cycle slice."""

    return [
        theta_from_duty_cycle(
            freq_hz=float(freq_hz),
            alpha=float(alpha),
            duty_cycle=float(duty_cycle),
            cycle_ms=float(cycle_ms),
        )
        for freq_hz in freq_values
        for duty_cycle in duty_cycle_values
    ]


def full_space_lhs_points(config: SimulationConfig, n_samples: int, seed: int) -> list[PatternParameters]:
    """Generate Latin-hypercube points in the full theta search space."""

    lhs = latin_hypercube_samples(dim=len(config.theta_bounds.names), n_samples=n_samples, seed=seed)
    return [config.theta_bounds.decode_unit(sample) for sample in lhs]


def evaluate_theta_set(
    theta_values: Iterable[PatternParameters],
    seeds: Iterable[int],
    config: SimulationConfig,
    *,
    label: str,
    budget_norm: float | None = None,
    reference_emg_by_seed: dict[int, np.ndarray] | None = None,
) -> list[dict]:
    """Evaluate a batch of theta vectors and return flat records."""

    records: list[dict] = []
    theta_list = list(theta_values)
    for index, theta in enumerate(progress(theta_list, desc=f"Sweep {label}")):
        summary = evaluate_pattern(
            theta=theta,
            seeds=seeds,
            config=config,
            budget_norm=budget_norm,
            reference_emg_by_seed=reference_emg_by_seed,
        )
        records.append(summary_to_record(summary, extra={"slice": label, "eval_index": index}))
    return records


def run_sweep_suite(
    config: SimulationConfig,
    seeds: Iterable[int],
    *,
    seed_trial_budget: int = DEFAULT_SWEEP_SEED_TRIALS,
    duty_cycle_alpha: float = 0.5,
    lhs_seed: int = 123,
    reference_emg_by_seed: dict[int, np.ndarray] | None = None,
) -> dict[str, list[dict]]:
    """Run the tonic, duty-cycle, and full-theta sweep suite."""

    seeds_tuple = tuple(int(seed) for seed in seeds)
    preset = sweep_grid_values(seed_trial_budget=seed_trial_budget, seed_count=len(seeds_tuple))
    tonic_records = evaluate_theta_set(
        tonic_grid_points(
            preset["tonic_freqs"],
            preset["tonic_alpha"],
            config.simulation_duration_ms,
        ),
        seeds=seeds_tuple,
        config=config,
        label="tonic",
        reference_emg_by_seed=reference_emg_by_seed,
    )
    duty_records = evaluate_theta_set(
        duty_cycle_grid_points(
            freq_values=preset["duty_freqs"],
            duty_cycle_values=preset["duty_cycle"],
            alpha=duty_cycle_alpha,
            cycle_ms=config.baseline_cycle_ms,
        ),
        seeds=seeds_tuple,
        config=config,
        label="duty_cycle",
        reference_emg_by_seed=reference_emg_by_seed,
    )
    full_records = evaluate_theta_set(
        full_space_lhs_points(config, int(preset["full_theta_samples"][0]), seed=lhs_seed),
        seeds=seeds_tuple,
        config=config,
        label="lhs_full_theta",
        reference_emg_by_seed=reference_emg_by_seed,
    )
    all_records = tonic_records + duty_records + full_records
    frontier = build_best_under_limit_frontier(all_records)
    return {
        "tonic": tonic_records,
        "duty_cycle": duty_records,
        "lhs_full_theta": full_records,
        "all": all_records,
        "frontier": frontier,
        "preset": {
            "seed_trial_budget": int(seed_trial_budget),
            "seed_count": len(seeds_tuple),
            "candidate_budget": max(1, int(seed_trial_budget) // max(1, len(seeds_tuple))),
            "tonic": len(tonic_records),
            "duty_cycle": len(duty_records),
            "lhs_full_theta": len(full_records),
        },
    }
