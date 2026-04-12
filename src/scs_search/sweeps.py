"""Grid and Latin-hypercube sweep utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .analysis import build_best_under_limit_frontier, summary_to_record
from .config import DEFAULT_SWEEP_SEED_TRIALS, PatternParameters, SimulationConfig
from .patterns import invalid_theta_reason, theta_from_duty_cycle, theta_from_tonic
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
    """Choose a balanced rectangular grid shape near a target count."""

    if target <= 1:
        return (1, max(1, target))

    preferred_ratio = preferred_rows / preferred_cols
    approx_rows = max(1, int(round(np.sqrt(target * preferred_ratio))))
    approx_cols = max(1, int(round(np.sqrt(target / preferred_ratio))))
    row_candidates = range(max(1, approx_rows - 6), approx_rows + 7)
    col_candidates = range(max(1, approx_cols - 6), approx_cols + 7)

    best_rows = approx_rows
    best_cols = approx_cols
    best_objective = float("inf")
    best_count_error = float("inf")
    best_ratio_error = float("inf")
    for rows in row_candidates:
        for cols in col_candidates:
            points = rows * cols
            count_error = abs(points - target)
            ratio_error = abs(np.log((rows / cols) / preferred_ratio))
            objective = 5.0 * count_error + target * ratio_error
            if (
                objective < best_objective
                or (objective == best_objective and count_error < best_count_error)
                or (
                    objective == best_objective
                    and count_error == best_count_error
                    and ratio_error < best_ratio_error
                )
            ):
                best_rows = rows
                best_cols = cols
                best_objective = objective
                best_count_error = count_error
                best_ratio_error = ratio_error
    return best_rows, best_cols


def sweep_grid_values(
    seed_trial_budget: int = DEFAULT_SWEEP_SEED_TRIALS,
    seed_count: int = 1,
) -> dict[str, np.ndarray]:
    """Derive a sweep preset from the requested seed-trial budget."""

    candidate_budget = max(1, int(seed_trial_budget) // max(1, int(seed_count)))
    tonic_target, duty_target, lhs_target = _split_counts(candidate_budget, (0.2, 0.2, 0.6))
    pw_anchors = np.asarray([60.0, 210.0, 600.0], dtype=float)
    tonic_rows, tonic_cols = _grid_shape_for_budget(max(1, tonic_target // len(pw_anchors)), preferred_rows=4, preferred_cols=5)
    duty_rows, duty_cols = _grid_shape_for_budget(max(1, duty_target // len(pw_anchors)), preferred_rows=4, preferred_cols=5)
    actual_tonic = tonic_rows * tonic_cols * len(pw_anchors)
    actual_duty = duty_rows * duty_cols * len(pw_anchors)
    lhs_count = max(1, candidate_budget - actual_tonic - actual_duty)

    return {
        "tonic_freqs": np.linspace(10.0, 400.0, num=tonic_rows),
        "tonic_alpha": np.linspace(0.1, 0.9, num=tonic_cols),
        "tonic_pw_us": pw_anchors,
        "duty_freqs": np.linspace(10.0, 400.0, num=duty_rows),
        "duty_cycle": np.linspace(0.1, 0.9, num=duty_cols),
        "duty_pw_us": pw_anchors,
        "full_theta_samples": np.asarray([lhs_count]),
    }


def tonic_grid_points(
    freq_values: Iterable[float],
    alpha_values: Iterable[float],
    pw_values: Iterable[float],
    duration_ms: int,
    config: SimulationConfig,
) -> list[PatternParameters]:
    """Generate theta values for the tonic slice."""

    alpha_list = [float(alpha) for alpha in alpha_values]
    freq_count = max(1, len(list(freq_values)))
    theta_values: list[PatternParameters] = []
    for pw_us in pw_values:
        max_freq = min(float(config.device_config.max_master_rate_hz), 1e6 / float(pw_us))
        valid_freqs = np.linspace(10.0, max_freq, num=freq_count)
        theta_values.extend(
            theta_from_tonic(
                freq_hz=float(freq_hz),
                pw_us=float(pw_us),
                alpha=float(alpha),
                t_end_ms=float(duration_ms),
            )
            for freq_hz in valid_freqs
            for alpha in alpha_list
        )
    return theta_values


def duty_cycle_grid_points(
    freq_values: Iterable[float],
    duty_cycle_values: Iterable[float],
    pw_values: Iterable[float],
    alpha: float,
    cycle_ms: float,
    config: SimulationConfig,
) -> list[PatternParameters]:
    """Generate theta values for the duty-cycle slice."""

    duty_cycle_list = [float(duty_cycle) for duty_cycle in duty_cycle_values]
    freq_count = max(1, len(list(freq_values)))
    theta_values: list[PatternParameters] = []
    for pw_us in pw_values:
        max_freq = min(float(config.device_config.max_master_rate_hz), 1e6 / float(pw_us))
        valid_freqs = np.linspace(10.0, max_freq, num=freq_count)
        theta_values.extend(
            theta_from_duty_cycle(
                freq_hz=float(freq_hz),
                pw_us=float(pw_us),
                alpha=float(alpha),
                duty_cycle=float(duty_cycle),
                cycle_ms=float(cycle_ms),
            )
            for freq_hz in valid_freqs
            for duty_cycle in duty_cycle_list
        )
    return theta_values


def full_space_lhs_points(config: SimulationConfig, n_samples: int, seed: int) -> list[PatternParameters]:
    """Generate Latin-hypercube points in the full theta search space."""

    rng_seed = int(seed)
    valid_points: list[PatternParameters] = []
    while len(valid_points) < n_samples:
        batch_size = max(4, 2 * (n_samples - len(valid_points)))
        lhs = latin_hypercube_samples(dim=len(config.theta_bounds.names), n_samples=batch_size, seed=rng_seed)
        rng_seed += 1
        for sample in lhs:
            theta = config.theta_bounds.decode_unit(sample)
            if (
                invalid_theta_reason(
                    theta,
                    config.device_config,
                    enforce_no_overlap=config.transduction_config.enforce_no_overlap,
                )
                is None
            ):
                valid_points.append(theta)
                if len(valid_points) >= n_samples:
                    break
    return valid_points[:n_samples]


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
            preset["tonic_pw_us"],
            config.simulation_duration_ms,
            config,
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
            pw_values=preset["duty_pw_us"],
            alpha=duty_cycle_alpha,
            cycle_ms=config.baseline_cycle_ms,
            config=config,
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
