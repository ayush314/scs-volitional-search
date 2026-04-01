"""Grid and Latin-hypercube sweep utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .analysis import build_upper_hull_frontier, summary_to_record
from .config import PatternParameters, SimulationConfig
from .patterns import theta_from_duty_cycle, theta_from_tonic
from .simulator_adapter import evaluate_pattern
from .utils import latin_hypercube_samples, progress


def sweep_grid_values() -> dict[str, np.ndarray]:
    """Return the default sweep preset used for the main experiments."""

    return {
        "tonic_freqs": np.linspace(10.0, 1200.0, num=8),
        "tonic_alpha": np.linspace(0.1, 0.9, num=6),
        "duty_freqs": np.linspace(10.0, 1200.0, num=8),
        "duty_cycle": np.linspace(0.1, 0.9, num=6),
        "full_theta_samples": np.asarray([104]),
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
    duty_cycle_alpha: float = 0.5,
    lhs_seed: int = 123,
    reference_emg_by_seed: dict[int, np.ndarray] | None = None,
) -> dict[str, list[dict]]:
    """Run the tonic, duty-cycle, and full-theta sweep suite."""

    preset = sweep_grid_values()
    tonic_records = evaluate_theta_set(
        tonic_grid_points(
            preset["tonic_freqs"],
            preset["tonic_alpha"],
            config.simulation_duration_ms,
        ),
        seeds=seeds,
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
        seeds=seeds,
        config=config,
        label="duty_cycle",
        reference_emg_by_seed=reference_emg_by_seed,
    )
    full_records = evaluate_theta_set(
        full_space_lhs_points(config, int(preset["full_theta_samples"][0]), seed=lhs_seed),
        seeds=seeds,
        config=config,
        label="lhs_full_theta",
        reference_emg_by_seed=reference_emg_by_seed,
    )
    all_records = tonic_records + duty_records + full_records
    frontier = build_upper_hull_frontier(all_records)
    return {
        "tonic": tonic_records,
        "duty_cycle": duty_records,
        "lhs_full_theta": full_records,
        "all": all_records,
        "frontier": frontier,
    }
