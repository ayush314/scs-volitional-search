"""Sweep helpers for the task-burst physical-modulation search space."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..config import DEFAULT_SWEEP_SEED_TRIALS, PhysicalModulationParameters, SimulationConfig
from ..reporting.analysis import build_best_under_limit_frontier, summary_to_record
from ..simulation.evaluator import evaluate_pattern
from ..stimulation.patterns import invalid_theta_reason
from ..utils import latin_hypercube_samples, progress


def make_physical_modulation_simulation_config(
    *,
    backend: str = "neuron",
    supraspinal_drive_mode: str = "aperiodic_envelope",
) -> SimulationConfig:
    """Return the default simulation config for the physical-modulation study."""

    return SimulationConfig(backend=backend, supraspinal_drive_mode=supraspinal_drive_mode)


def theta_from_tonic_physical(
    i0_ma: float,
    f0_hz: float,
    *,
    period_ms: float = 500.0,
) -> PhysicalModulationParameters:
    """Build a constant-current constant-frequency pulse train."""

    return PhysicalModulationParameters(
        I0_ma=float(i0_ma),
        I1_ma=0.0,
        f0_hz=float(f0_hz),
        f1_hz=0.0,
        PW1_us=0.0,
        T_ms=float(period_ms),
    )


def physical_modulation_sweep_values(
    seed_trial_budget: int = DEFAULT_SWEEP_SEED_TRIALS,
    seed_count: int = 1,
) -> dict[str, object]:
    """Return the structured sweep preset for the physical-modulation search space."""

    candidate_budget = max(1, int(seed_trial_budget) // max(1, int(seed_count)))
    tonic_f0_hz = (10.0, 30.0, 60.0, 100.0, 205.0, 400.0)
    tonic_I0_ma = (2.0, 5.0, 8.0, 12.0, 16.0)
    tonic_count = len(tonic_f0_hz) * len(tonic_I0_ma)
    lhs_count = max(1, candidate_budget - tonic_count)
    return {
        "tonic_f0_hz": tonic_f0_hz,
        "tonic_I0_ma": tonic_I0_ma,
        "lhs_count": int(lhs_count),
    }


def tonic_physical_grid_points(
    current_values_ma: Sequence[float],
    frequency_values_hz: Sequence[float],
    *,
    period_ms: float = 500.0,
) -> list[PhysicalModulationParameters]:
    """Generate the constant pulse-train slice of the search space."""

    return [
        theta_from_tonic_physical(i0_ma=float(i0_ma), f0_hz=float(f0_hz), period_ms=float(period_ms))
        for i0_ma in current_values_ma
        for f0_hz in frequency_values_hz
    ]


def full_space_lhs_points(config: SimulationConfig, n_samples: int, seed: int) -> list[PhysicalModulationParameters]:
    """Generate valid Latin-hypercube points in the full theta search space."""

    rng_seed = int(seed)
    valid_points: list[PhysicalModulationParameters] = []
    while len(valid_points) < n_samples:
        batch_size = max(4, 2 * (n_samples - len(valid_points)))
        lhs = latin_hypercube_samples(dim=len(config.theta_bounds.names), n_samples=batch_size, seed=rng_seed)
        rng_seed += 1
        for sample in lhs:
            theta = config.theta_bounds.decode_unit(sample, device_config=config.device_config)
            if invalid_theta_reason(
                theta,
                config.device_config,
                enforce_no_overlap=config.transduction_config.enforce_no_overlap,
            ) is None:
                valid_points.append(theta)
                if len(valid_points) >= n_samples:
                    break
    return valid_points[:n_samples]


def evaluate_theta_set(
    theta_values: Iterable[PhysicalModulationParameters],
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


def run_physical_modulation_sweep_suite(
    config: SimulationConfig,
    seeds: Sequence[int],
    *,
    seed_trial_budget: int = DEFAULT_SWEEP_SEED_TRIALS,
    lhs_seed: int = 701,
    reference_emg_by_seed: dict[int, np.ndarray] | None = None,
) -> dict[str, list[dict]]:
    """Run the tonic slice plus full-space LHS for the physical-modulation model."""

    seeds_tuple = tuple(int(seed) for seed in seeds)
    preset = physical_modulation_sweep_values(seed_trial_budget=seed_trial_budget, seed_count=len(seeds_tuple))
    tonic_records = evaluate_theta_set(
        tonic_physical_grid_points(
            preset["tonic_I0_ma"],
            preset["tonic_f0_hz"],
            period_ms=config.baseline_cycle_ms,
        ),
        seeds=seeds_tuple,
        config=config,
        label="tonic_physical",
        reference_emg_by_seed=reference_emg_by_seed,
    )
    lhs_records = evaluate_theta_set(
        full_space_lhs_points(config, int(preset["lhs_count"]), lhs_seed),
        seeds=seeds_tuple,
        config=config,
        label="lhs_physical",
        reference_emg_by_seed=reference_emg_by_seed,
    )
    all_records = tonic_records + lhs_records
    return {
        "tonic": tonic_records,
        "lhs": lhs_records,
        "all": all_records,
        "frontier": build_best_under_limit_frontier(all_records),
        "preset": {
            "seed_trial_budget": int(seed_trial_budget),
            "candidate_budget": max(1, int(seed_trial_budget) // max(1, len(seeds_tuple))),
            "tonic_count": len(tonic_records),
            "lhs_count": len(lhs_records),
        },
    }
