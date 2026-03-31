"""Dose computations and penalty helpers."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import DoseConfig, PatternParameters, StimPattern


def raw_dose_from_pulse_alpha(pulse_alpha: Sequence[float]) -> float:
    """Return the recruited-fiber-pulse count."""

    return float(np.sum(np.asarray(pulse_alpha, dtype=float)))


def normalized_dose(raw_dose: float, duration_ms: float, max_frequency_hz: float = 120.0) -> float:
    """Normalize by the full-recruitment pulse budget over the same duration.

    A normalized dose of 1.0 means the pattern delivered the same recruited-fiber
    pulse count as alpha=1.0 stimulation at `max_frequency_hz` over the full run.
    """

    duration_seconds = float(duration_ms) / 1000.0
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive when normalizing dose.")
    return float(raw_dose) / (duration_seconds * float(max_frequency_hz))


def compute_pattern_dose(pattern: StimPattern, dose_config: DoseConfig) -> tuple[float, float]:
    """Compute raw and normalized dose for a stimulation pattern."""

    raw = raw_dose_from_pulse_alpha(pattern.pulse_alpha)
    norm = normalized_dose(raw, duration_ms=pattern.time_ms[-1] + 1.0 if pattern.time_ms.size else 1.0, max_frequency_hz=dose_config.max_frequency_hz)
    return raw, norm


def frequency_penalty(theta: PatternParameters, dose_config: DoseConfig) -> float:
    """Optional penalty for high stimulation frequency."""

    excess = max(0.0, float(theta.f) - float(dose_config.frequency_penalty_threshold_hz))
    return dose_config.frequency_penalty_weight * excess


def high_recruitment_penalty(pulse_alpha: Sequence[float], dose_config: DoseConfig) -> float:
    """Optional penalty for repeatedly driving recruitment near 1.0."""

    alpha = np.asarray(pulse_alpha, dtype=float)
    if alpha.size == 0:
        return 0.0
    excess = np.maximum(alpha - dose_config.high_recruitment_threshold, 0.0)
    return float(dose_config.high_recruitment_weight * np.mean(excess))


def combined_objective(
    mean_corr: float,
    std_corr: float,
    norm_dose: float,
    budget_norm: float | None,
    dose_config: DoseConfig,
    *,
    robust: bool,
    theta: PatternParameters | None = None,
    pulse_alpha: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Compute the robust and budget-penalized scalar objective."""

    robust_score = float(mean_corr) - (dose_config.robust_std_weight * float(std_corr) if robust else 0.0)
    budget_penalty = 0.0
    if budget_norm is not None:
        budget_penalty = dose_config.objective_penalty_weight * max(0.0, float(norm_dose) - float(budget_norm)) ** 2
    aux_penalty = 0.0
    if theta is not None:
        aux_penalty += frequency_penalty(theta, dose_config)
    if pulse_alpha is not None:
        aux_penalty += high_recruitment_penalty(pulse_alpha, dose_config)
    penalized = robust_score - budget_penalty - aux_penalty
    return robust_score, penalized


def is_feasible(norm_dose: float, budget_norm: float | None) -> bool:
    """Return whether a candidate satisfies the requested dose budget."""

    return budget_norm is None or float(norm_dose) <= float(budget_norm) + 1e-12
