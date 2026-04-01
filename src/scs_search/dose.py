"""Dose computations and hardware-budget approximation helpers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .config import DeviceConfig, DoseConfig, PatternParameters, StimPattern


def raw_dose_from_pulse_alpha(pulse_alpha: Sequence[float]) -> float:
    """Return the recruited-fiber-pulse count."""

    return float(np.sum(np.asarray(pulse_alpha, dtype=float)))


def normalized_recruitment_dose(raw_dose: float, duration_ms: float, max_frequency_hz: float = 1200.0) -> float:
    """Normalize recruited-fiber pulses by a full-recruitment reference run."""

    duration_seconds = float(duration_ms) / 1000.0
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive when normalizing recruitment dose.")
    return float(raw_dose) / (duration_seconds * float(max_frequency_hz))


def pulse_currents_ma(pulse_alpha: Sequence[float], device_config: DeviceConfig) -> np.ndarray:
    """Map recruitment fraction linearly to total program current."""

    alpha = np.clip(np.asarray(pulse_alpha, dtype=float), 0.0, 1.0)
    return alpha * float(device_config.max_total_current_ma)


def pulse_charge_uc(pulse_current_ma: Sequence[float], pulse_width_us: float) -> np.ndarray:
    """Return per-pulse charge in microcoulombs."""

    current_ma = np.asarray(pulse_current_ma, dtype=float)
    return current_ma * float(pulse_width_us) / 1000.0


def normalized_device_cost(total_charge_uc: float, duration_ms: float, device_config: DeviceConfig) -> float:
    """Normalize charge rate by the device maximum current/width/rate envelope."""

    duration_seconds = float(duration_ms) / 1000.0
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive when normalizing device cost.")
    max_charge_rate_uc_per_s = (
        float(device_config.max_total_current_ma)
        * float(device_config.max_pulse_width_us)
        / 1000.0
        * float(device_config.max_master_rate_hz)
    )
    return float(total_charge_uc / duration_seconds) / max_charge_rate_uc_per_s


def compute_pattern_dose(pattern: StimPattern, dose_config: DoseConfig, device_config: DeviceConfig) -> dict[str, float]:
    """Compute internal recruitment dose and public hardware-budget metrics."""

    duration_ms = pattern.time_ms[-1] + 1.0 if pattern.time_ms.size else 1.0
    pulse_width_us = float(pattern.metadata.get("pulse_width_us", pattern.theta.pulse_width_us))
    raw_recruitment_dose = raw_dose_from_pulse_alpha(pattern.pulse_alpha)
    recruitment_dose_norm = normalized_recruitment_dose(
        raw_recruitment_dose,
        duration_ms=duration_ms,
        max_frequency_hz=dose_config.max_frequency_hz,
    )

    currents_ma = pulse_currents_ma(pattern.pulse_alpha, device_config)
    charge_per_pulse_uc = pulse_charge_uc(currents_ma, pulse_width_us)
    total_charge_uc = float(np.sum(charge_per_pulse_uc))
    duration_seconds = float(duration_ms) / 1000.0
    charge_rate_uc_per_s = total_charge_uc / duration_seconds
    device_cost = normalized_device_cost(total_charge_uc, duration_ms=duration_ms, device_config=device_config)

    return {
        "raw_recruitment_dose": raw_recruitment_dose,
        "recruitment_dose_norm": recruitment_dose_norm,
        "device_cost": device_cost,
        "mean_total_current_ma": float(np.mean(currents_ma)) if currents_ma.size else 0.0,
        "mean_charge_per_pulse_uc": float(np.mean(charge_per_pulse_uc)) if charge_per_pulse_uc.size else 0.0,
        "charge_rate_uc_per_s": float(charge_rate_uc_per_s),
        "total_charge_uc": total_charge_uc,
        "pulse_width_us": pulse_width_us,
    }


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
    device_cost: float,
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
        budget_penalty = dose_config.objective_penalty_weight * max(0.0, float(device_cost) - float(budget_norm)) ** 2
    aux_penalty = 0.0
    if theta is not None:
        aux_penalty += frequency_penalty(theta, dose_config)
    if pulse_alpha is not None:
        aux_penalty += high_recruitment_penalty(pulse_alpha, dose_config)
    penalized = robust_score - budget_penalty - aux_penalty
    return robust_score, penalized


def is_feasible(device_cost: float, budget_norm: float | None) -> bool:
    """Return whether a candidate satisfies the requested device-budget constraint."""

    return budget_norm is None or float(device_cost) <= float(budget_norm) + 1e-12


def device_metric_summary(pattern: StimPattern, dose_config: DoseConfig, device_config: DeviceConfig) -> dict[str, Any]:
    """Convenience wrapper retained for scripts and debugging helpers."""

    return compute_pattern_dose(pattern, dose_config=dose_config, device_config=device_config)
