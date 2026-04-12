"""Dose computations and hardware-budget metrics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .config import DeviceConfig, DoseConfig, PatternParameters, StimPattern


def raw_dose_from_pulse_recruitment(pulse_recruitment_fraction: Sequence[float]) -> float:
    """Return the recruited-fiber-pulse count."""

    return float(np.sum(np.asarray(pulse_recruitment_fraction, dtype=float)))


def normalized_recruitment_dose(raw_dose: float, duration_ms: float, max_frequency_hz: float = 400.0) -> float:
    """Normalize recruited-fiber pulses by a full-recruitment reference run."""

    duration_seconds = float(duration_ms) / 1000.0
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive when normalizing recruitment dose.")
    return float(raw_dose) / (duration_seconds * float(max_frequency_hz))


def pulse_charge_uc(pulse_current_ma: Sequence[float], pulse_width_us: Sequence[float] | float) -> np.ndarray:
    """Return per-pulse charge in microcoulombs."""

    current_ma = np.asarray(pulse_current_ma, dtype=float)
    widths_us = np.asarray(pulse_width_us, dtype=float)
    return current_ma * widths_us / 1000.0


def normalized_current_rate_usage(total_current_pulse_ma: float, duration_ms: float, device_config: DeviceConfig) -> float:
    """Normalize current-rate usage by the device current/rate limit envelope."""

    duration_seconds = float(duration_ms) / 1000.0
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive when normalizing current-rate usage.")
    max_current_rate_ma_per_s = float(device_config.max_total_current_ma) * float(device_config.max_master_rate_hz)
    return float(total_current_pulse_ma / duration_seconds) / max_current_rate_ma_per_s


def normalized_device_cost(total_charge_uc: float, duration_ms: float, device_config: DeviceConfig) -> float:
    """Normalize charge-rate usage by the device current/width/rate envelope."""

    duration_seconds = float(duration_ms) / 1000.0
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive when normalizing device cost.")
    max_charge_product_us_hz = min(
        float(device_config.max_pulse_width_us) * float(device_config.max_master_rate_hz),
        1e6,
    )
    max_charge_rate_uc_per_s = (
        float(device_config.max_total_current_ma)
        * max_charge_product_us_hz
        / 1000.0
    )
    return float(total_charge_uc / duration_seconds) / max_charge_rate_uc_per_s


def compute_pattern_dose(
    pattern: StimPattern,
    pulse_recruitment_fraction: Sequence[float],
    dose_config: DoseConfig,
    device_config: DeviceConfig,
) -> dict[str, float]:
    """Compute recruitment diagnostics and delivered-pulse hardware metrics."""

    duration_ms = pattern.time_ms[-1] + 1.0 if pattern.time_ms.size else 1.0
    pulse_width_us = (
        float(np.mean(pattern.pulse_widths_us))
        if pattern.pulse_widths_us.size
        else float(pattern.metadata.get("pulse_width_us", device_config.default_pulse_width_us))
    )
    raw_recruitment_dose = raw_dose_from_pulse_recruitment(pulse_recruitment_fraction)
    recruitment_dose_norm = normalized_recruitment_dose(
        raw_recruitment_dose,
        duration_ms=duration_ms,
        max_frequency_hz=dose_config.max_frequency_hz,
    )

    currents_ma = np.asarray(pattern.pulse_current_ma, dtype=float)
    total_current_pulse_ma = float(np.sum(currents_ma))
    charge_per_pulse_uc = pulse_charge_uc(currents_ma, pattern.pulse_widths_us)
    total_charge_uc = float(np.sum(charge_per_pulse_uc))
    duration_seconds = float(duration_ms) / 1000.0
    charge_rate_uc_per_s = total_charge_uc / duration_seconds
    current_rate_usage = normalized_current_rate_usage(
        total_current_pulse_ma,
        duration_ms=duration_ms,
        device_config=device_config,
    )
    device_cost = normalized_device_cost(total_charge_uc, duration_ms=duration_ms, device_config=device_config)

    return {
        "raw_recruitment_dose": raw_recruitment_dose,
        "recruitment_dose_norm": recruitment_dose_norm,
        "device_cost": device_cost,
        "current_rate_usage": current_rate_usage,
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


def high_recruitment_penalty(
    pulse_recruitment_fraction: Sequence[float],
    dose_config: DoseConfig,
) -> float:
    """Optional penalty for repeatedly driving recruitment near 1.0."""

    recruitment = np.asarray(pulse_recruitment_fraction, dtype=float)
    if recruitment.size == 0:
        return 0.0
    excess = np.maximum(recruitment - dose_config.high_recruitment_threshold, 0.0)
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
    pulse_recruitment_fraction: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Compute the robust and budget-penalized scalar objective."""

    robust_score = float(mean_corr) - (dose_config.robust_std_weight * float(std_corr) if robust else 0.0)
    budget_penalty = 0.0
    if budget_norm is not None:
        budget_penalty = dose_config.objective_penalty_weight * max(0.0, float(device_cost) - float(budget_norm)) ** 2
    aux_penalty = 0.0
    if theta is not None:
        aux_penalty += frequency_penalty(theta, dose_config)
    if pulse_recruitment_fraction is not None:
        aux_penalty += high_recruitment_penalty(pulse_recruitment_fraction, dose_config)
    penalized = robust_score - budget_penalty - aux_penalty
    return robust_score, penalized


def device_metric_summary(pattern: StimPattern, dose_config: DoseConfig, device_config: DeviceConfig) -> dict[str, Any]:
    """Compute device metrics from a pattern."""

    return compute_pattern_dose(
        pattern,
        pulse_recruitment_fraction=np.zeros_like(pattern.pulse_alpha, dtype=float),
        dose_config=dose_config,
        device_config=device_config,
    )
