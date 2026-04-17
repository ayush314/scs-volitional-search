"""Pattern generation for the physical-modulation stimulation family."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..config import (
    DEFAULT_PULSE_WIDTH_US,
    DeviceConfig,
    PhysicalModulationParameters,
    SimulationConfig,
    StimPattern,
)


def _modulation_caps(theta: PhysicalModulationParameters, device_config: DeviceConfig) -> tuple[float, float, float]:
    """Return the modulation amplitudes allowed by the device box."""

    i1_cap = min(float(theta.I0_ma), float(device_config.max_total_current_ma) - float(theta.I0_ma))
    f1_cap = min(float(theta.f0_hz) - 10.0, float(device_config.max_master_rate_hz) - float(theta.f0_hz))
    pw1_cap = min(
        float(DEFAULT_PULSE_WIDTH_US) - float(device_config.min_pulse_width_us),
        float(device_config.max_pulse_width_us) - float(DEFAULT_PULSE_WIDTH_US),
    )
    return float(max(i1_cap, 0.0)), float(max(f1_cap, 0.0)), float(max(pw1_cap, 0.0))


def modulation_controls(
    theta: PhysicalModulationParameters,
    times_ms: np.ndarray,
    device_config: DeviceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate current, frequency, and pulse width over time."""

    period_ms = max(float(theta.T_ms), 1e-9)
    wave = np.sin((2.0 * np.pi * np.asarray(times_ms, dtype=float)) / period_ms)
    current_ma = np.clip(
        float(theta.I0_ma) + float(theta.I1_ma) * wave,
        0.0,
        float(device_config.max_total_current_ma),
    )
    frequency_hz = np.clip(
        float(theta.f0_hz) + float(theta.f1_hz) * wave,
        10.0,
        float(device_config.max_master_rate_hz),
    )
    pulse_width_us = np.clip(
        float(DEFAULT_PULSE_WIDTH_US) + float(theta.PW1_us) * wave,
        float(device_config.min_pulse_width_us),
        float(device_config.max_pulse_width_us),
    )
    return current_ma, frequency_hz, pulse_width_us


def invalid_theta_reason(
    theta: PhysicalModulationParameters,
    device_config: DeviceConfig,
    *,
    enforce_no_overlap: bool = True,
) -> str | None:
    """Return a feasibility violation string when the theta value is invalid."""

    params = PhysicalModulationParameters.from_any(theta)
    if params.I0_ma < 0.0 or params.I0_ma > float(device_config.max_total_current_ma):
        return "current_out_of_range"
    i1_cap, f1_cap, pw1_cap = _modulation_caps(params, device_config)
    if params.I1_ma < 0.0 or params.I1_ma > i1_cap + 1e-9:
        return "current_modulation_out_of_range"
    if params.f0_hz < 10.0 or params.f0_hz > float(device_config.max_master_rate_hz):
        return "frequency_out_of_range"
    if params.f1_hz < 0.0 or params.f1_hz > f1_cap + 1e-9:
        return "frequency_modulation_out_of_range"
    if params.PW1_us < 0.0 or params.PW1_us > pw1_cap + 1e-9:
        return "pulse_width_modulation_out_of_range"
    if params.T_ms <= 0.0:
        return "nonpositive_period"
    if enforce_no_overlap:
        max_frequency_hz = float(params.f0_hz) + float(params.f1_hz)
        max_pulse_width_us = float(DEFAULT_PULSE_WIDTH_US) + float(params.PW1_us)
        if max_frequency_hz * max_pulse_width_us > 1e6 + 1e-9:
            return "pulse_overlap"
    return None


def _generate_variable_frequency_pulses(
    theta: PhysicalModulationParameters,
    *,
    t_end_ms: float,
    scheduler_dt_ms: float,
    device_config: DeviceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a realized pulse schedule from the time-varying frequency control."""

    if t_end_ms <= 0.0 or scheduler_dt_ms <= 0.0:
        empty = np.asarray([], dtype=float)
        return empty, empty, empty

    control_times_ms = np.arange(0.0, float(t_end_ms) + float(scheduler_dt_ms), float(scheduler_dt_ms), dtype=float)
    currents_ma, frequencies_hz, pulse_widths_us = modulation_controls(theta, control_times_ms, device_config)

    pulse_times: list[float] = []
    pulse_currents: list[float] = []
    pulse_widths: list[float] = []

    initial_width_ms = float(pulse_widths_us[0]) / 1000.0
    if initial_width_ms <= float(t_end_ms) + 1e-9:
        pulse_times.append(0.0)
        pulse_currents.append(float(currents_ma[0]))
        pulse_widths.append(float(pulse_widths_us[0]))

    phase = 0.0
    next_crossing = 1.0
    for index in range(len(control_times_ms) - 1):
        t0_ms = float(control_times_ms[index])
        t1_ms = float(control_times_ms[index + 1])
        dt_ms = t1_ms - t0_ms
        phase_next = phase + 0.5 * (float(frequencies_hz[index]) + float(frequencies_hz[index + 1])) * dt_ms / 1000.0
        while phase_next >= next_crossing - 1e-12:
            fraction = (next_crossing - phase) / max(phase_next - phase, np.finfo(float).eps)
            pulse_time_ms = t0_ms + fraction * dt_ms
            pulse_current_ma, _, pulse_width_us = modulation_controls(
                theta,
                np.asarray([pulse_time_ms], dtype=float),
                device_config,
            )
            if pulse_time_ms + float(pulse_width_us[0]) / 1000.0 <= float(t_end_ms) + 1e-9:
                pulse_times.append(float(pulse_time_ms))
                pulse_currents.append(float(pulse_current_ma[0]))
                pulse_widths.append(float(pulse_width_us[0]))
            next_crossing += 1.0
        phase = phase_next

    return (
        np.asarray(pulse_times, dtype=float),
        np.asarray(pulse_currents, dtype=float),
        np.asarray(pulse_widths, dtype=float),
    )


def _schedule_invalid_reason(
    pulse_times_ms: np.ndarray,
    pulse_widths_us: np.ndarray,
    *,
    t_end_ms: float,
) -> str | None:
    """Return a realized-schedule feasibility violation string when present."""

    if pulse_times_ms.size == 0:
        return None
    pulse_end_ms = pulse_times_ms + (pulse_widths_us / 1000.0)
    if np.any(pulse_end_ms > float(t_end_ms) + 1e-9):
        return "pulse_out_of_bounds"
    if pulse_times_ms.size > 1 and np.any(pulse_end_ms[:-1] > pulse_times_ms[1:] + 1e-9):
        return "pulse_overlap"
    return None


def _build_physical_pattern(
    theta: PhysicalModulationParameters,
    t_end_ms: float,
    dt_ms: float,
    scheduler_dt_ms: float,
    device_config: DeviceConfig,
) -> StimPattern:
    """Assemble a `StimPattern` from the physical-modulation controls."""

    time_ms = np.arange(0.0, float(t_end_ms), float(dt_ms), dtype=float)
    current_t_ma, _, pulse_width_t_us = modulation_controls(theta, time_ms, device_config)
    alpha_t = current_t_ma / float(device_config.max_total_current_ma)
    pulse_times_ms, pulse_current_ma, pulse_widths_us = _generate_variable_frequency_pulses(
        theta,
        t_end_ms=float(t_end_ms),
        scheduler_dt_ms=float(scheduler_dt_ms),
        device_config=device_config,
    )
    pulse_alpha = (
        np.clip(pulse_current_ma / float(device_config.max_total_current_ma), 0.0, 1.0)
        if pulse_current_ma.size
        else np.asarray([], dtype=float)
    )
    invalid_reason = _schedule_invalid_reason(pulse_times_ms, pulse_widths_us, t_end_ms=float(t_end_ms))
    metadata = {
        "family": "physical_modulation",
        "baseline_pulse_width_us": float(DEFAULT_PULSE_WIDTH_US),
        "pulse_scheduler_dt_ms": float(scheduler_dt_ms),
        "pulse_width_us": float(np.mean(pulse_width_t_us)) if pulse_width_t_us.size else float(DEFAULT_PULSE_WIDTH_US),
        "mean_pulse_width_us": float(np.mean(pulse_widths_us)) if pulse_widths_us.size else float(DEFAULT_PULSE_WIDTH_US),
        "realized_schedule_invalid_reason": invalid_reason,
    }
    return StimPattern(
        family="physical_modulation",
        theta=theta,
        time_ms=time_ms,
        alpha_t=alpha_t,
        pulse_times_ms=pulse_times_ms,
        pulse_alpha=pulse_alpha,
        pulse_current_ma=pulse_current_ma,
        pulse_widths_us=pulse_widths_us,
        metadata=metadata,
    )


def generate_stim_pattern(
    theta: Sequence[float] | Mapping[str, float] | PhysicalModulationParameters,
    t_end_ms: int,
    dt_ms: float = 1.0,
    device_config: DeviceConfig | None = None,
    pulse_scheduler_dt_ms: float = 0.1,
) -> StimPattern:
    """Generate the realized stimulation pattern for one theta value."""

    config = device_config or DeviceConfig()
    params = PhysicalModulationParameters.from_any(theta)
    return _build_physical_pattern(
        params,
        t_end_ms=float(t_end_ms),
        dt_ms=float(dt_ms),
        scheduler_dt_ms=float(pulse_scheduler_dt_ms),
        device_config=config,
    )


def generate_tonic_pattern(
    *,
    freq_hz: float,
    current_ma: float | None = None,
    alpha: float | None = None,
    t_end_ms: int,
    pw_us: float = DEFAULT_PULSE_WIDTH_US,
    period_ms: float = 500.0,
    dt_ms: float = 1.0,
    device_config: DeviceConfig | None = None,
) -> StimPattern:
    """Generate a constant-current constant-frequency pulse train."""

    config = device_config or DeviceConfig()
    if current_ma is None:
        if alpha is None:
            raise ValueError("Tonic pattern generation requires either `current_ma` or `alpha`.")
        current_ma = float(alpha) * float(config.max_total_current_ma)
    pulse_width_us = float(np.clip(float(pw_us), float(config.min_pulse_width_us), float(config.max_pulse_width_us)))
    time_ms = np.arange(0.0, float(t_end_ms), float(dt_ms), dtype=float)
    pulse_interval_ms = 1000.0 / float(freq_hz)
    pulse_width_ms = pulse_width_us / 1000.0
    pulse_times = np.arange(0.0, float(t_end_ms), pulse_interval_ms, dtype=float)
    pulse_times = pulse_times[pulse_times + pulse_width_ms <= float(t_end_ms) + 1e-9]
    pulse_currents = np.full(pulse_times.shape, float(np.clip(current_ma, 0.0, config.max_total_current_ma)), dtype=float)
    pulse_widths = np.full(pulse_times.shape, pulse_width_us, dtype=float)
    pulse_alpha = (
        np.clip(pulse_currents / float(config.max_total_current_ma), 0.0, 1.0)
        if pulse_currents.size
        else np.asarray([], dtype=float)
    )
    alpha_t = np.full(time_ms.shape, float(np.clip(current_ma, 0.0, config.max_total_current_ma)) / float(config.max_total_current_ma))
    return StimPattern(
        family="tonic",
        theta={
            "I0_ma": float(np.clip(current_ma, 0.0, config.max_total_current_ma)),
            "I1_ma": 0.0,
            "f0_hz": float(freq_hz),
            "f1_hz": 0.0,
            "PW1_us": 0.0,
            "T_ms": float(period_ms),
        },
        time_ms=time_ms,
        alpha_t=alpha_t,
        pulse_times_ms=pulse_times,
        pulse_alpha=pulse_alpha,
        pulse_current_ma=pulse_currents,
        pulse_widths_us=pulse_widths,
        metadata={"pulse_width_us": pulse_width_us, "family": "tonic"},
    )
