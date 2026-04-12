"""Pattern generators for tonic, duty-cycled, and Fourier-envelope SCS."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .config import DEFAULT_PULSE_WIDTH_US, DeviceConfig, PatternParameters, SimulationConfig, StimPattern


def clip_recruitment(alpha: np.ndarray | Sequence[float]) -> np.ndarray:
    """Clip normalized pulse amplitudes into the valid closed interval."""

    return np.clip(np.asarray(alpha, dtype=float), 0.0, 1.0)


def theta_from_tonic(
    freq_hz: float,
    pw_us: float,
    alpha: float,
    t_end_ms: float,
) -> PatternParameters:
    """Build a tonic pattern as a restricted theta vector."""

    return PatternParameters(
        f=float(freq_hz),
        pw_us=float(pw_us),
        T_on=float(t_end_ms),
        T_off=0.0,
        alpha0=float(alpha),
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )


def theta_from_duty_cycle(
    freq_hz: float,
    pw_us: float,
    alpha: float,
    duty_cycle: float,
    cycle_ms: float,
) -> PatternParameters:
    """Build a constant-amplitude duty-cycled pattern."""

    duty = float(np.clip(duty_cycle, 0.0, 1.0))
    return PatternParameters(
        f=float(freq_hz),
        pw_us=float(pw_us),
        T_on=float(cycle_ms * duty),
        T_off=float(cycle_ms * (1.0 - duty)),
        alpha0=float(alpha),
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )


def evaluate_alpha_envelope(theta: PatternParameters, times_ms: np.ndarray) -> np.ndarray:
    """Evaluate the clipped Fourier envelope over a vector of times."""

    period_ms = max(theta.T_on + theta.T_off, 1e-9)
    tau = np.mod(times_ms, period_ms)
    inside_on_window = tau < theta.T_on
    phase = tau / period_ms
    envelope = (
        theta.alpha0
        + theta.alpha1 * np.sin(2.0 * np.pi * phase + theta.phi1)
        + theta.alpha2 * np.sin(4.0 * np.pi * phase + theta.phi2)
    )
    alpha = np.zeros_like(times_ms, dtype=float)
    alpha[inside_on_window] = clip_recruitment(envelope[inside_on_window])
    return alpha


def invalid_theta_reason(
    theta: PatternParameters,
    device_config: DeviceConfig,
    *,
    enforce_no_overlap: bool = True,
) -> str | None:
    """Return a feasibility violation string for the delivered-pulse pattern."""

    if theta.f < 10.0 or theta.f > float(device_config.max_master_rate_hz):
        return "frequency_out_of_range"
    if theta.pw_us < float(device_config.min_pulse_width_us) or theta.pw_us > float(device_config.max_pulse_width_us):
        return "pulse_width_out_of_range"
    if enforce_no_overlap and theta.pw_us * theta.f > 1e6 + 1e-9:
        return "pulse_overlap"
    if theta.T_on < 0.0 or theta.T_off < 0.0:
        return "negative_window_duration"
    return None


def theta_is_valid(
    theta: PatternParameters,
    device_config: DeviceConfig,
    *,
    enforce_no_overlap: bool = True,
) -> bool:
    """Return whether the candidate satisfies the delivered-pulse constraints."""

    return invalid_theta_reason(theta, device_config, enforce_no_overlap=enforce_no_overlap) is None


def _generate_pulse_times(
    freq_hz: float,
    pulse_width_us: float,
    T_on_ms: float,
    T_off_ms: float,
    t_end_ms: float,
) -> np.ndarray:
    """Generate pulse times whose full pulse width fits inside each on-window."""

    if freq_hz <= 0 or T_on_ms <= 0 or t_end_ms <= 0:
        return np.asarray([], dtype=float)

    period_ms = T_on_ms + T_off_ms
    if period_ms <= 0:
        return np.asarray([], dtype=float)
    pulse_width_ms = float(pulse_width_us) / 1000.0
    pulse_interval_ms = 1000.0 / freq_hz
    pulses: list[float] = []
    window_start = 0.0
    while window_start < t_end_ms:
        on_end = min(window_start + T_on_ms, t_end_ms)
        pulse_time = window_start
        while pulse_time + pulse_width_ms <= on_end + 1e-9:
            pulses.append(pulse_time)
            pulse_time += pulse_interval_ms
        window_start += period_ms
    return np.asarray(pulses, dtype=float)


def _build_pattern(
    theta: PatternParameters,
    t_end_ms: float,
    dt_ms: float,
    family: str,
    device_config: DeviceConfig,
    metadata: dict | None = None,
) -> StimPattern:
    """Assemble a `StimPattern` instance from shared components."""

    time_ms = np.arange(0.0, float(t_end_ms), float(dt_ms), dtype=float)
    alpha_t = evaluate_alpha_envelope(theta, time_ms)
    pulse_times_ms = _generate_pulse_times(theta.f, theta.pw_us, theta.T_on, theta.T_off, t_end_ms)
    pulse_alpha = evaluate_alpha_envelope(theta, pulse_times_ms) if pulse_times_ms.size else np.asarray([], dtype=float)
    pulse_widths_us = np.full(pulse_alpha.shape, float(theta.pw_us), dtype=float)
    pulse_current_ma = pulse_alpha * float(device_config.max_total_current_ma)
    return StimPattern(
        family=family,
        theta=theta,
        time_ms=time_ms,
        alpha_t=alpha_t,
        pulse_times_ms=pulse_times_ms,
        pulse_alpha=pulse_alpha,
        pulse_current_ma=pulse_current_ma,
        pulse_widths_us=pulse_widths_us,
        metadata={"pulse_width_us": float(theta.pw_us), **(metadata or {})},
    )


def generate_stim_pattern(
    theta: Sequence[float] | dict[str, float] | PatternParameters,
    t_end_ms: int,
    dt_ms: float = 1.0,
    device_config: DeviceConfig | None = None,
) -> StimPattern:
    """Generate the main Fourier-envelope stimulation structure."""

    params = PatternParameters.from_any(theta)
    if device_config is None:
        device_config = DeviceConfig()
    return _build_pattern(
        params,
        t_end_ms=float(t_end_ms),
        dt_ms=dt_ms,
        family="fourier",
        device_config=device_config,
        metadata={"family": "fourier"},
    )


def generate_tonic_pattern(
    freq_hz: float,
    alpha: float,
    t_end_ms: int,
    pw_us: float = DEFAULT_PULSE_WIDTH_US,
    dt_ms: float = 1.0,
    device_config: DeviceConfig | None = None,
) -> StimPattern:
    """Generate tonic stimulation for debugging and baseline comparisons."""

    if device_config is None:
        device_config = DeviceConfig()
    theta = theta_from_tonic(freq_hz=freq_hz, pw_us=pw_us, alpha=alpha, t_end_ms=t_end_ms)
    return _build_pattern(
        theta,
        t_end_ms=float(t_end_ms),
        dt_ms=dt_ms,
        family="tonic",
        device_config=device_config,
        metadata={"family": "tonic"},
    )


def generate_duty_cycled_constant_pattern(
    freq_hz: float,
    alpha: float,
    duty_cycle: float,
    t_end_ms: int,
    pw_us: float = DEFAULT_PULSE_WIDTH_US,
    cycle_ms: float = 500.0,
    dt_ms: float = 1.0,
    device_config: DeviceConfig | None = None,
) -> StimPattern:
    """Generate constant-amplitude stimulation gated by an on/off duty cycle."""

    if device_config is None:
        device_config = DeviceConfig()
    theta = theta_from_duty_cycle(
        freq_hz=freq_hz,
        pw_us=pw_us,
        alpha=alpha,
        duty_cycle=duty_cycle,
        cycle_ms=cycle_ms,
    )
    return _build_pattern(
        theta,
        t_end_ms=float(t_end_ms),
        dt_ms=dt_ms,
        family="duty_cycle",
        device_config=device_config,
        metadata={"family": "duty_cycle", "duty_cycle": float(duty_cycle), "cycle_ms": float(cycle_ms)},
    )


def pattern_from_family(
    family: str,
    t_end_ms: int,
    config: SimulationConfig,
    theta: Sequence[float] | dict[str, float] | PatternParameters | None = None,
    *,
    freq_hz: float | None = None,
    pw_us: float | None = None,
    alpha: float | None = None,
    duty_cycle: float | None = None,
) -> StimPattern:
    """Dispatch helper used by sweeps and baseline scripts."""

    family_normalized = family.lower()
    if family_normalized == "fourier":
        if theta is None:
            raise ValueError("Fourier patterns require a theta vector.")
        return generate_stim_pattern(theta=theta, t_end_ms=t_end_ms, dt_ms=config.dt_ms, device_config=config.device_config)
    if family_normalized == "tonic":
        if freq_hz is None or alpha is None:
            raise ValueError("Tonic patterns require `freq_hz` and `alpha`.")
        return generate_tonic_pattern(
            freq_hz=freq_hz,
            pw_us=float(DEFAULT_PULSE_WIDTH_US if pw_us is None else pw_us),
            alpha=alpha,
            t_end_ms=t_end_ms,
            dt_ms=config.dt_ms,
            device_config=config.device_config,
        )
    if family_normalized == "duty_cycle":
        if freq_hz is None or alpha is None or duty_cycle is None:
            raise ValueError("Duty-cycled patterns require `freq_hz`, `alpha`, and `duty_cycle`.")
        return generate_duty_cycled_constant_pattern(
            freq_hz=freq_hz,
            pw_us=float(DEFAULT_PULSE_WIDTH_US if pw_us is None else pw_us),
            alpha=alpha,
            duty_cycle=duty_cycle,
            t_end_ms=t_end_ms,
            cycle_ms=config.baseline_cycle_ms,
            dt_ms=config.dt_ms,
            device_config=config.device_config,
        )
    raise ValueError(f"Unsupported pattern family: {family}")
