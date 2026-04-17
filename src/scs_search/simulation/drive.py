"""Local supraspinal drive generation and spike sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit

from ..config import SimulationConfig


@dataclass(frozen=True)
class SupraspinalDrive:
    """Seed-specific descending-command waveform used for supraspinal spike generation."""

    mode: str
    time_ms: np.ndarray
    envelope: np.ndarray
    rate_hz: np.ndarray
    metadata: dict[str, Any]


def _time_axis_ms(simulation_duration_ms: int) -> np.ndarray:
    """Return the 1 ms supraspinal sampling axis."""

    return np.arange(0.0, float(simulation_duration_ms), 1.0, dtype=float)


def _control_times_ms(simulation_duration_ms: int, control_dt_ms: float) -> np.ndarray:
    """Return control-point locations including the simulation endpoint."""

    control_times = np.arange(0.0, float(simulation_duration_ms), float(control_dt_ms), dtype=float)
    if control_times.size == 0 or not np.isclose(control_times[-1], float(simulation_duration_ms)):
        control_times = np.append(control_times, float(simulation_duration_ms))
    return control_times


def _standardize(values: np.ndarray) -> np.ndarray:
    """Standardize a latent sequence when it has nonzero variance."""

    std = float(np.std(values))
    if std <= 0.0:
        return values.copy()
    return (values - float(np.mean(values))) / std


def _build_task_burst_support(config: SimulationConfig, trial_seed: int, time_ms: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
    """Construct a seed-specific burst support with real active and silent intervals."""

    rng = np.random.RandomState(int(trial_seed) + 17_311)
    support = np.zeros(time_ms.shape, dtype=float)
    burst_intervals_ms: list[list[float]] = []
    time_cursor_ms = float(rng.uniform(0.0, max(float(config.supraspinal_task_gap_max_ms), 1.0)))
    burst_min_ms = float(config.supraspinal_task_burst_min_ms)
    burst_max_ms = float(max(config.supraspinal_task_burst_max_ms, burst_min_ms))
    gap_min_ms = float(config.supraspinal_task_gap_min_ms)
    gap_max_ms = float(max(config.supraspinal_task_gap_max_ms, gap_min_ms))
    duration_ms = float(config.simulation_duration_ms)

    while time_cursor_ms < duration_ms:
        burst_duration_ms = float(rng.uniform(burst_min_ms, burst_max_ms))
        burst_start_ms = time_cursor_ms
        burst_end_ms = min(burst_start_ms + burst_duration_ms, duration_ms)
        start_index = int(np.floor(burst_start_ms))
        end_index = int(np.ceil(burst_end_ms))
        if end_index > start_index:
            support[start_index:end_index] = 1.0
            burst_intervals_ms.append([float(burst_start_ms), float(burst_end_ms)])
        gap_duration_ms = float(rng.uniform(gap_min_ms, gap_max_ms))
        time_cursor_ms = burst_end_ms + gap_duration_ms

    if not burst_intervals_ms and support.size:
        center_ms = 0.5 * duration_ms
        half_width_ms = 0.5 * burst_min_ms
        burst_start_ms = max(center_ms - half_width_ms, 0.0)
        burst_end_ms = min(center_ms + half_width_ms, duration_ms)
        start_index = int(np.floor(burst_start_ms))
        end_index = int(np.ceil(burst_end_ms))
        support[start_index:end_index] = 1.0
        burst_intervals_ms.append([float(burst_start_ms), float(burst_end_ms)])
    return support, burst_intervals_ms


def _build_task_amplitude_envelope(config: SimulationConfig, trial_seed: int, time_ms: np.ndarray) -> np.ndarray:
    """Construct the slow common-drive amplitude that modulates active task intervals."""

    control_times_ms = _control_times_ms(
        config.simulation_duration_ms,
        config.supraspinal_envelope_control_dt_ms,
    )
    rng = np.random.RandomState(int(trial_seed))
    rho = float(config.supraspinal_envelope_ar_rho)
    innovation_scale = float(np.sqrt(max(1.0 - (rho * rho), 0.0)))
    latent = np.empty(control_times_ms.size, dtype=float)
    latent[0] = float(rng.normal())
    for index in range(1, control_times_ms.size):
        latent[index] = rho * latent[index - 1] + innovation_scale * float(rng.normal())
    amplitude_control = expit(0.9 * _standardize(latent))
    interpolator = PchipInterpolator(control_times_ms, amplitude_control)
    return np.asarray(interpolator(time_ms), dtype=float)


def _build_aperiodic_envelope(config: SimulationConfig, trial_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Construct the smooth aperiodic envelope on the 1 ms time grid."""

    time_ms = _time_axis_ms(config.simulation_duration_ms)
    task_support, burst_intervals_ms = _build_task_burst_support(config, trial_seed, time_ms)
    task_amplitude = _build_task_amplitude_envelope(config, trial_seed, time_ms)
    envelope = task_support * task_amplitude
    sigma_samples = float(config.supraspinal_envelope_smoothing_sigma_ms)
    if sigma_samples > 0.0:
        envelope = gaussian_filter1d(envelope, sigma=sigma_samples, mode="nearest")
    clipped = np.clip(envelope, 0.0, 1.0)
    clipped[clipped < 1e-4] = 0.0
    return time_ms, clipped, burst_intervals_ms


def _build_sinusoidal_envelope(config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Construct the legacy half-wave sinusoidal envelope on the 1 ms time grid."""

    time_ms = _time_axis_ms(config.simulation_duration_ms)
    phase = 2.0 * np.pi * float(config.supraspinal_inhomogeneous_rate_hz) * time_ms
    envelope = np.maximum(np.sin(phase), 0.0)
    return time_ms, np.clip(envelope, 0.0, 1.0)


def build_supraspinal_drive(config: SimulationConfig, trial_seed: int) -> SupraspinalDrive:
    """Return the seed-specific supraspinal rate profile for one trial."""

    mode = str(config.supraspinal_drive_mode).lower()
    if mode == "aperiodic_envelope":
        time_ms, envelope, burst_intervals_ms = _build_aperiodic_envelope(config, int(trial_seed))
        floor_hz = float(config.supraspinal_rate_floor_hz)
        peak_hz = float(config.supraspinal_rate_hz)
        rate_hz = floor_hz + (peak_hz - floor_hz) * envelope
        metadata = {
            "supraspinal_drive_mode": "aperiodic_envelope",
            "supraspinal_rate_floor_hz": floor_hz,
            "supraspinal_rate_peak_hz": peak_hz,
            "supraspinal_envelope_control_dt_ms": float(config.supraspinal_envelope_control_dt_ms),
            "supraspinal_envelope_smoothing_sigma_ms": float(config.supraspinal_envelope_smoothing_sigma_ms),
            "supraspinal_envelope_ar_rho": float(config.supraspinal_envelope_ar_rho),
            "supraspinal_task_burst_min_ms": float(config.supraspinal_task_burst_min_ms),
            "supraspinal_task_burst_max_ms": float(config.supraspinal_task_burst_max_ms),
            "supraspinal_task_gap_min_ms": float(config.supraspinal_task_gap_min_ms),
            "supraspinal_task_gap_max_ms": float(config.supraspinal_task_gap_max_ms),
            "supraspinal_task_burst_count": int(len(burst_intervals_ms)),
            "supraspinal_task_burst_intervals_ms": burst_intervals_ms,
            "supraspinal_task_active_fraction": float(np.mean(envelope > 1e-3)) if envelope.size else 0.0,
            "supraspinal_trial_seed": int(trial_seed),
        }
        return SupraspinalDrive(
            mode="aperiodic_envelope",
            time_ms=time_ms,
            envelope=envelope,
            rate_hz=np.asarray(rate_hz, dtype=float),
            metadata=metadata,
        )
    if mode == "sinusoidal":
        time_ms, envelope = _build_sinusoidal_envelope(config)
        peak_hz = float(config.supraspinal_rate_hz)
        rate_hz = peak_hz * envelope
        metadata = {
            "supraspinal_drive_mode": "sinusoidal",
            "supraspinal_rate_floor_hz": 0.0,
            "supraspinal_rate_peak_hz": peak_hz,
            "supraspinal_inhomogeneous_rate_hz": float(config.supraspinal_inhomogeneous_rate_hz),
            "supraspinal_trial_seed": int(trial_seed),
        }
        return SupraspinalDrive(
            mode="sinusoidal",
            time_ms=time_ms,
            envelope=envelope,
            rate_hz=np.asarray(rate_hz, dtype=float),
            metadata=metadata,
        )
    raise ValueError(f"Unsupported supraspinal_drive_mode: {config.supraspinal_drive_mode}")


def sample_supraspinal_spike_trains(
    config: SimulationConfig,
    trial_seed: int,
    *,
    num_neurons: int | None = None,
) -> tuple[list[np.ndarray], SupraspinalDrive]:
    """Sample per-neuron supraspinal spike trains on 1 ms bins for one trial seed."""

    drive = build_supraspinal_drive(config, int(trial_seed))
    neuron_count = int(config.num_supraspinal_total if num_neurons is None else num_neurons)
    if neuron_count <= 0:
        return [], drive
    probability = np.clip(np.asarray(drive.rate_hz, dtype=float) / 1000.0, 0.0, 1.0)
    rng = np.random.RandomState(int(trial_seed))
    draws = rng.rand(neuron_count, probability.size) < probability[None, :]
    spike_trains = [
        np.asarray(drive.time_ms[mask], dtype=float)
        for mask in draws
    ]
    return spike_trains, drive
