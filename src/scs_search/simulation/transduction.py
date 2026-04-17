"""Pulse-to-afferent transduction helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SimulationConfig, StimPattern
from .structural import _StructuralState, get_structural_state


@dataclass(frozen=True)
class _TransductionResult:
    """Delivered-pulse transduction into afferent-fiber spike trains."""

    afferent_pulse_times: list[np.ndarray]
    pulse_recruitment_fraction: np.ndarray


def _transduce_pattern_to_afferent_fibers(
    pattern: StimPattern,
    config: SimulationConfig,
    structural_state: _StructuralState,
) -> _TransductionResult:
    """Map delivered pulses into deterministic afferent-fiber spike trains."""

    num_afferents = int(config.num_scs_total)
    quantiles = np.empty(num_afferents, dtype=float)
    quantiles[structural_state.recruitment_order] = (
        np.arange(num_afferents, dtype=float) + 1.0
    ) / float(num_afferents)
    chronaxie_us = float(config.transduction_config.chronaxie_us)
    max_current_ma = float(config.device_config.max_total_current_ma)
    max_pulse_width_us = float(config.device_config.max_pulse_width_us)
    absolute_refractory_ms = float(config.transduction_config.absolute_refractory_ms)
    relative_refractory_end_ms = float(config.transduction_config.relative_refractory_end_ms)
    rheobase_max_ma = max_current_ma / (1.0 + (chronaxie_us / max_pulse_width_us))
    rheobase_ma = quantiles * rheobase_max_ma
    afferent_pulse_times: list[list[float]] = [[] for _ in range(num_afferents)]
    pulse_recruitment_fraction: list[float] = []
    last_spike_times_ms = np.full(num_afferents, -np.inf, dtype=float)
    for pulse_time, pulse_current_ma, pulse_width_us in zip(
        pattern.pulse_times_ms,
        pattern.pulse_current_ma,
        pattern.pulse_widths_us,
    ):
        baseline_threshold_ma = rheobase_ma * (1.0 + (chronaxie_us / float(pulse_width_us)))
        dt_since_last_spike_ms = float(pulse_time) - last_spike_times_ms
        eligible = dt_since_last_spike_ms >= absolute_refractory_ms
        effective_threshold_ma = baseline_threshold_ma.copy()
        if relative_refractory_end_ms > absolute_refractory_ms:
            recovering = eligible & (dt_since_last_spike_ms < relative_refractory_end_ms)
            recovery_fraction = (
                (dt_since_last_spike_ms[recovering] - absolute_refractory_ms)
                / (relative_refractory_end_ms - absolute_refractory_ms)
            )
            recovery_fraction = np.clip(recovery_fraction, np.finfo(float).eps, 1.0)
            effective_threshold_ma[recovering] = effective_threshold_ma[recovering] / recovery_fraction
        active_afferents = np.flatnonzero(
            eligible & (effective_threshold_ma <= float(pulse_current_ma) + 1e-12)
        )
        pulse_recruitment_fraction.append(float(active_afferents.size) / float(num_afferents))
        for afferent_index in active_afferents:
            afferent_pulse_times[int(afferent_index)].append(float(pulse_time))
        last_spike_times_ms[active_afferents] = float(pulse_time)
    return _TransductionResult(
        afferent_pulse_times=[np.asarray(times, dtype=float) for times in afferent_pulse_times],
        pulse_recruitment_fraction=np.asarray(pulse_recruitment_fraction, dtype=float),
    )


def _pattern_to_afferent_fibers(
    pattern: StimPattern,
    config: SimulationConfig,
    structural_state: _StructuralState,
) -> _TransductionResult:
    """Dispatch pulse-to-afferent mapping according to the configured control stage."""

    mode = str(config.transduction_config.mode).lower()
    if mode == "strength_duration":
        return _transduce_pattern_to_afferent_fibers(pattern, config, structural_state)
    raise ValueError(f"Unsupported transduction mode: {config.transduction_config.mode}")


def transduce_pattern_to_afferent_fibers(
    pattern: StimPattern,
    config: SimulationConfig,
    structural_state: _StructuralState | None = None,
) -> _TransductionResult:
    """Public wrapper used by analysis code that needs realized recruitment traces."""

    state = structural_state or get_structural_state(config)
    return _pattern_to_afferent_fibers(pattern, config, state)
