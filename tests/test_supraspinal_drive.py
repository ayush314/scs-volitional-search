"""Tests for local supraspinal drive generation and spike sampling."""

from __future__ import annotations

import numpy as np

from scs_search.config import SimulationConfig
from scs_search.simulation.drive import build_supraspinal_drive, sample_supraspinal_spike_trains


def test_same_seed_produces_same_aperiodic_rate_profile() -> None:
    config = SimulationConfig()

    first = build_supraspinal_drive(config, 101)
    second = build_supraspinal_drive(config, 101)

    assert np.allclose(first.envelope, second.envelope)
    assert np.allclose(first.rate_hz, second.rate_hz)


def test_different_seeds_produce_different_aperiodic_rate_profiles() -> None:
    config = SimulationConfig()

    first = build_supraspinal_drive(config, 101)
    second = build_supraspinal_drive(config, 202)

    assert not np.allclose(first.rate_hz, second.rate_hz)


def test_aperiodic_envelope_and_rate_stay_within_bounds() -> None:
    config = SimulationConfig()

    drive = build_supraspinal_drive(config, 101)

    assert np.all((drive.envelope >= 0.0) & (drive.envelope <= 1.0))
    assert np.all((drive.rate_hz >= config.supraspinal_rate_floor_hz) & (drive.rate_hz <= config.supraspinal_rate_hz))
    assert drive.metadata["supraspinal_task_burst_count"] >= 1


def test_aperiodic_drive_has_real_low_drive_and_high_drive_intervals() -> None:
    config = SimulationConfig()

    drive = build_supraspinal_drive(config, 101)
    low_drive = drive.rate_hz <= (config.supraspinal_rate_floor_hz + 0.05 * (config.supraspinal_rate_hz - config.supraspinal_rate_floor_hz))
    high_drive = drive.rate_hz >= (config.supraspinal_rate_floor_hz + 0.5 * (config.supraspinal_rate_hz - config.supraspinal_rate_floor_hz))

    assert np.mean(low_drive) > 0.15
    assert np.mean(high_drive) > 0.10


def test_aperiodic_drive_is_not_half_periodic() -> None:
    config = SimulationConfig(simulation_duration_ms=1000)

    drive = build_supraspinal_drive(config, 101)
    first_half = drive.rate_hz[:500]
    second_half = drive.rate_hz[500:1000]

    assert not np.allclose(first_half, second_half)
    assert not any(np.allclose(first_half, np.roll(second_half, shift)) for shift in range(second_half.size))


def test_sinusoidal_mode_is_deterministic_and_periodic() -> None:
    config = SimulationConfig(supraspinal_drive_mode="sinusoidal")

    first = build_supraspinal_drive(config, 101)
    second = build_supraspinal_drive(config, 202)

    assert np.allclose(first.rate_hz, second.rate_hz)
    assert np.allclose(first.rate_hz[:500], first.rate_hz[500:1000], atol=1e-6)
    assert np.all(first.rate_hz >= 0.0)
    assert np.all(first.rate_hz <= config.supraspinal_rate_hz)


def test_spike_sampling_is_deterministic_for_same_seed() -> None:
    config = SimulationConfig()

    trains_a, drive_a = sample_supraspinal_spike_trains(config, 101, num_neurons=4)
    trains_b, drive_b = sample_supraspinal_spike_trains(config, 101, num_neurons=4)

    assert np.allclose(drive_a.rate_hz, drive_b.rate_hz)
    assert len(trains_a) == len(trains_b) == 4
    for spikes_a, spikes_b in zip(trains_a, trains_b):
        assert np.array_equal(spikes_a, spikes_b)


def test_spike_sampling_changes_with_trial_seed() -> None:
    config = SimulationConfig()

    trains_a, _ = sample_supraspinal_spike_trains(config, 101, num_neurons=4)
    trains_b, _ = sample_supraspinal_spike_trains(config, 202, num_neurons=4)

    assert any(not np.array_equal(spikes_a, spikes_b) for spikes_a, spikes_b in zip(trains_a, trains_b))
