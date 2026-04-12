"""Tests for reference EMG cache reuse and pulse transduction."""

from __future__ import annotations

import numpy as np

from scs_search.config import DeviceConfig, ParameterBounds, PatternParameters, SimulationConfig
from scs_search.dose import compute_pattern_dose
from scs_search.patterns import generate_stim_pattern
from scs_search.simulator_adapter import (
    _make_structural_state,
    _transduce_pattern_to_afferent_fibers,
    evaluate_pattern,
    load_reference_emg_cache,
    resolve_reference_emg_cache,
)


def test_load_reference_emg_cache_reads_healthy_arrays_only(tmp_path) -> None:
    np.savez(
        tmp_path / "emg_arrays.npz",
        healthy_prelesion_seed_101=np.array([1.0, 2.0]),
        lesion_no_stim_seed_101=np.array([3.0, 4.0]),
    )

    cache = load_reference_emg_cache(tmp_path, [101, 202])

    assert set(cache) == {101}
    assert np.allclose(cache[101], np.array([1.0, 2.0]))


def test_resolve_reference_emg_cache_builds_only_missing_seeds(tmp_path, monkeypatch) -> None:
    np.savez(tmp_path / "emg_arrays.npz", healthy_prelesion_seed_101=np.array([1.0, 2.0]))

    def fake_build_reference_emg_cache(seeds, _config):
        assert tuple(seeds) == (202,)
        return {202: np.array([5.0, 6.0])}

    monkeypatch.setattr(
        "scs_search.simulator_adapter.build_reference_emg_cache",
        fake_build_reference_emg_cache,
    )

    cache = resolve_reference_emg_cache([101, 202], SimulationConfig(), reference_dir=tmp_path)

    assert set(cache) == {101, 202}
    assert np.allclose(cache[101], np.array([1.0, 2.0]))
    assert np.allclose(cache[202], np.array([5.0, 6.0]))
    persisted = load_reference_emg_cache(tmp_path, [101, 202])
    assert set(persisted) == {101, 202}


def test_transduction_matches_nested_recruitment_order_at_max_pulse_width() -> None:
    config = SimulationConfig()
    state = _make_structural_state(config)
    theta = PatternParameters(
        f=100.0,
        pw_us=config.device_config.max_pulse_width_us,
        T_on=1000.0,
        T_off=0.0,
        alpha0=0.5,
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )
    pattern = generate_stim_pattern(theta, t_end_ms=20, device_config=config.device_config)
    transduction = _transduce_pattern_to_afferent_fibers(pattern, config, state)

    recruited = {index for index, times in enumerate(transduction.afferent_pulse_times) if times.size}
    expected = set(state.recruitment_order[: config.num_scs_total // 2])
    assert recruited == expected
    assert np.isclose(transduction.pulse_recruitment_fraction[0], 0.5)


def test_wider_pulses_recruit_at_least_as_many_afferents_for_same_amplitude() -> None:
    config = SimulationConfig()
    state = _make_structural_state(config)
    narrow_theta = PatternParameters(
        f=100.0,
        pw_us=60.0,
        T_on=1000.0,
        T_off=0.0,
        alpha0=0.2,
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )
    wide_theta = PatternParameters(
        f=100.0,
        pw_us=1000.0,
        T_on=1000.0,
        T_off=0.0,
        alpha0=0.2,
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )
    narrow_pattern = generate_stim_pattern(narrow_theta, t_end_ms=20, device_config=config.device_config)
    wide_pattern = generate_stim_pattern(wide_theta, t_end_ms=20, device_config=config.device_config)

    narrow = _transduce_pattern_to_afferent_fibers(narrow_pattern, config, state)
    wide = _transduce_pattern_to_afferent_fibers(wide_pattern, config, state)

    assert np.all(wide.pulse_recruitment_fraction >= narrow.pulse_recruitment_fraction)
    assert wide.pulse_recruitment_fraction[0] > narrow.pulse_recruitment_fraction[0]


def test_relative_refractory_reduces_recruitment_at_high_following_rates() -> None:
    theta = PatternParameters(
        f=500.0,
        pw_us=1000.0,
        T_on=1000.0,
        T_off=0.0,
        alpha0=1.0,
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )
    with_relative = SimulationConfig()
    without_relative = SimulationConfig(
        transduction_config=with_relative.transduction_config.__class__(
            enforce_no_overlap=True,
            chronaxie_us=with_relative.transduction_config.chronaxie_us,
            absolute_refractory_ms=with_relative.transduction_config.absolute_refractory_ms,
            relative_refractory_end_ms=with_relative.transduction_config.absolute_refractory_ms,
        )
    )
    state = _make_structural_state(with_relative)
    pattern = generate_stim_pattern(theta, t_end_ms=10, device_config=with_relative.device_config)

    relative = _transduce_pattern_to_afferent_fibers(pattern, with_relative, state)
    no_relative = _transduce_pattern_to_afferent_fibers(pattern, without_relative, state)

    assert np.isclose(no_relative.pulse_recruitment_fraction[0], 1.0)
    assert np.any(relative.pulse_recruitment_fraction[1:] < no_relative.pulse_recruitment_fraction[1:])
    assert np.mean(relative.pulse_recruitment_fraction[1:]) < np.mean(
        no_relative.pulse_recruitment_fraction[1:]
    )


def test_realized_recruitment_dose_comes_from_transduced_afferent_spikes() -> None:
    config = SimulationConfig()
    state = _make_structural_state(config)
    narrow_theta = PatternParameters(
        f=100.0,
        pw_us=60.0,
        T_on=1000.0,
        T_off=0.0,
        alpha0=0.2,
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )
    wide_theta = PatternParameters(
        f=100.0,
        pw_us=1000.0,
        T_on=1000.0,
        T_off=0.0,
        alpha0=0.2,
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )
    narrow_pattern = generate_stim_pattern(narrow_theta, t_end_ms=20, device_config=config.device_config)
    wide_pattern = generate_stim_pattern(wide_theta, t_end_ms=20, device_config=config.device_config)
    narrow_transduction = _transduce_pattern_to_afferent_fibers(narrow_pattern, config, state)
    wide_transduction = _transduce_pattern_to_afferent_fibers(wide_pattern, config, state)

    narrow_metrics = compute_pattern_dose(
        narrow_pattern,
        pulse_recruitment_fraction=narrow_transduction.pulse_recruitment_fraction,
        dose_config=config.dose_config,
        device_config=config.device_config,
    )
    wide_metrics = compute_pattern_dose(
        wide_pattern,
        pulse_recruitment_fraction=wide_transduction.pulse_recruitment_fraction,
        dose_config=config.dose_config,
        device_config=config.device_config,
    )

    assert wide_metrics["raw_recruitment_dose"] > narrow_metrics["raw_recruitment_dose"]


def test_invalid_pattern_returns_invalid_summary_without_running_neuron() -> None:
    device = DeviceConfig(
        max_total_current_ma=20.0,
        min_pulse_width_us=60.0,
        max_pulse_width_us=1000.0,
        pulse_width_step_us=10.0,
        max_master_rate_hz=1200.0,
        default_pulse_width_us=210.0,
    )
    bounds = ParameterBounds(
        lower=(10.0, 60.0, 50.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0),
        upper=(1200.0, 1000.0, 500.0, 500.0, 0.9, 0.5, 2.0 * np.pi, 0.5, 2.0 * np.pi),
    )
    config = SimulationConfig(device_config=device, theta_bounds=bounds)
    theta = PatternParameters(
        f=1200.0,
        pw_us=1000.0,
        T_on=100.0,
        T_off=0.0,
        alpha0=0.5,
        alpha1=0.0,
        phi1=0.0,
        alpha2=0.0,
        phi2=0.0,
    )

    summary = evaluate_pattern(theta, seeds=(101, 202), config=config)

    assert summary.valid is False
    assert summary.invalid_reason == "pulse_overlap"
    assert summary.mean_corr == config.dose_config.invalid_objective_floor
    assert all(record["valid"] is False for record in summary.per_seed_records)
