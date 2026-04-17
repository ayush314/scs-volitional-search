"""Tests for reference EMG cache reuse and pulse transduction."""

from __future__ import annotations

import numpy as np

from scs_search.config import PatientConditionSpec, PhysicalModulationParameters, SimulationConfig
from scs_search.dose import compute_pattern_dose
from scs_search.simulation import backend as simulation_backend
from scs_search.simulation import drive as simulation_drive
from scs_search.simulation import evaluator as simulation_evaluator
from scs_search.simulation.backend import _create_supraspinal_population, _run_neuron_condition
from scs_search.simulation.drive import SupraspinalDrive
from scs_search.simulation.evaluator import load_reference_emg_cache, resolve_reference_emg_cache
from scs_search.simulation.structural import make_structural_state
from scs_search.simulation.transduction import _transduce_pattern_to_afferent_fibers
from scs_search.stimulation.patterns import generate_stim_pattern, generate_tonic_pattern


class _FakeVector(list):
    def __init__(self, values=()):
        super().__init__(values)


class _FakeVecStim:
    def __init__(self) -> None:
        self.played: list[float] = []

    def play(self, vector: _FakeVector) -> None:
        self.played = list(vector)


class _FakeH:
    def __init__(self) -> None:
        self.tstop = 0.0

    def load_file(self, _path: str) -> None:
        return None

    def VecStim(self) -> _FakeVecStim:
        return _FakeVecStim()

    def Vector(self, values=()) -> _FakeVector:
        return _FakeVector(values)

    def finitialize(self) -> None:
        return None

    def run(self) -> None:
        return None


class _FakeNF:
    def create_spike_recorder_input_neurons(self, neurons):
        return [_FakeVector(neuron.played) for neuron in neurons]

    def create_spike_recorder_mns(self, neurons):
        return [_FakeVector() for _ in neurons]

    def create_exponential_synapses(self, *_args, **_kwargs):
        return [object()], [object()]

    def estimate_emg_signal(self, _mn_spike_times, simulation_duration):
        return np.zeros(int(simulation_duration), dtype=float)


class _FakeCLL:
    @staticmethod
    def MotoneuronNoDendrites(*_args, **_kwargs):
        return object()


def _fake_drive(config: SimulationConfig, trial_seed: int) -> SupraspinalDrive:
    time_ms = np.arange(0.0, float(config.simulation_duration_ms), 1.0, dtype=float)
    rate_hz = np.full(time_ms.shape, float(config.supraspinal_rate_floor_hz), dtype=float)
    return SupraspinalDrive(
        mode=str(config.supraspinal_drive_mode),
        time_ms=time_ms,
        envelope=np.zeros_like(time_ms, dtype=float),
        rate_hz=rate_hz,
        metadata={
            "supraspinal_drive_mode": str(config.supraspinal_drive_mode),
            "supraspinal_rate_floor_hz": float(config.supraspinal_rate_floor_hz),
            "supraspinal_trial_seed": int(trial_seed),
        },
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

    monkeypatch.setattr(simulation_evaluator, "build_reference_emg_cache", fake_build_reference_emg_cache)

    cache = resolve_reference_emg_cache([101, 202], SimulationConfig(), reference_dir=tmp_path)

    assert set(cache) == {101, 202}
    assert np.allclose(cache[101], np.array([1.0, 2.0]))
    assert np.allclose(cache[202], np.array([5.0, 6.0]))


def test_transduction_matches_nested_recruitment_order_at_max_pulse_width() -> None:
    config = SimulationConfig()
    state = make_structural_state(config)
    pattern = generate_tonic_pattern(
        freq_hz=100.0,
        current_ma=0.5 * config.device_config.max_total_current_ma,
        pw_us=config.device_config.max_pulse_width_us,
        t_end_ms=20,
        device_config=config.device_config,
    )

    transduction = _transduce_pattern_to_afferent_fibers(pattern, config, state)

    recruited = {index for index, times in enumerate(transduction.afferent_pulse_times) if times.size}
    expected = set(state.recruitment_order[: config.num_scs_total // 2])
    assert recruited == expected
    assert np.isclose(transduction.pulse_recruitment_fraction[0], 0.5)


def test_wider_pulses_recruit_at_least_as_many_afferents_for_same_amplitude() -> None:
    config = SimulationConfig()
    state = make_structural_state(config)
    narrow_pattern = generate_tonic_pattern(
        freq_hz=100.0,
        current_ma=0.2 * config.device_config.max_total_current_ma,
        pw_us=60.0,
        t_end_ms=20,
        device_config=config.device_config,
    )
    wide_pattern = generate_tonic_pattern(
        freq_hz=100.0,
        current_ma=0.2 * config.device_config.max_total_current_ma,
        pw_us=config.device_config.max_pulse_width_us,
        t_end_ms=20,
        device_config=config.device_config,
    )

    narrow = _transduce_pattern_to_afferent_fibers(narrow_pattern, config, state)
    wide = _transduce_pattern_to_afferent_fibers(wide_pattern, config, state)

    assert np.all(wide.pulse_recruitment_fraction >= narrow.pulse_recruitment_fraction)
    assert wide.pulse_recruitment_fraction[0] > narrow.pulse_recruitment_fraction[0]


def test_relative_refractory_reduces_recruitment_at_high_following_rates() -> None:
    theta = PhysicalModulationParameters(
        I0_ma=20.0,
        I1_ma=0.0,
        f0_hz=400.0,
        f1_hz=0.0,
        PW1_us=150.0,
        T_ms=200.0,
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
    state = make_structural_state(with_relative)
    pattern = generate_stim_pattern(theta, t_end_ms=10, device_config=with_relative.device_config)

    relative = _transduce_pattern_to_afferent_fibers(pattern, with_relative, state)
    no_relative = _transduce_pattern_to_afferent_fibers(pattern, without_relative, state)

    assert np.isclose(relative.pulse_recruitment_fraction[0], no_relative.pulse_recruitment_fraction[0])
    assert np.any(relative.pulse_recruitment_fraction[1:] < no_relative.pulse_recruitment_fraction[1:])


def test_realized_recruitment_dose_comes_from_transduced_afferent_spikes() -> None:
    config = SimulationConfig()
    state = make_structural_state(config)
    narrow_pattern = generate_tonic_pattern(
        freq_hz=100.0,
        current_ma=0.2 * config.device_config.max_total_current_ma,
        pw_us=60.0,
        t_end_ms=20,
        device_config=config.device_config,
    )
    wide_pattern = generate_tonic_pattern(
        freq_hz=100.0,
        current_ma=0.2 * config.device_config.max_total_current_ma,
        pw_us=config.device_config.max_pulse_width_us,
        t_end_ms=20,
        device_config=config.device_config,
    )
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


def test_create_supraspinal_population_uses_local_spike_sampler(monkeypatch) -> None:
    config = SimulationConfig(num_supraspinal_total=4, simulation_duration_ms=10)
    h = _FakeH()
    nf = _FakeNF()
    calls: list[tuple[int, int]] = []

    def fake_sample(_config, trial_seed, *, num_neurons=None):
        calls.append((int(trial_seed), int(num_neurons)))
        spike_trains = [
            np.asarray([0.0, 4.0], dtype=float),
            np.asarray([1.0], dtype=float),
            np.asarray([2.0], dtype=float),
            np.asarray([3.0], dtype=float),
        ]
        return spike_trains, _fake_drive(_config, int(trial_seed))

    monkeypatch.setattr(simulation_backend, "sample_supraspinal_spike_trains", fake_sample)

    neurons, spike_times, vectors, drive = _create_supraspinal_population(
        h,
        nf,
        config,
        trial_seed=101,
        num_supraspinal=2,
    )

    assert calls == [(101, 4)]
    assert len(neurons) == len(spike_times) == len(vectors) == 2
    assert np.array_equal(np.asarray(spike_times[0], dtype=float), np.asarray([0.0, 4.0], dtype=float))
    assert drive.metadata["supraspinal_drive_mode"] == "aperiodic_envelope"


def test_run_neuron_condition_uses_local_supraspinal_population(monkeypatch) -> None:
    config = SimulationConfig(
        num_scs_total=4,
        num_supraspinal_total=5,
        num_mn=2,
        simulation_duration_ms=10,
    )
    theta = PhysicalModulationParameters(
        I0_ma=8.0,
        I1_ma=0.0,
        f0_hz=100.0,
        f1_hz=0.0,
        PW1_us=0.0,
        T_ms=500.0,
    )
    pattern = generate_stim_pattern(theta, t_end_ms=10, device_config=config.device_config)
    state = make_structural_state(config)
    condition = PatientConditionSpec(label="lesion", perc_supra_intact=0.4)
    calls: list[tuple[int, int]] = []

    def fake_create_population(_h, _nf, _config, trial_seed, num_supraspinal):
        calls.append((int(trial_seed), int(num_supraspinal)))
        neurons = [_FakeVecStim() for _ in range(num_supraspinal)]
        spike_times = [_FakeVector([0.0, 4.0]) for _ in range(num_supraspinal)]
        vectors = [_FakeVector([0.0, 4.0]) for _ in range(num_supraspinal)]
        return neurons, spike_times, vectors, _fake_drive(_config, int(trial_seed))

    monkeypatch.setattr(simulation_backend, "_load_neuron_backend", lambda _config: (_FakeH(), _FakeCLL(), _FakeNF()))
    monkeypatch.setattr(simulation_backend, "_create_supraspinal_population", fake_create_population)

    result = _run_neuron_condition(condition, pattern, 101, config, state)

    assert calls == [(101, 2)]
    assert result.metadata["supraspinal_drive_mode"] == "aperiodic_envelope"
    assert result.metadata["num_supraspinal"] == 2


def test_run_condition_iterates_requested_seeds(monkeypatch) -> None:
    config = SimulationConfig()
    condition = PatientConditionSpec(label="lesion", perc_supra_intact=0.2)
    pattern = generate_tonic_pattern(freq_hz=20.0, alpha=0.0, t_end_ms=10)
    calls: list[int] = []

    monkeypatch.setattr(simulation_backend, "progress", lambda iterable, **_kwargs: iterable)
    monkeypatch.setattr(simulation_backend, "get_structural_state", lambda _config: object())

    def fake_run_single_condition(_condition, _stim_pattern, trial_seed, _config, _state, *, transduction=None):
        calls.append(int(trial_seed))
        return f"seed-{trial_seed}"

    monkeypatch.setattr(simulation_backend, "run_single_condition", fake_run_single_condition)

    results = simulation_backend.run_condition(condition, pattern, seeds=[101, 202], config=config)

    assert calls == [101, 202]
    assert results == ["seed-101", "seed-202"]
