"""NEURON backend loading and per-condition execution."""

from __future__ import annotations

import importlib
import platform
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import PatientConditionSpec, SimulationConfig, SimulationResult, StimPattern, coerce_seed_sequence
from ..utils import progress
from .drive import SupraspinalDrive, build_supraspinal_drive, sample_supraspinal_spike_trains
from .structural import _StructuralState, get_structural_state
from .transduction import _TransductionResult, _pattern_to_afferent_fibers

_LOADED_MECHANISM_LIBRARIES: set[str] = set()


def neuron_available() -> bool:
    """Return whether the `neuron` Python package is currently importable."""

    try:
        importlib.import_module("neuron")
    except ImportError:
        return False
    return True


def _external_repo_root(config: SimulationConfig) -> Path:
    """Return the checked-out simulator root."""

    return Path(config.external_root).resolve() / "SCSInSCIMechanisms"


def _ensure_external_path(config: SimulationConfig) -> Path:
    """Make the simulator repo importable by Python."""

    repo_root = _external_repo_root(config)
    if not repo_root.exists():
        raise FileNotFoundError(
            f"Expected upstream simulator at {repo_root}. Run scripts/setup_external.sh first."
        )
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)
    return repo_root


def _compiled_mechanism_library(repo_root: Path) -> Path:
    """Return the preferred compiled mechanism library."""

    preferred = [
        repo_root / "arm64" / "libnrnmech.dylib",
        repo_root / "x86_64" / "libnrnmech.dylib",
        repo_root / "arm64" / "libnrnmech.so",
        repo_root / "x86_64" / "libnrnmech.so",
    ]
    for candidate in preferred:
        if candidate.exists():
            return candidate
    discovered = sorted(repo_root.glob("**/libnrnmech.dylib")) + sorted(repo_root.glob("**/libnrnmech.so"))
    if not discovered:
        raise FileNotFoundError(
            f"No compiled NEURON mechanisms found under {repo_root}. Run scripts/build_neuron.sh first."
        )
    return discovered[0]


def _load_mechanism_library(h: Any, library: Path) -> None:
    """Load the compiled mechanism library once per process."""

    library_key = str(library.resolve())
    if library_key in _LOADED_MECHANISM_LIBRARIES:
        return
    try:
        h.nrn_load_dll(library_key)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "already exists" in message or "already loaded" in message:
            _LOADED_MECHANISM_LIBRARIES.add(library_key)
            return
        remediation = ""
        if platform.system() == "Darwin":
            remediation = (
                "\nmacOS remediation:\n"
                f"  xattr -dr com.apple.quarantine {library.parent}\n"
                f"  codesign --force --sign - {library}"
            )
        raise RuntimeError(
            f"Failed to load compiled NEURON mechanisms from {library}: {exc}.{remediation}"
        ) from exc
    _LOADED_MECHANISM_LIBRARIES.add(library_key)


def _load_neuron_backend(config: SimulationConfig) -> tuple[Any, Any, Any]:
    """Load NEURON plus the simulator cells and modules."""

    repo_root = _ensure_external_path(config)
    from neuron import h  # Imported lazily so unit tests can run without NEURON.

    _load_mechanism_library(h, _compiled_mechanism_library(repo_root))

    cll = importlib.import_module("cells")
    nf = importlib.import_module("tools.neuron_functions")
    return h, cll, nf


def _set_trial_seed(seed: int) -> None:
    """Synchronize Python and NumPy RNGs for one trial seed."""

    np.random.seed(int(seed))
    random.seed(int(seed))


def _create_vecstim_population(h: Any, spike_trains: Iterable[np.ndarray]) -> tuple[list[Any], list[Any]]:
    """Build a VecStim population from explicit spike-time arrays."""

    neurons: list[Any] = []
    vectors: list[Any] = []
    for spike_times in spike_trains:
        vec_stim = h.VecStim()
        vec = h.Vector(np.asarray(spike_times, dtype=float).tolist())
        vec_stim.play(vec)
        neurons.append(vec_stim)
        vectors.append(vec)
    return neurons, vectors


def _create_supraspinal_population(
    h: Any,
    nf: Any,
    config: SimulationConfig,
    trial_seed: int,
    num_supraspinal: int,
) -> tuple[list[Any], list[Any], list[Any], SupraspinalDrive]:
    """Create supraspinal VecStim inputs and recorders from the local drive generator."""

    if num_supraspinal <= 0:
        drive = build_supraspinal_drive(config, int(trial_seed))
        return [], [], [], drive
    full_spike_trains, drive = sample_supraspinal_spike_trains(
        config,
        int(trial_seed),
        num_neurons=config.num_supraspinal_total,
    )
    supraspinal_neurons, supraspinal_vectors = _create_vecstim_population(h, full_spike_trains[:num_supraspinal])
    supraspinal_spike_times = nf.create_spike_recorder_input_neurons(supraspinal_neurons)
    return supraspinal_neurons, supraspinal_spike_times, supraspinal_vectors, drive


def _run_neuron_condition(
    condition: PatientConditionSpec,
    stim_pattern: StimPattern,
    trial_seed: int,
    config: SimulationConfig,
    structural_state: _StructuralState,
    *,
    transduction: _TransductionResult | None = None,
) -> SimulationResult:
    """Run the mechanistic fine-motor-task simulation."""

    h, cll, nf = _load_neuron_backend(config)
    h.load_file("stdrun.hoc")

    num_supraspinal = int(round(config.num_supraspinal_total * condition.perc_supra_intact))
    transduction_result = transduction or _pattern_to_afferent_fibers(
        stim_pattern,
        config=config,
        structural_state=structural_state,
    )

    _set_trial_seed(trial_seed)
    supraspinal_neurons, supraspinal_spike_times, supraspinal_vectors, supraspinal_drive = _create_supraspinal_population(
        h,
        nf,
        config,
        trial_seed,
        num_supraspinal,
    )

    scs_neurons, scs_vectors = _create_vecstim_population(h, transduction_result.afferent_pulse_times)

    mns = [
        cll.MotoneuronNoDendrites("WT", drug=config.mn_drug, L=structural_state.mn_lengths[index])
        for index in range(config.num_mn)
    ]
    mn_spike_recorders = nf.create_spike_recorder_mns(mns)

    _ = (supraspinal_vectors, scs_vectors)
    syn_supraspinal = nc_supraspinal = syn_scs = nc_scs = None
    if num_supraspinal > 0:
        syn_supraspinal, nc_supraspinal = nf.create_exponential_synapses(
            supraspinal_neurons,
            mns,
            structural_state.W_supraspinal[:num_supraspinal],
            config.synapse_tau_ms,
        )
    syn_scs, nc_scs = nf.create_exponential_synapses(
        scs_neurons,
        mns,
        structural_state.W_scs,
        config.synapse_tau_ms,
        structural_state.scs_delay,
    )

    h.finitialize()
    h.tstop = config.simulation_duration_ms
    h.run()

    mn_spike_times = [np.asarray(vector) if len(vector) > 0 else np.asarray([], dtype=float) for vector in mn_spike_recorders]
    supraspinal_spikes = [np.asarray(vector) if len(vector) > 0 else np.asarray([], dtype=float) for vector in supraspinal_spike_times]

    _set_trial_seed(trial_seed + 10_000)
    emg_signal = nf.estimate_emg_signal(mn_spike_times, simulation_duration=config.simulation_duration_ms)
    return SimulationResult(
        condition_label=condition.label,
        trial_seed=int(trial_seed),
        structural_seed=int(config.structural_seed),
        backend=config.backend,
        perc_supra_intact=condition.perc_supra_intact,
        stim_pattern=stim_pattern,
        emg_signal=np.asarray(emg_signal, dtype=float),
        mn_spike_times=mn_spike_times,
        supraspinal_spike_times=supraspinal_spikes,
        scs_pulse_times=transduction_result.afferent_pulse_times,
        metadata={
            "num_supraspinal": num_supraspinal,
            "supraspinal_seed": int(trial_seed),
            "emg_seed": int(trial_seed + 10_000),
            **supraspinal_drive.metadata,
            "supraspinal_mean_rate_hz": float(np.mean(supraspinal_drive.rate_hz)),
            "supraspinal_max_rate_hz": float(np.max(supraspinal_drive.rate_hz)),
            "mean_afferent_recruitment_fraction": (
                float(np.mean(transduction_result.pulse_recruitment_fraction))
                if transduction_result.pulse_recruitment_fraction.size
                else 0.0
            ),
            "kept_refs": all(reference is not None for reference in (syn_scs, nc_scs)),
        },
    )


def run_single_condition(
    condition: PatientConditionSpec,
    stim_pattern: StimPattern,
    trial_seed: int,
    config: SimulationConfig,
    structural_state: _StructuralState | None = None,
    *,
    transduction: _TransductionResult | None = None,
) -> SimulationResult:
    """Run one seed for one condition using the configured backend."""

    state = structural_state or get_structural_state(config)
    backend = config.backend.lower()
    if backend == "neuron":
        if not neuron_available():
            raise RuntimeError(
                "NEURON backend requested but `import neuron` failed. "
                "Install the project with `uv pip install -e \".[optim,dev]\"` first."
            )
        return _run_neuron_condition(condition, stim_pattern, trial_seed, config, state, transduction=transduction)
    raise ValueError(f"Unsupported backend: {config.backend}. Only `neuron` is supported.")


def run_condition(
    condition: PatientConditionSpec,
    stim_pattern: StimPattern,
    seeds: Iterable[int],
    config: SimulationConfig,
    *,
    progress_desc: str | None = None,
    structural_state: _StructuralState | None = None,
    transduction: _TransductionResult | None = None,
) -> list[SimulationResult]:
    """Run the same condition over multiple trial seeds."""

    state = structural_state or get_structural_state(config)
    seeds_tuple = coerce_seed_sequence(seeds, config.seed_config.train_seeds)
    return [
        run_single_condition(condition, stim_pattern, int(seed), config, state, transduction=transduction)
        for seed in progress(seeds_tuple, desc=progress_desc)
    ]
