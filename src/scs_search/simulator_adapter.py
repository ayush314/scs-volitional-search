"""Adapter for the fine motor task with patterned SCS."""

from __future__ import annotations

import importlib
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import (
    PatientConditionSpec,
    EvaluationSummary,
    PatternParameters,
    SimulationConfig,
    SimulationResult,
    StimPattern,
    coerce_seed_sequence,
    condition_defaults,
)
from .dose import combined_objective, compute_pattern_dose
from .metrics import compute_emg_similarity, mean_and_std_over_seeds
from .patterns import generate_stim_pattern, generate_tonic_pattern, invalid_theta_reason
from .plotting import display_name, lesion_label, plot_emg_seed_panels
from .utils import ensure_dir, progress, write_json


@dataclass(frozen=True)
class _StructuralState:
    """Fixed structural randomness shared across repeated trial seeds."""

    mn_lengths: np.ndarray
    W_supraspinal: np.ndarray
    W_scs: np.ndarray
    scs_delay: np.ndarray
    recruitment_order: np.ndarray


@dataclass(frozen=True)
class _TransductionResult:
    """Delivered-pulse transduction into afferent-fiber spike trains."""

    afferent_pulse_times: list[np.ndarray]
    pulse_recruitment_fraction: np.ndarray


_LOADED_MECHANISM_LIBRARIES: set[str] = set()
_HEALTHY_REFERENCE_PREFIX = "healthy_prelesion_seed_"


def neuron_available() -> bool:
    """Return whether the `neuron` Python package is currently importable."""

    try:
        importlib.import_module("neuron")
    except ImportError:
        return False
    return True


def _external_repo_root(config: SimulationConfig) -> Path:
    """Return the checked-out upstream simulator root."""

    return Path(config.external_root).resolve() / "SCSInSCIMechanisms"


def _ensure_external_path(config: SimulationConfig) -> Path:
    """Make the upstream repo importable by Python."""

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
    """Load NEURON plus the upstream cells and modules."""

    repo_root = _ensure_external_path(config)
    from neuron import h  # Imported lazily so unit tests can run without NEURON.

    _load_mechanism_library(h, _compiled_mechanism_library(repo_root))

    cll = importlib.import_module("cells")
    nf = importlib.import_module("tools.neuron_functions")
    return h, cll, nf


def _make_structural_state(config: SimulationConfig) -> _StructuralState:
    """Generate the structural randomness once and reuse it across trial seeds."""

    rng = np.random.RandomState(config.structural_seed)
    mn_lengths = config.mn_avg_diameter + rng.randn(config.num_mn) * 0.1 * config.mn_avg_diameter
    W_supraspinal = rng.gamma(
        config.synapse_shape,
        scale=config.synaptic_weight_supra / config.synapse_shape,
        size=(config.num_supraspinal_total, config.num_mn),
    )
    W_scs = rng.gamma(
        config.synapse_shape,
        scale=config.synaptic_weight_scs / config.synapse_shape,
        size=(config.num_scs_total, config.num_mn),
    )
    scs_delay = rng.lognormal(-0.47, 0.37, size=(config.num_scs_total, config.num_mn))
    recruitment_order = rng.permutation(config.num_scs_total)
    return _StructuralState(
        mn_lengths=mn_lengths,
        W_supraspinal=W_supraspinal,
        W_scs=W_scs,
        scs_delay=scs_delay,
        recruitment_order=recruitment_order,
    )


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
            effective_threshold_ma[recovering] = (
                effective_threshold_ma[recovering] / recovery_fraction
            )
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


def _set_trial_seed(seed: int) -> None:
    """Synchronize Python and NumPy RNGs for one trial seed."""

    np.random.seed(int(seed))
    random.seed(int(seed))


def _run_neuron_condition(
    condition: PatientConditionSpec,
    stim_pattern: StimPattern,
    trial_seed: int,
    config: SimulationConfig,
    structural_state: _StructuralState,
) -> SimulationResult:
    """Run the actual mechanistic fine-motor-task simulation."""

    h, cll, nf = _load_neuron_backend(config)
    h.load_file("stdrun.hoc")

    num_supraspinal = int(round(config.num_supraspinal_total * condition.perc_supra_intact))
    transduction = _transduce_pattern_to_afferent_fibers(
        stim_pattern,
        config=config,
        structural_state=structural_state,
    )

    _set_trial_seed(trial_seed)
    supraspinal_neurons = []
    supraspinal_spike_times = []
    if num_supraspinal > 0:
        supraspinal_neurons = nf.create_inhomogeneous_input_neurons(
            num_supraspinal,
            config.supraspinal_rate_hz,
            config.simulation_duration_ms,
            frequency=config.supraspinal_inhomogeneous_rate_hz,
        )
        supraspinal_spike_times = nf.create_spike_recorder_input_neurons(supraspinal_neurons)

    scs_neurons = []
    scs_vectors = []
    for pulse_times in transduction.afferent_pulse_times:
        vec_stim = h.VecStim()
        vec = h.Vector(pulse_times.tolist())
        vec_stim.play(vec)
        scs_neurons.append(vec_stim)
        scs_vectors.append(vec)

    mns = [
        cll.MotoneuronNoDendrites("WT", drug=config.mn_drug, L=structural_state.mn_lengths[index])
        for index in range(config.num_mn)
    ]
    mn_spike_recorders = nf.create_spike_recorder_mns(mns)

    _ = scs_vectors
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
    supraspinal_spikes = [
        np.asarray(vector) if len(vector) > 0 else np.asarray([], dtype=float)
        for vector in supraspinal_spike_times
    ]

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
        scs_pulse_times=transduction.afferent_pulse_times,
        metadata={
            "num_supraspinal": num_supraspinal,
            "supraspinal_seed": int(trial_seed),
            "emg_seed": int(trial_seed + 10_000),
            "mean_afferent_recruitment_fraction": (
                float(np.mean(transduction.pulse_recruitment_fraction))
                if transduction.pulse_recruitment_fraction.size
                else 0.0
            ),
            "kept_refs": all(
                reference is not None
                for reference in (syn_scs, nc_scs)
            ),
        },
    )

def run_single_condition(
    condition: PatientConditionSpec,
    stim_pattern: StimPattern,
    trial_seed: int,
    config: SimulationConfig,
    structural_state: _StructuralState | None = None,
) -> SimulationResult:
    """Run one seed for one condition using the configured backend."""

    state = structural_state or _make_structural_state(config)
    backend = config.backend.lower()
    if backend == "neuron":
        if not neuron_available():
            raise RuntimeError(
                "NEURON backend requested but `import neuron` failed. "
                "Install the project with `uv pip install -e \".[optim,dev]\"` first."
            )
        return _run_neuron_condition(condition, stim_pattern, trial_seed, config, state)
    raise ValueError(f"Unsupported backend: {config.backend}. Only `neuron` is supported.")


def run_condition(
    condition: PatientConditionSpec,
    stim_pattern: StimPattern,
    seeds: Iterable[int],
    config: SimulationConfig,
    *,
    progress_desc: str | None = None,
) -> list[SimulationResult]:
    """Run the same condition over multiple trial seeds."""

    structural_state = _make_structural_state(config)
    seeds_tuple = coerce_seed_sequence(seeds, config.seed_config.train_seeds)
    return [
        run_single_condition(condition, stim_pattern, int(seed), config, structural_state)
        for seed in progress(seeds_tuple, desc=progress_desc)
    ]


def _invalid_evaluation_summary(
    theta: PatternParameters,
    seeds: tuple[int, ...],
    config: SimulationConfig,
    invalid_reason: str,
    budget_norm: float | None,
) -> EvaluationSummary:
    """Return a non-simulated summary for an infeasible stimulation pattern."""

    floor = float(config.dose_config.invalid_objective_floor)
    per_seed_records = [
        {
            "seed": int(seed),
            "corr": floor,
            "raw_recruitment_dose": 0.0,
            "recruitment_dose_norm": 0.0,
            "device_cost": 1.0,
            "current_rate_usage": 1.0,
            "total_current_ma": 0.0,
            "charge_per_pulse_uc": 0.0,
            "charge_rate_uc_per_s": 0.0,
            "pulse_width_us": float(theta.pw_us),
            "backend": config.backend,
            "condition_label": "lesion_scs_invalid",
            "theta": theta.to_dict(),
            "valid": False,
            "invalid_reason": invalid_reason,
        }
        for seed in seeds
    ]
    return EvaluationSummary(
        theta=theta,
        family="fourier",
        seeds=seeds,
        per_seed_records=per_seed_records,
        mean_corr=floor,
        std_corr=0.0,
        mean_raw_dose=0.0,
        std_raw_dose=0.0,
        mean_norm_dose=0.0,
        std_norm_dose=0.0,
        mean_device_cost=1.0,
        std_device_cost=0.0,
        mean_current_rate_usage=1.0,
        std_current_rate_usage=0.0,
        mean_total_current_ma=0.0,
        std_total_current_ma=0.0,
        mean_charge_per_pulse_uc=0.0,
        std_charge_per_pulse_uc=0.0,
        mean_charge_rate_uc_per_s=0.0,
        std_charge_rate_uc_per_s=0.0,
        penalized_objective=floor,
        robust_objective=floor,
        valid=False,
        invalid_reason=invalid_reason,
        metadata={
            "budget_norm": budget_norm,
            "backend": config.backend,
            "pulse_width_us": float(theta.pw_us),
            "usage_metric": "normalized_charge_rate_usage",
            "valid": False,
            "invalid_reason": invalid_reason,
        },
    )


def evaluate_pattern(
    theta: PatternParameters | dict[str, float] | tuple[float, ...] | list[float],
    seeds: Iterable[int],
    config: SimulationConfig,
    budget_norm: float | None = None,
    *,
    reference_emg_by_seed: dict[int, np.ndarray] | None = None,
    robust_objective: bool = False,
) -> EvaluationSummary:
    """Evaluate lesion+SCS restoration against the healthy pre-lesion reference."""

    healthy_condition, lesion_condition = condition_defaults(config)
    theta_params = config.theta_bounds.clip(theta)
    seeds_tuple = coerce_seed_sequence(seeds, config.seed_config.train_seeds)
    invalid_reason = invalid_theta_reason(
        theta_params,
        config.device_config,
        enforce_no_overlap=config.transduction_config.enforce_no_overlap,
    )
    if invalid_reason is not None:
        return _invalid_evaluation_summary(theta_params, seeds_tuple, config, invalid_reason, budget_norm)
    stim_pattern = generate_stim_pattern(
        theta_params,
        t_end_ms=config.simulation_duration_ms,
        dt_ms=config.dt_ms,
        device_config=config.device_config,
    )
    zero_pattern = generate_tonic_pattern(
        freq_hz=max(theta_params.f, 1.0),
        alpha=0.0,
        t_end_ms=config.simulation_duration_ms,
        dt_ms=config.dt_ms,
        device_config=config.device_config,
    )

    structural_state = _make_structural_state(config)
    if reference_emg_by_seed is None:
        healthy_results = [
            run_single_condition(healthy_condition, zero_pattern, int(seed), config, structural_state)
            for seed in seeds_tuple
        ]
        reference_emg_by_seed = {result.trial_seed: result.emg_signal for result in healthy_results}
    lesion_results = [
        run_single_condition(
            PatientConditionSpec(label="lesion_scs", perc_supra_intact=lesion_condition.perc_supra_intact),
            stim_pattern,
            int(seed),
            config,
            structural_state,
        )
        for seed in seeds_tuple
    ]

    metric_values: list[float] = []
    raw_dose_values: list[float] = []
    norm_dose_values: list[float] = []
    device_cost_values: list[float] = []
    current_rate_values: list[float] = []
    total_current_values: list[float] = []
    charge_per_pulse_values: list[float] = []
    charge_rate_values: list[float] = []
    per_seed_records: list[dict[str, Any]] = []
    transduction = _transduce_pattern_to_afferent_fibers(stim_pattern, config, structural_state)
    dose_metrics = compute_pattern_dose(
        stim_pattern,
        pulse_recruitment_fraction=transduction.pulse_recruitment_fraction,
        dose_config=config.dose_config,
        device_config=config.device_config,
    )

    for lesion_result in lesion_results:
        seed = int(lesion_result.trial_seed)
        reference_emg = np.asarray(reference_emg_by_seed[seed], dtype=float)
        corr = compute_emg_similarity(
            reference_emg=reference_emg,
            candidate_emg=lesion_result.emg_signal,
            envelope_window_ms=config.metric_config.envelope_window_ms,
            max_lag_ms=config.metric_config.max_lag_ms,
            use_envelope=config.metric_config.use_envelope,
        )
        metric_values.append(corr)
        raw_dose_values.append(float(dose_metrics["raw_recruitment_dose"]))
        norm_dose_values.append(float(dose_metrics["recruitment_dose_norm"]))
        device_cost_values.append(float(dose_metrics["device_cost"]))
        current_rate_values.append(float(dose_metrics["current_rate_usage"]))
        total_current_values.append(float(dose_metrics["mean_total_current_ma"]))
        charge_per_pulse_values.append(float(dose_metrics["mean_charge_per_pulse_uc"]))
        charge_rate_values.append(float(dose_metrics["charge_rate_uc_per_s"]))
        per_seed_records.append(
            {
                "seed": seed,
                "corr": float(corr),
                "raw_recruitment_dose": float(dose_metrics["raw_recruitment_dose"]),
                "recruitment_dose_norm": float(dose_metrics["recruitment_dose_norm"]),
                "device_cost": float(dose_metrics["device_cost"]),
                "current_rate_usage": float(dose_metrics["current_rate_usage"]),
                "total_current_ma": float(dose_metrics["mean_total_current_ma"]),
                "charge_per_pulse_uc": float(dose_metrics["mean_charge_per_pulse_uc"]),
                "charge_rate_uc_per_s": float(dose_metrics["charge_rate_uc_per_s"]),
                "pulse_width_us": float(dose_metrics["pulse_width_us"]),
                "backend": lesion_result.backend,
                "condition_label": lesion_result.condition_label,
                "theta": theta_params.to_dict(),
                "valid": True,
                "invalid_reason": None,
            }
        )

    mean_corr, std_corr = mean_and_std_over_seeds(metric_values)
    mean_raw_dose, std_raw_dose = mean_and_std_over_seeds(raw_dose_values)
    mean_norm_dose, std_norm_dose = mean_and_std_over_seeds(norm_dose_values)
    mean_device_cost, std_device_cost = mean_and_std_over_seeds(device_cost_values)
    mean_current_rate_usage, std_current_rate_usage = mean_and_std_over_seeds(current_rate_values)
    mean_total_current_ma, std_total_current_ma = mean_and_std_over_seeds(total_current_values)
    mean_charge_per_pulse_uc, std_charge_per_pulse_uc = mean_and_std_over_seeds(charge_per_pulse_values)
    mean_charge_rate_uc_per_s, std_charge_rate_uc_per_s = mean_and_std_over_seeds(charge_rate_values)
    robust_score, penalized_score = combined_objective(
        mean_corr=mean_corr,
        std_corr=std_corr,
        device_cost=mean_device_cost,
        budget_norm=budget_norm,
        dose_config=config.dose_config,
        robust=robust_objective,
        theta=theta_params,
        pulse_recruitment_fraction=transduction.pulse_recruitment_fraction,
    )
    return EvaluationSummary(
        theta=theta_params,
        family=stim_pattern.family,
        seeds=seeds_tuple,
        per_seed_records=per_seed_records,
        mean_corr=mean_corr,
        std_corr=std_corr,
        mean_raw_dose=mean_raw_dose,
        std_raw_dose=std_raw_dose,
        mean_norm_dose=mean_norm_dose,
        std_norm_dose=std_norm_dose,
        mean_device_cost=mean_device_cost,
        std_device_cost=std_device_cost,
        mean_current_rate_usage=mean_current_rate_usage,
        std_current_rate_usage=std_current_rate_usage,
        mean_total_current_ma=mean_total_current_ma,
        std_total_current_ma=std_total_current_ma,
        mean_charge_per_pulse_uc=mean_charge_per_pulse_uc,
        std_charge_per_pulse_uc=std_charge_per_pulse_uc,
        mean_charge_rate_uc_per_s=mean_charge_rate_uc_per_s,
        std_charge_rate_uc_per_s=std_charge_rate_uc_per_s,
        penalized_objective=penalized_score,
        robust_objective=robust_score,
        valid=True,
        invalid_reason=None,
        metadata={
            "budget_norm": budget_norm,
            "backend": config.backend,
            "pulse_width_us": float(dose_metrics["pulse_width_us"]),
            "current_cap_ma": float(config.device_config.max_total_current_ma),
            "usage_metric": "normalized_charge_rate_usage",
            "valid": True,
        },
    )


def build_reference_emg_cache(seeds: Iterable[int], config: SimulationConfig) -> dict[int, np.ndarray]:
    """Generate healthy pre-lesion reference EMG traces keyed by seed."""

    healthy_condition, _ = condition_defaults(config)
    zero_pattern = generate_tonic_pattern(
        freq_hz=40.0,
        alpha=0.0,
        t_end_ms=config.simulation_duration_ms,
        dt_ms=config.dt_ms,
        device_config=config.device_config,
    )
    return {
        result.trial_seed: result.emg_signal
        for result in run_condition(
            healthy_condition,
            zero_pattern,
            seeds,
            config,
            progress_desc="Reference EMG",
        )
    }


def load_reference_emg_cache(reference_dir: str | Path, seeds: Iterable[int]) -> dict[int, np.ndarray]:
    """Load cached healthy pre-lesion EMG traces for the requested seeds."""

    seeds_tuple = tuple(int(seed) for seed in seeds)
    cache_path = Path(reference_dir) / "emg_arrays.npz"
    if not cache_path.exists():
        return {}
    loaded: dict[int, np.ndarray] = {}
    with np.load(cache_path) as arrays:
        for seed in seeds_tuple:
            key = f"{_HEALTHY_REFERENCE_PREFIX}{seed}"
            if key in arrays:
                loaded[seed] = np.asarray(arrays[key], dtype=float)
    return loaded


def _persist_reference_emg_cache(reference_dir: str | Path, cache_by_seed: dict[int, np.ndarray]) -> None:
    """Merge healthy reference traces into the on-disk reference cache."""

    reference_path = ensure_dir(reference_dir)
    cache_path = reference_path / "emg_arrays.npz"
    arrays: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path) as existing:
            arrays = {name: np.asarray(existing[name], dtype=float) for name in existing.files}
    for seed, emg_signal in cache_by_seed.items():
        arrays[f"{_HEALTHY_REFERENCE_PREFIX}{int(seed)}"] = np.asarray(emg_signal, dtype=float)
    np.savez(cache_path, **arrays)


def resolve_reference_emg_cache(
    seeds: Iterable[int],
    config: SimulationConfig,
    reference_dir: str | Path | None = None,
) -> dict[int, np.ndarray]:
    """Load saved healthy references first, then build only any missing seeds."""

    seeds_tuple = coerce_seed_sequence(seeds, config.seed_config.train_seeds)
    reference_cache = load_reference_emg_cache(reference_dir, seeds_tuple) if reference_dir is not None else {}
    missing_seeds = tuple(seed for seed in seeds_tuple if seed not in reference_cache)
    if missing_seeds:
        built_cache = build_reference_emg_cache(missing_seeds, config)
        reference_cache.update(built_cache)
        if reference_dir is not None:
            _persist_reference_emg_cache(reference_dir, built_cache)
    return {seed: np.asarray(reference_cache[seed], dtype=float) for seed in seeds_tuple}


def write_best_emg_panel(
    *,
    method_key: str,
    theta: PatternParameters,
    output_dir: str | Path,
    config: SimulationConfig,
    reference_dir: str | Path,
) -> None:
    """Evaluate one best candidate on train seeds and write its EMG comparison panel."""

    output_path = ensure_dir(output_dir)
    train_seeds = tuple(int(seed) for seed in config.seed_config.train_seeds)
    reference_cache = resolve_reference_emg_cache(train_seeds, config, reference_dir=reference_dir)
    stim_pattern = generate_stim_pattern(
        theta,
        t_end_ms=config.simulation_duration_ms,
        dt_ms=config.dt_ms,
        device_config=config.device_config,
    )
    lesion_condition = PatientConditionSpec(
        label=f"lesion_{method_key}",
        perc_supra_intact=config.lesion_perc_supra_intact,
    )
    lesion_results = run_condition(lesion_condition, stim_pattern, train_seeds, config)
    lesion_by_seed = {int(result.trial_seed): result.emg_signal for result in lesion_results}
    train_corrs = [
        compute_emg_similarity(
            reference_emg=reference_cache[int(seed)],
            candidate_emg=lesion_by_seed[int(seed)],
            envelope_window_ms=config.metric_config.envelope_window_ms,
            max_lag_ms=config.metric_config.max_lag_ms,
            use_envelope=config.metric_config.use_envelope,
        )
        for seed in train_seeds
    ]
    mean_corr, _ = mean_and_std_over_seeds(train_corrs)
    comparison_label = lesion_label(method_key)
    comparison_label = comparison_label[0].lower() + comparison_label[1:]
    plot_emg_seed_panels(
        {int(seed): reference_cache[int(seed)] for seed in train_seeds},
        lesion_by_seed,
        output_path / "best_emg.png",
        f"Healthy pre-lesion vs {comparison_label} | corr={mean_corr:.3f}",
        reference_label="Healthy pre-lesion",
        candidate_label=lesion_label(method_key),
    )


def evaluate_best_candidate_report_summary(
    *,
    theta: PatternParameters,
    output_dir: str | Path,
    config: SimulationConfig,
    reference_dir: str | Path,
) -> EvaluationSummary:
    """Reevaluate one candidate on report seeds and write the final report summary."""

    output_path = ensure_dir(output_dir)
    report_seeds = tuple(int(seed) for seed in config.seed_config.report_seeds)
    reference_cache = resolve_reference_emg_cache(report_seeds, config, reference_dir=reference_dir)
    summary = evaluate_pattern(
        theta=theta,
        seeds=report_seeds,
        config=config,
        reference_emg_by_seed=reference_cache,
    )
    write_json(output_path / "final_report_summary.json", summary)
    return summary
