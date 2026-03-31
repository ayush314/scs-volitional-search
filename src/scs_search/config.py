"""Configuration objects and shared dataclasses for SCS search experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

THETA_NAMES: tuple[str, ...] = ("f", "T_on", "T_off", "alpha0", "alpha1", "phi1", "alpha2", "phi2")
DEFAULT_BUDGET_LEVELS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
DEFAULT_SWEEP_EVALUATIONS: int = 472
DEFAULT_SWEEP_SEED_TRIALS: int = DEFAULT_SWEEP_EVALUATIONS * 3
UPSTREAM_REPOS: dict[str, str] = {
    "SCSInSCIMechanisms": "ea349460de2a245ec5d3a929a00006b9ac821825",
    "GeneticAlgorithmSCSMotorControl": "67267ae076baa826812051ce81c8c20fe327808e",
}

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
EXTERNAL_ROOT = REPO_ROOT / "external"


def default_train_seeds() -> tuple[int, ...]:
    """Default repeated-evaluation seeds for candidate scoring."""

    return (101, 202, 303)


def default_report_seeds() -> tuple[int, ...]:
    """Default final-reporting seeds for incumbent evaluation."""

    return (1001, 1002, 1003)


@dataclass(frozen=True)
class PatternParameters:
    """Main stimulation parameterization."""

    f: float
    T_on: float
    T_off: float
    alpha0: float
    alpha1: float
    phi1: float
    alpha2: float
    phi2: float

    @classmethod
    def from_any(cls, theta: Sequence[float] | Mapping[str, float] | "PatternParameters") -> "PatternParameters":
        """Normalize a user-provided theta representation to a dataclass."""

        if isinstance(theta, cls):
            return theta
        if isinstance(theta, Mapping):
            values = dict(theta)
            if "a1" in values and "alpha1" not in values:
                values["alpha1"] = values["a1"]
            if "a2" in values and "alpha2" not in values:
                values["alpha2"] = values["a2"]
            return cls(**{name: float(values[name]) for name in THETA_NAMES})
        if len(theta) != len(THETA_NAMES):
            raise ValueError(f"Expected {len(THETA_NAMES)} parameters, received {len(theta)}.")
        return cls(**{name: float(value) for name, value in zip(THETA_NAMES, theta)})

    def to_array(self) -> np.ndarray:
        """Return the parameter vector in optimizer-friendly order."""

        return np.asarray([getattr(self, name) for name in THETA_NAMES], dtype=float)

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-friendly mapping."""

        return {name: float(getattr(self, name)) for name in THETA_NAMES}


@dataclass(frozen=True)
class ParameterBounds:
    """Box constraints for the stimulation search space."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    names: tuple[str, ...] = THETA_NAMES

    def __post_init__(self) -> None:
        if len(self.lower) != len(self.upper) or len(self.lower) != len(self.names):
            raise ValueError("Bounds must have the same dimensionality as the theta names.")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("Each lower bound must be strictly less than its upper bound.")

    def clip(self, theta: Sequence[float] | Mapping[str, float] | PatternParameters) -> PatternParameters:
        """Clip a theta vector into the configured box."""

        params = PatternParameters.from_any(theta).to_array()
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        clipped = np.clip(params, lower, upper)
        return PatternParameters.from_any(clipped)

    def decode_unit(self, unit_point: Sequence[float]) -> PatternParameters:
        """Map a point from [0, 1]^d into the bounded parameter space."""

        point = np.clip(np.asarray(unit_point, dtype=float), 0.0, 1.0)
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        return PatternParameters.from_any(lower + point * (upper - lower))

    def encode_unit(self, theta: Sequence[float] | Mapping[str, float] | PatternParameters) -> np.ndarray:
        """Map a theta vector into [0, 1]^d."""

        params = PatternParameters.from_any(theta).to_array()
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        return np.clip((params - lower) / (upper - lower), 0.0, 1.0)


def default_theta_bounds() -> ParameterBounds:
    """Return the default search box from the study plan."""

    return ParameterBounds(
        lower=(10.0, 50.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0),
        upper=(120.0, 500.0, 500.0, 0.9, 0.5, 2.0 * np.pi, 0.5, 2.0 * np.pi),
    )


@dataclass(frozen=True)
class SeedConfig:
    """Seed sets for structural randomness and repeated candidate evaluation."""

    structural_seed: int = 672945
    train_seeds: tuple[int, ...] = field(default_factory=default_train_seeds)
    report_seeds: tuple[int, ...] = field(default_factory=default_report_seeds)

    def seeds_for_budget(self, budget: int | float) -> tuple[int, ...]:
        """Return a deterministic prefix of the training seeds."""

        seed_count = max(1, min(int(round(float(budget))), len(self.train_seeds)))
        return self.train_seeds[:seed_count]


@dataclass(frozen=True)
class MetricConfig:
    """Configuration for EMG restoration metrics."""

    use_envelope: bool = True
    envelope_window_ms: int = 25
    max_lag_ms: int = 0


@dataclass(frozen=True)
class DoseConfig:
    """Configuration for normalized dose and penalties."""

    max_frequency_hz: float = 120.0
    budget_levels: tuple[float, ...] = DEFAULT_BUDGET_LEVELS
    objective_penalty_weight: float = 10.0
    robust_std_weight: float = 0.25
    frequency_penalty_threshold_hz: float = 100.0
    frequency_penalty_weight: float = 0.0
    high_recruitment_threshold: float = 0.8
    high_recruitment_weight: float = 0.0


@dataclass(frozen=True)
class SimulationConfig:
    """Shared simulation constants across scripts, sweeps, and optimizers."""

    backend: str = "neuron"
    dt_ms: float = 1.0
    simulation_duration_ms: int = 1000
    num_scs_total: int = 60
    num_supraspinal_total: int = 300
    num_mn: int = 100
    mn_drug: bool = True
    mn_avg_diameter: float = 36.0
    synaptic_weight_scs: float = 0.000148
    synaptic_weight_supra: float = 0.000148
    synapse_shape: float = 1.2
    synapse_tau_ms: float = 2.0
    supraspinal_rate_hz: float = 60.0
    supraspinal_inhomogeneous_rate_hz: float = 0.001
    healthy_perc_supra_intact: float = 1.0
    lesion_perc_supra_intact: float = 0.2
    baseline_cycle_ms: float = 500.0
    structural_seed: int = 672945
    external_root: str = str(EXTERNAL_ROOT)
    theta_bounds: ParameterBounds = field(default_factory=default_theta_bounds)
    seed_config: SeedConfig = field(default_factory=SeedConfig)
    metric_config: MetricConfig = field(default_factory=MetricConfig)
    dose_config: DoseConfig = field(default_factory=DoseConfig)


@dataclass(frozen=True)
class ConditionSpec:
    """Description of a healthy or lesioned simulation condition."""

    label: str
    perc_supra_intact: float


@dataclass
class StimPattern:
    """Time-indexed stimulation structure consumed by the simulator adapter."""

    family: str
    theta: PatternParameters
    time_ms: np.ndarray
    alpha_t: np.ndarray
    pulse_times_ms: np.ndarray
    pulse_alpha: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationResult:
    """Raw output from a single simulator run."""

    condition_label: str
    trial_seed: int
    structural_seed: int
    backend: str
    perc_supra_intact: float
    stim_pattern: StimPattern
    emg_signal: np.ndarray
    mn_spike_times: list[np.ndarray] | None = None
    supraspinal_spike_times: list[np.ndarray] | None = None
    scs_pulse_times: list[np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationSummary:
    """Aggregate statistics for a stimulation pattern across repeated seeds."""

    theta: PatternParameters
    family: str
    seeds: tuple[int, ...]
    per_seed_records: list[dict[str, Any]]
    mean_corr: float
    std_corr: float
    mean_raw_dose: float
    std_raw_dose: float
    mean_norm_dose: float
    std_norm_dose: float
    penalized_objective: float
    robust_objective: float
    feasible_by_budget: dict[str, bool]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerConfig:
    """Shared settings for BOHB, TuRBO, and CMA-ES runs."""

    algorithm: str
    seed_trial_budget: int = DEFAULT_SWEEP_SEED_TRIALS
    budget_norm: float = 1.0
    robust_objective: bool = False
    turbo_initial_points: int = 4
    cmaes_population_size: int = 8
    turbo_initial_length: float = 0.8
    turbo_min_length: float = 0.05
    turbo_max_length: float = 1.6
    turbo_success_tolerance: int = 3
    turbo_failure_tolerance: int = 3


@dataclass
class OptimizerRunResult:
    """Optimizer run summary saved by the CLI runners."""

    algorithm: str
    output_dir: str
    incumbent_theta: PatternParameters
    incumbent_summary: EvaluationSummary
    history: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


def condition_defaults(config: SimulationConfig) -> tuple[ConditionSpec, ConditionSpec]:
    """Return the default healthy and lesioned conditions."""

    return (
        ConditionSpec(label="healthy_prelesion", perc_supra_intact=config.healthy_perc_supra_intact),
        ConditionSpec(label="lesion", perc_supra_intact=config.lesion_perc_supra_intact),
    )


def dataclass_config_bundle(
    simulation_config: SimulationConfig,
    optimizer_config: OptimizerConfig | None = None,
) -> dict[str, Any]:
    """Return a serializable configuration bundle."""

    bundle: dict[str, Any] = {
        "simulation": simulation_config,
        "seed_config": simulation_config.seed_config,
        "metric_config": simulation_config.metric_config,
        "dose_config": simulation_config.dose_config,
        "theta_bounds": simulation_config.theta_bounds,
    }
    if optimizer_config is not None:
        bundle["optimizer"] = optimizer_config
    return bundle


def coerce_seed_sequence(seeds: Iterable[int] | None, default: Sequence[int]) -> tuple[int, ...]:
    """Normalize seed iterables coming from scripts and optimizer budgets."""

    if seeds is None:
        return tuple(int(seed) for seed in default)
    return tuple(int(seed) for seed in seeds)
