"""Configuration objects and shared dataclasses for the task-burst physical-modulation study."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

DEFAULT_PULSE_WIDTH_US: float = 210.0
THETA_NAMES: tuple[str, ...] = (
    "I0_ma",
    "I1_ma",
    "f0_hz",
    "f1_hz",
    "PW1_us",
    "T_ms",
)
DEFAULT_SWEEP_SEED_TRIALS: int = 1000

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent.parent
EXTERNAL_ROOT = REPO_ROOT / "external"


def default_train_seeds() -> tuple[int, ...]:
    """Default repeated-evaluation seeds for candidate scoring."""

    return (101, 202, 303)


def default_report_seeds() -> tuple[int, ...]:
    """Default final-reporting seeds for incumbent evaluation."""

    return tuple(range(1001, 1011))


@dataclass(frozen=True)
class PhysicalModulationParameters:
    """Shared-wave physical pulse-control parameterization."""

    I0_ma: float
    I1_ma: float
    f0_hz: float
    f1_hz: float
    PW1_us: float
    T_ms: float

    @classmethod
    def from_any(
        cls,
        theta: Sequence[float] | Mapping[str, float] | "PhysicalModulationParameters",
    ) -> "PhysicalModulationParameters":
        """Normalize one user-provided theta representation."""

        if isinstance(theta, cls):
            return theta
        if isinstance(theta, Mapping):
            values = dict(theta)
            aliases = {
                "I0_ma": ("I0_ma", "I0", "i0_ma", "i0"),
                "I1_ma": ("I1_ma", "I1", "i1_ma", "i1"),
                "f0_hz": ("f0_hz", "f0"),
                "f1_hz": ("f1_hz", "f1"),
                "PW1_us": ("PW1_us", "PW1", "pw1_us", "pw1"),
                "T_ms": ("T_ms", "T", "period_ms"),
            }
            normalized: dict[str, float] = {}
            for name, candidates in aliases.items():
                for candidate in candidates:
                    if candidate in values:
                        normalized[name] = float(values[candidate])
                        break
                else:
                    raise KeyError(f"Missing physical modulation parameter `{name}`.")
            return cls(**normalized)
        if len(theta) != len(THETA_NAMES):
            raise ValueError(f"Expected {len(THETA_NAMES)} parameters, received {len(theta)}.")
        return cls(**{name: float(value) for name, value in zip(THETA_NAMES, theta)})

    def to_array(self) -> np.ndarray:
        """Return the parameter vector in optimizer-friendly order."""

        return np.asarray([getattr(self, name) for name in THETA_NAMES], dtype=float)

    def to_dict(self) -> dict[str, float]:
        """Return a JSON-friendly mapping."""

        return {name: float(getattr(self, name)) for name in THETA_NAMES}


def theta_to_dict(theta: Any) -> dict[str, float]:
    """Return a JSON-friendly theta mapping."""

    if hasattr(theta, "to_dict"):
        values = theta.to_dict()
        return {str(key): float(value) for key, value in values.items()}
    if isinstance(theta, Mapping):
        return {str(key): float(value) for key, value in theta.items()}
    raise TypeError("Theta values must provide `to_dict()` or be a mapping.")


@dataclass(frozen=True)
class DeviceConfig:
    """Hardware-budget limits used for reporting and constraints."""

    max_total_current_ma: float = 20.0
    min_pulse_width_us: float = 60.0
    max_pulse_width_us: float = 600.0
    pulse_width_step_us: float = 10.0
    max_master_rate_hz: float = 400.0
    default_pulse_width_us: float = DEFAULT_PULSE_WIDTH_US


@dataclass(frozen=True)
class PhysicalModulationBounds:
    """Box constraints for the physical-modulation search space."""

    lower: tuple[float, ...] = (0.0, 0.0, 10.0, 0.0, 0.0, 50.0)
    upper: tuple[float, ...] = (20.0, 20.0, 400.0, 390.0, 150.0, 1000.0)
    names: tuple[str, ...] = THETA_NAMES
    baseline_pulse_width_us: float = DEFAULT_PULSE_WIDTH_US

    def __post_init__(self) -> None:
        if len(self.lower) != len(self.upper) or len(self.lower) != len(self.names):
            raise ValueError("Bounds must have the same dimensionality as the theta names.")
        if any(lo >= hi for lo, hi in zip(self.lower, self.upper)):
            raise ValueError("Each lower bound must be strictly less than its upper bound.")

    def _i1_cap(self, i0_ma: float, device_config: DeviceConfig) -> float:
        return float(
            min(
                max(i0_ma, 0.0),
                max(float(device_config.max_total_current_ma) - i0_ma, 0.0),
                self.upper[1],
            )
        )

    def _f1_cap(self, f0_hz: float, device_config: DeviceConfig) -> float:
        f_min = float(self.lower[2])
        f_max = float(device_config.max_master_rate_hz)
        return float(min(max(f0_hz - f_min, 0.0), max(f_max - f0_hz, 0.0), self.upper[3]))

    def _pw1_cap(self, device_config: DeviceConfig) -> float:
        pw0_us = float(self.baseline_pulse_width_us)
        return float(
            min(
                max(pw0_us - float(device_config.min_pulse_width_us), 0.0),
                max(float(device_config.max_pulse_width_us) - pw0_us, 0.0),
                self.upper[4],
            )
        )

    def _coerce_mapping(
        self,
        theta: Sequence[float] | Mapping[str, float] | PhysicalModulationParameters,
    ) -> dict[str, float]:
        if isinstance(theta, PhysicalModulationParameters):
            return theta.to_dict()
        if isinstance(theta, Mapping):
            return PhysicalModulationParameters.from_any(theta).to_dict()
        if len(theta) != len(self.names):
            raise ValueError(f"Expected {len(self.names)} parameters, received {len(theta)}.")
        return {name: float(value) for name, value in zip(self.names, theta)}

    def _clip_mapping(self, values: Mapping[str, float], device_config: DeviceConfig) -> dict[str, float]:
        clipped = {
            "I0_ma": float(
                np.clip(values["I0_ma"], self.lower[0], min(self.upper[0], float(device_config.max_total_current_ma)))
            ),
            "I1_ma": float(np.clip(values["I1_ma"], self.lower[1], self.upper[1])),
            "f0_hz": float(
                np.clip(values["f0_hz"], self.lower[2], min(self.upper[2], float(device_config.max_master_rate_hz)))
            ),
            "f1_hz": float(np.clip(values["f1_hz"], self.lower[3], self.upper[3])),
            "PW1_us": float(np.clip(values["PW1_us"], self.lower[4], self._pw1_cap(device_config))),
            "T_ms": float(np.clip(values["T_ms"], self.lower[5], self.upper[5])),
        }
        clipped["I1_ma"] = float(np.clip(clipped["I1_ma"], self.lower[1], self._i1_cap(clipped["I0_ma"], device_config)))
        clipped["f1_hz"] = float(np.clip(clipped["f1_hz"], self.lower[3], self._f1_cap(clipped["f0_hz"], device_config)))
        clipped["PW1_us"] = float(np.clip(clipped["PW1_us"], self.lower[4], self._pw1_cap(device_config)))
        return clipped

    def clip(
        self,
        theta: Sequence[float] | Mapping[str, float] | PhysicalModulationParameters,
        *,
        device_config: DeviceConfig | None = None,
    ) -> PhysicalModulationParameters:
        """Clip a theta value into the configured box."""

        config = device_config or DeviceConfig()
        values = self._coerce_mapping(theta)
        return PhysicalModulationParameters.from_any(self._clip_mapping(values, config))

    def decode_unit(
        self,
        unit_point: Sequence[float],
        *,
        device_config: DeviceConfig | None = None,
    ) -> PhysicalModulationParameters:
        """Map one point from [0, 1]^d into the bounded parameter space."""

        point = np.clip(np.asarray(unit_point, dtype=float), 0.0, 1.0)
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        values = {name: float(value) for name, value in zip(self.names, lower + point * (upper - lower))}
        return self.clip(values, device_config=device_config)

    def encode_unit(
        self,
        theta: Sequence[float] | Mapping[str, float] | PhysicalModulationParameters,
        *,
        device_config: DeviceConfig | None = None,
    ) -> np.ndarray:
        """Map one theta value into [0, 1]^d."""

        config = device_config or DeviceConfig()
        values = self._clip_mapping(self._coerce_mapping(theta), config)
        params = np.asarray([values[name] for name in self.names], dtype=float)
        lower = np.asarray(self.lower, dtype=float)
        upper = np.asarray(self.upper, dtype=float)
        return np.clip((params - lower) / (upper - lower), 0.0, 1.0)


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
    """Configuration for internal recruitment diagnostics and budget penalties."""

    max_frequency_hz: float = 400.0
    objective_penalty_weight: float = 10.0
    robust_std_weight: float = 0.25
    frequency_penalty_threshold_hz: float = 100.0
    frequency_penalty_weight: float = 0.0
    high_recruitment_threshold: float = 0.8
    high_recruitment_weight: float = 0.0
    invalid_objective_floor: float = -1.0


@dataclass(frozen=True)
class TransductionConfig:
    """Internal settings for mapping delivered pulses to afferent spikes."""

    enforce_no_overlap: bool = True
    mode: str = "strength_duration"
    chronaxie_us: float = 360.0
    absolute_refractory_ms: float = 0.7
    relative_refractory_end_ms: float = 3.0


@dataclass(frozen=True)
class SimulationConfig:
    """Shared simulation constants across scripts, sweeps, and optimizers."""

    backend: str = "neuron"
    dt_ms: float = 1.0
    pulse_scheduler_dt_ms: float = 0.1
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
    supraspinal_drive_mode: str = "aperiodic_envelope"
    supraspinal_rate_floor_hz: float = 0.0
    supraspinal_envelope_control_dt_ms: float = 100.0
    supraspinal_envelope_smoothing_sigma_ms: float = 25.0
    supraspinal_envelope_ar_rho: float = 0.85
    supraspinal_task_burst_min_ms: float = 120.0
    supraspinal_task_burst_max_ms: float = 260.0
    supraspinal_task_gap_min_ms: float = 80.0
    supraspinal_task_gap_max_ms: float = 220.0
    supraspinal_inhomogeneous_rate_hz: float = 0.002
    healthy_perc_supra_intact: float = 1.0
    lesion_perc_supra_intact: float = 0.2
    baseline_cycle_ms: float = 500.0
    structural_seed: int = 672945
    external_root: str = str(EXTERNAL_ROOT)
    theta_bounds: PhysicalModulationBounds = field(default_factory=PhysicalModulationBounds)
    seed_config: SeedConfig = field(default_factory=SeedConfig)
    metric_config: MetricConfig = field(default_factory=MetricConfig)
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    dose_config: DoseConfig = field(default_factory=DoseConfig)
    transduction_config: TransductionConfig = field(default_factory=TransductionConfig)


@dataclass(frozen=True)
class PatientConditionSpec:
    """Description of a healthy or lesioned simulation condition."""

    label: str
    perc_supra_intact: float


@dataclass
class StimPattern:
    """Time-indexed stimulation structure consumed by the simulator adapter."""

    family: str
    theta: Any
    time_ms: np.ndarray
    alpha_t: np.ndarray
    pulse_times_ms: np.ndarray
    pulse_alpha: np.ndarray
    pulse_current_ma: np.ndarray
    pulse_widths_us: np.ndarray
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
    """Aggregate statistics for one stimulation pattern across repeated seeds."""

    theta: Any
    family: str
    seeds: tuple[int, ...]
    per_seed_records: list[dict[str, Any]]
    mean_corr: float
    std_corr: float
    mean_raw_dose: float
    std_raw_dose: float
    mean_norm_dose: float
    std_norm_dose: float
    mean_device_cost: float
    std_device_cost: float
    mean_current_rate_usage: float
    std_current_rate_usage: float
    mean_total_current_ma: float
    std_total_current_ma: float
    mean_charge_per_pulse_uc: float
    std_charge_per_pulse_uc: float
    mean_charge_rate_uc_per_s: float
    std_charge_rate_uc_per_s: float
    mean_relative_envelope_rmse: float
    std_relative_envelope_rmse: float
    penalized_objective: float
    robust_objective: float
    valid: bool = True
    invalid_reason: str | None = None
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
    incumbent_theta: Any
    incumbent_summary: EvaluationSummary
    history: list[dict[str, Any]]
    metadata: dict[str, Any] = field(default_factory=dict)


def condition_defaults(config: SimulationConfig) -> tuple[PatientConditionSpec, PatientConditionSpec]:
    """Return the default healthy and lesioned conditions."""

    return (
        PatientConditionSpec(label="healthy_prelesion", perc_supra_intact=config.healthy_perc_supra_intact),
        PatientConditionSpec(label="lesion", perc_supra_intact=config.lesion_perc_supra_intact),
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
        "device_config": simulation_config.device_config,
        "dose_config": simulation_config.dose_config,
        "transduction_config": simulation_config.transduction_config,
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
