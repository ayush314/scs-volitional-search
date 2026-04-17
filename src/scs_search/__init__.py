"""Core package for the task-burst physical-modulation SCS study."""

from .config import (
    DeviceConfig,
    DoseConfig,
    EvaluationSummary,
    MetricConfig,
    OptimizerConfig,
    OptimizerRunResult,
    PhysicalModulationBounds,
    PhysicalModulationParameters,
    SeedConfig,
    SimulationConfig,
    SimulationResult,
    StimPattern,
    THETA_NAMES,
    TransductionConfig,
)

__all__ = [
    "DeviceConfig",
    "DoseConfig",
    "EvaluationSummary",
    "MetricConfig",
    "OptimizerConfig",
    "OptimizerRunResult",
    "PhysicalModulationBounds",
    "PhysicalModulationParameters",
    "SeedConfig",
    "SimulationConfig",
    "SimulationResult",
    "StimPattern",
    "THETA_NAMES",
    "TransductionConfig",
]
