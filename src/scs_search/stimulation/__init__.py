"""Pulse-train construction helpers."""

from .patterns import (
    generate_stim_pattern,
    generate_tonic_pattern,
    invalid_theta_reason,
    modulation_controls,
)

__all__ = [
    "generate_stim_pattern",
    "generate_tonic_pattern",
    "invalid_theta_reason",
    "modulation_controls",
]
