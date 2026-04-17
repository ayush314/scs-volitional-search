"""Structural-state helpers shared across repeated trial seeds."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..config import SimulationConfig


@dataclass(frozen=True)
class _StructuralState:
    """Fixed structural randomness shared across repeated trial seeds."""

    mn_lengths: np.ndarray
    W_supraspinal: np.ndarray
    W_scs: np.ndarray
    scs_delay: np.ndarray
    recruitment_order: np.ndarray


def make_structural_state(config: SimulationConfig) -> _StructuralState:
    """Generate the structural randomness once from the structural seed."""

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


@lru_cache(maxsize=8)
def get_structural_state(config: SimulationConfig) -> _StructuralState:
    """Cache structural state by simulation configuration."""

    return make_structural_state(config)
