"""Tests for normalized stimulation dose."""

from __future__ import annotations

import numpy as np

from scs_search.config import DoseConfig
from scs_search.dose import combined_objective, compute_pattern_dose, raw_dose_from_pulse_alpha
from scs_search.patterns import generate_tonic_pattern


def test_raw_dose_is_sum_of_pulse_alpha() -> None:
    pulse_alpha = np.array([0.2, 0.4, 0.6])
    assert np.isclose(raw_dose_from_pulse_alpha(pulse_alpha), 1.2)


def test_normalized_dose_matches_full_recruitment_reference() -> None:
    pattern = generate_tonic_pattern(freq_hz=120.0, alpha=1.0, t_end_ms=1000)
    raw_dose, norm_dose = compute_pattern_dose(pattern, DoseConfig())
    assert np.isclose(raw_dose, 120.0)
    assert np.isclose(norm_dose, 1.0)


def test_combined_objective_penalizes_budget_violation() -> None:
    robust_score, penalized = combined_objective(
        mean_corr=0.8,
        std_corr=0.1,
        norm_dose=1.2,
        budget_norm=1.0,
        dose_config=DoseConfig(objective_penalty_weight=10.0),
        robust=False,
    )
    assert np.isclose(robust_score, 0.8)
    assert penalized < robust_score
