"""Tests for sweep frontier analysis helpers."""

from __future__ import annotations

from scs_search.analysis import build_upper_hull_frontier


def test_upper_hull_removes_interior_points() -> None:
    records = [
        {"norm_dose": 0.1, "mean_corr": 0.1, "label": "a"},
        {"norm_dose": 0.2, "mean_corr": 0.4, "label": "b"},
        {"norm_dose": 0.3, "mean_corr": 0.2, "label": "c"},
        {"norm_dose": 0.4, "mean_corr": 0.5, "label": "d"},
    ]

    hull = build_upper_hull_frontier(records)

    assert [record["label"] for record in hull] == ["a", "b", "d"]


def test_upper_hull_keeps_best_point_per_dose() -> None:
    records = [
        {"norm_dose": 0.2, "mean_corr": 0.3, "label": "low"},
        {"norm_dose": 0.2, "mean_corr": 0.5, "label": "high"},
        {"norm_dose": 0.4, "mean_corr": 0.6, "label": "end"},
    ]

    hull = build_upper_hull_frontier(records)

    assert [record["label"] for record in hull] == ["high", "end"]
