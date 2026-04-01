"""Tests for sweep frontier analysis helpers."""

from __future__ import annotations

from scs_search.analysis import build_best_under_limit_frontier


def test_frontier_tracks_best_observed_under_limit() -> None:
    records = [
        {"device_cost": 0.1, "mean_corr": 0.1, "label": "a"},
        {"device_cost": 0.2, "mean_corr": 0.4, "label": "b"},
        {"device_cost": 0.3, "mean_corr": 0.2, "label": "c"},
        {"device_cost": 0.4, "mean_corr": 0.5, "label": "d"},
    ]

    hull = build_best_under_limit_frontier(records)

    assert [record["label"] for record in hull] == ["a", "b", "b", "d"]
    assert [record["device_cost"] for record in hull] == [0.1, 0.2, 0.3, 0.4]
    assert [record["mean_corr"] for record in hull] == [0.1, 0.4, 0.4, 0.5]


def test_frontier_keeps_best_point_per_limit() -> None:
    records = [
        {"device_cost": 0.2, "mean_corr": 0.3, "label": "low"},
        {"device_cost": 0.2, "mean_corr": 0.5, "label": "high"},
        {"device_cost": 0.4, "mean_corr": 0.6, "label": "end"},
    ]

    hull = build_best_under_limit_frontier(records)

    assert [record["label"] for record in hull] == ["high", "end"]
