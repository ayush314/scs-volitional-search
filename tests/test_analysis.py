"""Tests for frontier analysis and summary regeneration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import scs_search.reporting.summarize as summarize_results
import scs_search.reporting.analysis as internal_analysis
from scs_search.reporting import analysis
from scs_search.reporting.analysis import build_best_under_limit_frontier, comparable_optimizer_history, filter_history_by_seed_budget
from scs_search.config import MetricConfig, theta_to_dict


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


def test_filter_history_by_seed_budget_keeps_requested_budget_and_non_bohb_rows() -> None:
    records = [
        {"label": "bohb-low", "seed_budget": 1, "mean_corr": 0.2, "device_cost": 0.1},
        {"label": "bohb-full", "seed_budget": 3, "mean_corr": 0.5, "device_cost": 0.2},
        {"label": "grid", "mean_corr": 0.4, "device_cost": 0.15},
    ]

    filtered = filter_history_by_seed_budget(records, required_seed_budget=3)

    assert [record["label"] for record in filtered] == ["bohb-full", "grid"]


def test_comparable_optimizer_history_uses_full_fidelity_bohb_rows_when_available() -> None:
    records = [
        {"label": "bohb-low", "seed_budget": 1, "mean_corr": 0.2, "device_cost": 0.1},
        {"label": "bohb-mid", "seed_budget": 2, "mean_corr": 0.3, "device_cost": 0.12},
        {"label": "bohb-full", "seed_budget": 3, "mean_corr": 0.5, "device_cost": 0.2},
    ]

    filtered = comparable_optimizer_history(records, algorithm="bohb", required_seed_budget=3)

    assert [record["label"] for record in filtered] == ["bohb-full"]


def test_reevaluate_best_patterns_refreshes_when_theta_changes(tmp_path, monkeypatch) -> None:
    results_root = tmp_path
    grid_dir = results_root / "grid_sweep"
    ref_dir = results_root / "reference"
    grid_dir.mkdir(parents=True)
    ref_dir.mkdir(parents=True)
    (ref_dir / "summary.json").write_text(json.dumps({}), encoding="utf-8")
    theta = {"I0_ma": 5.0, "I1_ma": 0.0, "f0_hz": 100.0, "f1_hz": 0.0, "PW1_us": 0.0, "T_ms": 500.0}
    (grid_dir / "summary.json").write_text(
        json.dumps({"best_pattern": {"theta": theta, "record": {"mean_corr": 0.7}}}),
        encoding="utf-8",
    )
    (grid_dir / "final_report_summary.json").write_text(
        json.dumps({"theta": {**theta, "I0_ma": 4.0}, "mean_relative_envelope_rmse": 0.5}),
        encoding="utf-8",
    )

    refreshed = {"called": False}

    def fake_eval(*, theta, output_dir, config, reference_dir):
        refreshed["called"] = True
        Path(output_dir, "final_report_summary.json").write_text(
            json.dumps({"theta": theta_to_dict(theta), "mean_relative_envelope_rmse": 0.4}),
            encoding="utf-8",
        )

    monkeypatch.setattr(summarize_results, "evaluate_best_candidate_report_summary", fake_eval)
    monkeypatch.setattr(summarize_results, "write_best_emg_panel", lambda **kwargs: None)

    config = summarize_results._config_from_results(results_root)
    summarize_results.reevaluate_best_patterns(results_root, config)

    assert refreshed["called"] is True


def test_reference_baseline_stats_returns_requested_split_from_summary(tmp_path) -> None:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    (reference_dir / "summary.json").write_text(
        json.dumps(
            {
                "train_seeds": [101, 202, 303],
                "report_seeds": [1001, 1002],
                "seeds": [101, 202, 303, 1001, 1002],
                "lesion_no_stim_baseline": {"mean_corr": 0.4, "std_corr": 0.1, "seeds": [101, 202, 303, 1001, 1002]},
                "lesion_no_stim_baseline_all": {"mean_corr": 0.4, "std_corr": 0.1, "seeds": [101, 202, 303, 1001, 1002]},
                "lesion_no_stim_baseline_train": {"mean_corr": 0.5, "std_corr": 0.05, "seeds": [101, 202, 303]},
                "lesion_no_stim_baseline_report": {"mean_corr": 0.6, "std_corr": 0.02, "seeds": [1001, 1002]},
            }
        ),
        encoding="utf-8",
    )

    metric_config = MetricConfig()

    assert analysis.reference_baseline_stats(reference_dir, metric_config)["mean_corr"] == 0.4
    assert analysis.reference_baseline_stats(reference_dir, metric_config, seeds=[101, 202, 303])["mean_corr"] == 0.5
    assert analysis.reference_baseline_stats(reference_dir, metric_config, seeds=[1001, 1002])["mean_corr"] == 0.6


def test_reference_baseline_stats_recomputes_for_requested_seed_subset(tmp_path, monkeypatch) -> None:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    np.savez(
        reference_dir / "emg_arrays.npz",
        healthy_prelesion_seed_1=np.asarray([0.0]),
        lesion_no_stim_seed_1=np.asarray([0.1]),
        healthy_prelesion_seed_2=np.asarray([0.0]),
        lesion_no_stim_seed_2=np.asarray([0.5]),
        healthy_prelesion_seed_3=np.asarray([0.0]),
        lesion_no_stim_seed_3=np.asarray([0.9]),
    )

    monkeypatch.setattr(
        internal_analysis,
        "compute_emg_similarity",
        lambda reference_emg, candidate_emg, **kwargs: float(candidate_emg[0]),
    )

    stats = analysis.reference_baseline_stats(reference_dir, MetricConfig(), seeds=[1, 3])

    assert stats is not None
    assert stats["seeds"] == [1, 3]
    assert stats["mean_corr"] == 0.5
