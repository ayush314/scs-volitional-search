#!/usr/bin/env python3
"""Run the TuRBO optimizer."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from scs_search.analysis import best_so_far_trace, build_upper_hull_frontier, reference_baseline_stats
from scs_search.config import DEFAULT_SWEEP_SEED_TRIALS, OptimizerConfig, SimulationConfig, dataclass_config_bundle
from scs_search.optimizers.turbo_runner import run_optimizer
from scs_search.plotting import plot_best_so_far, plot_frontier, plot_frontier_overlay
from scs_search.utils import ensure_dir, read_json, read_jsonl, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TuRBO under a fixed seed-level trial budget.")
    parser.add_argument("--output-dir", default="results/turbo")
    parser.add_argument("--seed-trial-budget", type=int, default=DEFAULT_SWEEP_SEED_TRIALS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulation = SimulationConfig(backend="neuron")
    optimizer = replace(OptimizerConfig(algorithm="turbo"), seed_trial_budget=args.seed_trial_budget)
    output_dir = ensure_dir(args.output_dir)
    result = run_optimizer({"simulation": simulation, "optimizer": optimizer}, str(output_dir))
    history = result.history
    trace = best_so_far_trace(history, budget_norm=optimizer.budget_norm)
    frontier = build_upper_hull_frontier(history)
    baseline = reference_baseline_stats(Path(output_dir).parent / "reference", simulation.metric_config)

    write_json(output_dir / "config.json", dataclass_config_bundle(simulation, optimizer))
    write_json(output_dir / "summary.json", result)
    write_json(output_dir / "history.json", history)
    write_json(output_dir / "frontier.json", frontier)
    write_jsonl(output_dir / "metrics.jsonl", history)
    write_csv(output_dir / "metrics.csv", history)
    write_json(output_dir / "trace.json", trace)
    plot_best_so_far(
        {"TuRBO": trace},
        output_dir / "best_so_far.png",
        baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
    )
    plot_frontier(
        history,
        frontier,
        output_dir / "device_budget_vs_corr.png",
        baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
    )

    grid_dir = Path(output_dir).parent / "grid_sweep"
    grid_frontier_file = grid_dir / "frontier.json"
    grid_metrics_file = grid_dir / "metrics.jsonl"
    if grid_frontier_file.exists() and grid_metrics_file.exists():
        plot_frontier_overlay(
            read_jsonl(grid_metrics_file),
            read_json(grid_frontier_file),
            history,
            frontier,
            output_dir / "device_budget_vs_corr_with_grid.png",
            base_name="Grid sweep",
            overlay_name="TuRBO",
            baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
        )


if __name__ == "__main__":
    main()
