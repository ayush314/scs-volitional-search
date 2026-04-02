#!/usr/bin/env python3
"""Run the CMA-ES optimizer."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from scs_search.analysis import best_so_far_trace, build_best_under_limit_frontier, reference_baseline_stats
from scs_search.config import DEFAULT_SWEEP_SEED_TRIALS, OptimizerConfig, SimulationConfig, dataclass_config_bundle
from scs_search.optimizers import optimizer_summary_payload
from scs_search.optimizers.cmaes_runner import run_optimizer
from scs_search.plotting import lesion_label, plot_best_so_far, plot_frontier, plot_frontier_overlay
from scs_search.simulator_adapter import write_best_emg_panel
from scs_search.utils import ensure_dir, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CMA-ES under a fixed seed-level trial budget.")
    parser.add_argument("--output-dir", default="results/cmaes")
    parser.add_argument("--seed-trial-budget", type=int, default=DEFAULT_SWEEP_SEED_TRIALS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    simulation = SimulationConfig(backend="neuron")
    optimizer = replace(OptimizerConfig(algorithm="cmaes"), seed_trial_budget=args.seed_trial_budget)
    output_dir = ensure_dir(args.output_dir)
    result = run_optimizer({"simulation": simulation, "optimizer": optimizer}, str(output_dir))
    history = result.history
    trace = best_so_far_trace(history)
    frontier = build_best_under_limit_frontier(history)
    baseline = reference_baseline_stats(Path(output_dir).parent / "reference", simulation.metric_config)

    write_json(
        output_dir / "summary.json",
        optimizer_summary_payload(result=result, config_bundle=dataclass_config_bundle(simulation, optimizer)),
    )
    write_jsonl(output_dir / "history.jsonl", history)
    write_best_emg_panel(
        method_key="cmaes",
        theta=result.incumbent_theta,
        output_dir=output_dir,
        config=simulation,
        reference_dir=Path(output_dir).parent / "reference",
    )
    plot_best_so_far(
        {lesion_label("cmaes"): trace},
        output_dir / "search_progress.png",
        baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
        title="CMA-ES search progress across seed-level trials",
    )
    plot_frontier(
        history,
        frontier,
        output_dir / "frontier.png",
        baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
        title="CMA-ES frontier",
    )

    grid_dir = Path(output_dir).parent / "grid_sweep"
    grid_patterns_file = grid_dir / "patterns.jsonl"
    if grid_patterns_file.exists():
        grid_records = read_jsonl(grid_patterns_file)
        plot_frontier_overlay(
            grid_records,
            build_best_under_limit_frontier(grid_records),
            history,
            frontier,
            output_dir / "frontier_with_grid.png",
            base_name="Grid sweep",
            overlay_name="CMA-ES",
            baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
        )


if __name__ == "__main__":
    main()
