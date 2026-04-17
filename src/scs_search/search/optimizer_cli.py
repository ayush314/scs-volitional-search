"""Shared CLI implementation for the optimizer entrypoints."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..config import DEFAULT_SWEEP_SEED_TRIALS, OptimizerConfig, dataclass_config_bundle
from ..reporting.analysis import (
    best_so_far_trace,
    build_best_under_limit_frontier,
    comparable_optimizer_history,
    reference_baseline_stats,
)
from ..reporting.plotting import display_name, lesion_label, plot_best_so_far, plot_frontier, plot_frontier_overlay
from ..simulation.evaluator import write_best_emg_panel
from ..utils import ensure_dir, read_jsonl, write_json, write_jsonl
from .optimizer_history import history_seed_trials, load_optimizer_history, optimizer_summary_payload
from .optimizers.bohb import run_optimizer as run_bohb
from .optimizers.cmaes import run_optimizer as run_cmaes
from .optimizers.turbo import run_optimizer as run_turbo
from .sweep import make_physical_modulation_simulation_config

RUNNERS = {
    "cmaes": run_cmaes,
    "turbo": run_turbo,
    "bohb": run_bohb,
}


def parse_args(algorithm: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Run {display_name(algorithm)} on the physical modulation search space.")
    parser.add_argument("--output-dir", default=f"results/{algorithm}")
    parser.add_argument("--seed-trial-budget", type=int, default=DEFAULT_SWEEP_SEED_TRIALS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--additional-seed-trial-budget", type=int, default=None)
    parser.add_argument(
        "--supraspinal-drive-mode",
        choices=("aperiodic_envelope", "sinusoidal"),
        default="aperiodic_envelope",
    )
    return parser.parse_args()


def main(algorithm: str) -> None:
    args = parse_args(algorithm)
    simulation = make_physical_modulation_simulation_config(
        backend="neuron",
        supraspinal_drive_mode=args.supraspinal_drive_mode,
    )
    output_dir = ensure_dir(args.output_dir)
    existing_history = load_optimizer_history(output_dir) if args.resume else []
    existing_seed_trials = history_seed_trials(existing_history)
    target_seed_trial_budget = int(args.seed_trial_budget)
    if args.additional_seed_trial_budget is not None:
        target_seed_trial_budget = existing_seed_trials + int(args.additional_seed_trial_budget)
    optimizer = replace(OptimizerConfig(algorithm=algorithm), seed_trial_budget=target_seed_trial_budget)
    result = RUNNERS[algorithm]({"simulation": simulation, "optimizer": optimizer}, str(output_dir), resume=args.resume)
    history = result.history
    comparable_history = comparable_optimizer_history(
        history,
        algorithm=algorithm,
        required_seed_budget=len(simulation.seed_config.train_seeds),
    )
    trace = best_so_far_trace(comparable_history)
    frontier = build_best_under_limit_frontier(comparable_history)
    baseline = reference_baseline_stats(
        Path(output_dir).parent / "reference",
        simulation.metric_config,
        seeds=simulation.seed_config.train_seeds,
    )

    write_json(
        output_dir / "summary.json",
        optimizer_summary_payload(
            result=result,
            config_bundle=dataclass_config_bundle(simulation, optimizer),
            history_for_best_pattern=comparable_history,
        ),
    )
    write_jsonl(output_dir / "history.jsonl", history)
    write_best_emg_panel(
        method_key=algorithm,
        theta=result.incumbent_theta,
        output_dir=output_dir,
        config=simulation,
        reference_dir=Path(output_dir).parent / "reference",
    )
    plot_best_so_far(
        {lesion_label(algorithm): trace},
        output_dir / "search_progress.png",
        baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
        title=f"{display_name(algorithm)} search progress across seed-level trials",
    )
    plot_frontier(
        comparable_history,
        frontier,
        output_dir / "frontier.png",
        baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
        title=f"{display_name(algorithm)} frontier",
    )

    grid_dir = Path(output_dir).parent / "grid_sweep"
    grid_patterns_file = grid_dir / "patterns.jsonl"
    if grid_patterns_file.exists():
        grid_records = read_jsonl(grid_patterns_file)
        plot_frontier_overlay(
            grid_records,
            build_best_under_limit_frontier(grid_records),
            comparable_history,
            frontier,
            output_dir / "frontier_with_grid.png",
            base_name="Grid sweep",
            overlay_name=display_name(algorithm),
            baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
        )
