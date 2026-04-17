#!/usr/bin/env python3
"""Run the structured sweep for the physical-modulation study."""

from __future__ import annotations

import argparse
from pathlib import Path

from scs_search.reporting.analysis import best_record, reference_baseline_stats
from scs_search.config import DEFAULT_SWEEP_SEED_TRIALS, dataclass_config_bundle
from scs_search.reporting.plotting import plot_frontier
from scs_search.search.sweep import make_physical_modulation_simulation_config, run_physical_modulation_sweep_suite
from scs_search.simulation.evaluator import resolve_reference_emg_cache, write_best_emg_panel
from scs_search.utils import ensure_dir, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the default sweep over the physical-modulation pattern family.")
    parser.add_argument("--output-dir", default="results/grid_sweep")
    parser.add_argument("--seed-trial-budget", type=int, default=DEFAULT_SWEEP_SEED_TRIALS)
    parser.add_argument(
        "--supraspinal-drive-mode",
        choices=("aperiodic_envelope", "sinusoidal"),
        default="aperiodic_envelope",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = make_physical_modulation_simulation_config(
        backend="neuron",
        supraspinal_drive_mode=args.supraspinal_drive_mode,
    )
    output_dir = ensure_dir(args.output_dir)
    seeds = tuple(config.seed_config.train_seeds)
    reference_dir = Path(output_dir).parent / "reference"
    reference_cache = resolve_reference_emg_cache(seeds, config, reference_dir=reference_dir)
    baseline = reference_baseline_stats(reference_dir, config.metric_config, seeds=seeds)
    sweep_results = run_physical_modulation_sweep_suite(
        config,
        seeds=seeds,
        seed_trial_budget=args.seed_trial_budget,
        reference_emg_by_seed=reference_cache,
    )
    best_pattern_record = best_record(sweep_results["all"])
    best_theta = config.theta_bounds.clip(
        {key.removeprefix("theta_"): value for key, value in best_pattern_record.items() if key.startswith("theta_")},
        device_config=config.device_config,
    )

    write_jsonl(output_dir / "patterns.jsonl", sweep_results["all"])
    write_json(
        output_dir / "summary.json",
        {
            "config": dataclass_config_bundle(config),
            "preset": "physical_modulation_default",
            "cost_metric": "device_cost",
            "cost_metric_label": "normalized_charge_rate_usage",
            "num_patterns": len(sweep_results["all"]),
            "evaluation_seed_policy": "three_train_seeds_per_pattern",
            "train_seeds": list(seeds),
            "seed_trials": len(sweep_results["all"]) * len(seeds),
            "requested_seed_trial_budget": int(args.seed_trial_budget),
            "allocation": sweep_results["preset"],
            "reference_dir": str(reference_dir),
            "lesion_no_stim_baseline": baseline,
            "best_pattern": {
                "theta": best_theta,
                "record": best_pattern_record,
            },
        },
    )
    write_best_emg_panel(
        method_key="grid_sweep",
        theta=best_theta,
        output_dir=output_dir,
        config=config,
        reference_dir=reference_dir,
    )
    plot_frontier(
        sweep_results["all"],
        sweep_results["frontier"],
        output_dir / "frontier.png",
        baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
        title="Grid sweep frontier",
    )


if __name__ == "__main__":
    main()
