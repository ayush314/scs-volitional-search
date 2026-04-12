#!/usr/bin/env python3
"""Run tonic, duty-cycle, and full-space sweep suites."""

from __future__ import annotations

import argparse
from pathlib import Path

from scs_search.analysis import best_record, reference_baseline_stats
from scs_search.config import DEFAULT_SWEEP_SEED_TRIALS, PatternParameters, SimulationConfig, dataclass_config_bundle
from scs_search.plotting import plot_frontier
from scs_search.simulator_adapter import resolve_reference_emg_cache, write_best_emg_panel
from scs_search.sweeps import run_sweep_suite
from scs_search.utils import ensure_dir, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the default sweep over tonic, duty-cycle, and Fourier patterns.")
    parser.add_argument("--output-dir", default="results/grid_sweep")
    parser.add_argument("--seed-trial-budget", type=int, default=DEFAULT_SWEEP_SEED_TRIALS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(backend="neuron")
    output_dir = ensure_dir(args.output_dir)
    seeds = tuple(config.seed_config.train_seeds)
    reference_dir = Path(output_dir).parent / "reference"
    reference_cache = resolve_reference_emg_cache(seeds, config, reference_dir=reference_dir)
    baseline = reference_baseline_stats(reference_dir, config.metric_config)
    sweep_results = run_sweep_suite(
        config,
        seeds=seeds,
        seed_trial_budget=args.seed_trial_budget,
        reference_emg_by_seed=reference_cache,
    )
    best_pattern_record = best_record(sweep_results["all"])

    write_jsonl(output_dir / "patterns.jsonl", sweep_results["all"])
    write_json(
        output_dir / "summary.json",
        {
            "config": dataclass_config_bundle(config),
            "preset": "default",
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
                "theta": {
                    key.removeprefix("theta_"): value
                    for key, value in best_pattern_record.items()
                    if key.startswith("theta_")
                },
                "record": best_pattern_record,
            },
        },
    )
    best_theta = PatternParameters.from_any(
        {key.removeprefix("theta_"): value for key, value in best_pattern_record.items() if key.startswith("theta_")}
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
