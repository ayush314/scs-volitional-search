#!/usr/bin/env python3
"""Run tonic, duty-cycle, and full-space sweep suites."""

from __future__ import annotations

import argparse
from pathlib import Path

from scs_search.analysis import reference_baseline_stats
from scs_search.config import SimulationConfig, dataclass_config_bundle
from scs_search.plotting import plot_frontier
from scs_search.simulator_adapter import resolve_reference_emg_cache
from scs_search.sweeps import run_sweep_suite
from scs_search.utils import ensure_dir, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full study sweep over tonic, duty-cycle, and Fourier patterns.")
    parser.add_argument("--output-dir", default="results/grid_sweep")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(backend="neuron")
    output_dir = ensure_dir(args.output_dir)
    seeds = config.seed_config.train_seeds
    reference_dir = Path(output_dir).parent / "reference"
    reference_cache = resolve_reference_emg_cache(seeds, config, reference_dir=reference_dir)
    baseline = reference_baseline_stats(reference_dir, config.metric_config)
    sweep_results = run_sweep_suite(config, seeds=seeds, reference_emg_by_seed=reference_cache)

    write_json(output_dir / "config.json", dataclass_config_bundle(config))
    write_jsonl(output_dir / "metrics.jsonl", sweep_results["all"])
    write_csv(output_dir / "metrics.csv", sweep_results["all"])
    write_json(output_dir / "frontier.json", sweep_results["frontier"])
    write_json(
        output_dir / "summary.json",
        {
            "preset": "study",
            "num_records": len(sweep_results["all"]),
            "train_seeds": list(seeds),
            "seed_trials": len(sweep_results["all"]) * len(seeds),
            "reference_dir": str(reference_dir),
            "lesion_no_stim_baseline": baseline,
        },
    )
    plot_frontier(
        sweep_results["all"],
        sweep_results["frontier"],
        output_dir / "frontier.png",
        baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
    )


if __name__ == "__main__":
    main()
