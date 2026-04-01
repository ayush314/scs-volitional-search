#!/usr/bin/env python3
"""Summarize result directories into comparison plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scs_search.analysis import reference_baseline_stats
from scs_search.config import PatientConditionSpec, PatternParameters, SimulationConfig
from scs_search.patterns import generate_stim_pattern
from scs_search.plotting import plot_best_so_far, plot_emg_examples, plot_frontier, plot_seed_sensitivity
from scs_search.simulator_adapter import run_condition
from scs_search.utils import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison figures from completed result directories.")
    parser.add_argument("--results-root", default="results")
    return parser.parse_args()


def _theta_from_flat_record(record: dict[str, float]) -> dict[str, float]:
    """Recover a theta mapping from a flattened record."""

    return {
        key.removeprefix("theta_"): float(value)
        for key, value in record.items()
        if key.startswith("theta_")
    }


def _candidate_pattern_records(results_root: Path) -> list[dict[str, object]]:
    """Collect candidate patterns from sweep and optimizer outputs."""

    candidates: list[dict[str, object]] = []

    sweep_metrics = results_root / "grid_sweep" / "metrics.jsonl"
    if sweep_metrics.exists():
        with sweep_metrics.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                theta = _theta_from_flat_record(record)
                if theta:
                    candidates.append(
                        {
                            "source": str(record.get("slice", "grid_sweep")),
                            "mean_corr": float(record["mean_corr"]),
                            "theta": theta,
                        }
                    )

    for algorithm in ("bohb", "turbo", "cmaes"):
        summary_file = results_root / algorithm / "summary.json"
        if not summary_file.exists():
            continue
        summary = read_json(summary_file)
        incumbent = summary["incumbent_summary"]
        candidates.append(
            {
                "source": algorithm,
                "mean_corr": float(incumbent["mean_corr"]),
                "theta": incumbent["theta"],
            }
        )
    return candidates


def maybe_plot_reference(results_root: Path) -> None:
    reference_dir = results_root / "reference"
    emg_path = reference_dir / "emg_arrays.npz"
    if not emg_path.exists():
        return
    candidates = _candidate_pattern_records(results_root)
    if not candidates:
        return

    arrays = np.load(emg_path)
    healthy_key = next((key for key in arrays.files if key.startswith("healthy_prelesion")), None)
    if healthy_key is None:
        return

    best_candidate = max(candidates, key=lambda record: float(record["mean_corr"]))
    healthy_emg = arrays[healthy_key]
    seed = int(healthy_key.rsplit("_", 1)[-1])
    baseline = reference_baseline_stats(reference_dir, SimulationConfig().metric_config)
    baseline_text = ""
    if baseline is not None:
        baseline_text = f" | lesion baseline corr={float(baseline['mean_corr']):.3f}"

    config = SimulationConfig(backend="neuron")
    lesion_condition = PatientConditionSpec("lesion_best_stim", config.lesion_perc_supra_intact)
    stim_pattern = generate_stim_pattern(
        PatternParameters.from_any(best_candidate["theta"]),
        t_end_ms=config.simulation_duration_ms,
        dt_ms=config.dt_ms,
        device_config=config.device_config,
    )
    lesion_result = run_condition(lesion_condition, stim_pattern, [seed], config)[0]
    plot_emg_examples(
        healthy_emg,
        lesion_result.emg_signal,
        results_root / "reference_emg.png",
        f"Pre-lesion vs lesion + best stimulation ({best_candidate['source']}){baseline_text}",
    )


def maybe_plot_frontier(results_root: Path) -> None:
    sweep_dir = results_root / "grid_sweep"
    if not sweep_dir.exists():
        return
    frontier_file = sweep_dir / "frontier.json"
    metrics_file = sweep_dir / "metrics.jsonl"
    if frontier_file.exists() and metrics_file.exists():
        frontier = read_json(frontier_file)
        with metrics_file.open("r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle]
        baseline = reference_baseline_stats(results_root / "reference", SimulationConfig().metric_config)
        plot_frontier(
            records,
            frontier,
            results_root / "frontier.png",
            baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
        )


def maybe_plot_optimizer_traces(results_root: Path) -> None:
    trace_map = {}
    for algorithm in ("bohb", "turbo", "cmaes"):
        trace_file = results_root / algorithm / "trace.json"
        if trace_file.exists():
            trace_map[algorithm.upper()] = read_json(trace_file)
    if trace_map:
        baseline = reference_baseline_stats(results_root / "reference", SimulationConfig().metric_config)
        plot_best_so_far(
            trace_map,
            results_root / "optimizer_comparison.png",
            baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
        )


def maybe_plot_seed_sensitivity(results_root: Path) -> None:
    summary_labels = []
    summary_means = []
    summary_stds = []
    for algorithm in ("bohb", "turbo", "cmaes"):
        summary_file = results_root / algorithm / "summary.json"
        if summary_file.exists():
            summary = read_json(summary_file)
            incumbent = summary["incumbent_summary"]
            summary_labels.append(algorithm.upper())
            summary_means.append(float(incumbent["mean_corr"]))
            summary_stds.append(float(incumbent["std_corr"]))
    if summary_labels:
        baseline = reference_baseline_stats(results_root / "reference", SimulationConfig().metric_config)
        plot_seed_sensitivity(
            summary_labels,
            summary_means,
            summary_stds,
            results_root / "seed_sensitivity.png",
            baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
        )


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    maybe_plot_reference(results_root)
    maybe_plot_frontier(results_root)
    maybe_plot_optimizer_traces(results_root)
    maybe_plot_seed_sensitivity(results_root)


if __name__ == "__main__":
    main()
