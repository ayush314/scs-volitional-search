#!/usr/bin/env python3
"""Summarize result directories into comparison plots."""

from __future__ import annotations

import argparse
from pathlib import Path

from scs_search.analysis import best_so_far_trace, build_best_under_limit_frontier, reference_baseline_stats
from scs_search.config import PatternParameters, SimulationConfig
from scs_search.plotting import display_name, lesion_label, plot_best_so_far, plot_frontier_comparison, plot_seed_sensitivity
from scs_search.simulator_adapter import evaluate_best_candidate_report_summary
from scs_search.utils import read_json, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison figures from completed result directories.")
    parser.add_argument("--results-root", default="results")
    return parser.parse_args()


def _best_candidate_specs(results_root: Path) -> list[dict[str, object]]:
    """Collect the best pattern from the sweep and each optimizer."""

    candidates: list[dict[str, object]] = []
    sweep_summary_file = results_root / "grid_sweep" / "summary.json"
    if sweep_summary_file.exists():
        sweep_summary = read_json(sweep_summary_file)
        best_pattern = sweep_summary.get("best_pattern")
        if best_pattern is not None:
            candidates.append(
                {
                    "method_key": "grid_sweep",
                    "display_name": display_name("grid_sweep"),
                    "theta": best_pattern["theta"],
                    "output_dir": results_root / "grid_sweep",
                }
            )

    for algorithm in ("bohb", "turbo", "cmaes"):
        summary_file = results_root / algorithm / "summary.json"
        if not summary_file.exists():
            continue
        summary = read_json(summary_file)
        best_pattern = summary["best_pattern"]
        candidates.append(
            {
                "method_key": algorithm,
                "display_name": display_name(algorithm),
                "theta": best_pattern["theta"],
                "output_dir": results_root / algorithm,
            }
        )
    return candidates


def maybe_plot_optimizer_traces(results_root: Path) -> None:
    trace_map = {}
    for algorithm in ("bohb", "turbo", "cmaes"):
        history_file = results_root / algorithm / "history.jsonl"
        if history_file.exists():
            history = read_jsonl(history_file)
            trace = best_so_far_trace(history)
            trace_map[lesion_label(algorithm)] = trace
            baseline = reference_baseline_stats(results_root / "reference", SimulationConfig().metric_config)
            plot_best_so_far(
                {lesion_label(algorithm): trace},
                results_root / algorithm / "search_progress.png",
                baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
                title=f"{display_name(algorithm)} search progress across seed-level trials",
            )
    if trace_map:
        baseline = reference_baseline_stats(results_root / "reference", SimulationConfig().metric_config)
        plot_best_so_far(
            trace_map,
            results_root / "optimizer_search_progress.png",
            baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
            title="Optimizer search progress across seed-level trials",
        )


def reevaluate_best_patterns(results_root: Path) -> dict[str, dict]:
    """Load or build the final report summary for the best pattern from each method."""

    config = SimulationConfig(backend="neuron")
    reference_dir = results_root / "reference"
    reevaluated: dict[str, dict] = {}
    for candidate in _best_candidate_specs(results_root):
        summary_path = Path(candidate["output_dir"]) / "final_report_summary.json"
        if not summary_path.exists():
            evaluate_best_candidate_report_summary(
                theta=PatternParameters.from_any(candidate["theta"]),
                output_dir=candidate["output_dir"],
                config=config,
                reference_dir=reference_dir,
            )
        reevaluated[str(candidate["method_key"])] = read_json(summary_path)
    return reevaluated


def maybe_plot_seed_sensitivity(results_root: Path, reevaluated: dict[str, dict]) -> None:
    summary_labels = []
    summary_means = []
    summary_stds = []
    for method_key in ("grid_sweep", "bohb", "turbo", "cmaes"):
        summary = reevaluated.get(method_key)
        if summary is not None:
            summary_labels.append(display_name(method_key))
            summary_means.append(float(summary["mean_corr"]))
            summary_stds.append(float(summary["std_corr"]))
    if summary_labels:
        baseline = reference_baseline_stats(results_root / "reference", SimulationConfig().metric_config)
        report_seed_count = len(SimulationConfig().seed_config.report_seeds)
        plot_seed_sensitivity(
            summary_labels,
            summary_means,
            summary_stds,
            results_root / "optimizer_robustness.png",
            baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
            title=f"Optimizer robustness on {report_seed_count} report seeds",
        )


def _compress_frontier(frontier: list[dict]) -> list[dict]:
    """Keep only change points from a step frontier."""

    compressed: list[dict] = []
    last_corr = None
    for record in frontier:
        corr = float(record["mean_corr"])
        if last_corr is None or corr != last_corr:
            compressed.append(record)
            last_corr = corr
    if frontier and (not compressed or compressed[-1] is not frontier[-1]):
        compressed.append(frontier[-1])
    return compressed


def maybe_plot_frontier_comparison(results_root: Path) -> None:
    frontier_map = {}
    for method_key, path in (
        ("grid_sweep", results_root / "grid_sweep" / "patterns.jsonl"),
        ("bohb", results_root / "bohb" / "history.jsonl"),
        ("turbo", results_root / "turbo" / "history.jsonl"),
        ("cmaes", results_root / "cmaes" / "history.jsonl"),
    ):
        if path.exists():
            frontier_map[display_name(method_key)] = _compress_frontier(build_best_under_limit_frontier(read_jsonl(path)))
    if frontier_map:
        baseline = reference_baseline_stats(results_root / "reference", SimulationConfig().metric_config)
        plot_frontier_comparison(
            frontier_map,
            results_root / "optimizer_frontiers.png",
            baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
            title="Optimizer frontiers",
        )


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    reevaluated = reevaluate_best_patterns(results_root)
    maybe_plot_optimizer_traces(results_root)
    maybe_plot_seed_sensitivity(results_root, reevaluated)
    maybe_plot_frontier_comparison(results_root)


if __name__ == "__main__":
    main()
