"""Result summarization entrypoint used by the CLI script."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from ..config import theta_to_dict
from ..search.sweep import make_physical_modulation_simulation_config
from ..simulation.drive import build_supraspinal_drive
from ..simulation.evaluator import evaluate_best_candidate_report_summary, write_best_emg_panel
from ..utils import read_json, read_jsonl, write_json
from .analysis import best_so_far_trace, build_best_under_limit_frontier, comparable_optimizer_history, reference_baseline_stats
from .plotting import (
    display_name,
    lesion_label,
    plot_best_so_far,
    plot_frontier,
    plot_frontier_comparison,
    plot_frontier_overlay,
    plot_seed_sensitivity,
    plot_supraspinal_drive_examples,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison figures from completed result directories.")
    parser.add_argument("--results-root", default="results")
    return parser.parse_args()


def _config_from_results(_results_root: Path):
    config = make_physical_modulation_simulation_config(backend="neuron")
    for summary_path in (
        _results_root / "reference" / "summary.json",
        _results_root / "grid_sweep" / "summary.json",
        _results_root / "cmaes" / "summary.json",
        _results_root / "turbo" / "summary.json",
        _results_root / "bohb" / "summary.json",
    ):
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        simulation_dict = summary.get("config", {}).get("simulation", {})
        if not isinstance(simulation_dict, dict):
            continue
        return replace(
            config,
            dt_ms=float(simulation_dict.get("dt_ms", config.dt_ms)),
            pulse_scheduler_dt_ms=float(
                simulation_dict.get("pulse_scheduler_dt_ms", config.pulse_scheduler_dt_ms)
            ),
            simulation_duration_ms=int(
                simulation_dict.get("simulation_duration_ms", config.simulation_duration_ms)
            ),
            supraspinal_drive_mode=str(
                simulation_dict.get("supraspinal_drive_mode", config.supraspinal_drive_mode)
            ),
            supraspinal_rate_hz=float(simulation_dict.get("supraspinal_rate_hz", config.supraspinal_rate_hz)),
            supraspinal_rate_floor_hz=float(
                simulation_dict.get("supraspinal_rate_floor_hz", config.supraspinal_rate_floor_hz)
            ),
            supraspinal_envelope_control_dt_ms=float(
                simulation_dict.get(
                    "supraspinal_envelope_control_dt_ms",
                    config.supraspinal_envelope_control_dt_ms,
                )
            ),
            supraspinal_envelope_smoothing_sigma_ms=float(
                simulation_dict.get(
                    "supraspinal_envelope_smoothing_sigma_ms",
                    config.supraspinal_envelope_smoothing_sigma_ms,
                )
            ),
            supraspinal_envelope_ar_rho=float(
                simulation_dict.get("supraspinal_envelope_ar_rho", config.supraspinal_envelope_ar_rho)
            ),
            supraspinal_task_burst_min_ms=float(
                simulation_dict.get("supraspinal_task_burst_min_ms", config.supraspinal_task_burst_min_ms)
            ),
            supraspinal_task_burst_max_ms=float(
                simulation_dict.get("supraspinal_task_burst_max_ms", config.supraspinal_task_burst_max_ms)
            ),
            supraspinal_task_gap_min_ms=float(
                simulation_dict.get("supraspinal_task_gap_min_ms", config.supraspinal_task_gap_min_ms)
            ),
            supraspinal_task_gap_max_ms=float(
                simulation_dict.get("supraspinal_task_gap_max_ms", config.supraspinal_task_gap_max_ms)
            ),
            supraspinal_inhomogeneous_rate_hz=float(
                simulation_dict.get("supraspinal_inhomogeneous_rate_hz", config.supraspinal_inhomogeneous_rate_hz)
            ),
            healthy_perc_supra_intact=float(
                simulation_dict.get("healthy_perc_supra_intact", config.healthy_perc_supra_intact)
            ),
            lesion_perc_supra_intact=float(
                simulation_dict.get("lesion_perc_supra_intact", config.lesion_perc_supra_intact)
            ),
            baseline_cycle_ms=float(simulation_dict.get("baseline_cycle_ms", config.baseline_cycle_ms)),
            structural_seed=int(simulation_dict.get("structural_seed", config.structural_seed)),
        )
    return config


def _best_candidate_specs(results_root: Path) -> list[dict[str, object]]:
    """Collect the best pattern from the sweep and each optimizer."""

    candidates: list[dict[str, object]] = []
    for method_key, summary_path in (
        ("grid_sweep", results_root / "grid_sweep" / "summary.json"),
        ("cmaes", results_root / "cmaes" / "summary.json"),
        ("turbo", results_root / "turbo" / "summary.json"),
        ("bohb", results_root / "bohb" / "summary.json"),
    ):
        if not summary_path.exists():
            continue
        summary = read_json(summary_path)
        best_pattern = summary.get("best_pattern")
        if best_pattern is None:
            continue
        candidates.append(
            {
                "method_key": method_key,
                "display_name": display_name(method_key),
                "theta": best_pattern["theta"],
                "output_dir": results_root / method_key,
            }
        )
    return candidates


def refresh_reference_summary_baselines(results_root: Path, config) -> None:
    """Rewrite the reference summary with explicit all/train/report baselines."""

    reference_dir = results_root / "reference"
    summary_file = reference_dir / "summary.json"
    if not summary_file.exists():
        return

    summary = read_json(summary_file)
    train_baseline = reference_baseline_stats(reference_dir, config.metric_config, seeds=config.seed_config.train_seeds)
    report_baseline = reference_baseline_stats(reference_dir, config.metric_config, seeds=config.seed_config.report_seeds)
    all_baseline = reference_baseline_stats(reference_dir, config.metric_config)

    updated = False
    for key, baseline in (
        ("lesion_no_stim_baseline", all_baseline),
        ("lesion_no_stim_baseline_all", all_baseline),
        ("lesion_no_stim_baseline_train", train_baseline),
        ("lesion_no_stim_baseline_report", report_baseline),
    ):
        if baseline is not None and summary.get(key) != baseline:
            summary[key] = baseline
            updated = True

    if updated:
        write_json(summary_file, summary)


def maybe_plot_reference_drive_examples(results_root: Path, config) -> None:
    reference_dir = results_root / "reference"
    if not reference_dir.exists():
        return
    train_seeds = tuple(int(seed) for seed in config.seed_config.train_seeds)
    plot_supraspinal_drive_examples(
        {int(seed): build_supraspinal_drive(config, int(seed)) for seed in train_seeds},
        reference_dir / "supraspinal_drive.png",
        title="Train-seed supraspinal drive examples",
    )


def reevaluate_best_patterns(results_root: Path, config) -> dict[str, dict]:
    """Load or build held-out summaries for the best pattern from each method."""

    reference_dir = results_root / "reference"
    reevaluated: dict[str, dict] = {}
    for candidate in _best_candidate_specs(results_root):
        summary_path = Path(candidate["output_dir"]) / "final_report_summary.json"
        candidate_theta = config.theta_bounds.clip(candidate["theta"], device_config=config.device_config)
        candidate_theta_dict = theta_to_dict(candidate_theta)
        needs_refresh = True
        if summary_path.exists():
            existing = read_json(summary_path)
            existing_theta = existing.get("theta")
            needs_refresh = (
                "mean_relative_envelope_rmse" not in existing
                or existing_theta is None
                or theta_to_dict(existing_theta) != candidate_theta_dict
            )
        if needs_refresh:
            evaluate_best_candidate_report_summary(
                theta=candidate_theta,
                output_dir=candidate["output_dir"],
                config=config,
                reference_dir=reference_dir,
            )
        write_best_emg_panel(
            method_key=str(candidate["method_key"]),
            theta=candidate_theta,
            output_dir=candidate["output_dir"],
            config=config,
            reference_dir=reference_dir,
        )
        reevaluated[str(candidate["method_key"])] = read_json(summary_path)
    return reevaluated


def maybe_plot_optimizer_traces(results_root: Path, config) -> None:
    trace_map = {}
    baseline = reference_baseline_stats(
        results_root / "reference",
        config.metric_config,
        seeds=config.seed_config.train_seeds,
    )
    baseline_corr = None if baseline is None else float(baseline["mean_corr"])
    for algorithm in ("bohb", "turbo", "cmaes"):
        history_file = results_root / algorithm / "history.jsonl"
        if not history_file.exists():
            continue
        history = read_jsonl(history_file)
        comparable_history = comparable_optimizer_history(
            history,
            algorithm=algorithm,
            required_seed_budget=len(config.seed_config.train_seeds),
        )
        trace = best_so_far_trace(comparable_history)
        trace_map[lesion_label(algorithm)] = trace
        plot_best_so_far(
            {lesion_label(algorithm): trace},
            results_root / algorithm / "search_progress.png",
            baseline_corr=baseline_corr,
            title=f"{display_name(algorithm)} search progress across seed-level trials",
        )
    if trace_map:
        plot_best_so_far(
            trace_map,
            results_root / "optimizer_search_progress.png",
            baseline_corr=baseline_corr,
            title="Optimizer search progress across seed-level trials",
        )


def maybe_plot_seed_sensitivity(results_root: Path, reevaluated: dict[str, dict], config) -> None:
    labels = []
    means = []
    stds = []
    for method_key in ("grid_sweep", "cmaes", "turbo", "bohb"):
        summary = reevaluated.get(method_key)
        if summary is None:
            continue
        labels.append(display_name(method_key))
        means.append(float(summary["mean_corr"]))
        stds.append(float(summary["std_corr"]))
    if labels:
        baseline = reference_baseline_stats(
            results_root / "reference",
            config.metric_config,
            seeds=config.seed_config.report_seeds,
        )
        plot_seed_sensitivity(
            labels,
            means,
            stds,
            results_root / "optimizer_robustness.png",
            baseline_corr=None if baseline is None else float(baseline["mean_corr"]),
            title=f"Optimizer robustness on {len(config.seed_config.report_seeds)} report seeds",
        )


def maybe_plot_method_frontiers(results_root: Path, config) -> None:
    baseline = reference_baseline_stats(
        results_root / "reference",
        config.metric_config,
        seeds=config.seed_config.train_seeds,
    )
    baseline_corr = None if baseline is None else float(baseline["mean_corr"])
    grid_records: list[dict] = []
    grid_frontier: list[dict] = []
    grid_patterns_file = results_root / "grid_sweep" / "patterns.jsonl"
    if grid_patterns_file.exists():
        grid_records = read_jsonl(grid_patterns_file)
        grid_frontier = build_best_under_limit_frontier(grid_records)
        plot_frontier(
            grid_records,
            grid_frontier,
            results_root / "grid_sweep" / "frontier.png",
            baseline_corr=baseline_corr,
            title="Grid sweep frontier",
        )

    for algorithm in ("bohb", "turbo", "cmaes"):
        history_file = results_root / algorithm / "history.jsonl"
        if not history_file.exists():
            continue
        history = read_jsonl(history_file)
        comparable_history = comparable_optimizer_history(
            history,
            algorithm=algorithm,
            required_seed_budget=len(config.seed_config.train_seeds),
        )
        frontier = build_best_under_limit_frontier(comparable_history)
        plot_frontier(
            comparable_history,
            frontier,
            results_root / algorithm / "frontier.png",
            baseline_corr=baseline_corr,
            title=f"{display_name(algorithm)} frontier",
        )
        if grid_records:
            plot_frontier_overlay(
                grid_records,
                grid_frontier,
                comparable_history,
                frontier,
                results_root / algorithm / "frontier_with_grid.png",
                base_name="Grid sweep",
                overlay_name=display_name(algorithm),
                baseline_corr=baseline_corr,
            )


def maybe_plot_frontier_comparison(results_root: Path, config) -> None:
    frontier_map = {}
    for method_key, path in (
        ("grid_sweep", results_root / "grid_sweep" / "patterns.jsonl"),
        ("cmaes", results_root / "cmaes" / "history.jsonl"),
        ("turbo", results_root / "turbo" / "history.jsonl"),
        ("bohb", results_root / "bohb" / "history.jsonl"),
    ):
        if not path.exists():
            continue
        records = read_jsonl(path)
        records = comparable_optimizer_history(
            records,
            algorithm=method_key,
            required_seed_budget=len(config.seed_config.train_seeds),
        )
        frontier_map[display_name(method_key)] = build_best_under_limit_frontier(records)
    if frontier_map:
        baseline = reference_baseline_stats(
            results_root / "reference",
            config.metric_config,
            seeds=config.seed_config.train_seeds,
        )
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
    config = _config_from_results(results_root)
    refresh_reference_summary_baselines(results_root, config)
    maybe_plot_reference_drive_examples(results_root, config)
    reevaluated = reevaluate_best_patterns(results_root, config)
    maybe_plot_optimizer_traces(results_root, config)
    maybe_plot_seed_sensitivity(results_root, reevaluated, config)
    maybe_plot_method_frontiers(results_root, config)
    maybe_plot_frontier_comparison(results_root, config)


__all__ = [
    "_config_from_results",
    "main",
    "maybe_plot_frontier_comparison",
    "maybe_plot_method_frontiers",
    "maybe_plot_optimizer_traces",
    "maybe_plot_reference_drive_examples",
    "maybe_plot_seed_sensitivity",
    "parse_args",
    "reevaluate_best_patterns",
    "refresh_reference_summary_baselines",
]
