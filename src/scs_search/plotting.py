"""Plotting helpers for references, frontiers, and optimizer traces."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from .config import StimPattern
from .utils import ensure_dir

_DISPLAY_NAMES = {
    "grid_sweep": "Grid sweep",
    "cmaes": "CMA-ES",
    "turbo": "TuRBO",
    "bohb": "BOHB",
}
WAVEFORM_Y_LABEL = "EMG"
WAVEFORM_Y_LIMS = (-10.0, 10.0)
SCORE_Y_LABEL = "Pre-lesion EMG correlation (Pearson r)"


def _prepare_output(path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    return output_path


def display_name(method_key: str) -> str:
    """Return the public-facing display name for a method or source."""

    return _DISPLAY_NAMES.get(method_key, method_key.replace("_", " ").title())


def lesion_label(method_key: str) -> str:
    """Return the legend label for a lesion + method trace."""

    return f"Lesion + {display_name(method_key)}"


def _cost_value(record: Mapping[str, float]) -> float:
    """Return the public hardware-budget cost from a flat record."""

    return float(record["device_cost"])


def plot_emg_examples(
    reference_emg: np.ndarray,
    candidate_emg: np.ndarray,
    output_path: str | Path,
    title: str,
    *,
    reference_label: str = "Pre-lesion",
    candidate_label: str = "Lesion + SCS",
) -> None:
    """Plot two EMG traces with caller-provided labels."""

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    duration_ms = max(len(reference_emg), len(candidate_emg))
    reference_time = np.arange(len(reference_emg), dtype=float)
    candidate_time = np.arange(len(candidate_emg), dtype=float)
    ax.plot(reference_time, reference_emg, label=reference_label, linewidth=1.5)
    ax.plot(candidate_time, candidate_emg, label=candidate_label, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(f"Time (ms, 0-{max(duration_ms - 1, 0)})")
    ax.set_ylabel(WAVEFORM_Y_LABEL)
    ax.set_ylim(*WAVEFORM_Y_LIMS)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_emg_seed_panels(
    seed_to_reference_emg: Mapping[int, np.ndarray],
    seed_to_candidate_emg: Mapping[int, np.ndarray],
    output_path: str | Path,
    title: str,
    *,
    reference_label: str = "Pre-lesion",
    candidate_label: str = "Lesion + SCS",
) -> None:
    """Plot one EMG comparison panel per seed."""

    output = _prepare_output(output_path)
    seeds = [int(seed) for seed in seed_to_reference_emg if int(seed) in {int(k) for k in seed_to_candidate_emg}]
    if not seeds:
        raise ValueError("No overlapping seeds available for EMG panel plotting.")

    ordered_candidate = {int(seed): seed_to_candidate_emg[int(seed)] for seed in seed_to_candidate_emg}
    seed_count = len(seeds)
    fig, axes = plt.subplots(seed_count, 1, figsize=(10, max(3.0 * seed_count, 3.5)), sharex=False)
    if seed_count == 1:
        axes = [axes]

    max_duration_ms = 0
    for ax, seed in zip(axes, seeds):
        reference_emg = np.asarray(seed_to_reference_emg[int(seed)], dtype=float)
        candidate_emg = np.asarray(ordered_candidate[int(seed)], dtype=float)
        duration_ms = max(len(reference_emg), len(candidate_emg))
        max_duration_ms = max(max_duration_ms, duration_ms)
        ax.plot(np.arange(len(reference_emg), dtype=float), reference_emg, label=reference_label, linewidth=1.4)
        ax.plot(np.arange(len(candidate_emg), dtype=float), candidate_emg, label=candidate_label, linewidth=1.2)
        ax.set_ylabel(WAVEFORM_Y_LABEL)
        ax.set_ylim(*WAVEFORM_Y_LIMS)
        ax.set_title(f"Seed {seed}")

    axes[0].legend()
    axes[-1].set_xlabel(f"Time (ms, 0-{max(max_duration_ms - 1, 0)})")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _scatter_with_frontier(
    ax: plt.Axes,
    records: Iterable[Mapping[str, float]],
    frontier: Iterable[Mapping[str, float]],
    *,
    color: str,
    scatter_label: str,
    frontier_label: str,
    scatter_alpha: float = 0.4,
) -> None:
    """Plot evaluated points together with a best-under-limit frontier."""

    records_list = list(records)
    if records_list:
        ax.scatter(
            [_cost_value(record) for record in records_list],
            [float(record["mean_corr"]) for record in records_list],
            s=16,
            alpha=scatter_alpha,
            color=color,
            label=scatter_label,
        )

    frontier_list = list(frontier)
    if frontier_list:
        ax.step(
            [_cost_value(record) for record in frontier_list],
            [float(record["mean_corr"]) for record in frontier_list],
            where="post",
            linewidth=1.5,
            color=color,
            label=frontier_label,
        )


def plot_frontier(
    records: Iterable[Mapping[str, float]],
    frontier: Iterable[Mapping[str, float]],
    output_path: str | Path,
    *,
    baseline_corr: float | None = None,
    baseline_label: str = "Lesion no stim baseline",
    title: str = "Pattern frontier",
) -> None:
    """Plot restoration vs normalized charge-rate usage with the frontier overlay."""

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    _scatter_with_frontier(
        ax,
        records,
        frontier,
        color="C0",
        scatter_label="Evaluated patterns",
        frontier_label="Best observed up to usage level",
    )
    if baseline_corr is not None:
        ax.axhline(float(baseline_corr), color="C3", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.set_xlabel("Normalized charge-rate usage")
    ax.set_ylabel(SCORE_Y_LABEL)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_frontier_overlay(
    base_records: Iterable[Mapping[str, float]],
    base_frontier: Iterable[Mapping[str, float]],
    overlay_records: Iterable[Mapping[str, float]],
    overlay_frontier: Iterable[Mapping[str, float]],
    output_path: str | Path,
    *,
    base_name: str,
    overlay_name: str,
    baseline_corr: float | None = None,
    baseline_label: str = "Lesion no stim baseline",
) -> None:
    """Plot one charge-rate frontier on top of another reference frontier."""

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    _scatter_with_frontier(
        ax,
        base_records,
        base_frontier,
        color="0.7",
        scatter_label=f"{base_name} patterns",
        frontier_label=f"{base_name} frontier",
        scatter_alpha=0.25,
    )
    _scatter_with_frontier(
        ax,
        overlay_records,
        overlay_frontier,
        color="C1",
        scatter_label=f"{overlay_name} patterns",
        frontier_label=f"{overlay_name} frontier",
    )
    if baseline_corr is not None:
        ax.axhline(float(baseline_corr), color="C3", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.set_xlabel("Normalized charge-rate usage")
    ax.set_ylabel(SCORE_Y_LABEL)
    ax.set_title(f"{overlay_name} vs {base_name} frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_best_so_far(
    algorithm_to_trace: Mapping[str, Iterable[Mapping[str, float]]],
    output_path: str | Path,
    *,
    baseline_corr: float | None = None,
    baseline_label: str = "Lesion no stim baseline",
    title: str = "Search progress across seed-level trials",
) -> None:
    """Plot best-so-far restoration versus seed-level trials."""

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    for algorithm, trace in algorithm_to_trace.items():
        trace_list = list(trace)
        ax.plot(
            [float(row.get("seed_trials_used", row.get("evaluation_index", row.get("call_index", 0)))) for row in trace_list],
            [float(row["best_so_far"]) for row in trace_list],
            label=algorithm,
        )
    if baseline_corr is not None:
        ax.axhline(float(baseline_corr), color="C3", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.set_xlabel("Seed-level trials")
    ax.set_ylabel(SCORE_Y_LABEL)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_seed_sensitivity(
    labels: list[str],
    means: list[float],
    stds: list[float],
    output_path: str | Path,
    *,
    baseline_corr: float | None = None,
    baseline_label: str = "Lesion no stim baseline",
    title: str = "Pattern robustness across report seeds",
) -> None:
    """Plot mean ± std across repeated seeds."""

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4)
    if baseline_corr is not None:
        ax.axhline(float(baseline_corr), color="C3", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel(SCORE_Y_LABEL)
    ax.set_title(title)
    if baseline_corr is not None:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_frontier_comparison(
    frontier_map: Mapping[str, Iterable[Mapping[str, float]]],
    output_path: str | Path,
    *,
    baseline_corr: float | None = None,
    baseline_label: str = "Lesion no stim baseline",
    title: str = "Optimizer frontiers",
) -> None:
    """Plot multiple charge-rate frontiers on one shared axis."""

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, frontier in frontier_map.items():
        frontier_list = list(frontier)
        if not frontier_list:
            continue
        ax.step(
            [_cost_value(record) for record in frontier_list],
            [float(record["mean_corr"]) for record in frontier_list],
            where="post",
            linewidth=1.6,
            label=label,
        )
    if baseline_corr is not None:
        ax.axhline(float(baseline_corr), color="C3", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.set_xlabel("Normalized charge-rate usage")
    ax.set_ylabel(SCORE_Y_LABEL)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_pattern(pattern: StimPattern, output_path: str | Path) -> None:
    """Visualize alpha(t) and the pulse-time amplitudes of a pattern."""

    output = _prepare_output(output_path)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(pattern.time_ms, pattern.alpha_t)
    axes[0].set_ylabel("alpha(t)")
    axes[0].set_title(f"Pattern family: {pattern.family}")
    axes[1].stem(pattern.pulse_times_ms, pattern.pulse_alpha, basefmt=" ")
    axes[1].set_xlabel(f"Time (ms, 0-{max(int(pattern.time_ms[-1]) if pattern.time_ms.size else 0, 0)})")
    axes[1].set_ylabel("Pulse amplitude (alpha)")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
