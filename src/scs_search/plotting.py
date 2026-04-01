"""Plotting helpers for references, frontiers, and optimizer traces."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from .config import StimPattern
from .utils import ensure_dir


def _prepare_output(path: str | Path) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    return output_path


def _cost_value(record: Mapping[str, float]) -> float:
    """Return the public hardware-budget cost from a flat record."""

    return float(record.get("device_cost", record.get("norm_dose", 0.0)))


def plot_emg_examples(reference_emg: np.ndarray, candidate_emg: np.ndarray, output_path: str | Path, title: str) -> None:
    """Plot a pre-lesion vs lesion+SCS EMG comparison."""

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(10, 4))
    duration_ms = max(len(reference_emg), len(candidate_emg))
    reference_time = np.arange(len(reference_emg), dtype=float)
    candidate_time = np.arange(len(candidate_emg), dtype=float)
    ax.plot(reference_time, reference_emg, label="Pre-lesion", linewidth=1.5)
    ax.plot(candidate_time, candidate_emg, label="Lesion + SCS", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel(f"Time (ms, 0-{max(duration_ms - 1, 0)})")
    ax.set_ylabel("EMG")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _scatter_with_hull(
    ax: plt.Axes,
    records: Iterable[Mapping[str, float]],
    frontier: Iterable[Mapping[str, float]],
    *,
    color: str,
    scatter_label: str,
    hull_label: str,
    scatter_alpha: float = 0.4,
) -> None:
    """Plot evaluated points together with an upper-hull overlay."""

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
        ax.plot(
            [_cost_value(record) for record in frontier_list],
            [float(record["mean_corr"]) for record in frontier_list],
            marker="o",
            linewidth=1.5,
            color=color,
            label=hull_label,
        )


def plot_frontier(
    records: Iterable[Mapping[str, float]],
    frontier: Iterable[Mapping[str, float]],
    output_path: str | Path,
    *,
    baseline_corr: float | None = None,
    baseline_label: str = "Lesion no stim baseline",
) -> None:
    """Plot restoration vs normalized device budget with the upper-hull overlay."""

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    _scatter_with_hull(
        ax,
        records,
        frontier,
        color="C0",
        scatter_label="Evaluated patterns",
        hull_label="Upper hull",
    )
    if baseline_corr is not None:
        ax.axhline(float(baseline_corr), color="C3", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.set_xlabel("Normalized device budget usage")
    ax.set_ylabel("EMG restoration correlation")
    ax.set_title("Device budget vs restoration hull")
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
    """Plot one device-budget/correlation hull on top of another reference hull."""

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(6, 4))
    _scatter_with_hull(
        ax,
        base_records,
        base_frontier,
        color="0.7",
        scatter_label=f"{base_name} patterns",
        hull_label=f"{base_name} hull",
        scatter_alpha=0.25,
    )
    _scatter_with_hull(
        ax,
        overlay_records,
        overlay_frontier,
        color="C1",
        scatter_label=f"{overlay_name} patterns",
        hull_label=f"{overlay_name} hull",
    )
    if baseline_corr is not None:
        ax.axhline(float(baseline_corr), color="C3", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.set_xlabel("Normalized device budget usage")
    ax.set_ylabel("EMG restoration correlation")
    ax.set_title(f"Device budget vs restoration: {overlay_name} over {base_name}")
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
) -> None:
    """Plot best-so-far restoration versus equivalent full evaluations."""

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
    ax.set_ylabel("Best-so-far restoration")
    ax.set_title("Best-so-far restoration vs seed-level trials")
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
) -> None:
    """Plot mean ± std across repeated seeds."""

    output = _prepare_output(output_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4)
    if baseline_corr is not None:
        ax.axhline(float(baseline_corr), color="C3", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel("Correlation")
    ax.set_title("Sensitivity to seed-level noise")
    if baseline_corr is not None:
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
    axes[1].set_ylabel("Pulse alpha")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)
