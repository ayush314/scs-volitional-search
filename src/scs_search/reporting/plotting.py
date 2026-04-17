"""Plotting helpers for the canonical task-burst physical-modulation study."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np

from ..config import StimPattern
from ..simulation.drive import SupraspinalDrive
from ..utils import ensure_dir

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
    """Return the public-facing display name for one method or source."""

    return _DISPLAY_NAMES.get(method_key, method_key.replace("_", " ").title())


def lesion_label(method_key: str) -> str:
    """Return the legend label for a lesion + method trace."""

    return f"Lesion + {display_name(method_key)}"


def _cost_value(record: Mapping[str, float]) -> float:
    return float(record["device_cost"])


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
    fig, axes = plt.subplots(seed_count, 1, figsize=(10.0, max(3.0 * seed_count, 3.5)), sharex=False)
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
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
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
    ax.legend(loc="lower right", frameon=True)
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
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
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
    ax.legend(loc="lower right", frameon=True)
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
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
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
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=4)
    if baseline_corr is not None:
        ax.axhline(float(baseline_corr), color="C3", linestyle="--", linewidth=1.2, label=baseline_label)
    ax.set_xticks(x, labels, rotation=22, ha="right")
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
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
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
    ax.legend(frameon=True, loc="lower right")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _instantaneous_pulse_rate(pattern: StimPattern) -> tuple[np.ndarray, np.ndarray]:
    """Return realized pulse-rate samples from consecutive pulse intervals."""

    if pattern.pulse_times_ms.size < 2:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    pulse_intervals_ms = np.diff(pattern.pulse_times_ms)
    valid = pulse_intervals_ms > 0.0
    if not np.any(valid):
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    return pattern.pulse_times_ms[1:][valid], 1000.0 / pulse_intervals_ms[valid]


def _pulse_train_trace(
    pattern: StimPattern,
    *,
    t_start_ms: float,
    t_end_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a piecewise-constant current waveform for the realized pulse train."""

    if t_end_ms <= t_start_ms:
        return np.asarray([t_start_ms, t_end_ms], dtype=float), np.zeros(2, dtype=float)

    x_values: list[float] = [float(t_start_ms)]
    y_values: list[float] = [0.0]
    last_time = float(t_start_ms)
    for pulse_time_ms, pulse_current_ma, pulse_width_us in zip(
        pattern.pulse_times_ms,
        pattern.pulse_current_ma,
        pattern.pulse_widths_us,
    ):
        pulse_start_ms = float(pulse_time_ms)
        pulse_end_ms = pulse_start_ms + (float(pulse_width_us) / 1000.0)
        if pulse_end_ms < t_start_ms or pulse_start_ms > t_end_ms:
            continue
        clipped_start_ms = max(pulse_start_ms, float(t_start_ms))
        clipped_end_ms = min(pulse_end_ms, float(t_end_ms))
        if clipped_start_ms > last_time:
            x_values.append(clipped_start_ms)
            y_values.append(0.0)
        x_values.extend([clipped_start_ms, clipped_end_ms, clipped_end_ms])
        y_values.extend([float(pulse_current_ma), float(pulse_current_ma), 0.0])
        last_time = clipped_end_ms
    if last_time < t_end_ms:
        x_values.append(float(t_end_ms))
        y_values.append(0.0)
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def _select_zoom_window(pattern: StimPattern) -> tuple[float, float]:
    """Choose one pulse window that exposes local variation in spacing, height, and width."""

    pulse_end_max_ms = (
        float(np.max(pattern.pulse_times_ms + (pattern.pulse_widths_us / 1000.0)))
        if pattern.pulse_times_ms.size
        else 0.0
    )
    if pattern.pulse_times_ms.size == 0:
        x_max = max(float(pattern.time_ms[-1]) if pattern.time_ms.size else 0.0, pulse_end_max_ms, 1.0)
        return 0.0, x_max
    if pattern.pulse_times_ms.size <= 8:
        start_ms = max(float(pattern.pulse_times_ms[0]) - 5.0, 0.0)
        end_ms = min(
            float(pattern.pulse_times_ms[-1]) + float(pattern.pulse_widths_us[-1]) / 1000.0 + 5.0,
            max(float(pattern.time_ms[-1]) if pattern.time_ms.size else 0.0, pulse_end_max_ms, 0.0),
        )
        return start_ms, max(end_ms, start_ms + 1.0)

    pulse_count = int(pattern.pulse_times_ms.size)
    window_pulses = min(18, pulse_count)
    intervals_ms = np.diff(pattern.pulse_times_ms)
    interval_values = np.concatenate([intervals_ms[:1], intervals_ms]) if intervals_ms.size else np.ones(pulse_count, dtype=float)
    current_norm = pattern.pulse_current_ma / max(float(np.ptp(pattern.pulse_current_ma)), 1e-9)
    width_norm = pattern.pulse_widths_us / max(float(np.ptp(pattern.pulse_widths_us)), 1e-9)
    interval_norm = interval_values / max(float(np.ptp(interval_values)), 1e-9)

    best_start = 0
    best_score = -np.inf
    for start in range(0, pulse_count - window_pulses + 1):
        stop = start + window_pulses
        score = (
            float(np.std(current_norm[start:stop]))
            + float(np.std(width_norm[start:stop]))
            + float(np.std(interval_norm[start:stop]))
        )
        if score > best_score:
            best_score = score
            best_start = start

    best_stop = best_start + window_pulses
    start_ms = float(pattern.pulse_times_ms[best_start])
    end_ms = float(pattern.pulse_times_ms[best_stop - 1]) + float(pattern.pulse_widths_us[best_stop - 1]) / 1000.0
    margin_ms = max(2.0, 0.1 * (end_ms - start_ms))
    full_end_ms = max(float(pattern.time_ms[-1]) if pattern.time_ms.size else end_ms, pulse_end_max_ms, end_ms)
    return max(start_ms - margin_ms, 0.0), min(end_ms + margin_ms, full_end_ms)


def _format_series_axis(ax: plt.Axes, values: np.ndarray, unit: str, *, decimals: int = 2) -> None:
    """Apply readable limits and annotate the displayed value range."""

    series = np.asarray(values, dtype=float)
    if series.size == 0:
        return

    value_min = float(np.min(series))
    value_max = float(np.max(series))
    tolerance = max(1e-9, 1e-6 * max(1.0, abs(value_min), abs(value_max)))

    if np.isclose(value_min, value_max, atol=tolerance, rtol=0.0):
        padding = max(0.05 * max(abs(value_max), 1.0), 1.0 if unit == "us" else 0.1)
        ax.set_ylim(value_min - padding, value_max + padding)
        label = f"constant: {value_min:.{decimals}f} {unit}"
    else:
        span = value_max - value_min
        padding = max(0.12 * span, 0.02 * max(abs(value_min), abs(value_max), 1.0))
        ax.set_ylim(value_min - padding, value_max + padding)
        label = f"range: {value_min:.{decimals}f}-{value_max:.{decimals}f} {unit}"

    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.text(
        0.99,
        0.9,
        label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.9},
    )


def plot_pattern_detail(pattern: StimPattern, output_path: str | Path) -> None:
    """Visualize the realized temporal stimulation pattern in detail."""

    output = _prepare_output(output_path)
    rate_times_ms, rate_hz = _instantaneous_pulse_rate(pattern)
    pulse_end_max_ms = (
        float(np.max(pattern.pulse_times_ms + (pattern.pulse_widths_us / 1000.0)))
        if pattern.pulse_times_ms.size
        else 0.0
    )
    x_max = max(float(pattern.time_ms[-1]) if pattern.time_ms.size else 0.0, pulse_end_max_ms, 0.0)
    zoom_start_ms, zoom_end_ms = _select_zoom_window(pattern)
    full_x, full_y = _pulse_train_trace(pattern, t_start_ms=0.0, t_end_ms=x_max)
    zoom_x, zoom_y = _pulse_train_trace(pattern, t_start_ms=zoom_start_ms, t_end_ms=zoom_end_ms)
    fig, axes = plt.subplots(5, 1, figsize=(11.0, 10.0), constrained_layout=True)

    axes[0].plot(full_x, full_y, color="C0", linewidth=1.35)
    axes[0].fill_between(full_x, 0.0, full_y, color="C0", alpha=0.22)
    axes[0].set_ylabel("Current (mA)")
    axes[0].set_title(f"Delivered pulse train: {pattern.family}")
    axes[0].set_xlim(-5.0, x_max + 5.0)

    axes[1].plot(zoom_x, zoom_y, color="C0", linewidth=1.5)
    axes[1].fill_between(zoom_x, 0.0, zoom_y, color="C0", alpha=0.24)
    axes[1].set_ylabel("Current (mA)")
    axes[1].set_xlim(zoom_start_ms, zoom_end_ms)
    axes[1].set_title(f"Zoomed pulse train ({zoom_start_ms:.1f}-{zoom_end_ms:.1f} ms)")

    if pattern.pulse_times_ms.size:
        axes[2].plot(pattern.pulse_times_ms, pattern.pulse_current_ma, color="C1", linewidth=1.2, alpha=0.9)
        axes[2].scatter(pattern.pulse_times_ms, pattern.pulse_current_ma, s=12, color="C1", alpha=0.85)
    axes[2].set_ylabel("Pulse current (mA)")
    axes[2].set_xlim(-5.0, x_max + 5.0)
    _format_series_axis(axes[2], pattern.pulse_current_ma, "mA")

    if rate_times_ms.size:
        axes[3].plot(rate_times_ms, rate_hz, color="C2", linewidth=1.2, alpha=0.9)
        axes[3].scatter(rate_times_ms, rate_hz, s=12, color="C2", alpha=0.85)
    axes[3].set_ylabel("Pulse rate (Hz)")
    axes[3].set_xlim(-5.0, x_max + 5.0)
    _format_series_axis(axes[3], rate_hz, "Hz", decimals=1)

    if pattern.pulse_times_ms.size:
        axes[4].plot(pattern.pulse_times_ms, pattern.pulse_widths_us, color="C3", linewidth=1.2, alpha=0.9)
        axes[4].scatter(pattern.pulse_times_ms, pattern.pulse_widths_us, s=12, color="C3", alpha=0.85)
    axes[4].set_ylabel("PW (us)")
    axes[4].set_xlabel(f"Time (ms, 0-{max(int(x_max), 0)})")
    axes[4].set_xlim(-5.0, x_max + 5.0)
    _format_series_axis(axes[4], pattern.pulse_widths_us, "us", decimals=1)

    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_supraspinal_drive_examples(
    seed_to_drive: Mapping[int, SupraspinalDrive],
    output_path: str | Path,
    *,
    title: str = "Supraspinal drive examples",
) -> None:
    """Plot one supraspinal drive rate trace per seed."""

    output = _prepare_output(output_path)
    seeds = [int(seed) for seed in seed_to_drive]
    if not seeds:
        raise ValueError("Cannot plot supraspinal drive examples without at least one seed.")

    fig, axes = plt.subplots(
        len(seeds),
        1,
        figsize=(10.5, max(2.1 * len(seeds), 3.0)),
        sharex=True,
        constrained_layout=True,
    )
    if len(seeds) == 1:
        axes = [axes]

    for axis, seed in zip(axes, seeds):
        drive = seed_to_drive[int(seed)]
        axis.plot(drive.time_ms, drive.rate_hz, color="#355070", linewidth=1.8)
        axis.fill_between(drive.time_ms, 0.0, drive.rate_hz, color="#8fbcd4", alpha=0.35)
        for burst_start_ms, burst_end_ms in drive.metadata.get("supraspinal_task_burst_intervals_ms", []):
            axis.axvspan(float(burst_start_ms), float(burst_end_ms), color="#d8e7d2", alpha=0.18, linewidth=0.0)
        axis.set_ylabel(f"Seed {seed}\nRate (Hz)")
        axis.set_ylim(0.0, max(1.0, float(np.max(drive.rate_hz)) * 1.08 if drive.rate_hz.size else 1.0))
        axis.grid(axis="y", alpha=0.18, linewidth=0.8)

    axes[0].set_title(title)
    axes[-1].set_xlabel(
        f"Time (ms, 0-{max(int(seed_to_drive[seeds[0]].time_ms[-1]) if seed_to_drive[seeds[0]].time_ms.size else 0, 0)})"
    )
    fig.savefig(output, dpi=220)
    plt.close(fig)
