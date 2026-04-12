#!/usr/bin/env python3
"""Generate, save, and plot healthy and lesioned baseline EMG waveforms for several virtual patients."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scs_search.config import SimulationConfig, PatientConditionSpec
from scs_search.patterns import generate_tonic_pattern
from scs_search.simulator_adapter import (
    resolve_reference_emg_cache,
    run_single_condition,
)
from scs_search.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and plot healthy and lesioned baseline EMG waveforms."
    )
    parser.add_argument(
        "--output-dir",
        default="results/baseline_demo",
        help="Directory to save the plots and cache.",
    )
    parser.add_argument(
        "--num-patients",
        type=int,
        default=3,
        help="Number of virtual patients (seeds) to simulate.",
    )
    parser.add_argument(
        "--plot-lesioned",
        action="store_true",
        help="Also generate and overlay simulated lesioned EMG responses.",
    )
    return parser.parse_args()


def generate_and_plot_healthy(
    patient_seeds: list[int],
    config: SimulationConfig,
    reference_dir: Path,
    output_dir: Path,
) -> dict[int, np.ndarray]:
    """Generate healthy reference traces and plot them alone."""
    print(
        f"[Healthy] Simulating healthy pre-lesion models for {len(patient_seeds)} patients..."
    )

    # This automatically generates missing traces and SAVES them as an .npz cache to reference_dir
    emg_by_seed = resolve_reference_emg_cache(
        seeds=patient_seeds, config=config, reference_dir=reference_dir
    )

    print("[Healthy] Plotting waveforms using seaborn...")
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        len(patient_seeds), 1, figsize=(10, 3 * len(patient_seeds)), sharex=True
    )
    if len(patient_seeds) == 1:
        axes = [axes]

    for ax, seed in zip(axes, patient_seeds):
        emg_signal = emg_by_seed[seed]

        # Build the exact time vector the simulation used (down to dt_ms resolution)
        duration_ms = float(config.simulation_duration_ms)
        dt_ms = float(config.dt_ms)
        time_ms = np.arange(0, duration_ms, dt_ms)

        min_len = min(len(time_ms), len(emg_signal))

        # Plot patient wave using seaborn
        sns.lineplot(
            x=time_ms[:min_len],
            y=emg_signal[:min_len],
            ax=ax,
            label="Healthy",
            color=sns.color_palette("muted")[0],
            alpha=0.9,
            linewidth=1.0,
        )

        ax.set_ylabel("EMG Amplitude (mV)")
        ax.set_title(f"Patient Trial (Seed {seed}) - Pre-Lesion Healthy Form")
        ax.set_xlim(0, 500)
        ax.legend()

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()

    plot_path = output_dir / "healthy_waveforms.png"
    plt.savefig(plot_path, dpi=300)
    print(f"[Healthy] Plot saved successfully to {plot_path}")

    return emg_by_seed


def generate_and_plot_lesioned(
    patient_seeds: list[int],
    config: SimulationConfig,
    healthy_emg_by_seed: dict[int, np.ndarray],
    output_dir: Path,
) -> dict[int, np.ndarray]:
    """Generate lesioned (unhealthy, no-SCS) traces and overlay them on the healthy plots."""
    print(
        f"\n[Lesioned] Simulating lesioned (no-SCS) models for {len(patient_seeds)} patients with varying severities..."
    )

    # A zero-alpha tonic pattern guarantees zero SCS stimulation (unhealthy baseline)
    zero_pattern = generate_tonic_pattern(
        freq_hz=40.0,
        alpha=0.0,
        t_end_ms=config.simulation_duration_ms,
        pw_us=config.device_config.default_pulse_width_us,
        dt_ms=config.dt_ms,
        device_config=config.device_config,
    )

    lesioned_emg_by_seed = {}
    lesion_degrees = {}

    for seed in patient_seeds:
        # Randomly vary the degree of lesion from 5% to 35% intact
        perc_intact = round(random.uniform(0.05, 0.35), 3)
        custom_lesioned_condition = PatientConditionSpec(
            label=f"lesion_{int(perc_intact*100)}", perc_supra_intact=perc_intact
        )
        # Run simulation for the randomized lesioned state individually
        result = run_single_condition(
            custom_lesioned_condition, zero_pattern, seed, config
        )
        lesioned_emg_by_seed[seed] = result.emg_signal
        lesion_degrees[seed] = perc_intact

    print("[Lesioned] Overlaying varying lesioned waveforms using seaborn...")
    sns.set_theme(style="whitegrid")

    fig, axes = plt.subplots(
        len(patient_seeds), 1, figsize=(10, 3 * len(patient_seeds)), sharex=True
    )
    if len(patient_seeds) == 1:
        axes = [axes]

    for ax, seed in zip(axes, patient_seeds):
        healthy_signal = healthy_emg_by_seed[seed]
        lesioned_signal = lesioned_emg_by_seed[seed]
        perc_intact = lesion_degrees[seed]

        duration_ms = float(config.simulation_duration_ms)
        dt_ms = float(config.dt_ms)
        time_ms = np.arange(0, duration_ms, dt_ms)

        min_len_h = min(len(time_ms), len(healthy_signal))
        min_len_l = min(len(time_ms), len(lesioned_signal))

        # Plot Healthy as a faded baseline
        sns.lineplot(
            x=time_ms[:min_len_h],
            y=healthy_signal[:min_len_h],
            ax=ax,
            label="Healthy Reference",
            color=sns.color_palette("muted")[0],
            alpha=0.3,
            linewidth=1.0,
        )

        # Overlay the Lesioned signal distinctly
        sns.lineplot(
            x=time_ms[:min_len_l],
            y=lesioned_signal[:min_len_l],
            ax=ax,
            label="Lesioned (No SCS)",
            color=sns.color_palette("muted")[3],
            alpha=0.9,
            linewidth=1.5,
        )

        ax.set_ylabel("EMG Amplitude (mV)")
        ax.set_title(
            f"Patient Trial (Seed {seed}) - Lesioned ({perc_intact*100:.1f}% Intact) vs. Healthy"
        )
        ax.set_xlim(0, 500)
        ax.legend()

    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()

    plot_path = output_dir / "lesioned_comparisons_overlay.png"
    plt.savefig(plot_path, dpi=300)
    print(f"[Lesioned] Comparative plot saved successfully to {plot_path}")

    return lesioned_emg_by_seed


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    reference_dir = output_dir / "reference"

    # Initialize the core config just like in run_cmaes.py
    simulation = SimulationConfig(backend="neuron")

    # Generate varied virtual patients using separate seeds
    patient_seeds = [1000 + i for i in range(args.num_patients)]

    # Always generate and plot the baseline healthy response
    healthy_emg = generate_and_plot_healthy(
        patient_seeds, simulation, reference_dir, output_dir
    )

    # If explicitly requested, also generate and plot the lesioned overlay
    if args.plot_lesioned:
        generate_and_plot_lesioned(patient_seeds, simulation, healthy_emg, output_dir)


if __name__ == "__main__":
    main()
