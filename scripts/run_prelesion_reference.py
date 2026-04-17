#!/usr/bin/env python3
"""Generate healthy and lesioned reference conditions."""

from __future__ import annotations

import argparse

import numpy as np

from scs_search.config import PatientConditionSpec, dataclass_config_bundle
from scs_search.metrics import compute_emg_similarity, mean_and_std_over_seeds
from scs_search.reporting.plotting import plot_emg_seed_panels, plot_supraspinal_drive_examples
from scs_search.search.sweep import make_physical_modulation_simulation_config
from scs_search.simulation.backend import run_condition
from scs_search.simulation.drive import build_supraspinal_drive
from scs_search.stimulation.patterns import generate_tonic_pattern
from scs_search.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate healthy pre-lesion and lesion-without-stimulation references.")
    parser.add_argument("--output-dir", default="results/reference")
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
    seeds = tuple(dict.fromkeys(config.seed_config.train_seeds + config.seed_config.report_seeds))

    healthy = PatientConditionSpec("healthy_prelesion", config.healthy_perc_supra_intact)
    lesion = PatientConditionSpec("lesion_no_stim", config.lesion_perc_supra_intact)

    zero_pattern = generate_tonic_pattern(
        freq_hz=40.0,
        current_ma=0.0,
        t_end_ms=config.simulation_duration_ms,
        dt_ms=config.dt_ms,
        device_config=config.device_config,
    )

    healthy_results = run_condition(healthy, zero_pattern, seeds, config, progress_desc="Reference healthy")
    lesion_results = run_condition(lesion, zero_pattern, seeds, config, progress_desc="Reference lesion")

    healthy_by_seed = {result.trial_seed: result.emg_signal for result in healthy_results}
    lesion_by_seed = {result.trial_seed: result.emg_signal for result in lesion_results}
    baseline_scores = [
        compute_emg_similarity(
            reference_emg=healthy_by_seed[seed],
            candidate_emg=lesion_by_seed[seed],
            envelope_window_ms=config.metric_config.envelope_window_ms,
            max_lag_ms=config.metric_config.max_lag_ms,
            use_envelope=config.metric_config.use_envelope,
        )
        for seed in seeds
    ]
    baseline_mean, baseline_std = mean_and_std_over_seeds(baseline_scores)
    train_seeds = tuple(int(seed) for seed in config.seed_config.train_seeds)
    report_seeds = tuple(int(seed) for seed in config.seed_config.report_seeds)
    train_baseline_scores = [
        compute_emg_similarity(
            reference_emg=healthy_by_seed[seed],
            candidate_emg=lesion_by_seed[seed],
            envelope_window_ms=config.metric_config.envelope_window_ms,
            max_lag_ms=config.metric_config.max_lag_ms,
            use_envelope=config.metric_config.use_envelope,
        )
        for seed in train_seeds
    ]
    train_baseline_mean, train_baseline_std = mean_and_std_over_seeds(train_baseline_scores)
    report_baseline_scores = [
        compute_emg_similarity(
            reference_emg=healthy_by_seed[seed],
            candidate_emg=lesion_by_seed[seed],
            envelope_window_ms=config.metric_config.envelope_window_ms,
            max_lag_ms=config.metric_config.max_lag_ms,
            use_envelope=config.metric_config.use_envelope,
        )
        for seed in report_seeds
    ]
    report_baseline_mean, report_baseline_std = mean_and_std_over_seeds(report_baseline_scores)

    baseline_all = {
        "label": "Lesion no stim",
        "mean_corr": baseline_mean,
        "std_corr": baseline_std,
        "seeds": list(seeds),
    }
    baseline_train = {
        "label": "Lesion no stim",
        "mean_corr": train_baseline_mean,
        "std_corr": train_baseline_std,
        "seeds": list(train_seeds),
    }
    baseline_report = {
        "label": "Lesion no stim",
        "mean_corr": report_baseline_mean,
        "std_corr": report_baseline_std,
        "seeds": list(report_seeds),
    }

    write_json(
        output_dir / "summary.json",
        {
            "config": dataclass_config_bundle(config),
            "seeds": list(seeds),
            "train_seeds": list(config.seed_config.train_seeds),
            "report_seeds": list(config.seed_config.report_seeds),
            "conditions": ["healthy_prelesion", "lesion_no_stim"],
            "backend": config.backend,
            "seed_trials": len(seeds) * 2,
            "lesion_no_stim_baseline": baseline_all,
            "lesion_no_stim_baseline_all": baseline_all,
            "lesion_no_stim_baseline_train": baseline_train,
            "lesion_no_stim_baseline_report": baseline_report,
        },
    )

    emg_arrays = {
        f"{result.condition_label}_seed_{result.trial_seed}": result.emg_signal
        for result in healthy_results + lesion_results
    }
    np.savez(output_dir / "emg_arrays.npz", **emg_arrays)
    plot_emg_seed_panels(
        {seed: healthy_by_seed[seed] for seed in train_seeds},
        {seed: lesion_by_seed[seed] for seed in train_seeds},
        output_dir / "reference_emg.png",
        f"Healthy pre-lesion vs lesion no stim | corr={train_baseline_mean:.3f}",
        reference_label="Healthy pre-lesion",
        candidate_label="Lesion no stim",
    )
    plot_supraspinal_drive_examples(
        {int(seed): build_supraspinal_drive(config, int(seed)) for seed in train_seeds},
        output_dir / "supraspinal_drive.png",
        title="Train-seed supraspinal drive examples",
    )


if __name__ == "__main__":
    main()
