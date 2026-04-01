#!/usr/bin/env python3
"""Generate healthy and lesioned reference conditions."""

from __future__ import annotations

import argparse

import numpy as np

from scs_search.config import PatientConditionSpec, SimulationConfig, dataclass_config_bundle
from scs_search.metrics import compute_emg_similarity, mean_and_std_over_seeds
from scs_search.patterns import generate_tonic_pattern
from scs_search.simulator_adapter import run_condition
from scs_search.utils import ensure_dir, write_csv, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate healthy pre-lesion and lesion-without-stimulation references.")
    parser.add_argument("--output-dir", default="results/reference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(backend="neuron")
    output_dir = ensure_dir(args.output_dir)
    seeds = tuple(dict.fromkeys(config.seed_config.train_seeds + config.seed_config.report_seeds))

    healthy = PatientConditionSpec("healthy_prelesion", config.healthy_perc_supra_intact)
    lesion = PatientConditionSpec("lesion_no_stim", config.lesion_perc_supra_intact)

    zero_pattern = generate_tonic_pattern(
        freq_hz=40.0,
        alpha=0.0,
        t_end_ms=config.simulation_duration_ms,
        dt_ms=config.dt_ms,
        pulse_width_us=config.device_config.default_pulse_width_us,
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

    per_seed_rows = []
    for result_group in (healthy_results, lesion_results):
        for result in result_group:
            row = {
                "condition": result.condition_label,
                "seed": result.trial_seed,
                "backend": result.backend,
                "emg_mean": float(result.emg_signal.mean()),
                "emg_std": float(result.emg_signal.std()),
            }
            per_seed_rows.append(row)

    write_json(output_dir / "config.json", dataclass_config_bundle(config))
    write_jsonl(output_dir / "metrics.jsonl", per_seed_rows)
    write_csv(output_dir / "metrics.csv", per_seed_rows)
    write_json(
        output_dir / "summary.json",
        {
            "seeds": list(seeds),
            "train_seeds": list(config.seed_config.train_seeds),
            "report_seeds": list(config.seed_config.report_seeds),
            "conditions": ["healthy_prelesion", "lesion_no_stim"],
            "backend": config.backend,
            "seed_trials": len(seeds) * 2,
            "lesion_no_stim_baseline": {
                "label": "Lesion no stim",
                "mean_corr": baseline_mean,
                "std_corr": baseline_std,
                "seeds": list(seeds),
            },
        },
    )

    emg_arrays = {
        f"{result.condition_label}_seed_{result.trial_seed}": result.emg_signal
        for result in healthy_results + lesion_results
    }
    write_json(output_dir / "emg_index.json", {"arrays": list(emg_arrays.keys())})

    np.savez(output_dir / "emg_arrays.npz", **emg_arrays)


if __name__ == "__main__":
    main()
