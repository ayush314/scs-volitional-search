"""Candidate evaluation, reference caching, and report-figure helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..config import (
    EvaluationSummary,
    PatientConditionSpec,
    SimulationConfig,
    coerce_seed_sequence,
    condition_defaults,
    theta_to_dict,
)
from ..dose import combined_objective, compute_pattern_dose
from ..metrics import compute_emg_similarity, mean_and_std_over_seeds, relative_envelope_rmse
from ..reporting.plotting import (
    display_name,
    lesion_label,
    plot_emg_seed_panels,
    plot_pattern_detail,
    plot_supraspinal_drive_examples,
)
from ..stimulation.patterns import generate_stim_pattern, generate_tonic_pattern, invalid_theta_reason
from ..utils import ensure_dir, write_json
from .backend import run_condition, run_single_condition
from .drive import build_supraspinal_drive
from .structural import get_structural_state
from .transduction import _pattern_to_afferent_fibers

_HEALTHY_REFERENCE_PREFIX = "healthy_prelesion_seed_"


def _invalid_evaluation_summary(
    theta: Any,
    seeds: tuple[int, ...],
    config: SimulationConfig,
    invalid_reason: str,
    budget_norm: float | None,
) -> EvaluationSummary:
    """Return a non-simulated summary for an infeasible stimulation pattern."""

    floor = float(config.dose_config.invalid_objective_floor)
    theta_values = theta_to_dict(theta)
    pulse_width_us = float(theta_values.get("pw_us", theta_values.get("PW1_us", config.device_config.default_pulse_width_us)))
    per_seed_records = [
        {
            "seed": int(seed),
            "corr": floor,
            "relative_envelope_rmse": 1.0,
            "raw_recruitment_dose": 0.0,
            "recruitment_dose_norm": 0.0,
            "device_cost": 1.0,
            "current_rate_usage": 1.0,
            "total_current_ma": 0.0,
            "charge_per_pulse_uc": 0.0,
            "charge_rate_uc_per_s": 0.0,
            "pulse_width_us": pulse_width_us,
            "backend": config.backend,
            "condition_label": "lesion_scs_invalid",
            "theta": theta_values,
            "valid": False,
            "invalid_reason": invalid_reason,
        }
        for seed in seeds
    ]
    return EvaluationSummary(
        theta=theta,
        family="physical_modulation",
        seeds=seeds,
        per_seed_records=per_seed_records,
        mean_corr=floor,
        std_corr=0.0,
        mean_raw_dose=0.0,
        std_raw_dose=0.0,
        mean_norm_dose=0.0,
        std_norm_dose=0.0,
        mean_device_cost=1.0,
        std_device_cost=0.0,
        mean_current_rate_usage=1.0,
        std_current_rate_usage=0.0,
        mean_total_current_ma=0.0,
        std_total_current_ma=0.0,
        mean_charge_per_pulse_uc=0.0,
        std_charge_per_pulse_uc=0.0,
        mean_charge_rate_uc_per_s=0.0,
        std_charge_rate_uc_per_s=0.0,
        mean_relative_envelope_rmse=1.0,
        std_relative_envelope_rmse=0.0,
        penalized_objective=floor,
        robust_objective=floor,
        valid=False,
        invalid_reason=invalid_reason,
        metadata={
            "budget_norm": budget_norm,
            "backend": config.backend,
            "pulse_width_us": pulse_width_us,
            "transduction_mode": str(config.transduction_config.mode),
            "usage_metric": "normalized_charge_rate_usage",
            "dose_definitions": {
                "r_k": "fraction of afferents recruited by pulse k",
                "D_raw": "sum_k r_k",
                "D_norm": "D_raw / (T_run_s * f_max)",
                "device_cost": "sum_k(I_k * PW_k) / (T_run_s * I_max * PW_max * f_max)",
                "relative_envelope_rmse": "||e_ref - e_cand||_2 / ||e_ref||_2 on rectified, smoothed EMG envelopes",
            },
            "valid": False,
            "invalid_reason": invalid_reason,
        },
    )


def evaluate_pattern(
    theta: Any,
    seeds: Iterable[int],
    config: SimulationConfig,
    budget_norm: float | None = None,
    *,
    reference_emg_by_seed: dict[int, np.ndarray] | None = None,
    robust_objective: bool = False,
) -> EvaluationSummary:
    """Evaluate lesion+SCS restoration against the healthy pre-lesion reference."""

    healthy_condition, lesion_condition = condition_defaults(config)
    theta_params = config.theta_bounds.clip(theta, device_config=config.device_config)
    seeds_tuple = coerce_seed_sequence(seeds, config.seed_config.train_seeds)
    invalid_reason = invalid_theta_reason(
        theta_params,
        config.device_config,
        enforce_no_overlap=config.transduction_config.enforce_no_overlap,
    )
    if invalid_reason is not None:
        return _invalid_evaluation_summary(theta_params, seeds_tuple, config, invalid_reason, budget_norm)

    stim_pattern = generate_stim_pattern(
        theta_params,
        t_end_ms=config.simulation_duration_ms,
        dt_ms=config.dt_ms,
        device_config=config.device_config,
        pulse_scheduler_dt_ms=config.pulse_scheduler_dt_ms,
    )
    realized_invalid_reason = stim_pattern.metadata.get("realized_schedule_invalid_reason")
    if realized_invalid_reason is not None:
        return _invalid_evaluation_summary(theta_params, seeds_tuple, config, str(realized_invalid_reason), budget_norm)

    zero_pattern = generate_tonic_pattern(
        freq_hz=max(float(theta_to_dict(theta_params).get("f0_hz", 10.0)), 1.0),
        current_ma=0.0,
        t_end_ms=config.simulation_duration_ms,
        dt_ms=config.dt_ms,
        device_config=config.device_config,
    )

    structural_state = get_structural_state(config)
    transduction = _pattern_to_afferent_fibers(stim_pattern, config, structural_state)

    if reference_emg_by_seed is None:
        healthy_results = [
            run_single_condition(healthy_condition, zero_pattern, int(seed), config, structural_state)
            for seed in seeds_tuple
        ]
        reference_emg_by_seed = {result.trial_seed: result.emg_signal for result in healthy_results}

    lesion_results = [
        run_single_condition(
            PatientConditionSpec(label="lesion_scs", perc_supra_intact=lesion_condition.perc_supra_intact),
            stim_pattern,
            int(seed),
            config,
            structural_state,
            transduction=transduction,
        )
        for seed in seeds_tuple
    ]

    dose_metrics = compute_pattern_dose(
        stim_pattern,
        pulse_recruitment_fraction=transduction.pulse_recruitment_fraction,
        dose_config=config.dose_config,
        device_config=config.device_config,
    )

    metric_values: list[float] = []
    raw_dose_values: list[float] = []
    norm_dose_values: list[float] = []
    device_cost_values: list[float] = []
    current_rate_values: list[float] = []
    total_current_values: list[float] = []
    charge_per_pulse_values: list[float] = []
    charge_rate_values: list[float] = []
    rrmse_values: list[float] = []
    per_seed_records: list[dict[str, Any]] = []

    for lesion_result in lesion_results:
        seed = int(lesion_result.trial_seed)
        reference_emg = np.asarray(reference_emg_by_seed[seed], dtype=float)
        corr = compute_emg_similarity(
            reference_emg=reference_emg,
            candidate_emg=lesion_result.emg_signal,
            envelope_window_ms=config.metric_config.envelope_window_ms,
            max_lag_ms=config.metric_config.max_lag_ms,
            use_envelope=config.metric_config.use_envelope,
        )
        rrmse = relative_envelope_rmse(
            reference_emg=reference_emg,
            candidate_emg=lesion_result.emg_signal,
            envelope_window_ms=config.metric_config.envelope_window_ms,
            max_lag_ms=config.metric_config.max_lag_ms,
            use_envelope=config.metric_config.use_envelope,
        )
        metric_values.append(corr)
        rrmse_values.append(rrmse)
        raw_dose_values.append(float(dose_metrics["raw_recruitment_dose"]))
        norm_dose_values.append(float(dose_metrics["recruitment_dose_norm"]))
        device_cost_values.append(float(dose_metrics["device_cost"]))
        current_rate_values.append(float(dose_metrics["current_rate_usage"]))
        total_current_values.append(float(dose_metrics["mean_total_current_ma"]))
        charge_per_pulse_values.append(float(dose_metrics["mean_charge_per_pulse_uc"]))
        charge_rate_values.append(float(dose_metrics["charge_rate_uc_per_s"]))
        per_seed_records.append(
            {
                "seed": seed,
                "corr": float(corr),
                "relative_envelope_rmse": float(rrmse),
                "raw_recruitment_dose": float(dose_metrics["raw_recruitment_dose"]),
                "recruitment_dose_norm": float(dose_metrics["recruitment_dose_norm"]),
                "device_cost": float(dose_metrics["device_cost"]),
                "current_rate_usage": float(dose_metrics["current_rate_usage"]),
                "total_current_ma": float(dose_metrics["mean_total_current_ma"]),
                "charge_per_pulse_uc": float(dose_metrics["mean_charge_per_pulse_uc"]),
                "charge_rate_uc_per_s": float(dose_metrics["charge_rate_uc_per_s"]),
                "pulse_width_us": float(dose_metrics["pulse_width_us"]),
                "backend": lesion_result.backend,
                "condition_label": lesion_result.condition_label,
                "theta": theta_to_dict(theta_params),
                "supraspinal_drive_mode": lesion_result.metadata.get("supraspinal_drive_mode"),
                "supraspinal_rate_floor_hz": lesion_result.metadata.get("supraspinal_rate_floor_hz"),
                "valid": True,
                "invalid_reason": None,
            }
        )

    mean_corr, std_corr = mean_and_std_over_seeds(metric_values)
    mean_raw_dose, std_raw_dose = mean_and_std_over_seeds(raw_dose_values)
    mean_norm_dose, std_norm_dose = mean_and_std_over_seeds(norm_dose_values)
    mean_device_cost, std_device_cost = mean_and_std_over_seeds(device_cost_values)
    mean_current_rate_usage, std_current_rate_usage = mean_and_std_over_seeds(current_rate_values)
    mean_total_current_ma, std_total_current_ma = mean_and_std_over_seeds(total_current_values)
    mean_charge_per_pulse_uc, std_charge_per_pulse_uc = mean_and_std_over_seeds(charge_per_pulse_values)
    mean_charge_rate_uc_per_s, std_charge_rate_uc_per_s = mean_and_std_over_seeds(charge_rate_values)
    mean_relative_envelope_rmse, std_relative_envelope_rmse = mean_and_std_over_seeds(rrmse_values)
    robust_score, penalized_score = combined_objective(
        mean_corr=mean_corr,
        std_corr=std_corr,
        device_cost=mean_device_cost,
        budget_norm=budget_norm,
        dose_config=config.dose_config,
        robust=robust_objective,
        theta=theta_params,
        pulse_recruitment_fraction=transduction.pulse_recruitment_fraction,
    )
    return EvaluationSummary(
        theta=theta_params,
        family=stim_pattern.family,
        seeds=seeds_tuple,
        per_seed_records=per_seed_records,
        mean_corr=mean_corr,
        std_corr=std_corr,
        mean_raw_dose=mean_raw_dose,
        std_raw_dose=std_raw_dose,
        mean_norm_dose=mean_norm_dose,
        std_norm_dose=std_norm_dose,
        mean_device_cost=mean_device_cost,
        std_device_cost=std_device_cost,
        mean_current_rate_usage=mean_current_rate_usage,
        std_current_rate_usage=std_current_rate_usage,
        mean_total_current_ma=mean_total_current_ma,
        std_total_current_ma=std_total_current_ma,
        mean_charge_per_pulse_uc=mean_charge_per_pulse_uc,
        std_charge_per_pulse_uc=std_charge_per_pulse_uc,
        mean_charge_rate_uc_per_s=mean_charge_rate_uc_per_s,
        std_charge_rate_uc_per_s=std_charge_rate_uc_per_s,
        mean_relative_envelope_rmse=mean_relative_envelope_rmse,
        std_relative_envelope_rmse=std_relative_envelope_rmse,
        penalized_objective=penalized_score,
        robust_objective=robust_score,
        valid=True,
        invalid_reason=None,
        metadata={
            "budget_norm": budget_norm,
            "backend": config.backend,
            "pulse_width_us": float(dose_metrics["pulse_width_us"]),
            "current_cap_ma": float(config.device_config.max_total_current_ma),
            "transduction_mode": str(config.transduction_config.mode),
            "supraspinal_drive_mode": str(config.supraspinal_drive_mode),
            "supraspinal_rate_floor_hz": float(config.supraspinal_rate_floor_hz),
            "supraspinal_envelope_control_dt_ms": float(config.supraspinal_envelope_control_dt_ms),
            "supraspinal_envelope_smoothing_sigma_ms": float(config.supraspinal_envelope_smoothing_sigma_ms),
            "supraspinal_envelope_ar_rho": float(config.supraspinal_envelope_ar_rho),
            "supraspinal_task_burst_min_ms": float(config.supraspinal_task_burst_min_ms),
            "supraspinal_task_burst_max_ms": float(config.supraspinal_task_burst_max_ms),
            "supraspinal_task_gap_min_ms": float(config.supraspinal_task_gap_min_ms),
            "supraspinal_task_gap_max_ms": float(config.supraspinal_task_gap_max_ms),
            "usage_metric": "normalized_charge_rate_usage",
            "dose_definitions": {
                "r_k": "fraction of afferents recruited by pulse k",
                "D_raw": "sum_k r_k",
                "D_norm": "D_raw / (T_run_s * f_max)",
                "device_cost": "sum_k(I_k * PW_k) / (T_run_s * I_max * PW_max * f_max)",
                "relative_envelope_rmse": "||e_ref - e_cand||_2 / ||e_ref||_2 on rectified, smoothed EMG envelopes",
            },
            "reference_matching": "each lesion or lesion+SCS seed is compared against the healthy pre-lesion trace generated with the same seed",
            "valid": True,
        },
    )


def build_reference_emg_cache(seeds: Iterable[int], config: SimulationConfig) -> dict[int, np.ndarray]:
    """Generate healthy pre-lesion reference EMG traces keyed by seed."""

    healthy_condition, _ = condition_defaults(config)
    zero_pattern = generate_tonic_pattern(
        freq_hz=40.0,
        current_ma=0.0,
        t_end_ms=config.simulation_duration_ms,
        dt_ms=config.dt_ms,
        device_config=config.device_config,
    )
    return {
        result.trial_seed: result.emg_signal
        for result in run_condition(
            healthy_condition,
            zero_pattern,
            seeds,
            config,
            progress_desc="Reference EMG",
        )
    }


def load_reference_emg_cache(reference_dir: str | Path, seeds: Iterable[int]) -> dict[int, np.ndarray]:
    """Load cached healthy pre-lesion EMG traces for the requested seeds."""

    seeds_tuple = tuple(int(seed) for seed in seeds)
    cache_path = Path(reference_dir) / "emg_arrays.npz"
    if not cache_path.exists():
        return {}
    loaded: dict[int, np.ndarray] = {}
    with np.load(cache_path) as arrays:
        for seed in seeds_tuple:
            key = f"{_HEALTHY_REFERENCE_PREFIX}{seed}"
            if key in arrays:
                loaded[seed] = np.asarray(arrays[key], dtype=float)
    return loaded


def _persist_reference_emg_cache(reference_dir: str | Path, cache_by_seed: dict[int, np.ndarray]) -> None:
    """Merge healthy reference traces into the on-disk reference cache."""

    reference_path = ensure_dir(reference_dir)
    cache_path = reference_path / "emg_arrays.npz"
    arrays: dict[str, np.ndarray] = {}
    if cache_path.exists():
        with np.load(cache_path) as existing:
            arrays = {name: np.asarray(existing[name], dtype=float) for name in existing.files}
    for seed, emg_signal in cache_by_seed.items():
        arrays[f"{_HEALTHY_REFERENCE_PREFIX}{int(seed)}"] = np.asarray(emg_signal, dtype=float)
    np.savez(cache_path, **arrays)


def resolve_reference_emg_cache(
    seeds: Iterable[int],
    config: SimulationConfig,
    reference_dir: str | Path | None = None,
) -> dict[int, np.ndarray]:
    """Load saved healthy references first, then build only any missing seeds."""

    seeds_tuple = coerce_seed_sequence(seeds, config.seed_config.train_seeds)
    reference_cache = load_reference_emg_cache(reference_dir, seeds_tuple) if reference_dir is not None else {}
    missing_seeds = tuple(seed for seed in seeds_tuple if seed not in reference_cache)
    if missing_seeds:
        built_cache = build_reference_emg_cache(missing_seeds, config)
        reference_cache.update(built_cache)
        if reference_dir is not None:
            _persist_reference_emg_cache(reference_dir, built_cache)
    return {seed: np.asarray(reference_cache[seed], dtype=float) for seed in seeds_tuple}


def write_best_emg_panel(
    *,
    method_key: str,
    theta: Any,
    output_dir: str | Path,
    config: SimulationConfig,
    reference_dir: str | Path,
) -> None:
    """Evaluate one best candidate on train seeds and write its EMG comparison panel."""

    output_path = ensure_dir(output_dir)
    train_seeds = tuple(int(seed) for seed in config.seed_config.train_seeds)
    reference_cache = resolve_reference_emg_cache(train_seeds, config, reference_dir=reference_dir)
    stim_pattern = generate_stim_pattern(
        theta,
        t_end_ms=config.simulation_duration_ms,
        dt_ms=config.dt_ms,
        device_config=config.device_config,
        pulse_scheduler_dt_ms=config.pulse_scheduler_dt_ms,
    )
    structural_state = get_structural_state(config)
    transduction = _pattern_to_afferent_fibers(stim_pattern, config, structural_state)
    lesion_condition = PatientConditionSpec(
        label=f"lesion_{method_key}",
        perc_supra_intact=config.lesion_perc_supra_intact,
    )
    lesion_results = run_condition(
        lesion_condition,
        stim_pattern,
        train_seeds,
        config,
        structural_state=structural_state,
        transduction=transduction,
    )
    lesion_by_seed = {int(result.trial_seed): result.emg_signal for result in lesion_results}
    train_corrs = [
        compute_emg_similarity(
            reference_emg=reference_cache[int(seed)],
            candidate_emg=lesion_by_seed[int(seed)],
            envelope_window_ms=config.metric_config.envelope_window_ms,
            max_lag_ms=config.metric_config.max_lag_ms,
            use_envelope=config.metric_config.use_envelope,
        )
        for seed in train_seeds
    ]
    mean_corr, _ = mean_and_std_over_seeds(train_corrs)
    comparison_label = lesion_label(method_key)
    comparison_label = comparison_label[0].lower() + comparison_label[1:]
    plot_emg_seed_panels(
        {int(seed): reference_cache[int(seed)] for seed in train_seeds},
        lesion_by_seed,
        output_path / "best_emg.png",
        f"Healthy pre-lesion vs {comparison_label} | corr={mean_corr:.3f}",
        reference_label="Healthy pre-lesion",
        candidate_label=lesion_label(method_key),
    )
    plot_pattern_detail(stim_pattern, output_path / "stim_pattern.png")
    plot_supraspinal_drive_examples(
        {int(seed): build_supraspinal_drive(config, int(seed)) for seed in train_seeds},
        output_path / "supraspinal_drive.png",
        title="Train-seed supraspinal drive examples",
    )


def evaluate_best_candidate_report_summary(
    *,
    theta: Any,
    output_dir: str | Path,
    config: SimulationConfig,
    reference_dir: str | Path,
) -> EvaluationSummary:
    """Reevaluate one candidate on report seeds and write the final report summary."""

    output_path = ensure_dir(output_dir)
    report_seeds = tuple(int(seed) for seed in config.seed_config.report_seeds)
    reference_cache = resolve_reference_emg_cache(report_seeds, config, reference_dir=reference_dir)
    summary = evaluate_pattern(
        theta=theta,
        seeds=report_seeds,
        config=config,
        reference_emg_by_seed=reference_cache,
    )
    write_json(output_path / "final_report_summary.json", summary)
    return summary
