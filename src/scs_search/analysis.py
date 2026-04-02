"""Analysis helpers for sweeps and optimizer histories."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .config import EvaluationSummary, MetricConfig
from .metrics import compute_emg_similarity, mean_and_std_over_seeds
from .utils import read_json


def summary_to_record(summary: EvaluationSummary, extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Flatten an evaluation summary into a row-like record."""

    record = {
        "family": summary.family,
        "mean_corr": summary.mean_corr,
        "std_corr": summary.std_corr,
        "device_cost": summary.mean_device_cost,
        "std_device_cost": summary.std_device_cost,
        "total_current_ma": summary.mean_total_current_ma,
        "charge_per_pulse_uc": summary.mean_charge_per_pulse_uc,
        "charge_rate_uc_per_s": summary.mean_charge_rate_uc_per_s,
        "recruitment_raw_dose": summary.mean_raw_dose,
        "recruitment_norm_dose": summary.mean_norm_dose,
        "penalized_objective": summary.penalized_objective,
        "robust_objective": summary.robust_objective,
        **{f"theta_{key}": value for key, value in summary.theta.to_dict().items()},
    }
    if extra:
        record.update(dict(extra))
    return record


def build_best_under_limit_frontier(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the best observed correlation under each device-usage limit."""

    best_by_cost: dict[float, dict[str, Any]] = {}
    for record in records:
        dose = float(record["device_cost"])
        score = float(record["mean_corr"])
        existing = best_by_cost.get(dose)
        if existing is None or score > float(existing["mean_corr"]):
            best_by_cost[dose] = dict(record)

    ordered = [best_by_cost[dose] for dose in sorted(best_by_cost)]
    frontier: list[dict[str, Any]] = []
    best_so_far = float("-inf")
    best_record: dict[str, Any] | None = None
    for record in ordered:
        score = float(record["mean_corr"])
        if best_record is None or score > best_so_far:
            best_so_far = score
            best_record = dict(record)
        point = dict(best_record)
        point["device_cost"] = float(record["device_cost"])
        point["device_cost_limit"] = float(record["device_cost"])
        point["source_device_cost"] = float(best_record["device_cost"])
        point["mean_corr"] = float(best_so_far)
        frontier.append(point)
    return frontier


def build_upper_hull_frontier(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Compatibility wrapper for older call sites."""

    return build_best_under_limit_frontier(records)


def best_record(
    records: Iterable[Mapping[str, Any]],
    *,
    score_key: str = "mean_corr",
) -> dict[str, Any]:
    """Return the highest-scoring flat record from a sequence."""

    records_list = [dict(record) for record in records]
    if not records_list:
        raise ValueError("Cannot choose a best record from an empty sequence.")
    return max(records_list, key=lambda record: float(record[score_key]))


def best_so_far_trace(
    records: Iterable[Mapping[str, Any]],
    score_key: str = "mean_corr",
) -> list[dict[str, Any]]:
    """Compute best-so-far traces over an ordered record sequence."""

    trace: list[dict[str, Any]] = []
    best_any = float("-inf")
    for index, record in enumerate(records, start=1):
        score = float(record.get(score_key, float("-inf")))
        best_any = max(best_any, score)
        trace.append(
            {
                "evaluation_index": index,
                "seed_trials_used": float(record.get("seed_trials_used", index)),
                "best_so_far": best_any,
            }
        )
    return trace


def reference_baseline_stats(reference_dir: str | Path, metric_config: MetricConfig) -> dict[str, Any] | None:
    """Return lesion-without-stim correlation against healthy pre-lesion."""

    reference_path = Path(reference_dir)
    summary_file = reference_path / "summary.json"
    if summary_file.exists():
        summary = read_json(summary_file)
        baseline = summary.get("lesion_no_stim_baseline")
        if baseline is not None:
            return baseline

    emg_path = reference_path / "emg_arrays.npz"
    if not emg_path.exists():
        return None

    scores: list[float] = []
    seeds: list[int] = []
    with np.load(emg_path) as arrays:
        healthy_by_seed = {
            int(name.rsplit("_", 1)[-1]): np.asarray(arrays[name], dtype=float)
            for name in arrays.files
            if name.startswith("healthy_prelesion_seed_")
        }
        lesion_by_seed = {
            int(name.rsplit("_", 1)[-1]): np.asarray(arrays[name], dtype=float)
            for name in arrays.files
            if name.startswith("lesion_no_stim_seed_")
        }
    common_seeds = sorted(set(healthy_by_seed) & set(lesion_by_seed))
    for seed in common_seeds:
        scores.append(
            compute_emg_similarity(
                healthy_by_seed[seed],
                lesion_by_seed[seed],
                envelope_window_ms=metric_config.envelope_window_ms,
                max_lag_ms=metric_config.max_lag_ms,
                use_envelope=metric_config.use_envelope,
            )
        )
        seeds.append(seed)
    if not scores:
        return None

    mean_corr, std_corr = mean_and_std_over_seeds(scores)
    return {
        "label": "Lesion no stim",
        "mean_corr": mean_corr,
        "std_corr": std_corr,
        "seeds": seeds,
    }
