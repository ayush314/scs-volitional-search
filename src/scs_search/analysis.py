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
        "raw_dose": summary.mean_raw_dose,
        "norm_dose": summary.mean_norm_dose,
        "penalized_objective": summary.penalized_objective,
        "robust_objective": summary.robust_objective,
        **{f"theta_{key}": value for key, value in summary.theta.to_dict().items()},
    }
    if extra:
        record.update(dict(extra))
    return record


def build_upper_hull_frontier(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the upper hull in dose-correlation space."""

    best_by_dose: dict[float, dict[str, Any]] = {}
    for record in records:
        dose = float(record["norm_dose"])
        score = float(record["mean_corr"])
        existing = best_by_dose.get(dose)
        if existing is None or score > float(existing["mean_corr"]):
            best_by_dose[dose] = dict(record)

    ordered = [best_by_dose[dose] for dose in sorted(best_by_dose)]
    if len(ordered) <= 2:
        return ordered

    def cross(o: Mapping[str, Any], a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
        return (
            (float(a["norm_dose"]) - float(o["norm_dose"])) * (float(b["mean_corr"]) - float(o["mean_corr"]))
            - (float(a["mean_corr"]) - float(o["mean_corr"])) * (float(b["norm_dose"]) - float(o["norm_dose"]))
        )

    hull: list[dict[str, Any]] = []
    for record in ordered:
        point = dict(record)
        while len(hull) >= 2 and cross(hull[-2], hull[-1], point) >= 0.0:
            hull.pop()
        hull.append(point)
    return hull


def best_so_far_trace(
    records: Iterable[Mapping[str, Any]],
    score_key: str = "mean_corr",
    budget_norm: float | None = None,
) -> list[dict[str, Any]]:
    """Compute best-so-far traces over an ordered record sequence."""

    trace: list[dict[str, Any]] = []
    best_any = float("-inf")
    best_feasible = float("-inf")
    for index, record in enumerate(records, start=1):
        score = float(record.get(score_key, float("-inf")))
        best_any = max(best_any, score)
        if budget_norm is None or float(record.get("norm_dose", float("inf"))) <= float(budget_norm):
            best_feasible = max(best_feasible, score)
        trace.append(
            {
                "evaluation_index": index,
                "seed_trials_used": float(record.get("seed_trials_used", index)),
                "best_so_far": best_any,
                "best_feasible_so_far": best_feasible if best_feasible > float("-inf") else None,
            }
        )
    return trace


def load_run_bundle(run_dir: str | Path) -> dict[str, Any]:
    """Load the common summary files from a run directory."""

    path = Path(run_dir)
    bundle: dict[str, Any] = {}
    for filename in ("config.json", "summary.json", "frontier.json", "history.json", "metrics.json"):
        file_path = path / filename
        if file_path.exists():
            bundle[filename] = read_json(file_path)
    return bundle


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
