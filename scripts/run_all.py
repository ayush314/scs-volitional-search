#!/usr/bin/env python3
"""Run the full experiment pipeline sequentially."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scs_search.config import DEFAULT_SWEEP_SEED_TRIALS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full reference, sweep, optimizer, and summary pipeline.")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--seed-trial-budget", type=int, default=DEFAULT_SWEEP_SEED_TRIALS)
    return parser.parse_args()


def run_step(args: list[str], repo_root: Path) -> None:
    command = [sys.executable, *args]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) if not existing_pythonpath else f"{SRC_ROOT}:{existing_pythonpath}"
    print(f"\n==> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def main() -> None:
    args = parse_args()
    repo_root = REPO_ROOT
    results_root = Path(args.results_root)

    run_step(["scripts/run_prelesion_reference.py", "--output-dir", str(results_root / "reference")], repo_root)
    run_step(
        [
            "scripts/run_grid_sweep.py",
            "--seed-trial-budget",
            str(args.seed_trial_budget),
            "--output-dir",
            str(results_root / "grid_sweep"),
        ],
        repo_root,
    )
    run_step(
        [
            "scripts/run_cmaes.py",
            "--seed-trial-budget",
            str(args.seed_trial_budget),
            "--output-dir",
            str(results_root / "cmaes"),
        ],
        repo_root,
    )
    run_step(
        [
            "scripts/run_turbo.py",
            "--seed-trial-budget",
            str(args.seed_trial_budget),
            "--output-dir",
            str(results_root / "turbo"),
        ],
        repo_root,
    )
    run_step(
        [
            "scripts/run_bohb.py",
            "--seed-trial-budget",
            str(args.seed_trial_budget),
            "--output-dir",
            str(results_root / "bohb"),
        ],
        repo_root,
    )
    run_step(["scripts/summarize_results.py", "--results-root", str(results_root)], repo_root)


if __name__ == "__main__":
    main()
