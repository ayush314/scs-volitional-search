# scs-volitional-search

More than a million people in the United States live with paralysis due to spinal cord injury, and many never recover motor function to a satisfactory degree. Epidural spinal cord stimulation (SCS) is promising because injury can disrupt descending motor commands while leaving much of the spinal motor circuitry itself intact, creating the possibility that targeted stimulation could amplify residual commands and help re-engage natural spinal pathways.

This project studies that idea in a mechanistic NEURON simulation of a fine-motor task. We simulate a patient performing the task before lesion to get a healthy EMG target, then simulate the same task after lesion with SCS and search for stimulation patterns that make the lesioned EMG match the healthy one. The motivation for searching over patterned stimulation, rather than only standard tonic stimulation, is that prior work suggests the temporal structure of stimulation matters and may interact with the nonlinear dynamics of spinal circuits.

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[optim,dev]"
./scripts/setup_external.sh
./scripts/build_neuron.sh
python -c "from neuron import h; print('NEURON OK')"
```

## Experiment

We search over an 8-parameter stimulation pattern
`theta = (f, T_on, T_off, alpha0, alpha1, phi1, alpha2, phi2)`.
Here, `f` sets pulse frequency, `T_on` and `T_off` set the repeating on/off structure, and the remaining parameters define a clipped two-harmonic envelope
`alpha(t) = clip(alpha0 + alpha1 sin(2πt/T + phi1) + alpha2 sin(4πt/T + phi2), 0, 1)`,
with `T = T_on + T_off`. Inside the simulator, `alpha(t)` is the recruited-fiber fraction. In the hardware-cost layer, the same `alpha(t)` is also treated as a simple fraction of maximum program current.

We chose this pattern family as a compromise between flexibility and tractability. It can represent standard tonic stimulation, turn stimulation on and off in repeating blocks through `T_on` and `T_off`, and add modest within-cycle modulation through the two harmonics. That gives the study enough temporal variety to test non-tonic patterns without making the search space too large.

## Scoring

The y-axis is restoration correlation. For each seed, the healthy pre-lesion EMG trace is compared with the lesion + SCS EMG trace from the same fine-motor task. Both traces are rectified and smoothed with a `25 ms` moving-average window, then scored by Pearson correlation. Lesion without stimulation is also generated as a reference condition, but the main study is broader than just the unstimulated lesion comparison: the grid sweep maps the pattern space, and the optimizers are compared by how efficiently they find high-correlation patterns.

The x-axis is a hardware-aware cost metric,
`device_cost = sum_k q_k / (T_run * 100 mA * 1000 us * 1200 Hz)`,
with `q_k = (100 mA * alpha_k) * 210 us`.
In words, `device_cost` is delivered charge divided by the maximum charge the device could have delivered over the same run under Medtronic-style limits of `0-100 mA`, `60-1000 us`, and `10-1200 Hz`:
[`Medtronic Intellis 97715 manual mirror`](https://manuals.plus/m/bd8d5a123e572f58dbaa2dd8d7366ae8aee93c5247b73efb75873da0bd0a1ad6)
We fix pulse width at `210 us` for the current study; that choice is within the motor-control epidural SCS range summarized here:
[`motor-control epidural SCS review`](https://pmc.ncbi.nlm.nih.gov/articles/PMC10208259/)

## Runs

```bash
python scripts/run_prelesion_reference.py --output-dir results/reference
python scripts/run_grid_sweep.py --output-dir results/grid_sweep
python scripts/run_cmaes.py --seed-trial-budget 100 --output-dir results/cmaes
python scripts/run_turbo.py --seed-trial-budget 100 --output-dir results/turbo
python scripts/run_bohb.py --seed-trial-budget 100 --output-dir results/bohb
python scripts/summarize_results.py --results-root results
```

Run the scripts in that order:
- `run_prelesion_reference.py` builds the healthy target and lesion-no-stim reference condition.
- `run_grid_sweep.py` samples the stimulation space broadly and produces the sweep frontier.
- `run_cmaes.py`, `run_turbo.py`, and `run_bohb.py` apply the three adaptive search methods under matched seed-level compute.
- `summarize_results.py` combines finished runs into shared comparison plots.

## Defaults

- Simulation duration: `1000 ms`
- Lesion severity: `perc_supra_intact = 0.2`
- Sweep: `100` candidates total = `20` tonic + `20` duty-cycle + `60` full-theta LHS
- Sweep evaluation policy: `1` fixed train seed per candidate for coverage
- Optimizer training seeds: up to `3` per candidate
- Optimizer budget: `100` seed-level trials by default
- Final reporting seeds: `3`

## Outputs

- `results/reference/`
  - `config.json`, `metrics.jsonl`, `metrics.csv`: reference-run settings and per-condition rows
  - `summary.json`: lesion-no-stim correlation summary
  - `emg_arrays.npz`, `emg_index.json`: healthy and lesion EMG traces by seed
  - `reference_emg.png`: healthy pre-lesion versus lesion no stim
- `results/grid_sweep/`
  - `config.json`: sweep settings
  - `metrics.jsonl`, `metrics.csv`: one row per sampled candidate
  - `summary.json`: candidate count and seed-level trial count
  - `frontier.json`: sweep hull
  - `frontier.png`: sampled candidates and sweep frontier
- `results/cmaes/`, `results/turbo/`, `results/bohb/`
  - `config.json`: optimizer settings
  - `history.json`, `metrics.jsonl`, `metrics.csv`: candidate evaluations during search
  - `trace.json`: best-so-far progress
  - `summary.json`: final incumbent summary
  - `frontier.json`: that method's hull
  - `best_so_far.png`: search progress versus seed-level trials
  - `device_budget_vs_corr.png`: that method's visited device-cost/correlation region
  - `device_budget_vs_corr_with_grid.png`: overlay against the grid sweep when sweep outputs are present
- `results/`
  - `reference_emg.png`: healthy pre-lesion versus lesion + best stimulation found so far
  - `frontier.png`: top-level copy of the sweep frontier
  - `optimizer_comparison.png`: all optimizer best-so-far traces on one plot
  - `seed_sensitivity.png`: final incumbent mean ± std across report seeds for each optimizer

## Tests

```bash
pytest
```
