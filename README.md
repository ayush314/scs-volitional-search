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
with `T = T_on + T_off`. Inside the simulator, `alpha(t)` is the recruited-fiber fraction. In the hardware-usage metric, the same `alpha(t)` is also treated as a simple fraction of maximum program current.

We chose this pattern family as a compromise between flexibility and tractability. It can represent standard tonic stimulation, turn stimulation on and off in repeating blocks through `T_on` and `T_off`, and add modest within-cycle modulation through the two harmonics. That allows for enough temporal variety to test non-tonic patterns without making the search space too large.

## Scoring

The y-axis is pre-lesion EMG correlation. For each seed, the healthy pre-lesion EMG trace is compared with the lesion + SCS EMG trace from the same fine-motor task. Both traces are rectified and smoothed with a `25 ms` moving-average window, then scored by Pearson correlation. Lesion without stimulation is also generated as a reference condition, but the main experiment is broader than just the unstimulated lesion comparison: the grid sweep maps the pattern space, and the optimizers are compared by how efficiently they find high-correlation patterns.

The x-axis is a normalized hardware-usage metric. The device manual gives a broad operating regime, including total program current up to `100 mA`, pulse width from `60-1000 us`, and frequency from `10-1200 Hz`:
[`Medtronic Intellis 97715 manual mirror`](https://manuals.plus/m/bd8d5a123e572f58dbaa2dd8d7366ae8aee93c5247b73efb75873da0bd0a1ad6).
Within that, this project uses a narrower setup: frequency is searched over `10-1200 Hz`, pulse width is fixed at `210 us`, and `alpha_k` is mapped to a conservative `50 mA` cap through `I_k = 50 mA * alpha_k`. Since larger `alpha_k` makes each pulse stronger and higher frequency produces more pulses over the run, both should increase hardware usage, so we sum the currents per pulse across the run and normalize by the amount that would be delivered at `50 mA` and `1200 Hz` over the same duration:
`device_usage = sum_k I_k / (T_run * 50 mA * 1200 Hz)`.

## Runs

Run the full pipeline with one command:

```bash
python scripts/run_all.py --results-root results --seed-trial-budget 700
```

Or run the stages individually:

```bash
python scripts/run_prelesion_reference.py --output-dir results/reference
python scripts/run_grid_sweep.py --seed-trial-budget 700 --output-dir results/grid_sweep
python scripts/run_cmaes.py --seed-trial-budget 700 --output-dir results/cmaes
python scripts/run_turbo.py --seed-trial-budget 700 --output-dir results/turbo
python scripts/run_bohb.py --seed-trial-budget 700 --output-dir results/bohb
python scripts/summarize_results.py --results-root results
```

If you run the stages manually, use them in that order:
- `run_prelesion_reference.py` builds the healthy target and lesion-no-stim reference condition.
- `run_grid_sweep.py` samples the stimulation space broadly and produces the sweep frontier.
- `run_cmaes.py`, `run_turbo.py`, and `run_bohb.py` apply the three adaptive search methods under matched seed-level compute.
- `summarize_results.py` reevaluates the best pattern from each folder on the report seeds and builds the shared comparison plots.

## Defaults

- Simulation duration: `1000 ms`
- Lesion severity: `perc_supra_intact = 0.2`
- Sweep budget: `700` seed-level trials by default with exactly `3` seeds per pattern
- Sweep allocation: `~20%` tonic patterns, `~20%` duty-cycle patterns, and `~60%` Latin hypercube samples over the full 8-parameter space
- Optimizer budget: `700` seed-level trials by default with up to `3` seeds per pattern
- Final reporting seeds: `10`

## Outputs

- `results/reference/`
  - `summary.json`: reference settings, seed lists, and the lesion-no-stim baseline correlation
  - `emg_arrays.npz`: saved healthy pre-lesion and lesion-no-stim EMG traces reused by later scripts
  - `reference_emg.png`: three-panel train-seed comparison showing what the lesion does before any stimulation
- `results/grid_sweep/`
  - `patterns.jsonl`: one row per evaluated sweep pattern with its parameters, device usage, and correlation
  - `summary.json`: sweep settings, budget allocation, and the best pattern found in the sweep
  - `frontier.png`: the sweep frontier, showing the best observed correlation available up to each device-usage level
  - `best_emg.png`: three-panel train-seed EMG comparison for the best grid-sweep pattern, so you can see the waveform match directly
  - `final_report_summary.json`: 10-seed reevaluation of that pattern
- `results/cmaes/`, `results/turbo/`, `results/bohb/`
  - `history.jsonl`: one row per evaluated pattern during search
  - `summary.json`: run settings, search metadata, and the best pattern found during search
  - `best_emg.png`: three-panel train-seed EMG comparison for that best pattern, so you can inspect the waveform match for that method
  - `search_progress.png`: how quickly that optimizer improves the best pattern it has found as seed-level trials accumulate
  - `frontier.png`: that optimizer's own frontier in device-usage/correlation space
  - `frontier_with_grid.png`: that optimizer's frontier overlaid on the grid-sweep frontier for direct comparison
  - `final_report_summary.json`: 10-seed reevaluation of that same pattern on the report seeds
- `results/`
  - `optimizer_search_progress.png`: all optimizer search-progress traces on one plot, showing relative search efficiency
  - `optimizer_robustness.png`: final-pattern mean and standard deviation across the 10 report seeds, showing stability
  - `optimizer_frontiers.png`: sweep and optimizer frontiers on one shared plot, showing the best tradeoff each method reached

## Tests

```bash
pytest
```
