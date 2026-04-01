# scs-volitional-search

Research codebase for the CMU 18-879 mechanistic SCS project on volitional motor control.

## Conventions

- `alpha(t)` is normalized recruitment fraction in `[0, 1]` instead of current in mA.
- The restoration score is Pearson correlation between pre-lesion EMG envelope and lesion+SCS EMG envelope under the same supraspinal drive.
- This correlation uses full-wave rectified EMG with a 25 ms moving-average envelope.
- The lesion-without-stimulation correlation against pre-lesion is the baseline to beat and is saved in `results/reference/summary.json`.
- The default simulation duration is `1000 ms` for the sweep and optimizer runs.
- The public cost axis is a normalized device-budget approximation:
  `I_k = 100 * alpha_k mA`, `q_k = I_k * tau`, and
  `device_cost = sum_k q_k / (T * 100 mA * 1000 us * 1200 Hz)`.
- This hardware-aware cost is a reporting and constraint layer only; the simulator itself still uses recruitment fraction internally.
- These hardware limits are taken from the Medtronic Intellis 97715 implant manual:
  program intensity `0-100 mA`, pulse width `60-1000 us`, master rate `10-1200 Hz`
  ([manual mirror](https://manuals.plus/m/bd8d5a123e572f58dbaa2dd8d7366ae8aee93c5247b73efb75873da0bd0a1ad6)).
- Pulse width is fixed at `210 us` based on human epidural SCS motor studies
  ([review](https://pmc.ncbi.nlm.nih.gov/articles/PMC10208259/)).

## Parameterization

The main stimulation family is:

```text
theta = (f, T_on, T_off, alpha0, alpha1, phi1, alpha2, phi2)
```

with:

```text
alpha(t) = clip(
  alpha0
  + alpha1 * sin(2*pi*t/T + phi1)
  + alpha2 * sin(4*pi*t/T + phi2),
  0,
  1
)
T = T_on + T_off
```

- Pulses are placed at frequency `f` only during on-windows.
- `alpha(t)` is evaluated at pulse times and mapped to recruited afferent fraction.
- Pulse width is fixed at `210 us` for all evaluated patterns.
- Tonic stimulation is the restricted case `T_off=0`, `alpha1=0`, `alpha2=0`.
- Duty-cycled constant-amplitude stimulation is the restricted case `T_off>=0`, `alpha1=0`, `alpha2=0`.

## Sweep

`python scripts/run_grid_sweep.py --output-dir results/grid_sweep` evaluates:

- tonic grid: `8 x 6 = 48` patterns over `(f, alpha0)`
- duty-cycle grid: `8 x 6 = 48` patterns over `(f, duty_cycle)` at fixed `alpha=0.5`
- full-theta Latin hypercube: `104` patterns over the full 8D `theta` space

Total:

- `200` candidate patterns
- `3` train seeds per pattern
- `600` seed-level trials

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[optim,dev]"
./scripts/setup_external.sh
./scripts/build_neuron.sh
python -c "from neuron import h; print('NEURON OK')"
```

## Main Workflow

```bash
python scripts/run_prelesion_reference.py --output-dir results/reference
python scripts/run_grid_sweep.py --output-dir results/grid_sweep
python scripts/run_cmaes.py --seed-trial-budget 600 --output-dir results/cmaes
python scripts/run_turbo.py --seed-trial-budget 600 --output-dir results/turbo
python scripts/run_bohb.py --seed-trial-budget 600 --output-dir results/bohb
python scripts/summarize_results.py --results-root results
```

Use this order:

1. Generate pre-lesion and lesion references.
2. Run the full sweep and inspect the device-budget/correlation upper hull over evaluated patterns.
3. Run CMA-ES, TuRBO, and BOHB.
4. Summarize combined plots directly under `results/`.

`run_prelesion_reference.py` saves the healthy reference traces for both the train seeds and report seeds. The sweep and optimizer scripts reuse `results/reference/emg_arrays.npz` when it exists and only build missing healthy traces.

## Defaults

- lesion severity: `perc_supra_intact = 0.2`
- train/eval seeds per candidate: `3`
- reporting seeds: `3`
- simulation duration: `1000 ms`
- fixed pulse width for all runs: `210 us`
- default sweep size: `200` candidate patterns
- default sweep compute: `200 x 3 = 600` seed-level trials
- optimizer fairness budget: `600` seed-level trials by default, configurable with `--seed-trial-budget`
- restoration metric: mean seed-averaged EMG-envelope correlation
- device budget normalization reference: `100 mA`, `1000 us`, `1200 Hz` over the full run duration
- optimizer comparison x-axis: seed-level trials used during search

## Outputs

Each run writes:

- `config.json`
- `metrics.jsonl`
- `metrics.csv`
- summary JSON
- `frontier.json` for sweep and optimizer device-budget/correlation hulls
- plots where relevant, including local device-budget/correlation hulls and optimizer best-so-far traces
- device-budget/correlation and optimizer-trace plots include a dashed lesion-without-stimulation baseline when `results/reference` exists

The optimizer scripts also write:

- `trace.json` with best-so-far correlation versus seed-level trials
- `best_so_far.png`
- `device_budget_vs_corr.png`
- `device_budget_vs_corr_with_grid.png` when `results/grid_sweep` exists

`scripts/summarize_results.py` writes combined plots directly to `results/`:

- `reference_emg.png`
- `frontier.png`
- `optimizer_comparison.png`
- `seed_sensitivity.png`

## Tests

```bash
pytest
```
