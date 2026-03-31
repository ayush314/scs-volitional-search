# scs-volitional-search

Research codebase for the CMU 18-879 mechanistic SCS project on volitional motor control.

## Conventions

- `alpha(t)` is normalized recruitment fraction in `[0, 1]` instead of current in mA.
- The restoration score is Pearson correlation between pre-lesion EMG envelope and lesion+SCS EMG envelope under the same supraspinal drive.
- This correlation uses full-wave rectified EMG with a 25 ms moving-average envelope.
- The lesion-without-stimulation correlation against pre-lesion is the baseline to beat and is saved in `results/reference/summary.json`.
- Dose is in terms of recruited-fiber-pulses, and normalized dose is `raw_dose / (t_end_seconds * 120 Hz)`.
- A normalized dose of `1.0` means the same recruited-fiber-pulse count as `alpha=1` at `120 Hz` over the full simulation.
- The default simulation duration is `1000 ms` for the sweep and optimizer runs.

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
- Tonic stimulation is the restricted case `T_off=0`, `alpha1=0`, `alpha2=0`.
- Duty-cycled constant-amplitude stimulation is the restricted case `T_off>=0`, `alpha1=0`, `alpha2=0`.

## Sweep

`python scripts/run_grid_sweep.py --output-dir results/grid_sweep` evaluates:

- tonic grid: `12 x 9 = 108` patterns over `(f, alpha0)`
- duty-cycle grid: `12 x 9 = 108` patterns over `(f, duty_cycle)` at fixed amplitude
- full-theta Latin hypercube: `256` patterns over the full 8D `theta` space

Total:

- `472` candidate patterns
- `3` train seeds per pattern
- `1416` seed-level trials

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
python scripts/run_cmaes.py --seed-trial-budget 1416 --output-dir results/cmaes
python scripts/run_turbo.py --seed-trial-budget 1416 --output-dir results/turbo
python scripts/run_bohb.py --seed-trial-budget 1416 --output-dir results/bohb
python scripts/summarize_results.py --results-root results
```

Use this order:

1. Generate pre-lesion and lesion references.
2. Run the full sweep and inspect the dose-correlation upper hull over evaluated patterns.
3. Run CMA-ES, TuRBO, and BOHB.
4. Summarize combined plots directly under `results/`.

`run_prelesion_reference.py` saves the healthy reference traces for both the train seeds and report seeds. The sweep and optimizer scripts reuse `results/reference/emg_arrays.npz` when it exists and only build missing healthy traces.

## Defaults

- lesion severity: `perc_supra_intact = 0.2`
- train/eval seeds per candidate: `3`
- reporting seeds: `3`
- simulation duration: `1000 ms`
- default sweep size: `472` candidate patterns
- default sweep compute: `472 x 3 = 1416` seed-level trials
- optimizer fairness budget: `1416` seed-level trials by default, configurable with `--seed-trial-budget`
- restoration metric: mean seed-averaged EMG-envelope correlation
- dose normalization reference: `alpha=1` at `120 Hz` over the full run duration
- optimizer comparison x-axis: seed-level trials used during search

## Outputs

Each run writes:

- `config.json`
- `metrics.jsonl`
- `metrics.csv`
- summary JSON
- `frontier.json` for sweep and optimizer dose-correlation hulls
- plots where relevant, including local dose-correlation hulls and optimizer best-so-far traces
- dose-correlation and optimizer-trace plots include a dashed lesion-without-stimulation baseline when `results/reference` exists

The optimizer scripts also write:

- `trace.json` with best-so-far correlation versus seed-level trials
- `best_so_far.png`
- `dose_vs_corr.png`
- `dose_vs_corr_with_grid.png` when `results/grid_sweep` exists

`scripts/summarize_results.py` writes combined plots directly to `results/`:

- `reference_emg.png`
- `frontier.png`
- `optimizer_comparison.png`
- `seed_sensitivity.png`

## Tests

```bash
pytest
```
