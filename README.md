# scs-volitional-search

More than a million people in the United States live with paralysis due to spinal cord injury, and many never recover motor function to a satisfactory degree. Epidural spinal cord stimulation (SCS) is promising because injury can disrupt descending motor commands while leaving much of the spinal motor circuitry itself intact, creating the possibility that targeted stimulation could amplify residual commands and help re-engage natural spinal pathways.

This project studies that idea in a mechanistic NEURON simulation of a fine-motor task. We simulate a patient performing the task before lesion to get a healthy EMG target, then simulate the same task after lesion with SCS and search for stimulation patterns that make the lesioned EMG match the healthy one. The motivation for searching over patterned stimulation, rather than only standard tonic stimulation, is that prior work suggests the temporal structure of stimulation matters and may interact with the nonlinear dynamics of spinal circuits.

## Setup

Environment:

- Python `>=3.9`
- NEURON `8.2.7`
- `nrnivmodl` available on `PATH` before running `./scripts/build_neuron.sh`

From the repository root:

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[optim,dev]"
./scripts/setup_external.sh
./scripts/build_neuron.sh
python -c "from neuron import h; print('NEURON OK')"
```

`setup_external.sh` populates `external/SCSInSCIMechanisms`, which provides the fine-motor NEURON model used here. `build_neuron.sh` compiles the NEURON mechanism library. 
`external/` and `results/` are generated, git-ignored working directories. `external/` holds the checked-out simulator dependency and `results/` holds run outputs.

## Experiment

Each trial begins with one `1000 ms` pattern of voluntary supraspinal input, meaning descending input from the brain to the spinal cord. This input has irregular active periods separated by gaps, and during active periods its firing rate varies smoothly between `0` and `60 Hz`. For a given seed, exactly the same generated input is used in every condition.

We then simulate three versions of that same trial. In the healthy pre-lesion condition, the supraspinal input is left intact and the resulting EMG trace is treated as the target. In the lesioned condition, only `20%` of the supraspinal inputs are kept by default. We evaluate the lesioned model both without stimulation and with SCS. The search problem is to find stimulation patterns that make the lesioned EMG with SCS resemble the healthy EMG from the same seed.

The stimulation pattern has six parameters,
`theta = (I0_ma, I1_ma, f0_hz, f1_hz, PW1_us, T_ms)`:

- `I0_ma`: baseline current
- `I1_ma`: how much current varies over time
- `f0_hz`: baseline pulse frequency
- `f1_hz`: how much frequency varies over time
- `PW1_us`: how much pulse width varies around `210 us`
- `T_ms`: modulation period

These control the baseline value and the time-varying part of current, pulse frequency, and pulse width over a repeating cycle of length `T_ms`:

- `I(t) = clip(I0_ma + I1_ma sin(2πt/T_ms), 0, 20)`
- `f(t) = clip(f0_hz + f1_hz sin(2πt/T_ms), 10, 400)`
- `PW(t) = clip(210 + PW1_us sin(2πt/T_ms), 60, 600)`

The frequency function determines when pulses happen. When a pulse occurs at time `t_k`, its current and pulse width are taken from `I(t_k)` and `PW(t_k)`. In the implementation, pulse times are generated on a `0.1 ms` scheduler grid. The limits in these equations are the study limits for delivered stimulation: `0-20 mA`, `10-400 Hz`, and `60-600 us`. They were chosen as conservative values within reported spinal stimulation ranges and programmable device limits ([Howell et al., 2014](https://doi.org/10.1371/journal.pone.0114938); [Darrow et al., 2019](https://doi.org/10.1089/neu.2018.6006); [Medtronic Intellis manual](https://manuals.plus/m/bd8d5a123e572f58dbaa2dd8d7366ae8aee93c5247b73efb75873da0bd0a1ad6)). Within that `60-600 us` range, pulse-width modulation is centered at `210 us`, the baseline pulse width used in this study, which is a common choice according to [D'hondt et al., 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10208259/). The modulation amplitudes are clipped so the realized current, frequency, and pulse width stay inside those limits throughout the trial. Candidates are also rejected if the realized pulse train contains overlapping pulses.

Before stimulation reaches the spinal network, each delivered pulse acts on the modeled afferent fibers, which represent the sensory fibers activated by epidural stimulation. The simulator uses `60` afferent inputs. Stronger pulses recruit more of these fibers, and wider pulses lower the activation threshold through a standard strength-duration relation with chronaxie `360 us` ([Wesselink et al., 1999](https://doi.org/10.1007/BF02513291)). After a fiber fires, it cannot fire again for `0.7 ms` ([Tackmann and Lehmann, 1974](https://doi.org/10.1159/000114626)), and its threshold returns to baseline over the next `3.0 ms` ([Gordineer et al., 2026](https://doi.org/10.1152/jn.00463.2025)). The resulting afferent spikes, together with the supraspinal spikes, drive the NEURON motoneuron network, and EMG is estimated from the motoneuron spike trains.

We explore this six-dimensional stimulation space in four ways. The structured sweep evaluates a constant-parameter tonic slice together with a space-filling Latin hypercube sample of the full box. The three adaptive methods are CMA-ES, TuRBO, and BOHB. At the default `1000` seed-level budget and `3` train seeds per full evaluation, the sweep evaluates `333` candidates: `30` tonic points (`5` current anchors x `6` frequency anchors) and `303` Latin hypercube samples. CMA-ES and TuRBO evaluate every candidate on all three train seeds, while BOHB also uses `1`- and `2`-seed stages internally before promoting candidates to `3` seeds.

## Scoring

The primary score is restoration correlation. For each seed, the healthy pre-lesion EMG trace is compared with the lesion + SCS EMG trace from the same task and the same generated supraspinal input. Both traces are rectified and smoothed with a `25 ms` moving-average window, then scored by Pearson correlation.

Held-out reevaluations also report relative envelope RMSE,
`relative_envelope_rmse = ||e_ref - e_cand||_2 / ||e_ref||_2`.

Hardware usage is summarized with
`device_cost = sum_k (I_k * PW_k) / (T_run_s * I_max * PW_max * f_max)`,
which is the delivered charge rate normalized by the maximum charge rate allowed by the study limits. Here `I_max = 20 mA`, `PW_max = 600 us`, and `f_max = 400 Hz`, consistent with conservative values inside reported spinal-stimulation ranges and programmable device limits ([Howell et al., 2014](https://doi.org/10.1371/journal.pone.0114938); [Darrow et al., 2019](https://doi.org/10.1089/neu.2018.6006); [Medtronic Intellis manual](https://manuals.plus/m/bd8d5a123e572f58dbaa2dd8d7366ae8aee93c5247b73efb75873da0bd0a1ad6)). Summaries also report realized recruitment diagnostics such as `recruitment_raw_dose`, `recruitment_norm_dose`, `current_rate_usage`, `charge_rate_uc_per_s`, and `mean_charge_per_pulse_uc`.

## Runs

Run the full pipeline with one command:

```bash
python scripts/run_all.py
```

By default, this writes to `results/`, uses the aperiodic supraspinal input described above, and allocates `1000` seed-level trials to the sweep and to each optimizer.

To run the stages individually:

```bash
python scripts/run_prelesion_reference.py --output-dir results/reference
python scripts/run_grid_sweep.py --seed-trial-budget 1000 --output-dir results/grid_sweep
python scripts/run_cmaes.py --seed-trial-budget 1000 --output-dir results/cmaes
python scripts/run_turbo.py --seed-trial-budget 1000 --output-dir results/turbo
python scripts/run_bohb.py --seed-trial-budget 1000 --output-dir results/bohb
python scripts/summarize_results.py --results-root results
```

Interrupted optimizer runs can be resumed in place:

```bash
python scripts/run_cmaes.py --output-dir results/cmaes --seed-trial-budget 1000 --resume
python scripts/run_turbo.py --output-dir results/turbo --seed-trial-budget 1000 --resume
python scripts/run_bohb.py --output-dir results/bohb --seed-trial-budget 1000 --resume
```

To extend an existing optimizer run, use `--resume` with an additional budget:

```bash
python scripts/run_cmaes.py \
  --output-dir results/cmaes \
  --resume \
  --additional-seed-trial-budget 500
```

## Defaults

- simulation duration: `1000 ms`
- healthy intact supraspinal inputs: `100%`
- lesioned intact supraspinal inputs: `20%`
- fixed model seed: `672945`
- train seeds: `101, 202, 303`
- report seeds: `1001-1010`
- supraspinal input mode: `aperiodic_envelope`
- current range: `0-20 mA`
- frequency range: `10-400 Hz`
- pulse-width range: `60-600 us`
- pulse-width baseline: `210 us`
- modulation-period range: `50-1000 ms`
- sweep budget: `1000` seed-level trials
- optimizer budget: `1000` seed-level trials
- results root: `results/`

## Results

After a full run, `results/` looks like this:

```text
results/
  reference/
    summary.json                     baseline numbers
    emg_arrays.npz                   cached healthy and lesion EMG traces
    reference_emg.png                healthy vs lesion-no-stim EMG
    supraspinal_drive.png            example supraspinal inputs

  grid_sweep/
    patterns.jsonl                   all sweep evaluations
    summary.json                     train-seed sweep summary
    final_report_summary.json        held-out reevaluation of the sweep winner
    frontier.png                     sweep frontier
    stim_pattern.png                 realized pulse train for the sweep winner
    best_emg.png                     healthy vs lesion+SCS EMG for the sweep winner
    supraspinal_drive.png            train-seed supraspinal inputs used for the sweep plots

  cmaes/
  turbo/
  bohb/
    history.jsonl                    search history
    resume_state.pkl                 resume checkpoint
    summary.json                     train-seed search summary
    final_report_summary.json        held-out reevaluation of the method winner
    search_progress.png              best-so-far correlation versus seed-level trials
    frontier.png                     method frontier
    frontier_with_grid.png           method frontier overlaid on the grid-sweep frontier
    stim_pattern.png                 realized pulse train for the method winner
    best_emg.png                     healthy vs lesion+SCS EMG for the method winner
    supraspinal_drive.png            train-seed supraspinal inputs used for the method plots

  optimizer_frontiers.png            shared frontier comparison across methods
  optimizer_robustness.png           held-out mean and standard deviation across methods
  optimizer_search_progress.png      shared search-progress comparison across optimizers
```

## Tests

```bash
pytest
```
