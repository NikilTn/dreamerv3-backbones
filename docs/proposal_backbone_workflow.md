# Proposal Backbone Workflow

This document explains the current state of the repository for the DreamerV3 backbone-comparison project, what has already been implemented, how the implementation is structured, how to run the experiments on an HPC system, how to aggregate the results, and what still remains before the repository matches the full scope of the proposal.

## Goal

The repository is being reshaped into a single PyTorch experiment harness where the Dreamer-style world-model backbone can be swapped while the rest of the training stack stays as fixed as possible.

The intended comparison target from the proposal is:

- `GRU / RSSM` baseline
- `Transformer / STORM-style`
- `Mamba / Mamba-2 style`
- `S4 / S3M style`
- `S5 style`

The proposal benchmark focus is:

- `Atari100k`
- `BSuite` memory tasks
- `POPGym` memory tasks

## What Has Been Implemented

### 1. Backbone selection in the main Dreamer model

The world model is no longer hardwired to a single RSSM implementation.

Implemented files:

- `dreamer.py`
- `rssm_factory.py`
- `rssm_base.py`
- `rssm.py`
- `rssm_transformer.py`
- `rssm_mamba.py`
- `rssm_s4.py`
- `rssm_s5.py`

What changed:

- `dreamer.py` now builds the dynamics model through `rssm_factory.py`.
- `configs/model/_base_.yaml` now exposes `model.backbone`.
- All backbone variants implement the same Dreamer-facing API:
  - `initial()`
  - `observe()`
  - `obs_step()`
  - `prior()`
  - `img_step()`
  - `imagine_with_action()`
  - `get_feat()`
  - `kl_loss()`

Why it was done this way:

- The rest of the Dreamer training code already assumes a narrow interface.
- By keeping the interface stable, the actor, value, reward head, continuation head, replay buffer, and trainer can stay shared.
- This makes it possible to compare backbones with minimal non-backbone code drift.

### 2. Shared categorical RSSM scaffold

`rssm_base.py` provides the shared categorical latent machinery:

- posterior network
- prior network
- discrete stochastic latent sampling
- KL balancing loss
- feature extraction from `(stoch, deter)`

Only the deterministic transition model is swapped between backbones.

This is the main reason the repository can now expose multiple backbones without duplicating the entire Dreamer world model for every variant.

### 3. Proposal benchmark environment wrappers

Implemented files:

- `envs/bsuite.py`
- `envs/popgym.py`
- `envs/__init__.py`

Implemented configs:

- `configs/env/bsuite_memory_len.yaml`
- `configs/env/bsuite_memory_size.yaml`
- `configs/env/bsuite_discounting_chain.yaml`
- `configs/env/popgym_repeat_previous.yaml`
- `configs/env/popgym_autoencode.yaml`
- `configs/env/popgym_concentration.yaml`

What these wrappers do:

- flatten vector or structured observations into a single `vector` key
- keep the Dreamer observation contract:
  - `is_first`
  - `is_last`
  - `is_terminal`
- wrap discrete actions so they match the repo's action handling

Important note about BSuite:

- `bsuite` does not install on Python 3.13 because of an upstream packaging issue.
- The repo is set up so that BSuite is expected to work on `Python 3.11` on the HPC.
- This is why the local smoke tests skip BSuite when the package is unavailable.

### 4. HPC-friendly run metadata and resolved configs

Implemented files:

- `tools.py`
- `train.py`

What changed:

- each run now saves:
  - `resolved_config.yaml`
  - `resolved_config.json`
  - `run_metadata.json`
- metadata includes:
  - status
  - start and end time
  - elapsed time
  - seed
  - backbone
  - task
  - trainer step budget

Why this matters:

- analysis scripts no longer need to guess which config produced a run
- wall-clock efficiency metrics can be computed directly
- failed runs can be detected and filtered automatically

### 5. Sweep, job-list, and SLURM scripts

Implemented files:

- `scripts/build_proposal_sweep.py`
- `scripts/run_joblist.py`
- `scripts/slurm_array_template.sh`
- `scripts/verify_setup.py`

What they do:

`build_proposal_sweep.py`

- generates commands for benchmark/backbone/seed grids
- supports proposal benchmark presets
- writes a reusable newline-delimited job list

`run_joblist.py`

- runs one indexed job from a job list
- or runs all jobs sequentially

`slurm_array_template.sh`

- runs one line from a generated job list through a SLURM array index
- keeps SLURM usage generic so it can be adapted to local cluster settings

`verify_setup.py`

- checks whether the key runtime packages are importable
- useful right after setting up the cluster environment

### 6. Result aggregation and plotting script

Implemented file:

- `scripts/analyze_proposal_results.py`

Current outputs:

- `runs.csv`
- `learning_curves.csv`
- `per_task_summary.csv`
- `suite_summary.csv`
- `pairwise_probability_of_improvement.csv`
- `performance_profiles.csv`
- `summary.md`
- learning-curve plots
- performance-profile plots

Current metrics implemented:

- final per-run score
- per-run subexperiment labeling (`none`, `cpc`, `dfs`, `cpc_dfs`)
- per-task mean / std / median / IQM
- per-suite mean / std / median / IQM
- bootstrap confidence intervals for summary metrics
- pairwise probability of improvement
- wall-clock seconds per environment step
- Atari100k human-normalized scores using standard human/random baselines

## Local Validation Done So Far

Smoke tests were added for:

- backbone construction and one training update on CPU
- CPC-enabled training update on CPU
- DFS replay sampling
- POPGym environment wrapper
- sweep script generation
- analysis script output generation

Test files:

- `tests/test_backbones_smoke.py`
- `tests/test_buffer_dfs_smoke.py`
- `tests/test_proposal_envs_smoke.py`
- `tests/test_scripts_smoke.py`

Recommended local check:

```bash
.venv/bin/python -m unittest discover -s tests
```

## How the Current Backbone Implementations Should Be Interpreted

This is important.

The repository is now in a good `backbone-comparison harness` state, but not every backbone is yet a fully paper-faithful reproduction.

Current status:

- `gru / rssm`: closest to the existing baseline implementation
- `transformer`: a real transformer-style deterministic backbone within the shared harness
- `mamba`, `s4`, `s5`: runnable structured-sequence approximations inside the shared harness

What this means in practice:

- You can now train and compare them from one repository.
- You can start running seeds and collecting results on the cluster.
- But if the final report requires strict paper fidelity, these variants still need refinement.

## How to Run on the HPC

Use Python 3.11.

```bash
git clone <your-copy> dreamerv3
cd dreamerv3
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
python scripts/verify_setup.py
```

### Single run example

```bash
python train.py \
  env=popgym_repeat_previous \
  subexp=none \
  model=size12M \
  model.backbone=transformer \
  seed=0 \
  logdir=./logdir/proposal/popgym_repeat_previous/none/RepeatPreviousEasy/transformer/seed0
```

### Subexperiment presets

The repository now exposes the proposal subexperiments as Hydra config presets:

- `subexp=none`
- `subexp=cpc`
- `subexp=dfs`
- `subexp=cpc_dfs`

Examples:

```bash
python train.py env=atari100k model.backbone=transformer subexp=cpc env.task=atari_pong
python train.py env=atari100k model.backbone=mamba2 subexp=dfs env.task=atari_pong
python train.py env=atari100k model.backbone=transformer subexp=cpc_dfs env.task=atari_pong
```

These presets expand to the underlying config flags:

- `model.cpc.enabled`
- `buffer.sampling.strategy`

### Build a proposal sweep

```bash
python scripts/build_proposal_sweep.py \
  --write-joblist joblists/proposal.txt \
  --root-logdir ./logdir/proposal \
  --model size12M \
  --subexps none cpc dfs cpc_dfs
```

### Run locally from the job list

```bash
python scripts/run_joblist.py joblists/proposal.txt --all
```

### Run with SLURM

```bash
wc -l joblists/proposal.txt
sbatch --array=0-99 scripts/slurm_array_template.sh joblists/proposal.txt
```

### Aggregate finished runs

```bash
python scripts/analyze_proposal_results.py \
  --logdir-root ./logdir/proposal \
  --output-dir ./logdir/proposal/analysis
```

## Recommended Experimental Workflow

### Phase 1: Sanity

Run one short debug job per backbone:

- one Atari task
- one POPGym task
- one BSuite task

Confirm:

- run completes
- metrics are written
- checkpoints are written
- aggregation script sees the run

### Phase 2: Baseline

Lock down the `GRU / RSSM` baseline first.

Before spending large cluster budgets on alternative backbones, make sure:

- `gru` runs converge on at least a small subset of Atari100k
- `gru` runs behave sensibly on BSuite and POPGym
- your logging and analysis scripts produce the expected summaries

### Phase 3: Backbone sweep

Once the baseline is stable:

- hold model size fixed
- hold seeds fixed
- hold environment budget fixed
- vary only `model.backbone` or, for subexperiment tables, only `subexp`

This is the core controlled-comparison stage of the proposal.

### Phase 4: Final aggregation

After enough runs are complete:

- aggregate all logdirs
- inspect suite summaries
- inspect per-task learning curves
- inspect pairwise probability of improvement
- use the plots and CSVs as the base for the report

## What Is Still Missing Relative to the Full Proposal

The main software pieces from the proposal are now present:

- selectable backbone variants
- CPC as an auxiliary subexperiment
- DFS as an alternative replay-sampling strategy
- benchmark configs for Atari100k, BSuite, and POPGym
- sweep generation, SLURM support, and result aggregation

What remains is mostly `research completion`, not missing repository plumbing.

### 1. Backbone fidelity improvements

Still needed if you want closer paper fidelity:

- transformer segment caching / relative-position design details
- stricter Mamba / Mamba-2 design choices
- closer S4 / S3M behavior
- closer S5 world-model recurrence details

### 2. Benchmark sweep completeness

The benchmark entrypoints exist, but you still need to decide the exact final task lists:

- which Atari100k 26 games to include
- which BSuite ids to include from the discovered sweep
- which POPGym difficulty levels to include in the final table

### 3. Final report assets

The repository can produce the data artifacts, but you still need to turn them into:

- final paper tables
- final plots for the report
- slide deck material

## Recommended Next Steps

The best next order is:

1. Stabilize the GRU baseline on the exact benchmark subset you want to report.
2. Run small-scale sweeps for all backbones to catch runtime or scaling issues early.
3. Run the `none`, `cpc`, `dfs`, and `cpc_dfs` presets on a reduced benchmark subset to verify that the new options behave sensibly.
4. Re-run the benchmark matrix with fixed seeds and fixed budgets.
5. Aggregate and inspect the results.
6. Only then spend effort on making any single backbone more paper-faithful if the results suggest it is necessary.

## Short Practical Summary

Right now this repo is good for:

- cloning to the HPC
- installing with Python 3.11
- launching benchmark/backbone sweeps
- launching `none`, `cpc`, `dfs`, and `cpc_dfs` ablation sweeps
- collecting run metadata
- aggregating metrics into proposal-oriented summaries

The main reason this is not yet the final submission-ready research artifact is that some non-GRU backbones are still unified-harness approximations rather than exact reproductions, and the final benchmark/task matrix still needs to be fixed and run at scale.

That means the repo is now `operational for the full planned ablation workflow`, but the final research-quality comparison still depends on large-scale runs and, if needed, tighter per-paper backbone fidelity.
