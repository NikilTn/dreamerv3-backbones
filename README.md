# R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation

This repository provides a PyTorch implementation of [R2-Dreamer][r2dreamer] (ICLR 2026), a computationally efficient world model that achieves high performance on continuous control benchmarks. It also includes an efficient PyTorch DreamerV3 reproduction that trains **~5x faster** than a widely used [codebase][dreamerv3-torch], along with other baselines. Selecting R2-Dreamer via the config provides an additional **~1.6x speedup** over this baseline.

## Instructions

Install dependencies. This repository is tested with Ubuntu 24.04 and Python 3.11.

If you prefer Docker, follow [`docs/docker.md`](docs/docker.md).

```bash
# Installing via a virtual env like uv is recommended.
pip install -r requirements.txt
```

Run training on default settings:

```bash
python3 train.py logdir=./logdir/test
```

Monitoring results: training metrics are logged to [Weights & Biases](https://wandb.ai). Authenticate once with `wandb login` (or set `WANDB_API_KEY`) and runs will appear in the `dreamerv3-backbones` project by default. Override the wandb project/entity/run name from the CLI:

```bash
python3 train.py wandb.project=my-project wandb.entity=my-team wandb.name=walker-walk-r2
```

Switching algorithms:

```bash
# Choose an algorithm via model.rep_loss:
# r2dreamer|dreamer|infonce|dreamerpro
python3 train.py model.rep_loss=r2dreamer
```

Switching world-model backbones:

```bash
# Choose a backbone via model.backbone:
# gru|rssm|transformer|storm|mamba2|s4|s3m|s5
python3 train.py model.backbone=transformer
```

Switching proposal sub-experiments:

```bash
# Choose a subexperiment via Hydra config group:
# none|cpc|dfs|cpc_dfs
python3 train.py subexp=cpc
python3 train.py subexp=dfs
python3 train.py subexp=cpc_dfs
```

You can also set the raw config flags directly:

```bash
python3 train.py model.cpc.enabled=true
python3 train.py buffer.sampling.strategy=dfs
python3 train.py model.cpc.enabled=true buffer.sampling.strategy=dfs
```

For easier code reading, inline tensor shape annotations are provided. See [`docs/tensor_shapes.md`](docs/tensor_shapes.md).


## Available Benchmarks
At the moment, the following benchmarks are available in this repository.

| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [Meta-World](https://github.com/Farama-Foundation/Metaworld) | Image | Continuous | 1M | Robotic manipulation with complex contact interactions.|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [DMC Subtle](envs/dmc_subtle.py) | Image | Continuous |1M| DeepMind Control Suite with tiny task-relevant objects. |
| [Atari 100k](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | Image | Discrete |400K| 26 Atari games. |
| [BSuite](https://github.com/google-deepmind/bsuite) | Vector | Discrete | User-defined | Memory-length, memory-size, and discounting-chain diagnostics. |
| [POPGym](https://pypi.org/project/popgym/) | Vector | Discrete | User-defined | Memory-intensive POMDP tasks like RepeatPrevious, Autoencode, and Concentration. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|

Use Hydra to select a benchmark and a specific task using `env` and `env.task`, respectively.

```bash
python3 train.py ... env=dmc_vision env.task=dmc_walker_walk
```

Proposal benchmark quick start:

```bash
# Atari 100k baseline/backbone sweeps
python3 train.py env=atari100k model.backbone=gru env.task=atari_pong
python3 train.py env=atari100k model.backbone=transformer env.task=atari_pong

# BSuite memory diagnostics
python3 train.py env=bsuite_memory_len model.backbone=s5
python3 train.py env=bsuite_memory_size model.backbone=mamba2
python3 train.py env=bsuite_discounting_chain model.backbone=transformer

# POPGym memory tasks
python3 train.py env=popgym_repeat_previous model.backbone=transformer
python3 train.py env=popgym_autoencode model.backbone=s4
python3 train.py env=popgym_concentration model.backbone=s5

# CPC and DFS ablations
python3 train.py env=atari100k model.backbone=transformer subexp=cpc env.task=atari_pong
python3 train.py env=atari100k model.backbone=mamba2 subexp=dfs env.task=atari_pong
python3 train.py env=atari100k model.backbone=transformer subexp=cpc_dfs env.task=atari_pong
```

## HPC Quick Start

Use Python 3.11 on the cluster. `bsuite` currently does not install on Python 3.13, so the proposal benchmark stack should be created in a Python 3.11 environment.

```bash
git clone <your-fork-or-copy> dreamerv3
cd dreamerv3
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
python scripts/verify_setup.py
```

Example backbone sweep command template:

```bash
python3 train.py \
  env=popgym_repeat_previous \
  subexp=none \
  model.backbone=transformer \
  seed=0 \
  logdir=./logdir/proposal/popgym_repeat_previous/none/transformer/seed0
```

To compare backbones fairly, keep `env`, `env.task`, `model` size, `trainer.steps`, and the seed list fixed while only changing `model.backbone` and, when relevant, the `subexp` preset.

## Sweep and Analysis Scripts

Build a backbone sweep job list:

```bash
python scripts/build_proposal_sweep.py \
  --write-joblist joblists/proposal.txt \
  --root-logdir ./logdir/proposal \
  --subexps none cpc dfs cpc_dfs
```

Run a whole job list locally:

```bash
python scripts/run_joblist.py joblists/proposal.txt --all
```

Run the generated commands on SLURM as an array:

```bash
wc -l joblists/proposal.txt
sbatch --array=0-9 scripts/slurm_array_template.sh joblists/proposal.txt
```

Aggregate completed runs into CSV summaries and plots:

```bash
python scripts/analyze_proposal_results.py \
  --logdir-root ./logdir/proposal \
  --output-dir ./logdir/proposal/analysis
```

For the full implementation notes, workflow details, and remaining research steps, see [`docs/proposal_backbone_workflow.md`](docs/proposal_backbone_workflow.md).

## Headless rendering

If you run MuJoCo-based environments (DMC / MetaWorld) on headless machines, you may need to set `MUJOCO_GL` for offscreen rendering. **Using EGL is recommended** as it accelerates rendering, leading to faster simulation throughput.

```bash
# For example, when using EGL (GPU)
export MUJOCO_GL=egl
# (optional) Choose which GPU EGL uses
export MUJOCO_EGL_DEVICE_ID=0
```

More details: [Working with MuJoCo-based environments](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)

## Code formatting

If you want automatic formatting/basic checks before commits, you can enable `pre-commit`:

```bash
pip install pre-commit
# This sets up a pre-commit hook so that checks are run every time you commit
pre-commit install
# Manual pre-commit run on all files
pre-commit run --all-files
```

## Citation

If you find this code useful, please consider citing:

```bibtex
@inproceedings{
morihira2026rdreamer,
title={R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation},
author={Naoki Morihira and Amal Nahar and Kartik Bharadwaj and Yasuhiro Kato and Akinobu Hayashi and Tatsuya Harada},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Je2QqXrcQq}
}
```

[r2dreamer]: https://openreview.net/forum?id=Je2QqXrcQq&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)
[dreamerv3-torch]: https://github.com/NM512/dreamerv3-torch
