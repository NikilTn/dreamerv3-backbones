# HPC Runbook

This runbook explains how to set up the repo on the SJSU COE HPC, generate
DreamerV3 backbone joblists, submit Slurm arrays, monitor runs, move pending
jobs between partitions, and collect logs/checkpoints.

It is written for teammates who have their own HPC login. Replace placeholders
such as `<USER>` and `<REPO_URL>` with your own values.

## Mental Model

- One experiment run is one command in a joblist.
- One Slurm array task runs one joblist line.
- One array task requests one H100:

```bash
--gres=gpu:h100:1
```

- The array concurrency suffix controls the maximum number of simultaneous
  runs:

```bash
--array=0-44%5    # 45 total tasks, at most 5 running at once
--array=0-23%10   # 24 total tasks, at most 10 running at once
```

`%10` does not require 10 GPUs to be free immediately. If only one H100 is
free, Slurm starts one task and keeps the rest pending. As more matching H100s
open, Slurm starts more tasks up to the `%10` cap.

## Login

Use the login node to submit jobs. Do not SSH directly into compute nodes for
training.

```bash
ssh <USER>@coe-hpc2.sjsu.edu
```

For long monitoring sessions from a local machine, an SSH ControlMaster socket
avoids repeated password prompts:

```bash
mkdir -p ~/.ssh/cm && chmod 700 ~/.ssh/cm
rm -f ~/.ssh/cm/sjsu-hpc2

ssh -MN \
  -o ControlMaster=yes \
  -o ControlPath="$HOME/.ssh/cm/sjsu-hpc2" \
  -o ControlPersist=6h \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  <USER>@coe-hpc2.sjsu.edu
```

In another local terminal, verify:

```bash
ssh -S ~/.ssh/cm/sjsu-hpc2 -O check <USER>@coe-hpc2.sjsu.edu
```

## Cluster Partitions

A Slurm partition is a queue/pool of nodes. Observed useful H100 partitions:

```text
condo   PI-owned/shared condo nodes, including condo7-condo11 H100s
gpuqs   general GPU short queue
gpuqm   general GPU medium queue
gpuql   general GPU long queue
nsfqs   NSF H100 short queue
nsfqm   NSF H100 medium queue
nsfql   NSF H100 long queue
```

Short queues usually have higher priority but shorter wall-time limits. Medium
and long queues have longer limits but may start later.

Check node/partition state:

```bash
sinfo -N -o "%18N %18P %12t %24G %E" | awk 'NR==1 || tolower($0) ~ /h100/'
squeue -u "$USER" -o "%.18i %.10P %.34j %.8T %.12M %.12l %.8D %R"
```

Ask Slurm when a hypothetical H100 job would start:

```bash
sbatch --test-only \
  --partition=gpuqs \
  --gres=gpu:h100:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=12:00:00 \
  --wrap='hostname'
```

Repeat with `nsfqs`, `gpuqm`, `nsfqm`, or `condo` as needed.

## Repo Setup On HPC

Use Python 3.11 on the cluster.

```bash
git clone <REPO_URL> ~/dreamerv3
cd ~/dreamerv3

python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt

python scripts/verify_setup.py
python -m unittest discover -s tests
```

If `bsuite` is unavailable on a local Python version, BSuite tests may skip.
The HPC environment should use Python 3.11 for the broadest compatibility.

## Single Run

Example POPGym RepeatPrevious run:

```bash
source .venv/bin/activate

python train.py \
  env=popgym_repeat_previous \
  model=size12M \
  subexp=none \
  model.backbone=gru \
  seed=0 \
  logdir=logdir/manual/popgym_repeat_previous/easy/gru/seed0 \
  env.task=popgym_RepeatPreviousEasy-v0 \
  batch_size=384 \
  batch_length=128 \
  env.env_num=4 \
  trainer.burn_in=8 \
  model.compile=False \
  trainer.eval_every=1000000 \
  trainer.eval_episode_num=0 \
  trainer.video_pred_log=False \
  trainer.params_hist_log=False
```

The POPGym config provides:

```yaml
steps: 1e6
action_repeat: 1
```

So a POPGym run trains for 1,000,000 actual POPGym environment steps unless
you override `trainer.steps` or `env.steps`.

For Atari100k in this codebase, remember that trainer steps count
action-repeat-adjusted frames:

```text
100k agent decisions x action_repeat=4 = 400k trainer steps
```

Use `trainer.steps=400000` for Atari100k-style budgets unless the env config
already sets it.

## Reduced POPGym Sweep

The reduced sweep used for the current project run is:

```text
POPGym RepeatPrevious Easy/Medium/Hard
backbones: gru, s3m, s5
seeds: 0,1,2,3,4
subexp: none
total: 3 x 3 x 5 = 45 runs
```

Current shared overrides:

```text
batch_size=384
batch_length=128
env.env_num=4
trainer.burn_in=8
model.compile=False
trainer.eval_every=1000000
trainer.eval_episode_num=0
trainer.video_pred_log=False
trainer.params_hist_log=False
```

Generate a joblist:

```bash
mkdir -p joblists

python scripts/build_proposal_sweep.py \
  --experiments popgym_repeat_previous \
  --backbones gru s3m s5 \
  --seeds 0 1 2 3 4 \
  --subexps none \
  --python .venv/bin/python \
  --override batch_size=384 \
  --override batch_length=128 \
  --override env.env_num=4 \
  --override trainer.burn_in=8 \
  --override model.compile=False \
  --override trainer.eval_every=1000000 \
  --override trainer.eval_episode_num=0 \
  --override trainer.video_pred_log=False \
  --override trainer.params_hist_log=False \
  --write-joblist joblists/repeat_previous_reduced.txt \
  --root-logdir ./logdir/repeat_previous_reduced
```

Verify the job count and inspect the first few commands before spending cluster
time:

```bash
wc -l joblists/repeat_previous_reduced.txt
head -n 3 joblists/repeat_previous_reduced.txt
```

## Slurm Array Submission

Submit to condo H100s:

```bash
mkdir -p slurm

sbatch \
  --partition=condo \
  --gres=gpu:h100:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=12:00:00 \
  --job-name=rp-reduced \
  --array=0-44%5 \
  scripts/slurm_array_template.sh \
  joblists/repeat_previous_reduced.txt
```

Submit to general/NSF H100 queues:

```bash
sbatch \
  --partition=gpuqs,nsfqs,gpuqm,nsfqm \
  --gres=gpu:h100:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=12:00:00 \
  --job-name=rp-fastgrab \
  --array=0-44%10 \
  scripts/slurm_array_template.sh \
  joblists/repeat_previous_reduced.txt
```

The wrapper script activates `.venv` and runs:

```bash
python scripts/run_joblist.py <joblist> --index "$SLURM_ARRAY_TASK_ID"
```

Thus `ARRAY_ID_20` runs line 20 of the joblist.

## Monitoring

Queue:

```bash
squeue -u "$USER" -o "%.18i %.10P %.34j %.8T %.12M %.12l %.8D %R"
```

Specific array:

```bash
squeue -r -j <ARRAY_ID> -o "%.18i %.10P %.34j %.8T %.12M %.12l %.8D %R"
```

Slurm logs:

```bash
tail -n 100 slurm/<job-name>_<array-id>_<task-id>.out
tail -n 100 slurm/<job-name>_<array-id>_<task-id>.err
```

Run logs:

```bash
tail -f logdir/.../console.log
tail -f logdir/.../metrics.jsonl
cat logdir/.../run_metadata.json
```

Healthy early signs:

- Slurm stdout prints the full command.
- `console.log` reaches `Create envs.` and `Simulate agent.`
- model parameter counts are printed.
- `metrics.jsonl` starts writing `episode/score`, `episode/length`, and
  `fps/fps`.

Useful metrics keys in `metrics.jsonl`:

```text
episode/score
episode/length
fps/fps
train/loss/dyn
train/loss/rep
train/loss/rew
train/loss/con
train/loss/policy
train/loss/value
train/loss/repval
train/opt/loss
train/opt/lr
train/opt/updates
```

## Moving Pending Jobs Between Partitions

Sometimes a partition gets stuck, or another H100 opens elsewhere. Move only
pending jobs. Do not cancel running jobs unless you are intentionally throwing
away their progress.

Example: original array `38888` has tasks `21..44` pending on `condo`, and you
want to move them to general/NSF H100 queues.

Create a new joblist from original lines 21..44. Because `sed` is 1-indexed,
use `22,45p`:

```bash
cd ~/dreamerv3

sed -n '22,45p' joblists/repeat_previous_reduced.txt \
  > joblists/repeat_previous_reduced_fastgrab_21_44.txt
```

Optional allocator setting for PyTorch fragmentation-sensitive runs:

```bash
sed -i 's#^#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #' \
  joblists/repeat_previous_reduced_fastgrab_21_44.txt
```

Confirm those tasks are really pending:

```bash
squeue -r -j 38888 -t PENDING -o "%i %T %R"
```

Cancel only the pending tasks:

```bash
scancel '38888_[21-44]'
```

Submit the replacement array:

```bash
sbatch \
  --partition=gpuqs,nsfqs,gpuqm,nsfqm \
  --gres=gpu:h100:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=12:00:00 \
  --job-name=rp-fastgrab \
  --array=0-23%10 \
  scripts/slurm_array_template.sh \
  joblists/repeat_previous_reduced_fastgrab_21_44.txt
```

This does not migrate running jobs. It only resubmits not-yet-started work to
other partitions.

## Admin-Held Or Suspended Jobs

If Slurm shows:

```text
Reason=JobHeldAdmin
STATE=SUSPENDED
```

the user usually cannot release the job:

```bash
scontrol release <jobid>   # may fail with access denied
```

Options:

- Wait for admin/owner policy to release it.
- Cancel it and rerun from scratch.

Current training code saves `latest.pt` at completion and can also save periodic
model snapshots during training when `trainer.checkpoint_every > 0`. For the
current reduced sweep config this is set to `1e5`, so newly started jobs write
`checkpoint_000100000.pt`, `checkpoint_000200000.pt`, and so on, plus a fresh
`latest.pt`.

Important caveat: these snapshots contain the agent and optimizer states. They
are enough for checkpoint evaluation and partial model inspection, but they are
not a full exact resume package because the replay buffer and environment worker
state are not persisted. Jobs that were already running before this code was
synced will not gain periodic snapshots retroactively.

## Logs, Checkpoints, And Analysis

Each run directory contains:

```text
metrics.jsonl
console.log
run_metadata.json
resolved_config.json
resolved_config.yaml
latest.pt              # refreshed at periodic checkpoints and completion
checkpoint_*.pt        # optional numbered model snapshots every N steps
events.out.tfevents... # TensorBoard scalar events
```

Analyze a completed root:

```bash
python scripts/analyze_proposal_results.py \
  --logdir-root ./logdir/repeat_previous_reduced \
  --output-dir ./logdir/repeat_previous_reduced/analysis
```

Outputs include CSV summaries, learning curves, performance profiles, and
`summary.md`.

Download completed checkpoints and scalar logs to a local machine:

```bash
mkdir -p checkpoints/repeat_previous_reduced/full_runs

rsync -av --prune-empty-dirs \
  --include='*/' \
  --include='latest.pt' \
  --include='checkpoint_*.pt' \
  --include='run_metadata.json' \
  --include='resolved_config.json' \
  --include='resolved_config.yaml' \
  --include='metrics.jsonl' \
  --exclude='*' \
  <USER>@coe-hpc2.sjsu.edu:~/dreamerv3/logdir/repeat_previous_reduced/ \
  checkpoints/repeat_previous_reduced/full_runs/
```

Do not commit downloaded checkpoints to Git. They are large artifacts, not
source code.

Local checkpoint sanity/evaluation:

```bash
python scripts/eval_checkpoint.py \
  --manifest checkpoints/repeat_previous_reduced/CHECKPOINT_MANIFEST.tsv \
  --episodes 20 \
  --env-num 4 \
  --device cpu
```

Use a small smoke first, for example `--episodes 2 --env-num 2`, before running
the full validation over every downloaded checkpoint.

## Resuming Training

The repo supports practical model-level resume:

```bash
python train.py \
  ...same run overrides... \
  resume_checkpoint=/path/to/checkpoint_000500000.pt \
  logdir=logdir/resumed/my_run
```

This restores:

- model weights,
- optimizer state,
- `trainer_step` from the checkpoint.

It does not restore replay buffer contents or environment worker state. After
resuming, the trainer starts from the saved global step, collects fresh replay,
and begins updates again once the new replay has enough sequence length.

Keep `trainer.steps` as the total target budget. For example, if a job saved
`checkpoint_000500000.pt` and the intended run budget is 1M steps, resume with
`trainer.steps=1000000`. To extend that run to 1.5M total steps, use
`trainer.steps=1500000`.

Checkpoint size guidance:

- Current model-level checkpoints are usually manageable because they contain
  parameters and optimizer states.
- Exact replay-buffer checkpoints would be much larger. The POPGym buffer can
  hold up to `5e5` transitions, each storing latent state tensors such as
  `deter` and `stoch`; saving this repeatedly across many runs could easily
  become many GB per run and TB-scale across a sweep.
- For this project, prefer model-level resume unless exact bit-for-bit
  continuation is scientifically required.

## Smoke And Throughput Pilots

Before a large sweep, run short pilots:

```bash
PARTITION=gpuqs GRES=gpu:h100:1 bash scripts/smoke_throughput.sh
python scripts/smoke_analyze.py --smoke-root logdir/smoke
```

The smoke scripts estimate end-to-end training throughput and can reveal
backbone-specific slowdowns, OOMs, or config regressions before spending many
H100-hours.

## Current Reduced Sweep Reference

The active reduced sweep used these Slurm IDs and paths:

```text
Main array:      38888
Fastgrab array:  39051
Rerun job:       38902
Root logdir:     logdir/repeat_previous_reduced_20260504_004958
Main joblist:    joblists/repeat_previous_reduced_20260504_004958.txt
Fastgrab list:   joblists/repeat_previous_reduced_20260504_004958_fastgrab_21_44.txt
Fastgrab map:    joblists/repeat_previous_reduced_20260504_004958_fastgrab_21_44_manifest.tsv
```

The fastgrab array was created by moving only pending original tasks `21..44`
off the condo-only array and submitting them to `gpuqs,nsfqs,gpuqm,nsfqm`.

Use `docs/training_progress_log.md` for the latest chronological status, but
treat it as a live project log rather than a reusable setup guide.
