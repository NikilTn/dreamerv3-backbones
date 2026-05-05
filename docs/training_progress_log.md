# Training Progress Log

This file tracks long-running cluster training so status survives chat
summaries and SSH session loss.

## RepeatPrevious Reduced Sweep

- Launch time: 2026-05-04 01:07 PDT.
- Main Slurm array: `38888`.
- Rerun job: `38902` for `RepeatPreviousEasy-v0 / s3m / seed4` after an OOM
  in main task `38888_9`.
- Cluster root: `~/dreamerv3/logdir/repeat_previous_reduced_20260504_004958`.
- Rerun root:
  `~/dreamerv3/logdir/repeat_previous_reduced_20260504_004958_reruns`.
- Joblist: `~/dreamerv3/joblists/repeat_previous_reduced_20260504_004958.txt`.
- Main sweep: POPGym RepeatPrevious Easy/Medium/Hard x `gru,s3m,s5` x seeds
  `0..4`, `subexp=none`, 45 planned runs.
- Main overrides: `batch_size=384 batch_length=128 env.env_num=4
  trainer.burn_in=8 model.compile=False trainer.eval_every=1000000
  trainer.eval_episode_num=0 trainer.video_pred_log=False
  trainer.params_hist_log=False`.

### 2026-05-04 06:02 PDT

- Downloaded locally completed checkpoints for:
  - `RepeatPreviousEasy-v0 / gru / seed1`
  - `RepeatPreviousEasy-v0 / gru / seed2`
- Local checkpoint folder:
  `checkpoints/repeat_previous_reduced_20260504_004958`.

### 2026-05-04 08:17 PDT

- Main array `38888`:
  - Completed: 5.
  - Running: 5.
  - Pending: 34.
  - Failed: 1.
- Completed with checkpoints:
  - `RepeatPreviousEasy-v0 / gru / seed0`
  - `RepeatPreviousEasy-v0 / gru / seed1`
  - `RepeatPreviousEasy-v0 / gru / seed2`
  - `RepeatPreviousEasy-v0 / gru / seed3`
  - `RepeatPreviousEasy-v0 / gru / seed4`
- Running:
  - `38888_5`: `RepeatPreviousEasy-v0 / s3m / seed0`, about 980k steps.
  - `38888_6`: `RepeatPreviousEasy-v0 / s3m / seed1`, about 945k steps.
  - `38888_7`: `RepeatPreviousEasy-v0 / s3m / seed2`, about 340k steps.
  - `38888_8`: `RepeatPreviousEasy-v0 / s3m / seed3`, about 335k steps.
  - `38888_10`: `RepeatPreviousEasy-v0 / s5 / seed0`, about 195k steps.
- Failed:
  - `38888_9`: `RepeatPreviousEasy-v0 / s3m / seed4`, OOM at about 105k
    steps on an 80 GB H100 with `batch_size=384`.
- Rerun submitted:
  - `38902`: `RepeatPreviousEasy-v0 / s3m / seed4`, `batch_size=256`,
    pending at last verified check.

### 2026-05-04 08:52 PDT

- Status check blocked because the local SSH ControlMaster socket for
  `coe-hpc2` had expired:
  `~/.ssh/cm/sjsu-hpc2` was missing.
- Refresh with:

```bash
mkdir -p ~/.ssh/cm && chmod 700 ~/.ssh/cm
ssh -M -S ~/.ssh/cm/sjsu-hpc2 \
  -o ServerAliveInterval=60 -o ServerAliveCountMax=3 \
  -fN 019157047@coe-hpc2.sjsu.edu
```

- After refreshing, check:

```bash
ssh -S ~/.ssh/cm/sjsu-hpc2 -o BatchMode=yes 019157047@coe-hpc2.sjsu.edu \
  'cd ~/dreamerv3 && squeue -j 38888,38902 -o "%.18i %.9P %.40j %.8T %.10M %.10l %.6D %R"'
```

### 2026-05-04 09:44 PDT

- SSH socket refreshed and cluster status checked.
- Main array `38888`:
  - Completed: 7.
  - Running: 3.
  - Suspended/admin-held: 2.
  - Pending: 32.
  - Failed: 1.
- Completed with checkpoints:
  - `RepeatPreviousEasy-v0 / gru / seed0..4`
  - `RepeatPreviousEasy-v0 / s3m / seed0`
  - `RepeatPreviousEasy-v0 / s3m / seed1`
- Running:
  - `38888_10`: `RepeatPreviousEasy-v0 / s5 / seed0`, about 525k steps,
    about 59.5 fps.
  - `38888_11`: `RepeatPreviousEasy-v0 / s5 / seed1`, about 535k steps,
    about 103.2 fps.
  - `38888_12`: `RepeatPreviousEasy-v0 / s5 / seed2`, about 480k steps,
    about 102.2 fps.
- Suspended/admin-held:
  - `38888_7`: `RepeatPreviousEasy-v0 / s3m / seed2`, about 505k steps.
  - `38888_8`: `RepeatPreviousEasy-v0 / s3m / seed3`, about 500k steps.
- Failed:
  - `38888_9`: `RepeatPreviousEasy-v0 / s3m / seed4`, OOM at about 105k
    steps with `batch_size=384`.
- Rerun:
  - `38902`: `RepeatPreviousEasy-v0 / s3m / seed4`, `batch_size=256`,
    pending with `Reason=JobHeldAdmin`.
- Tried `scontrol release 38902 38888_7 38888_8`; Slurm returned
  `Access/permission denied`, so the admin hold cannot be released by this
  user account.
- Practical implication: the main array is currently using only 3 active H100s;
  two array slots are occupied by suspended jobs. Do not cancel suspended jobs
  without a deliberate decision, because they have partial progress and may
  resume if the condo hold lifts.

### Admin Hold Interpretation

Slurm reports `Reason=JobHeldAdmin` for rerun job `38902` and suspended jobs
`38888_7` and `38888_8`. `scontrol show partition condo` reports
`MaxTime=30-01:00:00`, `PreemptMode=OFF`, and the jobs requested only
`--time=12:00:00`, so this does not look like a normal walltime limit. It also
does not look like a normal resource wait; normal waiting jobs show reasons
such as `Resources`, `Priority`, or `JobArrayTaskLimit`.

Most likely this is a condo/admin policy or owner-side intervention on those
nodes. Exact cause is not visible from the user account. The user cannot release
these jobs: `scontrol release` returned `Access/permission denied`.

### 750-Run Planning Estimate

For a large expansion of 750 one-million-step runs using the current reduced
backbone set (`gru`, `s3m`, `s5`) and measured end-to-end POPGym rates:

- Optimistic `batch_size=384` for all three backbones: about `2443 GPU-hours`.
- Safer plan with S3M lowered to `env8_bs256` after the 80 GB H100 OOM:
  about `2599 GPU-hours`.
- Very conservative plan with S3M at `env4_bs256`: about `3156 GPU-hours`.

Recommended planning number: use the safer S3M estimate, about
`2600 GPU-hours`, because `s3m batch_size=384` OOMed on an 80 GB condo H100.

Approximate wall clock for `2600 GPU-hours`, before/after 30% retry and
contention slack:

| Total Active GPUs | Ideal | With Slack |
|---:|---:|---:|
| 4 | 27.1 days | 38.7 days |
| 8 | 13.5 days | 19.3 days |
| 12 | 9.0 days | 12.9 days |
| 16 | 6.8 days | 9.7 days |
| 20 | 5.4 days | 7.7 days |

Splitting across 4 people only speeds things up if it increases the total
number of active GPUs. Four people on the same 4-5 available condo H100s will
not be much faster than one well-managed array.

### 2026-05-04 10:30 PDT

- Checked via `coe-hpc2.hpc.coe`; jobs are submitted from `coe-hpc2` to the
  shared Slurm `condo` partition. `hpc3/g17` was an alternate internal access
  path, not the active submission host for this sweep.
- Main array `38888`:
  - Completed: 7.
  - Running: 3.
  - Suspended/admin-held: 2.
  - Pending: 32.
  - Failed: 1.
- Completed with checkpoints:
  - `RepeatPreviousEasy-v0 / gru / seed0..4`
  - `RepeatPreviousEasy-v0 / s3m / seed0`
  - `RepeatPreviousEasy-v0 / s3m / seed1`
- Running:
  - `38888_10`: `RepeatPreviousEasy-v0 / s5 / seed0`, about 690k steps,
    about 59.5 fps on `condo7`.
  - `38888_11`: `RepeatPreviousEasy-v0 / s5 / seed1`, about 825k steps,
    about 103.6 fps on `condo9`.
  - `38888_12`: `RepeatPreviousEasy-v0 / s5 / seed2`, about 765k steps,
    about 101.7 fps on `condo8`.
- Suspended/admin-held:
  - `38888_7`: `RepeatPreviousEasy-v0 / s3m / seed2`, about 505k steps on
    `condo11`.
  - `38888_8`: `RepeatPreviousEasy-v0 / s3m / seed3`, about 500k steps on
    `condo10`.
- Failed:
  - `38888_9`: `RepeatPreviousEasy-v0 / s3m / seed4`, OOM at about 105k
    steps with `batch_size=384`.
- Rerun:
  - `38902`: `RepeatPreviousEasy-v0 / s3m / seed4`, `batch_size=256`,
    still pending with `Reason=JobHeldAdmin`.
- All H100 condo nodes `condo7..condo11` show `mix`; only `condo7/8/9` have
  actively running jobs from this array, while `condo10/11` hold suspended
  array jobs.

### 2026-05-04 10:34 PDT H100 Availability Snapshot

Checked from `coe-hpc2`. H100 nodes visible to Slurm:

- Condo H100s:
  - `condo7`, `condo8`, `condo9`: running our jobs.
  - `condo10`, `condo11`: allocated to our suspended/admin-held array jobs.
- General GPU H100s:
  - `g19`: `IDLE`, available through `gpuqs/gpuqm/gpuql`; `sbatch
    --test-only` predicted immediate start on `gpuqs`.
  - `g16`, `g18`: `MIXED`, currently running other users' H100 jobs.
- NSF H100s:
  - `g26`, `g28`: `IDLE` with reason `osp`; `sbatch --test-only` predicted
    immediate start on `g26` through `nsfqs/nsfqm/nsfql`.
  - `g22`: `ALLOCATED`.
  - `g20`, `g21`, `g23`, `g24`, `g25`, `g27`, `g29`, `g30`, `g31`, `g32`:
    `MIXED`, mostly occupied by other jobs or CPU allocations.

Do not SSH directly into compute nodes for training. Use Slurm allocations from
`coe-hpc2`. Quick probes:

```bash
sinfo -N -o "%18N %18P %12t %24G %E" | awk 'NR==1 || tolower($0) ~ /h100/'
sbatch --test-only --partition=gpuqs --gres=gpu:h100:1 --cpus-per-task=16 --mem=64G --time=12:00:00 --wrap='hostname'
sbatch --test-only --partition=nsfqs --gres=gpu:h100:1 --cpus-per-task=16 --mem=64G --time=12:00:00 --wrap='hostname'
```

To move not-yet-started tasks off the held condo array without duplicating work,
cancel only pending main-array tasks `13..44`, build a new joblist from those
lines, and submit to non-condo H100 partitions. Do this only after confirming
the cancel target is still pending:

```bash
cd ~/dreamerv3
squeue -j 38888 -o "%.18i %.9P %.40j %.8T %.10M %.10l %.6D %R"
sed -n '14,45p' joblists/repeat_previous_reduced_20260504_004958.txt \
  > joblists/repeat_previous_reduced_20260504_004958_remaining_13_44.txt
sed -i 's#^#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #' \
  joblists/repeat_previous_reduced_20260504_004958_remaining_13_44.txt
scancel '38888_[13-44]'
sbatch --partition=gpuqs,nsfqs,gpuqm,nsfqm \
  --gres=gpu:h100:1 \
  --cpus-per-task=16 \
  --mem=64G \
  --time=12:00:00 \
  --job-name=rp-reduced-h100 \
  --array=0-31%8 \
  scripts/slurm_array_template.sh \
  joblists/repeat_previous_reduced_20260504_004958_remaining_13_44.txt
```

The `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` prefix is intended to
reduce allocator fragmentation after the S3M OOM. It does not change model math.

### 2026-05-04 10:54 PDT H100 Queue Snapshot

Checked from `coe-hpc2` with `sinfo`, `squeue`, `scontrol show job`, and
`sbatch --test-only`.

- Visible H100 nodes: 21 total.
  - Condo: `condo7..condo11` = 5 H100s.
  - General GPU: `g16`, `g18`, `g19` = 3 H100s.
  - NSF: `g20..g32` = 13 H100s.
- Immediately free for a fresh 12h H100 job:
  - `gpuqs`: `g19`, predicted start `2026-05-04T10:54:16`.
- Current condo status:
  - `condo7`, `condo8`, `condo9`: running our jobs, nominal time-limit end
    between `2026-05-04T19:14:59` and `2026-05-04T20:24:26`.
  - `condo10`, `condo11`: allocated to our suspended/admin-held jobs. Slurm
    still reports nominal end around `2026-05-04T18:10`, but they are held, so
    do not rely on automatic progress or release.
  - New condo H100 job prediction: `2026-05-05T04:57:44` on `condo11`.
- General GPU H100 queue:
  - `g19`: idle now.
  - `g16`: other user job, nominal end `2026-05-10T19:39:30`.
  - `g18`: other user job, nominal end `2026-05-09T18:39:59`.
  - Start predictions for one 12h H100 job:
    - `gpuqs`: now on `g19`.
    - `gpuqm`: `2026-05-04T21:45:16`.
    - `gpuql`: `2026-05-04T13:04:16`.
- NSF H100 queue:
  - Earliest new 12h job prediction on NSF partitions:
    `2026-05-05T01:38:21` on `g24`.
  - Running H100 jobs have nominal end times ranging from `2026-05-05T01:38`
    (`g24`) through `2026-05-14T12:00` (`g22`, CPU allocation occupying the
    node).

These are scheduler predictions, not guarantees. Jobs can finish early, get
extended/requeued by policy, or be displaced by higher-priority scheduling.

### 2026-05-04 15:00 PDT Training Progress Snapshot

Access through the `hpc2` control socket is working again. Checked `squeue`,
run directories, checkpoints, and latest `metrics.jsonl` values.

- Main array `38888`:
  - Completed/checkpointed: 12 of 45 planned jobs.
  - Running/progressing: 3 jobs.
  - Suspended/admin-held: 2 jobs.
  - Failed: 1 job.
  - Pending: 27 jobs, `38888_[18-44%5]`.
- Rerun job `38902`:
  - `RepeatPreviousEasy-v0 / s3m / seed4`, `batch_size=256`, still pending
    with `Reason=JobHeldAdmin`.
- Completed/checkpointed runs:
  - `RepeatPreviousEasy-v0 / gru / seed0..4`
  - `RepeatPreviousEasy-v0 / s3m / seed0..1`
  - `RepeatPreviousEasy-v0 / s5 / seed0..4`
- Running/progressing runs:
  - `38888_15`: `RepeatPreviousMedium-v0 / gru / seed0`, about 562k steps,
    about 51.6 fps, estimated about 2.4h remaining.
  - `38888_16`: `RepeatPreviousMedium-v0 / gru / seed1`, about 427k steps,
    about 88.1 fps, estimated about 1.8h remaining.
  - `38888_17`: `RepeatPreviousMedium-v0 / gru / seed2`, about 348k steps,
    about 87.1 fps, estimated about 2.1h remaining.
- Still suspended/admin-held:
  - `38888_7`: `RepeatPreviousEasy-v0 / s3m / seed2`, about 508k steps.
  - `38888_8`: `RepeatPreviousEasy-v0 / s3m / seed3`, about 505k steps.
- Failed:
  - `38888_9`: `RepeatPreviousEasy-v0 / s3m / seed4`, OOM at about 106k
    steps with `batch_size=384`.
- Fresh H100 start predictions:
  - `gpuqs`: immediate start on `g19`.
  - `nsfqs`: `2026-05-05T01:38:21` on `g24`.
  - `condo`: `2026-05-04T19:21:44` on `condo11`.

If we leave the main array as condo-only, only three jobs are currently making
progress because two array slots are consumed by suspended admin-held jobs.
The next efficiency move is still to requeue the pending tail `18..44` onto
non-condo H100 partitions, after confirming no duplicate work will be launched.

### 2026-05-04 Local Checkpoint Download

Downloaded all currently completed checkpoints from the cluster to:

```text
checkpoints/repeat_previous_reduced_20260504_004958/full_runs/
```

Local checkpoint count matches the remote count: 12 `latest.pt` files.

Downloaded completed runs:

- `RepeatPreviousEasy-v0 / gru / seed0..4`
- `RepeatPreviousEasy-v0 / s3m / seed0..1`
- `RepeatPreviousEasy-v0 / s5 / seed0..4`

The local copy preserves the remote run directory layout and includes
`latest.pt`, `run_metadata.json`, `resolved_config.json`,
`resolved_config.yaml`, and `metrics.jsonl` for each available run. A local
manifest was written to:

```text
checkpoints/repeat_previous_reduced_20260504_004958/CHECKPOINT_MANIFEST.tsv
```

This manifest should be the input index for any local checkpoint sanity/eval
script.

### 2026-05-04 18:07 PDT Fastgrab H100 Array

SSH access through `coe-hpc2` was re-established. H100 scheduling check:

- `gpuqs`: immediate H100 start available on `g19`.
- `nsfqs/nsfqm/nsfql`: earliest predicted H100 around
  `2026-05-05T01:38:21` on `g24`.
- `condo`: new H100 start was predicted much later; existing condo jobs
  `38888_7` and `38888_8` remained suspended/admin-held.

Moved only not-yet-started original array tasks `38888_21..38888_44` off the
condo-only array:

```text
Original array: 38888
Cancelled:      38888_[21-44]
Left alone:     running 38888_18..20, suspended 38888_7..8, rerun 38902
```

Created fastgrab joblist and manifest:

```text
joblists/repeat_previous_reduced_20260504_004958_fastgrab_21_44.txt
joblists/repeat_previous_reduced_20260504_004958_fastgrab_21_44_manifest.tsv
```

Submitted:

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
  joblists/repeat_previous_reduced_20260504_004958_fastgrab_21_44.txt
```

New Slurm array:

```text
Fastgrab array: 39051
Array range:    0-23%10
Each task:      one training run, one H100
```

`39051_0` started immediately on `g19` through `gpuqs`. It maps to original
array task `21`:

```text
RepeatPreviousMedium-v0 / s3m / seed1
```

Early health check for `39051_0` passed: stdout reached `Create envs.`, the
run `console.log` reached `Simulate agent.` and printed model parameter counts,
and `metrics.jsonl` started writing episode scores plus an initial `fps/fps`
entry. No stderr output at launch.

### 2026-05-05 07:50 PDT Training Progress Snapshot

SSH access through the `hpc2` control socket is working. Checked `squeue`,
Slurm start predictions, `run_metadata.json`, checkpoints, and latest
`metrics.jsonl` values.

Original 45-run reduced sweep status:

- Completed/checkpointed: 25.
- Actively running: 3.
- Suspended/admin-held: 2.
- Failed: 1.
- Pending/not-started: 14.
- Rerun job `38902`: still pending with `Reason=JobHeldAdmin`.

Completed/checkpointed groups:

- `RepeatPreviousEasy-v0 / gru / seed0..4`
- `RepeatPreviousEasy-v0 / s3m / seed0..1`
- `RepeatPreviousEasy-v0 / s5 / seed0..4`
- `RepeatPreviousMedium-v0 / gru / seed0..4`
- `RepeatPreviousMedium-v0 / s3m / seed0..4`
- `RepeatPreviousMedium-v0 / s5 / seed0..2`

Actively running Slurm jobs:

- `39051_7` on `g24` (`nsfqs`):
  `RepeatPreviousMedium-v0 / s5 / seed3`, about 730k steps, about 77.2 fps,
  estimated about 1.0h remaining.
- `39051_8` on `g19` (`gpuqs`):
  `RepeatPreviousMedium-v0 / s5 / seed4`, about 316k steps, about 91.1 fps,
  estimated about 2.1h remaining.
- `39051_9` on `g18` (`gpuqs`):
  `RepeatPreviousHard-v0 / gru / seed0`, about 185k steps, about 75.0 fps,
  estimated about 3.0h remaining.

Still suspended/admin-held:

- `38888_7`: `RepeatPreviousEasy-v0 / s3m / seed2`, about 508k steps.
- `38888_8`: `RepeatPreviousEasy-v0 / s3m / seed3`, about 505k steps.

Failed:

- `38888_9`: `RepeatPreviousEasy-v0 / s3m / seed4`, OOM at about 106k steps
  with `batch_size=384`.

Pending/not-started:

- `39051_10..39051_23`: remaining fastgrab jobs, corresponding to
  `RepeatPreviousHard-v0 / gru / seed1..4`, `s3m / seed0..4`, and
  `s5 / seed0..4`.

Fresh H100 start predictions at this snapshot:

- `gpuqs`: next start predicted `2026-05-05T19:18:52` on `g18`.
- `nsfql`: next start predicted `2026-05-05T16:20:15` on `g25`.
- `nsfqm`: next start predicted `2026-05-05T17:15:15` on `g25`.
- `nsfqs`: next start predicted `2026-05-06T03:43:15` on `g25`.
- `condo`: next start predicted `2026-05-08T07:50:07` on `condo7`.

The fastgrab strategy worked overnight: `g24`, `g19`, and `g18` are currently
running our jobs through non-condo partitions. Condo remains unhealthy for new
work because of the admin-held jobs and poor start prediction.

### 2026-05-05 Local Checkpoint And Eval Update

- Local finished-checkpoint inventory now matches the remote finished-checkpoint
  inventory: 25 local `latest.pt` files and 25 remote `latest.pt` files under
  the reduced RepeatPrevious root.
- Regenerated:

```text
checkpoints/repeat_previous_reduced_20260504_004958/CHECKPOINT_MANIFEST.tsv
```

  with 25 rows, covering every locally downloaded checkpoint.
- Added periodic checkpointing for future runs:
  - `trainer.checkpoint_every: 1e5`
  - `trainer.checkpoint_keep_all: True`
  - snapshots are named `checkpoint_000100000.pt`,
    `checkpoint_000200000.pt`, etc.
- Periodic snapshots contain model and optimizer state for evaluation/inspection.
  They are not exact full-resume checkpoints because replay buffer and
  environment worker state are not saved.
- Synced the checkpointing/eval updates to `coe-hpc2`. Queued jobs that start
  after the sync should use the new checkpointing logic; already-running jobs
  keep the Python process they started with.
- Started local CPU eval smoke:

```bash
.venv/bin/python scripts/eval_checkpoint.py \
  --manifest checkpoints/repeat_previous_reduced_20260504_004958/CHECKPOINT_MANIFEST.tsv \
  --episodes 2 \
  --env-num 2 \
  --device cpu \
  --output-dir checkpoints/repeat_previous_reduced_20260504_004958/eval_2ep_20260505
```

- The 2-episode smoke finished with 25/25 rows and 0 errors.
- Full local deterministic evaluation then ran 20 episodes per checkpoint:

```bash
.venv/bin/python scripts/eval_checkpoint.py \
  --manifest checkpoints/repeat_previous_reduced_20260504_004958/CHECKPOINT_MANIFEST.tsv \
  --episodes 20 \
  --env-num 4 \
  --device cpu \
  --output-dir checkpoints/repeat_previous_reduced_20260504_004958/eval_20ep_20260505
```

  Results were written to:

```text
checkpoints/repeat_previous_reduced_20260504_004958/eval_20ep_20260505/eval_results.jsonl
checkpoints/repeat_previous_reduced_20260504_004958/eval_20ep_20260505/eval_summary.tsv
```

  Aggregate validation means:

| Task | Backbone | Seeds | Eval Mean | Seed Std | Last Train Mean |
|---|---:|---:|---:|---:|---:|
| Easy | `gru` | 5 | -0.3388 | 0.0863 | -0.3167 |
| Easy | `s3m` | 2 | -0.5000 | 0.0062 | -0.4583 |
| Easy | `s5` | 5 | -0.5079 | 0.0173 | -0.4833 |
| Medium | `gru` | 5 | -0.4983 | 0.0243 | -0.5111 |
| Medium | `s3m` | 5 | -0.5075 | 0.0152 | -0.4611 |
| Medium | `s5` | 3 | -0.4968 | 0.0146 | -0.4815 |

  Current interpretation: validation works and agrees broadly with the training
  logs, but the scientific comparison is still incomplete because Easy/S3M,
  Medium/S5, and all Hard runs are not fully checkpointed yet.

### 2026-05-05 Resume Support

- Added practical resume support:
  - `resume_checkpoint=/path/to/latest.pt` or
    `resume_checkpoint=/path/to/checkpoint_000500000.pt`
  - `resume_load_optim: True`
  - `resume_strict: True`
- Resume restores model weights, optimizer state, and the saved `trainer_step`.
- Replay buffer and env worker state are not restored. The resumed run starts
  from the saved global step, collects fresh replay, then resumes updates once
  enough sequence history exists in the new buffer.
- `trainer.steps` remains the total run budget. For example, a resume from
  `500k` toward a `1M` run should keep `trainer.steps=1000000`.
- Exact replay-buffer checkpointing was left off by default because it would be
  large: potentially many GB per run and TB-scale over full sweeps.
