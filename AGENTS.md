# Agent Notes

## Active Work Plan

**Read these before doing anything else:**

- [docs/backbone_work_plan.md](docs/backbone_work_plan.md) — current phase, session changelog, decisions log, open questions. **Update at the end of every session.**
- [docs/backbone_fidelity_status.md](docs/backbone_fidelity_status.md) — per-backbone audit of paper fidelity. Update after each Phase 2 backbone fix.

This file (`AGENTS.md`) is the operational reference: cluster access, partitions, smoke results, recommended sbatch templates. The two docs above own the implementation plan and per-backbone status.

## Operating Rules For This Repo

- Keep the user informed during HPC work. Do not silently wait on remote
  commands; shared filesystems and Slurm calls can hang or take much longer
  than expected. Send short progress updates while waiting.
- Avoid broad remote filesystem walks such as `find ~` on the HPC. Prefer
  targeted paths like `~/dreamerv3`, `~/slurm`, and known log directories.
- Use short SSH timeouts for probes, for example `-o BatchMode=yes
  -o ConnectTimeout=8`.
- Think like a senior engineer before acting: check the current state, identify
  the cheapest reliable probe, and avoid launching expensive Slurm jobs until
  the command, partition, path, environment, and expected output are verified.
- Fact-check assumptions against repo config or Slurm output before committing
  cluster time. For example, verify generated job counts with `wc -l`, verify
  node/partition availability with `sinfo`, and verify run health with
  `metrics.jsonl`, `console.log`, and `squeue`.
- Prefer short, bounded calibration jobs before full sweeps. Cancel pilots once
  they have produced enough signal.

## SJSU COE HPC Access

Do not ask the user to paste HPC passwords into chat. The user created local
OpenSSH ControlMaster sockets so Codex can reuse authenticated SSH sessions.

Persistent socket directory:

```bash
~/.ssh/cm
```

Reachable login paths:

```bash
# Main useful login node / full Slurm GPU cluster view.
ssh -S ~/.ssh/cm/sjsu-hpc2 -o BatchMode=yes 019157047@coe-hpc2.sjsu.edu

# Older/smaller login node, mainly useful as a jump host to coe-hpc3.
ssh -S ~/.ssh/cm/sjsu-hpc1 -o BatchMode=yes 019157047@coe-hpc1.sjsu.edu
```

`coe-hpc3.sjsu.edu` does not resolve directly from the local machine. To reach
the third system, reuse the HPC1 socket and then SSH internally without forcing
the username:

```bash
ssh -S ~/.ssh/cm/sjsu-hpc1 -o BatchMode=yes 019157047@coe-hpc1.sjsu.edu \
  'ssh -x -o BatchMode=yes coe-hpc3 "hostname; whoami"'
```

This internal route was verified to land on:

```text
g17.hpc.coe
019157047
```

If a socket expires, ask the user to re-run the ControlMaster command locally
and enter the password in their own terminal:

```bash
mkdir -p ~/.ssh/cm && chmod 700 ~/.ssh/cm

ssh -M -S ~/.ssh/cm/sjsu-hpc1 \
  -o ServerAliveInterval=60 -o ServerAliveCountMax=3 \
  -fN 019157047@coe-hpc1.sjsu.edu

ssh -M -S ~/.ssh/cm/sjsu-hpc2 \
  -o ServerAliveInterval=60 -o ServerAliveCountMax=3 \
  -fN 019157047@coe-hpc2.sjsu.edu
```

Cluster notes from `sinfo`/`scontrol` on 2026-05-02:

- `coe-hpc2` sees the useful GPU cluster.
- `coe-hpc1` only showed an old/drained P100 node, so it is not the preferred
  login for Dreamer experiments.
- H100 partitions include `nsfqs`, `nsfqm`, `nsfql` on `g20-g32`, plus
  `gpuqs`, `gpuqm`, `gpuql` on `g16`, `g18`, and `g19`.
- A100 nodes include `cs001-cs004` with `gpu:a100:4`.
- The `condo` partition was tested with a short H100 smoke job on
  2026-05-03. Slurm accepted the job and it ran successfully on
  `condo7.hpc.coe`, reporting `NVIDIA H100 PCIe, 81559 MiB`.
- Current condo H100 nodes observed idle after the smoke test:
  `condo7`, `condo8`, `condo9`, `condo10`, `condo11`.
- A 5-way condo H100 pilot array (`38585`) was submitted on 2026-05-03 and
  canceled after about 10 minutes once enough slope data was collected. It used
  POPGym RepeatPreviousEasy, seed 0, subexp none, 50k requested steps, and
  `model.compile=False` to avoid short-run compile overhead. Observed 5-to-10
  minute rates were roughly:
  - `gru`: 3.8 env steps/sec
  - `transformer`: 6.7 env steps/sec
  - `mamba2`: 7.6 env steps/sec
  - `s3m`: 1.6 env steps/sec
  - `s5`: 1.3 env steps/sec
- Projecting the 75-run RepeatPrevious sweep from that pilot gives roughly
  8,100 GPU-hours, or about 67 days on 5 H100s. Treat this as a warning that
  the current non-GRU harness, especially S3M/S5, is too slow for the full
  sweep without further optimization or a smaller step budget.
- The condo pilot did not come close to filling H100 VRAM. A live Slurm-attached
  `nvidia-smi` sample during the GRU pilot showed about 6.9 GiB used out of
  81.6 GiB and about 18% GPU utilization. Current `size12M` POPGym config uses
  `batch_size=16`, `batch_length=64`, and `env_num=1`, so the jobs are not
  configured as large-batch GPU-saturating runs.
- CPU still matters for these runs. POPGym is a CPU environment, the trainer
  uses one environment process by default for this config, and Slurm allocated
  8 CPUs/job via the current array template. Low GPU utilization plus slow
  env-step rates suggest the bottleneck is not H100 VRAM capacity.
- A follow-up condo tuning pilot tested GRU with `env.env_num in {4,8}` and
  `batch_size in {32,64}` using 16 CPUs and 64 GiB/job. Observed steady FPS:
  - `env4_bs32`: about 7.35 env steps/sec
  - `env4_bs64`: about 26.6 env steps/sec in the GRU-only tuning run
  - `env8_bs32`: about 7.9 env steps/sec
  - `env8_bs64`: about 15.4 env steps/sec
  Best tested setting was `env.env_num=4`, `batch_size=64`.
- An all-backbone pilot with `env.env_num=4`, `batch_size=64`, and
  `model.compile=False` observed:
  - `gru`: about 14.0 env steps/sec
  - `transformer`: about 21.8 env steps/sec
  - `mamba2`: about 14.2 env steps/sec
  - `s3m`: about 6.0 env steps/sec
  - `s5`: about 8.0 env steps/sec
  Projected 75-run / 1M-step RepeatPrevious sweep from those rates is about
  1,996 GPU-hours, or about 16.6 wall-clock days on 5 H100s. VRAM still was not
  close to full: live `nvidia-smi` samples were about 10 GiB used per H100.
- A larger-batch GRU-only pilot with `env.env_num=4` tested `batch_size=128`
  and `batch_size=256`:
  - `batch_size=128`: about 25.4 env steps/sec, about 14.8 GiB VRAM in one
    live sample
  - `batch_size=256`: about 71.0 env steps/sec, about 24.4 GiB VRAM in one
    live sample, completed the 20k pilot in under 6 minutes
  Both were valid on condo H100s. This suggests the prior threshold was mainly
  too-small batch / poor GPU occupancy rather than VRAM. Further testing should
  use `env.env_num=4`, `batch_size=256` as the new best GRU setting and then
  validate all backbones with that setting before launching a full sweep.
- A second larger grid tested GRU with `env.env_num in {4,8}` and
  `batch_size in {256,512,1024}`. All six 12k-step pilots completed. Observed
  max/steady FPS was roughly:
  - `env4_bs256`: about 41.7 env steps/sec
  - `env4_bs512`: about 99.0 env steps/sec
  - `env4_bs1024`: about 101.2 env steps/sec
  - `env8_bs256`: about 50.1 env steps/sec
  - `env8_bs512`: about 103.2 env steps/sec
  - `env8_bs1024`: about 109.0 env steps/sec
  For GRU, `env.env_num=8`, `batch_size=1024` is the best tested setting, but
  `env.env_num=4`, `batch_size=512/1024` is close. Before the full sweep, test
  all backbones at high batch size because S3M/S5 may have different memory and
  kernel behavior than GRU.
- A non-GRU backbone grid tested `transformer`, `mamba2`, `s3m`, and `s5` over
  `env4_bs512`, `env4_bs1024`, `env8_bs512`, and `env8_bs1024` for 12k steps.
  Results:
  - `transformer`: `bs1024` fails with CUDA scaled-dot-product attention
    `invalid configuration argument`; best valid setting was `env8_bs512`,
    about 61.2 env steps/sec.
  - `mamba2`: best was `env8_bs1024`, about 93.5 env steps/sec.
  - `s3m`: best was `env8_bs1024`, about 70.3 env steps/sec.
  - `s5`: best was `env8_bs1024`, about 56.9 env steps/sec.
  Combining these with GRU `env8_bs1024` at about 109 env steps/sec gives a
  projected 75-run RepeatPrevious sweep cost of about 283 GPU-hours, or about
  2.4 wall-clock days on 5 condo H100s, assuming 1M steps/run and similar
  throughput across Easy/Medium/Hard.
- Optimization pass on 2026-05-03 added a per-rollout `step_context` cache for
  Transformer relative-position projections, S4D discretization, and S5
  discretization. This reduces repeated tiny kernels inside the RSSM time loop
  without changing model math.
- H100 observe-only benchmark on `condo7` after that optimization, using
  `scripts/bench_backbones.py --device cuda --batch-size 512 --batch-length 128
  --deter 2048 --transformer-cache-length 128 --mamba-state-size 64
  --mamba-nheads 8 --s4-state-size 64 --s5-heads 4 --s5-state-size 32`:
  - Eager with TF32 matching `train.py`: `gru` 306 ms, `transformer` 468 ms,
    `mamba2` 403 ms, `s4` 257 ms, `s5` 296 ms per 512x128 observe rollout.
  - `torch.compile`: `gru` 75 ms, `transformer` 180 ms, `mamba2` 56 ms,
    `s4` 42 ms per rollout.
  - `s5` compile fails in TorchInductor because native `complex64` recurrence
    tensors hit a complex-stride/codegen limitation. `dreamer.py` now disables
    compile automatically only for `model.backbone=s5`; keep compile enabled
    for the other four production backbones.
  - These are RSSM observe-forward microbenchmarks, not end-to-end training
    fps. Still run a short real training pilot before launching any full
    75-run sweep.
- Latest reduced-sweep end-to-end calibration on 2026-05-04, after the
  persistent-state / Transformer-XL / Mamba / S4D / S5 changes, supersedes the
  older 512/1024-batch notes for the current code when running only
  `gru`, `s3m`, and `s5`. Calibration roots:
  - `logdir/reduced_batch_calib_20260504_003351`
  - `logdir/reduced_batch_calib_20260504_003351_followup_256_384`
  Shared overrides were POPGym RepeatPreviousEasy, `batch_length=128`,
  `trainer.burn_in=8`, `model.compile=False`, `trainer.steps=20000`, and
  one H100 per job on `partition=condo`.
  - `batch_size=1024` OOMs for `gru`, `s3m`, and `s5` in the current code.
  - `batch_size=512` is not a safe sweep default: `s3m` OOMs, and `gru`
    failed on an 80 GiB condo H100 even though one `gru env8_bs512` job ran on
    a larger-memory H100. Avoid 512 unless explicitly constraining node type.
  - Best safe measured configs for the reduced 45-run sweep:
    - `gru`: `env.env_num=4`, `batch_size=384`, about 88.4 env steps/sec.
    - `s3m`: `env.env_num=4`, `batch_size=384`, about 69.9 env steps/sec.
    - `s5`: `env.env_num=4`, `batch_size=384`, about 104.7 env steps/sec.
  - Recommended reduced-sweep shared overrides:
    `batch_size=384 batch_length=128 env.env_num=4 trainer.burn_in=8
    model.compile=False`. This keeps one common safe config across `gru/s3m/s5`
    and avoids mixed-node VRAM failures.
  - Projected 45-run RepeatPrevious reduced sweep
    (`3 difficulties x 3 backbones x 5 seeds`, 1M steps/run) is about
    146.6 GPU-hours from the measured best safe rates. Wall clock is roughly
    29.3 h on 5 sustained H100s, or about 42 h with 30% retry/contention slack.
- Active production run launched on 2026-05-04 at 01:07 PDT:
  - Slurm array ID: `38888`
  - Array shape: `0-44%5`
  - Partition/GPU: `condo`, `gpu:h100:1`
  - Joblist: `joblists/repeat_previous_reduced_20260504_004958.txt`
  - Manifest: `joblists/repeat_previous_reduced_20260504_004958_manifest.txt`
  - Root logdir: `logdir/repeat_previous_reduced_20260504_004958`
  - Sweep: POPGym RepeatPrevious Easy/Medium/Hard x `gru,s3m,s5` x seeds
    `0..4`, `subexp=none`, 45 runs total.
  - Overrides: `batch_size=384 batch_length=128 env.env_num=4
    trainer.burn_in=8 model.compile=False trainer.eval_every=1000000
    trainer.eval_episode_num=0 trainer.video_pred_log=False
    trainer.params_hist_log=False`
  - First wave `38888_0..38888_4` started immediately on `condo7..condo11`
    and reached nonzero training fps by 01:11 PDT. Example monitor command:
    `squeue -j 38888 -o "%.18i %.9P %.40j %.8T %.10M %.10l %.6D %R"`.
  - At about 05:54 PDT, tasks `38888_1` and `38888_2` had written
    `run_metadata.json` with `"status": "completed"` and reached step
    `999807`, but the Python process returned exit code 120 during interpreter
    shutdown. `scripts/run_joblist.py` was patched and synced to the cluster so
    future pending array tasks treat exit 120 as success only when completed
    metadata already exists.
- Checkpoint/eval update on 2026-05-05:
  - Local and remote finished-checkpoint counts matched at 25 `latest.pt`
    files under `repeat_previous_reduced_20260504_004958`.
  - Local manifest regenerated at
    `checkpoints/repeat_previous_reduced_20260504_004958/CHECKPOINT_MANIFEST.tsv`
    with 25 rows.
  - Training now supports periodic model snapshots via
    `trainer.checkpoint_every: 1e5` and `trainer.checkpoint_keep_all: True`.
    New jobs started after the sync write `latest.pt` plus numbered
    `checkpoint_*.pt` snapshots every ~100k trainer steps.
  - Caveat: periodic snapshots save the agent and optimizer state for
    evaluation/inspection. They are not exact full-resume checkpoints because
    replay buffer and environment worker state are not persisted.
  - Existing jobs already running before the sync keep their old Python process;
    queued jobs should pick up the new checkpointing code when they start.
  - Local CPU validation is fast enough for current POPGym checkpoints. Use:
    `.venv/bin/python scripts/eval_checkpoint.py --manifest checkpoints/repeat_previous_reduced_20260504_004958/CHECKPOINT_MANIFEST.tsv --episodes 20 --env-num 4 --device cpu`.
  - The dedicated A100 is better reserved for actual training/reruns or
    high-volume validation; it is not necessary for the current 25-checkpoint
    POPGym eval.
  - Practical resume is now supported with `resume_checkpoint=/path/to/*.pt`.
    It restores model weights, optimizer state, and `trainer_step`, then refills
    replay from fresh environment interaction. Keep `trainer.steps` as the
    total target step budget, not the number of additional steps. Exact
    replay-buffer resume is intentionally not enabled by default because it can
    become multi-GB per run and TB-scale across sweeps.
- Remote tmux sessions created on `coe-hpc2`:
  - `dreamerv3_codex`
  - `dreamerv3_hpc2_a`
  - `dreamerv3_hpc2_b`
- Remote tmux sessions created on internal `coe-hpc3`/`g17.hpc.coe`:
  - `dreamerv3_hpc3_a`
  - `dreamerv3_hpc3_b`

Useful commands:

```bash
ssh -t -S ~/.ssh/cm/sjsu-hpc2 019157047@coe-hpc2.sjsu.edu \
  'tmux attach -t dreamerv3_codex'

ssh -t -S ~/.ssh/cm/sjsu-hpc2 019157047@coe-hpc2.sjsu.edu \
  'tmux attach -t dreamerv3_hpc2_a'

ssh -t -S ~/.ssh/cm/sjsu-hpc1 019157047@coe-hpc1.sjsu.edu \
  'ssh -x -t coe-hpc3 "tmux attach -t dreamerv3_hpc3_a"'

sinfo -o "%20P %6a %10l %5D %10t %24G %N"
sinfo -N -o "%18N %18P %10t %24G" | sort
squeue -u "$USER" -o "%.18i %.9P %.40j %.8T %.10M %.10l %.6D %R"
```

Recommended partitions for the DreamerV3/R2-Dreamer POPGym sweep:

```bash
# Smoke jobs
#SBATCH --partition=nsfqs,gpuqs
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00

# Main runs
#SBATCH --partition=nsfqm
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00

# Condo H100s, if intentionally using PI-owned / lower-priority nodes
#SBATCH --partition=condo
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
```
