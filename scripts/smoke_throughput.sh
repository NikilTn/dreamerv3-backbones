#!/bin/bash
# Submits smoke runs to measure steady-state throughput per (backbone, env, subexp).
#
# Usage (from repo root):
#   bash scripts/smoke_throughput.sh
#
# Each run writes logs under logdir/smoke/<name>/ and prints fps every
# `update_log_every` steps to metrics.jsonl. After all runs complete:
#   python scripts/smoke_analyze.py --smoke-root logdir/smoke
#
# Defaults target SJSU coe-hpc2's nsfqs partition (H100, 2-day limit, highest priority).

set -euo pipefail

ROOT="${ROOT:-$PWD}"
SMOKE_DIR="${SMOKE_DIR:-$ROOT/logdir/smoke}"
PARTITION="${PARTITION:-nsfqs}"
GRES="${GRES:-gpu:h100:1}"
VENV="${VENV:-$ROOT/.venv/bin/activate}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-64G}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

mkdir -p "$ROOT/slurm"

# Common overrides for every smoke run:
#   - eval disabled (no pauses)
#   - update_log_every=1000 -> ~20 fps samples per 20k-step run
#   - video / param-histogram logging off
#   - shared high-throughput production-ish settings:
#       * env.env_num=8: enough parallel envs to keep replay warm without
#         making env CPU the only thing we test
#       * batch_size=128, batch_length=128: largest conservative common
#         setting after H100 smoke attempts showed batch_size=512 OOMs several
#         backbones and batch_size=256 still OOMs Transformer at length 128
#       * one fair config across all backbones
#   - paper-faithful long-memory configs so smoke fps reflects what the
#     75-run sweep will actually launch with:
#       * trainer.burn_in=8 (Phase 1.5 replay burn-in mask)
#       * model.transformer.cache_length=128 (Phase 2a long-memory TXL;
#         default 8 is backward-compat only)
#     Mamba-2 (state_size=64, nheads=8) and S5 (heads=4, state_size=32)
#     defaults already match production via _base_.yaml.
COMMON="trainer.eval_every=1000000 trainer.eval_episode_num=0 \
trainer.update_log_every=1000 trainer.video_pred_log=False trainer.params_hist_log=False \
batch_size=128 batch_length=128 env.env_num=8 \
trainer.burn_in=8 model.transformer.cache_length=128 seed=0"

submit() {
  local name="$1"; shift
  echo "+ submit $name : $*"
  sbatch \
    --partition="$PARTITION" \
    --gres="$GRES" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --time=00:45:00 \
    --job-name="smoke_${name}" \
    --output="$ROOT/slurm/smoke_${name}_%j.out" \
    --wrap="set -euo pipefail; source $VENV; cd $ROOT; \
            python train.py $COMMON $EXTRA_OVERRIDES logdir=$SMOKE_DIR/$name $*"
}

# 1-5: backbone overhead on POPGym RepeatPreviousEasy (vector env, train_ratio=512).
# 20k env steps -> ~5-10 min wall-clock at ~50-150 fps.
for bb in gru transformer mamba2 s3m s5; do
  submit "popgym_${bb}_none" \
    env=popgym_repeat_previous \
    env.task=popgym_RepeatPreviousEasy-v0 \
    model.backbone="$bb" subexp=none \
    trainer.steps=20000
done

# 6: BSuite calibration (vector env, train_ratio=1024 -> 2x the gradient updates of POPGym).
submit "bsuite_gru_none" \
  env=bsuite_memory_len env.task=bsuite_memory_len/0 \
  model.backbone=gru subexp=none \
  trainer.steps=20000

# 7: Atari calibration (image env, action_repeat=4, train_ratio=128).
# trainer.steps counts FRAMES (after action_repeat). 30k frames = 7.5k env interactions.
submit "atari_gru_none" \
  env=atari100k env.task=atari_pong \
  model.backbone=gru subexp=none \
  trainer.steps=30000

# 8: CPC overhead on POPGym
submit "popgym_gru_cpc" \
  env=popgym_repeat_previous \
  env.task=popgym_RepeatPreviousEasy-v0 \
  model.backbone=gru subexp=cpc \
  trainer.steps=20000

# 9: DFS overhead on POPGym
submit "popgym_gru_dfs" \
  env=popgym_repeat_previous \
  env.task=popgym_RepeatPreviousEasy-v0 \
  model.backbone=gru subexp=dfs \
  trainer.steps=20000

# 10: cross-suite sanity -- non-GRU backbone on Atari.
# Confirms whether backbone overhead transfers from vector envs to image envs.
submit "atari_transformer_none" \
  env=atari100k env.task=atari_pong \
  model.backbone=transformer subexp=none \
  trainer.steps=30000

# 11: CPC overhead on a non-GRU backbone. CPC's per-step projection heads
# are the most likely surprise cost on top of an already-busy SSM forward.
submit "popgym_mamba2_cpc" \
  env=popgym_repeat_previous \
  env.task=popgym_RepeatPreviousEasy-v0 \
  model.backbone=mamba2 subexp=cpc \
  trainer.steps=20000

# 12: DFS overhead on a non-GRU backbone. DFS rewrites the replay sampler;
# any per-backbone interaction shows up here.
submit "popgym_mamba2_dfs" \
  env=popgym_repeat_previous \
  env.task=popgym_RepeatPreviousEasy-v0 \
  model.backbone=mamba2 subexp=dfs \
  trainer.steps=20000

echo
echo "Submitted 12 smoke jobs (5 backbones x POPGym + bsuite/atari sanity + GRU/Mamba CPC + GRU/Mamba DFS + Atari Transformer). Watch with:"
echo "  squeue -u \$USER"
echo
echo "Once they all show CD (completed) in sacct, run:"
echo "  python scripts/smoke_analyze.py --smoke-root $SMOKE_DIR"
