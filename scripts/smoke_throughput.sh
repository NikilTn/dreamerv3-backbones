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

mkdir -p "$ROOT/slurm"

# Common overrides for every smoke run:
#   - eval disabled (no pauses)
#   - update_log_every=1000 -> ~20 fps samples per 20k-step run
#   - video / param-histogram logging off
COMMON="trainer.eval_every=1000000 trainer.eval_episode_num=0 \
trainer.update_log_every=1000 trainer.video_pred_log=False trainer.params_hist_log=False seed=0"

submit() {
  local name="$1"; shift
  echo "+ submit $name : $*"
  sbatch \
    --partition="$PARTITION" \
    --gres="$GRES" \
    --cpus-per-task=8 \
    --mem=32G \
    --time=00:45:00 \
    --job-name="smoke_${name}" \
    --output="$ROOT/slurm/smoke_${name}_%j.out" \
    --wrap="set -euo pipefail; source $VENV; cd $ROOT; \
            python train.py $COMMON logdir=$SMOKE_DIR/$name $*"
}

# 1-5: backbone overhead on POPGym RepeatPreviousEasy (vector env, train_ratio=512).
# 20k env steps -> ~5-10 min wall-clock at ~50-150 fps.
for bb in gru transformer mamba s4 s5; do
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

echo
echo "Submitted 10 smoke jobs. Watch with:"
echo "  squeue -u \$USER"
echo
echo "Once they all show CD (completed) in sacct, run:"
echo "  python scripts/smoke_analyze.py --smoke-root $SMOKE_DIR"
