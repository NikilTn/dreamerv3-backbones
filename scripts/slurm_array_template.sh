#!/bin/bash
# Example SLURM array wrapper for the generated proposal job lists.
#
# Usage:
#   1) Build a job list:
#      python scripts/build_proposal_sweep.py --write-joblist joblists/proposal.txt ...
#   2) Count lines and set the array size accordingly:
#      wc -l joblists/proposal.txt
#   3) Submit as an array, limiting concurrency with %N:
#      sbatch --array=0-74%6 scripts/slurm_array_template.sh joblists/repeat_previous_main.txt

#SBATCH --job-name=dreamer-backbones
#SBATCH --output=slurm/%x_%A_%a.out
#SBATCH --error=slurm/%x_%A_%a.err
#SBATCH --partition=nsfqm
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:h100:1

set -euo pipefail

JOBLIST_PATH="${1:?Usage: sbatch ... slurm_array_template.sh <joblist>}"

cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

mkdir -p slurm
python scripts/run_joblist.py "${JOBLIST_PATH}" --index "${SLURM_ARRAY_TASK_ID}"
