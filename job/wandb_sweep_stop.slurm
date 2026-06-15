#!/bin/bash
# Tiny cleanup job: marks a W&B sweep as Finished after its agent array drains.
#
# Submitted by code/launch_wandb_sweep.py with
#     --dependency=afterany:<array_jobid>
# so SLURM only runs it once every task of the agent array has reached a
# terminal state (success or failure). At that point no more agents will
# claim work, so it is safe to stop the sweep unconditionally.
#
# Manual use (skips the launcher):
#     sbatch job/wandb_sweep_stop.slurm <SWEEP_ID>
#
#SBATCH -J disrnn_wb_stop
#SBATCH -p aind
#SBATCH -t 00:02:00
#SBATCH -c 1
#SBATCH --mem=256M

source "${CONDA_SH:?CONDA_SH not set; source job/user.env before sbatch}"
conda activate disrnn-cpu

SWEEP_ID="$1"
if [ -z "$SWEEP_ID" ]; then
    echo "Usage: sbatch job/wandb_sweep_stop.slurm <SWEEP_ID>" 1>&2
    exit 1
fi

echo "[STOP] Marking sweep $SWEEP_ID as Finished."
wandb sweep --stop "$SWEEP_ID"
