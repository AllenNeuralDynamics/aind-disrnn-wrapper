#!/bin/bash
# SLURM array script that runs a W&B sweep agent (GPU) for an existing sweep.
#
# Prefer launching via code/launch_wandb_sweep.py, which creates the sweep,
# injects git/launch lineage into each run's config (as Hydra +meta.* overrides),
# computes AGENT_COUNT from the grid size, and submits this script for you:
#
#     python -m code.launch_wandb_sweep --mode gpu
#
# Manual use (skips lineage injection and auto AGENT_COUNT):
#     wandb sweep sweeps/scaling_disrnn.yaml
#     sbatch job/wandb_sweep_gpu.slurm <SWEEP_ID>
#
#SBATCH -J disrnn_wb_gpu
#SBATCH -p aind
#SBATCH -t 05:00:00
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-11
#SBATCH --mail-type=FAIL,END,BEGIN

# Per-user settings (mail-user, output/error paths, CONDA_SH) come from
# job/user.env; source it before sbatch. See README "Per-user setup".
source "${CONDA_SH:?CONDA_SH not set; source job/user.env before sbatch}"
conda activate disrnn-gpu

WRAPPER_ROOT="$SLURM_SUBMIT_DIR"
cd "$WRAPPER_ROOT/code"
export PYTHONPATH="$WRAPPER_ROOT/code:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export DISRNN_OUTPUT_DIR="$HOME/outputs/disrnn"

python -c "import jax; print('JAX devices:', jax.devices())"

SWEEP_ID="$1"
if [ -z "$SWEEP_ID" ]; then
    echo "Usage: sbatch job/wandb_sweep_gpu.slurm <SWEEP_ID>" 1>&2
    exit 1
fi

AGENT_COUNT="${AGENT_COUNT:-15}"
echo "[GPU] SWEEP_ID=$SWEEP_ID AGENT_COUNT=$AGENT_COUNT ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
wandb agent --count "$AGENT_COUNT" "$SWEEP_ID"
