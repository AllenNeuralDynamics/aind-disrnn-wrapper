#!/bin/bash
#SBATCH -J disrnn_wandb_agent
#SBATCH -p aind
#SBATCH -t 12:00:00
#SBATCH -c 4
#SBATCH --mem=32G

#SBATCH --mail-user=han.hou@alleninstitute.org
#SBATCH --mail-type=FAIL,END,BEGIN
#SBATCH -o /home/han.hou/logfile/disrnn_agent_%j.out
#SBATCH -e /home/han.hou/logfile/disrnn_agent_%j.err

# --- Setup ---
source /allen/aind/scratch/han.hou/miniforge3/etc/profile.d/conda.sh
conda activate disrnn

WRAPPER_ROOT=/home/han.hou/code/aind-disrnn-wrapper
cd "$WRAPPER_ROOT"
export PYTHONPATH="$WRAPPER_ROOT:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
mkdir -p "$HOME/outputs"

# --- Sweep ID: first arg > env var ---
SWEEP_ID="${1:-${SWEEP_ID:-}}"
if [ -z "$SWEEP_ID" ]; then
    echo "Usage: sbatch disrnn_wandb_agent.slurm <SWEEP_ID>" 1>&2
    echo "Or set SWEEP_ID environment variable before submitting." 1>&2
    exit 1
fi

echo "Using SWEEP_ID: $SWEEP_ID"

# Run a single agent that will execute the command specified in your sweep's `command`.
wandb agent --count 1 "$SWEEP_ID"
