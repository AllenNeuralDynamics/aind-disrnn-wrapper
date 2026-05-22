#!/bin/bash
#SBATCH -J disrnn_wb_gpu
#SBATCH -p aind
#SBATCH -t 05:00:00
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-11

#SBATCH --mail-user=han.hou@alleninstitute.org
#SBATCH --mail-type=FAIL,END,BEGIN
#SBATCH -o /home/han.hou/logfile/job_%j.out
#SBATCH -e /home/han.hou/logfile/job_%j.err

source /allen/aind/scratch/han.hou/miniforge3/etc/profile.d/conda.sh
conda activate disrnn-gpu

WRAPPER_ROOT=/home/han.hou/code/aind-disrnn-wrapper
cd "$WRAPPER_ROOT"
export PYTHONPATH="$WRAPPER_ROOT:$PYTHONPATH"
export HYDRA_FULL_ERROR=1

python -c "import jax; print('JAX devices:', jax.devices())"

SWEEP_ID="$1"
if [ -z "$SWEEP_ID" ]; then
    echo "Usage: sbatch job/wandb_sweep_gpu.slurm <SWEEP_ID>" 1>&2
    exit 1
fi

AGENT_COUNT="${AGENT_COUNT:-15}"
echo "[GPU] SWEEP_ID=$SWEEP_ID AGENT_COUNT=$AGENT_COUNT ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
wandb agent --count "$AGENT_COUNT" "$SWEEP_ID"
