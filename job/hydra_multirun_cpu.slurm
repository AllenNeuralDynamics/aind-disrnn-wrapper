#!/bin/bash
# SLURM array script that runs Hydra multirun (CPU) for deterministic config
# enumeration without W&B sweep orchestration.
#
# See the "HPC run modes" section of README.md for usage and tradeoffs vs.
# W&B sweeps. Typical use:
#
#     sbatch job/hydra_multirun_cpu.slurm
#
#SBATCH -J disrnn_hydra_cpu
#SBATCH -p aind
#SBATCH -t 05:00:00
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH --array=0-11

#SBATCH --mail-user=han.hou@alleninstitute.org
#SBATCH --mail-type=FAIL,END,BEGIN
#SBATCH -o /home/han.hou/logfile/job_%j.out
#SBATCH -e /home/han.hou/logfile/job_%j.err

source /allen/aind/scratch/han.hou/miniforge3/etc/profile.d/conda.sh
conda activate disrnn-cpu

WRAPPER_ROOT=/home/han.hou/code/aind-disrnn-wrapper
cd "$WRAPPER_ROOT"
export PYTHONPATH="$WRAPPER_ROOT:$PYTHONPATH"
export HYDRA_FULL_ERROR=1

python -m code.run_hpc -m \
  wandb.project=hpc_test \
  data=synthetic \
  data.num_sessions=50,100 \
  model.penalties.beta=0.001,0.003,0.01 \
  'seed=range(0,10)'
