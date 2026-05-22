#!/bin/bash
#SBATCH -J disrnn_hydra_gpu
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

python -m code.run_hpc -m \
  data=synthetic \
  data.num_sessions=50,100 \
  model.penalties.beta=0.001,0.003,0.01 \
  'seed=range(0,10)'
