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
  data.num_trials=500 \
  data/task=random_walk \
  data.task.reward_baiting=true \
  'data.agent.agent_params_session_var.biasL={type:uniform,min:-3.0,max:3.0}' \
  data.batch_size=512 \
  model.architecture.update_net_n_units_per_layer=5 \
  model.architecture.update_net_n_layers=3 \
  model.architecture.choice_net_n_units_per_layer=4 \
  model.architecture.choice_net_n_layers=0 \
  model.training.lr=0.005 \
  model.training.n_warmup_steps=500 \
  model.training.n_steps=5000 \
  model.training.max_grad_norm=1 \
  data.num_sessions=50,100 \
  model.penalties.beta=0.001,0.003,0.01 \
  'seed=range(0,10)'
