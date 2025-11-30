#!/bin/bash
#SBATCH -J disrnn_job
#SBATCH -p aind
#SBATCH -t 12:00:00
#SBATCH -c 16
#SBATCH --mem=150G

#SBATCH --mail-user=han.hou@alleninstitute.org
#SBATCH --mail-type=FAIL,END,BEGIN
#SBATCH -o /home/han.hou/logfile/disrnn_%j.out
#SBATCH -e /home/han.hou/logfile/disrnn_%j.err

# --- Environment ---
source /allen/aind/scratch/han.hou/miniforge3/etc/profile.d/conda.sh
conda activate disrnn

WRAPPER_ROOT=/home/han.hou/code/aind-disrnn-wrapper
cd "$WRAPPER_ROOT"
export PYTHONPATH="$WRAPPER_ROOT:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
mkdir -p "$HOME/outputs"

# --- Hydra overrides ---
if [ $# -gt 0 ]; then
    CMD=(python -m code.run_hpc "$@")
elif [ -n "$HYDRA_OVERRIDES" ]; then
    # shellcheck disable=SC2206
    OVERRIDES_ARRAY=($HYDRA_OVERRIDES)
    CMD=(python -m code.run_hpc "${OVERRIDES_ARRAY[@]}")
else
    echo "No Hydra overrides supplied. Pass them as script arguments or set HYDRA_OVERRIDES."
    echo "Example: sbatch disrnn_hydra.slurm job_id=123 data=mice model=disrnn"
    exit 1
fi

echo "Launching Hydra run: ${CMD[*]}"
"${CMD[@]}"
