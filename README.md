# aind-disrnn-wrapper

The wrapper capsule in the AIND-disRNN MLOps stack:

<img width="1380" height="875" alt="image" src="https://github.com/user-attachments/assets/e029c0e3-ce47-4f65-b61f-42c8bd5b053a" />

## Installation in HPC
To install the capsule in an HPC environment, follow these steps:

0. If you need to install Conda on HPC first, follow this guide:
   https://gist.github.com/rhngla/a2bfac4d1f343836cd69090747b6f952#set-up-virtual-environments-on-hpc
1. Create CPU and GPU conda environments:
   ```bash
   conda create -n disrnn-cpu python=3.12 -y
   conda activate disrnn-cpu
   pip install -e .

   conda create -n disrnn-gpu python=3.12 -y
   conda activate disrnn-gpu
   pip install -e ".[gpu]"
   ```
2. Optional: install dev tools in either environment:
   ```bash
   pip install -e ".[dev]"
   ```

## HPC run modes

There are three supported execution modes in HPC.

### 1) Default: W&B sweep + SLURM array (parallel scan)

Use this for hyperparameter scans.

W&B sweeps provide better experiment visualization and comparison in the W&B UI (best-run ranking, sweep table, grouped sweep analytics).

```bash
python -m code.launch_wandb_sweep --mode cpu
python -m code.launch_wandb_sweep --mode gpu
```

Manual equivalent:

```bash
wandb sweep sweeps/scaling_disrnn.yaml
sbatch job/wandb_sweep_cpu.slurm <SWEEP_ID>
sbatch job/wandb_sweep_gpu.slurm <SWEEP_ID>
```

### 2) Hydra multirun + SLURM array

Use this for deterministic config enumeration without W&B sweep orchestration.

Hydra multirun jobs are generated and launched correctly, but they are not automatically grouped as a single sweep in W&B.

CPU multirun jobs:

```bash
sbatch job/hydra_multirun_cpu.slurm
```

GPU multirun jobs:

```bash
sbatch job/hydra_multirun_gpu.slurm
```

Hydra `-m` generates combinations; SLURM array gives scheduler-level fan-out.

### 3) Single run

Use this for one experiment only.

```bash
python -m code.run_hpc job_id=42 data=mice model=disrnn
```

Hydra writes outputs under `~/outputs/disrnn/...` (see `config/config.yaml`), including `inputs.yaml`/`inputs.json` and copied config inputs for reproducibility.
