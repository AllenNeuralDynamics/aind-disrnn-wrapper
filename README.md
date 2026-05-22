# aind-disrnn-wrapper

The wrapper capsule in the AIND-disRNN MLOps stack:

<img width="1380" height="875" alt="image" src="https://github.com/user-attachments/assets/e029c0e3-ce47-4f65-b61f-42c8bd5b053a" />

## Installation in HPC
To install the capsule in an HPC environment, follow these steps:
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

## Running single experiments

Two entry points are available:

- `code.run_capsule` – legacy path used by the Code Ocean pipeline. It expects configs under `/data/jobs` and writes artifacts to `/results`.
- `code.run_hpc` – new Hydra-managed entry point for SLURM/HPC runs. It composes configs from the repo's `config/` folder and writes to `~/outputs`.

For local/HPC runs launch:

```bash
python -m code.run_hpc job_id=42 data=mice model=disrnn
```

Hydra writes everything into `~/outputs/${job_id}` (see `config/config.yaml`). The run directory also stores `inputs.yaml`/`inputs.json` plus a copy of the `config/` tree for auditing.

## Hydra multirun sweeps

Use Hydra's `-m` flag to iterate over datasets or other overrides:

```bash
python -m code.run_hpc -m data=mice,synthetic model=baseline_rl job_id=multirun
```

Each child run lands under `~/outputs/multirun/<hydra_job_id>`. Combine this with SLURM arrays by wrapping the command in your submission script; `job/disrnn_hydra.slurm` shows a template that simply forwards overrides to `python -m code.run_hpc` while Hydra manages per-run working directories automatically.

## Default workflow: W&B sweeps on SLURM arrays

W&B sweep + SLURM arrays is the default path for parallel hyperparameter scans in HPC.

### One-command launcher (recommended)

Use the wrapper to create the sweep, parse `SWEEP_ID`, compute an `AGENT_COUNT`, and submit `sbatch` automatically:

```bash
python -m code.launch_wandb_sweep --mode cpu
python -m code.launch_wandb_sweep --mode gpu
```

Useful options:

```bash
# preview commands without executing
python -m code.launch_wandb_sweep --mode cpu --dry-run

# override array spec used for AGENT_COUNT estimation
python -m code.launch_wandb_sweep --mode gpu --sbatch-extra='--array=0-1'

# force a manual AGENT_COUNT
python -m code.launch_wandb_sweep --mode cpu --agent-count 2
```

### Manual workflow

1. Create a sweep from a YAML spec (example included at `sweeps/scaling_disrnn.yaml`):

   ```bash
   wandb sweep sweeps/scaling_disrnn.yaml
   ```

2. Copy the returned `SWEEP_ID` and submit CPU or GPU agents:

   CPU agents:
   ```bash
   sbatch job/wandb_agent_cpu.slurm <SWEEP_ID>
   ```

   GPU agents:
   ```bash
   sbatch job/wandb_agent_gpu.slurm <SWEEP_ID>
   ```

3. Tune sweep parallelism by changing array size and `AGENT_COUNT` in the job script.
   - CPU default: `--array=0-19` and `AGENT_COUNT=3` (about 60 runs)
   - GPU default: `--array=0-3` and `AGENT_COUNT=15` (about 60 runs)

## Legacy note

Hydra multirun still works well for deterministic config enumeration. W&B sweeps can coexist with Hydra by letting Hydra define structural settings (data/model families) while W&B optimizes numeric hyperparameters.
