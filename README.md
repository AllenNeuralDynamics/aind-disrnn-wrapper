# aind-disrnn-wrapper

The wrapper capsule in the AIND-disRNN MLOps stack:

<img width="1380" height="875" alt="image" src="https://github.com/user-attachments/assets/e029c0e3-ce47-4f65-b61f-42c8bd5b053a" />

## Installation in HPC
To install the capsule in an HPC environment, follow these steps:
1. Create a new conda environment:
   ```bash
   conda create -n disrnn python=3.12 -y
   conda activate disrnn
   ```
2. Install `aind-disrnn-wrapper` in editable mode:
   ```bash
   pip install -e .
   ```
    if want to use GPU version, use
    ```bash
    pip install -e ".[gpu]"
    ```
   with dev tools
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

## WandB sweep workflow

1. Create a sweep spec that launches `python -m code.run_hpc ...` with any fixed Hydra overrides.
2. Start the sweep to receive a `SWEEP_ID` (e.g., `wandb sweep sweep.yaml`).
3. Submit `job/disrnn_wandb_agent.slurm` to schedule agents on SLURM:

   ```bash
   sbatch job/disrnn_wandb_agent.slurm <SWEEP_ID>
   ```

Hydra multiruns can coexist with wandb sweeps by letting Hydra enumerate structural choices (datasets, trainers) while wandb tunes hyperparameters. Document the division of labor in your sweep YAML so each tool controls a distinct dimension.