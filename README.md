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

Project routing note:

- Sweep runs are routed by top-level `entity` and `project` in `sweeps/scaling_disrnn.yaml`.
- In sweep mode, W&B ignores per-run `wandb.project`/`wandb.entity` overrides passed to `code.run_hpc`.

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

## Reproducibility

Every run produced by this repo can be traced back to the exact code and command that launched it. This is achieved with three complementary mechanisms:

1. Local Hydra run artifacts.
   - On every run, `code/run_hpc.py` copies the `config/` directory and writes `inputs.yaml` + `inputs.json` (the fully resolved Hydra config) into the run output directory and into the W&B run folder.
   - This captures the full effective configuration after defaults, includes, and CLI overrides have been applied.

2. W&B run config.
   - `wandb.init(config=...)` records the `data`, `model`, and `meta` blocks from the resolved Hydra config, so they are filterable and groupable in the W&B UI.

3. Lineage injection for sweeps (`code/launch_wandb_sweep.py`).
   - At sweep-creation time, the launcher captures git state and launch context and appends them as Hydra `+meta.*` overrides to the sweep `command` list, via a patched temp YAML passed to `wandb sweep`.
   - Each run in the sweep therefore records these fields under `meta.*` in its W&B config:

     | field | meaning |
     | --- | --- |
     | `meta.git_commit` | full SHA of `HEAD` at sweep launch |
     | `meta.git_branch` | current branch |
     | `meta.git_dirty` | `yes` if working tree had uncommitted changes |
     | `meta.sweep_yaml` | sweep YAML path relative to repo root |
     | `meta.owner` | Unix user who launched the sweep |
     | `meta.launcher_cmd` | exact argv used to invoke the launcher |
     | `meta.mode` | `cpu` or `gpu` |

   - These fields are visible to every teammate with W&B access; no shared filesystem or separate registry is required.
   - W&B sweep mode ignores per-run `wandb.entity`/`wandb.project` overrides; sweep routing is set by the top-level `entity`/`project` keys in the sweep YAML.

Recommended practice: commit (or at least record) your changes before launching a sweep so `meta.git_dirty` is `no` and `meta.git_commit` uniquely identifies the code state.
