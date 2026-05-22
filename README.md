# aind-disrnn-wrapper

The wrapper capsule in the AIND-disRNN MLOps stack:

<img width="1380" height="875" alt="image" src="https://github.com/user-attachments/assets/e029c0e3-ce47-4f65-b61f-42c8bd5b053a" />

## Installation in HPC
To install the capsule in an HPC environment, follow these steps:

0. If you need to install Conda on HPC first, follow this guide:
   https://gist.github.com/rhngla/a2bfac4d1f343836cd69090747b6f952#set-up-virtual-environments-on-hpc
1. Create CPU and/or GPU conda environments (install only the one(s) you need):
   ```bash
   # CPU environment
   conda create -n disrnn-cpu python=3.12 -y
   conda activate disrnn-cpu
   pip install -e .

   # GPU environment
   conda create -n disrnn-gpu python=3.12 -y
   conda activate disrnn-gpu
   pip install -e ".[gpu]"
   ```
2. Optional: install dev tools in either environment:
   ```bash
   pip install -e ".[dev]"
   ```

## Before running the launcher

`python -m code.launch_wandb_sweep` runs on the login node and only needs `pyyaml` importable (it shells out to the `wandb` CLI rather than importing it). Activate either env on the login side before invoking it:

```bash
conda activate disrnn-cpu   # or disrnn-gpu — choice does not matter here
```

The local activation does **not** affect the compute env. The compute env is selected by `--mode cpu/gpu`, and `job/wandb_sweep_{cpu,gpu}.slurm` activates the correct env on the compute node itself. Activating `disrnn-gpu` locally and passing `--mode cpu` still produces a CPU sweep.

## GPU tier selection

The Allen HPC cluster exposes several GPU tiers (inspect with `sinfo -o "%20N %10c %10m %25f %10G"`):

- `titanxp` / `1080ti` — debugging and sanity checks; almost always free.
- `v100` — the **default for this repo**. Plenty of capacity, low queue wait, more than enough for current disRNN sizes.
- `l40s` / `a100` — wider models, more sweep agents per node, or when v100 is saturated.
- `h200` — reserved for genuinely large training (wide nets, long sequences, multi-GPU). Contended; avoid for small models.

The GPU slurm scripts (`job/wandb_sweep_gpu.slurm`, `job/hydra_multirun_gpu.slurm`) default to `--gres=gpu:v100:1`. Override per launch without editing the script:

```bash
# W&B sweep with a specific GPU tier
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode gpu --gpu-type a100

# Hydra multirun with a specific GPU tier
sbatch --gres=gpu:a100:1 job/hydra_multirun_gpu.slurm
```

Rule of thumb: pick the smallest tier that keeps the GPU >50% utilized (`nvidia-smi dmon -s u 1` on the node while a run is going). For the current disRNN configs (`update_net=5`, `choice_net=4`, sequences of length 50), `v100` is the sweet spot; H200 wins essentially nothing because the model is too small to fill the SMs.

## HPC run modes

There are three supported execution modes in HPC.

### 1) Default: W&B sweep + SLURM array (parallel scan)

Use this for hyperparameter scans.

W&B sweeps provide better experiment visualization and comparison in the W&B UI (best-run ranking, sweep table, grouped sweep analytics).

Project routing note:

- Sweep runs are routed by top-level `entity` and `project` in `sweeps/scaling_disrnn.yaml`.
- In sweep mode, W&B ignores per-run `wandb.project`/`wandb.entity` overrides passed to `code.run_hpc`.

```bash
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode cpu
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode gpu
```

Manual equivalent:

```bash
wandb sweep sweeps/scaling_disrnn.yaml
sbatch job/wandb_sweep_cpu.slurm <SWEEP_ID>
sbatch job/wandb_sweep_gpu.slurm <SWEEP_ID>
```

#### `launch_wandb_sweep.py` in detail

The launcher is the recommended way to start a sweep because it does three things `wandb sweep` + `sbatch` alone do not:

1. Reads the sweep YAML and patches a temp copy with **lineage overrides** (`+meta.git_commit`, `+meta.git_branch`, `+meta.git_dirty`, `+meta.sweep_yaml`, `+meta.owner`, `+meta.launcher_cmd`, `+meta.mode`) appended to the sweep's `command` list, so every run records where it came from. See the [Reproducibility](#reproducibility) section.
2. Creates the sweep on W&B via `wandb sweep <patched_yaml>`, parses the returned sweep ID, and submits `job/wandb_sweep_{cpu,gpu}.slurm` as a SLURM array of agents.
3. Auto-computes `AGENT_COUNT` from the grid size and array size, and warns if the planned coverage (`array_tasks * agent_count`) is smaller than the full grid.

Execution model:

- Each SLURM array task runs one `wandb agent --count $AGENT_COUNT` invocation.
- Each agent claims `AGENT_COUNT` runs from the sweep, runs them sequentially, then exits.
- Total runs scheduled = `num_array_tasks * AGENT_COUNT`. The remaining grid points stay unclaimed; the W&B sweep state will stay `Running` until someone calls `wandb sweep --stop <id>`.
- The default `--array=0-11` in the slurm scripts gives 12 parallel tasks; the launcher will pick `AGENT_COUNT = ceil(total_grid_runs / 12)`.

CLI flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--sweep-yaml` | **required** | Path to the W&B sweep YAML (e.g. `sweeps/scaling_disrnn.yaml`). |
| `--mode` | `cpu` | `cpu` or `gpu`; selects `job/wandb_sweep_{cpu,gpu}.slurm`. |
| `--agent-count` | auto | Override `AGENT_COUNT` (runs per agent). Auto = `ceil(grid_size / array_size)`. |
| `--sbatch-extra` | `""` | Extra sbatch arguments. Must use `=`-form for value-bearing flags (e.g. `--sbatch-extra=--array=0-1`). |
| `--gpu-type` | none | When `--mode gpu`, inject `--gres=gpu:<type>:1` (e.g. `v100`, `a100`, `h200`). See [GPU tier selection](#gpu-tier-selection). |
| `--dry-run` | off | Print the `wandb sweep` and `sbatch` commands without executing. |

Examples:

```bash
# Full sweep with cluster-side defaults (CPU, 12 array tasks, auto AGENT_COUNT).
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode cpu

# Same, but on V100s (default for --mode gpu).
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode gpu

# Constrain to A100 nodes instead.
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode gpu --gpu-type a100

# Bounded validation: 2 array tasks, 1 agent each => 2 runs sampled out of 60.
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode cpu \
  --sbatch-extra=--array=0-1 --agent-count 1

# Inspect what would be run without launching anything.
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode gpu --dry-run

# Use a different sweep YAML.
python -m code.launch_wandb_sweep \
  --sweep-yaml sweeps/my_new_sweep.yaml --mode gpu
```

Tips and caveats:

- Commit your changes before launching, so `meta.git_dirty` is `no` and `meta.git_commit` uniquely identifies the code state.
- After a bounded sweep finishes, the W&B UI will keep showing the sweep as `Running`. The launcher prints the exact `wandb sweep --stop <id>` command to run; do this so the state matches reality.
- Hardcoded sbatch defaults (partition, walltime, memory, mail config) live in the slurm scripts under `job/`. Edit them there; the launcher passes them through.
- The sweep YAML's `command` list controls the per-run `python -m code.run_hpc` invocation. Fixed Hydra overrides (e.g. `data.batch_size=512`) belong there; swept axes go under `parameters`.

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

## Code Ocean compatibility

The HPC migration is additive; the Code Ocean capsule path still works:

- `code/run_capsule.py` is the Code Ocean entry point and is unchanged.
- `code/run` (`python -u run_capsule.py "$@"`) is still wired as the capsule's "Reproducible Run" command.
- `environment/Dockerfile` is unchanged and pins the same dependencies used on HPC.
- Shared helpers in `code/utils/run_helpers.py` (`find_hydra_config`, `copy_input_folder`, `save_resolved_config`, `start_wandb_run`) are used by both the Code Ocean and HPC entry points.
- `CO_COMPUTATION_ID` is still recorded into the W&B run config on Code Ocean; on HPC it is simply absent.
- The hardware tag (`cpu` or `gpu` + model) is detected via `nvidia-smi` with a safe fallback, so it works on both Code Ocean and HPC.

HPC-only additions (sweep lineage injection via `code/launch_wandb_sweep.py`, sbatch scripts under `job/`) are opt-in and have no effect on Code Ocean runs.
