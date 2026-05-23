# aind-disrnn-wrapper

The wrapper capsule in the AIND-disRNN MLOps stack:

<img width="1380" height="875" alt="image" src="https://github.com/user-attachments/assets/e029c0e3-ce47-4f65-b61f-42c8bd5b053a" />

The diagram above shows the Code Ocean (CO) architecture, where this repo runs as the "wrapper capsule" orchestrated by the AIND dispatcher.

This repo also supports running directly on HPC (SLURM), without Code Ocean or the dispatcher. The two paths share the same training code and Hydra configs; only the entry points and launch tooling differ.

# Use it in Code Ocean

See the CO pipeline: https://github.com/AllenNeuralDynamics/aind-disrnn-pipeline

# Use it in HPC

## Installation
To run this repo directly on HPC, follow these steps. The HPC path does **not** need the AIND dispatcher capsule on Code Ocean to run, but it does read its Hydra configs from the [`aind-disrnn-dispatcher`](https://github.com/AllenNeuralDynamics/aind-disrnn-dispatcher) repo (single source of truth for configs across CO and HPC).

0. If you need to install Conda on HPC first, follow this guide:
   https://gist.github.com/rhngla/a2bfac4d1f343836cd69090747b6f952#set-up-virtual-environments-on-hpc
1. Clone this repo **and** `aind-disrnn-dispatcher` as siblings (the wrapper's Hydra entry point reads `../../aind-disrnn-dispatcher/code/config` relative to itself):
   ```bash
   cd /path/to/parent
   git clone https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper.git
   git clone https://github.com/AllenNeuralDynamics/aind-disrnn-dispatcher.git
   ```
2. Create CPU and/or GPU conda environments (install only the one(s) you need):
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
3. Optional: install dev tools in either environment:
   ```bash
   pip install -e ".[dev]"
   ```

## Per-user setup

The slurm scripts under `job/` and the launcher need a couple of per-user values (your email, where to write SLURM logs, the path to your `conda.sh`). They live in a gitignored `job/user.env`:

```bash
cp job/user.env.example job/user.env
# edit job/user.env (email, log dir, CONDA_SH)
```

Add one line to your `~/.bashrc` (or `~/.zshrc`) so the values are loaded for every shell:

```bash
echo 'source /path/to/aind-disrnn-wrapper/job/user.env' >> ~/.bashrc
```

The `SBATCH_*` variables are picked up by `sbatch` automatically; `CONDA_SH` is read by the slurm scripts themselves.

## HPC run modes

There are three supported execution modes in HPC.

Conventions for all three:

- The launcher is invoked from the **repo root**; the slurm scripts then `cd` into `code/` before running `python -m run_hpc`. Don't move the launcher invocation into `code/` — `code/launch_wandb_sweep.py` is meant to be called as a module from the repo root.
- The Hydra entry point reads its config from `../../aind-disrnn-dispatcher/code/config` (a sibling clone). Make sure the dispatcher repo is checked out before launching.
- Activate either `disrnn-cpu` or `disrnn-gpu` locally before invoking the launcher — the choice does not matter, it only satisfies `import yaml`. The compute env is selected by `--mode cpu/gpu` and activated inside the slurm script.

### 1) Default: W&B sweep + SLURM array (parallel scan)

Use this for hyperparameter scans.

W&B sweeps provide better experiment visualization and comparison in the W&B UI (best-run ranking, sweep table, grouped sweep analytics).

Project routing note:

- Sweep runs are routed by top-level `entity` and `project` in `sweeps/scaling_disrnn.yaml`.
- In sweep mode, W&B ignores per-run `wandb.project`/`wandb.entity` overrides passed to `run_hpc`.

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

The launcher wraps `wandb sweep` + `sbatch` and adds three things: (1) it injects git/launch lineage into each run as Hydra `+meta.*` overrides (see [Reproducibility](#reproducibility)); (2) it auto-computes `AGENT_COUNT` from the grid size and the array size; (3) it submits a tiny cleanup job that marks the sweep `Finished` in W&B once every array task ends. Each array task runs `wandb agent --count $AGENT_COUNT`; total runs scheduled = `num_array_tasks * AGENT_COUNT`. Defaults: CPU `--array=0-11`, GPU `--array=0-5` (matches the 6-GPU per-user cap; see [GPU tier selection](#gpu-tier-selection)).

CLI flags:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--sweep-yaml` | **required** | Path to the W&B sweep YAML (e.g. `sweeps/scaling_disrnn.yaml`). |
| `--mode` | `cpu` | `cpu` or `gpu`; selects `job/wandb_sweep_{cpu,gpu}.slurm`. |
| `--agent-count` | auto | Override `AGENT_COUNT` (runs per agent). Auto = `ceil(grid_size / array_size)`. |
| `--sbatch-extra` | `""` | Extra sbatch arguments. Must use `=`-form for value-bearing flags (e.g. `--sbatch-extra=--array=0-1`). Pass multiple flags as a single quoted string (e.g. `--sbatch-extra="--array=0-5 --time=01:00:00"`). |
| `--gpu-type` | none | When `--mode gpu`, inject `--gres=gpu:<type>:1` (e.g. `v100`, `a100`, `h200`). See [GPU tier selection](#gpu-tier-selection). |
| `--no-autostop` | off | Keep the sweep open after the array drains (instead of marking it `Finished`). Use when you plan to submit more agents later. |
| `--dry-run` | off | Print the `wandb sweep` and `sbatch` commands without executing. |

Examples:

```bash
# Full sweep, CPU defaults.
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode cpu

# GPU, A100 instead of the default V100.
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode gpu --gpu-type a100

# Bounded validation: 2 array tasks, 1 agent each => 2 runs sampled.
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode cpu \
  --sbatch-extra=--array=0-1 --agent-count 1

# Multiple sbatch overrides: bundle into ONE quoted --sbatch-extra string.
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode cpu \
  --sbatch-extra="--array=0-5 --time=01:00:00"

# Inspect without launching.
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode gpu --dry-run
```

Example output from a successful launch:

```text
Computed AGENT_COUNT=5 from total_grid_runs=60 and num_array_tasks=12
Planned coverage: 60 runs (12 array tasks x 5 agents) out of 60 grid points.
Running: wandb sweep /home/han.hou/code/aind-disrnn-wrapper/sweeps/scaling_disrnn.yaml
Patched sweep YAML with lineage at: /tmp/sweep_62wj64pp.yaml
wandb: Creating sweep from: /tmp/sweep_62wj64pp.yaml
wandb: Creating sweep with ID: 999g0yyr
wandb: View sweep at: https://wandb.ai/AIND-disRNN/hpc_test/sweeps/999g0yyr
wandb: Run sweep agent with: wandb agent AIND-disRNN/hpc_test/999g0yyr
Parsed SWEEP_ID: AIND-disRNN/hpc_test/999g0yyr
Running: sbatch --export ALL,AGENT_COUNT=5 /home/han.hou/code/aind-disrnn-wrapper/job/wandb_sweep_gpu.slurm AIND-disRNN/hpc_test/999g0yyr
Submitted batch job 19640526
```

Note the `Submitted batch job <jobid>` line at the end — that's the SLURM array job ID you'll use to monitor or cancel the run.

Monitoring and cancelling jobs:

```bash
# See all of your jobs (pending + running). Useful right after a launch.
squeue -u $USER

# Watch the queue update every 5s.
watch -n 5 squeue -u $USER

# Inspect one job in detail (state, reason, node, time, etc.).
scontrol show job <jobid>

# Tail the SLURM log for a job (paths come from job/user.env: SBATCH_OUTPUT/SBATCH_ERROR).
tail -f $HOME/logfile/job_<jobid>.out
tail -f $HOME/logfile/job_<jobid>.err

# History (including completed/failed tasks) for an array job.
sacct -j <array_jobid> --format=JobID,State,ExitCode,Start,End

# Cancel one job or a whole array.
scancel <jobid>

# Cancel ALL your pending+running jobs at once (use with care).
scancel -u $USER
```

Cancelling a sweep cleanly: `scancel <array_jobid>` kills compute; the auto-stop cleanup job then fires (because `afterany` treats cancelled tasks as terminal) and marks the W&B sweep `Finished`. If you launched with `--no-autostop`, also run `wandb sweep --stop <SWEEP_ID>` afterwards.

Tips and caveats:

- **Where to change what.** Experiment parameters (the `parameters:` grid and the fixed Hydra overrides in `command:`) belong in the sweep YAML — they define the sweep. Per-launch knobs (array size, walltime, GPU tier, agent count) go on the launcher CLI. The launcher does **not** let you override the sweep `parameters:` from the command line by design; if you need a different grid, copy the YAML and launch the copy.
- **Commit before launching** so `meta.git_dirty` is `no` and `meta.git_commit` uniquely identifies the code state.
- **Auto-stop** marks the sweep `Finished` after every array task ends. Pass `--no-autostop` if you plan to submit more agents to the same sweep later (e.g. `sbatch job/wandb_sweep_cpu.slurm <SWEEP_ID>`); the manual `sbatch` path never auto-stops either.
- **GPU cap.** The `aind` QOS limits each user to 6 concurrent GPUs across all tiers. The GPU slurm script defaults to `--array=0-5` for that reason; oversizing just queues the extras.
- **Slurm scripts rarely need edits.** Per-user values come from `job/user.env`; common overrides are launcher flags. Edit `job/*.slurm` only for non-routine changes (different partition, longer walltime, etc.).

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

Use this for one experiment only. Run from inside `code/` with `code/` on `PYTHONPATH`, and set `DISRNN_OUTPUT_DIR` so Hydra resolves outputs to your HPC home:

```bash
cd /path/to/aind-disrnn-wrapper/code
export PYTHONPATH=/path/to/aind-disrnn-wrapper/code:$PYTHONPATH
export DISRNN_OUTPUT_DIR=$HOME/outputs/disrnn
python -m run_hpc job_id=42 data=mice model=disrnn
```

Hydra writes outputs under `$DISRNN_OUTPUT_DIR/...` (see `aind-disrnn-dispatcher/code/config/config.yaml`), including `inputs.yaml`/`inputs.json` and copied config inputs for reproducibility.

## GPU tier selection

The `aind` partition exposes several GPU tiers (inspect with `sinfo -o "%20N %10c %10m %25f %15G" -p aind`):

| Tier | Nodes (count × GPUs/node) | When to use |
| --- | --- | --- |
| `titanxp` / `1080ti` | 1 × 4 each | Debugging and sanity checks; almost always free. |
| `titanx` | 1 × 2 | Same niche as `titanxp`; older silicon. |
| `v100` | 4 × 4 = 16 GPUs | **Default for this repo.** Plenty of capacity, low queue wait, more than enough for current disRNN sizes. |
| `l40s` | 1 × 2 | Wider models, when v100 is saturated. |
| `a100` | 8 × 1 + 4 × 4 = 24 GPUs | Wider models, more sweep agents per node, or when v100 is saturated. |
| `h200` | 3 × 4 = 12 GPUs | Reserved for genuinely large training (wide nets, long sequences, multi-GPU). Contended; avoid for small models. |

Per-user concurrency cap: the `aind` QOS limits each user to **6 concurrent GPUs total** (`gres/gpu=6` in `MaxTRESPU`; verify with `sacctmgr show qos aind format=Name,MaxTRESPU`). The cap counts all GPU types together, so mixing tiers (e.g. half on `v100`, half on `a100`) does **not** raise the ceiling. The GPU slurm scripts default to `--array=0-5` for this reason.

The GPU slurm scripts (`job/wandb_sweep_gpu.slurm`, `job/hydra_multirun_gpu.slurm`) default to `--gres=gpu:v100:1`. Override per launch without editing the script:

```bash
# W&B sweep with a specific GPU tier
python -m code.launch_wandb_sweep --sweep-yaml sweeps/scaling_disrnn.yaml --mode gpu --gpu-type a100

# Hydra multirun with a specific GPU tier
sbatch --gres=gpu:a100:1 job/hydra_multirun_gpu.slurm
```

Rule of thumb: pick the smallest tier that keeps the GPU >50% utilized (`nvidia-smi dmon -s u 1` on the node while a run is going). For the current disRNN configs (`update_net=5`, `choice_net=4`, sequences of length 50), `v100` is the sweet spot; H200 wins essentially nothing because the model is too small to fill the SMs.

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
