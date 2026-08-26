# aind-disrnn-wrapper

The wrapper capsule in the AIND-disRNN MLOps stack:

<img width="1380" height="875" alt="image" src="https://github.com/user-attachments/assets/e029c0e3-ce47-4f65-b61f-42c8bd5b053a" />

The diagram above shows the Code Ocean (CO) architecture, where this repo runs
as the wrapper capsule orchestrated by the AIND dispatcher.

# Use it in Code Ocean

See the CO pipeline: https://github.com/AllenNeuralDynamics/aind-disrnn-pipeline

# Use it on Allen HPC

This repo supplies the disRNN runtime entry point, primarily
`code/run_hpc.py`. Allen HPC launch orchestration lives in the dispatcher repo
so the CO, Beaker, and HPC paths are managed in parallel:

```text
aind-disrnn-dispatcher/code/launch_hpc.py
aind-disrnn-dispatcher/code/hpc/
```

Expected local layout:

```bash
/path/to/parent/
  aind-disrnn-dispatcher/
  aind-disrnn-wrapper/
```

Create the runtime environments from this wrapper repo:

```bash
cd /path/to/parent/aind-disrnn-wrapper

conda create -n disrnn-cpu python=3.12 -y
conda activate disrnn-cpu
pip install -e .

conda create -n disrnn-gpu python=3.12 -y
conda activate disrnn-gpu
pip install -e ".[gpu]"
```

Launch W&B sweeps and SLURM jobs from the dispatcher repo. See:

```text
/path/to/parent/aind-disrnn-dispatcher/code/hpc/README.md
```

The launcher is invoked from `disrnn-cpu`; the SLURM script activates
`disrnn-cpu` or `disrnn-gpu` on the compute node based on the selected mode.

For one-off debug runs only, submit through SLURM and run the module from
`code/` with this repo on `PYTHONPATH`:

```bash
cd /path/to/aind-disrnn-wrapper/code
export PYTHONPATH=/path/to/aind-disrnn-wrapper/code:${PYTHONPATH:-}
export DISRNN_OUTPUT_DIR=$HOME/outputs/disrnn
python -m run_hpc job_id=42 data=mice model=disrnn
```

# Reproducibility

Every run produced by this repo can be traced back to the exact code and command
that launched it. This is achieved with three complementary mechanisms:

1. Local Hydra run artifacts.
   - On every run, `code/run_hpc.py` copies the `config/` directory and writes
     `inputs.yaml` and `inputs.json` into the run output directory and into the
     W&B run folder.
   - This captures the full effective configuration after defaults, includes,
     and CLI overrides have been applied.

2. W&B run config.
   - `wandb.init(config=...)` records the `data`, `model`, and `meta` blocks
     from the resolved Hydra config, so they are filterable and groupable in the
     W&B UI.
   - Beaker runs also stamp `wrapper_commit`, `dispatcher_commit`, and
     `foraging_models_commit` after refreshing all three repositories.

3. Dispatcher lineage injection for HPC sweeps.
   - At sweep creation time, `aind-disrnn-dispatcher/code/launch_hpc.py`
     captures dispatcher git state, wrapper git state, and launch context, then
     appends them as Hydra `+meta.*` overrides to the sweep `command` list.
   - Each run in the sweep records fields such as
     `meta.dispatcher_git_commit`, `meta.wrapper_git_commit`,
     `meta.sweep_yaml`, `meta.owner`, `meta.launcher_cmd`, and `meta.mode`.
   - W&B sweep mode ignores per-run `wandb.entity` and `wandb.project`
     overrides; sweep routing is set by the top-level `entity` and `project`
     keys in the sweep YAML.

Recommended practice: commit or record your changes before launching a sweep so
the git lineage uniquely identifies the code state.
