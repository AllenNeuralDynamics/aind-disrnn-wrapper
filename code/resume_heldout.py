#!/usr/bin/env python
"""Resume-only held-out fine-tune for an OOM-killed grid cell.

Training finished and the checkpoint is on disk; only the auto held-out step
was OOM-killed. Re-run JUST the held-out fine-tune with more RAM, logging into
the ORIGINAL W&B run so the grid stays uniform. Reproduces the auto path
(training_runner._run_auto_heldout_finetune) EXACTLY, including reading the
run's own seed + auto_heldout_finetune settings from inputs.yaml -- the seed
seeds held-out embedding init, so it must match the cell's seed for the number
to equal what the grid cell would have produced.
"""
import sys, argparse, logging, json
from pathlib import Path
# This script lives in code/; make sibling wrapper modules importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml
import wandb
from post_training_analysis import run_heldout_subject_finetuning_from_config
from utils.run_helpers import configure_sys_logger
logging.basicConfig(level=logging.INFO)
p = argparse.ArgumentParser()
p.add_argument("--model-dir", required=True)
p.add_argument("--wandb-run-id", required=True)
p.add_argument("--output-root", default="/home/han.hou/outputs/heldout_subject_finetuning")
p.add_argument("--entity", default="AIND-disRNN")
p.add_argument("--project", default="mice_ignore_scaling")
a = p.parse_args()
configure_sys_logger()

# Read the run's OWN inputs.yaml so seed + auto_heldout settings match the cell
inputs = yaml.safe_load(open(Path(a.model_dir) / "inputs.yaml"))
def _dig(d, *ks):
    for k in ks:
        d = d.get(k, {}) if isinstance(d, dict) else {}
    return d
auto = _dig(inputs, "model", "training", "auto_heldout_finetune")
run_seed = inputs.get("seed", None)
print(f"RESUME using seed={run_seed} checkpoint_policy={auto.get('checkpoint_policy','best_eval')}")

finetune_config = {
    "source_run": {"model_dir": a.model_dir,
                   "checkpoint_policy": str(auto.get("checkpoint_policy", "best_eval"))},
    "heldout_subjects": {k: None for k in
        ("test_subject_ids","curricula","min_sessions","heldout_every_n","mature_only","cols_to_retain")},
    "heldout_finetuning": {
        "n_steps": int(auto.get("n_steps", 500)),
        "lr": float(auto.get("lr", 0.001)),
        "checkpoint_every_n_steps": int(auto.get("checkpoint_every_n_steps", 100)),
        "batch_size": auto.get("batch_size", None),
        "batch_mode": str(auto.get("batch_mode", "single")),
        "keep_media_files": bool(auto.get("keep_media_files", True)),
        "checkpoint_plot_split_examples_every_n": int(auto.get("checkpoint_plot_split_examples_every_n", 100)),
        "checkpoint_save_output_df_every_n": int(auto.get("checkpoint_save_output_df_every_n", 0)),
        "train_example_sessions_per_subject": int(auto.get("train_example_sessions_per_subject", 1)),
        "eval_example_sessions_per_subject": int(auto.get("eval_example_sessions_per_subject", 1)),
        "example_max_subjects": int(auto.get("example_max_subjects", 1)),
    },
    "output": {"output_root": a.output_root, "run_name_suffix": None},
    "seed": run_seed,
}
run = wandb.init(project=a.project, entity=a.entity,
                 id=a.wandb_run_id, resume="must")
step_offset = int(getattr(run, "step", 0)) + 1
res = run_heldout_subject_finetuning_from_config(
    finetune_config, wandb_run=run, wandb_key_prefix="heldout",
    wandb_step_offset=step_offset)
summ = res.get("summary") or {}
print("RESUME DONE output_dir=", res.get("output_dir"))
print("RESUME_METRICS=" + json.dumps({k: v for k, v in summ.items()
      if "engage" in k or "LR_engaged" in k or "likelihood" in k.lower()}))
run.finish()
