"""Entry point for disRNN wrapper experiments on HPC/SLURM (Beaker / AI Hub).

Loads the dispatcher's Hydra config via the sibling-repo relative path, prepares
it, starts a W&B run, then delegates the train + held-out evaluation/fine-tuning
to the shared ``training_runner.run_training`` (the same body the Code Ocean
entry point run_capsule.py uses), keeping the two paths in parity.
"""

import json
import logging
import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from training_runner import run_training
from utils.json_helpers import dictconfig_to_json
from utils.run_helpers import (
    apply_dynamic_run_name_components,
    apply_model_penalty_multipliers,
    configure_sys_logger,
    copy_inputs_for_run,
    copy_run_to_wandb,
    maybe_restore_checkpoint_from_wandb,
    save_resolved_config,
    start_wandb_run,
)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../aind-disrnn-dispatcher/code/config",
    config_name="config",
)
def main(hydra_config: DictConfig) -> None:
    configure_sys_logger()

    # Prepare config (resolve disRNN penalty multipliers, set dynamic run-name
    # components) before saving inputs / starting W&B — same as run_capsule.py.
    apply_model_penalty_multipliers(hydra_config)
    apply_dynamic_run_name_components(hydra_config)

    hydra_runtime = HydraConfig.get().runtime
    run_dir = Path(hydra_runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Backup inputs for reproducibility
    config_root = Path(__file__).parent.parent.parent / "aind-disrnn-dispatcher" / "code" / "config"
    copy_inputs_for_run(config_root, run_dir / "inputs")
    save_resolved_config(hydra_config, run_dir / "inputs.yaml")
    json_destination = run_dir / "inputs.json"
    json_destination.write_text(
        json.dumps(dictconfig_to_json(hydra_config), indent=2, sort_keys=True)
    )
    logger.info("Saved resolved config JSON to %s", json_destination)

    # --- Prepare wandb ---
    wandb_run = start_wandb_run(hydra_config)

    # Lay the run out as <run_output_base>/inputs.yaml + <run_output_base>/outputs/
    # so post-training analysis (resolve_model_run / auto held-out fine-tuning) can
    # locate it, mirroring the Code Ocean layout. With W&B, use the W&B run dir
    # (copy_run_to_wandb brings inputs.yaml there); otherwise the Hydra run dir.
    #
    # Resumable mode (Beaker autoResume): when DISRNN_RESUMABLE_OUTPUT_DIR is set,
    # anchor outputs at that STABLE path instead of the per-run W&B dir, so a
    # preempted+restarted job re-finds its checkpoints/<step_N>/train_state.pkl
    # and the trainer resumes. Each Beaker task owns its own /results dataset, so
    # a fixed path there is unique to this run. W&B continuity across the restart
    # is handled out-of-band via WANDB_RUN_ID + WANDB_RESUME env vars.
    resumable_output_dir = os.environ.get("DISRNN_RESUMABLE_OUTPUT_DIR")
    run_output_base = run_dir
    if resumable_output_dir:
        run_output_base = Path(resumable_output_dir).resolve()
        run_output_base.mkdir(parents=True, exist_ok=True)
        logger.info("Resumable mode: stable run output base %s", run_output_base)
        # Bring inputs.yaml/inputs/.hydra into the stable base so post-training
        # analysis (resolve_model_run / auto held-out fine-tuning) can locate the
        # run — same as the wandb branch below. Without this the held-out fine-tune
        # fails with "Could not find run inputs at <base>/inputs.yaml". Idempotent
        # on autoResume (dirs_exist_ok; does not touch outputs/checkpoints).
        copy_run_to_wandb(run_dir, run_output_base)
    elif wandb_run is not None:
        run_output_base = Path(wandb_run.dir).resolve()
        logger.info("wandb run directory: %s", run_output_base)
        # Copy hydra run folder (incl. inputs.yaml) into the W&B run dir.
        copy_run_to_wandb(run_dir, run_output_base)
    if hasattr(hydra_config, "model") and "output_dir" in hydra_config.model:
        hydra_config.model.output_dir = str(run_output_base / "outputs")
        logger.info("Model output_dir set to %s", hydra_config.model.output_dir)

    # Extend-later: if training.restore_from_run_id (or DISRNN_RESTORE_FROM_RUN_ID)
    # is set, seed this run's output dir from that prior run's W&B checkpoint
    # artifact so the trainer continues from it (skips warmup) instead of starting
    # fresh. Trainer-agnostic; no-op when not requested; only seeds when no local
    # checkpoint exists yet (preemption-restart safe). See run_helpers docstring.
    maybe_restore_checkpoint_from_wandb(hydra_config, run_output_base)

    resolved_yaml = OmegaConf.to_yaml(hydra_config, resolve=True)
    logger.info("Hydra config (resolved):\n%s", resolved_yaml)

    # --- Train + held-out evaluation/fine-tuning (shared with run_capsule.py) ---
    run_training(hydra_config, wandb_run, run_output_dir=str(run_output_base))
    if wandb_run is not None:
        wandb_run.finish()

    logger.info("All done, goodbye")


if __name__ == "__main__":
    main()
