"""Entry point for disRNN wrapper experiments (Code Ocean pipeline).

Loads the active Hydra config from the Code Ocean job mount (/data/jobs),
prepares it, starts a W&B run, then delegates the train + held-out
evaluation/fine-tuning to the shared ``training_runner.run_training`` (the same
body the Beaker/HPC entry point run_hpc.py uses), keeping the two paths in parity.
"""

import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

from training_runner import run_training
from utils.run_helpers import (
    apply_dynamic_run_name_components,
    apply_model_penalty_multipliers,
    configure_sys_logger,
    copy_input_folder,
    find_hydra_config,
    save_resolved_config,
    start_wandb_run,
)

logger = logging.getLogger(__name__)


def main() -> None:
    configure_sys_logger()

    # --- Load Hydra config ---
    config_path = find_hydra_config()
    if config_path is None:
        logger.error("No config.yaml found. Exiting.")
        sys.exit(1)

    logger.info("Loading Hydra config from %s", config_path)
    hydra_config = OmegaConf.load(config_path)
    apply_model_penalty_multipliers(hydra_config)
    apply_dynamic_run_name_components(hydra_config)

    # Backup inputs
    copy_input_folder(config_path)
    save_resolved_config(hydra_config, Path("/results/inputs.yaml"))

    # --- Prepare wandb ---
    wandb_run = start_wandb_run(hydra_config)
    resolved = OmegaConf.to_container(hydra_config, resolve=True)
    logger.info("Hydra config (resolved):\n%s", OmegaConf.to_yaml(resolved))

    # --- Train + held-out evaluation/fine-tuning (shared with run_hpc.py) ---
    run_training(hydra_config, wandb_run)
    if wandb_run is not None:
        wandb_run.finish()

    logger.info("All done, goodbye")


if __name__ == "__main__":
    main()
