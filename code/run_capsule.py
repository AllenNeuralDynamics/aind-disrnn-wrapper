"""Entry point for disRNN wrapper experiments (Code Ocean pipeline)."""

import logging
import sys
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from base.interfaces import DatasetLoader, ModelTrainer
from utils.run_helpers import (
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

    # Backup inputs
    copy_input_folder(config_path)
    save_resolved_config(hydra_config, Path("/results/inputs.yaml"))

    # --- Prepare wandb ---
    wandb_run = start_wandb_run(hydra_config)
    resolved = OmegaConf.to_container(hydra_config, resolve=True)
    logger.info("Hydra config (resolved):\n%s", OmegaConf.to_yaml(resolved))

    # --- Load data ---
    dataset_loader: DatasetLoader = instantiate(hydra_config.data)
    dataset_bundle = dataset_loader.load()
    logger.info("Loaded dataset bundle with metadata: %s", dataset_bundle.metadata)

    # --- Train model ---
    model_trainer: ModelTrainer = instantiate(hydra_config.model)
    loggers = {"wandb": wandb_run} if wandb_run is not None else None
    model_trainer.fit(dataset_bundle, loggers=loggers)
    if wandb_run is not None:
        wandb_run.finish()

    logger.info("All done, goodbye")


if __name__ == "__main__":
    main()
