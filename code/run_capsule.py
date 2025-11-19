"""Entry point for disRNN wrapper experiments."""

import logging
import time
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from capsule_core.interfaces import DatasetLoader, ModelTrainer
from capsule_core.types import TrainerResult
from utils.run_helpers import (
    configure_logger,
    copy_input_folder,
    find_hydra_config,
    persist_output,
    save_resolved_config,
    start_wandb_run,
)

logger = logging.getLogger(__name__)


def main() -> None:
    configure_logger()

    # --- Load Hydra config ---
    config_path = find_hydra_config()
    if config_path is None:
        logger.error("No config.yaml found. Exiting.")
        sys.exit(1)

    logger.info("Loading Hydra config from %s", config_path)
    hydra_config = OmegaConf.load(config_path)
    
    #  Backup inputs
    copy_input_folder(config_path)
    save_resolved_config(hydra_config, Path("/results/inputs.yaml"))

    #  Sync seeds, if needed
    seed = hydra_config.get("seed")
    if seed is None:
        seed = int(time.time())
        logger.warning("No seed provided in config; using fallback seed %s", seed)
    hydra_config.seed = seed
    hydra_config.data.seed = hydra_config.data.get("seed", seed)
    hydra_config.model.seed = hydra_config.model.get("seed", seed)
    
    # --- Prepare wandb ---
    # Note: wandb_run doesn't have a run name and config at this point,
    #       You should set them in your model trainer.
    wandb_run = start_wandb_run(hydra_config.wandb)
    loggers = {"wandb": wandb_run} if wandb_run is not None else None

    # --- Load data ---
    dataset_loader: DatasetLoader = instantiate(hydra_config.data)
    dataset_bundle = dataset_loader.load()
    logger.info("Loaded dataset bundle with metadata: %s", dataset_bundle.metadata)

    # --- Train model ---
    model_trainer: ModelTrainer = instantiate(hydra_config.model)
    trainer_result: TrainerResult = model_trainer.fit(dataset_bundle, loggers=loggers)
    if wandb_run is not None:
        wandb_run.finish()

    # --- Save outputs ---
    persist_output(trainer_result.output, Path("/results/outputs.json"))
    logger.info("All done, goodbye")


if __name__ == "__main__":
    main()
