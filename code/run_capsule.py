"""Entry point for disRNN wrapper experiments."""

import logging
import sys
import time
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from base.interfaces import DatasetLoader, ModelTrainer
from utils.load_mice_snapshot import load_mice_snapshot
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

    # When using snapshot-based loading, surface the key selection parameters
    # so they are clearly visible at the top of the run log.
    data_cfg = hydra_config.data
    if hasattr(data_cfg, "mature_only"):
        logger.info(
            "Snapshot data loading: mature_only=%s, subject_ids=%s, start=%s, end=%s",
            data_cfg.mature_only,
            getattr(data_cfg, "subject_ids", None),
            getattr(data_cfg, "subject_start", None),
            getattr(data_cfg, "subject_end", None),
        )

    dataset_bundle = dataset_loader.load()
    logger.info("Loaded dataset bundle with metadata: %s", dataset_bundle.metadata)

    # Append resolved subject IDs to the wandb run name so rank-based slices
    # (where subject_ids is null in config) are still identifiable in the UI.
    resolved_subject_ids = dataset_bundle.metadata.get("subject_ids")
    if wandb_run is not None and resolved_subject_ids is not None:
        ids_str = "-".join(str(s) for s in sorted(resolved_subject_ids))
        new_name = f"{wandb_run.name}_subjs-{ids_str}" if wandb_run.name else f"subs-{ids_str}"
        wandb_run.name = new_name
        wandb_run.config.update({"resolved_subject_ids": list(resolved_subject_ids)})
        logger.info("Updated wandb run name to: %s", new_name)

    # --- Train model ---
    model_trainer: ModelTrainer = instantiate(hydra_config.model)
    loggers = {"wandb": wandb_run} if wandb_run is not None else None
    output = model_trainer.fit(dataset_bundle, loggers=loggers)
    if wandb_run is not None:
        wandb_run.finish()

    logger.info("All done, goodbye")


if __name__ == "__main__":
    main()