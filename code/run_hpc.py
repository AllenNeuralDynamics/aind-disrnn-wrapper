"""Entry point for disRNN wrapper experiments on HPC/SLURM."""

import json
import logging
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from .base.interfaces import DatasetLoader, ModelTrainer
from .utils.json_helpers import dictconfig_to_json
from .utils.run_helpers import (
    configure_sys_logger,
    copy_inputs_for_run,
    save_resolved_config,
    start_wandb_run,
)

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(hydra_config: DictConfig) -> None:
    configure_sys_logger()

    run_dir = Path.cwd()

    # Backup inputs for reproducibility
    config_root = Path(__file__).parent.parent / "config"
    copy_inputs_for_run(config_root, run_dir / "inputs")
    save_resolved_config(hydra_config, run_dir / "inputs.yaml")
    json_destination = run_dir / "inputs.json"
    json_destination.write_text(
        json.dumps(dictconfig_to_json(hydra_config), indent=2, sort_keys=True)
    )
    logger.info("Saved resolved config JSON to %s", json_destination)

    # --- Prepare wandb ---
    wandb_run = start_wandb_run(hydra_config)
    resolved_yaml = OmegaConf.to_yaml(hydra_config, resolve=True)
    logger.info("Hydra config (resolved):\n%s", resolved_yaml)

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
