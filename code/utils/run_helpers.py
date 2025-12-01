"""Utility helpers for capsule run orchestration."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

import wandb

logger = logging.getLogger(__name__)


def find_hydra_config() -> Path | None:
    """Locate the first config.yaml under /data/jobs."""

    candidates = list(Path("/data/jobs").rglob("config.yaml"))
    if not candidates:
        logger.warning("No config.yaml found under /data/jobs/")
        return None
    if len(candidates) > 1:
        logger.warning(
            "Multiple config.yaml files found: %s. Using the first one.", candidates
        )
    return candidates[0]


def copy_input_folder(config_path: Path) -> None:
    source_dir = config_path.resolve().parents[1]
    destination_root = Path("/results/inputs")
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_dir = destination_root / source_dir.name
    logger.info("Copying Hydra inputs from %s to %s", source_dir, destination_dir)
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)


def save_resolved_config(config: DictConfig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=str(destination), resolve=True)
    logger.info("Saved resolved config to %s", destination)


def configure_sys_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(levelname)s:%(asctime)s:%(filename)s:%(lineno)d:    %(message)s",
        datefmt="%Y-%m-%d %H-%M-%S",
    )


def start_wandb_run(
    hydra_config: DictConfig,
) -> Optional[wandb.sdk.wandb_run.Run]:
    dict_config = OmegaConf.to_container(hydra_config, resolve=True)

    # Extract & remove original tags
    base_tags = wandb_cfg.pop("tags", []) or []
    extra_tags = [hydra_config.data.type, hydra_config.model.type]

    # Init wandb with merged tags
    run = wandb.init(
        **wandb_cfg,
        config={k: dict_config[k] for k in ("data", "model") if k in dict_config},
        tags=base_tags + extra_tags,
    )
    
    # System environment variable for CO 
    run.config.update({"CO_COMPUTATION_ID": os.environ.get("CO_COMPUTATION_ID")})
    return run
