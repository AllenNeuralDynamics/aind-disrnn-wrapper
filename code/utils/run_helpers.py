"""Utility helpers for capsule run orchestration."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

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
    destination_root = Path("/results/input")
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_dir = destination_root / source_dir.name
    logger.info("Copying Hydra inputs from %s to %s", source_dir, destination_dir)
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)


def save_resolved_config(config: DictConfig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=str(destination), resolve=True)
    logger.info("Saved resolved config to %s", destination)


def persist_output(output_obj: Any, destination: Path) -> None:
    if output_obj is None:
        logger.info("Trainer returned no output payload; skipping persistence.")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(output_obj, "model_dump_json"):
        payload = output_obj.model_dump_json(indent=4)
    elif hasattr(output_obj, "model_dump"):
        payload = json.dumps(output_obj.model_dump(), indent=4)
    else:
        payload = json.dumps(output_obj, indent=4, default=_json_default)
    destination.write_text(payload)
    logger.info("Wrote trainer output to %s", destination)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def configure_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(levelname)s:%(asctime)s:%(filename)s:%(lineno)d:    %(message)s",
        datefmt="%Y-%m-%d %H-%M-%S",
    )


def start_wandb_run(
    config: DictConfig | Mapping[str, Any] | None,
) -> Optional[wandb.sdk.wandb_run.Run]:
    if config is None:
        return None

    params = OmegaConf.to_container(config, resolve=True)
    if not isinstance(params, dict):
        params = dict(params)

    run_config = params.pop("config", {})
    run = wandb.init(**params, config=run_config)
    run.config.update({"CO_COMPUTATION_ID": os.environ.get("CO_COMPUTATION_ID")})
    return run
