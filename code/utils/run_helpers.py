"""Utility helpers for capsule run orchestration."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from omegaconf import DictConfig, OmegaConf

import wandb

logger = logging.getLogger(__name__)


def find_hydra_config() -> Path | None:
    """Locate the first config.yaml under /data/jobs (Code Ocean compatibility)."""

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
    """Copy Hydra inputs from Code Ocean mounted folders into /results."""

    source_dir = config_path.resolve().parents[1]
    destination_root = Path("/results/inputs")
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_dir = destination_root / source_dir.name
    logger.info("Copying Hydra inputs from %s to %s", source_dir, destination_dir)
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)


def copy_inputs_for_run(source_dir: Path, destination_dir: Path) -> None:
    """Copy input tree into the supplied destination (HPC workflow)."""

    destination_dir.mkdir(parents=True, exist_ok=True)
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


def _hardware_tags() -> list[str]:
    """Return tags describing the compute hardware for this run.

    If a GPU is visible via nvidia-smi, return ['gpu', '<model>'] where
    <model> is a normalized short GPU name (e.g. 'a100', 'h200', 'l40s',
    'titanxp'). Otherwise return ['cpu'].

    The list of GPU short tags below covers the Allen cluster GPU types.
    To inspect the current cluster GPU inventory, run:
        sinfo -o "%20N %10c %10m %25f %10G"
    Note that the SLURM GRES name (e.g. 'a100sx') may differ from what
    nvidia-smi reports (e.g. 'NVIDIA A100-SXM...'); detection is based on
    nvidia-smi, so SXM-variant A100s are normalized to 'a100' like other
    A100 nodes.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            name = out.stdout.strip().splitlines()[0].strip().lower()
            # Normalize common Allen GPU types into short tags.
            for short in ("h200", "h100", "a100", "l40s", "v100", "titanxp", "titanx", "1080ti"):
                if short in name.replace(" ", ""):
                    return ["gpu", short]
            # Fallback: collapse to alphanumerics.
            model = "".join(c for c in name if c.isalnum()) or "unknown"
            return ["gpu", model]
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass
    return ["cpu"]


def start_wandb_run(
    hydra_config: DictConfig,
) -> Optional[wandb.sdk.wandb_run.Run]:
    dict_config = OmegaConf.to_container(hydra_config, resolve=True)

    # Extract & remove original tags
    wandb_cfg = dict_config.get("wandb", {})
    base_tags = wandb_cfg.pop("tags", []) or []
    extra_tags = [hydra_config.data.type, hydra_config.model.type, *_hardware_tags()]

    # Init wandb with merged tags
    run = wandb.init(
        **wandb_cfg,
        config={k: dict_config[k] for k in ("data", "model", "meta") if k in dict_config},
        tags=base_tags + extra_tags,
    )

    # Tag sweep-associated runs so they can be filtered in the W&B UI.
    if getattr(run, "sweep_id", None):
        run.tags = tuple(run.tags) + ("sweep",)

    # System environment variable for CO 
    run.config.update({"CO_COMPUTATION_ID": os.environ.get("CO_COMPUTATION_ID")})
    return run


def copy_run_to_wandb(run_dir: Path, wandb_dir: Path) -> None:
    """Copy Hydra run directory contents into the wandb run folder."""

    resolved_run_dir = run_dir.resolve()
    resolved_wandb_dir = wandb_dir.resolve()

    if resolved_run_dir == resolved_wandb_dir:
        logger.info("Hydra run dir and wandb dir are identical, skipping copy")
        return

    resolved_wandb_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Copying Hydra run dir %s to wandb dir %s", resolved_run_dir, resolved_wandb_dir)

    for item in resolved_run_dir.iterdir():
        destination = resolved_wandb_dir / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
        else:
            shutil.copy2(item, destination)
