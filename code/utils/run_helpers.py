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


def resolve_disrnn_penalties(penalties_cfg: Any) -> dict[str, Any]:
    """Resolve optional per-penalty multiplier fields for disRNN configs.

    Any ``<penalty_name>_multiplier`` entry scales the corresponding base
    penalty. If the base penalty is omitted, ``beta`` is used as the default.
    Multiplier keys are removed from the returned dictionary so repeated calls
    are idempotent.
    """
    penalties = (
        OmegaConf.to_container(penalties_cfg, resolve=True)
        if isinstance(penalties_cfg, DictConfig)
        else dict(penalties_cfg)
    )
    resolved = dict(penalties or {})
    beta = resolved.get("beta")
    multiplier_suffix = "_multiplier"

    for key in list(resolved.keys()):
        if not key.endswith(multiplier_suffix):
            continue

        base_key = key[: -len(multiplier_suffix)]
        if base_key == "beta":
            continue

        multiplier = float(resolved.pop(key))
        base_value = resolved.get(base_key, beta)
        if base_value is None:
            raise ValueError(
                f"Cannot resolve {base_key}: set either {base_key} or beta before "
                f"using {key}."
            )
        resolved[base_key] = float(base_value) * multiplier

    return resolved


def _append_multisubject_suffix(component: Any, *, enabled: bool) -> Any:
    """Append a multisubject suffix to a run-name component when needed."""
    if not enabled or component is None:
        return component
    component_str = str(component)
    if "multisubject" in component_str.lower():
        return component_str
    return f"{component_str}_multisubject"


def apply_dynamic_run_name_components(hydra_config: DictConfig) -> None:
    """Mutate run-name components based on config-driven runtime mode.

    This keeps the YAML simple: base run-name components stay generic, and
    multisubject mode is reflected automatically in the resolved config and the
    downstream W&B run name.
    """
    data_cfg = getattr(hydra_config, "data", None)
    if data_cfg is not None and "run_name_component" in data_cfg:
        data_cfg.run_name_component = _append_multisubject_suffix(
            data_cfg.run_name_component,
            enabled=bool(getattr(data_cfg, "multisubject", False)),
        )

    model_cfg = getattr(hydra_config, "model", None)
    architecture_cfg = getattr(model_cfg, "architecture", None) if model_cfg is not None else None
    if model_cfg is not None and "run_name_component" in model_cfg:
        model_cfg.run_name_component = _append_multisubject_suffix(
            model_cfg.run_name_component,
            enabled=bool(getattr(architecture_cfg, "multisubject", False)),
        )


def apply_model_penalty_multipliers(hydra_config: DictConfig) -> None:
    """Resolve disRNN penalty multipliers into effective numeric penalties."""
    model_cfg = getattr(hydra_config, "model", None)
    if model_cfg is None or getattr(model_cfg, "type", None) != "disrnn":
        return

    penalties_cfg = getattr(model_cfg, "penalties", None)
    if penalties_cfg is None:
        return

    model_cfg.penalties = OmegaConf.create(resolve_disrnn_penalties(penalties_cfg))


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
    wandb_cfg = dict_config.get("wandb", {})
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
