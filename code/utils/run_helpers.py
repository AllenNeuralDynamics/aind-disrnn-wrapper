"""Utility helpers for capsule run orchestration."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
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


def resolve_heldout_test_likelihood(summary: Any) -> Optional[float]:
    """Return the held-out test likelihood from an evaluation summary, or None.

    Fresh evaluations (``evaluate_*_on_heldout_subjects``) report the value under
    ``test_likelihood``; the dedup-hit path reuses a checkpoint summary keyed
    ``heldout_test_likelihood``. Returns ``None`` for non-dict summaries or when
    neither key is present (e.g. a failure-fallback summary).
    """
    if not isinstance(summary, dict):
        return None
    value = summary.get("test_likelihood")
    if value is None:
        value = summary.get("heldout_test_likelihood")
    return value


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


def _code_versions() -> dict[str, Optional[str]]:
    """Resolved commit SHAs of the wrapper + dispatcher repos, for reproducibility.

    Stamped into the W&B run config so every run is traceable to the exact code it
    ran. Prefers the WRAPPER_COMMIT / DISPATCHER_COMMIT env vars (e.g. set by
    beaker/entrypoint.sh) and falls back to `git rev-parse` on the on-disk repos;
    None if neither is available (so it's harmless on any platform).
    """

    def sha(repo_dir: Path, env_var: str) -> Optional[str]:
        if os.environ.get(env_var):
            return os.environ[env_var]
        try:
            out = subprocess.run(
                ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
        return None

    wrapper_dir = Path(__file__).resolve().parents[2]  # .../aind-disrnn-wrapper
    dispatcher_dir = wrapper_dir.parent / "aind-disrnn-dispatcher"
    return {
        "wrapper_commit": sha(wrapper_dir, "WRAPPER_COMMIT"),
        "dispatcher_commit": sha(dispatcher_dir, "DISPATCHER_COMMIT"),
    }


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

    # Reproducibility / provenance metadata stamped into the W&B config.
    # allow_val_change so resuming a run after a code bump (WANDB_RESUME=allow with
    # a new WRAPPER_COMMIT) updates the SHA instead of crashing on a config conflict.
    run.config.update({
        "CO_COMPUTATION_ID": os.environ.get("CO_COMPUTATION_ID"),
        **_code_versions(),
    }, allow_val_change=True)
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
