"""Utility helpers for capsule run orchestration."""

from __future__ import annotations

import logging
import os
import random
import shutil
import subprocess
import sys
import time
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


def _init_wandb_with_fallback(init_kwargs: dict) -> wandb.sdk.wandb_run.Run:
    """``wandb.init`` with bounded jittered retries, then an offline fallback.

    Online run creation hits the W&B backend; when many tasks launch at once (a
    grid sweep starts all its Beaker tasks together) that handshake can time out.
    Retry a few times with *random* backoff — the jitter de-correlates simultaneous
    retries so they don't all collide again — then fall back to ``mode="offline"``
    so training never blocks on W&B. Offline runs log locally and can be synced to
    W&B afterward (``wandb sync``). If ``WANDB_MODE`` is already offline/disabled,
    the first attempt just succeeds locally and the retry/fallback never engages.
    Knobs: ``WANDB_INIT_ATTEMPTS`` (default 3), ``WANDB_INIT_TIMEOUT`` (default 120).
    """
    attempts = int(os.environ.get("WANDB_INIT_ATTEMPTS", "3"))
    timeout = int(os.environ.get("WANDB_INIT_TIMEOUT", "120"))
    last_exc: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return wandb.init(**init_kwargs, settings=wandb.Settings(init_timeout=timeout))
        except Exception as exc:  # online run-creation failed (e.g. CommError timeout)
            last_exc = exc
            try:
                wandb.teardown()  # reset any half-started wandb-core service before retry
            except Exception:
                pass
            if attempt < attempts:
                delay = random.uniform(5.0, 30.0) * attempt  # jittered, grows per attempt
                logger.warning(
                    "wandb.init failed (attempt %d/%d): %s; retrying in %.0fs",
                    attempt, attempts, exc, delay,
                )
                time.sleep(delay)
    logger.warning(
        "wandb.init failed after %d attempts (%s); falling back to OFFLINE mode "
        "(run logged locally under the wandb dir; sync to W&B later)",
        attempts, last_exc,
    )
    return wandb.init(
        **init_kwargs, settings=wandb.Settings(init_timeout=timeout), mode="offline"
    )


def start_wandb_run(
    hydra_config: DictConfig,
) -> Optional[wandb.sdk.wandb_run.Run]:
    dict_config = OmegaConf.to_container(hydra_config, resolve=True)

    # Extract & remove original tags
    wandb_cfg = dict_config.get("wandb", {})
    base_tags = wandb_cfg.pop("tags", []) or []
    extra_tags = [hydra_config.data.type, hydra_config.model.type, *_hardware_tags()]

    # Init wandb with merged tags, retrying then falling back to offline so training
    # never blocks on a flaky online run-creation handshake.
    run = _init_wandb_with_fallback({
        **wandb_cfg,
        "config": {k: dict_config[k] for k in ("data", "model", "meta") if k in dict_config},
        "tags": base_tags + extra_tags,
    })

    # Provenance stamped into the W&B config (allow_val_change so a resume after a
    # code/ref bump updates values instead of raising ConfigError). Two layers:
    #  - platform-native cross-ref ids (alongside CO_COMPUTATION_ID): Beaker exp/job +
    #    code SHAs — for jumping to the exact run on each platform.
    #  - `meta`: our portable, human-readable system (study / variant / launch_id /
    #    label / note / config_hash), set by launch_beaker_resumable.py via DISRNN_META_*
    #    env — consistent across CO / Beaker / AI1 HPC. Merge-safe with any config `meta`.
    #    `note` is a free-text "why this run exists + what we want to learn" so humans and
    #    agents can read intent straight from the run record.
    meta_env = {
        "study": os.environ.get("DISRNN_META_STUDY"),
        "variant": os.environ.get("DISRNN_META_VARIANT"),
        "launch_id": os.environ.get("DISRNN_META_LAUNCH_ID"),
        "label": os.environ.get("DISRNN_META_LABEL"),
        "note": os.environ.get("DISRNN_META_NOTE"),
        "config_hash": os.environ.get("DISRNN_META_CONFIG_HASH"),
    }
    meta = {**(run.config.get("meta") or {}), **{k: v for k, v in meta_env.items() if v}}
    provenance = {
        "CO_COMPUTATION_ID": os.environ.get("CO_COMPUTATION_ID"),
        "BEAKER_EXPERIMENT_ID": os.environ.get("BEAKER_EXPERIMENT_ID"),
        "BEAKER_JOB_ID": os.environ.get("BEAKER_JOB_ID"),
        **_code_versions(),
    }
    if meta:
        provenance["meta"] = meta
    run.config.update(provenance, allow_val_change=True)
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


def compute_bottleneck_sparsity_metrics(
    params: Any,
    open_thresh: float = 0.1,
    closed_thresh: float = 0.9,
) -> dict[str, float]:
    """Compute real-time bottleneck-sparsity scalars for a Multisubject DisRNN.

    Wraps the upstream ``multisubject_disrnn.get_auxiliary_metrics`` (which returns
    ``total_sigma`` plus per-family open/closed *counts*) and augments it with:

    - an isolated ``update_net_latent`` bottleneck breakout (mean sigma + open/closed
      counts). This is the channel that ``update_net_latent_penalty_multiplier``
      directly targets, so it is the primary readout for the beta-scan study; the
      upstream ``update_bottlenecks_*`` aggregates subj+obs+latent and hides it.
    - a ``bottlenecks/`` W&B namespace and fraction-open values for each family so a
      run's interaction sparsity is visible in real time on the W&B dashboard.

    Returns a flat dict of floats/ints keyed under ``bottlenecks/*``. On any failure
    (unexpected param layout, non-multisubject params) it returns ``{}`` so logging
    never breaks training.
    """
    try:
        import numpy as np
        from disentangled_rnns.library import disrnn as _disrnn
        from disentangled_rnns.library import (
            multisubject_disrnn as _ms_disrnn,
        )

        out: dict[str, float] = {}

        # Upstream aggregate metrics (total_sigma + per-family open/closed counts).
        try:
            aux = _ms_disrnn.get_auxiliary_metrics(
                params, open_thresh=open_thresh, closed_thresh=closed_thresh
            )
            for k, v in aux.items():
                out[f"bottlenecks/{k}"] = float(v)
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_auxiliary_metrics failed: %s", exc)

        # Isolated per-family sigmas, incl. the update_net_latent channel the
        # multiplier drives. Param module key is 'multisubject_dis_rnn'.
        module_params = None
        if isinstance(params, dict):
            module_params = params.get("multisubject_dis_rnn")
        if module_params is not None:
            families = {
                "latent": "latent_sigma_params",
                "update_net_subj": "update_net_subj_sigma_params",
                "update_net_obs": "update_net_obs_sigma_params",
                "update_net_latent": "update_net_latent_sigma_params",
                "choice_net_subj": "choice_net_subj_sigma_params",
                "choice_net_latent": "choice_net_latent_sigma_params",
            }
            for fam, key in families.items():
                if key not in module_params:
                    continue
                sig = np.array(
                    _disrnn.reparameterize_sigma(module_params[key])
                ).ravel()
                n = int(sig.size)
                if n == 0:
                    continue
                n_open = int(np.sum(sig < open_thresh))
                n_closed = int(np.sum(sig > closed_thresh))
                out[f"bottlenecks/{fam}_mean_sigma"] = float(np.mean(sig))
                out[f"bottlenecks/{fam}_min_sigma"] = float(np.min(sig))
                out[f"bottlenecks/{fam}_n_open"] = n_open
                out[f"bottlenecks/{fam}_n_closed"] = n_closed
                out[f"bottlenecks/{fam}_frac_open"] = float(n_open) / n

        return out
    except Exception as exc:  # noqa: BLE001
        logger.warning("compute_bottleneck_sparsity_metrics failed: %s", exc)
        return {}
