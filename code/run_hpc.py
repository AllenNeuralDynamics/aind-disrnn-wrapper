"""Entry point for disRNN wrapper experiments on HPC/SLURM (Beaker / AI Hub).

Loads the dispatcher's Hydra config via the sibling-repo relative path, prepares
it, starts a W&B run, then delegates the train + held-out evaluation/fine-tuning
to the shared ``training_runner.run_training`` (the same body the Code Ocean
entry point run_capsule.py uses), keeping the two paths in parity.
"""

import json
import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from training_runner import run_training
from utils.json_helpers import dictconfig_to_json
from utils.run_helpers import (
    apply_dynamic_run_name_components,
    apply_model_penalty_multipliers,
    configure_sys_logger,
    copy_inputs_for_run,
    copy_run_to_wandb,
    save_resolved_config,
    start_wandb_run,
)

logger = logging.getLogger(__name__)


@hydra.main(
    version_base=None,
    config_path="../../aind-disrnn-dispatcher/code/config",
    config_name="config",
)
def main(hydra_config: DictConfig) -> None:
    configure_sys_logger()

    # Prepare config (resolve disRNN penalty multipliers, set dynamic run-name
    # components) before saving inputs / starting W&B — same as run_capsule.py.
    apply_model_penalty_multipliers(hydra_config)
    apply_dynamic_run_name_components(hydra_config)

    hydra_runtime = HydraConfig.get().runtime
    run_dir = Path(hydra_runtime.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Backup inputs for reproducibility
    config_root = Path(__file__).parent.parent.parent / "aind-disrnn-dispatcher" / "code" / "config"
    copy_inputs_for_run(config_root, run_dir / "inputs")
    save_resolved_config(hydra_config, run_dir / "inputs.yaml")
    json_destination = run_dir / "inputs.json"
    json_destination.write_text(
        json.dumps(dictconfig_to_json(hydra_config), indent=2, sort_keys=True)
    )
    logger.info("Saved resolved config JSON to %s", json_destination)

    # --- Prepare wandb ---
    wandb_run = start_wandb_run(hydra_config)
    wandb_run_dir: Path | None = None
    if wandb_run is not None:
        wandb_run_dir = Path(wandb_run.dir).resolve()
        logger.info("wandb run directory: %s", wandb_run_dir)
        if hasattr(hydra_config, "model") and "output_dir" in hydra_config.model:
            hydra_config.model.output_dir = str(wandb_run_dir)
            logger.info(f"Model output_dir overridden to wandb dir {wandb_run_dir}")
        # Copy hydra run folder to wandb directory for full reproducibility
        copy_run_to_wandb(run_dir, wandb_run_dir)

    resolved_yaml = OmegaConf.to_yaml(hydra_config, resolve=True)
    logger.info("Hydra config (resolved):\n%s", resolved_yaml)

    # --- Train + held-out evaluation/fine-tuning (shared with run_capsule.py) ---
    run_training(hydra_config, wandb_run)
    if wandb_run is not None:
        wandb_run.finish()

    logger.info("All done, goodbye")


if __name__ == "__main__":
    main()
