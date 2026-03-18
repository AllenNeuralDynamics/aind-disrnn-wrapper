
"""Entry point for disRNN wrapper experiments."""

# Set PYTHONPATH for correct module resolution
import os
import sys
pythonpath = "/root/capsule/src/disentangled-rnns"
if pythonpath not in sys.path:
    sys.path.insert(0, pythonpath)

import logging
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf

from base.interfaces import DatasetLoader, ModelTrainer
from utils.baseline_rl_evaluation import evaluate_baseline_rl_on_heldout_subjects
from utils.disrnn_evaluation import (
    HeldoutEvalConfig,
    evaluate_disrnn_on_heldout_subjects,
    load_disrnn_heldout_subject_data,
)
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

    heldout_cfg = HeldoutEvalConfig.from_data_cfg(
        hydra_config.data,
        default_example_max_subjects=int(getattr(hydra_config, "example_max_subjects", 6)),
    )

    # Append resolved subject IDs to the wandb run name so rank-based slices
    # (where subject_ids is null in config) are still identifiable in the UI.
    resolved_subject_ids = dataset_bundle.metadata.get("subject_ids")
    if wandb_run is not None and resolved_subject_ids is not None:
        ids_str = "-".join(str(s) for s in sorted(resolved_subject_ids))
        new_name = f"{wandb_run.name}_subjs-{ids_str}" if wandb_run.name else f"subs-{ids_str}"
        wandb_run.name = new_name
        wandb_run.config.update({"resolved_subject_ids": list(resolved_subject_ids)})
        logger.info("Updated wandb run name to: %s", new_name)

    heldout_test_data = None
    if (
        getattr(hydra_config.model, "type", None) == "disrnn"
        and hasattr(hydra_config.data, "mature_only")
        and heldout_cfg.enabled
    ):
        try:
            heldout_test_data = load_disrnn_heldout_subject_data(heldout_cfg)
        except Exception as exc:
            logger.warning(
                "Preloading held-out test data failed; evaluation will fall back to lazy loading: %s",
                exc,
            )

    # --- Train model ---
    model_trainer: ModelTrainer = instantiate(
        hydra_config.model,
        heldout_data=heldout_test_data,
    )
    loggers = {"wandb": wandb_run} if wandb_run is not None else None
    output = model_trainer.fit(dataset_bundle, loggers=loggers)
    if heldout_test_data is None:
        heldout_test_data = getattr(model_trainer, "heldout_data", None)

    checkpoint_heldout_summaries = []
    if isinstance(output, dict):
        raw_checkpoint_heldout = output.get("heldout_test_checkpoints", [])
        if isinstance(raw_checkpoint_heldout, list):
            checkpoint_heldout_summaries = [
                item for item in raw_checkpoint_heldout if isinstance(item, dict)
            ]

    if hasattr(hydra_config.data, "mature_only") and heldout_cfg.enabled:
        model_type = getattr(hydra_config.model, "type", None)
        heldout_summary = None
        try:
            if model_type == "disrnn":
                training_cfg = getattr(hydra_config.model, "training", {})
                checkpoint_keep_media_files = bool(
                    getattr(training_cfg, "checkpoint_keep_media_files", True)
                )
                total_steps = int(
                    getattr(getattr(hydra_config.model, "training", {}), "n_steps", 0)
                )
                evaluated_checkpoint_steps = {
                    int(item.get("step", -1))
                    for item in checkpoint_heldout_summaries
                    if isinstance(item, dict)
                }
                if total_steps in evaluated_checkpoint_steps:
                    logger.info(
                        "Skipping duplicate final held-out eval; step_%s already evaluated in checkpoint loop.",
                        total_steps,
                    )
                    heldout_summary = next(
                        (
                            item
                            for item in checkpoint_heldout_summaries
                            if int(item.get("step", -1)) == total_steps
                        ),
                        None,
                    )
                else:
                    heldout_summary = evaluate_disrnn_on_heldout_subjects(
                        hydra_config,
                        wandb_run=wandb_run,
                        output_subdir=f"heldout_test/checkpoints/step_{total_steps}",
                        log_to_wandb=False,
                        heldout_data=heldout_test_data,
                    )
                    if heldout_summary is not None and wandb_run is not None:
                        warmup_steps = int(getattr(training_cfg, "n_warmup_steps", 0))
                        wandb_step = warmup_steps + total_steps
                        wandb_run.log(
                            {
                                "checkpoint/heldout_test_likelihood": float(
                                    heldout_summary["test_likelihood"]
                                ),
                                "checkpoint/step": total_steps,
                            },
                            step=wandb_step,
                        )
                        try:
                            import wandb

                            trial_plot_paths = heldout_summary.get("plots", {}).get(
                                "latents_over_trials_examples", []
                            )
                            space_plot_paths = heldout_summary.get("plots", {}).get(
                                "latents_in_space_examples", []
                            )
                            checkpoint_plot_payload = {}
                            if trial_plot_paths:
                                checkpoint_plot_payload[
                                    "checkpoint/heldout/latents_over_trials_examples"
                                ] = [wandb.Image(str(path)) for path in trial_plot_paths]
                            if space_plot_paths:
                                checkpoint_plot_payload[
                                    "checkpoint/heldout/latents_in_space_examples"
                                ] = [wandb.Image(str(path)) for path in space_plot_paths]
                            if checkpoint_plot_payload:
                                wandb_run.log(checkpoint_plot_payload, step=wandb_step)
                                if not checkpoint_keep_media_files:
                                    for path in trial_plot_paths + space_plot_paths:
                                        try:
                                            Path(str(path)).unlink(missing_ok=True)
                                        except Exception as path_exc:
                                            logger.warning(
                                                "Failed to remove final held-out media file %s: %s",
                                                path,
                                                path_exc,
                                            )
                                    plots = heldout_summary.get("plots", {})
                                    if isinstance(plots, dict):
                                        plots["latents_over_trials_examples"] = []
                                        plots["latents_in_space_examples"] = []
                        except Exception as exc:
                            logger.warning(
                                "Final held-out image logging failed for step=%s: %s",
                                total_steps,
                                exc,
                            )
            elif model_type == "baseline_rl":
                heldout_summary = evaluate_baseline_rl_on_heldout_subjects(
                    hydra_config,
                    wandb_run=wandb_run,
                )
        except Exception as exc:
            logger.warning("Held-out evaluation failed and will be skipped: %s", exc)
            heldout_summary = {
                "enabled": True,
                "evaluation_failed": True,
                "error": str(exc),
            }

        if heldout_summary is not None:
            output["heldout_test"] = heldout_summary

    if wandb_run is not None:
        wandb_run.finish()

    logger.info("All done, goodbye")


if __name__ == "__main__":
    main()