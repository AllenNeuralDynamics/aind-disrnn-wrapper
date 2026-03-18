from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Mapping
from dataclasses import asdict

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf

import aind_disrnn_utils.data_loader as dl
import types
from disentangled_rnns.library import disrnn, plotting, rnn_utils

from base.interfaces import ModelTrainer
from base.types import DatasetBundle
from utils.disrnn_evaluation import plot_disrnn_examples_for_split
from utils.disrnn_evaluation import (
    HeldoutEvalConfig,
    evaluate_disrnn_on_heldout_subjects,
    load_disrnn_heldout_subject_data,
)

logger = logging.getLogger(__name__)


def _to_dict(config: Mapping[str, Any] | DictConfig) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


class DisrnnTrainer(ModelTrainer):
    """Trainer that reproduces the legacy disRNN pipeline."""

    def __init__(
        self,
        architecture: Mapping[str, Any] | DictConfig,
        penalties: Mapping[str, Any] | DictConfig,
        training: Mapping[str, Any] | DictConfig,
        heldout_data: dict[str, Any] | None = None,
        output_dir: str = "/results/outputs",
        seed: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__(seed=seed)
        self.architecture = _to_dict(architecture)
        self.penalties = _to_dict(penalties)
        self.training = _to_dict(training)
        self.heldout_data = heldout_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        bundle: DatasetBundle,
        loggers: dict[str, Any] | None = None,
    ):
        metadata = dict(bundle.metadata)
        ignore_policy = metadata.get("ignore_policy", "exclude")

        wandb_run = None
        if loggers and "wandb" in loggers:
            wandb_run = loggers["wandb"]

        dataset = bundle.extras.get("dataset") if bundle.extras else None
        if dataset is None:
            raise ValueError(
                "Dataset bundle must include the constructed disRNN dataset."
            )

        dataset_train = bundle.train_set
        dataset_eval = bundle.eval_set
        if dataset_train is None or dataset_eval is None:
            raise ValueError("Dataset bundle must include train and eval splits.")

        output = {
            "num_trials": metadata.get("num_trials"),
            "num_sessions": metadata.get("num_sessions"),
        }

        logger.info(
            "Dataset details: input %s, output %s", dataset._xs.shape, dataset._ys.shape
        )
        logger.info(
            "Train/Eval shapes: train=%s eval=%s",
            dataset_train._ys.shape,
            dataset_eval._ys.shape,
        )

        key = jax.random.PRNGKey(self.seed)
        warmup_key, training_key = jax.random.split(key)
        output["random_key"] = [int(x) for x in np.asarray(key).reshape(-1)]

        args = types.SimpleNamespace(
            num_latents=self.architecture["latent_size"],
            update_net_n_units_per_layer=self.architecture[
                "update_net_n_units_per_layer"
            ],
            update_net_n_layers=self.architecture["update_net_n_layers"],
            choice_net_n_units_per_layer=self.architecture[
                "choice_net_n_units_per_layer"
            ],
            choice_net_n_layers=self.architecture["choice_net_n_layers"],
            activation=self.architecture["activation"],
            latent_penalty=self.penalties["latent_penalty"],
            choice_net_latent_penalty=self.penalties["choice_net_latent_penalty"],
            update_net_obs_penalty=self.penalties["update_net_obs_penalty"],
            update_net_latent_penalty=self.penalties["update_net_latent_penalty"],
            max_grad_norm=self.training["max_grad_norm"],
            n_steps=self.training["n_steps"],
            n_warmup_steps=self.training["n_warmup_steps"],
            learning_rate=self.training["lr"],
            loss=self.training["loss"],
            loss_param=self.training["loss_param"],
            checkpoint_every_n_steps=int(self.training.get("checkpoint_every_n_steps", 0)),
            checkpoint_eval_on_eval_split=bool(
                self.training.get("checkpoint_eval_on_eval_split", True)
            ),
            checkpoint_log_eval_to_wandb=bool(
                self.training.get("checkpoint_log_eval_to_wandb", True)
            ),
            checkpoint_plot_split_examples_every_n=int(
                self.training.get("checkpoint_plot_split_examples_every_n", 5)
            ),
            checkpoint_save_output_df_every_n=int(
                self.training.get("checkpoint_save_output_df_every_n", 0)
            ),
            checkpoint_log_split_examples_to_wandb=bool(
                self.training.get("checkpoint_log_split_examples_to_wandb", True)
            ),
            checkpoint_plot_choice_rule=bool(
                self.training.get("checkpoint_plot_choice_rule", False)
            ),
            checkpoint_plot_update_rules=bool(
                self.training.get("checkpoint_plot_update_rules", False)
            ),
            checkpoint_keep_media_files=bool(
                self.training.get("checkpoint_keep_media_files", True)
            ),
            checkpoint_run_heldout_eval=bool(
                self.training.get("checkpoint_run_heldout_eval", True)
            ),
            checkpoint_include_final_in_heldout=bool(
                self.training.get("checkpoint_include_final_in_heldout", False)
            ),
            plot_choice_rule=bool(self.training.get("plot_choice_rule", False)),
            plot_update_rules=bool(self.training.get("plot_update_rules", False)),
        )

        logger.info(f"max_grad_norm = {args.max_grad_norm}")

        output_size = 2 if ignore_policy == "exclude" else 3
        disrnn_config = disrnn.DisRnnConfig(
            obs_size=dataset._xs.shape[2],
            output_size=output_size,
            x_names=dataset.x_names,
            y_names=dataset.y_names,
            latent_size=args.num_latents,
            update_net_n_units_per_layer=args.update_net_n_units_per_layer,
            update_net_n_layers=args.update_net_n_layers,
            choice_net_n_units_per_layer=args.choice_net_n_units_per_layer,
            choice_net_n_layers=args.choice_net_n_layers,
            activation=args.activation,
            noiseless_mode=False,
            latent_penalty=args.latent_penalty,
            choice_net_latent_penalty=args.choice_net_latent_penalty,
            update_net_obs_penalty=args.update_net_obs_penalty,
            update_net_latent_penalty=args.update_net_latent_penalty,
        )

        noiseless_network = copy.deepcopy(disrnn_config)
        noiseless_network.latent_penalty = 0
        noiseless_network.choice_net_latent_penalty = 0
        noiseless_network.update_net_obs_penalty = 0
        noiseless_network.update_net_latent_penalty = 0
        noiseless_network.l2_scale = 0
        noiseless_network.noiseless_mode = True

        logger.info("Running warmup training phase")
        warmup_start = time.time()
        params, warmup_opt_state, warmup_losses = rnn_utils.train_network(
            lambda: disrnn.HkDisentangledRNN(noiseless_network),
            dataset_train,
            dataset_eval,
            opt=optax.adam(args.learning_rate),
            loss=args.loss,
            loss_param=args.loss_param,
            n_steps=args.n_warmup_steps,
            max_grad_norm=args.max_grad_norm,
            random_key=warmup_key,
            report_progress_by="wandb",
            wandb_run=wandb_run,
        )
        warmup_duration = time.time() - warmup_start
        warmup_path = self._plot_losses(
            warmup_losses,
            title="Loss over warmup training",
            output_name="warmup_validation.png",
        )
        if wandb_run is not None:
            wandb_run.log({"fig/warmup_loss_curve": wandb.Image(str(warmup_path))})

        logger.info("Running full training phase")
        start = time.time()

        if args.checkpoint_every_n_steps < 0:
            raise ValueError("training.checkpoint_every_n_steps must be >= 0")
        if args.checkpoint_plot_split_examples_every_n < 0:
            raise ValueError("training.checkpoint_plot_split_examples_every_n must be >= 0")
        if args.checkpoint_save_output_df_every_n < 0:
            raise ValueError("training.checkpoint_save_output_df_every_n must be >= 0")

        checkpoint_records: list[dict[str, Any]] = []
        checkpoint_heldout_summaries: list[dict[str, Any]] = []
        if args.checkpoint_every_n_steps == 0:
            params, opt_state, losses = rnn_utils.train_network(
                lambda: disrnn.HkDisentangledRNN(disrnn_config),
                dataset_train,
                dataset_eval,
                loss=args.loss,
                loss_param=args.loss_param,
                params=params,
                opt_state=None,
                opt=optax.adam(args.learning_rate),
                n_steps=args.n_steps,
                max_grad_norm=args.max_grad_norm,
                do_plot=True,
                random_key=training_key,
                report_progress_by="wandb",
                wandb_run=wandb_run,
                wandb_step_offset=args.n_warmup_steps,
            )
        else:
            optimizer = optax.adam(args.learning_rate)
            checkpoint_root = self.output_dir / "checkpoints"
            checkpoint_root.mkdir(parents=True, exist_ok=True)

            all_training_losses: list[float] = []
            all_validation_losses: list[float] = []
            steps_completed = 0
            opt_state = None
            xs_eval_all, ys_eval_all = dataset_eval.get_all()
            xs_full_for_checkpoint, _ = dataset.get_all()
            df_for_checkpoint = bundle.raw
            random_key = training_key
            heldout_eval_cfg = None
            heldout_test_data = self.heldout_data
            runtime_heldout_cfg = HeldoutEvalConfig.from_data_cfg(metadata)
            if runtime_heldout_cfg.enabled:
                heldout_eval_cfg = OmegaConf.create(
                    {
                        "data": asdict(runtime_heldout_cfg),
                        "model": {
                            "architecture": self.architecture,
                            "penalties": self.penalties,
                            "output_dir": str(self.output_dir),
                        },
                    }
                )
                if heldout_test_data is None:
                    try:
                        heldout_test_data = load_disrnn_heldout_subject_data(runtime_heldout_cfg)
                        self.heldout_data = heldout_test_data
                    except Exception as exc:
                        logger.warning(
                            "Preloading held-out test data failed; evaluation will fall back to lazy loading: %s",
                            exc,
                        )

            while steps_completed < args.n_steps:
                chunk_steps = min(
                    args.checkpoint_every_n_steps,
                    args.n_steps - steps_completed,
                )
                params, opt_state, chunk_losses = rnn_utils.train_network(
                    lambda: disrnn.HkDisentangledRNN(disrnn_config),
                    dataset_train,
                    dataset_eval,
                    loss=args.loss,
                    loss_param=args.loss_param,
                    params=params,
                    opt_state=opt_state,
                    opt=optimizer,
                    n_steps=chunk_steps,
                    max_grad_norm=args.max_grad_norm,
                    do_plot=False,
                    random_key=random_key,
                    report_progress_by="wandb",
                    wandb_run=wandb_run,
                    wandb_step_offset=args.n_warmup_steps + steps_completed,
                )

                all_training_losses.extend(np.asarray(chunk_losses["training_loss"]).tolist())
                all_validation_losses.extend(np.asarray(chunk_losses["validation_loss"]).tolist())

                random_key = jax.random.split(random_key, 2)[0]
                steps_completed += chunk_steps

                checkpoint_dir = checkpoint_root / f"step_{steps_completed}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_params_path = checkpoint_dir / "params.json"
                with checkpoint_params_path.open("w") as f:
                    f.write(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))
                eval_likelihood_ckpt: float | None = None
                if args.checkpoint_eval_on_eval_split:
                    yhat_eval_ckpt, _ = rnn_utils.eval_network(
                        lambda: disrnn.HkDisentangledRNN(noiseless_network),
                        params,
                        xs_eval_all,
                    )
                    n_action_logits_ckpt = int(getattr(dataset_eval, "n_classes", 0))
                    if n_action_logits_ckpt <= 0:
                        n_action_logits_ckpt = int(np.asarray(yhat_eval_ckpt).shape[2] - 1)
                    if n_action_logits_ckpt <= 0:
                        raise ValueError(
                            "Invalid number of action logits inferred for checkpoint eval likelihood."
                        )
                    eval_likelihood_ckpt = float(
                        rnn_utils.normalized_likelihood(
                            ys_eval_all,
                            yhat_eval_ckpt[:, :, :n_action_logits_ckpt],
                        )
                    )

                checkpoint_record = {
                    "step": int(steps_completed),
                    "params_path": str(checkpoint_params_path),
                }
                if eval_likelihood_ckpt is not None:
                    checkpoint_record["eval_likelihood"] = eval_likelihood_ckpt

                checkpoint_plot_paths: dict[str, Any] = {
                    "bottlenecks": None,
                    "choice_rule": None,
                    "update_rules": [],
                }

                try:
                    bottlenecks_fig_ckpt = plotting.plot_bottlenecks(
                        params, disrnn_config, sort_latents=False
                    )
                    bottlenecks_fig_ckpt.tight_layout()
                    bottlenecks_path_ckpt = checkpoint_dir / "bottlenecks.png"
                    bottlenecks_fig_ckpt.savefig(bottlenecks_path_ckpt)
                    plt.close(bottlenecks_fig_ckpt)
                    checkpoint_plot_paths["bottlenecks"] = str(bottlenecks_path_ckpt)

                    if args.checkpoint_plot_choice_rule:
                        choice_fig_ckpt = plotting.plot_choice_rule(params, disrnn_config)
                        if choice_fig_ckpt is not None:
                            axes = choice_fig_ckpt.get_axes()
                            for ax in axes:
                                ax.axhline(0, alpha=.5)
                                ax.axvline(0, alpha=.5)
                            choice_fig_ckpt.tight_layout()
                            choice_path_ckpt = checkpoint_dir / "choice_rule.png"
                            choice_fig_ckpt.savefig(choice_path_ckpt)
                            plt.close(choice_fig_ckpt)
                            checkpoint_plot_paths["choice_rule"] = str(choice_path_ckpt)

                    if args.checkpoint_plot_update_rules:
                        update_figs_ckpt = plotting.plot_update_rules(params, disrnn_config)
                        update_rule_paths_ckpt: list[str] = []
                        for index, fig_ckpt in enumerate(update_figs_ckpt):
                            fig_ckpt.tight_layout()
                            update_path_ckpt = checkpoint_dir / f"update_rule_{index}.png"
                            fig_ckpt.savefig(update_path_ckpt)
                            plt.close(fig_ckpt)
                            update_rule_paths_ckpt.append(str(update_path_ckpt))
                        checkpoint_plot_paths["update_rules"] = update_rule_paths_ckpt
                except Exception as exc:
                    logger.warning(
                        "Checkpoint plotting failed for step=%s: %s",
                        steps_completed,
                        exc,
                    )

                checkpoint_record["plot_paths"] = checkpoint_plot_paths

                should_plot_split_examples_ckpt = (
                    args.checkpoint_plot_split_examples_every_n > 0
                    and (
                        steps_completed % args.checkpoint_plot_split_examples_every_n == 0
                        or steps_completed == args.n_steps
                    )
                )
                should_save_output_df_ckpt = (
                    args.checkpoint_save_output_df_every_n > 0
                    and (
                        steps_completed % args.checkpoint_save_output_df_every_n == 0
                        or steps_completed == args.n_steps
                    )
                )
                if should_plot_split_examples_ckpt or should_save_output_df_ckpt:
                    yhat_full_ckpt, network_states_full_ckpt = rnn_utils.eval_network(
                        lambda: disrnn.HkDisentangledRNN(noiseless_network),
                        params,
                        xs_full_for_checkpoint,
                    )
                    output_df_ckpt = dl.add_model_results(
                        df_for_checkpoint.copy(),
                        np.asarray(network_states_full_ckpt),
                        yhat_full_ckpt,
                        ignore_policy=ignore_policy,
                    )

                    if should_save_output_df_ckpt:
                        output_df_ckpt_path = checkpoint_dir / "output_df.csv"
                        output_df_ckpt.to_csv(output_df_ckpt_path, index=False)
                        checkpoint_record["output_df_path"] = str(output_df_ckpt_path)

                    if should_plot_split_examples_ckpt:
                        n_action_logits_ckpt_full = int(getattr(dataset_eval, "n_classes", 0))
                        if n_action_logits_ckpt_full <= 0:
                            n_action_logits_ckpt_full = int(
                                np.asarray(yhat_full_ckpt).shape[2] - 1
                            )
                        split_summaries_ckpt = self._generate_split_examples(
                            output_dir=checkpoint_dir,
                            output_df=output_df_ckpt,
                            network_states_full=np.asarray(network_states_full_ckpt),
                            yhat_full=np.asarray(yhat_full_ckpt),
                            params=params,
                            metadata=metadata,
                            n_action_logits=n_action_logits_ckpt_full,
                            wandb_run=(
                                wandb_run
                                if args.checkpoint_log_split_examples_to_wandb
                                else None
                            ),
                            wandb_step=args.n_warmup_steps + int(steps_completed),
                            wandb_key_prefix="checkpoint",
                        )
                        checkpoint_record["split_examples"] = split_summaries_ckpt

                checkpoint_records.append(checkpoint_record)

                if (
                    wandb_run is not None
                    and eval_likelihood_ckpt is not None
                    and args.checkpoint_log_eval_to_wandb
                ):
                    wandb_run.log(
                        {
                            "checkpoint/eval_likelihood": eval_likelihood_ckpt,
                            "checkpoint/step": int(steps_completed),
                        },
                        step=args.n_warmup_steps + int(steps_completed),
                    )

                if wandb_run is not None and args.checkpoint_log_eval_to_wandb:
                    checkpoint_plot_payload = {}
                    checkpoint_bottlenecks = checkpoint_plot_paths.get("bottlenecks")
                    checkpoint_choice_rule = checkpoint_plot_paths.get("choice_rule")
                    checkpoint_update_rules = checkpoint_plot_paths.get("update_rules", [])
                    if checkpoint_bottlenecks:
                        checkpoint_plot_payload[
                            "checkpoint/fig/bottlenecks"
                        ] = wandb.Image(str(checkpoint_bottlenecks))
                    if checkpoint_choice_rule:
                        checkpoint_plot_payload[
                            "checkpoint/fig/choice_rule"
                        ] = wandb.Image(str(checkpoint_choice_rule))
                    for index, update_rule_path in enumerate(checkpoint_update_rules):
                        checkpoint_plot_payload[
                            f"checkpoint/fig/update_rule_{index}"
                        ] = wandb.Image(str(update_rule_path))
                    if checkpoint_plot_payload:
                        wandb_run.log(
                            checkpoint_plot_payload,
                            step=args.n_warmup_steps + int(steps_completed),
                        )

                    if not args.checkpoint_keep_media_files:
                        for maybe_path in [
                            checkpoint_bottlenecks,
                            checkpoint_choice_rule,
                            *checkpoint_update_rules,
                        ]:
                            if not maybe_path:
                                continue
                            try:
                                Path(str(maybe_path)).unlink(missing_ok=True)
                            except Exception as exc:
                                logger.warning(
                                    "Failed to remove checkpoint media file %s: %s",
                                    maybe_path,
                                    exc,
                                )
                        checkpoint_plot_paths["bottlenecks"] = None
                        checkpoint_plot_paths["choice_rule"] = None
                        checkpoint_plot_paths["update_rules"] = []

                if (
                    wandb_run is not None
                    and args.checkpoint_log_split_examples_to_wandb
                    and not args.checkpoint_keep_media_files
                    and "split_examples" in checkpoint_record
                ):
                    split_examples = checkpoint_record.get("split_examples", {})
                    for split_summary in split_examples.values():
                        if not isinstance(split_summary, dict):
                            continue
                        plots = split_summary.get("plots", {})
                        if not isinstance(plots, dict):
                            continue
                        for key in (
                            "latents_over_trials_examples",
                            "latents_in_space_examples",
                        ):
                            raw_paths = plots.get(key, [])
                            if not isinstance(raw_paths, list):
                                continue
                            for raw_path in raw_paths:
                                try:
                                    Path(str(raw_path)).unlink(missing_ok=True)
                                except Exception as exc:
                                    logger.warning(
                                        "Failed to remove checkpoint split-example media file %s: %s",
                                        raw_path,
                                        exc,
                                    )
                            plots[key] = []

                should_run_checkpoint_heldout = (
                    heldout_eval_cfg is not None
                    and args.checkpoint_run_heldout_eval
                    and (
                        int(steps_completed) < int(args.n_steps)
                        or (
                            args.checkpoint_include_final_in_heldout and int(steps_completed) == int(args.n_steps)
                        )
                    )
                )
                if should_run_checkpoint_heldout:
                    try:
                        ckpt_summary = evaluate_disrnn_on_heldout_subjects(
                            heldout_eval_cfg,
                            wandb_run=wandb_run,
                            params_path=checkpoint_params_path,
                            output_subdir=f"heldout_test/checkpoints/step_{steps_completed}",
                            log_to_wandb=False,
                            heldout_data=heldout_test_data,
                        )
                        if ckpt_summary is not None:
                            checkpoint_heldout_summaries.append(
                                {
                                    "step": int(steps_completed),
                                    "params_path": str(checkpoint_params_path),
                                    "heldout_test_likelihood": float(
                                        ckpt_summary["test_likelihood"]
                                    ),
                                }
                            )
                            if wandb_run is not None:
                                wandb_step = args.n_warmup_steps + int(steps_completed)
                                wandb_run.log(
                                    {
                                        "checkpoint/heldout_test_likelihood": float(
                                            ckpt_summary["test_likelihood"]
                                        ),
                                        "checkpoint/step": int(steps_completed),
                                    },
                                    step=wandb_step,
                                )
                                try:
                                    trial_plot_paths = ckpt_summary.get("plots", {}).get(
                                        "latents_over_trials_examples", []
                                    )
                                    space_plot_paths = ckpt_summary.get("plots", {}).get(
                                        "latents_in_space_examples", []
                                    )
                                    checkpoint_plot_payload = {}
                                    if trial_plot_paths:
                                        checkpoint_plot_payload[
                                            "checkpoint/heldout/latents_over_trials_examples"
                                        ] = [
                                            wandb.Image(str(path)) for path in trial_plot_paths
                                        ]
                                    if space_plot_paths:
                                        checkpoint_plot_payload[
                                            "checkpoint/heldout/latents_in_space_examples"
                                        ] = [
                                            wandb.Image(str(path)) for path in space_plot_paths
                                        ]
                                    if checkpoint_plot_payload:
                                        wandb_run.log(checkpoint_plot_payload, step=wandb_step)
                                        if not args.checkpoint_keep_media_files:
                                            for path in trial_plot_paths + space_plot_paths:
                                                try:
                                                    Path(str(path)).unlink(missing_ok=True)
                                                except Exception as path_exc:
                                                    logger.warning(
                                                        "Failed to remove checkpoint held-out media file %s: %s",
                                                        path,
                                                        path_exc,
                                                    )
                                            plots = ckpt_summary.get("plots", {})
                                            if isinstance(plots, dict):
                                                plots["latents_over_trials_examples"] = []
                                                plots["latents_in_space_examples"] = []
                                except Exception as exc:
                                    logger.warning(
                                        "Checkpoint held-out image logging failed for step=%s: %s",
                                        steps_completed,
                                        exc,
                                    )
                    except Exception as exc:
                        logger.warning(
                            "Held-out evaluation failed for checkpoint step=%s: %s",
                            steps_completed,
                            exc,
                        )

                checkpoint_records.append(checkpoint_record)

            losses = {
                "training_loss": np.asarray(all_training_losses, dtype=float),
                "validation_loss": np.asarray(all_validation_losses, dtype=float),
            }

        training_time = time.time() - start
        output["training_time"] = training_time
        output["checkpoint_every_n_steps"] = int(args.checkpoint_every_n_steps)
        if checkpoint_records:
            output["checkpoints"] = checkpoint_records
        if checkpoint_heldout_summaries:
            output["heldout_test_checkpoints"] = checkpoint_heldout_summaries

        losses_path = self._plot_losses(
            losses,
            title="Loss over Training",
            output_name="validation.png",
        )
        if wandb_run is not None:
            wandb_run.log({"fig/validation_loss_curve": wandb.Image(str(losses_path))})

        bottlenecks_fig = plotting.plot_bottlenecks(params, disrnn_config, sort_latents=False)
        bottlenecks_fig.tight_layout()
        bottlenecks_path = self._save_figure(bottlenecks_fig, "bottlenecks.png")
        if wandb_run is not None:
            wandb_run.log({"fig/bottlenecks": wandb.Image(str(bottlenecks_path))})

        if args.plot_choice_rule:
            choice_fig = plotting.plot_choice_rule(params, disrnn_config)
            if choice_fig is not None:
                axes = choice_fig.get_axes()
                for ax in axes:
                    ax.axhline(0, alpha=.5)
                    ax.axvline(0, alpha=.5)
                choice_fig.tight_layout()
                choice_path = self._save_figure(choice_fig, "choice_rule.png")
                if wandb_run is not None:
                    wandb_run.log({"fig/choice_rule": wandb.Image(str(choice_path))})

        if args.plot_update_rules:
            update_figs = plotting.plot_update_rules(params, disrnn_config)
            for index, fig in enumerate(update_figs):
                fig.tight_layout()
                path = self._save_figure(fig, f"update_rule_{index}.png")
                if wandb_run is not None:
                    wandb_run.log({f"fig/update_rule_{index}": wandb.Image(str(path))})

        # Get model predictions on full dataset, including the training set
        xs_full, ys_full = dataset.get_all()
        yhat_full, network_states_full = rnn_utils.eval_network(
            lambda: disrnn.HkDisentangledRNN(noiseless_network), params, xs_full
        )

        df = bundle.raw
        output_df = dl.add_model_results(
            df, network_states_full.__array__(), yhat_full, ignore_policy=ignore_policy
        )
        output_path = self.output_dir / "output_df.csv"
        output_df.to_csv(output_path, index=False)

        params_path = self.output_dir / "params.json"
        with params_path.open("w") as f:
            f.write(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))

        # Get likelihood evaluated on just the evaluation dataset
        xs_eval, ys_eval = dataset_eval.get_all()
        yhat_eval, network_states_eval = rnn_utils.eval_network(
            lambda: disrnn.HkDisentangledRNN(noiseless_network), params, xs_eval
        )
        n_action_logits = int(getattr(dataset_eval, "n_classes", 0))
        if n_action_logits <= 0:
            n_action_logits = int(yhat_eval.shape[2] - 1)
        if n_action_logits <= 0:
            raise ValueError(
                f"Invalid number of action logits inferred for eval likelihood: {n_action_logits}"
            )

        likelihood = rnn_utils.normalized_likelihood(
            ys_eval,
            yhat_eval[:, :, :n_action_logits],
        )
        output["likelihood"] = float(likelihood)

        final_output_dir = self.output_dir
        if args.checkpoint_every_n_steps > 0:
            final_output_dir = self.output_dir / "checkpoints" / f"step_{args.n_steps}"
            final_output_dir.mkdir(parents=True, exist_ok=True)

        split_summaries = self._generate_split_examples(
            output_dir=final_output_dir,
            output_df=output_df,
            network_states_full=np.asarray(network_states_full),
            yhat_full=np.asarray(yhat_full),
            params=params,
            metadata=metadata,
            n_action_logits=n_action_logits,
            wandb_run=(
                wandb_run
                if (args.checkpoint_every_n_steps == 0)
                else None
            ),
        )
        output["split_examples"] = split_summaries
        
        # -- Compare to groundtruth likelihood if available --
        gt_likelihood = metadata.get("avg_eval_likelihood_groundtruth")
        if gt_likelihood is not None:
            output["groundtruth_likelihood"] = float(gt_likelihood)
            output["likelihood_relative_to_groundtruth"] = float(likelihood) / float(gt_likelihood)

        # save output to json
        with open(self.output_dir / "output_summary.json", "w") as f:
            json.dump(output, f, indent=4)

        if checkpoint_records:
            checkpoint_metrics_path = self.output_dir / "checkpoint_metrics.json"
            with checkpoint_metrics_path.open("w") as f:
                json.dump(checkpoint_records, f, indent=2)

            checkpoint_root = self.output_dir / "checkpoints"
            checkpoint_root.mkdir(parents=True, exist_ok=True)
            checkpoint_index_path = checkpoint_root / "index.json"
            checkpoint_index = {
                "checkpoint_every_n_steps": int(args.checkpoint_every_n_steps),
                "n_steps": int(args.n_steps),
                "n_warmup_steps": int(args.n_warmup_steps),
                "count": int(len(checkpoint_records)),
                "checkpoints": checkpoint_records,
            }
            with checkpoint_index_path.open("w") as f:
                json.dump(checkpoint_index, f, indent=2)

        # Save config as diciontary
        disrnn_config_dict = asdict(disrnn_config)
        with open(self.output_dir / "disrnn_config.json", "w") as f:
            json.dump(disrnn_config_dict, f, indent=4)

        if wandb_run is not None:
            wandb_run.summary["final/val_loss"] = float(losses["validation_loss"][-1])
            wandb_run.summary["final/train_loss"] = float(losses["training_loss"][-1])
            wandb_run.summary["likelihood"] = float(likelihood)
            
            if gt_likelihood is not None:
                wandb_run.summary["groundtruth_likelihood"] = float(gt_likelihood)
                wandb_run.summary["likelihood_relative_to_groundtruth"] = float(likelihood) / float(gt_likelihood)

            wandb_run.summary["elapsed_seconds"] = float(training_time)
            wandb_run.summary["warmup_seconds"] = float(warmup_duration)

            # Upload the whole /results/output folder as an artifact
            # Here I'm using the random id as the name, meaning each run will has its own artifact.
            artifact_name = (
                f"disrnn-output-{getattr(wandb_run, 'id', None) or 'latest'}"
            )
            artifact = wandb.Artifact(artifact_name, type="training-output")
            artifact.add_dir(str(self.output_dir))
            wandb_run.log_artifact(artifact)

        return output

    def _plot_losses(
        self, losses: Mapping[str, Any], title: str, output_name: str, log_loss_every: int = 10
    ) -> Path:
        fig = plt.figure()
        timepoints = np.array(np.arange(0, len(losses['training_loss'])*log_loss_every, log_loss_every))
        timepoints[0] = 1
        plt.semilogy(timepoints, losses["training_loss"], color="black")
        plt.semilogy(timepoints, losses["validation_loss"], color="tab:red", linestyle="dashed")
        plt.xlabel("Training Step")
        plt.ylabel("Mean Loss")
        plt.legend(("Training Set", "Validation Set"))
        plt.title(title)
        path = self._save_figure(fig, output_name)
        return path

    def _generate_split_examples(
        self,
        *,
        output_dir: Path,
        output_df: pd.DataFrame,
        network_states_full: np.ndarray,
        yhat_full: np.ndarray,
        params: Any,
        metadata: dict[str, Any],
        n_action_logits: int,
        wandb_run: Any | None = None,
        wandb_step: int | None = None,
        wandb_key_prefix: str | None = None,
    ) -> dict[str, Any]:
        default_sessions_per_subject = int(metadata.get("heldout_example_sessions_per_subject", 1))
        train_sessions_per_subject = int(
            metadata.get("train_example_sessions_per_subject", default_sessions_per_subject)
        )
        eval_sessions_per_subject = int(
            metadata.get("eval_example_sessions_per_subject", default_sessions_per_subject)
        )
        if train_sessions_per_subject < 0:
            raise ValueError("train_example_sessions_per_subject must be >= 0")
        if eval_sessions_per_subject < 0:
            raise ValueError("eval_example_sessions_per_subject must be >= 0")
        max_subjects_to_plot = int(metadata.get("example_max_subjects", 6))
        if max_subjects_to_plot < 0:
            raise ValueError("example_max_subjects must be >= 0")

        session_order = list(dict.fromkeys(output_df["ses_idx"].tolist()))
        n_sessions_full = int(np.asarray(network_states_full).shape[1])
        if len(session_order) != n_sessions_full:
            raise ValueError(
                "Session mismatch between dataframe and model outputs: "
                f"df_sessions={len(session_order)} model_sessions={n_sessions_full}"
            )

        eval_every_n = int(metadata.get("eval_every_n", 2))
        if eval_every_n <= 0:
            raise ValueError(f"Invalid eval_every_n in metadata: {eval_every_n}")

        eval_indices = np.arange(eval_every_n - 1, n_sessions_full, eval_every_n)
        train_indices = np.array(
            [idx for idx in range(n_sessions_full) if idx not in set(eval_indices)],
            dtype=int,
        )

        split_summaries: dict[str, Any] = {}
        sessions_per_subject_by_split = {
            "training": train_sessions_per_subject,
            "eval": eval_sessions_per_subject,
        }
        for split_name, split_indices in (("training", train_indices), ("eval", eval_indices)):
            split_sessions_per_subject = sessions_per_subject_by_split[split_name]
            if split_sessions_per_subject == 0:
                logger.info("Skipping %s example plotting (sessions_per_subject=0).", split_name)
                split_summaries[split_name] = {
                    "split": split_name,
                    "plotting_skipped": True,
                    "example_sessions_per_subject": 0,
                    "example_max_subjects": max_subjects_to_plot,
                    "plots": {
                        "latents_over_trials_examples": [],
                        "latents_in_space_examples": [],
                    },
                }
                continue

            if split_indices.size == 0:
                logger.warning("Skipping %s example plotting: no sessions selected.", split_name)
                continue

            split_session_ids = [session_order[int(i)] for i in split_indices.tolist()]
            split_output_df = output_df[output_df["ses_idx"].isin(split_session_ids)].copy()
            split_output_df["ses_idx"] = pd.Categorical(
                split_output_df["ses_idx"],
                categories=split_session_ids,
                ordered=True,
            )
            split_output_df = split_output_df.sort_values(["ses_idx", "trial"]).copy()
            split_output_df["ses_idx"] = split_output_df["ses_idx"].astype(str)

            try:
                split_summary = plot_disrnn_examples_for_split(
                    split_name=split_name,
                    output_dir=output_dir,
                    output_df=split_output_df,
                    network_states=np.asarray(network_states_full)[:, split_indices, :],
                    yhat_logits=np.asarray(yhat_full)[:, split_indices, :],
                    params=params,
                    sessions_per_subject=split_sessions_per_subject,
                    max_subjects_to_plot=max_subjects_to_plot,
                    n_action_logits=n_action_logits,
                    wandb_run=None,
                )
                split_summaries[split_name] = split_summary
            except Exception as exc:
                logger.warning(
                    "Skipping %s example plotting due to plotting error: %s",
                    split_name,
                    exc,
                )
                split_summaries[split_name] = {
                    "split": split_name,
                    "plotting_failed": True,
                    "error": str(exc),
                    "example_sessions_per_subject": split_sessions_per_subject,
                    "example_max_subjects": max_subjects_to_plot,
                    "plots": {
                        "latents_over_trials_examples": [],
                        "latents_in_space_examples": [],
                    },
                }

        if wandb_run is not None:
            for split_name, split_summary in split_summaries.items():
                if split_summary.get("plotting_failed", False):
                    continue
                key_prefix = f"{split_name}"
                if wandb_key_prefix:
                    key_prefix = f"{wandb_key_prefix}/{split_name}"
                payload = {
                    f"{key_prefix}/latents_over_trials_examples": [
                        wandb.Image(str(path))
                        for path in split_summary.get("plots", {}).get(
                            "latents_over_trials_examples", []
                        )
                    ],
                    f"{key_prefix}/latents_in_space_examples": [
                        wandb.Image(str(path))
                        for path in split_summary.get("plots", {}).get(
                            "latents_in_space_examples", []
                        )
                    ],
                }
                if wandb_step is None:
                    wandb_run.log(payload)
                else:
                    wandb_run.log(payload, step=wandb_step)

        return split_summaries

    def _save_figure(self, fig: Any, filename: str) -> Path:
        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        return path