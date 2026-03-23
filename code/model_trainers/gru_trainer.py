from __future__ import annotations

import json
import logging
import time
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Mapping

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf

from disentangled_rnns.library import rnn_utils

from base.interfaces import ModelTrainer
from base.types import DatasetBundle
from models.gru_network import make_gru_network
from utils.disrnn_evaluation import HeldoutEvalConfig
from utils.gru_evaluation import (
    add_gru_model_results,
    evaluate_gru_on_heldout_subjects,
    load_gru_heldout_subject_data,
    plot_gru_examples_for_split,
)

logger = logging.getLogger(__name__)


def _to_dict(config: Mapping[str, Any] | DictConfig) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


def _require_n_action_logits(dataset: Any, yhat: np.ndarray, *, context: str) -> int:
    n_action_logits = int(getattr(dataset, "n_classes", 0))
    if n_action_logits <= 0:
        raise ValueError(
            f"GRU {context} requires dataset.n_classes to be set to a positive integer."
        )
    actual_output_size = int(np.asarray(yhat).shape[2])
    if actual_output_size != n_action_logits:
        raise ValueError(
            f"GRU {context} logits shape mismatch: dataset.n_classes={n_action_logits} "
            f"but yhat.shape[2]={actual_output_size}"
        )
    return n_action_logits


def _log_heldout_dataset_details(heldout_data: dict[str, Any] | None) -> None:
    if not heldout_data:
        return

    xs_test = heldout_data.get("xs_test")
    ys_test = heldout_data.get("ys_test")
    if xs_test is None or ys_test is None:
        dataset_test = heldout_data.get("dataset_test")
        if dataset_test is not None:
            xs_test = getattr(dataset_test, "_xs", None)
            ys_test = getattr(dataset_test, "_ys", None)

    if xs_test is None or ys_test is None:
        return

    logger.info(
        "Held-out test shapes: input %s, output %s",
        np.asarray(xs_test).shape,
        np.asarray(ys_test).shape,
    )


def _log_dataset_split_details(
    *,
    dataset: Any,
    dataset_train: Any,
    dataset_eval: Any,
) -> None:
    logger.info(
        "Full dataset shapes: input %s, output %s",
        dataset._xs.shape,
        dataset._ys.shape,
    )
    logger.info(
        "Training split shapes: input %s, output %s",
        dataset_train._xs.shape,
        dataset_train._ys.shape,
    )
    logger.info(
        "Eval split shapes: input %s, output %s",
        dataset_eval._xs.shape,
        dataset_eval._ys.shape,
    )


class GruTrainer(ModelTrainer):
    """Trainer that mirrors the disRNN pipeline for GRU models."""

    def __init__(
        self,
        architecture: Mapping[str, Any] | DictConfig,
        training: Mapping[str, Any] | DictConfig,
        heldout_data: dict[str, Any] | None = None,
        output_dir: str = "/results/outputs",
        seed: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__(seed=seed)
        self.architecture = _to_dict(architecture)
        self.training = _to_dict(training)
        self.heldout_data = heldout_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _remove_media_files(self, paths: list[Any]) -> None:
        for raw_path in paths:
            if not raw_path:
                continue
            try:
                Path(str(raw_path)).unlink(missing_ok=True)
            except Exception as exc:
                logger.warning("Failed to remove media file %s: %s", raw_path, exc)

    def _cleanup_split_example_media(self, split_examples: dict[str, Any]) -> None:
        for split_summary in split_examples.values():
            if not isinstance(split_summary, dict):
                continue
            plots = split_summary.get("plots", {})
            if not isinstance(plots, dict):
                continue
            for key in ("latents_over_trials_examples", "latents_in_space_examples"):
                raw_paths = plots.get(key, [])
                if not isinstance(raw_paths, list):
                    continue
                self._remove_media_files(raw_paths)
                plots[key] = []

    def _cleanup_heldout_media(self, heldout_summary: dict[str, Any]) -> None:
        plots = heldout_summary.get("plots", {})
        if not isinstance(plots, dict):
            return
        all_paths = []
        for key in ("latents_over_trials_examples", "latents_in_space_examples"):
            raw_paths = plots.get(key, [])
            if not isinstance(raw_paths, list):
                continue
            all_paths.extend(raw_paths)
            plots[key] = []
        self._remove_media_files(all_paths)

    def _log_heldout_summary_to_wandb(
        self,
        *,
        heldout_summary: dict[str, Any],
        wandb_run: Any | None,
        wandb_step: int | None,
        wandb_key_prefix: str,
    ) -> None:
        if wandb_run is None:
            return

        metric_payload = {
            f"{wandb_key_prefix}/heldout_test_likelihood": float(
                heldout_summary["test_likelihood"]
            ),
        }
        if wandb_step is None:
            wandb_run.log(metric_payload)
        else:
            wandb_run.log(metric_payload, step=wandb_step)

        trial_plot_paths = heldout_summary.get("plots", {}).get(
            "latents_over_trials_examples",
            [],
        )
        space_plot_paths = heldout_summary.get("plots", {}).get(
            "latents_in_space_examples",
            [],
        )
        image_payload = {}
        if trial_plot_paths:
            image_payload[f"{wandb_key_prefix}/heldout/latents_over_trials_examples"] = [
                wandb.Image(str(path)) for path in trial_plot_paths
            ]
        if space_plot_paths:
            image_payload[f"{wandb_key_prefix}/heldout/latents_in_space_examples"] = [
                wandb.Image(str(path)) for path in space_plot_paths
            ]
        if image_payload:
            if wandb_step is None:
                wandb_run.log(image_payload)
            else:
                wandb_run.log(image_payload, step=wandb_step)

    def _evaluate_initialization_snapshot(
        self,
        *,
        stage_name: str,
        params: Any,
        bundle: DatasetBundle,
        metadata: dict[str, Any],
        dataset: Any,
        dataset_train: Any,
        dataset_eval: Any,
        ignore_policy: str,
        make_network: Any,
        heldout_eval_cfg: Any | None,
        heldout_data: dict[str, Any] | None,
        wandb_run: Any | None,
        wandb_step: int | None,
        keep_media_files: bool,
    ) -> dict[str, Any]:
        stage_dir = self.output_dir / "initialization" / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        log_scope = stage_name.replace("_", " ").title()
        wandb_key_prefix = f"initialization/{stage_name}"

        params_path = stage_dir / "params.json"
        with params_path.open("w") as f:
            f.write(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))

        xs_train, ys_train = dataset_train.get_all()
        yhat_train, _ = rnn_utils.eval_network(make_network, params, xs_train)
        n_action_logits_train = _require_n_action_logits(
            dataset_train,
            np.asarray(yhat_train),
            context="initialization train",
        )
        train_likelihood = float(
            rnn_utils.normalized_likelihood(
                ys_train,
                np.asarray(yhat_train)[:, :, :n_action_logits_train],
            )
        )

        xs_eval, ys_eval = dataset_eval.get_all()
        yhat_eval, _ = rnn_utils.eval_network(make_network, params, xs_eval)
        n_action_logits_eval = _require_n_action_logits(
            dataset_eval,
            np.asarray(yhat_eval),
            context="initialization eval",
        )
        eval_likelihood = float(
            rnn_utils.normalized_likelihood(
                ys_eval,
                np.asarray(yhat_eval)[:, :, :n_action_logits_eval],
            )
        )

        logger.info("%s training likelihood: %.4f", log_scope, train_likelihood)
        logger.info("%s eval likelihood: %.4f", log_scope, eval_likelihood)

        if wandb_run is not None:
            metric_payload = {
                f"{wandb_key_prefix}/train_likelihood": train_likelihood,
                f"{wandb_key_prefix}/eval_likelihood": eval_likelihood,
            }
            if wandb_step is None:
                wandb_run.log(metric_payload)
            else:
                wandb_run.log(metric_payload, step=wandb_step)

        xs_full, _ = dataset.get_all()
        yhat_full, network_states_full = rnn_utils.eval_network(
            make_network,
            params,
            xs_full,
        )
        output_df = add_gru_model_results(
            bundle.raw.copy(),
            np.asarray(network_states_full),
            np.asarray(yhat_full),
            ignore_policy=ignore_policy,
        )
        output_df_path = stage_dir / "output_df.csv"
        output_df.to_csv(output_df_path, index=False)

        n_action_logits_full = _require_n_action_logits(
            dataset,
            np.asarray(yhat_full),
            context="initialization full-dataset plotting",
        )
        split_summaries = self._generate_split_examples(
            output_dir=stage_dir,
            output_df=output_df,
            network_states_full=np.asarray(network_states_full),
            yhat_full=np.asarray(yhat_full),
            params=params,
            metadata=metadata,
            n_action_logits=n_action_logits_full,
            wandb_run=wandb_run,
            log_scope=log_scope,
            wandb_step=wandb_step,
            wandb_key_prefix=wandb_key_prefix,
        )
        if wandb_run is not None and not keep_media_files:
            self._cleanup_split_example_media(split_summaries)

        snapshot_summary: dict[str, Any] = {
            "stage": stage_name,
            "params_path": str(params_path),
            "output_df_path": str(output_df_path),
            "train_likelihood": train_likelihood,
            "eval_likelihood": eval_likelihood,
            "split_examples": split_summaries,
        }

        if heldout_eval_cfg is not None:
            try:
                heldout_summary = evaluate_gru_on_heldout_subjects(
                    heldout_eval_cfg,
                    wandb_run=wandb_run,
                    params_path=params_path,
                    output_subdir=f"initialization/{stage_name}/heldout_test",
                    log_to_wandb=False,
                    heldout_data=heldout_data,
                    log_scope=log_scope,
                )
                if heldout_summary is not None:
                    snapshot_summary["heldout"] = heldout_summary
                    snapshot_summary["heldout_test_likelihood"] = float(
                        heldout_summary["test_likelihood"]
                    )
                    self._log_heldout_summary_to_wandb(
                        heldout_summary=heldout_summary,
                        wandb_run=wandb_run,
                        wandb_step=wandb_step,
                        wandb_key_prefix=wandb_key_prefix,
                    )
                    if wandb_run is not None and not keep_media_files:
                        self._cleanup_heldout_media(heldout_summary)
            except Exception as exc:
                logger.warning(
                    "Initialization held-out evaluation failed for %s: %s",
                    stage_name,
                    exc,
                )

        summary_path = stage_dir / "summary.json"
        with summary_path.open("w") as f:
            json.dump(snapshot_summary, f, indent=2)
        return snapshot_summary

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
            raise ValueError("Dataset bundle must include the constructed RNN dataset.")

        dataset_train = bundle.train_set
        dataset_eval = bundle.eval_set
        if dataset_train is None or dataset_eval is None:
            raise ValueError("Dataset bundle must include train and eval splits.")

        output = {
            "num_trials": metadata.get("num_trials"),
            "num_sessions": metadata.get("num_sessions"),
        }

        _log_dataset_split_details(
            dataset=dataset,
            dataset_train=dataset_train,
            dataset_eval=dataset_eval,
        )
        _log_heldout_dataset_details(self.heldout_data)

        key = jax.random.PRNGKey(self.seed)
        output["random_key"] = [int(x) for x in np.asarray(key).reshape(-1)]

        expected_output_size = 2 if ignore_policy == "exclude" else 3
        args = types.SimpleNamespace(
            hidden_size=int(self.architecture["hidden_size"]),
            num_layers=int(self.architecture.get("num_layers", 1)),
            output_size=int(self.architecture.get("output_size", expected_output_size)),
            max_grad_norm=self.training["max_grad_norm"],
            n_steps=int(self.training["n_steps"]),
            learning_rate=self.training["lr"],
            loss=self.training["loss"],
            loss_param=self.training["loss_param"],
            checkpoint_every_n_steps=int(self.training.get("checkpoint_every_n_steps", 0)),
            checkpoint_eval_on_eval_split=bool(
                self.training.get("checkpoint_eval_on_eval_split", True)
            ),
            checkpoint_eval_on_train_split=bool(
                self.training.get("checkpoint_eval_on_train_split", True)
            ),
            checkpoint_log_eval_to_wandb=bool(
                self.training.get("checkpoint_log_eval_to_wandb", True)
            ),
            checkpoint_log_train_to_wandb=bool(
                self.training.get("checkpoint_log_train_to_wandb", True)
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
            checkpoint_keep_media_files=bool(
                self.training.get("checkpoint_keep_media_files", True)
            ),
            checkpoint_run_heldout_eval=bool(
                self.training.get("checkpoint_run_heldout_eval", True)
            ),
            checkpoint_include_final_in_heldout=bool(
                self.training.get("checkpoint_include_final_in_heldout", False)
            ),
            save_output_df=bool(self.training.get("save_output_df", False)),
        )

        if args.num_layers != 1:
            raise NotImplementedError(
                "GRU integration currently supports architecture.num_layers == 1 only."
            )
        if args.output_size != expected_output_size:
            raise ValueError(
                "Configured GRU output_size does not match dataset ignore_policy: "
                f"configured={args.output_size} expected={expected_output_size}"
            )
        if args.n_steps < 0:
            raise ValueError("training.n_steps must be >= 0")
        if args.checkpoint_every_n_steps < 0:
            raise ValueError("training.checkpoint_every_n_steps must be >= 0")
        if args.checkpoint_plot_split_examples_every_n < 0:
            raise ValueError("training.checkpoint_plot_split_examples_every_n must be >= 0")
        if args.checkpoint_save_output_df_every_n < 0:
            raise ValueError("training.checkpoint_save_output_df_every_n must be >= 0")

        logger.info("max_grad_norm = %s", args.max_grad_norm)

        make_network = make_gru_network(
            hidden_size=args.hidden_size,
            output_size=args.output_size,
        )

        heldout_eval_cfg = None
        heldout_test_data = self.heldout_data
        runtime_heldout_cfg = HeldoutEvalConfig.from_data_cfg(metadata)
        if runtime_heldout_cfg.enabled:
            heldout_eval_cfg = OmegaConf.create(
                {
                    "data": asdict(runtime_heldout_cfg),
                    "model": {
                        "architecture": self.architecture,
                        "output_dir": str(self.output_dir),
                    },
                }
            )
            if heldout_test_data is None:
                try:
                    heldout_test_data = load_gru_heldout_subject_data(runtime_heldout_cfg)
                    self.heldout_data = heldout_test_data
                    _log_heldout_dataset_details(heldout_test_data)
                except Exception as exc:
                    logger.warning(
                        "Preloading held-out test data failed; evaluation will fall back to lazy loading: %s",
                        exc,
                    )

        start = time.time()
        checkpoint_records: list[dict[str, Any]] = []
        checkpoint_heldout_summaries: list[dict[str, Any]] = []

        params, init_opt_state, _ = rnn_utils.train_network(
            make_network,
            dataset_train,
            dataset_eval,
            loss=args.loss,
            loss_param=args.loss_param,
            opt=optax.adam(args.learning_rate),
            n_steps=0,
            max_grad_norm=args.max_grad_norm,
            random_key=key,
            report_progress_by="wandb",
            wandb_run=wandb_run,
        )
        output["initial_evaluations"] = {
            "before_training": self._evaluate_initialization_snapshot(
                stage_name="before_training",
                params=params,
                bundle=bundle,
                metadata=metadata,
                dataset=dataset,
                dataset_train=dataset_train,
                dataset_eval=dataset_eval,
                ignore_policy=ignore_policy,
                make_network=make_network,
                heldout_eval_cfg=heldout_eval_cfg,
                heldout_data=heldout_test_data,
                wandb_run=wandb_run,
                wandb_step=0,
                keep_media_files=args.checkpoint_keep_media_files,
            )
        }

        if args.checkpoint_every_n_steps == 0:
            params, opt_state, losses = rnn_utils.train_network(
                make_network,
                dataset_train,
                dataset_eval,
                loss=args.loss,
                loss_param=args.loss_param,
                params=params,
                opt_state=init_opt_state,
                opt=optax.adam(args.learning_rate),
                n_steps=args.n_steps,
                max_grad_norm=args.max_grad_norm,
                do_plot=True,
                random_key=key,
                report_progress_by="wandb",
                wandb_run=wandb_run,
            )
        else:
            optimizer = optax.adam(args.learning_rate)
            checkpoint_root = self.output_dir / "checkpoints"
            checkpoint_root.mkdir(parents=True, exist_ok=True)

            params, opt_state, _ = rnn_utils.train_network(
                make_network,
                dataset_train,
                dataset_eval,
                opt=optimizer,
                loss=args.loss,
                loss_param=args.loss_param,
                n_steps=0,
                max_grad_norm=args.max_grad_norm,
                random_key=key,
                report_progress_by="wandb",
                wandb_run=wandb_run,
            )

            all_training_losses: list[float] = []
            all_validation_losses: list[float] = []
            steps_completed = 0
            xs_train_all, ys_train_all = dataset_train.get_all()
            xs_eval_all, ys_eval_all = dataset_eval.get_all()
            xs_full_for_checkpoint, _ = dataset.get_all()
            df_for_checkpoint = bundle.raw
            random_key = key
            opt_state = init_opt_state

            while steps_completed < args.n_steps:
                chunk_steps = min(
                    args.checkpoint_every_n_steps,
                    args.n_steps - steps_completed,
                )
                params, opt_state, chunk_losses = rnn_utils.train_network(
                    make_network,
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
                    wandb_step_offset=steps_completed,
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

                train_likelihood_ckpt: float | None = None
                if args.checkpoint_eval_on_train_split:
                    yhat_train_ckpt, _ = rnn_utils.eval_network(
                        make_network,
                        params,
                        xs_train_all,
                    )
                    n_action_logits_train_ckpt = _require_n_action_logits(
                        dataset_train,
                        np.asarray(yhat_train_ckpt),
                        context="checkpoint train",
                    )
                    train_likelihood_ckpt = float(
                        rnn_utils.normalized_likelihood(
                            ys_train_all,
                            np.asarray(yhat_train_ckpt)[:, :, :n_action_logits_train_ckpt],
                        )
                    )

                eval_likelihood_ckpt: float | None = None
                if args.checkpoint_eval_on_eval_split:
                    yhat_eval_ckpt, _ = rnn_utils.eval_network(
                        make_network,
                        params,
                        xs_eval_all,
                    )
                    n_action_logits_ckpt = _require_n_action_logits(
                        dataset_eval,
                        np.asarray(yhat_eval_ckpt),
                        context="checkpoint eval",
                    )
                    eval_likelihood_ckpt = float(
                        rnn_utils.normalized_likelihood(
                            ys_eval_all,
                            np.asarray(yhat_eval_ckpt)[:, :, :n_action_logits_ckpt],
                        )
                    )

                checkpoint_record = {
                    "step": int(steps_completed),
                    "params_path": str(checkpoint_params_path),
                }
                if train_likelihood_ckpt is not None:
                    checkpoint_record["train_likelihood"] = train_likelihood_ckpt
                    logger.info(
                        "Checkpoint step %s, training likelihood: %.4f",
                        steps_completed,
                        train_likelihood_ckpt,
                    )
                if eval_likelihood_ckpt is not None:
                    checkpoint_record["eval_likelihood"] = eval_likelihood_ckpt
                    logger.info(
                        "Checkpoint step %s, eval likelihood: %.4f",
                        steps_completed,
                        eval_likelihood_ckpt,
                    )

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
                        make_network,
                        params,
                        xs_full_for_checkpoint,
                    )
                    output_df_ckpt = add_gru_model_results(
                        df_for_checkpoint.copy(),
                        np.asarray(network_states_full_ckpt),
                        np.asarray(yhat_full_ckpt),
                        ignore_policy=ignore_policy,
                    )

                    if should_save_output_df_ckpt:
                        output_df_ckpt_path = checkpoint_dir / "output_df.csv"
                        output_df_ckpt.to_csv(output_df_ckpt_path, index=False)
                        checkpoint_record["output_df_path"] = str(output_df_ckpt_path)

                    if should_plot_split_examples_ckpt:
                        n_action_logits_ckpt_full = _require_n_action_logits(
                            dataset,
                            np.asarray(yhat_full_ckpt),
                            context="checkpoint full-dataset plotting",
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
                            log_scope=f"Checkpoint step {steps_completed}",
                            wandb_step=int(steps_completed),
                            wandb_key_prefix="checkpoint",
                        )
                        checkpoint_record["split_examples"] = split_summaries_ckpt

                checkpoint_records.append(checkpoint_record)

                if (
                    wandb_run is not None
                    and args.checkpoint_log_train_to_wandb
                    and train_likelihood_ckpt is not None
                ):
                    wandb_run.log(
                        {
                            "checkpoint/train_likelihood": train_likelihood_ckpt,
                            "checkpoint/step": int(steps_completed),
                        },
                        step=int(steps_completed),
                    )

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
                        step=int(steps_completed),
                    )

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
                        for key_name in (
                            "latents_over_trials_examples",
                            "latents_in_space_examples",
                        ):
                            raw_paths = plots.get(key_name, [])
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
                            plots[key_name] = []

                should_run_checkpoint_heldout = (
                    heldout_eval_cfg is not None
                    and args.checkpoint_run_heldout_eval
                    and (
                        int(steps_completed) < int(args.n_steps)
                        or (
                            args.checkpoint_include_final_in_heldout
                            and int(steps_completed) == int(args.n_steps)
                        )
                    )
                )
                if should_run_checkpoint_heldout:
                    try:
                        ckpt_summary = evaluate_gru_on_heldout_subjects(
                            heldout_eval_cfg,
                            wandb_run=wandb_run,
                            params_path=checkpoint_params_path,
                            output_subdir=f"heldout_test/checkpoints/step_{steps_completed}",
                            log_to_wandb=False,
                            heldout_data=heldout_test_data,
                            log_scope=f"Checkpoint step {steps_completed}",
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
                                wandb_step = int(steps_completed)
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

        xs_full, _ = dataset.get_all()
        yhat_full, network_states_full = rnn_utils.eval_network(
            make_network,
            params,
            xs_full,
        )

        df = bundle.raw
        output_df = add_gru_model_results(
            df,
            np.asarray(network_states_full),
            np.asarray(yhat_full),
            ignore_policy=ignore_policy,
        )
        if args.save_output_df:
            output_path = self.output_dir / "output_df.csv"
            output_df.to_csv(output_path, index=False)

        params_path = self.output_dir / "params.json"
        with params_path.open("w") as f:
            f.write(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))

        xs_train, ys_train = dataset_train.get_all()
        yhat_train, _ = rnn_utils.eval_network(make_network, params, xs_train)
        n_action_logits_train = _require_n_action_logits(
            dataset_train,
            np.asarray(yhat_train),
            context="final train",
        )
        likelihood_train = rnn_utils.normalized_likelihood(
            ys_train,
            np.asarray(yhat_train)[:, :, :n_action_logits_train],
        )
        output["likelihood_train"] = float(likelihood_train)
        logger.info("Final training likelihood: %.4f", float(likelihood_train))

        xs_eval, ys_eval = dataset_eval.get_all()
        yhat_eval, _ = rnn_utils.eval_network(make_network, params, xs_eval)
        n_action_logits = _require_n_action_logits(
            dataset_eval,
            np.asarray(yhat_eval),
            context="final eval",
        )

        likelihood = rnn_utils.normalized_likelihood(
            ys_eval,
            np.asarray(yhat_eval)[:, :, :n_action_logits],
        )
        output["likelihood"] = float(likelihood)
        logger.info("Final eval likelihood: %.4f", float(likelihood))

        final_output_dir = self.output_dir
        if args.checkpoint_every_n_steps > 0:
            final_output_dir = self.output_dir / "checkpoints" / f"step_{args.n_steps}"
            final_output_dir.mkdir(parents=True, exist_ok=True)

        split_summaries: dict[str, Any] | None = None
        if checkpoint_records:
            last_checkpoint = checkpoint_records[-1]
            if (
                int(last_checkpoint.get("step", -1)) == int(args.n_steps)
                and "split_examples" in last_checkpoint
            ):
                split_summaries = last_checkpoint["split_examples"]

        if split_summaries is None:
            split_summaries = self._generate_split_examples(
                output_dir=final_output_dir,
                output_df=output_df,
                network_states_full=np.asarray(network_states_full),
                yhat_full=np.asarray(yhat_full),
                params=params,
                metadata=metadata,
                n_action_logits=n_action_logits,
                wandb_run=wandb_run if args.checkpoint_every_n_steps == 0 else None,
                log_scope="Final",
            )
        output["split_examples"] = split_summaries

        gt_likelihood = metadata.get("avg_eval_likelihood_groundtruth")
        if gt_likelihood is not None:
            output["groundtruth_likelihood"] = float(gt_likelihood)
            output["likelihood_relative_to_groundtruth"] = float(likelihood) / float(
                gt_likelihood
            )

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
                "count": int(len(checkpoint_records)),
                "checkpoints": checkpoint_records,
            }
            with checkpoint_index_path.open("w") as f:
                json.dump(checkpoint_index, f, indent=2)

        gru_config = {
            "architecture": dict(self.architecture),
            "training": dict(self.training),
            "output_size": int(args.output_size),
        }
        with open(self.output_dir / "gru_config.json", "w") as f:
            json.dump(gru_config, f, indent=4)

        if wandb_run is not None:
            if len(losses["validation_loss"]) > 0:
                wandb_run.summary["final/val_loss"] = float(losses["validation_loss"][-1])
            if len(losses["training_loss"]) > 0:
                wandb_run.summary["final/train_loss"] = float(losses["training_loss"][-1])
            wandb_run.summary["likelihood"] = float(likelihood)
            wandb_run.summary["likelihood_train"] = float(likelihood_train)

            if gt_likelihood is not None:
                wandb_run.summary["groundtruth_likelihood"] = float(gt_likelihood)
                wandb_run.summary["likelihood_relative_to_groundtruth"] = float(
                    likelihood
                ) / float(gt_likelihood)

            wandb_run.summary["elapsed_seconds"] = float(training_time)

            artifact_name = f"gru-output-{getattr(wandb_run, 'id', None) or 'latest'}"
            artifact = wandb.Artifact(artifact_name, type="training-output")
            artifact.add_dir(str(self.output_dir))
            wandb_run.log_artifact(artifact)

        return output

    def _plot_losses(
        self,
        losses: Mapping[str, Any],
        title: str,
        output_name: str,
        log_loss_every: int = 10,
    ) -> Path:
        fig = plt.figure()
        timepoints = np.array(
            np.arange(0, len(losses["training_loss"]) * log_loss_every, log_loss_every)
        )
        if len(timepoints) > 0:
            timepoints[0] = 1
        plt.semilogy(timepoints, losses["training_loss"], color="black")
        plt.semilogy(
            timepoints,
            losses["validation_loss"],
            color="tab:red",
            linestyle="dashed",
        )
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
        log_scope: str | None = None,
        wandb_step: int | None = None,
        wandb_key_prefix: str | None = None,
    ) -> dict[str, Any]:
        default_sessions_per_subject = int(
            metadata.get("heldout_example_sessions_per_subject", 1)
        )
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
                split_summary = plot_gru_examples_for_split(
                    split_name=split_name,
                    output_dir=output_dir,
                    output_df=split_output_df,
                    network_states=np.asarray(network_states_full)[:, split_indices, :],
                    yhat_logits=np.asarray(yhat_full)[:, split_indices, :],
                    sessions_per_subject=split_sessions_per_subject,
                    max_subjects_to_plot=max_subjects_to_plot,
                    n_action_logits=n_action_logits,
                    wandb_run=None,
                    log_scope=log_scope,
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
