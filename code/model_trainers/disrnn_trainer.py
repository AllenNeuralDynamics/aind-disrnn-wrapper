from __future__ import annotations

import copy
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence
from dataclasses import asdict

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
import wandb
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf

import aind_disrnn_utils.data_loader as dl
import types
from disentangled_rnns.library import disrnn, multisubject_disrnn, plotting, rnn_utils

from base.interfaces import ModelTrainer
from base.types import DatasetBundle
from model_trainers.base_multisubject_trainer import BaseMultisubjectTrainer
from model_trainers.checkpoint_resume import (
    find_latest_resumable_state,
    save_train_state,
)
from models import multisubject_disrnn as local_multisubject_disrnn
from models.session_conditioning import (
    compute_session_curriculum_lambda,
    resolve_session_conditioning_from_architecture,
    resolve_session_curriculum_steps,
)
from utils.disrnn_evaluation import plot_disrnn_examples_for_split
from utils.disrnn_evaluation import (
    HeldoutEvalConfig,
    evaluate_disrnn_on_heldout_subjects,
    load_disrnn_heldout_subject_data,
)
from utils.disrnn_distillation import (
    build_teacher_ensemble,
    evaluate_distillation_loss,
    resolve_distillation_config,
    resolve_penalty_scale,
    train_network_with_distillation,
)
from utils.multisubject import (
    compute_session_conditioned_context_dataframe,
    convert_local_params_to_upstream_multisubject,
    extract_subject_embeddings_from_params,
    normalize_subject_id,
    ordered_session_context_rows,
    prepend_session_index_to_multisubject_split_datasets,
    resolve_session_context_plot_subject_indices,
    save_multisubject_metadata,
    save_subject_index_map,
    save_session_context_map,
    session_regularization_index_arrays_from_session_context,
    subject_embeddings_to_dataframe,
)
from utils.run_helpers import (
    compute_bottleneck_sparsity_metrics,
    resolve_disrnn_penalties,
)
from utils.session_regularized_training import (
    build_zero_mean_session_delta_regularization_apply,
    train_network_with_session_regularization,
)

logger = logging.getLogger(__name__)


def _to_dict(config: Mapping[str, Any] | DictConfig) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


def _is_multisubject_mode(
    architecture: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> bool:
    return bool(architecture.get("multisubject", False) or metadata.get("multisubject", False))


def _require_n_action_logits(dataset: Any, yhat: np.ndarray, *, context: str) -> int:
    """Strict action-logit count for disRNN outputs.

    The disRNN network appends a trailing penalty channel to its output, so a
    valid output has width ``n_classes + 1`` (logits followed by the penalty
    term). Mirrors the GRU trainer's strict helper: requires ``dataset.n_classes``
    to be a positive integer and validates the output width, rather than silently
    inferring the count from ``yhat.shape[2] - 1``.
    """
    n_action_logits = int(getattr(dataset, "n_classes", 0))
    if n_action_logits <= 0:
        raise ValueError(
            f"disRNN {context} requires dataset.n_classes to be set to a positive integer."
        )
    actual_output_size = int(np.asarray(yhat).shape[2])
    if actual_output_size != n_action_logits + 1:
        raise ValueError(
            f"disRNN {context} logits shape mismatch: dataset.n_classes={n_action_logits} "
            f"but yhat.shape[2]={actual_output_size} "
            "(expected n_classes + 1 for the trailing penalty channel)."
        )
    return n_action_logits


def _get_architecture_value(
    architecture: Mapping[str, Any],
    key: str,
    *,
    default: Any = None,
    alias: str | None = None,
) -> Any:
    if key in architecture:
        return architecture[key]
    if alias is not None and alias in architecture:
        return architecture[alias]
    return default


def _resolve_disrnn_observation_x_names(
    dataset: Any,
    *,
    obs_size: int,
    context: str,
) -> list[str]:
    """Return the behavioral observation labels consumed by the disRNN core.

    Multisubject runs prepend subject and optional session index features to
    the dataset tensor, but upstream disRNN configs validate ``x_names``
    against ``obs_size`` only. We therefore keep the trailing observation
    labels and drop any packed context feature names.
    """
    resolved_obs_size = int(obs_size)
    if resolved_obs_size <= 0:
        raise ValueError(f"{context} requires obs_size > 0, got {resolved_obs_size}.")

    x_names = list(getattr(dataset, "x_names", []) or [])
    if len(x_names) < resolved_obs_size:
        raise ValueError(
            f"{context} expected at least {resolved_obs_size} x_names entries, "
            f"got {x_names}."
        )
    return x_names[-resolved_obs_size:]


def _bind_session_curriculum_lambda(
    make_network: Any,
    *,
    session_curriculum_lambda: float,
) -> Any:
    return lambda: make_network(session_curriculum_lambda)


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


def _maybe_prepend_session_indices_to_datasets(
    *,
    dataset: Any,
    dataset_train: Any,
    dataset_eval: Any,
    session_conditioning_enabled: bool,
    metadata: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    if not session_conditioning_enabled:
        return dataset, dataset_train, dataset_eval
    return prepend_session_index_to_multisubject_split_datasets(
        dataset=dataset,
        dataset_train=dataset_train,
        dataset_eval=dataset_eval,
        metadata=metadata,
    )


def _validate_multisubject_dataset_inputs(
    dataset: Any,
    *,
    max_n_subjects: int,
    session_conditioning_enabled: bool,
    session_max_index_by_subject_index: Sequence[int] | None,
    context: str,
) -> None:
    xs = dataset.get_all()["xs"]
    xs = np.asarray(xs)
    min_feature_count = 3 if session_conditioning_enabled else 2
    if xs.ndim != 3:
        raise ValueError(
            f"Multisubject disRNN {context} expects dataset inputs to be 3D, got shape={xs.shape}."
        )
    if xs.shape[2] < min_feature_count:
        raise ValueError(
            f"Multisubject disRNN {context} requires {min_feature_count} input features "
            "including prepended subject/session indices."
        )

    subject_ids = np.asarray(np.rint(xs[..., 0]), dtype=int)
    invalid_subject_mask = (subject_ids < -1) | (subject_ids >= int(max_n_subjects))
    if np.any(invalid_subject_mask):
        bad_values = np.unique(subject_ids[invalid_subject_mask]).tolist()
        raise ValueError(
            "Multisubject disRNN "
            f"{context} encountered out-of-range subject ids {bad_values}."
        )

    if not session_conditioning_enabled:
        return
    if session_max_index_by_subject_index is None:
        raise ValueError(
            "Multisubject disRNN session conditioning validation requires "
            "session_max_index_by_subject_index."
        )

    session_max = np.asarray(session_max_index_by_subject_index, dtype=int)
    session_ids = np.asarray(np.rint(xs[..., 1]), dtype=int)
    valid_subject_mask = subject_ids >= 0
    if np.any(valid_subject_mask):
        valid_session_ids = session_ids[valid_subject_mask]
        subject_specific_max = session_max[subject_ids[valid_subject_mask]]
        invalid_session_mask = np.logical_or(
            valid_session_ids < 1,
            valid_session_ids > subject_specific_max,
        )
        if np.any(invalid_session_mask):
            bad_values = np.unique(valid_session_ids[invalid_session_mask]).tolist()
            raise ValueError(
                "Multisubject disRNN "
                f"{context} encountered out-of-range session ids {bad_values}."
            )

    invalid_padded_session_mask = np.logical_and(subject_ids < 0, session_ids != -1)
    if np.any(invalid_padded_session_mask):
        bad_values = np.unique(session_ids[invalid_padded_session_mask]).tolist()
        raise ValueError(
            "Multisubject disRNN "
            f"{context} expected padded rows to use session id -1, got {bad_values}."
        )


class DisrnnTrainer(BaseMultisubjectTrainer):
    """Trainer that reproduces the legacy disRNN pipeline."""

    _MODEL_LABEL = "disRNN"
    _TRAINER_CONTEXT_NAME = "DisrnnTrainer"

    def __init__(
        self,
        architecture: Mapping[str, Any] | DictConfig,
        penalties: Mapping[str, Any] | DictConfig,
        training: Mapping[str, Any] | DictConfig,
        distillation: Mapping[str, Any] | DictConfig | None = None,
        heldout_data: dict[str, Any] | None = None,
        output_dir: str = "/results/outputs",
        seed: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__(
            architecture=architecture,
            training=training,
            heldout_data=heldout_data,
            output_dir=output_dir,
            seed=seed,
        )
        self.penalties = resolve_disrnn_penalties(penalties)
        self.distillation = resolve_distillation_config(
            _to_dict(distillation) if distillation is not None else None
        )

    def _build_network_configs(
        self,
        *,
        dataset: Any,
        ignore_policy: str,
        metadata: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        """Create train/eval configs for single- or multisubject disRNN."""
        output_size = 2 if ignore_policy == "exclude" else 3
        is_multisubject = _is_multisubject_mode(self.architecture, metadata)

        if is_multisubject:
            num_subjects = int(
                metadata.get("num_subjects")
                or len(metadata.get("subject_ids", []))
            )
            if num_subjects <= 0:
                raise ValueError(
                    "Multisubject disRNN requires metadata.num_subjects or metadata.subject_ids."
                )
            plot_subject_index = int(self.training.get("plot_subject_index", 0))
            if plot_subject_index < 0 or plot_subject_index >= num_subjects:
                raise ValueError(
                    "training.plot_subject_index must be in "
                    f"[0, {num_subjects - 1}], got {plot_subject_index}."
                )
            subject_embedding_size = _get_architecture_value(
                self.architecture,
                "subject_embedding_size",
            )
            if subject_embedding_size is None or int(subject_embedding_size) <= 0:
                raise ValueError(
                    "Multisubject disRNN requires architecture.subject_embedding_size > 0."
                )
            session_conditioning_cfg = resolve_session_conditioning_from_architecture(
                architecture=self.architecture,
                metadata=metadata,
                multisubject=is_multisubject,
                max_n_subjects=num_subjects,
                subject_embedding_size=int(subject_embedding_size),
                context="DisrnnTrainer",
            )
            packed_context_feature_count = (
                2 if bool(session_conditioning_cfg["enabled"]) else 1
            )
            obs_size = int(dataset._xs.shape[2] - packed_context_feature_count)

            config = local_multisubject_disrnn.MultisubjectDisRnnConfig(
                obs_size=obs_size,
                output_size=output_size,
                x_names=_resolve_disrnn_observation_x_names(
                    dataset,
                    obs_size=obs_size,
                    context="Multisubject disRNN config",
                ),
                y_names=dataset.y_names,
                latent_size=self.architecture["latent_size"],
                update_net_n_units_per_layer=self.architecture[
                    "update_net_n_units_per_layer"
                ],
                update_net_n_layers=self.architecture["update_net_n_layers"],
                choice_net_n_units_per_layer=self.architecture[
                    "choice_net_n_units_per_layer"
                ],
                choice_net_n_layers=self.architecture["choice_net_n_layers"],
                activation=self.architecture["activation"],
                noiseless_mode=False,
                latent_penalty=self.penalties["latent_penalty"],
                choice_net_latent_penalty=self.penalties["choice_net_latent_penalty"],
                update_net_obs_penalty=self.penalties["update_net_obs_penalty"],
                update_net_latent_penalty=self.penalties["update_net_latent_penalty"],
                max_n_subjects=num_subjects,
                subject_embedding_size=int(subject_embedding_size),
                subject_embedding_init=str(
                    self.architecture.get("subject_embedding_init", "zeros")
                ),
                session_encoding_type=str(
                    session_conditioning_cfg["session_encoding_type"]
                ),
                session_integration_type=str(
                    session_conditioning_cfg["session_integration_type"]
                ),
                session_fourier_k=int(session_conditioning_cfg["session_fourier_k"]),
                session_delta_n_layers=int(session_conditioning_cfg["session_delta_n_layers"]),
                session_delta_hidden_size=int(
                    session_conditioning_cfg["session_delta_hidden_size"]
                ),
                # `.get(key, 0)` only defaults on a *missing* key; the config
                # ships these as explicit `null`, so a present-but-None value
                # would reach int() and raise. The forward pass gates on
                # session_curriculum_lambda (default 1.0), so 0 here is correct
                # whenever these aren't pre-resolved (e.g. held-out fine-tuning,
                # which calls _build_network_configs directly without the
                # resolve_session_curriculum_steps write-back that train() does).
                session_n_pretrain_steps=int(
                    self.architecture.get("session_n_pretrain_steps") or 0
                ),
                session_n_warmup_steps=int(
                    self.architecture.get("session_n_warmup_steps") or 0
                ),
                session_max_index_by_subject_index=list(
                    session_conditioning_cfg["session_max_index_by_subject_index"]
                ),
                use_global_subject_bottleneck=bool(
                    self.architecture.get("use_global_subject_bottleneck", True)
                ),
                subj_penalty=float(self.penalties.get("subject_penalty", 0.0)),
                update_net_subj_penalty=float(
                    self.penalties.get("update_net_subject_penalty", 0.0)
                ),
                choice_net_subj_penalty=float(
                    self.penalties.get("choice_net_subject_penalty", 0.0)
                ),
            )
        else:
            resolve_session_conditioning_from_architecture(
                architecture=self.architecture,
                metadata=metadata,
                multisubject=is_multisubject,
                max_n_subjects=None,
                subject_embedding_size=None,
                context="DisrnnTrainer",
            )
            obs_size = int(dataset._xs.shape[2])
            config = disrnn.DisRnnConfig(
                obs_size=obs_size,
                output_size=output_size,
                x_names=_resolve_disrnn_observation_x_names(
                    dataset,
                    obs_size=obs_size,
                    context="Single-subject disRNN config",
                ),
                y_names=dataset.y_names,
                latent_size=self.architecture["latent_size"],
                update_net_n_units_per_layer=self.architecture[
                    "update_net_n_units_per_layer"
                ],
                update_net_n_layers=self.architecture["update_net_n_layers"],
                choice_net_n_units_per_layer=self.architecture[
                    "choice_net_n_units_per_layer"
                ],
                choice_net_n_layers=self.architecture["choice_net_n_layers"],
                activation=self.architecture["activation"],
                noiseless_mode=False,
                latent_penalty=self.penalties["latent_penalty"],
                choice_net_latent_penalty=self.penalties["choice_net_latent_penalty"],
                update_net_obs_penalty=self.penalties["update_net_obs_penalty"],
                update_net_latent_penalty=self.penalties["update_net_latent_penalty"],
            )

        # Warmup trains against this *noiseless* copy with every bottleneck penalty
        # zeroed, so the warmup phase is effectively penalty-free even though it
        # still uses loss="penalized_categorical" (the penalty term sums to 0).
        # This zeroing is load-bearing; see
        # test_warmup_uses_penalty_free_noiseless_network.
        noiseless_network = copy.deepcopy(config)
        noiseless_network.latent_penalty = 0
        noiseless_network.choice_net_latent_penalty = 0
        noiseless_network.update_net_obs_penalty = 0
        noiseless_network.update_net_latent_penalty = 0
        noiseless_network.l2_scale = 0
        noiseless_network.noiseless_mode = True
        return config, noiseless_network

    def _make_network_factory(
        self,
        config: Any,
        *,
        multisubject: bool,
    ) -> Callable[..., Any]:
        if multisubject:
            def _make_network(session_curriculum_lambda: float = 1.0) -> Any:
                runtime_config = copy.copy(config)
                setattr(
                    runtime_config,
                    "session_curriculum_lambda",
                    session_curriculum_lambda,
                )
                return local_multisubject_disrnn.MultisubjectDisRnn(runtime_config)

            return _make_network

        return lambda session_curriculum_lambda=1.0: disrnn.HkDisentangledRNN(config)

    def _subject_plot_context(
        self,
        *,
        params: Any,
        multisubject: bool,
    ) -> tuple[np.ndarray | None, int | None]:
        if not multisubject:
            return None, None

        plot_subject_index = int(self.training.get("plot_subject_index", 0))
        subject_embeddings = extract_subject_embeddings_from_params(params)
        if plot_subject_index < 0 or plot_subject_index >= subject_embeddings.shape[0]:
            raise ValueError(
                "training.plot_subject_index is out of range for saved subject embeddings: "
                f"{plot_subject_index} not in [0, {subject_embeddings.shape[0] - 1}]"
            )
        return np.asarray(subject_embeddings[plot_subject_index], dtype=float), plot_subject_index












    def _resolve_n_action_logits(self, dataset, yhat, *, context):
        return _require_n_action_logits(dataset, yhat, context=context)

    def _add_model_results(self, raw_df, network_states, yhat, *, ignore_policy):
        return dl.add_model_results(
            raw_df, network_states, yhat, ignore_policy=ignore_policy
        )

    def _evaluate_heldout_subjects(
        self, *, heldout_eval_cfg, wandb_run, params_path, output_subdir, heldout_data, log_scope
    ):
        return evaluate_disrnn_on_heldout_subjects(
            heldout_eval_cfg,
            wandb_run=wandb_run,
            params_path=params_path,
            output_subdir=output_subdir,
            log_to_wandb=False,
            heldout_data=heldout_data,
            log_scope=log_scope,
        )

    def _snapshot_extra_summary_fields(self, *, session_curriculum_lambda):
        return {"session_curriculum_lambda": float(session_curriculum_lambda)}

    def _plot_model_specific_diagnostics(
        self,
        *,
        params: Any,
        disrnn_config: Any,
        stage_dir: Path,
        stage_name: str,
        is_multisubject: bool,
        metadata: dict[str, Any],
        raw_df: Any,
        session_curriculum_lambda: float,
        wandb_run: Any | None,
        wandb_step: int | None,
        keep_media_files: bool,
    ) -> dict[str, Any]:
        plot_paths: dict[str, Any] = {
            "bottlenecks": None,
            "choice_rule": None,
            "subject_embedding_state_space": None,
            "subject_session_context_state_space": None,
            "update_rules": [],
        }
        try:
            bottlenecks_fig = plotting.plot_bottlenecks(
                params,
                disrnn_config,
                sort_latents=False,
            )
            bottlenecks_fig.tight_layout()
            bottlenecks_path = stage_dir / "bottlenecks.png"
            bottlenecks_fig.savefig(bottlenecks_path)
            plt.close(bottlenecks_fig)
            plot_paths["bottlenecks"] = str(bottlenecks_path)

            if is_multisubject:
                subject_embedding_fig = self._plot_subject_embedding_state_space(
                    params=params,
                    raw_df=raw_df,
                    metadata=metadata,
                )
                if subject_embedding_fig is not None:
                    subject_embedding_path = stage_dir / "subject_embedding_state_space.png"
                    subject_embedding_fig.savefig(subject_embedding_path)
                    plt.close(subject_embedding_fig)
                    plot_paths["subject_embedding_state_space"] = str(subject_embedding_path)
                subject_session_context_fig = (
                    self._plot_subject_session_context_state_space(
                        params=params,
                        raw_df=raw_df,
                        metadata=metadata,
                        session_curriculum_lambda=session_curriculum_lambda,
                    )
                )
                if subject_session_context_fig is not None:
                    subject_session_context_path = (
                        stage_dir / "subject_session_context_state_space.png"
                    )
                    subject_session_context_fig.savefig(subject_session_context_path)
                    plt.close(subject_session_context_fig)
                    plot_paths["subject_session_context_state_space"] = str(
                        subject_session_context_path
                    )

            plot_subject_embedding, plot_subject_index = self._subject_plot_context(
                params=params,
                multisubject=is_multisubject,
            )
            should_plot_choice_rule = bool(
                self.training.get("plot_choice_rule", False)
                or self.training.get("checkpoint_plot_choice_rule", False)
            )
            if should_plot_choice_rule:
                choice_fig = plotting.plot_choice_rule(
                    params,
                    disrnn_config,
                    subj_embedding=plot_subject_embedding,
                )
                if choice_fig is not None:
                    for ax in choice_fig.get_axes():
                        ax.axhline(0, alpha=.5)
                        ax.axvline(0, alpha=.5)
                    choice_fig.tight_layout()
                    choice_path = stage_dir / "choice_rule.png"
                    choice_fig.savefig(choice_path)
                    plt.close(choice_fig)
                    plot_paths["choice_rule"] = str(choice_path)

            should_plot_update_rules = bool(
                self.training.get("plot_update_rules", False)
                or self.training.get("checkpoint_plot_update_rules", False)
            )
            if should_plot_update_rules:
                params_for_update_rules = (
                    convert_local_params_to_upstream_multisubject(params)
                    if is_multisubject
                    else params
                )
                update_figs = plotting.plot_update_rules(
                    params_for_update_rules,
                    disrnn_config,
                    subj_ind=plot_subject_index,
                )
                update_rule_paths: list[str] = []
                for index, fig in enumerate(update_figs):
                    fig.tight_layout()
                    update_path = stage_dir / f"update_rule_{index}.png"
                    fig.savefig(update_path)
                    plt.close(fig)
                    update_rule_paths.append(str(update_path))
                plot_paths["update_rules"] = update_rule_paths
        except Exception as exc:
            logger.warning("Initialization plotting failed for %s: %s", stage_name, exc)

        if wandb_run is not None:
            plot_payload = {}
            if plot_paths["bottlenecks"]:
                plot_payload["checkpoint/fig/bottlenecks"] = wandb.Image(
                    str(plot_paths["bottlenecks"])
                )
            if plot_paths["subject_embedding_state_space"]:
                plot_payload["checkpoint/fig/subject_embedding_state_space"] = wandb.Image(
                    str(plot_paths["subject_embedding_state_space"])
                )
            if plot_paths["subject_session_context_state_space"]:
                plot_payload["checkpoint/fig/subject_session_context_state_space"] = (
                    wandb.Image(str(plot_paths["subject_session_context_state_space"]))
                )
            if plot_paths["choice_rule"]:
                plot_payload["checkpoint/fig/choice_rule"] = wandb.Image(
                    str(plot_paths["choice_rule"])
                )
            for index, update_rule_path in enumerate(plot_paths["update_rules"]):
                plot_payload[f"checkpoint/fig/update_rule_{index}"] = wandb.Image(
                    str(update_rule_path)
                )
            if plot_payload:
                if wandb_step is None:
                    wandb_run.log(plot_payload)
                else:
                    wandb_run.log(plot_payload, step=wandb_step)
                if not keep_media_files:
                    self._remove_media_files(
                        [
                            plot_paths["bottlenecks"],
                            plot_paths["subject_embedding_state_space"],
                            plot_paths["subject_session_context_state_space"],
                            plot_paths["choice_rule"],
                            *plot_paths["update_rules"],
                        ]
                    )
                    plot_paths["bottlenecks"] = None
                    plot_paths["subject_embedding_state_space"] = None
                    plot_paths["subject_session_context_state_space"] = None
                    plot_paths["choice_rule"] = None
                    plot_paths["update_rules"] = []
        return plot_paths

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
        is_multisubject = _is_multisubject_mode(self.architecture, metadata)
        output["multisubject"] = bool(is_multisubject)
        max_n_subjects = None
        if is_multisubject:
            max_n_subjects = int(
                metadata.get("num_subjects")
                or len(metadata.get("subject_ids", []))
            )
        session_conditioning_cfg = resolve_session_conditioning_from_architecture(
            architecture=self.architecture,
            metadata=metadata,
            multisubject=is_multisubject,
            max_n_subjects=max_n_subjects,
            subject_embedding_size=(
                int(self.architecture["subject_embedding_size"]) if is_multisubject else None
            ),
            context="DisrnnTrainer",
        )
        dataset, dataset_train, dataset_eval = _maybe_prepend_session_indices_to_datasets(
            dataset=dataset,
            dataset_train=dataset_train,
            dataset_eval=dataset_eval,
            session_conditioning_enabled=bool(session_conditioning_cfg["enabled"]),
            metadata=metadata,
        )

        # --- Length-bucketed batching (opt-in via training.length_bucketing) ---
        # Trim the per-batch unroll to the batch's session length instead of the
        # global T_max, cutting padding compute. The bucketed-sampling logic lives
        # in the shared _sample_batch (utils/session_regularized_training.py); it
        # only fires when the train dataset carries `length_bucketing=True` and
        # batch_mode == "random". Mirrors gru_trainer.py's wiring so both trainers
        # share one implementation. Set on the train dataset that session-
        # regularized training samples from (after the session-index prepend so
        # feature-0 padding markers are already in place).
        if bool(self.training.get("length_bucketing", False)):
            dataset_train.length_bucketing = True
            dataset_train.length_bucket_grid = int(
                self.training.get("length_bucket_grid", 128)
            )
            logger.info(
                "Length-bucketed batching enabled (grid=%s)",
                dataset_train.length_bucket_grid,
            )

        total_session_curriculum_steps = int(self.training.get("n_warmup_steps", 0)) + int(
            self.training.get("n_steps", 0)
        )
        session_curriculum_cfg = resolve_session_curriculum_steps(
            total_training_steps=total_session_curriculum_steps,
            session_n_pretrain_steps=self.architecture.get("session_n_pretrain_steps"),
            session_n_warmup_steps=self.architecture.get("session_n_warmup_steps"),
            context="DisrnnTrainer",
        )
        self.architecture["session_n_pretrain_steps"] = int(
            session_curriculum_cfg["session_n_pretrain_steps"]
        )
        self.architecture["session_n_warmup_steps"] = int(
            session_curriculum_cfg["session_n_warmup_steps"]
        )
        output["session_curriculum"] = {
            **session_curriculum_cfg,
            "total_training_steps": int(total_session_curriculum_steps),
        }

        _log_dataset_split_details(
            dataset=dataset,
            dataset_train=dataset_train,
            dataset_eval=dataset_eval,
        )
        _log_heldout_dataset_details(self.heldout_data)
        if is_multisubject and max_n_subjects is not None:
            _validate_multisubject_dataset_inputs(
                dataset,
                max_n_subjects=max_n_subjects,
                session_conditioning_enabled=bool(session_conditioning_cfg["enabled"]),
                session_max_index_by_subject_index=session_conditioning_cfg[
                    "session_max_index_by_subject_index"
                ],
                context="full dataset",
            )
            _validate_multisubject_dataset_inputs(
                dataset_train,
                max_n_subjects=max_n_subjects,
                session_conditioning_enabled=bool(session_conditioning_cfg["enabled"]),
                session_max_index_by_subject_index=session_conditioning_cfg[
                    "session_max_index_by_subject_index"
                ],
                context="training split",
            )
            _validate_multisubject_dataset_inputs(
                dataset_eval,
                max_n_subjects=max_n_subjects,
                session_conditioning_enabled=bool(session_conditioning_cfg["enabled"]),
                session_max_index_by_subject_index=session_conditioning_cfg[
                    "session_max_index_by_subject_index"
                ],
                context="eval split",
            )
            self._log_session_conditioning_details(
                metadata=metadata,
                dataset=dataset,
                session_conditioning_cfg=session_conditioning_cfg,
                session_curriculum_cfg=session_curriculum_cfg,
                total_training_steps=total_session_curriculum_steps,
            )
            if bool(session_conditioning_cfg["enabled"]) and bool(
                self.training.get("plot_session_context_state_space", True)
            ):
                self._resolve_session_context_plot_subject_indices(metadata=metadata)

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
            auto_resume=bool(self.training.get("auto_resume", True)),
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
            initialization_eval_before_warmup=bool(
                self.training.get("initialization_eval_before_warmup", True)
            ),
            initialization_eval_after_warmup=bool(
                self.training.get("initialization_eval_after_warmup", True)
            ),
            plot_choice_rule=bool(self.training.get("plot_choice_rule", False)),
            plot_update_rules=bool(self.training.get("plot_update_rules", False)),
            save_output_df=bool(self.training.get("save_output_df", False)),
        )

        logger.info(f"max_grad_norm = {args.max_grad_norm}")
        # Fail fast on invalid step counts before any warmup/initialization eval
        # runs (mirrors GruTrainer's validation just after the max_grad_norm log).
        if args.n_steps < 0:
            raise ValueError("training.n_steps must be >= 0")
        if args.n_warmup_steps < 0:
            raise ValueError("training.n_warmup_steps must be >= 0")
        disrnn_config, noiseless_network = self._build_network_configs(
            dataset=dataset,
            ignore_policy=ignore_policy,
            metadata=metadata,
        )
        if hasattr(disrnn_config, "session_n_pretrain_steps"):
            disrnn_config.session_n_pretrain_steps = int(
                session_curriculum_cfg["session_n_pretrain_steps"]
            )
            disrnn_config.session_n_warmup_steps = int(
                session_curriculum_cfg["session_n_warmup_steps"]
            )
        if hasattr(noiseless_network, "session_n_pretrain_steps"):
            noiseless_network.session_n_pretrain_steps = int(
                session_curriculum_cfg["session_n_pretrain_steps"]
            )
            noiseless_network.session_n_warmup_steps = int(
                session_curriculum_cfg["session_n_warmup_steps"]
            )
        make_train_network = self._make_network_factory(
            disrnn_config,
            multisubject=is_multisubject,
        )
        make_noiseless_network = self._make_network_factory(
            noiseless_network,
            multisubject=is_multisubject,
        )

        def _session_curriculum_lambda_for_total_step(current_step: int) -> float:
            if not bool(session_conditioning_cfg["enabled"]):
                return 1.0
            return compute_session_curriculum_lambda(
                current_step=current_step,
                session_n_pretrain_steps=int(
                    session_curriculum_cfg["session_n_pretrain_steps"]
                ),
                session_n_warmup_steps=int(
                    session_curriculum_cfg["session_n_warmup_steps"]
                ),
            )

        def _session_curriculum_lambda_for_completed_total_steps(
            completed_total_steps: int,
        ) -> float:
            if int(completed_total_steps) <= 0:
                return _session_curriculum_lambda_for_total_step(0)
            return _session_curriculum_lambda_for_total_step(
                int(completed_total_steps) - 1
            )

        def _make_noiseless_eval_network_for_completed_total_steps(
            completed_total_steps: int,
        ) -> Any:
            return _bind_session_curriculum_lambda(
                make_noiseless_network,
                session_curriculum_lambda=(
                    _session_curriculum_lambda_for_completed_total_steps(
                        completed_total_steps
                    )
                ),
            )

        expected_n_action_logits = int(getattr(dataset, "n_classes", 0))
        if expected_n_action_logits <= 0:
            expected_n_action_logits = 2 if ignore_policy == "exclude" else 3
        lambda_reg_session = float(self.training.get("lambda_reg_session", 0.0))
        if lambda_reg_session < 0:
            raise ValueError("training.lambda_reg_session must be >= 0.")
        if lambda_reg_session > 0 and not bool(session_conditioning_cfg["enabled"]):
            logger.warning(
                "training.lambda_reg_session=%s but session conditioning is disabled "
                "(session_encoding_type=none); skipping session-delta regularization.",
                lambda_reg_session,
            )
            lambda_reg_session = 0.0
        train_session_regularization_apply = None
        warmup_session_regularization_apply = None
        if lambda_reg_session > 0:
            session_context = metadata.get("session_context")
            if not isinstance(session_context, Mapping):
                raise ValueError(
                    "training.lambda_reg_session requires metadata.session_context."
                )
            if max_n_subjects is None:
                raise ValueError(
                    "training.lambda_reg_session requires multisubject metadata.num_subjects."
                )
            reg_subject_indices, reg_session_indices = (
                session_regularization_index_arrays_from_session_context(session_context)
            )
            warmup_session_regularization_apply = (
                build_zero_mean_session_delta_regularization_apply(
                    make_network=make_noiseless_network,
                    subject_indices=reg_subject_indices,
                    session_indices=reg_session_indices,
                    max_n_subjects=int(max_n_subjects),
                )
            )
            train_session_regularization_apply = (
                build_zero_mean_session_delta_regularization_apply(
                    make_network=make_train_network,
                    subject_indices=reg_subject_indices,
                    session_indices=reg_session_indices,
                    max_n_subjects=int(max_n_subjects),
                )
            )
            logger.info(
                "Zero-mean session-delta regularization enabled: lambda_reg_session=%s "
                "over %d subject-session pairs.",
                lambda_reg_session,
                int(reg_subject_indices.shape[0]),
            )
        _train_all = dataset_train.get_all()
        xs_train_all, ys_train_all = _train_all["xs"], _train_all["ys"]
        _eval_all = dataset_eval.get_all()
        xs_eval_all, ys_eval_all = _eval_all["xs"], _eval_all["ys"]
        distillation_penalty_scale = resolve_penalty_scale(args.loss_param)
        distillation_ensemble = None
        if self.distillation.enabled:
            logger.info(
                "Building GRU teacher ensemble for distillation from %d run(s).",
                len(self.distillation.teacher_model_dirs),
            )
            distillation_ensemble = build_teacher_ensemble(
                distillation=self.distillation,
                dataset=dataset,
                dataset_train=dataset_train,
                dataset_eval=dataset_eval,
                metadata=metadata,
                output_dir=self.output_dir,
                expected_output_size=expected_n_action_logits,
            )
            output["distillation"] = dict(distillation_ensemble.manifest)

        def _train_supervised_with_optional_session_regularization(
            *,
            make_current_network: Callable[..., Any],
            current_session_regularization_apply: Any | None,
            params: Any | None = None,
            opt_state: Any | None = None,
            n_steps: int,
            random_key: Any,
            optimizer: Any,
            wandb_step_offset: int = 0,
            total_step_offset: int = 0,
            log_bottleneck_sparsity: bool = False,
        ):
            return train_network_with_session_regularization(
                make_current_network,
                dataset_train,
                dataset_eval,
                loss=args.loss,
                loss_param=args.loss_param,
                n_action_logits=expected_n_action_logits,
                session_regularization_apply=current_session_regularization_apply,
                session_regularization_scale=lambda_reg_session,
                session_curriculum_lambda_schedule=(
                    lambda local_step: _session_curriculum_lambda_for_total_step(
                        total_step_offset + local_step
                    )
                ),
                opt=optimizer,
                params=params,
                opt_state=opt_state,
                n_steps=n_steps,
                max_grad_norm=args.max_grad_norm,
                random_key=random_key,
                report_progress_by="wandb",
                wandb_run=wandb_run,
                wandb_step_offset=wandb_step_offset,
                bottleneck_sparsity_fn=(
                    compute_bottleneck_sparsity_metrics
                    if log_bottleneck_sparsity
                    else None
                ),
            )

        def _compute_distillation_metrics(
            current_params: Any,
            *,
            completed_total_steps: int,
        ) -> tuple[float, float] | None:
            if distillation_ensemble is None:
                return None
            train_metric = evaluate_distillation_loss(
                make_network=_make_noiseless_eval_network_for_completed_total_steps(
                    completed_total_steps
                ),
                params=current_params,
                xs=xs_train_all,
                ys=ys_train_all,
                teacher_probs=distillation_ensemble.train_probs,
                temperature=distillation_ensemble.config.temperature,
                n_action_logits=expected_n_action_logits,
                include_penalty=False,
                penalty_scale=distillation_penalty_scale,
            )
            eval_metric = evaluate_distillation_loss(
                make_network=_make_noiseless_eval_network_for_completed_total_steps(
                    completed_total_steps
                ),
                params=current_params,
                xs=xs_eval_all,
                ys=ys_eval_all,
                teacher_probs=distillation_ensemble.eval_probs,
                temperature=distillation_ensemble.config.temperature,
                n_action_logits=expected_n_action_logits,
                include_penalty=False,
                penalty_scale=distillation_penalty_scale,
            )
            return float(train_metric), float(eval_metric)

        heldout_eval_cfg = None
        heldout_test_data = self.heldout_data
        runtime_heldout_cfg = HeldoutEvalConfig.from_data_cfg(metadata)
        if runtime_heldout_cfg.enabled and is_multisubject:
            logger.info(
                "Skipping PER-CHECKPOINT held-out eval (checkpoint_run_heldout_eval) "
                "for multisubject disRNN; v1 supports seen-subject personalization only. "
                "NOTE: this does NOT disable the end-of-training held-out fine-tune "
                "(auto_heldout_finetune) — that runs separately and logs heldout/* metrics."
            )
        elif runtime_heldout_cfg.enabled:
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
                    _log_heldout_dataset_details(heldout_test_data)
                except Exception as exc:
                    logger.warning(
                        "Preloading held-out test data failed; evaluation will fall back to lazy loading: %s",
                        exc,
                    )

        # --- Resume from latest full-state checkpoint (preemption recovery) ---
        # Only the chunked checkpoint path writes resumable state, so resume is
        # scoped to it. A found checkpoint already has warmup folded into its
        # params, so the warmup phase (and its one-time init evals) is skipped.
        warmup_duration = 0.0
        resume_state = None
        if args.auto_resume and args.checkpoint_every_n_steps > 0:
            resume_state = find_latest_resumable_state(self.output_dir / "checkpoints")
        if resume_state is not None:
            logger.info(
                "Resuming disRNN training from step %s; skipping warmup",
                resume_state.steps_completed,
            )
            params = resume_state.params
        else:
            params, warmup_opt_state, _ = _train_supervised_with_optional_session_regularization(
                make_current_network=make_noiseless_network,
                current_session_regularization_apply=warmup_session_regularization_apply,
                n_steps=0,
                random_key=warmup_key,
                optimizer=optax.adam(args.learning_rate),
                total_step_offset=0,
            )
            # Step-0 bottleneck-sparsity baseline. The library initializes every
            # bottleneck sigma from RandomUniform(0, 0.05) -> all bottlenecks OPEN
            # (frac_open == 1.0) at init. Logging here (at wandb step 0, before
            # warmup/penalty pressure) captures that baseline so the open->closed
            # trajectory driven by update_net_latent_penalty_multiplier is visible
            # rather than jumping straight to the first periodic checkpoint.
            if wandb_run is not None:
                init_sparsity = compute_bottleneck_sparsity_metrics(params)
                if init_sparsity:
                    wandb_run.log(
                        {**init_sparsity, "checkpoint/step": 0},
                        step=0,
                    )
            initial_evaluations: dict[str, Any] = {}
            if args.initialization_eval_before_warmup:
                initial_evaluations["before_warmup"] = self._evaluate_initialization_snapshot(
                    stage_name="before_warmup",
                    params=params,
                    bundle=bundle,
                    metadata=metadata,
                    dataset=dataset,
                    dataset_train=dataset_train,
                    dataset_eval=dataset_eval,
                    ignore_policy=ignore_policy,
                    make_eval_network=(
                        _make_noiseless_eval_network_for_completed_total_steps(0)
                    ),
                    disrnn_config=disrnn_config,
                    is_multisubject=is_multisubject,
                    heldout_eval_cfg=heldout_eval_cfg,
                    heldout_data=heldout_test_data,
                    wandb_run=wandb_run,
                    wandb_step=0,
                    keep_media_files=args.checkpoint_keep_media_files,
                    session_curriculum_lambda=(
                        _session_curriculum_lambda_for_completed_total_steps(0)
                    ),
                )
                distillation_metrics = _compute_distillation_metrics(
                    params,
                    completed_total_steps=0,
                )
                if distillation_metrics is not None:
                    (
                        initial_evaluations["before_warmup"]["train_distillation_loss"],
                        initial_evaluations["before_warmup"]["eval_distillation_loss"],
                    ) = distillation_metrics
            if initial_evaluations:
                output["initial_evaluations"] = initial_evaluations

            logger.info("Running warmup training phase")
            warmup_start = time.time()
            if distillation_ensemble is None:
                params, warmup_opt_state, warmup_losses = (
                    _train_supervised_with_optional_session_regularization(
                        make_current_network=make_noiseless_network,
                        current_session_regularization_apply=warmup_session_regularization_apply,
                        params=params,
                        opt_state=warmup_opt_state,
                        n_steps=args.n_warmup_steps,
                        random_key=warmup_key,
                        optimizer=optax.adam(args.learning_rate),
                        total_step_offset=0,
                    )
                )
            else:
                params, warmup_opt_state, warmup_losses = train_network_with_distillation(
                    make_noiseless_network,
                    dataset_train,
                    dataset_eval,
                    training_teacher_probs=distillation_ensemble.train_probs,
                    validation_teacher_probs=distillation_ensemble.eval_probs,
                    opt=optax.adam(args.learning_rate),
                    params=params,
                    opt_state=warmup_opt_state,
                    n_steps=args.n_warmup_steps,
                    max_grad_norm=args.max_grad_norm,
                    random_key=warmup_key,
                    temperature=distillation_ensemble.config.temperature,
                    n_action_logits=expected_n_action_logits,
                    include_penalty=False,
                    penalty_scale=distillation_penalty_scale,
                    session_regularization_apply=warmup_session_regularization_apply,
                    session_regularization_scale=lambda_reg_session,
                    session_curriculum_lambda_schedule=_session_curriculum_lambda_for_total_step,
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
            if args.initialization_eval_after_warmup:
                initial_evaluations = output.setdefault("initial_evaluations", {})
                initial_evaluations["after_warmup"] = self._evaluate_initialization_snapshot(
                    stage_name="after_warmup",
                    params=params,
                    bundle=bundle,
                    metadata=metadata,
                    dataset=dataset,
                    dataset_train=dataset_train,
                    dataset_eval=dataset_eval,
                    ignore_policy=ignore_policy,
                    make_eval_network=_make_noiseless_eval_network_for_completed_total_steps(
                        int(args.n_warmup_steps)
                    ),
                    disrnn_config=disrnn_config,
                    is_multisubject=is_multisubject,
                    heldout_eval_cfg=heldout_eval_cfg,
                    heldout_data=heldout_test_data,
                    wandb_run=wandb_run,
                    wandb_step=int(args.n_warmup_steps),
                    keep_media_files=args.checkpoint_keep_media_files,
                    session_curriculum_lambda=(
                        _session_curriculum_lambda_for_completed_total_steps(
                            int(args.n_warmup_steps)
                        )
                    ),
                )
                distillation_metrics = _compute_distillation_metrics(
                    params,
                    completed_total_steps=int(args.n_warmup_steps),
                )
                if distillation_metrics is not None:
                    (
                        initial_evaluations["after_warmup"]["train_distillation_loss"],
                        initial_evaluations["after_warmup"]["eval_distillation_loss"],
                    ) = distillation_metrics

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
            if distillation_ensemble is None:
                params, opt_state, losses = _train_supervised_with_optional_session_regularization(
                    make_current_network=make_train_network,
                    current_session_regularization_apply=train_session_regularization_apply,
                    params=params,
                    opt_state=None,
                    n_steps=args.n_steps,
                    random_key=training_key,
                    optimizer=optax.adam(args.learning_rate),
                    wandb_step_offset=args.n_warmup_steps,
                    total_step_offset=int(args.n_warmup_steps),
                    log_bottleneck_sparsity=True,
                )
            else:
                params, opt_state, losses = train_network_with_distillation(
                    make_train_network,
                    dataset_train,
                    dataset_eval,
                    training_teacher_probs=distillation_ensemble.train_probs,
                    validation_teacher_probs=distillation_ensemble.eval_probs,
                    params=params,
                    opt_state=None,
                    opt=optax.adam(args.learning_rate),
                    n_steps=args.n_steps,
                    max_grad_norm=args.max_grad_norm,
                    random_key=training_key,
                    temperature=distillation_ensemble.config.temperature,
                    n_action_logits=expected_n_action_logits,
                    include_penalty=True,
                    penalty_scale=distillation_penalty_scale,
                    session_regularization_apply=train_session_regularization_apply,
                    session_regularization_scale=lambda_reg_session,
                    session_curriculum_lambda_schedule=(
                        lambda local_step: _session_curriculum_lambda_for_total_step(
                            int(args.n_warmup_steps) + local_step
                        )
                    ),
                    report_progress_by="wandb",
                    wandb_run=wandb_run,
                    wandb_step_offset=args.n_warmup_steps,
                )
        else:
            optimizer = optax.adam(args.learning_rate)
            checkpoint_root = self.output_dir / "checkpoints"
            checkpoint_root.mkdir(parents=True, exist_ok=True)

            if resume_state is not None:
                all_training_losses = list(resume_state.training_losses)
                all_validation_losses = list(resume_state.validation_losses)
                steps_completed = resume_state.steps_completed
                opt_state = resume_state.opt_state
                random_key = resume_state.random_key
            else:
                all_training_losses = []
                all_validation_losses = []
                steps_completed = 0
                opt_state = None
                random_key = training_key
            xs_full_for_checkpoint = dataset.get_all()["xs"]
            df_for_checkpoint = bundle.raw

            while steps_completed < args.n_steps:
                chunk_steps = min(
                    args.checkpoint_every_n_steps,
                    args.n_steps - steps_completed,
                )
                if distillation_ensemble is None:
                    params, opt_state, chunk_losses = (
                        _train_supervised_with_optional_session_regularization(
                            make_current_network=make_train_network,
                            current_session_regularization_apply=(
                                train_session_regularization_apply
                            ),
                            params=params,
                            opt_state=opt_state,
                            n_steps=chunk_steps,
                            random_key=random_key,
                            optimizer=optimizer,
                            wandb_step_offset=args.n_warmup_steps + steps_completed,
                            total_step_offset=int(args.n_warmup_steps + steps_completed),
                            log_bottleneck_sparsity=True,
                        )
                    )
                else:
                    params, opt_state, chunk_losses = train_network_with_distillation(
                        make_train_network,
                        dataset_train,
                        dataset_eval,
                        training_teacher_probs=distillation_ensemble.train_probs,
                        validation_teacher_probs=distillation_ensemble.eval_probs,
                        params=params,
                        opt_state=opt_state,
                        opt=optimizer,
                        n_steps=chunk_steps,
                        max_grad_norm=args.max_grad_norm,
                        random_key=random_key,
                        temperature=distillation_ensemble.config.temperature,
                        n_action_logits=expected_n_action_logits,
                        include_penalty=True,
                        penalty_scale=distillation_penalty_scale,
                        session_regularization_apply=train_session_regularization_apply,
                        session_regularization_scale=lambda_reg_session,
                        session_curriculum_lambda_schedule=(
                            lambda local_step, base_step=int(
                                args.n_warmup_steps + steps_completed
                            ): _session_curriculum_lambda_for_total_step(
                                base_step + local_step
                            )
                        ),
                        report_progress_by="wandb",
                        wandb_run=wandb_run,
                        wandb_step_offset=args.n_warmup_steps + steps_completed,
                    )

                all_training_losses.extend(np.asarray(chunk_losses["training_loss"]).tolist())
                all_validation_losses.extend(np.asarray(chunk_losses["validation_loss"]).tolist())

                random_key = jax.random.split(random_key, 2)[0]
                steps_completed += chunk_steps
                current_completed_total_steps = int(args.n_warmup_steps + steps_completed)
                current_session_curriculum_lambda = (
                    _session_curriculum_lambda_for_completed_total_steps(
                        current_completed_total_steps
                    )
                )
                current_noiseless_eval_network = (
                    _make_noiseless_eval_network_for_completed_total_steps(
                        current_completed_total_steps
                    )
                )

                checkpoint_dir = checkpoint_root / f"step_{steps_completed}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_params_path = checkpoint_dir / "params.json"
                with checkpoint_params_path.open("w") as f:
                    f.write(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))

                # Persist the full training state (params + optimizer + PRNG key +
                # step + loss history) so a preempted job can resume from here.
                # random_key/steps_completed are already advanced for the *next*
                # chunk, so the restored state continues seamlessly.
                save_train_state(
                    checkpoint_dir,
                    steps_completed=steps_completed,
                    params=params,
                    opt_state=opt_state,
                    random_key=random_key,
                    training_losses=all_training_losses,
                    validation_losses=all_validation_losses,
                )

                train_likelihood_ckpt: float | None = None
                if args.checkpoint_eval_on_train_split:
                    yhat_train_ckpt, _ = rnn_utils.eval_network(
                        current_noiseless_eval_network,
                        params,
                        xs_train_all,
                    )
                    n_action_logits_train_ckpt = _require_n_action_logits(
                        dataset_train,
                        yhat_train_ckpt,
                        context="checkpoint train likelihood",
                    )
                    train_likelihood_ckpt = float(
                        rnn_utils.normalized_likelihood(
                            ys_train_all,
                            yhat_train_ckpt[:, :, :n_action_logits_train_ckpt],
                        )
                    )

                eval_likelihood_ckpt: float | None = None
                if args.checkpoint_eval_on_eval_split:
                    yhat_eval_ckpt, _ = rnn_utils.eval_network(
                        current_noiseless_eval_network,
                        params,
                        xs_eval_all,
                    )
                    n_action_logits_ckpt = _require_n_action_logits(
                        dataset_eval,
                        yhat_eval_ckpt,
                        context="checkpoint eval likelihood",
                    )
                    eval_likelihood_ckpt = float(
                        rnn_utils.normalized_likelihood(
                            ys_eval_all,
                            yhat_eval_ckpt[:, :, :n_action_logits_ckpt],
                        )
                    )

                train_distillation_loss_ckpt: float | None = None
                eval_distillation_loss_ckpt: float | None = None
                if distillation_ensemble is not None:
                    train_distillation_loss_ckpt = evaluate_distillation_loss(
                        make_network=current_noiseless_eval_network,
                        params=params,
                        xs=xs_train_all,
                        ys=ys_train_all,
                        teacher_probs=distillation_ensemble.train_probs,
                        temperature=distillation_ensemble.config.temperature,
                        n_action_logits=expected_n_action_logits,
                        include_penalty=False,
                        penalty_scale=distillation_penalty_scale,
                    )
                    eval_distillation_loss_ckpt = evaluate_distillation_loss(
                        make_network=current_noiseless_eval_network,
                        params=params,
                        xs=xs_eval_all,
                        ys=ys_eval_all,
                        teacher_probs=distillation_ensemble.eval_probs,
                        temperature=distillation_ensemble.config.temperature,
                        n_action_logits=expected_n_action_logits,
                        include_penalty=False,
                        penalty_scale=distillation_penalty_scale,
                    )

                checkpoint_record = {
                    "step": int(steps_completed),
                    "params_path": str(checkpoint_params_path),
                    "session_curriculum_lambda": float(current_session_curriculum_lambda),
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
                if train_distillation_loss_ckpt is not None:
                    checkpoint_record["train_distillation_loss"] = float(
                        train_distillation_loss_ckpt
                    )
                    logger.info(
                        "Checkpoint step %s, train distillation loss: %.4f",
                        steps_completed,
                        train_distillation_loss_ckpt,
                    )
                if eval_distillation_loss_ckpt is not None:
                    checkpoint_record["eval_distillation_loss"] = float(
                        eval_distillation_loss_ckpt
                    )
                    logger.info(
                        "Checkpoint step %s, eval distillation loss: %.4f",
                        steps_completed,
                        eval_distillation_loss_ckpt,
                    )

                checkpoint_plot_paths: dict[str, Any] = {
                    "bottlenecks": None,
                    "choice_rule": None,
                    "subject_embedding_state_space": None,
                    "subject_session_context_state_space": None,
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

                    if is_multisubject:
                        subject_embedding_fig_ckpt = self._plot_subject_embedding_state_space(
                            params=params,
                            raw_df=bundle.raw,
                            metadata=metadata,
                        )
                        if subject_embedding_fig_ckpt is not None:
                            subject_embedding_path_ckpt = (
                                checkpoint_dir / "subject_embedding_state_space.png"
                            )
                            subject_embedding_fig_ckpt.savefig(subject_embedding_path_ckpt)
                            plt.close(subject_embedding_fig_ckpt)
                            checkpoint_plot_paths["subject_embedding_state_space"] = str(
                                subject_embedding_path_ckpt
                            )
                        subject_session_context_fig_ckpt = (
                            self._plot_subject_session_context_state_space(
                                params=params,
                                raw_df=bundle.raw,
                                metadata=metadata,
                                session_curriculum_lambda=current_session_curriculum_lambda,
                            )
                        )
                        if subject_session_context_fig_ckpt is not None:
                            subject_session_context_path_ckpt = (
                                checkpoint_dir / "subject_session_context_state_space.png"
                            )
                            subject_session_context_fig_ckpt.savefig(
                                subject_session_context_path_ckpt
                            )
                            plt.close(subject_session_context_fig_ckpt)
                            checkpoint_plot_paths["subject_session_context_state_space"] = (
                                str(subject_session_context_path_ckpt)
                            )

                    plot_subject_embedding_ckpt, plot_subject_index_ckpt = (
                        self._subject_plot_context(
                            params=params,
                            multisubject=is_multisubject,
                        )
                    )

                    if args.checkpoint_plot_choice_rule:
                        choice_fig_ckpt = plotting.plot_choice_rule(
                            params,
                            disrnn_config,
                            subj_embedding=plot_subject_embedding_ckpt,
                        )
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
                        params_for_update_rules_ckpt = (
                            convert_local_params_to_upstream_multisubject(params)
                            if is_multisubject
                            else params
                        )
                        update_figs_ckpt = plotting.plot_update_rules(
                            params_for_update_rules_ckpt,
                            disrnn_config,
                            subj_ind=plot_subject_index_ckpt,
                        )
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
                    yhat_full_ckpt, network_states_full_ckpt = self._eval_network_full(
                        current_noiseless_eval_network,
                        params,
                        xs_full_for_checkpoint,
                    )
                    # Only build the whole-cohort frame when it is persisted;
                    # plotting builds per-subject frames on demand below.
                    output_df_ckpt = None
                    if should_save_output_df_ckpt:
                        output_df_ckpt = dl.add_model_results(
                            df_for_checkpoint.copy(),
                            np.asarray(network_states_full_ckpt),
                            np.asarray(yhat_full_ckpt),
                            ignore_policy=ignore_policy,
                        )
                        output_df_ckpt_path = checkpoint_dir / "output_df.csv"
                        output_df_ckpt.to_csv(output_df_ckpt_path, index=False)
                        checkpoint_record["output_df_path"] = str(output_df_ckpt_path)

                    if should_plot_split_examples_ckpt:
                        n_action_logits_ckpt_full = _require_n_action_logits(
                            dataset_eval,
                            np.asarray(yhat_full_ckpt),
                            context="checkpoint full-dataset plotting",
                        )
                        split_summaries_ckpt = self._generate_split_examples(
                            output_dir=checkpoint_dir,
                            output_df=output_df_ckpt,
                            raw_df=df_for_checkpoint,
                            ignore_policy=ignore_policy,
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
                            wandb_step=args.n_warmup_steps + int(steps_completed),
                            wandb_key_prefix="checkpoint",
                        )
                        checkpoint_record["split_examples"] = split_summaries_ckpt

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
                        step=args.n_warmup_steps + int(steps_completed),
                    )
                if (
                    wandb_run is not None
                    and args.checkpoint_log_train_to_wandb
                    and train_distillation_loss_ckpt is not None
                ):
                    wandb_run.log(
                        {
                            "checkpoint/train_distillation_loss": float(
                                train_distillation_loss_ckpt
                            ),
                            "checkpoint/step": int(steps_completed),
                        },
                        step=args.n_warmup_steps + int(steps_completed),
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
                        step=args.n_warmup_steps + int(steps_completed),
                    )

                # NOTE: bottleneck-sparsity scalars are logged at loss pace inside
                # the training loop (train_network_with_session_regularization,
                # log_losses_every), plus a step-0 baseline before warmup, so no
                # per-checkpoint sparsity log is needed here. The end-of-run
                # final/bottlenecks/* summary write is still below.
                if (
                    wandb_run is not None
                    and eval_distillation_loss_ckpt is not None
                    and args.checkpoint_log_eval_to_wandb
                ):
                    wandb_run.log(
                        {
                            "checkpoint/eval_distillation_loss": float(
                                eval_distillation_loss_ckpt
                            ),
                            "checkpoint/step": int(steps_completed),
                        },
                        step=args.n_warmup_steps + int(steps_completed),
                    )

                if wandb_run is not None and args.checkpoint_log_eval_to_wandb:
                    checkpoint_plot_payload = {}
                    checkpoint_bottlenecks = checkpoint_plot_paths.get("bottlenecks")
                    checkpoint_choice_rule = checkpoint_plot_paths.get("choice_rule")
                    checkpoint_subject_embeddings = checkpoint_plot_paths.get(
                        "subject_embedding_state_space"
                    )
                    checkpoint_subject_session_context = checkpoint_plot_paths.get(
                        "subject_session_context_state_space"
                    )
                    checkpoint_update_rules = checkpoint_plot_paths.get("update_rules", [])
                    if checkpoint_bottlenecks:
                        checkpoint_plot_payload[
                            "checkpoint/fig/bottlenecks"
                        ] = wandb.Image(str(checkpoint_bottlenecks))
                    if checkpoint_subject_embeddings:
                        checkpoint_plot_payload[
                            "checkpoint/fig/subject_embedding_state_space"
                        ] = wandb.Image(str(checkpoint_subject_embeddings))
                    if checkpoint_subject_session_context:
                        checkpoint_plot_payload[
                            "checkpoint/fig/subject_session_context_state_space"
                        ] = wandb.Image(str(checkpoint_subject_session_context))
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
                            checkpoint_subject_embeddings,
                            checkpoint_subject_session_context,
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
                        checkpoint_plot_paths["subject_embedding_state_space"] = None
                        checkpoint_plot_paths["subject_session_context_state_space"] = None
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

        final_completed_total_steps = int(args.n_warmup_steps + args.n_steps)
        final_session_curriculum_lambda = _session_curriculum_lambda_for_completed_total_steps(
            final_completed_total_steps
        )
        output["session_curriculum"]["final_lambda"] = float(
            final_session_curriculum_lambda
        )
        final_noiseless_eval_network = (
            _make_noiseless_eval_network_for_completed_total_steps(
                final_completed_total_steps
            )
        )

        bottlenecks_fig = plotting.plot_bottlenecks(params, disrnn_config, sort_latents=False)
        bottlenecks_fig.tight_layout()
        bottlenecks_path = self._save_figure(bottlenecks_fig, "bottlenecks.png")
        if wandb_run is not None:
            wandb_run.log({"fig/bottlenecks": wandb.Image(str(bottlenecks_path))})

        if is_multisubject:
            subject_embedding_fig = self._plot_subject_embedding_state_space(
                params=params,
                raw_df=bundle.raw,
                metadata=metadata,
            )
            if subject_embedding_fig is not None:
                subject_embedding_path = self._save_figure(
                    subject_embedding_fig,
                    "subject_embedding_state_space.png",
                )
                output["subject_embedding_state_space_path"] = str(subject_embedding_path)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "fig/subject_embedding_state_space": wandb.Image(
                                str(subject_embedding_path)
                            )
                        }
                    )
            subject_session_context_fig = self._plot_subject_session_context_state_space(
                params=params,
                raw_df=bundle.raw,
                metadata=metadata,
                session_curriculum_lambda=final_session_curriculum_lambda,
            )
            if subject_session_context_fig is not None:
                subject_session_context_path = self._save_figure(
                    subject_session_context_fig,
                    "subject_session_context_state_space.png",
                )
                output["subject_session_context_state_space_path"] = str(
                    subject_session_context_path
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "fig/subject_session_context_state_space": wandb.Image(
                                str(subject_session_context_path)
                            )
                        }
                    )

        plot_subject_embedding, plot_subject_index = self._subject_plot_context(
            params=params,
            multisubject=is_multisubject,
        )

        if args.plot_choice_rule:
            choice_fig = plotting.plot_choice_rule(
                params,
                disrnn_config,
                subj_embedding=plot_subject_embedding,
            )
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
            params_for_update_rules = (
                convert_local_params_to_upstream_multisubject(params)
                if is_multisubject
                else params
            )
            update_figs = plotting.plot_update_rules(
                params_for_update_rules,
                disrnn_config,
                subj_ind=plot_subject_index,
            )
            for index, fig in enumerate(update_figs):
                fig.tight_layout()
                path = self._save_figure(fig, f"update_rule_{index}.png")
                if wandb_run is not None:
                    wandb_run.log({f"fig/update_rule_{index}": wandb.Image(str(path))})

        # Get model predictions on full dataset, including the training set
        _all = dataset.get_all()
        xs_full, ys_full = _all["xs"], _all["ys"]
        yhat_full, network_states_full = rnn_utils.eval_network(
            final_noiseless_eval_network, params, xs_full
        )

        df = bundle.raw
        output_df = dl.add_model_results(
            df, network_states_full.__array__(), yhat_full, ignore_policy=ignore_policy
        )
        if args.save_output_df:
            output_path = self.output_dir / "output_df.csv"
            output_df.to_csv(output_path, index=False)

        params_path = self.output_dir / "params.json"
        with params_path.open("w") as f:
            f.write(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))
        if is_multisubject:
            output["subject_artifacts"] = self._save_multisubject_artifacts(
                params=params,
                metadata=metadata,
                raw_df=df,
            )

        # Get likelihood evaluated on the training and evaluation datasets.
        _train = dataset_train.get_all()
        xs_train, ys_train = _train["xs"], _train["ys"]
        yhat_train, _ = rnn_utils.eval_network(
            final_noiseless_eval_network, params, xs_train
        )
        n_action_logits_train = _require_n_action_logits(
            dataset_train, yhat_train, context="final train likelihood"
        )
        likelihood_train = rnn_utils.normalized_likelihood(
            ys_train,
            yhat_train[:, :, :n_action_logits_train],
        )
        output["likelihood_train"] = float(likelihood_train)
        logger.info("Final training likelihood: %.4f", float(likelihood_train))

        _eval = dataset_eval.get_all()
        xs_eval, ys_eval = _eval["xs"], _eval["ys"]
        yhat_eval, network_states_eval = rnn_utils.eval_network(
            final_noiseless_eval_network, params, xs_eval
        )
        n_action_logits = _require_n_action_logits(
            dataset_eval, yhat_eval, context="final eval likelihood"
        )

        likelihood = rnn_utils.normalized_likelihood(
            ys_eval,
            yhat_eval[:, :, :n_action_logits],
        )
        output["likelihood"] = float(likelihood)
        logger.info("Final eval likelihood: %.4f", float(likelihood))
        if distillation_ensemble is not None:
            final_train_distillation_loss = evaluate_distillation_loss(
                make_network=final_noiseless_eval_network,
                params=params,
                xs=xs_train,
                ys=ys_train,
                teacher_probs=distillation_ensemble.train_probs,
                temperature=distillation_ensemble.config.temperature,
                n_action_logits=expected_n_action_logits,
                include_penalty=False,
                penalty_scale=distillation_penalty_scale,
            )
            final_eval_distillation_loss = evaluate_distillation_loss(
                make_network=final_noiseless_eval_network,
                params=params,
                xs=xs_eval,
                ys=ys_eval,
                teacher_probs=distillation_ensemble.eval_probs,
                temperature=distillation_ensemble.config.temperature,
                n_action_logits=expected_n_action_logits,
                include_penalty=False,
                penalty_scale=distillation_penalty_scale,
            )
            output.setdefault("distillation", {})
            output["distillation"]["train_distillation_loss"] = float(
                final_train_distillation_loss
            )
            output["distillation"]["eval_distillation_loss"] = float(
                final_eval_distillation_loss
            )
            output["distillation"]["teacher_count"] = int(
                len(distillation_ensemble.config.teacher_model_dirs)
            )
            output["distillation"]["temperature"] = float(
                distillation_ensemble.config.temperature
            )
            output["distillation"]["aggregation"] = distillation_ensemble.config.aggregation

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
                wandb_run=(
                    wandb_run
                    if (args.checkpoint_every_n_steps == 0)
                    else None
                ),
                log_scope="Final",
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
            if len(losses["validation_loss"]) > 0:
                wandb_run.summary["final/val_loss"] = float(losses["validation_loss"][-1])
            if len(losses["training_loss"]) > 0:
                wandb_run.summary["final/train_loss"] = float(losses["training_loss"][-1])
            wandb_run.summary["likelihood"] = float(likelihood)
            wandb_run.summary["likelihood_train"] = float(likelihood_train)
            if distillation_ensemble is not None:
                wandb_run.summary["distillation/train_loss"] = float(
                    output["distillation"]["train_distillation_loss"]
                )
                wandb_run.summary["distillation/eval_loss"] = float(
                    output["distillation"]["eval_distillation_loss"]
                )
                wandb_run.summary["distillation/teacher_count"] = int(
                    output["distillation"]["teacher_count"]
                )
                wandb_run.summary["distillation/temperature"] = float(
                    output["distillation"]["temperature"]
                )

            if gt_likelihood is not None:
                wandb_run.summary["groundtruth_likelihood"] = float(gt_likelihood)
                wandb_run.summary["likelihood_relative_to_groundtruth"] = float(likelihood) / float(gt_likelihood)

            wandb_run.summary["elapsed_seconds"] = float(training_time)
            wandb_run.summary["warmup_seconds"] = float(warmup_duration)

            # Final bottleneck-sparsity scalars -> wandb.summary so they are
            # queryable per-run (same route as `likelihood`) for post-hoc analysis.
            final_sparsity = compute_bottleneck_sparsity_metrics(params)
            for _k, _v in final_sparsity.items():
                wandb_run.summary[f"final/{_k}"] = _v

            # Upload the whole /results/output folder as an artifact
            # Here I'm using the random id as the name, meaning each run will has its own artifact.
            artifact_name = (
                f"disrnn-output-{getattr(wandb_run, 'id', None) or 'latest'}"
            )
            artifact = wandb.Artifact(artifact_name, type="training-output")
            artifact.add_dir(str(self.output_dir))
            wandb_run.log_artifact(artifact)

        return output


    def _plot_examples_for_split(
        self,
        *,
        split_name: str,
        output_dir: Path,
        output_df: pd.DataFrame,
        network_states: np.ndarray,
        yhat_logits: np.ndarray,
        params: Any,
        sessions_per_subject: int,
        max_subjects_to_plot: int,
        n_action_logits: int,
        wandb_run: Any | None,
        log_scope: str | None,
    ) -> dict[str, Any]:
        return plot_disrnn_examples_for_split(
            split_name=split_name,
            output_dir=output_dir,
            output_df=output_df,
            network_states=network_states,
            yhat_logits=yhat_logits,
            params=params,
            sessions_per_subject=sessions_per_subject,
            max_subjects_to_plot=max_subjects_to_plot,
            n_action_logits=n_action_logits,
            wandb_run=wandb_run,
            log_scope=log_scope,
        )

