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
from models import multisubject_disrnn as local_multisubject_disrnn
from models.session_conditioning import resolve_session_conditioning_from_architecture
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
    save_subject_index_map,
    save_session_context_map,
    subject_embeddings_to_dataframe,
)
from utils.run_helpers import resolve_disrnn_penalties

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
    xs, _ = dataset.get_all()
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


class DisrnnTrainer(ModelTrainer):
    """Trainer that reproduces the legacy disRNN pipeline."""

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
        super().__init__(seed=seed)
        self.architecture = _to_dict(architecture)
        self.penalties = resolve_disrnn_penalties(penalties)
        self.training = _to_dict(training)
        self.distillation = resolve_distillation_config(
            _to_dict(distillation) if distillation is not None else None
        )
        self.heldout_data = heldout_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
                context="DisrnnTrainer",
            )
            packed_context_feature_count = (
                2 if bool(session_conditioning_cfg["enabled"]) else 1
            )

            config = local_multisubject_disrnn.MultisubjectDisRnnConfig(
                obs_size=int(dataset._xs.shape[2] - packed_context_feature_count),
                output_size=output_size,
                x_names=dataset.x_names,
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
                context="DisrnnTrainer",
            )
            config = disrnn.DisRnnConfig(
                obs_size=dataset._xs.shape[2],
                output_size=output_size,
                x_names=dataset.x_names,
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
    ) -> Callable[[], Any]:
        if multisubject:
            return lambda: local_multisubject_disrnn.MultisubjectDisRnn(config)
        return lambda: disrnn.HkDisentangledRNN(config)

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

    def _save_multisubject_artifacts(
        self,
        *,
        params: Any,
        metadata: Mapping[str, Any],
        raw_df: pd.DataFrame,
    ) -> dict[str, str]:
        subject_id_to_index = metadata.get("subject_id_to_index")
        index_to_subject_id = metadata.get("index_to_subject_id")
        if not isinstance(subject_id_to_index, dict) or not isinstance(index_to_subject_id, dict):
            raise ValueError(
                "Multisubject artifact export requires subject_id_to_index and "
                "index_to_subject_id in bundle metadata."
            )

        subject_index_map_path = save_subject_index_map(
            self.output_dir / "subject_index_map.json",
            subject_id_to_index=subject_id_to_index,
            index_to_subject_id=index_to_subject_id,
        )
        subject_embeddings = extract_subject_embeddings_from_params(params)
        subject_embeddings_df = subject_embeddings_to_dataframe(
            index_to_subject_id,
            subject_embeddings,
        )
        subject_embeddings_df = self._attach_subject_metadata_to_embeddings(
            subject_embeddings_df,
            raw_df=raw_df,
            metadata=metadata,
        )
        subject_embeddings_path = self.output_dir / "subject_embeddings.pkl"
        subject_embeddings_df.to_pickle(subject_embeddings_path)
        artifacts = {
            "subject_index_map": str(subject_index_map_path),
            "subject_embeddings": str(subject_embeddings_path),
        }
        if str(self.architecture.get("session_encoding_type", "none")).strip().lower() != "none":
            session_context = metadata.get("session_context")
            if not isinstance(session_context, dict):
                raise ValueError(
                    "Session-conditioned multisubject disRNN export requires metadata.session_context."
                )
            session_context_map_path = save_session_context_map(
                self.output_dir / "session_context_map.json",
                session_context=session_context,
            )
            artifacts["session_context_map"] = str(session_context_map_path)
        return artifacts

    def _resolve_subject_curricula(
        self,
        *,
        raw_df: pd.DataFrame,
        metadata: Mapping[str, Any],
    ) -> dict[Any, str]:
        subject_curricula = metadata.get("subject_curricula")
        if isinstance(subject_curricula, dict) and subject_curricula:
            return {
                normalize_subject_id(subject_id): str(curriculum)
                for subject_id, curriculum in subject_curricula.items()
            }

        if "subject_id" not in raw_df.columns or "curriculum_name" not in raw_df.columns:
            return {}

        resolved: dict[Any, str] = {}
        for subject_id, subject_rows in raw_df.groupby("subject_id", sort=False):
            curricula = [
                str(value)
                for value in subject_rows["curriculum_name"].dropna().unique().tolist()
            ]
            normalized_subject_id = normalize_subject_id(subject_id)
            if not curricula:
                resolved[normalized_subject_id] = "Unknown"
            elif len(curricula) == 1:
                resolved[normalized_subject_id] = curricula[0]
            else:
                resolved[normalized_subject_id] = "Mixed"
        return resolved

    def _attach_subject_metadata_to_embeddings(
        self,
        subject_embeddings_df: pd.DataFrame,
        *,
        raw_df: pd.DataFrame,
        metadata: Mapping[str, Any],
    ) -> pd.DataFrame:
        enriched_df = subject_embeddings_df.copy()
        subject_curricula = self._resolve_subject_curricula(raw_df=raw_df, metadata=metadata)
        if subject_curricula:
            enriched_df["curriculum_name"] = enriched_df["subject_id"].map(
                lambda subject_id: subject_curricula.get(
                    normalize_subject_id(subject_id),
                    "Unknown",
                )
            )
        return enriched_df

    def _resolve_session_context_plot_subject_indices(
        self,
        *,
        metadata: Mapping[str, Any],
    ) -> list[int]:
        session_context = metadata.get("session_context")
        if not isinstance(session_context, Mapping):
            raise ValueError(
                "Session-context plotting requires metadata.session_context for multisubject runs."
            )
        return resolve_session_context_plot_subject_indices(
            session_context,
            requested_subject_indices=self.training.get("session_context_plot_subject_indices"),
            max_subjects=int(self.training.get("session_context_plot_max_subjects", 3)),
            random_seed=(
                None
                if self.training.get("session_context_plot_subject_indices") is not None
                else int(self.seed or 0)
            ),
        )

    def _log_session_conditioning_details(
        self,
        *,
        metadata: Mapping[str, Any],
        dataset: Any,
        session_conditioning_cfg: Mapping[str, Any],
    ) -> None:
        if not bool(session_conditioning_cfg.get("enabled")):
            return
        if not bool(self.training.get("session_conditioning_verbose_logging", True)):
            return

        session_max = np.asarray(
            session_conditioning_cfg["session_max_index_by_subject_index"],
            dtype=int,
        )
        logger.info(
            "Session conditioning enabled for disRNN: encoding=%s integration=%s "
            "fourier_k=%d packed_input=[subject_idx, session_idx, *obs]",
            session_conditioning_cfg["session_encoding_type"],
            session_conditioning_cfg["session_integration_type"],
            int(session_conditioning_cfg["session_fourier_k"]),
        )
        logger.info(
            "Session-conditioned input features: %s",
            list(getattr(dataset, "x_names", [])),
        )
        logger.info(
            "Per-subject session counts: min=%d median=%.1f max=%d across %d subjects",
            int(np.min(session_max)),
            float(np.median(session_max)),
            int(np.max(session_max)),
            int(session_max.shape[0]),
        )

        session_context = metadata.get("session_context")
        if isinstance(session_context, Mapping):
            ordered_rows = ordered_session_context_rows(session_context)
            train_session_ids = {
                str(session_id) for session_id in metadata.get("train_session_ids") or []
            }
            eval_session_ids = {
                str(session_id) for session_id in metadata.get("eval_session_ids") or []
            }
            preview_tokens: list[str] = []
            for row in ordered_rows[: min(5, len(ordered_rows))]:
                ordered_session_ids = [
                    str(session_id) for session_id in row.get("ordered_session_ids") or []
                ]
                train_count = sum(
                    1 for session_id in ordered_session_ids if session_id in train_session_ids
                )
                eval_count = sum(
                    1 for session_id in ordered_session_ids if session_id in eval_session_ids
                )
                preview_tokens.append(
                    f"{int(row['subject_index'])}:{len(ordered_session_ids)} "
                    f"(train={train_count}, eval={eval_count})"
                )
            if preview_tokens:
                logger.info(
                    "Session-order preview [subject_index:total (train, eval)]: %s",
                    ", ".join(preview_tokens),
                )

        if bool(self.training.get("plot_session_context_state_space", True)):
            selected_subject_indices = self._resolve_session_context_plot_subject_indices(
                metadata=metadata
            )
            if self.training.get("session_context_plot_subject_indices") is None:
                logger.info(
                    "Session-context state-space plot subjects (random, seed=%s): %s",
                    int(self.seed or 0),
                    selected_subject_indices,
                )
            else:
                logger.info(
                    "Session-context state-space plot subjects: %s",
                    selected_subject_indices,
                )

    def _plot_subject_embedding_state_space(
        self,
        *,
        params: Any,
        raw_df: pd.DataFrame,
        metadata: Mapping[str, Any],
    ) -> Any | None:
        index_to_subject_id = metadata.get("index_to_subject_id")
        if not isinstance(index_to_subject_id, dict):
            return None

        subject_embeddings = extract_subject_embeddings_from_params(params)
        if subject_embeddings.shape[1] < 2:
            logger.info(
                "Skipping subject embedding state-space plot because subject_embedding_size < 2."
            )
            return None

        plot_df = subject_embeddings_to_dataframe(index_to_subject_id, subject_embeddings)
        plot_df = self._attach_subject_metadata_to_embeddings(
            plot_df,
            raw_df=raw_df,
            metadata=metadata,
        )
        if "curriculum_name" not in plot_df.columns:
            plot_df["curriculum_name"] = "Unknown"
        else:
            plot_df["curriculum_name"] = plot_df["curriculum_name"].fillna("Unknown")

        embedding_columns = [
            column
            for column in plot_df.columns
            if column.startswith("embedding_")
        ]
        dim_pairs = list(itertools.combinations(embedding_columns, 2))
        if not dim_pairs:
            return None

        ncols = min(3, len(dim_pairs))
        nrows = int(np.ceil(len(dim_pairs) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.0 * nrows), squeeze=False)

        curriculum_values = list(dict.fromkeys(plot_df["curriculum_name"].astype(str).tolist()))
        cmap = plt.get_cmap("tab10")
        curriculum_to_color = {
            curriculum: cmap(index % max(1, cmap.N))
            for index, curriculum in enumerate(curriculum_values)
        }

        for ax, (x_column, y_column) in zip(axes.flat, dim_pairs):
            for curriculum in curriculum_values:
                curriculum_rows = plot_df[plot_df["curriculum_name"].astype(str) == curriculum]
                ax.scatter(
                    curriculum_rows[x_column],
                    curriculum_rows[y_column],
                    color=curriculum_to_color[curriculum],
                    label=curriculum,
                    s=55,
                    alpha=0.9,
                )
            for row in plot_df.itertuples(index=False):
                ax.text(
                    getattr(row, x_column),
                    getattr(row, y_column),
                    str(getattr(row, "subject_index")),
                    fontsize=8,
                    alpha=0.8,
                )
            ax.axhline(0, color="0.85", linewidth=1)
            ax.axvline(0, color="0.85", linewidth=1)
            ax.set_xlabel(x_column.replace("_", " ").title())
            ax.set_ylabel(y_column.replace("_", " ").title())
            ax.set_title(f"{x_column} vs {y_column}")

        for ax in axes.flat[len(dim_pairs):]:
            ax.axis("off")

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=curriculum_to_color[curriculum],
                markeredgecolor=curriculum_to_color[curriculum],
                label=curriculum,
            )
            for curriculum in curriculum_values
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=min(len(legend_handles), 4),
            title="Curriculum",
        )
        fig.suptitle("Subject Embedding State Space", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    def _plot_subject_session_context_state_space(
        self,
        *,
        params: Any,
        raw_df: pd.DataFrame,
        metadata: Mapping[str, Any],
    ) -> Any | None:
        if not bool(self.training.get("plot_session_context_state_space", True)):
            return None

        max_n_subjects = metadata.get("num_subjects")
        if max_n_subjects is None:
            subject_ids = metadata.get("subject_ids", [])
            max_n_subjects = len(subject_ids)
        session_conditioning_cfg = resolve_session_conditioning_from_architecture(
            architecture=self.architecture,
            metadata=metadata,
            multisubject=True,
            max_n_subjects=int(max_n_subjects) if max_n_subjects is not None else None,
            context="DisrnnTrainer session-context plotting",
        )
        if not bool(session_conditioning_cfg["enabled"]):
            return None

        selected_subject_indices = self._resolve_session_context_plot_subject_indices(
            metadata=metadata
        )
        if not selected_subject_indices:
            return None

        session_context = metadata.get("session_context")
        if not isinstance(session_context, Mapping):
            raise ValueError(
                "Session-context plotting requires metadata.session_context for multisubject runs."
            )

        plot_df = compute_session_conditioned_context_dataframe(
            params,
            session_context=session_context,
            session_encoding_type=str(session_conditioning_cfg["session_encoding_type"]),
            session_integration_type=str(
                session_conditioning_cfg["session_integration_type"]
            ),
            session_fourier_k=int(session_conditioning_cfg["session_fourier_k"]),
            session_max_index_by_subject_index=session_conditioning_cfg[
                "session_max_index_by_subject_index"
            ],
            train_session_ids=metadata.get("train_session_ids"),
            eval_session_ids=metadata.get("eval_session_ids"),
            selected_subject_indices=selected_subject_indices,
        )
        if plot_df.empty:
            return None

        subject_embeddings = extract_subject_embeddings_from_params(params)
        subject_curricula = self._resolve_subject_curricula(raw_df=raw_df, metadata=metadata)
        plot_df["curriculum_name"] = plot_df["subject_id"].map(
            lambda subject_id: subject_curricula.get(
                normalize_subject_id(subject_id),
                "Unknown",
            )
        )
        embedding_columns = [
            column for column in plot_df.columns if column.startswith("embedding_")
        ]
        if len(embedding_columns) < 2:
            logger.info(
                "Skipping session-context state-space plot because subject_embedding_size < 2."
            )
            return None

        dim_pairs = list(itertools.combinations(embedding_columns, 2))
        ncols = min(3, len(dim_pairs))
        block_rows = int(np.ceil(len(dim_pairs) / ncols))
        subject_keys = list(
            dict.fromkeys(
                [
                    (
                        int(row.subject_index),
                        normalize_subject_id(row.subject_id),
                    )
                    for row in plot_df.itertuples(index=False)
                ]
            )
        )
        fig, axes = plt.subplots(
            block_rows * len(subject_keys),
            ncols,
            figsize=(6.2 * ncols, 4.5 * block_rows * len(subject_keys)),
            squeeze=False,
        )

        cmap = plt.get_cmap("viridis")
        marker_by_split = {"train": "o", "eval": "s", "full": "^"}
        colorbar_specs: list[tuple[Any, list[Any], int, int]] = []

        for subject_position, (subject_index, subject_id) in enumerate(subject_keys):
            subject_rows = plot_df[
                plot_df["subject_index"].astype(int) == int(subject_index)
            ].copy()
            subject_rows = subject_rows.sort_values("session_index").reset_index(drop=True)
            curriculum_name = str(subject_rows["curriculum_name"].iloc[0])
            train_count = int((subject_rows["session_split"] == "train").sum())
            eval_count = int((subject_rows["session_split"] == "eval").sum())
            subject_max_session_index = int(subject_rows["subject_max_session_index"].iloc[0])
            subject_norm = plt.Normalize(
                vmin=1.0,
                vmax=float(max(subject_max_session_index, 2)),
            )
            subject_embedding = np.asarray(
                subject_embeddings[int(subject_index)],
                dtype=float,
            )
            subject_axes: list[Any] = []

            for pair_index, (x_column, y_column) in enumerate(dim_pairs):
                ax = axes[
                    subject_position * block_rows + pair_index // ncols,
                    pair_index % ncols,
                ]
                subject_axes.append(ax)
                ax.plot(
                    subject_rows[x_column],
                    subject_rows[y_column],
                    color="0.75",
                    linewidth=1.1,
                    alpha=0.8,
                    zorder=1,
                )
                for split_name, marker in marker_by_split.items():
                    split_rows = subject_rows[subject_rows["session_split"] == split_name]
                    if split_rows.empty:
                        continue
                    ax.scatter(
                        split_rows[x_column],
                        split_rows[y_column],
                        c=split_rows["session_index"].to_numpy(dtype=float),
                        cmap=cmap,
                        norm=subject_norm,
                        marker=marker,
                        s=62,
                        edgecolors="black",
                        linewidths=0.45,
                        alpha=0.95,
                        zorder=2,
                    )
                x_dim = int(x_column.split("_")[-1]) - 1
                y_dim = int(y_column.split("_")[-1]) - 1
                ax.scatter(
                    float(subject_embedding[x_dim]),
                    float(subject_embedding[y_dim]),
                    marker="*",
                    s=160,
                    c="black",
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                )
                ax.axhline(0, color="0.85", linewidth=1)
                ax.axvline(0, color="0.85", linewidth=1)
                ax.set_xlabel(x_column.replace("_", " ").title())
                ax.set_ylabel(y_column.replace("_", " ").title())
                ax.set_title(
                    f"Subj {subject_index} ({subject_id}) | {curriculum_name} | "
                    f"S={int(subject_rows['subject_max_session_index'].iloc[0])} "
                    f"(train={train_count}, eval={eval_count})\n"
                    f"{x_column} vs {y_column}"
                )

            for empty_index in range(len(dim_pairs), block_rows * ncols):
                axes[
                    subject_position * block_rows + empty_index // ncols,
                    empty_index % ncols,
                ].axis("off")

            scalar_mappable = plt.cm.ScalarMappable(cmap=cmap, norm=subject_norm)
            scalar_mappable.set_array([])
            colorbar_specs.append(
                (
                    scalar_mappable,
                    subject_axes,
                    int(subject_max_session_index),
                    int(subject_index),
                )
            )

        legend_handles = [
            Line2D(
                [0],
                [0],
                color="0.75",
                linewidth=1.1,
                label="Session trajectory",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="0.45",
                markeredgecolor="black",
                markersize=7,
                linewidth=0,
                label="Train session",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                markerfacecolor="0.45",
                markeredgecolor="black",
                markersize=7,
                linewidth=0,
                label="Eval session",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="black",
                markerfacecolor="black",
                markeredgecolor="white",
                markersize=11,
                linewidth=0,
                label="Base subject embedding",
            ),
        ]
        if (plot_df["session_split"] == "full").any():
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor="0.45",
                    markeredgecolor="black",
                    markersize=7,
                    linewidth=0,
                    label="Other session",
                )
            )
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(4, len(legend_handles)),
            frameon=False,
        )
        fig.suptitle("Session-Conditioned Subject Context State Space", fontsize=14)
        fig.tight_layout(rect=(0, 0.06, 0.86, 0.96))

        colorbar_left = 0.89
        colorbar_width = 0.018
        for scalar_mappable, subject_axes, subject_max_session_index, subject_index in colorbar_specs:
            subject_positions = [ax.get_position() for ax in subject_axes]
            if not subject_positions:
                continue
            y0 = min(position.y0 for position in subject_positions)
            y1 = max(position.y1 for position in subject_positions)
            colorbar_ax = fig.add_axes(
                [colorbar_left, y0, colorbar_width, max(y1 - y0, 0.05)]
            )
            colorbar = fig.colorbar(scalar_mappable, cax=colorbar_ax)
            if subject_max_session_index <= 6:
                colorbar.set_ticks(np.arange(1, subject_max_session_index + 1))
            else:
                colorbar.set_ticks([1, subject_max_session_index])
            colorbar.set_label(f"Session index (subj {subject_index})")
        return fig

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
            "checkpoint/step": int(wandb_step) if wandb_step is not None else 0,
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
        make_eval_network: Callable[[], Any],
        disrnn_config: Any,
        is_multisubject: bool,
        heldout_eval_cfg: Any | None,
        heldout_data: dict[str, Any] | None,
        wandb_run: Any | None,
        wandb_step: int | None,
        keep_media_files: bool,
    ) -> dict[str, Any]:
        stage_dir = self.output_dir / "initialization" / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        log_scope = stage_name.replace("_", " ").title()
        wandb_key_prefix = "checkpoint"

        params_path = stage_dir / "params.json"
        with params_path.open("w") as f:
            f.write(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))

        xs_train, ys_train = dataset_train.get_all()
        yhat_train, _ = rnn_utils.eval_network(make_eval_network, params, xs_train)
        n_action_logits_train = int(getattr(dataset_train, "n_classes", 0))
        if n_action_logits_train <= 0:
            n_action_logits_train = int(np.asarray(yhat_train).shape[2] - 1)
        if n_action_logits_train <= 0:
            raise ValueError(
                "Invalid number of action logits inferred for initialization train likelihood."
            )
        train_likelihood = float(
            rnn_utils.normalized_likelihood(
                ys_train,
                np.asarray(yhat_train)[:, :, :n_action_logits_train],
            )
        )

        xs_eval, ys_eval = dataset_eval.get_all()
        yhat_eval, _ = rnn_utils.eval_network(make_eval_network, params, xs_eval)
        n_action_logits_eval = int(getattr(dataset_eval, "n_classes", 0))
        if n_action_logits_eval <= 0:
            n_action_logits_eval = int(np.asarray(yhat_eval).shape[2] - 1)
        if n_action_logits_eval <= 0:
            raise ValueError(
                "Invalid number of action logits inferred for initialization eval likelihood."
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
                "checkpoint/step": int(wandb_step) if wandb_step is not None else 0,
                "checkpoint/train_likelihood": train_likelihood,
                "checkpoint/eval_likelihood": eval_likelihood,
            }
            if wandb_step is None:
                wandb_run.log(metric_payload)
            else:
                wandb_run.log(metric_payload, step=wandb_step)

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
                    raw_df=bundle.raw,
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
                        raw_df=bundle.raw,
                        metadata=metadata,
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

        xs_full, _ = dataset.get_all()
        yhat_full, network_states_full = rnn_utils.eval_network(
            make_eval_network,
            params,
            xs_full,
        )
        output_df = dl.add_model_results(
            bundle.raw.copy(),
            np.asarray(network_states_full),
            yhat_full,
            ignore_policy=ignore_policy,
        )

        n_action_logits_full = int(getattr(dataset_eval, "n_classes", 0))
        if n_action_logits_full <= 0:
            n_action_logits_full = int(np.asarray(yhat_full).shape[2] - 1)
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
            "train_likelihood": train_likelihood,
            "eval_likelihood": eval_likelihood,
            "plot_paths": plot_paths,
            "split_examples": split_summaries,
        }

        if heldout_eval_cfg is not None:
            try:
                heldout_summary = evaluate_disrnn_on_heldout_subjects(
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
            context="DisrnnTrainer",
        )
        dataset, dataset_train, dataset_eval = _maybe_prepend_session_indices_to_datasets(
            dataset=dataset,
            dataset_train=dataset_train,
            dataset_eval=dataset_eval,
            session_conditioning_enabled=bool(session_conditioning_cfg["enabled"]),
            metadata=metadata,
        )

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
        disrnn_config, noiseless_network = self._build_network_configs(
            dataset=dataset,
            ignore_policy=ignore_policy,
            metadata=metadata,
        )
        make_train_network = self._make_network_factory(
            disrnn_config,
            multisubject=is_multisubject,
        )
        make_noiseless_network = self._make_network_factory(
            noiseless_network,
            multisubject=is_multisubject,
        )
        expected_n_action_logits = int(getattr(dataset, "n_classes", 0))
        if expected_n_action_logits <= 0:
            expected_n_action_logits = 2 if ignore_policy == "exclude" else 3
        xs_train_all, ys_train_all = dataset_train.get_all()
        xs_eval_all, ys_eval_all = dataset_eval.get_all()
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

        def _compute_distillation_metrics(current_params: Any) -> tuple[float, float] | None:
            if distillation_ensemble is None:
                return None
            train_metric = evaluate_distillation_loss(
                make_network=make_noiseless_network,
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
                make_network=make_noiseless_network,
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
                "Skipping held-out evaluation for multisubject disRNN; "
                "v1 supports seen-subject personalization only."
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

        params, warmup_opt_state, _ = rnn_utils.train_network(
            make_noiseless_network,
            dataset_train,
            dataset_eval,
            opt=optax.adam(args.learning_rate),
            loss=args.loss,
            loss_param=args.loss_param,
            n_steps=0,
            max_grad_norm=args.max_grad_norm,
            random_key=warmup_key,
            report_progress_by="wandb",
            wandb_run=wandb_run,
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
                make_eval_network=make_noiseless_network,
                disrnn_config=disrnn_config,
                is_multisubject=is_multisubject,
                heldout_eval_cfg=heldout_eval_cfg,
                heldout_data=heldout_test_data,
                wandb_run=wandb_run,
                wandb_step=0,
                keep_media_files=args.checkpoint_keep_media_files,
            )
            distillation_metrics = _compute_distillation_metrics(params)
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
            params, warmup_opt_state, warmup_losses = rnn_utils.train_network(
                make_noiseless_network,
                dataset_train,
                dataset_eval,
                opt=optax.adam(args.learning_rate),
                loss=args.loss,
                loss_param=args.loss_param,
                params=params,
                opt_state=warmup_opt_state,
                n_steps=args.n_warmup_steps,
                max_grad_norm=args.max_grad_norm,
                random_key=warmup_key,
                report_progress_by="wandb",
                wandb_run=wandb_run,
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
                make_eval_network=make_noiseless_network,
                disrnn_config=disrnn_config,
                is_multisubject=is_multisubject,
                heldout_eval_cfg=heldout_eval_cfg,
                heldout_data=heldout_test_data,
                wandb_run=wandb_run,
                wandb_step=int(args.n_warmup_steps),
                keep_media_files=args.checkpoint_keep_media_files,
            )
            distillation_metrics = _compute_distillation_metrics(params)
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
                params, opt_state, losses = rnn_utils.train_network(
                    make_train_network,
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
            xs_full_for_checkpoint, _ = dataset.get_all()
            df_for_checkpoint = bundle.raw
            random_key = training_key

            while steps_completed < args.n_steps:
                chunk_steps = min(
                    args.checkpoint_every_n_steps,
                    args.n_steps - steps_completed,
                )
                if distillation_ensemble is None:
                    params, opt_state, chunk_losses = rnn_utils.train_network(
                        make_train_network,
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

                train_likelihood_ckpt: float | None = None
                if args.checkpoint_eval_on_train_split:
                    yhat_train_ckpt, _ = rnn_utils.eval_network(
                        make_noiseless_network,
                        params,
                        xs_train_all,
                    )
                    n_action_logits_train_ckpt = int(getattr(dataset_train, "n_classes", 0))
                    if n_action_logits_train_ckpt <= 0:
                        n_action_logits_train_ckpt = int(
                            np.asarray(yhat_train_ckpt).shape[2] - 1
                        )
                    if n_action_logits_train_ckpt <= 0:
                        raise ValueError(
                            "Invalid number of action logits inferred for checkpoint train likelihood."
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
                        make_noiseless_network,
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

                train_distillation_loss_ckpt: float | None = None
                eval_distillation_loss_ckpt: float | None = None
                if distillation_ensemble is not None:
                    train_distillation_loss_ckpt = evaluate_distillation_loss(
                        make_network=make_noiseless_network,
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
                        make_network=make_noiseless_network,
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
                    yhat_full_ckpt, network_states_full_ckpt = rnn_utils.eval_network(
                        make_noiseless_network,
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
        xs_full, ys_full = dataset.get_all()
        yhat_full, network_states_full = rnn_utils.eval_network(
            make_noiseless_network, params, xs_full
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
        xs_train, ys_train = dataset_train.get_all()
        yhat_train, _ = rnn_utils.eval_network(
            make_noiseless_network, params, xs_train
        )
        n_action_logits_train = int(getattr(dataset_train, "n_classes", 0))
        if n_action_logits_train <= 0:
            n_action_logits_train = int(yhat_train.shape[2] - 1)
        if n_action_logits_train <= 0:
            raise ValueError(
                "Invalid number of action logits inferred for final train likelihood: "
                f"{n_action_logits_train}"
            )
        likelihood_train = rnn_utils.normalized_likelihood(
            ys_train,
            yhat_train[:, :, :n_action_logits_train],
        )
        output["likelihood_train"] = float(likelihood_train)
        logger.info("Final training likelihood: %.4f", float(likelihood_train))

        xs_eval, ys_eval = dataset_eval.get_all()
        yhat_eval, network_states_eval = rnn_utils.eval_network(
            make_noiseless_network, params, xs_eval
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
        logger.info("Final eval likelihood: %.4f", float(likelihood))
        if distillation_ensemble is not None:
            final_train_distillation_loss = evaluate_distillation_loss(
                make_network=make_noiseless_network,
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
                make_network=make_noiseless_network,
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
            wandb_run.summary["final/val_loss"] = float(losses["validation_loss"][-1])
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
        log_scope: str | None = None,
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

        train_session_ids_meta = metadata.get("train_session_ids")
        eval_session_ids_meta = metadata.get("eval_session_ids")
        if train_session_ids_meta is not None or eval_session_ids_meta is not None:
            if not isinstance(train_session_ids_meta, list) or not isinstance(
                eval_session_ids_meta, list
            ):
                raise ValueError(
                    "metadata.train_session_ids and metadata.eval_session_ids must both be lists "
                    "when either is provided."
                )
            session_index_by_id = {
                session_id: index for index, session_id in enumerate(session_order)
            }
            missing_train_sessions = [
                session_id
                for session_id in train_session_ids_meta
                if session_id not in session_index_by_id
            ]
            missing_eval_sessions = [
                session_id
                for session_id in eval_session_ids_meta
                if session_id not in session_index_by_id
            ]
            if missing_train_sessions or missing_eval_sessions:
                raise ValueError(
                    "Split example metadata references sessions not present in output_df. "
                    f"Missing train={missing_train_sessions}, missing eval={missing_eval_sessions}"
                )
            train_indices = np.asarray(
                [session_index_by_id[session_id] for session_id in train_session_ids_meta],
                dtype=int,
            )
            eval_indices = np.asarray(
                [session_index_by_id[session_id] for session_id in eval_session_ids_meta],
                dtype=int,
            )
        else:
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
