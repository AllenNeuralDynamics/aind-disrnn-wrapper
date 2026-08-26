"""Shared base class for multisubject RNN trainers (GRU and disRNN).

Holds the logic that was historically duplicated verbatim between
``GruTrainer`` and ``DisrnnTrainer``: subject-embedding / session-context
state-space plots, multisubject artifact export, media cleanup, W&B held-out
logging, loss plotting, and session-conditioning logging. Subclasses set
``_MODEL_LABEL`` / ``_TRAINER_CONTEXT_NAME`` so emitted log/error strings remain
model-specific, and supply the model-specific training pipeline in ``fit``.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.lines import Line2D
from omegaconf import DictConfig, OmegaConf

from disentangled_rnns.library import rnn_utils

from base.interfaces import ModelTrainer
from base.types import DatasetBundle
from models.session_conditioning import resolve_session_conditioning_from_architecture
from utils.multisubject import (
    compute_session_conditioned_context_dataframe,
    extract_subject_embeddings_from_params,
    normalize_subject_id,
    ordered_session_context_rows,
    resolve_session_context_plot_subject_indices,
    save_multisubject_metadata,
    save_subject_index_map,
    save_session_context_map,
    subject_embeddings_to_dataframe,
)

logger = logging.getLogger(__name__)

# The state-space plots always show the leading _EMBEDDING_PLOT_N_PCS principal components of the
# subject embedding, paired up: C(4,2) = 6 panels per subject, at ANY embedding width. 4 matches
# the default subject_embedding_size, so the figure keeps the exact shape people already read.
#
# They used to pair up the RAW embedding dims, which is O(dim^2): subject_embedding_size=64 asks
# for C(64,2)=2016 panels PER SUBJECT -- a 1.69 GP figure that is unreadable, and that PIL rejects
# as a decompression bomb (it killed six training runs). Raw dims are not worth enumerating
# anyway: the embedding is defined only up to rotation, so "dim 3 vs dim 7" carries no special
# meaning. The PCs are rotation-invariant, use every dim, keep the panel count constant, and make
# runs at different embedding widths directly comparable -- while pushing far fewer images to W&B.
_EMBEDDING_PLOT_N_PCS = 4


def _project_embedding_columns_for_plot(
    plot_df: pd.DataFrame,
    embedding_columns: list[str],
) -> tuple[pd.DataFrame, list[str], Callable[[np.ndarray], np.ndarray]]:
    """Replace the raw embedding dims with their leading principal components, for plotting.

    Returns the augmented frame, the PC columns the caller should pair up, and the projection
    itself -- callers that overlay a raw embedding vector (e.g. the subject's base embedding, as
    a star) MUST push it through this same basis rather than indexing raw dims, or the overlay
    lands in a different space than the points around it.

    Coordinate ``i`` of a projected vector corresponds to column ``i`` of the returned columns.
    Fewer components are returned when the data cannot support 4 (a narrow embedding, or too few
    rows), in which case the raw columns and an identity projection are returned unchanged.
    """
    values = plot_df[embedding_columns].to_numpy(dtype=float)
    n_components = min(_EMBEDDING_PLOT_N_PCS, values.shape[0], values.shape[1])
    if n_components < 2:
        return plot_df, embedding_columns, lambda raw: raw

    mean = values.mean(axis=0, keepdims=True)
    centered = values - mean
    # Right singular vectors of the centered matrix are the principal axes.
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    basis = components[:n_components]
    projected = centered @ basis.T

    spectrum = np.square(singular_values)
    total = float(spectrum.sum())
    explained = (spectrum[:n_components] / total) if total > 0 else np.zeros(n_components)
    logger.info(
        "Projecting %d-dim subject embedding onto %d PCs for the state-space plot "
        "(explained variance: %s).",
        len(embedding_columns),
        n_components,
        ", ".join(
            f"PC{index + 1} {100.0 * float(value):.1f}%"
            for index, value in enumerate(explained)
        ),
    )

    def project(raw: np.ndarray) -> np.ndarray:
        """Map raw embedding coordinates into the plotted PC space (same centering + basis)."""
        raw = np.asarray(raw, dtype=float)
        return (raw - mean.reshape(-1)) @ basis.T

    plot_df = plot_df.copy()
    projected_columns = [f"embedding_pc{index + 1}" for index in range(n_components)]
    for index, column in enumerate(projected_columns):
        plot_df[column] = projected[:, index]
    return plot_df, projected_columns, project


def _to_dict(config: Mapping[str, Any] | DictConfig) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


class BaseMultisubjectTrainer(ModelTrainer):
    """Shared scaffolding + plotting/logging helpers for GRU and disRNN trainers."""

    # Overridden per subclass so emitted log/error text stays model-specific.
    _MODEL_LABEL: str = ""
    _TRAINER_CONTEXT_NAME: str = ""

    def __init__(
        self,
        architecture: Mapping[str, Any] | DictConfig,
        training: Mapping[str, Any] | DictConfig,
        heldout_data: dict[str, Any] | None = None,
        output_dir: str = "/results/outputs",
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.architecture = _to_dict(architecture)
        self.architecture.setdefault("session_delta_n_layers", 3)
        self.architecture.setdefault("session_delta_hidden_size", 16)
        self.training = _to_dict(training)
        self.heldout_data = heldout_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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
        requested_subject_indices = metadata.get("plot_subject_indices")
        if requested_subject_indices is None:
            requested_subject_indices = self.training.get("session_context_plot_subject_indices")
        return resolve_session_context_plot_subject_indices(
            session_context,
            requested_subject_indices=requested_subject_indices,
            max_subjects=int(self.training.get("session_context_plot_max_subjects", 3)),
            random_seed=(
                None if requested_subject_indices is not None else int(self.seed or 0)
            ),
        )

    def _log_session_conditioning_details(
        self,
        *,
        metadata: Mapping[str, Any],
        dataset: Any,
        session_conditioning_cfg: Mapping[str, Any],
        session_curriculum_cfg: Mapping[str, Any],
        total_training_steps: int,
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
            f"Session conditioning enabled for {self._MODEL_LABEL}: encoding=%s integration=%s fourier_k=%d "
            "delta_layers=%d delta_hidden=%d packed_input=[subject_idx, session_idx, *obs]",
            session_conditioning_cfg["session_encoding_type"],
            session_conditioning_cfg["session_integration_type"],
            int(session_conditioning_cfg["session_fourier_k"]),
            int(session_conditioning_cfg["session_delta_n_layers"]),
            int(session_conditioning_cfg["session_delta_hidden_size"]),
        )
        logger.info(
            "Session regularization lambda_reg_session=%s",
            float(self.training.get("lambda_reg_session", 0.0)),
        )
        logger.info(
            "Session curriculum steps: pretrain=%d warmup=%d total=%d",
            int(session_curriculum_cfg["session_n_pretrain_steps"]),
            int(session_curriculum_cfg["session_n_warmup_steps"]),
            int(total_training_steps),
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
        multisubject_metadata_path = save_multisubject_metadata(
            self.output_dir / "multisubject_metadata.json",
            metadata=metadata,
        )
        artifacts = {
            "subject_index_map": str(subject_index_map_path),
            "subject_embeddings": str(subject_embeddings_path),
            "multisubject_metadata": str(multisubject_metadata_path),
        }
        if str(self.architecture.get("session_encoding_type", "none")).strip().lower() != "none":
            session_context = metadata.get("session_context")
            if not isinstance(session_context, dict):
                raise ValueError(
                    f"Session-conditioned multisubject {self._MODEL_LABEL} export requires metadata.session_context."
                )
            session_context_map_path = save_session_context_map(
                self.output_dir / "session_context_map.json",
                session_context=session_context,
            )
            artifacts["session_context_map"] = str(session_context_map_path)
        return artifacts

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
        requested_plot_subject_indices = metadata.get("plot_subject_indices")
        if requested_plot_subject_indices is not None:
            requested_plot_subject_indices = {
                int(value) for value in requested_plot_subject_indices
            }
            plot_df = plot_df[
                plot_df["subject_index"].astype(int).isin(requested_plot_subject_indices)
            ].copy()
            if plot_df.empty:
                logger.info(
                    "Skipping subject embedding state-space plot because no requested plot "
                    "subjects are present."
                )
                return None
        if "curriculum_name" not in plot_df.columns:
            plot_df["curriculum_name"] = "Unknown"
        else:
            plot_df["curriculum_name"] = plot_df["curriculum_name"].fillna("Unknown")

        embedding_columns = [
            column
            for column in plot_df.columns
            if column.startswith("embedding_")
        ]
        plot_df, embedding_columns, _ = _project_embedding_columns_for_plot(
            plot_df, embedding_columns
        )
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
        session_curriculum_lambda: float = 1.0,
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
            subject_embedding_size=int(self.architecture["subject_embedding_size"]),
            context=f"{self._TRAINER_CONTEXT_NAME} session-context plotting",
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
            session_delta_n_layers=int(session_conditioning_cfg["session_delta_n_layers"]),
            session_delta_hidden_size=int(
                session_conditioning_cfg["session_delta_hidden_size"]
            ),
            session_curriculum_lambda=float(session_curriculum_lambda),
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

        # Same O(dim^2) blow-up as the embedding state-space plot above, but worse: this figure
        # repeats the whole pair grid PER SUBJECT, so a 64-dim embedding reached 1.69 GP.
        plot_df, embedding_columns, project_embedding = _project_embedding_columns_for_plot(
            plot_df, embedding_columns
        )
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
                # The star is the subject's base embedding. It must go through the SAME
                # projection as the session points around it -- indexing raw dims by the column
                # name would put it in a different space entirely once we plot PCs.
                projected_subject_embedding = project_embedding(subject_embedding)
                x_dim = embedding_columns.index(x_column)
                y_dim = embedding_columns.index(y_column)
                ax.scatter(
                    float(projected_subject_embedding[x_dim]),
                    float(projected_subject_embedding[y_dim]),
                    marker="*",
                    s=160,
                    c="black",
                    edgecolors="white",
                    linewidths=0.8,
                    zorder=3,
                )
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
        engage_plot_paths = heldout_summary.get("plots", {}).get(
            "latents_in_space_engage_examples",
            [],
        )
        if engage_plot_paths:
            image_payload[
                f"{wandb_key_prefix}/heldout/latents_in_space_engage_examples"
            ] = [wandb.Image(str(path)) for path in engage_plot_paths]
        if image_payload:
            if wandb_step is None:
                wandb_run.log(image_payload)
            else:
                wandb_run.log(image_payload, step=wandb_step)

    def _save_figure(self, fig: Any, filename: str) -> Path:
        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        return path

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
        network_states_full: np.ndarray,
        yhat_full: np.ndarray,
        params: Any,
        metadata: dict[str, Any],
        n_action_logits: int,
        output_df: pd.DataFrame | None = None,
        raw_df: pd.DataFrame | None = None,
        ignore_policy: str | None = None,
        wandb_run: Any | None = None,
        log_scope: str | None = None,
        wandb_step: int | None = None,
        wandb_key_prefix: str | None = None,
    ) -> dict[str, Any]:
        # Only a handful of subjects are ever plotted, so a caller can avoid
        # materializing the whole-cohort per-trial frame (the eval OOM at scale)
        # by passing raw_df (+ ignore_policy) and leaving output_df=None; each
        # split's frame is then built on demand from the sliced tensors below.
        if output_df is None and raw_df is None:
            raise ValueError(
                "_generate_split_examples requires either output_df or raw_df."
            )
        session_source_df = output_df if output_df is not None else raw_df
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

        session_order = list(dict.fromkeys(session_source_df["ses_idx"].tolist()))
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
            states_split = np.asarray(network_states_full)[:, split_indices, :]
            yhat_split = np.asarray(yhat_full)[:, split_indices, :]
            if output_df is not None:
                split_output_df = output_df[output_df["ses_idx"].isin(split_session_ids)].copy()
            else:
                # Build only this split's per-trial frame from the raw trials +
                # sliced tensors, so we never materialize the whole-cohort frame.
                split_raw = raw_df[raw_df["ses_idx"].isin(split_session_ids)].copy()
                split_raw["ses_idx"] = pd.Categorical(
                    split_raw["ses_idx"], categories=split_session_ids, ordered=True
                )
                split_raw = split_raw.sort_values(["ses_idx", "trial"]).copy()
                split_raw["ses_idx"] = split_raw["ses_idx"].astype(str)
                split_output_df = self._add_model_results(
                    split_raw, states_split, yhat_split, ignore_policy=ignore_policy
                )
            split_output_df["ses_idx"] = pd.Categorical(
                split_output_df["ses_idx"],
                categories=split_session_ids,
                ordered=True,
            )
            split_output_df = split_output_df.sort_values(["ses_idx", "trial"]).copy()
            split_output_df["ses_idx"] = split_output_df["ses_idx"].astype(str)

            try:
                split_summary = self._plot_examples_for_split(
                    split_name=split_name,
                    output_dir=output_dir,
                    output_df=split_output_df,
                    network_states=states_split,
                    yhat_logits=yhat_split,
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
                # 3-way (ignore-included) head: also log the P(ignore)-colored
                # engagement panel when the split plotter produced it. Absent for a
                # 2-way head, so the key is only added when non-empty.
                engage_paths = split_summary.get("plots", {}).get(
                    "latents_in_space_engage_examples", []
                )
                if engage_paths:
                    payload[f"{key_prefix}/latents_in_space_engage_examples"] = [
                        wandb.Image(str(path)) for path in engage_paths
                    ]
                if wandb_step is None:
                    wandb_run.log(payload)
                else:
                    wandb_run.log(payload, step=wandb_step)

        return split_summaries

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
        """Model-specific per-split example plotting. Implemented by subclasses."""
        raise NotImplementedError

    # Subclasses set a positive value to run eval_network in chunks along the
    # episode (session) axis. eval_network JIT-runs the model over all episodes
    # at once, which exhausts GPU memory for wide models on large multisubject
    # cohorts (e.g. GRU hidden_size=256 over ~18k sessions OOM'd a 48GB L40s).
    # None => single call (unchanged; the default for disRNN).
    _eval_max_episodes: int | None = None

    def _eval_network_full(self, make_network, params, xs):
        """eval_network, optionally chunked over the episode axis to bound GPU
        memory. Episodes are independent in the forward pass, so per-chunk
        results concatenated along axis=1 match a single call for a deterministic
        eval network."""
        xs = np.asarray(xs)
        cap = self._eval_max_episodes
        if not cap or xs.shape[1] <= cap:
            return rnn_utils.eval_network(make_network, params, xs)
        yhats, states = [], []
        for start in range(0, xs.shape[1], cap):
            y, s = rnn_utils.eval_network(
                make_network, params, xs[:, start : start + cap, :]
            )
            yhats.append(np.asarray(y))
            states.append(np.asarray(s))
        return np.concatenate(yhats, axis=1), np.concatenate(states, axis=1)

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
        make_eval_network: Any,
        is_multisubject: bool,
        heldout_eval_cfg: Any | None,
        heldout_data: dict[str, Any] | None,
        wandb_run: Any | None,
        wandb_step: int | None,
        keep_media_files: bool,
        session_curriculum_lambda: float = 1.0,
        disrnn_config: Any = None,
    ) -> dict[str, Any]:
        """Shared checkpoint/initialization snapshot pipeline for GRU and disRNN.

        Computes train/eval likelihood, logs metrics, runs model-specific
        diagnostics, evaluates the full dataset, generates split examples, runs
        held-out evaluation, and writes ``summary.json``. Model-specific behavior
        is delegated to the ``_resolve_n_action_logits``, ``_add_model_results``,
        ``_plot_model_specific_diagnostics``, ``_evaluate_heldout_subjects`` and
        ``_snapshot_extra_summary_fields`` hooks.
        """
        stage_dir = self.output_dir / "initialization" / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        log_scope = stage_name.replace("_", " ").title()
        wandb_key_prefix = "checkpoint"

        params_path = stage_dir / "params.json"
        with params_path.open("w") as f:
            f.write(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))

        _all = dataset_train.get_all()
        xs_train, ys_train = _all["xs"], _all["ys"]
        yhat_train, _ = self._eval_network_full(make_eval_network, params, xs_train)
        n_action_logits_train = self._resolve_n_action_logits(
            dataset_train, yhat_train, context="initialization train"
        )
        train_likelihood = float(
            rnn_utils.normalized_likelihood(
                ys_train, np.asarray(yhat_train)[:, :, :n_action_logits_train]
            )
        )

        _all = dataset_eval.get_all()
        xs_eval, ys_eval = _all["xs"], _all["ys"]
        yhat_eval, _ = self._eval_network_full(make_eval_network, params, xs_eval)
        n_action_logits_eval = self._resolve_n_action_logits(
            dataset_eval, yhat_eval, context="initialization eval"
        )
        eval_likelihood = float(
            rnn_utils.normalized_likelihood(
                ys_eval, np.asarray(yhat_eval)[:, :, :n_action_logits_eval]
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

        xs_full = dataset.get_all()["xs"]
        yhat_full, network_states_full = self._eval_network_full(
            make_eval_network, params, xs_full
        )
        # The snapshot's per-trial frame feeds only the split-example plots
        # below, which build per-subject frames on demand from the tensors; do
        # not materialize the whole-cohort frame here (the eval OOM at scale).

        plot_paths = self._plot_model_specific_diagnostics(
            params=params,
            disrnn_config=disrnn_config,
            stage_dir=stage_dir,
            stage_name=stage_name,
            is_multisubject=is_multisubject,
            metadata=metadata,
            raw_df=bundle.raw,
            session_curriculum_lambda=session_curriculum_lambda,
            wandb_run=wandb_run,
            wandb_step=wandb_step,
            keep_media_files=keep_media_files,
        )

        n_action_logits_full = self._resolve_n_action_logits(
            dataset, yhat_full, context="initialization full-dataset plotting"
        )
        split_summaries = self._generate_split_examples(
            output_dir=stage_dir,
            raw_df=bundle.raw,
            ignore_policy=ignore_policy,
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
        snapshot_summary.update(
            self._snapshot_extra_summary_fields(
                session_curriculum_lambda=session_curriculum_lambda
            )
        )

        if heldout_eval_cfg is not None:
            try:
                heldout_summary = self._evaluate_heldout_subjects(
                    heldout_eval_cfg=heldout_eval_cfg,
                    wandb_run=wandb_run,
                    params_path=params_path,
                    output_subdir=f"initialization/{stage_name}/heldout_test",
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

    # --- Model-specific hooks (overridden by subclasses) ----------------------

    def _resolve_n_action_logits(self, dataset: Any, yhat: Any, *, context: str) -> int:
        raise NotImplementedError

    def _add_model_results(self, raw_df, network_states, yhat, *, ignore_policy):
        raise NotImplementedError

    def _evaluate_heldout_subjects(
        self,
        *,
        heldout_eval_cfg,
        wandb_run,
        params_path,
        output_subdir,
        heldout_data,
        log_scope,
    ):
        raise NotImplementedError

    def _snapshot_extra_summary_fields(self, *, session_curriculum_lambda: float) -> dict:
        return {}
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
        raise NotImplementedError
