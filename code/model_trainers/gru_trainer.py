from __future__ import annotations

import itertools
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
from matplotlib.lines import Line2D
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
from utils.multisubject import (
    extract_subject_embeddings_from_params,
    normalize_subject_id,
    save_subject_index_map,
    subject_embeddings_to_dataframe,
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


def _resolve_multisubject_subject_count(metadata: Mapping[str, Any]) -> int:
    num_subjects = metadata.get("num_subjects")
    if num_subjects is not None:
        resolved = int(num_subjects)
        if resolved > 0:
            return resolved

    subject_ids = metadata.get("subject_ids")
    if isinstance(subject_ids, list) and subject_ids:
        return int(len(subject_ids))

    raise ValueError(
        "Multisubject GRU requires metadata.num_subjects or metadata.subject_ids."
    )


def _validate_multisubject_dataset_inputs(
    dataset: Any,
    *,
    max_n_subjects: int,
    context: str,
) -> None:
    xs, _ = dataset.get_all()
    xs = np.asarray(xs)
    if xs.ndim != 3:
        raise ValueError(
            f"Multisubject GRU {context} expects dataset inputs to be 3D, got shape={xs.shape}."
        )
    if xs.shape[2] < 2:
        raise ValueError(
            f"Multisubject GRU {context} requires a prepended subject-index feature."
        )

    subject_ids = np.asarray(xs[..., 0], dtype=float)
    if not np.all(np.isfinite(subject_ids)):
        raise ValueError(
            f"Multisubject GRU {context} encountered non-finite subject ids."
        )

    rounded_subject_ids = np.rint(subject_ids)
    integer_like_mask = np.isclose(subject_ids, rounded_subject_ids)
    if not np.all(integer_like_mask):
        bad_values = np.unique(subject_ids[~integer_like_mask]).tolist()
        raise ValueError(
            "Multisubject GRU "
            f"{context} expected integer-valued subject ids, got {bad_values}."
        )

    subject_ids_int = rounded_subject_ids.astype(int)
    invalid_mask = (subject_ids_int < -1) | (subject_ids_int >= int(max_n_subjects))
    if np.any(invalid_mask):
        bad_values = np.unique(subject_ids_int[invalid_mask]).tolist()
        raise ValueError(
            "Multisubject GRU "
            f"{context} encountered out-of-range subject ids {bad_values}. "
            f"Expected padding sentinel -1 or values in [0, {int(max_n_subjects) - 1}]."
        )


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
        return {
            "subject_index_map": str(subject_index_map_path),
            "subject_embeddings": str(subject_embeddings_path),
        }

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
        make_network: Any,
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
                "checkpoint/step": int(wandb_step) if wandb_step is not None else 0,
                "checkpoint/train_likelihood": train_likelihood,
                "checkpoint/eval_likelihood": eval_likelihood,
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

        plot_paths: dict[str, Any] = {"subject_embedding_state_space": None}
        if is_multisubject:
            try:
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
            except Exception as exc:
                logger.warning(
                    "Initialization subject-embedding plotting failed for %s: %s",
                    stage_name,
                    exc,
                )

        if wandb_run is not None and plot_paths["subject_embedding_state_space"]:
            plot_payload = {
                "checkpoint/fig/subject_embedding_state_space": wandb.Image(
                    str(plot_paths["subject_embedding_state_space"])
                )
            }
            if wandb_step is None:
                wandb_run.log(plot_payload)
            else:
                wandb_run.log(plot_payload, step=wandb_step)
            if not keep_media_files:
                self._remove_media_files([plot_paths["subject_embedding_state_space"]])
                plot_paths["subject_embedding_state_space"] = None

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
            "train_likelihood": train_likelihood,
            "eval_likelihood": eval_likelihood,
            "plot_paths": plot_paths,
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
        is_multisubject = _is_multisubject_mode(self.architecture, metadata)

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
        output["multisubject"] = bool(is_multisubject)

        _log_dataset_split_details(
            dataset=dataset,
            dataset_train=dataset_train,
            dataset_eval=dataset_eval,
        )
        _log_heldout_dataset_details(self.heldout_data)

        key = jax.random.PRNGKey(self.seed)
        output["random_key"] = [int(x) for x in np.asarray(key).reshape(-1)]

        max_n_subjects: int | None = None
        subject_embedding_size: int | None = None
        subject_embedding_init = str(self.architecture.get("subject_embedding_init", "zeros"))
        if is_multisubject:
            subject_embedding_size = self.architecture.get("subject_embedding_size")
            if subject_embedding_size is None or int(subject_embedding_size) <= 0:
                raise ValueError(
                    "Multisubject GRU requires architecture.subject_embedding_size > 0."
                )
            max_n_subjects = _resolve_multisubject_subject_count(metadata)
            if not isinstance(metadata.get("subject_id_to_index"), dict) or not isinstance(
                metadata.get("index_to_subject_id"),
                dict,
            ):
                raise ValueError(
                    "Multisubject GRU requires subject_id_to_index and "
                    "index_to_subject_id in bundle metadata."
                )
            _validate_multisubject_dataset_inputs(
                dataset,
                max_n_subjects=max_n_subjects,
                context="full dataset",
            )
            _validate_multisubject_dataset_inputs(
                dataset_train,
                max_n_subjects=max_n_subjects,
                context="training split",
            )
            _validate_multisubject_dataset_inputs(
                dataset_eval,
                max_n_subjects=max_n_subjects,
                context="eval split",
            )

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
            initialization_eval_before_training=bool(
                self.training.get("initialization_eval_before_training", True)
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
            multisubject=is_multisubject,
            max_n_subjects=max_n_subjects,
            subject_embedding_size=(
                int(subject_embedding_size) if subject_embedding_size is not None else None
            ),
            subject_embedding_init=subject_embedding_init,
        )

        heldout_eval_cfg = None
        heldout_test_data = self.heldout_data
        runtime_heldout_cfg = HeldoutEvalConfig.from_data_cfg(metadata)
        if runtime_heldout_cfg.enabled and is_multisubject:
            logger.info(
                "Skipping held-out evaluation for multisubject GRU; "
                "v1 supports seen-subject personalization only."
            )
        elif runtime_heldout_cfg.enabled:
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
        if args.initialization_eval_before_training:
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
                    is_multisubject=is_multisubject,
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
            opt_state = init_opt_state

            all_training_losses: list[float] = []
            all_validation_losses: list[float] = []
            steps_completed = 0
            xs_train_all, ys_train_all = dataset_train.get_all()
            xs_eval_all, ys_eval_all = dataset_eval.get_all()
            xs_full_for_checkpoint, _ = dataset.get_all()
            df_for_checkpoint = bundle.raw
            random_key = key

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

                checkpoint_plot_paths: dict[str, Any] = {
                    "subject_embedding_state_space": None,
                }
                if is_multisubject:
                    try:
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
                    except Exception as exc:
                        logger.warning(
                            "Checkpoint subject-embedding plotting failed for step=%s: %s",
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
                    and args.checkpoint_log_eval_to_wandb
                    and checkpoint_plot_paths["subject_embedding_state_space"]
                ):
                    wandb_run.log(
                        {
                            "checkpoint/fig/subject_embedding_state_space": wandb.Image(
                                str(checkpoint_plot_paths["subject_embedding_state_space"])
                            )
                        },
                        step=int(steps_completed),
                    )
                    if not args.checkpoint_keep_media_files:
                        self._remove_media_files(
                            [checkpoint_plot_paths["subject_embedding_state_space"]]
                        )
                        checkpoint_plot_paths["subject_embedding_state_space"] = None

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
        if is_multisubject:
            output["subject_artifacts"] = self._save_multisubject_artifacts(
                params=params,
                metadata=metadata,
                raw_df=df,
            )

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
