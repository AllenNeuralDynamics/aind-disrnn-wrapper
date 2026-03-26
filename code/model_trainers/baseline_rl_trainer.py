from __future__ import annotations

import concurrent.futures
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import wandb
from omegaconf import DictConfig, OmegaConf

from aind_dynamic_foraging_basic_analysis import plot_foraging_session
from aind_dynamic_foraging_models import generative_model
from aind_dynamic_foraging_models.generative_model.params import ParamsSymbols
from disentangled_rnns.library import rnn_utils
from utils.baseline_rl_evaluation import (
    _align_choice_prob_session,
    _align_q_session,
    _extract_q_histories,
    _normalize_identifier,
    _plot_q_values_for_session,
    _safe_filename_component,
    save_baseline_rl_output,
)

from base.interfaces import ModelTrainer
from base.types import DatasetBundle
from utils.multisubject import (
    compute_train_eval_session_ids,
    normalize_subject_id,
    save_subject_index_map,
)

logger = logging.getLogger(__name__)


def _to_dict(config: Mapping[str, Any] | DictConfig) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


def _compute_negLL_from_choice_prob(
    choice_prob: np.ndarray,
    choices: np.ndarray,
) -> float:
    """Compute negative log-likelihood from choice probabilities.

    Args:
        choice_prob: Array of shape (n_actions, n_trials) with choice probabilities.
        choices: Array of shape (n_trials,) with actual choices (0 or 1).

    Returns:
        Total negative log-likelihood.
    """
    n_trials = len(choices)
    log_lik = 0.0
    for t in range(n_trials):
        c = int(choices[t])
        prob = choice_prob[c, t]
        # Clamp probability to avoid log(0)
        prob = np.clip(prob, 1e-10, 1.0 - 1e-10)
        log_lik += np.log(prob)
    return -log_lik


def _compute_likelihood_stats(
    choice_sessions: List[np.ndarray],
    choice_prob_sessions: List[np.ndarray],
) -> dict[str, float | int]:
    """Return aggregate log-likelihood stats plus normalized likelihood."""
    total_log_lik = 0.0
    total_trials = 0

    for choices, choice_prob in zip(choice_sessions, choice_prob_sessions):
        choice_prob = np.asarray(choice_prob)
        n_trials = min(len(choices), int(choice_prob.shape[1]))
        if n_trials == 0:
            continue
        choices_idx = np.asarray(choices[:n_trials], dtype=int)
        valid = choices_idx >= 0
        if not np.any(valid):
            continue
        trial_idx = np.arange(n_trials)[valid]
        probs = choice_prob[choices_idx[valid], trial_idx]
        probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
        total_log_lik += float(np.sum(np.log(probs)))
        total_trials += int(np.sum(valid))

    normalized_likelihood = _normalized_likelihood_from_log_stats(
        total_log_lik,
        total_trials,
    )
    return {
        "total_log_likelihood": float(total_log_lik),
        "total_trials": int(total_trials),
        "normalized_likelihood": float(normalized_likelihood),
    }


def _normalized_likelihood_from_log_stats(
    total_log_likelihood: float,
    total_trials: int,
) -> float:
    """Return geometric-mean likelihood from aggregate log-likelihood stats."""
    if total_trials <= 0:
        return 0.0
    return float(np.exp(float(total_log_likelihood) / float(total_trials)))


def _json_default(value: Any) -> Any:
    """Convert numpy-backed values into JSON-serializable Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _fit_subject_sessions_impl(
    *,
    agent_class: str,
    agent_kwargs: Mapping[str, Any],
    fit_bounds_override: Mapping[str, Any],
    clamp_params: Mapping[str, Any],
    DE_kwargs: Mapping[str, Any],
    seed: int | None,
    train_choices: List[np.ndarray],
    train_rewards: List[np.ndarray],
    eval_choices: List[np.ndarray],
    eval_rewards: List[np.ndarray],
    include_choice_prob_sessions: bool,
) -> dict[str, Any]:
    """Fit one subject's sessions and return summary metrics."""
    agent_class_obj = getattr(generative_model, agent_class, None)
    if agent_class_obj is None:
        raise ValueError(
            f"Agent class '{agent_class}' not found in "
            f"aind_dynamic_foraging_models.generative_model"
        )

    agent = agent_class_obj(
        **dict(agent_kwargs),
        seed=seed,
    )
    fitting_result, _ = agent.fit(
        fit_choice_history=train_choices,
        fit_reward_history=train_rewards,
        fit_bounds_override=dict(fit_bounds_override),
        clamp_params=dict(clamp_params),
        DE_kwargs=dict(DE_kwargs),
    )

    fitted_params = {
        param_name: float(param_value)
        for param_name, param_value in fitting_result.params.items()
    }
    agent.set_params(**fitted_params)

    train_choice_prob_sessions = agent.perform_closed_loop_multi_session(
        train_choices,
        train_rewards,
    )
    train_choice_prob_sessions = [np.asarray(arr) for arr in train_choice_prob_sessions]
    train_stats = _compute_likelihood_stats(
        train_choices,
        train_choice_prob_sessions,
    )

    eval_agent = agent_class_obj(**dict(agent_kwargs), seed=seed)
    eval_agent.set_params(**fitted_params)
    eval_choice_prob_sessions = eval_agent.perform_closed_loop_multi_session(
        eval_choices,
        eval_rewards,
    )
    eval_choice_prob_sessions = [np.asarray(arr) for arr in eval_choice_prob_sessions]
    eval_stats = _compute_likelihood_stats(
        eval_choices,
        eval_choice_prob_sessions,
    )

    output = {
        "fitted_params": fitted_params,
        "n_free_params": int(fitting_result.k_model),
        "train_likelihood": float(train_stats["normalized_likelihood"]),
        "eval_likelihood": float(eval_stats["normalized_likelihood"]),
        "train_total_log_likelihood": float(train_stats["total_log_likelihood"]),
        "train_total_trials": int(train_stats["total_trials"]),
        "eval_total_log_likelihood": float(eval_stats["total_log_likelihood"]),
        "eval_total_trials": int(eval_stats["total_trials"]),
        "log_likelihood_train": float(fitting_result.log_likelihood),
        "LPT_train": float(fitting_result.LPT),
        "AIC": float(fitting_result.AIC),
        "BIC": float(fitting_result.BIC),
    }
    if include_choice_prob_sessions:
        output["train_choice_prob_sessions"] = train_choice_prob_sessions
        output["eval_choice_prob_sessions"] = eval_choice_prob_sessions
    return output


def _fit_multisubject_subject_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Worker entrypoint for fitting one subject in multisubject baseline RL."""
    fit_summary = _fit_subject_sessions_impl(
        agent_class=str(payload["agent_class"]),
        agent_kwargs=dict(payload["agent_kwargs"]),
        fit_bounds_override=dict(payload["fit_bounds_override"]),
        clamp_params=dict(payload["clamp_params"]),
        DE_kwargs=dict(payload["DE_kwargs"]),
        seed=payload.get("seed"),
        train_choices=list(payload["train_choices"]),
        train_rewards=list(payload["train_rewards"]),
        eval_choices=list(payload["eval_choices"]),
        eval_rewards=list(payload["eval_rewards"]),
        include_choice_prob_sessions=False,
    )
    return {
        "subject_id": payload["subject_id"],
        "subject_index": int(payload["subject_index"]),
        "curriculum_name": payload["curriculum_name"],
        "train_session_ids": list(payload["train_session_ids"]),
        "eval_session_ids": list(payload["eval_session_ids"]),
        **fit_summary,
    }


class BaselineRLTrainer(ModelTrainer):
    """Trainer for baseline RL model comparisons using aind-dynamic-foraging-models.

    This trainer fits RL cognitive models (e.g., Q-learning variants) to behavioral
    data and computes likelihood metrics on held-out evaluation sessions for
    comparison with disRNN models.
    """

    def __init__(
        self,
        agent_class: str,
        architecture: Mapping[str, Any] | DictConfig = {},
        agent_kwargs: Mapping[str, Any] | DictConfig = {},
        fit_bounds_override: Mapping[str, Any] | DictConfig = {},
        clamp_params: Mapping[str, Any] | DictConfig = {},
        DE_kwargs: Mapping[str, Any] | DictConfig = {"workers": 1},
        multisubject_subject_workers: int = 1,
        output_dir: str = "/results/outputs",
        seed: int | None = None,
        **_: Any,
    ) -> None:
        """Initialize the BaselineRLTrainer.

        Args:
            agent_class: Name of the agent class from aind_dynamic_foraging_models
                (e.g., "ForagerQLearning", "ForagerLossCounting").
            agent_kwargs: Keyword arguments defining agent hyperparameters
                (e.g., number_of_learning_rate, choice_kernel, action_selection).
            fit_bounds_override: Override default parameter bounds for MLE fitting.
            clamp_params: Parameters to fix to specified values during fitting.
            DE_kwargs: Keyword arguments for scipy's differential_evolution optimizer.
            output_dir: Directory to save output files.
            seed: Random seed for reproducibility.
        """
        super().__init__(seed=seed)
        self.agent_class = agent_class
        self.architecture = _to_dict(architecture)
        self.agent_kwargs = _to_dict(agent_kwargs)
        self.fit_bounds_override = _to_dict(fit_bounds_override)
        self.clamp_params = _to_dict(clamp_params)
        self.DE_kwargs = _to_dict(DE_kwargs)
        self.multisubject_subject_workers = int(multisubject_subject_workers)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _is_multisubject_mode(self, metadata: Mapping[str, Any]) -> bool:
        return bool(
            self.architecture.get("multisubject", False)
            or metadata.get("multisubject", False)
        )

    def _resolve_agent_class(self) -> Any:
        agent_class_obj = getattr(generative_model, self.agent_class, None)
        if agent_class_obj is None:
            raise ValueError(
                f"Agent class '{self.agent_class}' not found in "
                f"aind_dynamic_foraging_models.generative_model"
            )
        return agent_class_obj

    def _resolve_multisubject_subject_workers(self, n_subjects: int) -> int:
        if self.multisubject_subject_workers <= 0:
            raise ValueError("multisubject_subject_workers must be a positive integer.")
        if n_subjects <= 0:
            raise ValueError("n_subjects must be positive.")
        return min(int(self.multisubject_subject_workers), int(n_subjects))

    def _effective_multisubject_de_kwargs(self) -> dict[str, Any]:
        effective_kwargs = dict(self.DE_kwargs)
        effective_kwargs["workers"] = 1
        return effective_kwargs

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

    def _extract_sessions_from_raw_df(
        self,
        raw_df: pd.DataFrame,
        *,
        session_ids: List[Any] | None = None,
    ) -> tuple[List[np.ndarray], List[np.ndarray], List[Any]]:
        required_cols = {"ses_idx", "trial", "animal_response", "earned_reward"}
        missing = [column for column in required_cols if column not in raw_df.columns]
        if missing:
            raise ValueError(f"Raw dataframe missing required columns for baseline RL: {missing}")

        ordered_session_ids = (
            list(session_ids)
            if session_ids is not None
            else list(dict.fromkeys(raw_df["ses_idx"].tolist()))
        )
        choices_sessions: List[np.ndarray] = []
        rewards_sessions: List[np.ndarray] = []
        kept_session_ids: List[Any] = []

        for session_id in ordered_session_ids:
            session_df = raw_df[raw_df["ses_idx"] == session_id].sort_values("trial")
            if session_df.empty:
                continue
            choice_arr = session_df["animal_response"].to_numpy(dtype=int)
            valid_choice = (choice_arr == 0) | (choice_arr == 1)
            choice_arr = choice_arr[valid_choice]
            if len(choice_arr) == 0:
                continue
            reward_arr = session_df["earned_reward"].to_numpy(dtype=float)[valid_choice]
            choices_sessions.append(choice_arr.astype(int))
            rewards_sessions.append(reward_arr.astype(float))
            kept_session_ids.append(session_id)

        return choices_sessions, rewards_sessions, kept_session_ids

    def _build_multisubject_subject_records(
        self,
        *,
        raw_df: pd.DataFrame,
        metadata: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        if "subject_id" not in raw_df.columns:
            raise ValueError(
                "Multisubject baseline RL requires a 'subject_id' column in bundle.raw."
            )

        subject_id_to_index = metadata.get("subject_id_to_index")
        index_to_subject_id = metadata.get("index_to_subject_id")
        if not isinstance(subject_id_to_index, dict) or not isinstance(index_to_subject_id, dict):
            raise ValueError(
                "Multisubject baseline RL requires subject_id_to_index and "
                "index_to_subject_id in bundle metadata."
            )

        eval_every_n = int(metadata.get("eval_every_n", 2))
        train_session_ids_meta = metadata.get("train_session_ids")
        eval_session_ids_meta = metadata.get("eval_session_ids")
        train_session_id_set = (
            {str(session_id) for session_id in train_session_ids_meta}
            if isinstance(train_session_ids_meta, list)
            else None
        )
        eval_session_id_set = (
            {str(session_id) for session_id in eval_session_ids_meta}
            if isinstance(eval_session_ids_meta, list)
            else None
        )

        subject_curricula = self._resolve_subject_curricula(raw_df=raw_df, metadata=metadata)
        ordered_subjects = sorted(
            (
                (int(index), normalize_subject_id(subject_id))
                for index, subject_id in index_to_subject_id.items()
            ),
            key=lambda item: item[0],
        )

        subject_records: list[dict[str, Any]] = []
        normalized_subject_series = raw_df["subject_id"].map(normalize_subject_id)
        for subject_index, subject_id in ordered_subjects:
            subject_df = raw_df[normalized_subject_series == subject_id].copy()
            if subject_df.empty:
                continue
            subject_df = subject_df.sort_values(["ses_idx", "trial"]).reset_index(drop=True)
            session_ids = list(dict.fromkeys(subject_df["ses_idx"].tolist()))

            if train_session_id_set is not None and eval_session_id_set is not None:
                train_session_ids = [
                    session_id
                    for session_id in session_ids
                    if str(session_id) in train_session_id_set
                ]
                eval_session_ids = [
                    session_id
                    for session_id in session_ids
                    if str(session_id) in eval_session_id_set
                ]
                if not train_session_ids or not eval_session_ids:
                    logger.warning(
                        "Could not resolve train/eval session ids from metadata for subject_id=%s; "
                        "falling back to eval_every_n=%s.",
                        subject_id,
                        eval_every_n,
                    )
                    train_session_ids, eval_session_ids = compute_train_eval_session_ids(
                        session_ids,
                        eval_every_n=eval_every_n,
                    )
            else:
                train_session_ids, eval_session_ids = compute_train_eval_session_ids(
                    session_ids,
                    eval_every_n=eval_every_n,
                )

            train_choices, train_rewards, train_session_ids = self._extract_sessions_from_raw_df(
                subject_df,
                session_ids=train_session_ids,
            )
            eval_choices, eval_rewards, eval_session_ids = self._extract_sessions_from_raw_df(
                subject_df,
                session_ids=eval_session_ids,
            )

            if not train_choices or not eval_choices:
                raise ValueError(
                    "Subject split for baseline RL produced an empty train or eval set for "
                    f"subject_id={subject_id}."
                )

            subject_records.append(
                {
                    "subject_id": normalize_subject_id(subject_id),
                    "subject_index": int(subject_index),
                    "curriculum_name": subject_curricula.get(
                        normalize_subject_id(subject_id),
                        "Unknown",
                    ),
                    "train_choices": train_choices,
                    "train_rewards": train_rewards,
                    "eval_choices": eval_choices,
                    "eval_rewards": eval_rewards,
                    "train_session_ids": train_session_ids,
                    "eval_session_ids": eval_session_ids,
                }
            )

        if not subject_records:
            raise ValueError("No subject records were constructed for multisubject baseline RL.")

        return subject_records

    def _fit_subject_sessions(
        self,
        *,
        train_choices: List[np.ndarray],
        train_rewards: List[np.ndarray],
        eval_choices: List[np.ndarray],
        eval_rewards: List[np.ndarray],
        include_choice_prob_sessions: bool = True,
    ) -> dict[str, Any]:
        logger.info(
            "Fitting %s with kwargs: %s",
            self.agent_class,
            self.agent_kwargs,
        )
        agent_class_obj = self._resolve_agent_class()
        agent_for_metadata = agent_class_obj(**self.agent_kwargs, seed=self.seed)
        logger.info(
            "Agent has %d free parameters",
            len(agent_for_metadata.params_list_free),
        )
        return _fit_subject_sessions_impl(
            agent_class=self.agent_class,
            agent_kwargs=self.agent_kwargs,
            fit_bounds_override=self.fit_bounds_override,
            clamp_params=self.clamp_params,
            DE_kwargs=self.DE_kwargs,
            seed=self.seed,
            train_choices=train_choices,
            train_rewards=train_rewards,
            eval_choices=eval_choices,
            eval_rewards=eval_rewards,
            include_choice_prob_sessions=include_choice_prob_sessions,
        )

    def _parameter_display_name(self, param_name: str) -> str:
        try:
            return str(ParamsSymbols[param_name].value)
        except KeyError:
            return param_name

    def _plot_subject_parameter_state_space(
        self,
        subject_metrics_df: pd.DataFrame,
        *,
        parameter_columns: List[str],
    ) -> Figure | None:
        varying_param_columns = [
            column
            for column in parameter_columns
            if column in subject_metrics_df.columns
            and subject_metrics_df[column].nunique(dropna=False) > 1
        ]
        if len(varying_param_columns) < 2:
            logger.info(
                "Skipping subject parameter state-space plot because fewer than two fitted "
                "parameter columns vary across subjects."
            )
            return None

        plot_df = subject_metrics_df.copy()
        if "curriculum_name" not in plot_df.columns:
            plot_df["curriculum_name"] = "Unknown"
        else:
            plot_df["curriculum_name"] = plot_df["curriculum_name"].fillna("Unknown")

        dim_pairs = list(itertools.combinations(varying_param_columns, 2))
        ncols = min(3, len(dim_pairs))
        nrows = int(np.ceil(len(dim_pairs) / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(5.5 * ncols, 5.0 * nrows),
            squeeze=False,
        )

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
                    str(int(getattr(row, "subject_index"))),
                    fontsize=8,
                    alpha=0.8,
                )
            ax.axhline(0, color="0.85", linewidth=1)
            ax.axvline(0, color="0.85", linewidth=1)
            ax.set_xlabel(self._parameter_display_name(x_column))
            ax.set_ylabel(self._parameter_display_name(y_column))
            ax.set_title(
                f"{self._parameter_display_name(x_column)} vs "
                f"{self._parameter_display_name(y_column)}"
            )

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
        fig.suptitle("Subject Parameter State Space", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return fig

    def _plot_subject_likelihood_scatter(
        self,
        subject_metrics_df: pd.DataFrame,
        *,
        mean_subject_train_likelihood: float,
        mean_subject_eval_likelihood: float,
        pooled_train_trial_likelihood: float,
        pooled_eval_trial_likelihood: float,
    ) -> Figure:
        plot_df = subject_metrics_df.sort_values("subject_index").reset_index(drop=True).copy()
        if "curriculum_name" not in plot_df.columns:
            plot_df["curriculum_name"] = "Unknown"
        else:
            plot_df["curriculum_name"] = plot_df["curriculum_name"].fillna("Unknown")

        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), squeeze=False)
        curriculum_values = list(dict.fromkeys(plot_df["curriculum_name"].astype(str).tolist()))
        cmap = plt.get_cmap("tab10")
        curriculum_to_color = {
            curriculum: cmap(index % max(1, cmap.N))
            for index, curriculum in enumerate(curriculum_values)
        }
        rng = np.random.default_rng(0)
        plot_specs = [
            (
                axes[0, 0],
                "train_likelihood",
                "Train Likelihood",
                mean_subject_train_likelihood,
                pooled_train_trial_likelihood,
            ),
            (
                axes[0, 1],
                "eval_likelihood",
                "Eval Likelihood",
                mean_subject_eval_likelihood,
                pooled_eval_trial_likelihood,
            ),
        ]

        for panel_index, (ax, metric_column, title, mean_value, pooled_value) in enumerate(plot_specs):
            jitter = rng.uniform(-0.35, 0.35, size=len(plot_df))
            for curriculum in curriculum_values:
                curriculum_rows = plot_df[plot_df["curriculum_name"].astype(str) == curriculum]
                curriculum_indices = curriculum_rows.index.to_numpy(dtype=int)
                ax.scatter(
                    curriculum_rows[metric_column],
                    jitter[curriculum_indices],
                    color=curriculum_to_color[curriculum],
                    s=65,
                    alpha=0.9,
                    label=curriculum,
                )

            x_values = plot_df[metric_column].to_numpy(dtype=float)
            x_range = float(np.max(x_values) - np.min(x_values)) if len(x_values) > 0 else 0.0
            x_pad = max(0.0025, x_range * 0.01)
            for row_index, row in plot_df.iterrows():
                label_y_offset = 0.035 if row_index % 2 == 0 else -0.045
                ax.text(
                    float(row[metric_column]) + x_pad,
                    float(jitter[row_index]) + label_y_offset,
                    str(int(row["subject_index"])),
                    fontsize=8,
                    alpha=0.85,
                )

            ax.axvline(
                mean_value,
                color="black",
                linestyle="--",
                linewidth=1.8,
            )
            ax.axvline(
                pooled_value,
                color="dimgray",
                linestyle=":",
                linewidth=2.0,
            )
            x_margin = max(0.01, x_range * 0.08 if x_range > 0 else 0.02)
            ax.set_xlim(float(np.min(x_values)) - x_margin, float(np.max(x_values)) + x_margin)
            ax.set_ylim(-0.55, 0.55)
            ax.set_yticks([])
            ax.set_ylabel("")
            ax.set_xlabel("Likelihood")
            ax.set_title(f"{title} (n={len(plot_df)})")
            ax.grid(axis="x", color="0.9", linewidth=1)

        curriculum_handles = [
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
        line_handles = [
            Line2D([0], [0], color="black", linestyle="--", linewidth=1.8, label="Mean subject"),
            Line2D([0], [0], color="dimgray", linestyle=":", linewidth=2.0, label="Pooled trial"),
        ]
        fig.legend(
            handles=curriculum_handles + line_handles,
            loc="upper center",
            ncol=min(len(curriculum_handles) + len(line_handles), 5),
        )
        fig.suptitle("Per-Subject Likelihood Scatter", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        return fig

    def _fit_multisubject(
        self,
        *,
        bundle: DatasetBundle,
        metadata: dict[str, Any],
        wandb_run: Any | None,
        start_time: float,
    ) -> Dict[str, Any]:
        if bundle.raw is None or not hasattr(bundle.raw, "columns"):
            raise ValueError(
                "Multisubject baseline RL requires bundle.raw to be a dataframe-like object."
            )
        raw_df = pd.DataFrame(bundle.raw).copy()
        subject_records = self._build_multisubject_subject_records(
            raw_df=raw_df,
            metadata=metadata,
        )

        subject_id_to_index = metadata.get("subject_id_to_index")
        index_to_subject_id = metadata.get("index_to_subject_id")
        if not isinstance(subject_id_to_index, dict) or not isinstance(index_to_subject_id, dict):
            raise ValueError(
                "Multisubject baseline RL requires subject_id_to_index and "
                "index_to_subject_id in bundle metadata."
            )

        subject_output_dir = self.output_dir / "subjects"
        subject_output_dir.mkdir(parents=True, exist_ok=True)

        subject_parallel_workers = self._resolve_multisubject_subject_workers(
            len(subject_records)
        )
        effective_subject_de_kwargs = self._effective_multisubject_de_kwargs()
        worker_payloads = [
            {
                "agent_class": self.agent_class,
                "agent_kwargs": self.agent_kwargs,
                "fit_bounds_override": self.fit_bounds_override,
                "clamp_params": self.clamp_params,
                "DE_kwargs": effective_subject_de_kwargs,
                "seed": self.seed,
                "subject_id": subject_record["subject_id"],
                "subject_index": int(subject_record["subject_index"]),
                "curriculum_name": str(subject_record["curriculum_name"]),
                "train_choices": subject_record["train_choices"],
                "train_rewards": subject_record["train_rewards"],
                "eval_choices": subject_record["eval_choices"],
                "eval_rewards": subject_record["eval_rewards"],
                "train_session_ids": subject_record["train_session_ids"],
                "eval_session_ids": subject_record["eval_session_ids"],
            }
            for subject_record in subject_records
        ]

        per_subject_rows: list[dict[str, Any]] = []
        parameter_columns: List[str] | None = None
        per_subject_results: list[dict[str, Any]] = []

        if subject_parallel_workers == 1:
            for payload in worker_payloads:
                logger.info(
                    "Fitting multisubject baseline RL subject_id=%s (subject_index=%d) sequentially",
                    payload["subject_id"],
                    int(payload["subject_index"]),
                )
                per_subject_results.append(
                    _fit_multisubject_subject_worker(payload)
                )
        else:
            logger.info(
                "Fitting %d multisubject baseline RL subjects in parallel with %d subject workers; "
                "effective DE workers per subject=%d",
                len(worker_payloads),
                subject_parallel_workers,
                int(effective_subject_de_kwargs.get("workers", 1)),
            )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=subject_parallel_workers
            ) as executor:
                future_to_subject = {
                    executor.submit(_fit_multisubject_subject_worker, payload): (
                        payload["subject_id"],
                        int(payload["subject_index"]),
                    )
                    for payload in worker_payloads
                }
                for future in concurrent.futures.as_completed(future_to_subject):
                    subject_id, subject_index = future_to_subject[future]
                    result = future.result()
                    logger.info(
                        "Completed multisubject baseline RL fit for subject_id=%s "
                        "(subject_index=%d)",
                        subject_id,
                        subject_index,
                    )
                    per_subject_results.append(result)

        per_subject_results.sort(key=lambda item: int(item["subject_index"]))
        for fit_summary in per_subject_results:
            subject_id = fit_summary["subject_id"]
            subject_index = int(fit_summary["subject_index"])

            if parameter_columns is None:
                parameter_columns = list(fit_summary["fitted_params"].keys())

            subject_summary = {
                "agent_class": self.agent_class,
                "agent_kwargs": self.agent_kwargs,
                "subject_id": normalize_subject_id(subject_id),
                "subject_index": subject_index,
                "curriculum_name": str(fit_summary["curriculum_name"]),
                "num_train_sessions": int(len(fit_summary["train_session_ids"])),
                "num_eval_sessions": int(len(fit_summary["eval_session_ids"])),
                "num_train_trials": int(fit_summary["train_total_trials"]),
                "num_eval_trials": int(fit_summary["eval_total_trials"]),
                "train_session_ids": [
                    _normalize_identifier(session_id)
                    for session_id in fit_summary["train_session_ids"]
                ],
                "eval_session_ids": [
                    _normalize_identifier(session_id)
                    for session_id in fit_summary["eval_session_ids"]
                ],
                "fitted_params": fit_summary["fitted_params"],
                "train_likelihood": float(fit_summary["train_likelihood"]),
                "eval_likelihood": float(fit_summary["eval_likelihood"]),
                "train_total_log_likelihood": float(fit_summary["train_total_log_likelihood"]),
                "train_total_trials": int(fit_summary["train_total_trials"]),
                "eval_total_log_likelihood": float(fit_summary["eval_total_log_likelihood"]),
                "eval_total_trials": int(fit_summary["eval_total_trials"]),
                "log_likelihood_train": float(fit_summary["log_likelihood_train"]),
                "LPT_train": float(fit_summary["LPT_train"]),
                "AIC": float(fit_summary["AIC"]),
                "BIC": float(fit_summary["BIC"]),
                "n_free_params": int(fit_summary["n_free_params"]),
            }

            subject_row = {
                key: value
                for key, value in subject_summary.items()
                if key not in {"fitted_params", "train_session_ids", "eval_session_ids"}
            }
            for param_name, param_value in fit_summary["fitted_params"].items():
                subject_row[param_name] = float(param_value)
            per_subject_rows.append(subject_row)

            subject_dir = subject_output_dir / _safe_filename_component(subject_id)
            subject_dir.mkdir(parents=True, exist_ok=True)
            subject_summary_path = subject_dir / "fit_summary.json"
            with subject_summary_path.open("w") as f:
                json.dump(subject_summary, f, indent=2, default=_json_default)

        assert parameter_columns is not None
        subject_metrics_df = (
            pd.DataFrame(per_subject_rows).sort_values("subject_index").reset_index(drop=True)
        )
        subject_metrics_csv_path = self.output_dir / "subject_fit_metrics.csv"
        subject_metrics_pickle_path = self.output_dir / "subject_fit_metrics.pkl"
        subject_metrics_df.to_csv(subject_metrics_csv_path, index=False)
        subject_metrics_df.to_pickle(subject_metrics_pickle_path)

        subject_index_map_path = save_subject_index_map(
            self.output_dir / "subject_index_map.json",
            subject_id_to_index=subject_id_to_index,
            index_to_subject_id=index_to_subject_id,
        )

        mean_subject_train_likelihood = float(subject_metrics_df["train_likelihood"].mean())
        mean_subject_eval_likelihood = float(subject_metrics_df["eval_likelihood"].mean())
        pooled_train_trial_likelihood = _normalized_likelihood_from_log_stats(
            float(subject_metrics_df["train_total_log_likelihood"].sum()),
            int(subject_metrics_df["train_total_trials"].sum()),
        )
        pooled_eval_trial_likelihood = _normalized_likelihood_from_log_stats(
            float(subject_metrics_df["eval_total_log_likelihood"].sum()),
            int(subject_metrics_df["eval_total_trials"].sum()),
        )

        parameter_space_path = None
        parameter_space_fig = self._plot_subject_parameter_state_space(
            subject_metrics_df,
            parameter_columns=parameter_columns,
        )
        if parameter_space_fig is not None:
            parameter_space_path = self.output_dir / "subject_parameter_state_space.png"
            parameter_space_fig.savefig(parameter_space_path, dpi=150, bbox_inches="tight")
            plt.close(parameter_space_fig)

        likelihood_scatter_fig = self._plot_subject_likelihood_scatter(
            subject_metrics_df,
            mean_subject_train_likelihood=mean_subject_train_likelihood,
            mean_subject_eval_likelihood=mean_subject_eval_likelihood,
            pooled_train_trial_likelihood=pooled_train_trial_likelihood,
            pooled_eval_trial_likelihood=pooled_eval_trial_likelihood,
        )
        likelihood_scatter_path = self.output_dir / "subject_likelihood_scatter.png"
        likelihood_scatter_fig.savefig(likelihood_scatter_path, dpi=150, bbox_inches="tight")
        plt.close(likelihood_scatter_fig)

        elapsed_time = time.time() - start_time
        output: Dict[str, Any] = {
            "multisubject": True,
            "fit_strategy": "per_subject",
            "agent_class": self.agent_class,
            "agent_kwargs": self.agent_kwargs,
            "num_subjects": int(len(subject_metrics_df)),
            "num_trials": metadata.get("num_trials"),
            "num_sessions": metadata.get("num_sessions"),
            "mean_subject_train_likelihood": mean_subject_train_likelihood,
            "mean_subject_eval_likelihood": mean_subject_eval_likelihood,
            "pooled_train_trial_likelihood": pooled_train_trial_likelihood,
            "pooled_eval_trial_likelihood": pooled_eval_trial_likelihood,
            "train_likelihood": pooled_train_trial_likelihood,
            "eval_likelihood": pooled_eval_trial_likelihood,
            "elapsed_seconds": float(elapsed_time),
            "multisubject_subject_workers": int(subject_parallel_workers),
            "effective_de_workers_per_subject": int(
                effective_subject_de_kwargs.get("workers", 1)
            ),
            "parameter_columns": parameter_columns,
            "subject_artifacts": {
                "subject_index_map": str(subject_index_map_path),
                "subject_fit_metrics_csv": str(subject_metrics_csv_path),
                "subject_fit_metrics_pickle": str(subject_metrics_pickle_path),
            },
            "subject_parameter_state_space_path": (
                str(parameter_space_path) if parameter_space_path is not None else None
            ),
            "subject_likelihood_scatter_path": str(likelihood_scatter_path),
        }

        output_path = save_baseline_rl_output(self.output_dir, output, indent=2)
        logger.info("Saved multisubject baseline RL output to %s", output_path)

        if wandb_run is not None:
            wandb_run.summary["mean_subject_train_likelihood"] = mean_subject_train_likelihood
            wandb_run.summary["mean_subject_eval_likelihood"] = mean_subject_eval_likelihood
            wandb_run.summary["pooled_train_trial_likelihood"] = pooled_train_trial_likelihood
            wandb_run.summary["pooled_eval_trial_likelihood"] = pooled_eval_trial_likelihood
            wandb_run.summary["train_likelihood"] = pooled_train_trial_likelihood
            wandb_run.summary["eval_likelihood"] = pooled_eval_trial_likelihood
            wandb_run.summary["num_subjects"] = int(len(subject_metrics_df))
            wandb_run.summary["agent_class"] = self.agent_class
            wandb_run.summary["elapsed_seconds"] = float(elapsed_time)
            wandb_run.summary["multisubject_subject_workers"] = int(subject_parallel_workers)
            wandb_run.summary["effective_de_workers_per_subject"] = int(
                effective_subject_de_kwargs.get("workers", 1)
            )
            wandb_run.log(
                {
                    "subject_fit_metrics": wandb.Table(dataframe=subject_metrics_df),
                }
            )
            media_payload: dict[str, Any] = {
                "fig/subject_likelihood_scatter": wandb.Image(str(likelihood_scatter_path)),
            }
            if parameter_space_path is not None:
                media_payload["fig/subject_parameter_state_space"] = wandb.Image(
                    str(parameter_space_path)
                )
            wandb_run.log(media_payload)

            artifact_name = f"baseline-rl-output-{getattr(wandb_run, 'id', None) or 'latest'}"
            artifact = wandb.Artifact(artifact_name, type="training-output")
            artifact.add_dir(str(self.output_dir))
            wandb_run.log_artifact(artifact)

        logger.info(
            "Multisubject baseline RL fitting complete. "
            "Mean subject train likelihood: %.4f, mean subject eval likelihood: %.4f, "
            "pooled train likelihood: %.4f, pooled eval likelihood: %.4f",
            mean_subject_train_likelihood,
            mean_subject_eval_likelihood,
            pooled_train_trial_likelihood,
            pooled_eval_trial_likelihood,
        )
        return output

    def _fit_single_subject(
        self,
        *,
        bundle: DatasetBundle,
        metadata: dict[str, Any],
        wandb_run: Any | None,
        start_time: float,
    ) -> Dict[str, Any]:
        # --- Extract session data ---
        train_choices, train_rewards, eval_choices, eval_rewards = (
            self._extract_session_data(bundle)
        )

        n_train_sessions = len(train_choices)
        n_eval_sessions = len(eval_choices)
        n_train_trials = sum(len(c) for c in train_choices)
        n_eval_trials = sum(len(c) for c in eval_choices)

        logger.info(
            f"Training on {n_train_sessions} sessions ({n_train_trials} trials), "
            f"evaluating on {n_eval_sessions} sessions ({n_eval_trials} trials)"
        )

        agent_class_obj = self._resolve_agent_class()

        # Create agent for fitting
        agent = agent_class_obj(
            **self.agent_kwargs,
            seed=self.seed,
        )

        logger.info(f"Fitting {self.agent_class} with kwargs: {self.agent_kwargs}")
        logger.info(f"Agent has {len(agent.params_list_free)} free parameters")

        # Fit on training sessions
        fitting_result, _ = agent.fit(
            fit_choice_history=train_choices,
            fit_reward_history=train_rewards,
            fit_bounds_override=self.fit_bounds_override,
            clamp_params=self.clamp_params,
            DE_kwargs=self.DE_kwargs,
        )

        fitted_params = {
            param_name: float(param_value)
            for param_name, param_value in fitting_result.params.items()
        }
        logger.info(f"Fitted parameters: {fitted_params}")

        # --- Evaluate on held-out sessions ---
        # Create a fresh agent with fitted parameters
        eval_agent = agent_class_obj(**self.agent_kwargs, seed=self.seed)
        eval_agent.set_params(**fitted_params)
        agent.set_params(**fitted_params)

        # Run closed-loop simulation on evaluation sessions
        eval_choice_prob_sessions = eval_agent.perform_closed_loop_multi_session(
            eval_choices, eval_rewards
        )
        eval_choice_prob_sessions = [np.asarray(arr) for arr in eval_choice_prob_sessions]

        eval_likelihood = self._compute_normalized_likelihood(
            eval_choices, eval_choice_prob_sessions
        )

        train_choice_prob_sessions = agent.perform_closed_loop_multi_session(
            train_choices, train_rewards
        )
        train_choice_prob_sessions = [np.asarray(arr) for arr in train_choice_prob_sessions]
        train_likelihood = self._compute_normalized_likelihood(
            train_choices, train_choice_prob_sessions
        )

        train_q_histories = _extract_q_histories(agent, train_choice_prob_sessions)
        train_plot_choice_prob_sessions: List[np.ndarray] | None = None
        if train_q_histories is None:
            logger.warning(
                "Could not find explicit training Q-value histories; plotting model choice probabilities directly."
            )
            train_plot_choice_prob_sessions = train_choice_prob_sessions

        eval_q_histories = _extract_q_histories(eval_agent, eval_choice_prob_sessions)
        eval_plot_choice_prob_sessions: List[np.ndarray] | None = None
        if eval_q_histories is None:
            logger.warning(
                "Could not find explicit evaluation Q-value histories; plotting model choice probabilities directly."
            )
            eval_plot_choice_prob_sessions = eval_choice_prob_sessions

        output: Dict[str, Any] = {
            "multisubject": False,
            "fit_strategy": "single_subject",
            "agent_class": self.agent_class,
            "agent_kwargs": self.agent_kwargs,
            "fitted_params": fitted_params,
            "n_free_params": int(fitting_result.k_model),
            "num_train_sessions": n_train_sessions,
            "num_eval_sessions": n_eval_sessions,
            "num_train_trials": n_train_trials,
            "num_eval_trials": n_eval_trials,
            "mean_subject_train_likelihood": float(train_likelihood),
            "mean_subject_eval_likelihood": float(eval_likelihood),
            "pooled_train_trial_likelihood": float(train_likelihood),
            "pooled_eval_trial_likelihood": float(eval_likelihood),
            "eval_likelihood": float(eval_likelihood),
            "train_likelihood": float(train_likelihood),
            "log_likelihood_train": float(fitting_result.log_likelihood),
            "LPT_train": float(fitting_result.LPT),
            "AIC": float(fitting_result.AIC),
            "BIC": float(fitting_result.BIC),
        }

        gt_likelihood = metadata.get("avg_eval_likelihood_groundtruth")
        if gt_likelihood is not None:
            output["groundtruth_likelihood"] = float(gt_likelihood)
            output["likelihood_relative_to_groundtruth"] = (
                float(eval_likelihood) / float(gt_likelihood)
            )

        output["num_trials"] = metadata.get("num_trials")
        output["num_sessions"] = metadata.get("num_sessions")

        elapsed_time = time.time() - start_time
        output["elapsed_seconds"] = float(elapsed_time)

        extras = bundle.extras or {}
        session_details = extras.get("session_details")
        eval_every_n = metadata.get("eval_every_n", 2)
        param_recovery_fig = None

        if session_details is not None:
            plot_path = self.output_dir / "parameter_recovery.png"
            param_recovery_fig = self._plot_parameter_recovery(
                session_details=session_details,
                fitted_params=fitted_params,
                eval_every_n=eval_every_n,
                save_path=plot_path,
            )
            output["parameter_recovery_plot_path"] = str(plot_path)

        train_examples_summary: dict[str, Any] = {
            "split": "train",
            "plotting_failed": True,
            "error": "not run",
            "example_sessions": [],
            "plots": {"q_values_over_trials_examples": []},
        }
        eval_examples_summary: dict[str, Any] = {
            "split": "eval",
            "plotting_failed": True,
            "error": "not run",
            "example_sessions": [],
            "plots": {"q_values_over_trials_examples": []},
        }

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

        train_session_ids = [f"train_session_{i}" for i in range(n_train_sessions)]
        eval_session_ids = [f"eval_session_{i}" for i in range(n_eval_sessions)]
        train_session_subject_ids = ["unknown"] * n_train_sessions
        eval_session_subject_ids = ["unknown"] * n_eval_sessions

        if (
            bundle.raw is not None
            and hasattr(bundle.raw, "columns")
            and "ses_idx" in bundle.raw.columns
        ):
            session_order = list(dict.fromkeys(bundle.raw["ses_idx"].tolist()))
            n_total_sessions = len(session_order)
            expected_total = n_train_sessions + n_eval_sessions
            if n_total_sessions == expected_total:
                eval_every_n = int(metadata.get("eval_every_n", 2))
                if eval_every_n <= 0:
                    raise ValueError(f"Invalid eval_every_n in metadata: {eval_every_n}")

                eval_indices = np.arange(eval_every_n - 1, n_total_sessions, eval_every_n)
                eval_index_set = set(int(i) for i in eval_indices.tolist())
                train_indices = [
                    idx for idx in range(n_total_sessions) if idx not in eval_index_set
                ]

                if len(train_indices) == n_train_sessions and len(eval_indices) == n_eval_sessions:
                    train_session_ids = [session_order[idx] for idx in train_indices]
                    eval_session_ids = [session_order[int(idx)] for idx in eval_indices.tolist()]

                    if "subject_id" in bundle.raw.columns:
                        session_subject_map: dict[Any, Any] = {}
                        session_lookup = (
                            bundle.raw[["ses_idx", "subject_id"]]
                            .drop_duplicates(subset=["ses_idx"])
                            .set_index("ses_idx")["subject_id"]
                            .to_dict()
                        )
                        for session_id in session_order:
                            session_subject_map[session_id] = session_lookup.get(
                                session_id, "unknown"
                            )
                        train_session_subject_ids = [
                            session_subject_map.get(session_id, "unknown")
                            for session_id in train_session_ids
                        ]
                        eval_session_subject_ids = [
                            session_subject_map.get(session_id, "unknown")
                            for session_id in eval_session_ids
                        ]
                else:
                    logger.warning(
                        "Could not align train/eval split indices to extracted session lists. "
                        "Using fallback synthetic session IDs."
                    )
            else:
                logger.warning(
                    "Raw dataframe sessions (%d) do not match split sessions (%d). "
                    "Using fallback synthetic session IDs.",
                    n_total_sessions,
                    expected_total,
                )

        try:
            train_examples_summary = self._plot_q_value_examples_for_split(
                split_name="train",
                choice_sessions=train_choices,
                reward_sessions=train_rewards,
                q_histories=train_q_histories,
                choice_prob_sessions=train_plot_choice_prob_sessions,
                session_ids=train_session_ids,
                session_subject_ids=train_session_subject_ids,
                sessions_per_subject=train_sessions_per_subject,
                output_dir=self.output_dir,
            )
        except Exception as exc:
            logger.exception("Failed to generate train Q/prob example plots")
            train_examples_summary = {
                "split": "train",
                "plotting_failed": True,
                "error": str(exc),
                "example_sessions": [],
                "plots": {"q_values_over_trials_examples": []},
            }

        try:
            eval_examples_summary = self._plot_q_value_examples_for_split(
                split_name="eval",
                choice_sessions=eval_choices,
                reward_sessions=eval_rewards,
                q_histories=eval_q_histories,
                choice_prob_sessions=eval_plot_choice_prob_sessions,
                session_ids=eval_session_ids,
                session_subject_ids=eval_session_subject_ids,
                sessions_per_subject=eval_sessions_per_subject,
                output_dir=self.output_dir,
            )
        except Exception as exc:
            logger.exception("Failed to generate eval Q/prob example plots")
            eval_examples_summary = {
                "split": "eval",
                "plotting_failed": True,
                "error": str(exc),
                "example_sessions": [],
                "plots": {"q_values_over_trials_examples": []},
            }

        output["train_q_value_examples"] = train_examples_summary
        output["eval_q_value_examples"] = eval_examples_summary

        train_plot_paths = train_examples_summary.get("plots", {}).get(
            "q_values_over_trials_examples", []
        )
        eval_plot_paths = eval_examples_summary.get("plots", {}).get(
            "q_values_over_trials_examples", []
        )
        if train_plot_paths:
            output["train_choice_reward_fitted_prob_plot_path"] = str(train_plot_paths[0])
        if eval_plot_paths:
            output["eval_choice_reward_fitted_prob_plot_path"] = str(eval_plot_paths[0])

        if wandb_run is not None:
            wandb_run.summary["mean_subject_train_likelihood"] = float(train_likelihood)
            wandb_run.summary["mean_subject_eval_likelihood"] = float(eval_likelihood)
            wandb_run.summary["pooled_train_trial_likelihood"] = float(train_likelihood)
            wandb_run.summary["pooled_eval_trial_likelihood"] = float(eval_likelihood)
            wandb_run.summary["train_likelihood"] = float(train_likelihood)
            wandb_run.summary["eval_likelihood"] = float(eval_likelihood)
            wandb_run.summary["agent_class"] = self.agent_class
            wandb_run.summary["n_free_params"] = int(fitting_result.k_model)
            wandb_run.summary["elapsed_seconds"] = float(elapsed_time)

            if gt_likelihood is not None:
                wandb_run.summary["groundtruth_likelihood"] = float(gt_likelihood)
                wandb_run.summary["likelihood_relative_to_groundtruth"] = (
                    float(eval_likelihood) / float(gt_likelihood)
                )

            for param_name, param_value in fitted_params.items():
                wandb_run.summary[f"param/{param_name}"] = float(param_value)

            if param_recovery_fig is not None:
                wandb_run.log({"parameter_recovery": wandb.Image(param_recovery_fig)})
                logger.info("Logged parameter recovery plot to W&B")

            if train_plot_paths:
                wandb_run.log(
                    {
                        "train/q_values_over_trials_examples": [
                            wandb.Image(path) for path in train_plot_paths
                        ]
                    }
                )

            if eval_plot_paths:
                wandb_run.log(
                    {
                        "eval/q_values_over_trials_examples": [
                            wandb.Image(path) for path in eval_plot_paths
                        ]
                    }
                )

            artifact_name = f"baseline-rl-output-{getattr(wandb_run, 'id', None) or 'latest'}"
            artifact = wandb.Artifact(artifact_name, type="training-output")
            artifact.add_dir(str(self.output_dir))
            wandb_run.log_artifact(artifact)

        if param_recovery_fig is not None:
            plt.close(param_recovery_fig)

        output_path = save_baseline_rl_output(self.output_dir, output, indent=4)
        logger.info("Saved output to %s", output_path)

        logger.info(
            f"Baseline RL fitting complete. "
            f"Eval likelihood: {eval_likelihood:.4f}, "
            f"Train likelihood: {train_likelihood:.4f}"
        )

        return output

    def fit(
        self,
        bundle: DatasetBundle,
        loggers: dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Fit the RL model and compute likelihood on evaluation set.

        Args:
            bundle: DatasetBundle containing training/evaluation data.
            loggers: Optional dict with logger instances (e.g., {"wandb": wandb_run}).

        Returns:
            Dictionary with fitted parameters and likelihood metrics.
        """
        start_time = time.time()
        metadata = dict(bundle.metadata)

        wandb_run = None
        if loggers and "wandb" in loggers:
            wandb_run = loggers["wandb"]
        if self._is_multisubject_mode(metadata):
            return self._fit_multisubject(
                bundle=bundle,
                metadata=metadata,
                wandb_run=wandb_run,
                start_time=start_time,
            )
        return self._fit_single_subject(
            bundle=bundle,
            metadata=metadata,
            wandb_run=wandb_run,
            start_time=start_time,
        )

    def _plot_choice_reward_and_fitted_prob(
        self,
        choice_sessions: List[np.ndarray],
        reward_sessions: List[np.ndarray],
        choice_prob_sessions: List[np.ndarray],
        gt_params_per_session: list[dict[str, Any]] | None,
        param_names_for_gt: list[str] | None,
        label: str,
        save_path: Path,
    ) -> Figure:
        """Plot concatenated choice/reward with fitted choice probability overlay.

        This uses the existing plotting function from aind_dynamic_foraging_basic_analysis
        (the same one used by aind-dynamic-foraging-models), and adds session boundaries.
        """

        if len(choice_sessions) == 0:
            raise ValueError(f"No sessions provided for {label} plot")

        if len(choice_sessions) != len(reward_sessions) or len(choice_sessions) != len(
            choice_prob_sessions
        ):
            raise ValueError(
                f"{label} sessions length mismatch: choices={len(choice_sessions)}, "
                f"rewards={len(reward_sessions)}, choice_prob={len(choice_prob_sessions)}"
            )

        n_sessions = len(choice_sessions)
        n_cols = min(3, n_sessions)
        n_rows = (n_sessions + n_cols - 1) // n_cols
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5.5 * n_cols, 3.2 * n_rows),
            squeeze=False,
        )
        axes_flat = axes.flatten()

        # Add ground-truth parameter annotations at each session start if provided.
        def _empty_param_text(_: int) -> str:
            return ""

        _build_param_text = _empty_param_text

        if gt_params_per_session is not None:
            if len(gt_params_per_session) != len(choice_sessions):
                logger.warning(
                    f"Skipping GT param annotations for {label}: "
                    f"expected {len(choice_sessions)} sessions, got {len(gt_params_per_session)}"
                )
            else:
                # Match aind-dynamic-foraging-models parameter rendering:
                # use ParamsSymbols (latex) when available and sort by ParamsSymbols order.
                names_in = (
                    list(param_names_for_gt)
                    if param_names_for_gt is not None
                    else list(gt_params_per_session[0].keys())
                )
                default_order = list(ParamsSymbols.__members__.keys())

                def _sort_key(n: str) -> tuple[int, int | str]:
                    return (0, default_order.index(n)) if n in default_order else (1, n)

                names = sorted(names_in, key=_sort_key)

                def _render_name(n: str) -> str:
                    # Match aind-dynamic-foraging-models get_params_str(): latex symbol if known.
                    try:
                        return ParamsSymbols[n].value
                    except KeyError:
                        return n

                def build_param_text(sess_idx: int) -> str:
                    params = gt_params_per_session[sess_idx] or {}
                    parts: list[str] = []
                    for k in names:
                        if k in params:
                            try:
                                v = float(params[k])
                            except Exception:
                                continue
                            parts.append(f"{_render_name(k)} = {v:.3f}")
                    return ", ".join(parts)

                _build_param_text = build_param_text

        for sess_idx, ax in enumerate(axes_flat):
            if sess_idx >= n_sessions:
                ax.set_visible(False)
                continue

            choice = choice_sessions[sess_idx]
            reward = reward_sessions[sess_idx]
            choice_prob = choice_prob_sessions[sess_idx]
            denom = choice_prob.sum(axis=0)
            denom = np.where(denom == 0, 1.0, denom)
            p_right = choice_prob[1] / denom

            p_reward_dummy = np.full((2, len(choice)), np.nan)
            _, session_axes = plot_foraging_session(
                choice_history=choice,
                reward_history=reward,
                p_reward=p_reward_dummy,
                fitted_data=p_right,
                plot_list=["choice", "finished"],
                ax=ax,
            )

            ax_choice = session_axes[0]
            ax_reward = session_axes[1]
            choice_legend = ax_choice.get_legend()
            if choice_legend is not None:
                choice_legend.remove()
            reward_legend = ax_reward.get_legend()
            if reward_legend is not None:
                reward_legend.remove()
            ax_reward.set_visible(False)
            ax_reward.axis("off")

            if gt_params_per_session is not None:
                text = _build_param_text(sess_idx)
                if text:
                    ax_choice.text(
                        0.02,
                        1.12,
                        f"s{sess_idx}: {text}",
                        transform=ax_choice.transAxes,
                        rotation=0,
                        ha="left",
                        va="bottom",
                        fontsize=7,
                        color="0.25",
                        clip_on=False,
                    )

        fig.suptitle(f"{label}: choice/reward with fitted p(R)", fontsize=10)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved {label} overview plot to {save_path}")
        return fig

    def _plot_q_value_examples_for_split(
        self,
        *,
        split_name: str,
        choice_sessions: List[np.ndarray],
        reward_sessions: List[np.ndarray],
        q_histories: List[np.ndarray] | None,
        choice_prob_sessions: List[np.ndarray] | None,
        session_ids: List[Any],
        session_subject_ids: List[Any],
        sessions_per_subject: int,
        output_dir: Path,
    ) -> Dict[str, Any]:
        if sessions_per_subject < 0:
            raise ValueError(f"{split_name}_example_sessions_per_subject must be >= 0")
        if q_histories is None and choice_prob_sessions is None:
            raise ValueError(
                f"{split_name} example plotting requires q_histories or choice_prob_sessions."
            )
        history_sessions = q_histories if q_histories is not None else choice_prob_sessions
        assert history_sessions is not None
        if not (
            len(choice_sessions)
            == len(reward_sessions)
            == len(history_sessions)
            == len(session_ids)
            == len(session_subject_ids)
        ):
            raise ValueError(
                f"{split_name} split length mismatch: choices={len(choice_sessions)}, "
                f"rewards={len(reward_sessions)}, histories={len(history_sessions)}, "
                f"session_ids={len(session_ids)}, subject_ids={len(session_subject_ids)}"
            )

        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        examples: list[dict[str, Any]] = []
        q_plot_paths: list[str] = []

        if sessions_per_subject > 0:
            sessions_by_subject: dict[Any, list[int]] = {}
            for idx, subject_id in enumerate(session_subject_ids):
                sessions_by_subject.setdefault(subject_id, []).append(idx)

            for subject_id, indices in sessions_by_subject.items():
                for idx in indices[:sessions_per_subject]:
                    choices = choice_sessions[idx]
                    rewards = reward_sessions[idx]
                    if q_histories is not None:
                        q_session = _align_q_session(q_histories[idx], len(choices))
                        fig = _plot_q_values_for_session(
                            choices=choices,
                            rewards=rewards,
                            q_values=q_session,
                        )
                    else:
                        assert choice_prob_sessions is not None
                        choice_prob_session = _align_choice_prob_session(
                            choice_prob_sessions[idx],
                            len(choices),
                        )
                        fig = _plot_q_values_for_session(
                            choices=choices,
                            rewards=rewards,
                            choice_probabilities=choice_prob_session,
                        )
                    session_id = session_ids[idx]
                    fig.suptitle(f"Session {_normalize_identifier(session_id)}", fontsize=14)
                    fig.subplots_adjust(top=0.93)

                    out_path = (
                        split_dir
                        / (
                            f"q_values_over_trials_subject_{_safe_filename_component(subject_id)}"
                            f"_session_{_safe_filename_component(session_id)}.png"
                        )
                    )
                    fig.savefig(out_path)
                    plt.close(fig)

                    q_plot_paths.append(str(out_path))
                    examples.append(
                        {
                            "subject_id": _normalize_identifier(subject_id),
                            "session_id": _normalize_identifier(session_id),
                            "q_values_over_trials": str(out_path),
                        }
                    )

        summary: Dict[str, Any] = {
            "split": split_name,
            "num_sessions": int(len(choice_sessions)),
            "example_sessions_per_subject": int(sessions_per_subject),
            "example_sessions": examples,
            "plots": {
                "q_values_over_trials_examples": q_plot_paths,
            },
        }

        summary_path = split_dir / f"{split_name}_baseline_rl_examples_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)

        return summary

    def _extract_session_data(
        self, bundle: DatasetBundle
    ) -> tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Extract per-session choice and reward arrays for train/eval splits.

        Returns:
            Tuple of (train_choices, train_rewards, eval_choices, eval_rewards)
            where each is a list of 1D numpy arrays (one per session).
        """
        if bundle.train_set is None or bundle.eval_set is None:
            raise ValueError("Dataset bundle must include train and eval sets.")

        train_choices, train_rewards = self._extract_from_dataset(
            bundle.train_set, label="train"
        )
        eval_choices, eval_rewards = self._extract_from_dataset(
            bundle.eval_set, label="eval"
        )
        return train_choices, train_rewards, eval_choices, eval_rewards

    def _extract_from_dataset(
        self, dataset: Any, label: str
    ) -> tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract per-session choices and rewards from a DatasetRNN split."""
        xs, ys = dataset.get_all()
        x_names = list(dataset.x_names)
        if ys.ndim != 3 or ys.shape[2] != 1:
            raise ValueError(f"{label} ys has unexpected shape: {ys.shape}")

        reward_names = ("prev reward", "rewarded", "earned_reward", "reward")
        reward_idx = None
        for name in reward_names:
            if name in x_names:
                reward_idx = x_names.index(name)
                break
        if reward_idx is None:
            lower_names = [n.lower() for n in x_names]
            for name in reward_names:
                if name in lower_names:
                    reward_idx = lower_names.index(name)
                    break
        if reward_idx is None:
            raise ValueError(
                f"{label} dataset is missing a reward feature in x_names."
            )

        choices_sessions: List[np.ndarray] = []
        rewards_sessions: List[np.ndarray] = []
        n_sessions = ys.shape[1]
        for sess_idx in range(n_sessions):
            choices = ys[:, sess_idx, 0]
            valid = choices >= 0
            choices = choices[valid].astype(int)  # drop padded trials
            choices_sessions.append(choices)

            rewards_prev = xs[:, sess_idx, reward_idx]
            n_trials = len(choices)
            rewards = np.zeros(n_trials, dtype=float)
            if n_trials > 1:
                rewards[:-1] = rewards_prev[1:n_trials]  # shift prev reward to current trial
            rewards_sessions.append(rewards)

        return choices_sessions, rewards_sessions

    def _compute_normalized_likelihood(
        self,
        choice_sessions: List[np.ndarray],
        choice_prob_sessions: List[np.ndarray],
    ) -> float:
        """Compute normalized likelihood (geometric mean) across sessions.

        Uses the same computation as rnn_utils.normalized_likelihood for
        consistency with disRNN trainer.

        Args:
            choice_sessions: List of choice arrays, one per session.
            choice_prob_sessions: List of choice probability arrays (n_actions, n_trials).

        Returns:
            Normalized likelihood (geometric mean of per-trial probabilities).
        """
        stats = _compute_likelihood_stats(choice_sessions, choice_prob_sessions)
        return float(stats["normalized_likelihood"])

    def _plot_parameter_recovery(
        self,
        session_details: List[Dict[str, Any]],
        fitted_params: Dict[str, Any],
        eval_every_n: int,
        save_path: Path,
    ) -> Figure:
        """Plot comparison of true parameter distributions vs recovered parameters.

        Creates a figure with one subplot per parameter, showing:
        - Histogram of true parameter values across all sessions
        - Vertical line for the recovered (fitted) parameter value
        - Separate colors for train vs eval sessions

        Args:
            session_details: List of session detail dicts containing 'agent_params'.
            fitted_params: Dictionary of fitted parameter values.
            eval_every_n: Every n-th session is used for evaluation.
            save_path: Path to save the figure.

        Returns:
            matplotlib Figure object.
        """
        # Extract true parameters from each session
        n_sessions = len(session_details)
        param_names = list(fitted_params.keys())
        n_params = len(param_names)

        # Collect true parameter values per session
        true_params_per_session: Dict[str, List[float]] = {p: [] for p in param_names}
        session_is_eval: List[bool] = []

        for i, sess in enumerate(session_details):
            is_eval = (i % eval_every_n) == (eval_every_n - 1)
            session_is_eval.append(is_eval)
            agent_params = sess.get("agent_params", {})
            for param in param_names:
                if param in agent_params:
                    true_params_per_session[param].append(float(agent_params[param]))
                else:
                    true_params_per_session[param].append(np.nan)

        # Create figure with subplots
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if n_params == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, param in enumerate(param_names):
            ax = axes[idx]
            true_vals = np.array(true_params_per_session[param])
            fitted_val = float(fitted_params[param])

            # Separate train and eval values
            train_vals = true_vals[~np.array(session_is_eval)]
            eval_vals = true_vals[np.array(session_is_eval)]

            # Remove NaN values
            train_vals = train_vals[~np.isnan(train_vals)]
            eval_vals = eval_vals[~np.isnan(eval_vals)]
            all_vals = np.concatenate([train_vals, eval_vals]) if len(train_vals) > 0 or len(eval_vals) > 0 else np.array([])

            if len(all_vals) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(param)
                continue

            # Determine if parameter varies across sessions
            param_std = np.std(all_vals)
            param_varies = param_std > 1e-6

            if param_varies:
                # Plot side-by-side histogram of true values (train vs eval)
                bins = min(15, max(5, n_sessions // 2))
                
                # Calculate bin edges based on all values
                val_range = (np.min(all_vals), np.max(all_vals))
                bin_edges = np.linspace(val_range[0], val_range[1], bins + 1)
                bin_width = bin_edges[1] - bin_edges[0]
                
                # Compute histogram counts for train and eval
                train_counts, _ = np.histogram(train_vals, bins=bin_edges) if len(train_vals) > 0 else (np.zeros(bins), None)
                eval_counts, _ = np.histogram(eval_vals, bins=bin_edges) if len(eval_vals) > 0 else (np.zeros(bins), None)
                
                # Plot side-by-side bars
                bar_width = bin_width * 0.4
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                if len(train_vals) > 0:
                    ax.bar(bin_centers - bar_width/2, train_counts, width=bar_width, 
                           color="steelblue", alpha=0.8, edgecolor="white",
                           label=f"Train (n={len(train_vals)})")
                if len(eval_vals) > 0:
                    ax.bar(bin_centers + bar_width/2, eval_counts, width=bar_width,
                           color="coral", alpha=0.8, edgecolor="white",
                           label=f"Eval (n={len(eval_vals)})")

                # Add vertical line for fitted value
                ax.axvline(fitted_val, color="darkgreen", linestyle="--", linewidth=2.5,
                          label=f"Fitted: {fitted_val:.3f}")

                # Add mean of true values
                true_mean = np.mean(all_vals)
                ax.axvline(true_mean, color="purple", linestyle=":", linewidth=2,
                          label=f"True mean: {true_mean:.3f}")

                ax.legend(fontsize=8, loc="upper right")
            else:
                # Parameter is constant across sessions - show as bar comparison
                true_val = all_vals[0]
                x_pos = [0, 1]
                heights = [true_val, fitted_val]
                colors = ["steelblue", "darkgreen"]
                labels = ["True", "Fitted"]

                bars = ax.bar(x_pos, heights, color=colors, alpha=0.7, edgecolor="black")
                ax.set_xticks(x_pos)
                ax.set_xticklabels(labels)

                # Add value labels on bars
                for bar, height in zip(bars, heights):
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                           f"{height:.3f}", ha="center", va="bottom", fontsize=9)

                # Calculate and show error
                error = abs(fitted_val - true_val)
                rel_error = error / abs(true_val) if true_val != 0 else error
                ax.set_ylabel("Value")
                ax.text(0.5, 0.95, f"Error: {error:.4f} ({rel_error:.1%})",
                       ha="center", va="top", transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

            ax.set_title(param, fontsize=10, fontweight="bold")
            ax.set_xlabel("Parameter value")
            if param_varies:
                ax.set_ylabel("Count (sessions)")

        # Hide unused subplots
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle("Parameter Recovery: True Distribution vs Fitted Values",
                    fontsize=12, fontweight="bold", y=1.02)
        plt.tight_layout()

        # Save figure
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved parameter recovery plot to {save_path}")

        return fig
