from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import wandb
from omegaconf import DictConfig, OmegaConf

from aind_dynamic_foraging_basic_analysis import plot_foraging_session
from aind_dynamic_foraging_models import generative_model
from aind_dynamic_foraging_models.generative_model.params import ParamsSymbols
from disentangled_rnns.library import rnn_utils
from utils.baseline_rl_evaluation import (
    _align_q_session,
    _extract_q_histories,
    _normalize_identifier,
    _plot_q_values_for_session,
    _safe_filename_component,
)

from base.interfaces import ModelTrainer
from base.types import DatasetBundle

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


class BaselineRLTrainer(ModelTrainer):
    """Trainer for baseline RL model comparisons using aind-dynamic-foraging-models.

    This trainer fits RL cognitive models (e.g., Q-learning variants) to behavioral
    data and computes likelihood metrics on held-out evaluation sessions for
    comparison with disRNN models.
    """

    def __init__(
        self,
        agent_class: str,
        agent_kwargs: Mapping[str, Any] | DictConfig = {},
        fit_bounds_override: Mapping[str, Any] | DictConfig = {},
        clamp_params: Mapping[str, Any] | DictConfig = {},
        DE_kwargs: Mapping[str, Any] | DictConfig = {"workers": 1},
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
        self.agent_kwargs = _to_dict(agent_kwargs)
        self.fit_bounds_override = _to_dict(fit_bounds_override)
        self.clamp_params = _to_dict(clamp_params)
        self.DE_kwargs = _to_dict(DE_kwargs)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

        # --- Instantiate and fit the agent ---
        agent_class_obj = getattr(generative_model, self.agent_class, None)
        if agent_class_obj is None:
            raise ValueError(
                f"Agent class '{self.agent_class}' not found in "
                f"aind_dynamic_foraging_models.generative_model"
            )

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

        fitted_params = fitting_result.params
        logger.info(f"Fitted parameters: {fitted_params}")

        # --- Evaluate on held-out sessions ---
        # Create a fresh agent with fitted parameters
        eval_agent = agent_class_obj(**self.agent_kwargs, seed=self.seed)
        eval_agent.set_params(**fitted_params)

        # Run closed-loop simulation on evaluation sessions
        eval_choice_prob_sessions = eval_agent.perform_closed_loop_multi_session(
            eval_choices, eval_rewards
        )
        eval_choice_prob_sessions = [np.asarray(arr) for arr in eval_choice_prob_sessions]

        # Compute normalized likelihood on evaluation set
        # Convert to format expected by rnn_utils.normalized_likelihood
        # Shape: (n_timesteps, n_episodes, 1) for labels
        # Shape: (n_timesteps, n_episodes, n_classes) for logits
        eval_likelihood = self._compute_normalized_likelihood(
            eval_choices, eval_choice_prob_sessions
        )

        # Also compute likelihood on training set for reference
        train_choice_prob_sessions = agent.perform_closed_loop_multi_session(
            train_choices, train_rewards
        )
        train_choice_prob_sessions = [np.asarray(arr) for arr in train_choice_prob_sessions]
        train_likelihood = self._compute_normalized_likelihood(
            train_choices, train_choice_prob_sessions
        )

        train_q_histories = _extract_q_histories(agent, train_choice_prob_sessions)
        if train_q_histories is None:
            logger.warning(
                "Could not find explicit training Q-value histories; using fallback from action values"
            )
            train_q_histories = train_choice_prob_sessions

        eval_q_histories = _extract_q_histories(eval_agent, eval_choice_prob_sessions)
        if eval_q_histories is None:
            logger.warning(
                "Could not find explicit evaluation Q-value histories; using fallback from action values"
            )
            eval_q_histories = eval_choice_prob_sessions

        # --- Build output ---
        output: Dict[str, Any] = {
            "agent_class": self.agent_class,
            "agent_kwargs": self.agent_kwargs,
            "fitted_params": fitted_params,
            "n_free_params": fitting_result.k_model,
            "num_train_sessions": n_train_sessions,
            "num_eval_sessions": n_eval_sessions,
            "num_train_trials": n_train_trials,
            "num_eval_trials": n_eval_trials,
            "likelihood": float(eval_likelihood),
            "likelihood_train": float(train_likelihood),
            "log_likelihood_train": float(fitting_result.log_likelihood),
            "LPT_train": float(fitting_result.LPT),
            "AIC": float(fitting_result.AIC),
            "BIC": float(fitting_result.BIC),
        }

        # Compare to groundtruth likelihood if available
        gt_likelihood = metadata.get("avg_eval_likelihood_groundtruth")
        if gt_likelihood is not None:
            output["groundtruth_likelihood"] = float(gt_likelihood)
            output["likelihood_relative_to_groundtruth"] = (
                float(eval_likelihood) / float(gt_likelihood)
            )

        # Include metadata
        output["num_trials"] = metadata.get("num_trials")
        output["num_sessions"] = metadata.get("num_sessions")

        elapsed_time = time.time() - start_time
        output["elapsed_seconds"] = float(elapsed_time)

        # --- Save outputs ---
        output_path = self.output_dir / "baseline_rl_output.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4, default=str)
        logger.info(f"Saved output to {output_path}")

        # --- Generate parameter recovery plot if session_details available ---
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

        # --- Generate heldout-style Q/probability examples for train/eval sessions ---
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

        # --- Log to wandb ---
        if wandb_run is not None:
            wandb_run.summary["likelihood"] = float(eval_likelihood)
            wandb_run.summary["likelihood_train"] = float(train_likelihood)
            wandb_run.summary["agent_class"] = self.agent_class
            wandb_run.summary["n_free_params"] = fitting_result.k_model
            wandb_run.summary["elapsed_seconds"] = float(elapsed_time)

            if gt_likelihood is not None:
                wandb_run.summary["groundtruth_likelihood"] = float(gt_likelihood)
                wandb_run.summary["likelihood_relative_to_groundtruth"] = (
                    float(eval_likelihood) / float(gt_likelihood)
                )

            # Log fitted parameters individually
            for param_name, param_value in fitted_params.items():
                wandb_run.summary[f"param/{param_name}"] = float(param_value)

            # Log parameter recovery plot to wandb
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

            # Upload output as artifact
            artifact_name = f"baseline-rl-output-{getattr(wandb_run, 'id', None) or 'latest'}"
            artifact = wandb.Artifact(artifact_name, type="training-output")
            artifact.add_dir(str(self.output_dir))
            wandb_run.log_artifact(artifact)

        # Close figure to free memory
        if param_recovery_fig is not None:
            plt.close(param_recovery_fig)

        logger.info(
            f"Baseline RL fitting complete. "
            f"Eval likelihood: {eval_likelihood:.4f}, "
            f"Train likelihood: {train_likelihood:.4f}"
        )

        return output

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
        q_histories: List[np.ndarray],
        session_ids: List[Any],
        session_subject_ids: List[Any],
        sessions_per_subject: int,
        output_dir: Path,
    ) -> Dict[str, Any]:
        if sessions_per_subject < 0:
            raise ValueError(f"{split_name}_example_sessions_per_subject must be >= 0")
        if not (
            len(choice_sessions)
            == len(reward_sessions)
            == len(q_histories)
            == len(session_ids)
            == len(session_subject_ids)
        ):
            raise ValueError(
                f"{split_name} split length mismatch: choices={len(choice_sessions)}, "
                f"rewards={len(reward_sessions)}, q_histories={len(q_histories)}, "
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
                    q_session = _align_q_session(q_histories[idx], len(choices))

                    fig = _plot_q_values_for_session(
                        choices=choices,
                        rewards=rewards,
                        q_values=q_session,
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
        total_log_lik = 0.0
        total_trials = 0

        for choices, choice_prob in zip(choice_sessions, choice_prob_sessions):
            n_trials = min(len(choices), choice_prob.shape[1])
            if n_trials == 0:
                continue
            choices_idx = choices[:n_trials].astype(int)
            valid = choices_idx >= 0  # guard against padded labels
            if not np.any(valid):
                continue
            trial_idx = np.arange(n_trials)[valid]
            probs = choice_prob[choices_idx[valid], trial_idx]  # gather p(choice_t)
            probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
            total_log_lik += float(np.sum(np.log(probs)))
            total_trials += int(np.sum(valid))

        # Normalized likelihood = exp(mean log likelihood)
        if total_trials == 0:
            return 0.0
        return float(np.exp(total_log_lik / total_trials))

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