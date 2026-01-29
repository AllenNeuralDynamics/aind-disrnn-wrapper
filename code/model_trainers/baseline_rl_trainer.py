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

from aind_dynamic_foraging_models import generative_model
from disentangled_rnns.library import rnn_utils

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
        train_likelihood = self._compute_normalized_likelihood(
            train_choices, train_choice_prob_sessions
        )

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
            choices = choices[valid].astype(int)
            choices_sessions.append(choices)

            rewards_prev = xs[:, sess_idx, reward_idx]
            n_trials = len(choices)
            rewards = np.zeros(n_trials, dtype=float)
            if n_trials > 1:
                rewards[:-1] = rewards_prev[1:n_trials]
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
            n_trials = len(choices)
            for t in range(n_trials):
                c = int(choices[t])
                prob = choice_prob[c, t]
                # Clamp probability to avoid log(0)
                prob = np.clip(prob, 1e-10, 1.0 - 1e-10)
                total_log_lik += np.log(prob)
            total_trials += n_trials

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
