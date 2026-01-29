from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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

            # Upload output as artifact
            artifact_name = f"baseline-rl-output-{getattr(wandb_run, 'id', None) or 'latest'}"
            artifact = wandb.Artifact(artifact_name, type="training-output")
            artifact.add_dir(str(self.output_dir))
            wandb_run.log_artifact(artifact)

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
        metadata = bundle.metadata
        extras = bundle.extras or {}

        # Try to use session_details from synthetic data loader
        session_details = extras.get("session_details")
        eval_every_n = metadata.get("eval_every_n", 2)

        if session_details is not None:
            # Synthetic data path - session_details has per-session info
            n_sessions = len(session_details)
            train_choices = []
            train_rewards = []
            eval_choices = []
            eval_rewards = []

            for i, sess in enumerate(session_details):
                # Extract choices and rewards from session_details
                # choices are stored as (T, 1, 1), we need (T,)
                choices_arr = np.asarray(sess["choices"]).squeeze()
                # Rewards need to be extracted from raw dataframe
                raw_df = bundle.raw
                sess_df = raw_df[raw_df["ses_idx"] == i].sort_values("trial")
                rewards_arr = sess_df["earned_reward"].values.astype(float)

                # Determine if this is train or eval session
                # eval sessions are at indices 1, 1+eval_every_n, 1+2*eval_every_n, ...
                is_eval = (i % eval_every_n) == (eval_every_n - 1)

                if is_eval:
                    eval_choices.append(choices_arr)
                    eval_rewards.append(rewards_arr)
                else:
                    train_choices.append(choices_arr)
                    train_rewards.append(rewards_arr)

            return train_choices, train_rewards, eval_choices, eval_rewards

        # Fallback: reconstruct from raw dataframe
        raw_df = bundle.raw
        if raw_df is None:
            raise ValueError(
                "Cannot extract session data: no session_details or raw dataframe"
            )

        # Group by session
        session_groups = raw_df.groupby("ses_idx")
        n_sessions = len(session_groups)

        train_choices = []
        train_rewards = []
        eval_choices = []
        eval_rewards = []

        for ses_idx, group in session_groups:
            group = group.sort_values("trial")
            choices_arr = group["animal_response"].values.astype(int)
            rewards_arr = group["earned_reward"].values.astype(float)

            # Determine if this is train or eval session
            is_eval = (ses_idx % eval_every_n) == (eval_every_n - 1)

            if is_eval:
                eval_choices.append(choices_arr)
                eval_rewards.append(rewards_arr)
            else:
                train_choices.append(choices_arr)
                train_rewards.append(rewards_arr)

        return train_choices, train_rewards, eval_choices, eval_rewards

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
