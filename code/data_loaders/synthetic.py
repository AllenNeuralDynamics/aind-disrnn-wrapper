from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from aind_behavior_gym.dynamic_foraging.task import (
    CoupledBlockTask,
    RandomWalkTask,
    UncoupledBlockTask,
)
from aind_dynamic_foraging_models import generative_model

import aind_disrnn_utils.data_loader as dl
from disentangled_rnns.library import rnn_utils

from base.interfaces import DatasetLoader
from base.types import DatasetBundle
import logging

logger = logging.getLogger(__name__)

class SyntheticDatasetLoader(DatasetLoader):
    """Placeholder loader for synthetic experiments."""

    def __init__(self, seed: int | None = None, **settings: object) -> None:
        super().__init__(seed=seed)
        self.settings = settings

    def load(self) -> DatasetBundle:
        raise NotImplementedError(
            "Synthetic dataset loading is not implemented yet; config received:"
            f" {self.settings}"
        )


class SyntheticCognitiveAgents(DatasetLoader):
    """Generate synthetic data from cognitive agents in aind_dynamic_foraging_models."""

    def __init__(
        self,
        task: dict[str, Any],
        agent: dict[str, Any],
        num_trials: int,
        num_sessions: int,
        eval_every_n: int = 2,
        **metadata: object,
    ) -> None:
        # Random seeds are fully configured via the task/agent dictionaries.
        super().__init__()
        self.task_config = copy.deepcopy(task)
        self.agent_config = copy.deepcopy(agent)
        self.num_trials = int(num_trials)
        self.num_sessions = int(num_sessions)
        self.eval_every_n = int(eval_every_n)
        self.metadata_extras = dict(metadata)

    def load(self) -> DatasetBundle:
        # --- Resolve task and agent classes from config ---
        task_lookup = {
            "coupled_block": CoupledBlockTask,
            "uncoupled_block": UncoupledBlockTask,
            "random_walk": RandomWalkTask,
        }

        task_cfg = copy.deepcopy(self.task_config)
        task_type_key = str(task_cfg.pop("type", "")).lower()
        task_class = task_lookup[task_type_key]

        agent_cfg = copy.deepcopy(self.agent_config)
        agent_class_name = str(agent_cfg.get("agent_class", "")).strip()
        agent_class = getattr(generative_model, agent_class_name)

        agent_kwargs = copy.deepcopy(agent_cfg.get("agent_kwargs", {}))
        agent_params = copy.deepcopy(agent_cfg.get("agent_params", {}))
        agent_params_session_var_cfg = copy.deepcopy(agent_cfg.get("agent_params_session_var", {}))

        base_task_seed = task_cfg.pop("seed", None)
        base_agent_seed = agent_cfg.get("seed")
        if base_task_seed is not None:
            base_task_seed = int(base_task_seed)
        if base_agent_seed is not None:
            base_agent_seed = int(base_agent_seed)

        session_frames: list[pd.DataFrame] = []
        session_details: list[dict[str, Any]] = []

        # --- Simulate each session ---
        for session_idx in range(self.num_sessions):
            task_kwargs = copy.deepcopy(task_cfg)
            session_task_seed = None if base_task_seed is None else base_task_seed + session_idx
            if session_task_seed is not None:
                task_kwargs["seed"] = session_task_seed
            task_kwargs.setdefault("num_trials", self.num_trials)
            task_instance = task_class(**task_kwargs)

            # Sample session-specific agent parameters if non-stationary config provided
            session_agent_seed = None if base_agent_seed is None else base_agent_seed + session_idx
            session_agent_params = self._sample_session_agent_params(
                session_idx, session_agent_seed, agent_params, agent_params_session_var_cfg
            )

            forager_kwargs = copy.deepcopy(agent_kwargs)
            if session_agent_seed is not None:
                forager_kwargs["seed"] = session_agent_seed

            forager = agent_class(**forager_kwargs)
            forager.set_params(**session_agent_params)
            logger.info(
                f"Session {session_idx}: Using agent params: {session_agent_params}"
            )

            forager.perform(task_instance)

            session_df = self._session_dataframe(session_idx, forager, task_instance)
            session_frames.append(session_df)
            session_details.append(
                {
                    "session_index": session_idx,
                    "task_seed": session_task_seed,
                    "agent_seed": session_agent_seed,
                    "task_type": task_type_key,
                    "agent_class": agent_class_name,
                    "agent_params": forager.get_params(),
                    "task_kwargs": task_kwargs,
                }
            )

        # --- Assemble dataframe and disrnn dataset ---
        raw_df = pd.concat(session_frames, ignore_index=True).sort_values(["ses_idx", "trial"])  # type: ignore[arg-type]
        raw_df.reset_index(drop=True, inplace=True)

        dataset = dl.create_disrnn_dataset(raw_df, ignore_policy="exclude")

        xs, _ = dataset.get_all()
        n_sessions = xs.shape[1]
        if self.eval_every_n <= 0:
            raise ValueError("eval_every_n must be a positive integer.")
        eval_indices = np.arange(self.eval_every_n - 1, n_sessions, self.eval_every_n)
        if len(eval_indices) == 0:
            raise ValueError(
                f"No evaluation sessions selected. Increase 'num_sessions' ({self.num_sessions}) or reduce 'eval_every_n' ({self.eval_every_n})."
            )
        if len(eval_indices) >= n_sessions:
            raise ValueError(
                "All sessions would be used for evaluation only. Increase 'eval_every_n' or the number of sessions."
            )

        dataset_train, dataset_eval = rnn_utils.split_dataset(dataset, self.eval_every_n)

        # --- Package bundle metadata ---
        metadata: dict[str, Any] = {
            "num_trials": self.num_trials,
            "num_sessions": self.num_sessions,
            "eval_every_n": self.eval_every_n,
            "task": self.task_config,
            "agent": self.agent_config,
            "seeds": {
                "task": base_task_seed,
                "agent": base_agent_seed,
            },
        }
        metadata.update(self.metadata_extras)

        extras = {
            "dataset": dataset,
            "session_details": session_details,
        }

        return DatasetBundle(
            raw=raw_df,
            train_set=dataset_train,
            eval_set=dataset_eval,
            metadata=metadata,
            extras=extras,
        )

    def _sample_session_agent_params(
        self,
        session_idx: int,
        session_seed: int | None,
        base_params: dict[str, Any],
        agent_params_session_var: dict[str, Any],
    ) -> dict[str, Any]:
        """Return a dict of agent params for a session, sampling according to agent_params_session_var spec.

        Supported kinds:
        - type: "uniform" with 'min' and 'max'
        - type: "gaussian" with 'mean' and 'std'

        If a parameter is not present in agent_params_session_var, the base param is used.
        The session_seed (if provided) is combined with session_idx to seed numpy RNG
        deterministically per session.
        """
        rng = np.random.default_rng(None if session_seed is None else int(session_seed))

        session_params: dict[str, Any] = copy.deepcopy(base_params or {})

        for key, spec in (agent_params_session_var or {}).items():
            typ = str(spec.get("type", "")).lower()
            if typ == "uniform":
                lo = float(spec.get("min", 0.0))
                hi = float(spec.get("max", 1.0))
                session_params[key] = float(rng.uniform(lo, hi))
            elif typ in ("gaussian", "normal"):
                mean = float(spec.get("mean", 0.0))
                std = float(spec.get("std", 1.0))
                session_params[key] = float(rng.normal(mean, std))
            else:
                # Unknown spec: skip and fall back to base param if present
                continue

        return session_params

    def _session_dataframe(self, session_idx: int, forager: Any, task: Any) -> pd.DataFrame:
        choice_history = np.asarray(forager.get_choice_history(), dtype=int)
        reward_history = np.asarray(forager.get_reward_history(), dtype=float)

        p_reward = np.asarray(forager.get_p_reward())
        choice_prob = np.asarray(forager.choice_prob)

        n_trials = choice_history.shape[0]
        trials = np.arange(n_trials, dtype=int)

        data: dict[str, Any] = {
            "ses_idx": np.full(n_trials, session_idx, dtype=int),
            "trial": trials,
            "animal_response": choice_history.astype(int),
            "earned_reward": reward_history.astype(float),
            "reward_baiting": bool(getattr(task, "reward_baiting", False)),
        }

        data["p_reward_left"] = p_reward[0, :]
        data["p_reward_right"] = p_reward[1, :]
        data["choice_prob_left"] = choice_prob[0, :]
        data["choice_prob_right"] = choice_prob[1, :]

        return pd.DataFrame(data)