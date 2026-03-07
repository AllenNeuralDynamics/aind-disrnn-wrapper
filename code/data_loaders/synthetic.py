from __future__ import annotations

import copy
from typing import Any, Literal

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

    def __init__(
        self,
        seed: int | None = None,
        batch_size: int | None = None,
        batch_mode: Literal["single", "rolling", "random"] = "random",
        **settings: object,
    ) -> None:
        super().__init__(seed=seed)
        self.batch_size = batch_size
        self.batch_mode = batch_mode
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
        batch_size: int | None = None,
        batch_mode: Literal["single", "rolling", "random"] = "random",
        **metadata: object,
    ) -> None:
        # Random seeds are fully configured via the task/agent dictionaries.
        super().__init__()
        self.task_config = copy.deepcopy(task)
        self.agent_config = copy.deepcopy(agent)
        self.num_trials = int(num_trials)
        self.num_sessions = int(num_sessions)
        self.eval_every_n = int(eval_every_n)
        self.batch_size = batch_size
        self.batch_mode = batch_mode
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
            
            # Prepare choices (T, 1, 1) and logits (T, 1, 2) for this session
            choices = np.asarray(forager.choice_history)[:, np.newaxis, np.newaxis]
            probs = np.asarray(forager.choice_prob).T
            logits = np.log(probs + 1e-10)[:, np.newaxis, :]

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
                    "choices": choices,
                    "logits": logits,
                }
            )

        # --- Assemble dataframe and disrnn dataset ---
        raw_df = pd.concat(session_frames, ignore_index=True).sort_values(["ses_idx", "trial"])  # type: ignore[arg-type]
        raw_df.reset_index(drop=True, inplace=True)

        dataset = dl.create_disrnn_dataset(
            raw_df,
            ignore_policy="exclude",
            batch_size=self.batch_size,
            batch_mode=self.batch_mode,
        )

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

        # --- Identify evaluation sessions and compute global groundtruth likelihood ---
        eval_session_indices = np.arange(1, self.num_sessions, self.eval_every_n)
        
        # Concatenate all evaluation sessions along the episode dimension (axis 1)
        all_eval_choices = np.concatenate([session_details[i]["choices"] for i in eval_session_indices], axis=1)
        all_eval_logits = np.concatenate([session_details[i]["logits"] for i in eval_session_indices], axis=1)
        
        # Compute global normalized likelihood (geometric mean over all trials in all eval sessions)
        avg_eval_groundtruth = float(rnn_utils.normalized_likelihood(all_eval_choices, all_eval_logits))

        # --- Package bundle metadata ---
        metadata: dict[str, Any] = {
            "num_trials": self.num_trials,
            "num_sessions": self.num_sessions,
            "eval_every_n": self.eval_every_n,
            "eval_session_indices": eval_session_indices.tolist(),
            "avg_eval_likelihood_groundtruth": avg_eval_groundtruth,
            "batch_size": self.batch_size,
            "batch_mode": self.batch_mode,
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
        

def create_disrnn_dataset_task_trained_rnn(
    df_trials, ignore_policy="include", batch_size=None, features=None
) -> rnn_utils.DatasetRNN:
    """
    Creates a disrnn dataset object

    args:
    df_trials, a trial dataframe, created by aind_dynamic_foraging_data_utils
        must have 'ses_idx' as an column which indicates how to divide
        trials by session
    ignore_policy (str), must be "include" or "exclude", and determines
        how to use trials where the mouse did not response
    batch_size (int) input argument to disrnn dataset
    features (dict), keys must be columns in df_trials to be used as prediction
        features. values are the semantic labels for that feature. If None,
        use previous choice and previous reward
    """

    # Input checking
    if "ses_idx" not in df_trials:
        raise ValueError("df_trials must contain index of sessions ses_idx")
    if ignore_policy not in ["include", "exclude"]:
        raise ValueError('ignore_policy must be either "include" or "exclude"')

    # Copy so we can modify
    df_trials = df_trials.copy()

    # Determine the number of classes in the output prediction
    if ignore_policy == "include":
        n_classes = 3
    else:
        n_classes = 2
        # Remove trials without a response
        df_trials = df_trials[df_trials["animal_response"] != 2]

    # Format inputs
    # Make 0/1 coded reward vector
    df_trials["rewarded"] = df_trials["rewarded"].astype(int)

    # Break down feature dictionary
    if features is None:
        features = {
            "animal_response": "prev choice",
            "rewarded": "prev reward",
        }
    feature_cols = list(features.keys())
    feature_labels = [features[x] for x in feature_cols]

    # Ensure all feature columns are in df_trials
    for feature in feature_cols:
        if feature not in df_trials.columns:
            raise ValueError(
                "input feature '{}' not in df_trials".format(feature)
            )

    # Determine size of input matrix
    # Input matrix has size [# trials, # sessions, # features]
    max_session_length = df_trials.groupby("ses_idx")["trial"].count().max()

    num_sessions = len(df_trials["ses_idx"].unique())
    num_input_features = len(feature_cols)
    # Pad trials to be ignored with -1
    xs = np.full((max_session_length, num_sessions, num_input_features), -1)

    # Load each session into xs
    for dex, ses_idx in enumerate(df_trials["ses_idx"].unique()):
        temp = df_trials.query("ses_idx == @ses_idx")
        this_xs = temp[feature_cols].to_numpy()[:-1, :]
        xs[1 : len(temp), dex, :] = this_xs  # noqa E203

    # Determine size of output matrix
    # Output matrix has size [# trials, # sessions, # features]
    num_output_features = 1
    # pad trials to be ignored with -1
    ys = np.full((max_session_length, num_sessions, num_output_features), -1)

    # Load each session into ys
    for dex, ses_idx in enumerate(df_trials["ses_idx"].unique()):
        temp = df_trials.query("ses_idx == @ses_idx")
        this_ys = temp[["animal_response"]].to_numpy()
        ys[0 : len(temp), dex, :] = this_ys  # noqa E203

    # Pack into a DatasetRNN object
    dataset = rnn_utils.DatasetRNN(
        ys=ys,
        xs=xs,
        y_type="categorical",
        n_classes=n_classes,
        x_names=feature_labels,
        y_names=["choice"],
        batch_size=batch_size,
        batch_mode="random",
    )
    return dataset


class TaskTrainedRNNDatasetLoader(DatasetLoader):
    """Load the foraging dataset for mice experiments."""

    def __init__(
        self,
        subject_ids: list[str],
        ignore_policy: str,
        features: Mapping[str, str],
        eval_every_n: int,
        multisubject: bool = False,
        seed: int | None = None,
        **extras: object,
    ) -> None:
        super().__init__(seed=seed)
        self.subject_ids = subject_ids[0]  # currently assume one subject
        self.ignore_policy = ignore_policy
        self.features = dict(features)
        self.eval_every_n = eval_every_n
        self.multisubject = multisubject
        self.extras = extras

    def load(self) -> DatasetBundle:
        if self.multisubject:
            raise NotImplementedError("Multisubject loading is not yet supported.")

        df = pd.read_pickle(f'/code/TaskTrainedRNNData-logs_{self.subject_ids}.pkl')

        # df = pd.read_pickle(f'/data/DF-TaskTrainedRNNData/TaskTrainedRNNData-logs_{self.subject_ids.replace(":", "_")}.pkl')

        dataset = create_disrnn_dataset_task_trained_rnn(
            df, ignore_policy=self.ignore_policy, features=self.features
        )
        dataset_train, dataset_eval = rnn_utils.split_dataset(
            dataset, eval_every_n=self.eval_every_n
        )
        metadata = {
            "subject_ids": self.subject_ids,
            "ignore_policy": self.ignore_policy,
            "features": self.features,
            "eval_every_n": self.eval_every_n,
            "num_trials": len(df),
            "num_sessions": int(df["ses_idx"].nunique()) if "ses_idx" in df else None,
            "multisubject": self.multisubject,
        }
        metadata.update(self.extras)

        extras = {"dataset": dataset}
        return DatasetBundle(
            raw=df,
            train_set=dataset_train,
            eval_set=dataset_eval,
            metadata=metadata,
            extras=extras,
        )
