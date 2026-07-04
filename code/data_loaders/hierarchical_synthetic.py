"""Hierarchical synthetic cognitive-agent generator for embedding-recovery studies.

Generates multisubject synthetic foraging data with a *known* generative
structure so that a data-driven model's recovered subject/session embeddings
can be scored against ground truth:

  * Between-subject structure: each subject occupies a "subregion" of
    Q-learning parameter space -- a centroid drawn from a configurable
    population distribution (``subject_param_dist``).
  * Within-subject structure (optional): parameters drift smoothly across a
    subject's sessions (``drift``), e.g. learn_rate up, |biasL| toward zero,
    softmax inverse temperature up. With ``drift`` empty/None the subject is
    static across sessions (Stage 1); with ``drift`` set, sessions carry a
    deterministic trajectory (Stage 2).

The loader emits, in addition to the merged multisubject dataset that the
existing disRNN/GRU trainers already consume:

  * ``avg_eval_likelihood_groundtruth`` in metadata -- the normalized
    likelihood of the *generating* policy on the pooled eval sessions, which
    all three trainers read back and log as ``groundtruth_likelihood`` +
    ``likelihood_relative_to_groundtruth`` (the headline recovery score).
  * A per-(subject, session) ground-truth parameter table (parquet + json),
    including the resolved per-session RNG seed so any row is independently
    regenerable from (config, seed) alone -- no frozen dataset required.

Reproducibility: every (subject, session) is assigned unique deterministic
seeds via two parallel non-overlapping hierarchies (stride >> max sessions per
subject), one for the agent and one for the task:

  * agent choice sampling:
    ``session_seed = agent_base_seed + subject_idx * subject_seed_stride + 1 + session_idx``
    (agent_base_seed = ``agent.seed``, default 0), passed to ``forager(seed=...)``.
  * task reward schedule:
    ``task_seed = task_base_seed + subject_idx * subject_seed_stride + 1 + session_idx``
    (task_base_seed = ``task.seed`` if set, else agent_base_seed), passed to
    ``task(seed=...)``.

Both foragers and tasks use instance RNGs seeded from these values (verified),
so generation is byte-identical across machines. The two streams share the same
structure but are independently configurable; they coincide only when
``task.seed`` is left unset.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any, Literal, Mapping

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
from utils.multisubject import (
    build_subject_index_maps,
    compute_train_eval_session_ids,
    merge_datasets_with_subject_index,
    normalize_subject_id,
)

logger = logging.getLogger(__name__)

_TASK_LOOKUP = {
    "coupled_block": CoupledBlockTask,
    "uncoupled_block": UncoupledBlockTask,
    "random_walk": RandomWalkTask,
}

# Conservative clamp bounds for ForagerQLearning params (mirrors the pydantic
# field constraints in aind_dynamic_foraging_models: learn_rate in [0,1],
# rates in [0,1], softmax_inverse_temperature >= 0, biasL fit-bounds +/-5).
_PARAM_CLAMP: dict[str, tuple[float, float]] = {
    "learn_rate": (0.0, 1.0),
    "learn_rate_rew": (0.0, 1.0),
    "learn_rate_unrew": (0.0, 1.0),
    "forget_rate_unchosen": (0.0, 1.0),
    "choice_kernel_relative_weight": (0.0, 1.0),
    "choice_kernel_step_size": (0.0, 1.0),
    "softmax_inverse_temperature": (0.0, 100.0),
    "biasL": (-5.0, 5.0),
    "epsilon": (0.0, 1.0),
}


def _clamp(param_name: str, value: float) -> float:
    lo, hi = _PARAM_CLAMP.get(param_name, (-np.inf, np.inf))
    return float(min(hi, max(lo, value)))


def _sample_centroid_value(spec: Mapping[str, Any], rng: np.random.Generator) -> float:
    """Draw a subject centroid value for one parameter from a population spec."""
    typ = str(spec.get("type", "")).lower()
    if typ == "uniform":
        return float(rng.uniform(float(spec["min"]), float(spec["max"])))
    if typ in ("gaussian", "normal"):
        return float(rng.normal(float(spec["mean"]), float(spec["std"])))
    if typ in ("const", "constant", "fixed"):
        return float(spec["value"])
    raise ValueError(f"Unsupported subject_param_dist type {typ!r} for spec {dict(spec)!r}.")


def _apply_drift(
    param_name: str,
    centroid_value: float,
    spec: Mapping[str, Any],
    session_frac: float,
) -> float:
    """Return the drifted (pre-noise) value at a normalized session position.

    ``session_frac`` is in [0, 1]: 0 for the first session, 1 for the last.
    Modes:
      * linear      -> value = centroid + delta * frac
      * toward_zero -> value = centroid * (1 - frac * frac_shrink)   (|value| shrinks)
      * multiplicative -> value = centroid * (1 + rel * frac)
    """
    mode = str(spec.get("mode", "linear")).lower()
    if mode == "linear":
        return centroid_value + float(spec.get("delta", 0.0)) * session_frac
    if mode == "toward_zero":
        frac_shrink = float(spec.get("frac", 0.0))
        return centroid_value * (1.0 - frac_shrink * session_frac)
    if mode == "multiplicative":
        return centroid_value * (1.0 + float(spec.get("rel", 0.0)) * session_frac)
    raise ValueError(f"Unsupported drift mode {mode!r} for param {param_name!r}.")


class HierarchicalCognitiveAgents(DatasetLoader):
    """Generate hierarchical multisubject synthetic data with ground truth.

    Config (Hydra ``data`` group) is passed through ``task`` and ``agent`` dicts
    plus top-level counts, mirroring ``SyntheticCognitiveAgents``. The ``agent``
    dict adds three recovery-specific blocks:

      * ``subject_param_dist``: {param -> population spec} defining each
        subject's centroid (its subregion of parameter space).
      * ``drift`` (optional): {param -> drift spec} for within-subject
        across-session trajectories. Empty/None => static subjects (Stage 1).
      * ``session_noise`` (optional): {param -> std} additive per-session
        Gaussian jitter around the drifted centroid.
    """

    def __init__(
        self,
        task: dict[str, Any],
        agent: dict[str, Any],
        num_trials: int,
        num_subjects: int,
        num_sessions_per_subject: int,
        eval_every_n: int = 2,
        batch_size: int | None = None,
        batch_mode: Literal["single", "rolling", "random"] = "random",
        subject_seed_stride: int = 100_000,
        groundtruth_dir: str | None = None,
        **metadata: object,
    ) -> None:
        super().__init__()
        self.task_config = copy.deepcopy(task)
        self.agent_config = copy.deepcopy(agent)
        self.num_trials = int(num_trials)
        self.num_subjects = int(num_subjects)
        self.num_sessions_per_subject = int(num_sessions_per_subject)
        self.eval_every_n = int(eval_every_n)
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.subject_seed_stride = int(subject_seed_stride)
        # Where to write the ground-truth table; default to the run output dir.
        self.groundtruth_dir = groundtruth_dir or os.environ.get(
            "DISRNN_OUTPUT_DIR", "/results"
        )
        self.metadata_extras = dict(metadata)

    # ------------------------------------------------------------------ #
    # Deterministic non-overlapping seed hierarchy
    # ------------------------------------------------------------------ #
    def _subject_seed(self, base: int, subject_idx: int) -> int:
        """Base seed for a subject; subject centroid RNG uses this value."""
        return int(base) + int(subject_idx) * self.subject_seed_stride

    def _session_seed(self, base: int, subject_idx: int, session_idx: int) -> int:
        """Unique per-(subject, session) seed offset within a subject's block.

        Sessions occupy [subject_base + 1, subject_base + subject_seed_stride);
        since subject_seed_stride >> num_sessions_per_subject, no two
        (subject, session) pairs share a seed within one base stream.
        """
        return self._subject_seed(base, subject_idx) + 1 + int(session_idx)

    # ------------------------------------------------------------------ #
    # Per-(subject, session) parameter resolution
    # ------------------------------------------------------------------ #
    def _subject_centroid(
        self, subject_seed: int, subject_param_dist: Mapping[str, Any]
    ) -> dict[str, float]:
        rng = np.random.default_rng(subject_seed)
        centroid: dict[str, float] = {}
        # Iterate in sorted order so centroid draws are order-stable.
        for param_name in sorted(subject_param_dist.keys()):
            centroid[param_name] = _clamp(
                param_name, _sample_centroid_value(subject_param_dist[param_name], rng)
            )
        return centroid

    def _session_params(
        self,
        centroid: Mapping[str, float],
        base_params: Mapping[str, Any],
        drift: Mapping[str, Any],
        session_noise: Mapping[str, Any],
        session_frac: float,
        session_seed: int,
    ) -> dict[str, float]:
        rng = np.random.default_rng(session_seed)
        params: dict[str, Any] = copy.deepcopy(dict(base_params or {}))
        # Start from the subject centroid for every governed parameter.
        for param_name, centroid_value in centroid.items():
            value = float(centroid_value)
            if param_name in (drift or {}):
                value = _apply_drift(param_name, value, drift[param_name], session_frac)
            noise_std = float((session_noise or {}).get(param_name, 0.0))
            if noise_std > 0.0:
                value = value + float(rng.normal(0.0, noise_std))
            params[param_name] = _clamp(param_name, value)
        return params

    # ------------------------------------------------------------------ #
    # Main entry
    # ------------------------------------------------------------------ #
    def load(self) -> DatasetBundle:
        task_cfg = copy.deepcopy(self.task_config)
        task_type_key = str(task_cfg.pop("type", "")).lower()
        task_class = _TASK_LOOKUP[task_type_key]

        agent_cfg = copy.deepcopy(self.agent_config)
        agent_class_name = str(agent_cfg.get("agent_class", "")).strip()
        agent_class = getattr(generative_model, agent_class_name)
        agent_kwargs = copy.deepcopy(agent_cfg.get("agent_kwargs", {}))
        base_params = copy.deepcopy(agent_cfg.get("agent_params", {}))
        subject_param_dist = copy.deepcopy(agent_cfg.get("subject_param_dist", {}))
        drift = copy.deepcopy(agent_cfg.get("drift", {})) or {}
        session_noise = copy.deepcopy(agent_cfg.get("session_noise", {})) or {}

        base_seed = agent_cfg.get("seed")
        base_seed = 0 if base_seed is None else int(base_seed)
        base_task_seed = task_cfg.pop("seed", None)
        base_task_seed = base_seed if base_task_seed is None else int(base_task_seed)

        if not subject_param_dist:
            raise ValueError(
                "HierarchicalCognitiveAgents requires agent.subject_param_dist to "
                "define each subject's parameter subregion."
            )

        n_sessions = self.num_sessions_per_subject
        if n_sessions < 2:
            raise ValueError("num_sessions_per_subject must be >= 2 for a train/eval split.")

        # Per-subject accumulators (mirrors data_loaders/mice.py multisubject path).
        prepared: list[dict[str, Any]] = []
        groundtruth_rows: list[dict[str, Any]] = []
        # Per-session simulated choices/logits for the ground-truth likelihood.
        eval_choices_chunks: list[np.ndarray] = []
        eval_logits_chunks: list[np.ndarray] = []

        for subject_idx in range(self.num_subjects):
            subject_seed = self._subject_seed(base_seed, subject_idx)
            subject_id = f"synth{subject_idx:03d}"
            centroid = self._subject_centroid(subject_seed, subject_param_dist)

            session_frames: list[pd.DataFrame] = []
            ordered_session_ids: list[str] = []

            for session_idx in range(n_sessions):
                # Non-overlapping seed hierarchy (stride >> n_sessions), one
                # stream for the agent, one for the task.
                session_seed = self._session_seed(base_seed, subject_idx, session_idx)
                session_task_seed = self._session_seed(base_task_seed, subject_idx, session_idx)
                session_frac = 0.0 if n_sessions == 1 else session_idx / (n_sessions - 1)

                session_params = self._session_params(
                    centroid, base_params, drift, session_noise, session_frac, session_seed
                )

                task_kwargs = copy.deepcopy(task_cfg)
                task_kwargs["seed"] = session_task_seed
                task_kwargs.setdefault("num_trials", self.num_trials)
                task_instance = task_class(**task_kwargs)

                forager_kwargs = copy.deepcopy(agent_kwargs)
                forager_kwargs["seed"] = session_seed
                forager = agent_class(**forager_kwargs)
                forager.set_params(**session_params)
                forager.perform(task_instance)

                session_id = f"{subject_id}_s{session_idx:03d}"
                ordered_session_ids.append(session_id)
                session_frames.append(
                    self._session_dataframe(session_id, forager, task_instance)
                )

                # Store simulated choices/logits for ground-truth likelihood,
                # EXACTLY mirroring data_loaders/synthetic.py: choices (T,1,1),
                # logits (T,1,2), no ignore-masking. normalized_likelihood masks
                # only NEGATIVE labels; the softmax foragers never emit choice==2,
                # so pooling all trials matches both the single-subject loader and
                # the trainer's model-NL computation, keeping
                # likelihood_relative_to_groundtruth an apples-to-apples ratio.
                choices = np.asarray(forager.choice_history)[:, np.newaxis, np.newaxis]
                probs = np.asarray(forager.choice_prob).T
                logits = np.log(probs + 1e-10)[:, np.newaxis, :]

                # eval sessions per subject follow the same eval_every_n rule the
                # split uses: 1-based index positions (eval_every_n-1, 2*..-1, ...),
                # verified identical to compute_train_eval_session_ids.
                is_eval = ((session_idx + 1) % self.eval_every_n) == 0
                if is_eval:
                    eval_choices_chunks.append(choices)
                    eval_logits_chunks.append(logits)

                # Resolved ground-truth row (independently regenerable).
                gt_row = {
                    "subject_index": subject_idx,
                    "subject_id": subject_id,
                    "session_index_0based": session_idx,
                    "session_index_1based": session_idx + 1,
                    "session_id": session_id,
                    "is_eval": bool(is_eval),
                    "session_frac": float(session_frac),
                    "session_seed": int(session_seed),
                    "task_seed": int(session_task_seed),
                }
                for pname, pval in session_params.items():
                    gt_row[f"param_{pname}"] = float(pval)
                for pname, pval in centroid.items():
                    gt_row[f"centroid_{pname}"] = float(pval)
                groundtruth_rows.append(gt_row)

            subject_df = pd.concat(session_frames, ignore_index=True)
            prepared.append(
                {
                    "subject_id": subject_id,
                    "df": subject_df,
                    "ordered_session_ids": ordered_session_ids,
                }
            )

        # --- Build dense subject-index maps ---
        ordered_subject_ids, subject_id_to_index, index_to_subject_id = build_subject_index_maps(
            [item["subject_id"] for item in prepared]
        )

        # --- Per-subject datasets + splits, then merge with subject index ---
        full_datasets, train_datasets, eval_datasets = [], [], []
        subject_indices: list[int] = []
        session_max_index_by_subject_index: list[int] = []
        session_context_rows: list[dict[str, Any]] = []
        full_session_ids: list[str] = []
        train_session_ids: list[str] = []
        eval_session_ids: list[str] = []
        ordered_frames: list[pd.DataFrame] = []

        for item in prepared:
            subject_id = item["subject_id"]
            subject_index = subject_id_to_index[subject_id]
            subject_df = item["df"].copy()
            subject_df["subject_index"] = int(subject_index)

            dataset = dl.create_disrnn_dataset(
                subject_df,
                ignore_policy="exclude",
                batch_size=self.batch_size,
                batch_mode=("single" if self.batch_size is None else self.batch_mode),
            )
            dataset_train, dataset_eval = rnn_utils.split_dataset(
                dataset, eval_every_n=self.eval_every_n
            )
            sub_full_ids = item["ordered_session_ids"]
            sub_train_ids, sub_eval_ids = compute_train_eval_session_ids(
                sub_full_ids, eval_every_n=self.eval_every_n
            )

            full_datasets.append(dataset)
            train_datasets.append(dataset_train)
            eval_datasets.append(dataset_eval)
            subject_indices.append(subject_index)
            ordered_frames.append(subject_df)
            session_max_index_by_subject_index.append(int(len(sub_full_ids)))
            full_session_ids.extend(sub_full_ids)
            train_session_ids.extend(sub_train_ids)
            eval_session_ids.extend(sub_eval_ids)
            session_context_rows.append(
                {
                    "subject_id": normalize_subject_id(subject_id),
                    "subject_index": int(subject_index),
                    "ordered_session_ids": [str(s) for s in sub_full_ids],
                    "ordered_source_session_ids": [str(s) for s in sub_full_ids],
                }
            )

        merged_dataset = merge_datasets_with_subject_index(
            full_datasets, subject_indices, batch_size=self.batch_size, batch_mode=self.batch_mode
        )
        merged_train_dataset = merge_datasets_with_subject_index(
            train_datasets, subject_indices, batch_size=self.batch_size, batch_mode=self.batch_mode
        )
        merged_eval_dataset = merge_datasets_with_subject_index(
            eval_datasets, subject_indices, batch_size=self.batch_size, batch_mode=self.batch_mode
        )

        raw_df = pd.concat(ordered_frames, ignore_index=True)

        # --- Ground-truth eval likelihood (generating policy on pooled eval) ---
        if eval_choices_chunks:
            all_eval_choices = np.concatenate(eval_choices_chunks, axis=0)
            all_eval_logits = np.concatenate(eval_logits_chunks, axis=0)
            avg_eval_groundtruth = float(
                rnn_utils.normalized_likelihood(all_eval_choices, all_eval_logits)
            )
        else:
            avg_eval_groundtruth = float("nan")

        # --- Persist the ground-truth table (parquet + json) ---
        groundtruth_df = pd.DataFrame(groundtruth_rows)
        gt_paths = self._write_groundtruth(
            groundtruth_df,
            summary={
                "num_subjects": int(self.num_subjects),
                "num_sessions_per_subject": int(n_sessions),
                "num_trials_per_session": int(self.num_trials),
                "eval_every_n": int(self.eval_every_n),
                "agent_class": agent_class_name,
                "agent_kwargs": agent_kwargs,
                "subject_param_dist": subject_param_dist,
                "drift": drift,
                "session_noise": session_noise,
                "base_seed": int(base_seed),
                "subject_seed_stride": int(self.subject_seed_stride),
                "avg_eval_likelihood_groundtruth": avg_eval_groundtruth,
                "subject_id_to_index": subject_id_to_index,
            },
        )

        metadata: dict[str, Any] = {
            "multisubject": True,
            "num_subjects": int(len(ordered_subject_ids)),
            "subject_ids": ordered_subject_ids,
            "subject_id_to_index": subject_id_to_index,
            "index_to_subject_id": index_to_subject_id,
            "session_max_index_by_subject_index": session_max_index_by_subject_index,
            "session_context": {"indexing": "1_based", "per_subject": session_context_rows},
            "num_trials": int(len(raw_df)),
            "num_sessions": int(len(full_session_ids)),
            "full_session_ids": full_session_ids,
            "train_session_ids": train_session_ids,
            "eval_session_ids": eval_session_ids,
            "eval_every_n": self.eval_every_n,
            "avg_eval_likelihood_groundtruth": avg_eval_groundtruth,
            "batch_size": self.batch_size,
            "batch_mode": self.batch_mode,
            "groundtruth_table_paths": gt_paths,
            "task": self.task_config,
            "agent": self.agent_config,
        }
        metadata.update(self.metadata_extras)

        logger.info(
            "Hierarchical synthetic: %d subjects x %d sessions, merged full=%s train=%s eval=%s "
            "x_names=%s, groundtruth_eval_likelihood=%.4f",
            self.num_subjects,
            n_sessions,
            tuple(merged_dataset._xs.shape),
            tuple(merged_train_dataset._xs.shape),
            tuple(merged_eval_dataset._xs.shape),
            list(merged_dataset.x_names),
            avg_eval_groundtruth,
        )

        return DatasetBundle(
            raw=raw_df,
            train_set=merged_train_dataset,
            eval_set=merged_eval_dataset,
            metadata=metadata,
            extras={"dataset": merged_dataset, "groundtruth_table": groundtruth_df},
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _session_dataframe(self, session_id: str, forager: Any, task: Any) -> pd.DataFrame:
        choice_history = np.asarray(forager.get_choice_history(), dtype=int)
        reward_history = np.asarray(forager.get_reward_history(), dtype=float)
        p_reward = np.asarray(forager.get_p_reward())
        choice_prob = np.asarray(forager.choice_prob)
        n_trials = choice_history.shape[0]
        return pd.DataFrame(
            {
                "ses_idx": np.full(n_trials, session_id, dtype=object),
                "trial": np.arange(n_trials, dtype=int),
                "animal_response": choice_history.astype(int),
                "earned_reward": reward_history.astype(float),
                "reward_baiting": bool(getattr(task, "reward_baiting", False)),
                "p_reward_left": p_reward[0, :],
                "p_reward_right": p_reward[1, :],
                "choice_prob_left": choice_prob[0, :],
                "choice_prob_right": choice_prob[1, :],
            }
        )

    def _write_groundtruth(
        self, groundtruth_df: pd.DataFrame, summary: dict[str, Any]
    ) -> dict[str, str]:
        out_dir = self.groundtruth_dir
        try:
            os.makedirs(out_dir, exist_ok=True)
            parquet_path = os.path.join(out_dir, "groundtruth_params.parquet")
            json_path = os.path.join(out_dir, "groundtruth_summary.json")
            groundtruth_df.to_parquet(parquet_path, index=False)
            with open(json_path, "w") as f:
                json.dump(_jsonable(summary), f, indent=2)
            logger.info("Wrote ground-truth table: %s (%d rows) + %s",
                        parquet_path, len(groundtruth_df), json_path)
            return {"parquet": parquet_path, "json": json_path}
        except Exception as exc:  # pragma: no cover - best-effort artifact
            logger.warning("Could not persist ground-truth table to %s: %s", out_dir, exc)
            return {}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
