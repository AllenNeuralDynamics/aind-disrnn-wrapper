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
  * A per-(subject, session) ground-truth parameter table (CSV primary, plus
    parquet when an engine is available, and a JSON summary), including the
    resolved per-session RNG seed so any row is independently regenerable from
    (config, seed) alone -- no frozen dataset required. CSV is authoritative
    because the training conda env may lack pyarrow/fastparquet.

Performance: subject simulation is embarrassingly parallel (each subject's
sessions depend only on their own resolved seeds), so it fans out across a
``spawn`` multiprocessing pool (``generation_workers``, default auto =
min(SLURM_CPUS_PER_TASK or cpu_count, num_subjects); env override
``DISRNN_GEN_WORKERS``; set 1 to force serial). Because every (subject, session)
draws from its own seed, the merged dataset and ground-truth table are
byte-identical regardless of worker count. Only the pure-Python/numpy simulation
runs in workers; the jax dataset assembly + merge stays serial in the parent.

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
from typing import Any, Literal, Mapping, Sequence

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

    MONOTONIC modes (Stage 2):
      * linear         -> value = centroid + delta * frac
      * toward_zero    -> value = centroid * (1 - frac * frac_shrink)  (|value| shrinks)
      * multiplicative -> value = centroid * (1 + rel * frac)

    NON-MONOTONIC modes (Stage 2b): a smooth monotonic ramp can be absorbed into a
    static per-subject fit's session-average, so it costs little likelihood. A
    trajectory that reverses cannot be absorbed -> it is the sensitive stressor for
    session conditioning.
      * sinusoidal -> value = centroid + amp * sin(2*pi*cycles*frac + phase)
                      (oscillates ``cycles`` times over the subject's sessions)
      * inverted_u -> value = centroid + amp * 4 * frac * (1 - frac)
                      (0 at both ends, peak +amp at frac=0.5; ``amp`` may be negative
                      for a U shape)
      * piecewise  -> rises by ``delta`` up to frac=``peak`` (default 0.5), then
                      returns toward the centroid by frac=1 (tent / up-then-down)
    """
    mode = str(spec.get("mode", "linear")).lower()
    if mode == "linear":
        return centroid_value + float(spec.get("delta", 0.0)) * session_frac
    if mode == "toward_zero":
        frac_shrink = float(spec.get("frac", 0.0))
        return centroid_value * (1.0 - frac_shrink * session_frac)
    if mode == "multiplicative":
        return centroid_value * (1.0 + float(spec.get("rel", 0.0)) * session_frac)
    if mode == "sinusoidal":
        amp = float(spec.get("amp", 0.0))
        cycles = float(spec.get("cycles", 1.0))
        phase = float(spec.get("phase", 0.0))
        return centroid_value + amp * float(np.sin(2.0 * np.pi * cycles * session_frac + phase))
    if mode == "inverted_u":
        amp = float(spec.get("amp", 0.0))
        return centroid_value + amp * 4.0 * session_frac * (1.0 - session_frac)
    if mode == "piecewise":
        delta = float(spec.get("delta", 0.0))
        peak = float(spec.get("peak", 0.5))
        if session_frac <= peak:
            rise = (session_frac / peak) if peak > 0 else 0.0
        else:
            rise = ((1.0 - session_frac) / (1.0 - peak)) if peak < 1 else 0.0
        return centroid_value + delta * rise
    raise ValueError(f"Unsupported drift mode {mode!r} for param {param_name!r}.")


def _tail_eval_indices(n_sessions: int, heldout_frac: float) -> list[int]:
    """Contiguous TAIL held-out split: the last ``heldout_frac`` fraction of a
    subject's sessions are eval, the rest train. At least 1 eval + 1 train session.
    The model must EXTRAPOLATE drift into unseen later sessions (extrapolation),
    unlike the interleaved split which permits interpolation."""
    if n_sessions < 2:
        raise ValueError("Need >=2 sessions to split.")
    n_eval = max(1, min(n_sessions - 1, int(round(heldout_frac * n_sessions))))
    return list(range(n_sessions - n_eval, n_sessions))


def _assign_subject_presets(
    num_subjects: int,
    n_presets: int,
    *,
    mode: str = "balanced",
    weights: Sequence[float] | None = None,
    base_seed: int = 0,
) -> list[int]:
    """Assign each subject to a preset index (0..n_presets-1), deterministically.

    Stage 3+ mixture generators: a subject is a single model TYPE (preset) for its
    whole lifetime -- an animal is assumed not to switch algorithm within a session
    (per-session family switching is the separate Stage-4b axis). This picks which
    preset each subject uses.

      * ``balanced`` (default): round-robin ``subject_idx % n_presets`` -- equal
        counts, NO RNG, independent of the param/seed streams, fully reproducible.
      * ``random``: draw each subject's preset i.i.d. from ``weights`` (uniform if
        None) using a dedicated RNG stream seeded off ``base_seed`` via an
        independent SeedSequence entropy tag, so it never collides with the
        subject/session seed hierarchies.
    """
    if n_presets <= 0:
        raise ValueError("subject_presets.presets must be non-empty.")
    mode = str(mode).lower()
    if mode == "balanced":
        return [i % n_presets for i in range(num_subjects)]
    if mode == "random":
        if weights is None:
            p = None
        else:
            w = np.asarray([float(x) for x in weights], dtype=float)
            if w.shape[0] != n_presets:
                raise ValueError(
                    "subject_presets.weights length must equal the number of presets."
                )
            p = w / w.sum()
        rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), 0x5EED3]))
        return [int(v) for v in rng.choice(n_presets, size=num_subjects, p=p)]
    raise ValueError(f"Unsupported subject_presets.assignment mode {mode!r}.")


def _subject_mixture_weights(
    num_subjects: int,
    n_presets: int,
    *,
    concentration: float = 0.5,
    base_seed: int = 0,
) -> np.ndarray:
    """Stage-4b: per-subject MIXTURE weights over presets (families).

    Each subject is characterized by a probability vector over the presets -- the
    fraction of its sessions expected to be drawn from each family. This is the
    subject's stable IDENTITY (what the subject embedding should recover), distinct
    from the per-session realized family (what the session embedding should decode).

    Weights are drawn from a Dirichlet(concentration) per subject using a dedicated
    RNG stream (independent entropy tag) so they never collide with the
    subject/session param seeds or the Stage-3 assignment stream. Low concentration
    (<1) -> subjects lean toward one dominant family (sparse mixtures); high (>1) ->
    more uniform mixtures. Deterministic from ``base_seed``.
    """
    if n_presets <= 0:
        raise ValueError("subject_presets.presets must be non-empty.")
    rng = np.random.default_rng(np.random.SeedSequence([int(base_seed), 0x5E5510]))
    alpha = np.full(n_presets, float(concentration), dtype=float)
    # Draw subjects in order so the stream is stable regardless of num_subjects.
    return rng.dirichlet(alpha, size=num_subjects)


def _session_family_draws(
    mixture_weights: np.ndarray,
    n_sessions: int,
    *,
    subject_idx: int,
    base_seed: int = 0,
) -> list[int]:
    """Stage-4b: draw the realized family (preset index) for each of a subject's
    sessions from that subject's mixture weights.

    Uses a per-subject RNG stream (entropy tag + subject_idx) that is independent of
    the session param/task seed hierarchies, so the family sequence is reproducible
    and does not perturb the parameter draws. Returns a list of length ``n_sessions``.
    """
    w = np.asarray(mixture_weights, dtype=float)
    w = w / w.sum()
    rng = np.random.default_rng(
        np.random.SeedSequence([int(base_seed), 0x5E5511, int(subject_idx)])
    )
    return [int(v) for v in rng.choice(w.shape[0], size=int(n_sessions), p=w)]


def _split_dataset_tail(dataset: Any, heldout_frac: float) -> tuple[Any, Any]:
    """Tail-holdout analogue of rnn_utils.split_dataset (session axis = axis 1)."""
    from disentangled_rnns.library.rnn_utils import DatasetRNN

    data = dataset.get_all()
    xs, ys = data["xs"], data["ys"]
    n_sessions = xs.shape[1]
    eval_idx = set(_tail_eval_indices(n_sessions, heldout_frac))
    train_mask = np.array([i not in eval_idx for i in range(n_sessions)], dtype=bool)
    eval_mask = np.logical_not(train_mask)

    def _mk(mask: np.ndarray) -> Any:
        return DatasetRNN(
            xs[:, mask, :], ys[:, mask, :], x_names=dataset.x_names,
            y_names=dataset.y_names, y_type=dataset.y_type, n_classes=dataset.n_classes,
            batch_size=dataset.batch_size, batch_mode=dataset.batch_mode, rng=dataset.rng,
        )

    return _mk(train_mask), _mk(eval_mask)


def _tail_train_eval_session_ids(
    session_ids: Sequence[Any], heldout_frac: float
) -> tuple[list[Any], list[Any]]:
    """Session-id partition matching _split_dataset_tail (last frac -> eval)."""
    ordered = list(session_ids)
    eval_idx = set(_tail_eval_indices(len(ordered), heldout_frac))
    train_ids = [s for i, s in enumerate(ordered) if i not in eval_idx]
    eval_ids = [s for i, s in enumerate(ordered) if i in eval_idx]
    return train_ids, eval_ids


def _session_dataframe_from_forager(session_id: str, forager: Any, task: Any) -> pd.DataFrame:
    """Build the per-session trial dataframe from a performed forager (free fn).

    Free function (not a method) so it is usable inside a multiprocessing worker.
    """
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


def _simulate_subject(work: dict[str, Any]) -> dict[str, Any]:
    """Simulate all sessions for ONE subject. Runs in a worker process.

    Pure Python + numpy (foragers + task); no jax. Deterministic from the seeds
    already resolved by the parent, so any worker count yields identical output.
    Returns picklable numpy/pandas payloads (no jax dataset objects -- those are
    assembled serially in the parent).

    ``work`` keys: subject_idx, subject_id, task_type_key, task_cfg,
    agent_class_name, agent_kwargs, num_trials, sessions (list of dicts with
    session_id, session_params, session_seed, session_task_seed, is_eval).
    """
    task_class = _TASK_LOOKUP[work["task_type_key"]]
    default_agent_class_name = work["agent_class_name"]
    default_agent_kwargs = work["agent_kwargs"]
    num_trials = work["num_trials"]

    session_frames: list[pd.DataFrame] = []
    ordered_session_ids: list[str] = []
    eval_choices_chunks: list[np.ndarray] = []
    eval_logits_chunks: list[np.ndarray] = []

    for sess in work["sessions"]:
        task_kwargs = copy.deepcopy(work["task_cfg"])
        task_kwargs["seed"] = sess["session_task_seed"]
        task_kwargs.setdefault("num_trials", num_trials)
        task_instance = task_class(**task_kwargs)

        # Stage-4b: each session may use its OWN family; fall back to the subject's
        # default (Stage-3/4a, where every session repeats the same agent).
        agent_class = getattr(
            generative_model, sess.get("agent_class_name", default_agent_class_name)
        )
        forager_kwargs = copy.deepcopy(sess.get("agent_kwargs", default_agent_kwargs))
        forager_kwargs["seed"] = sess["session_seed"]
        forager = agent_class(**forager_kwargs)
        forager.set_params(**sess["session_params"])
        forager.perform(task_instance)

        session_id = sess["session_id"]
        ordered_session_ids.append(session_id)
        session_frames.append(
            _session_dataframe_from_forager(session_id, forager, task_instance)
        )

        # Ground-truth likelihood inputs, EXACTLY mirroring synthetic.py:
        # choices (T,1,1), logits (T,1,2), no ignore-masking.
        choices = np.asarray(forager.choice_history)[:, np.newaxis, np.newaxis]
        probs = np.asarray(forager.choice_prob).T
        logits = np.log(probs + 1e-10)[:, np.newaxis, :]
        if sess["is_eval"]:
            eval_choices_chunks.append(choices)
            eval_logits_chunks.append(logits)

    subject_df = pd.concat(session_frames, ignore_index=True)
    return {
        "subject_idx": work["subject_idx"],
        "subject_id": work["subject_id"],
        "df": subject_df,
        "ordered_session_ids": ordered_session_ids,
        "eval_choices": eval_choices_chunks,
        "eval_logits": eval_logits_chunks,
    }


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
        generation_workers: int | None = None,
        heldout_session_mode: Literal["interleaved", "tail"] = "interleaved",
        heldout_frac: float = 0.2,
        **metadata: object,
    ) -> None:
        super().__init__()
        self.task_config = copy.deepcopy(task)
        self.agent_config = copy.deepcopy(agent)
        self.num_trials = int(num_trials)
        self.num_subjects = int(num_subjects)
        self.num_sessions_per_subject = int(num_sessions_per_subject)
        self.eval_every_n = int(eval_every_n)
        # Held-out-session split mode (Stage 2b+, Han 2026-07-05):
        #   interleaved -> every eval_every_n-th session is eval (default; the model
        #     sees sessions on BOTH sides of each eval session -> it can INTERPOLATE
        #     drift, so a static fit pays little -> insensitive to drift).
        #   tail -> the LAST heldout_frac fraction of each subject's sessions are the
        #     held-out eval block (contiguous). The model trains only on earlier
        #     sessions and must EXTRAPOLATE the drift -> the fair, sensitive model-
        #     comparison split for drifting/mixture generators.
        self.heldout_session_mode = str(heldout_session_mode)
        self.heldout_frac = float(heldout_frac)
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.subject_seed_stride = int(subject_seed_stride)
        # Parallel subject simulation. None/<=0 => auto (min(cpu_count, num_subjects),
        # overridable via DISRNN_GEN_WORKERS). 1 => serial (no pool). Simulation is
        # embarrassingly parallel across subjects and deterministic per (subject,
        # session) seed, so any worker count yields byte-identical output.
        if generation_workers is None:
            env_workers = os.environ.get("DISRNN_GEN_WORKERS")
            generation_workers = int(env_workers) if env_workers else 0
        self.generation_workers = int(generation_workers)
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

    def _resolve_workers(self) -> int:
        """Number of simulation worker processes to use.

        ``generation_workers``: 1 => serial; >1 => that many; <=0 => auto. Auto
        prefers SLURM_CPUS_PER_TASK (the cores the scheduler actually gave us),
        else os.cpu_count(), capped at num_subjects. Any count is deterministic.
        """
        if self.generation_workers == 1:
            return 1
        if self.generation_workers > 1:
            return min(self.generation_workers, self.num_subjects)
        # auto
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        cpu_budget = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 1)
        return max(1, min(cpu_budget, self.num_subjects))

    # ------------------------------------------------------------------ #
    # Per-(subject, session) parameter resolution
    # ------------------------------------------------------------------ #
    def _resolve_preset_specs(self) -> tuple[list[dict[str, Any]], list[int]]:
        """Resolve the per-subject generative preset list + subject->preset assignment.

        Returns ``(presets, assignment)`` where each preset is a normalized dict with
        keys: ``name``, ``agent_class``, ``agent_kwargs``, ``agent_params``,
        ``subject_param_dist``, ``drift``, ``session_noise``; ``assignment[i]`` is the
        preset index for subject ``i``.

        BACKWARD-COMPATIBLE: when the agent config has no ``subject_presets`` block,
        returns a single preset built from the top-level agent fields and an all-zero
        assignment -- so the single-class Stage 1/2/2b path is byte-identical (same
        centroid RNG draws in the same order).

        Stage 3 (``subject_presets`` present): each subject is ONE preset for life
        (Bari / Hattori / Rescorla-Wagner, etc.), each preset carrying its own
        agent_class + agent_kwargs + subject_param_dist (+ optional per-preset drift /
        session_noise; falls back to the top-level drift/session_noise when a preset
        omits them).
        """
        agent_cfg = self.agent_config
        top_drift = copy.deepcopy(agent_cfg.get("drift", {})) or {}
        top_noise = copy.deepcopy(agent_cfg.get("session_noise", {})) or {}
        preset_block = agent_cfg.get("subject_presets")
        if not preset_block:
            # Single-class fallback (Stage 1/2/2b) -- identical behavior to before.
            single = {
                "name": str(agent_cfg.get("agent_class", "")).strip(),
                "agent_class": str(agent_cfg.get("agent_class", "")).strip(),
                "agent_kwargs": copy.deepcopy(agent_cfg.get("agent_kwargs", {})),
                "agent_params": copy.deepcopy(agent_cfg.get("agent_params", {})),
                "subject_param_dist": copy.deepcopy(agent_cfg.get("subject_param_dist", {})),
                "drift": top_drift,
                "session_noise": top_noise,
            }
            return [single], [0] * self.num_subjects
        # Stage 3: explicit per-subject presets.
        raw_presets = list(preset_block.get("presets", []))
        if not raw_presets:
            raise ValueError("agent.subject_presets.presets must be a non-empty list.")
        presets: list[dict[str, Any]] = []
        for i, pr in enumerate(raw_presets):
            if not pr.get("subject_param_dist"):
                raise ValueError(
                    f"subject_presets.presets[{i}] requires its own subject_param_dist."
                )
            presets.append(
                {
                    "name": str(pr.get("name", pr.get("agent_class", f"preset{i}"))),
                    "agent_class": str(pr["agent_class"]).strip(),
                    "agent_kwargs": copy.deepcopy(pr.get("agent_kwargs", {})),
                    "agent_params": copy.deepcopy(pr.get("agent_params", {})),
                    "subject_param_dist": copy.deepcopy(pr["subject_param_dist"]),
                    # per-preset drift/noise override the top-level defaults when given
                    "drift": copy.deepcopy(pr["drift"]) if "drift" in pr else top_drift,
                    "session_noise": (
                        copy.deepcopy(pr["session_noise"])
                        if "session_noise" in pr
                        else top_noise
                    ),
                }
            )
        base_seed = agent_cfg.get("seed")
        base_seed = 0 if base_seed is None else int(base_seed)
        assignment = _assign_subject_presets(
            self.num_subjects,
            len(presets),
            mode=str(preset_block.get("assignment", "balanced")),
            weights=preset_block.get("weights"),
            base_seed=base_seed,
        )
        return presets, assignment

    def _resolve_session_switching(self, presets):
        """Stage-4b: resolve per-subject family-mixture weights + per-session family
        draws, or return None when session switching is disabled.

        Returns ``(mixture_weights, session_families)`` or ``None``:
          * ``mixture_weights[i]`` -- probability vector over presets for subject i
            (its stable identity; subject embedding target).
          * ``session_families[i]`` -- list of length n_sessions giving the realized
            preset index for each of subject i's sessions (session embedding target).

        Enabled by ``subject_presets.session_switching.enabled: true``. When disabled
        (or no subject_presets), returns None -> the one-family-per-subject Stage-3/4a
        path is unchanged (byte-identical). The subject's centroid is still drawn ONCE
        per subject from a SHARED param space so a single continuous parameter vector
        threads across the families it visits (see load()).
        """
        preset_block = self.agent_config.get("subject_presets") or {}
        sw = preset_block.get("session_switching") or {}
        if not sw.get("enabled"):
            return None
        n_presets = len(presets)
        base_seed = self.agent_config.get("seed")
        base_seed = 0 if base_seed is None else int(base_seed)
        conc = float(sw.get("concentration", 0.5))
        mixture = _subject_mixture_weights(
            self.num_subjects, n_presets, concentration=conc, base_seed=base_seed
        )
        n_sessions = self.num_sessions_per_subject
        session_families = [
            _session_family_draws(
                mixture[i], n_sessions, subject_idx=i, base_seed=base_seed
            )
            for i in range(self.num_subjects)
        ]
        return mixture, session_families

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
    def groundtruth_table(self) -> "pd.DataFrame":
        """Regenerate ONLY the ground-truth parameter table (per-subject centroids
        + per-session drifted/noised params + is_eval flags), WITHOUT simulating any
        trials. Pure numpy RNG (~1s for 200x40), byte-identical to the table load()
        writes because it makes the same RNG draws in the same order.

        NOT the primary path for recovery analysis: since run_hpc now logs the
        groundtruth_params.csv into each run's W&B output artifact, the analysis reads
        that CSV directly. This method is the FALLBACK + AUDIT tool: (1) score runs
        whose W&B CSV is missing/contaminated (e.g. any launched before the per-run
        groundtruth_dir fix), and (2) cheaply verify a stored CSV still equals what its
        config regenerates (the guarantee behind trusting the CSV).
        """
        agent_cfg = copy.deepcopy(self.agent_config)
        presets, assignment = self._resolve_preset_specs()
        base_seed = agent_cfg.get("seed")
        base_seed = 0 if base_seed is None else int(base_seed)
        base_task_seed = self.task_config.get("seed")
        base_task_seed = base_seed if base_task_seed is None else int(base_task_seed)
        n_sessions = self.num_sessions_per_subject
        if n_sessions < 2:
            raise ValueError("num_sessions_per_subject must be >= 2 for a train/eval split.")
        if self.heldout_session_mode == "tail":
            _eval_idx_set = set(_tail_eval_indices(n_sessions, self.heldout_frac))
        else:
            _eval_idx_set = set(range(self.eval_every_n - 1, n_sessions, self.eval_every_n))

        switching = self._resolve_session_switching(presets)
        rows: list[dict[str, Any]] = []
        for subject_idx in range(self.num_subjects):
            subject_seed = self._subject_seed(base_seed, subject_idx)
            subject_id = f"synth{subject_idx:03d}"
            if switching is None:
                # Stage 3/4a: one preset for life (unchanged, byte-identical).
                preset = presets[assignment[subject_idx]]
                centroid = self._subject_centroid(subject_seed, preset["subject_param_dist"])
                per_session_preset_idx = [assignment[subject_idx]] * n_sessions
                centroids = {assignment[subject_idx]: centroid}
                mix = None
            else:
                # Stage 4b: subject visits multiple families across sessions. Draw ONE
                # centroid per preset the subject could use (offset seed per preset so
                # the draws are independent), and record the subject's mixture weights.
                mixture, session_families = switching
                mix = mixture[subject_idx]
                per_session_preset_idx = session_families[subject_idx]
                centroids = {
                    pi: self._subject_centroid(
                        subject_seed + 1000003 * pi, presets[pi]["subject_param_dist"]
                    )
                    for pi in sorted(set(per_session_preset_idx))
                }
            for session_idx in range(n_sessions):
                pi = per_session_preset_idx[session_idx]
                preset = presets[pi]
                centroid = centroids[pi]
                session_seed = self._session_seed(base_seed, subject_idx, session_idx)
                session_task_seed = self._session_seed(base_task_seed, subject_idx, session_idx)
                session_frac = 0.0 if n_sessions == 1 else session_idx / (n_sessions - 1)
                session_params = self._session_params(
                    centroid, preset["agent_params"], preset["drift"],
                    preset["session_noise"], session_frac, session_seed
                )
                is_eval = session_idx in _eval_idx_set
                row = {
                    "subject_index": subject_idx, "subject_id": subject_id,
                    "preset_index": int(pi),
                    "preset_name": preset["name"], "agent_class": preset["agent_class"],
                    "session_index_0based": session_idx, "session_index_1based": session_idx + 1,
                    "session_id": f"{subject_id}_s{session_idx:03d}",
                    "is_eval": bool(is_eval), "session_frac": float(session_frac),
                    "session_seed": int(session_seed), "task_seed": int(session_task_seed),
                }
                if mix is not None:
                    for k, pr in enumerate(presets):
                        row[f"mixweight_{pr['name']}"] = float(mix[k])
                for pname, pval in session_params.items():
                    row[f"param_{pname}"] = float(pval)
                for pname, pval in centroid.items():
                    row[f"centroid_{pname}"] = float(pval)
                rows.append(row)
        return pd.DataFrame(rows)

    def load(self) -> DatasetBundle:
        task_cfg = copy.deepcopy(self.task_config)
        task_type_key = str(task_cfg.pop("type", "")).lower()
        task_class = _TASK_LOOKUP[task_type_key]

        agent_cfg = copy.deepcopy(self.agent_config)
        presets, assignment = self._resolve_preset_specs()

        base_seed = agent_cfg.get("seed")
        base_seed = 0 if base_seed is None else int(base_seed)
        base_task_seed = task_cfg.pop("seed", None)
        base_task_seed = base_seed if base_task_seed is None else int(base_task_seed)

        n_sessions = self.num_sessions_per_subject
        if n_sessions < 2:
            raise ValueError("num_sessions_per_subject must be >= 2 for a train/eval split.")

        # Precompute the eval-session index set so the ground-truth table's is_eval
        # column matches the actual train/eval split for BOTH modes (all subjects
        # share n_sessions, so this is the same set per subject).
        if self.heldout_session_mode == "tail":
            _eval_idx_set = set(_tail_eval_indices(n_sessions, self.heldout_frac))
        else:
            _eval_idx_set = set(range(self.eval_every_n - 1, n_sessions, self.eval_every_n))

        # --- Resolve all per-(subject, session) params + seeds up front. This is
        # cheap (numpy RNG only) and lets simulation fan out across processes. The
        # ground-truth table is built here so it is independent of worker order.
        switching = self._resolve_session_switching(presets)
        groundtruth_rows: list[dict[str, Any]] = []
        work_items: list[dict[str, Any]] = []
        for subject_idx in range(self.num_subjects):
            subject_seed = self._subject_seed(base_seed, subject_idx)
            subject_id = f"synth{subject_idx:03d}"
            if switching is None:
                # Stage 3/4a: one preset for life (unchanged, byte-identical).
                per_session_preset_idx = [assignment[subject_idx]] * n_sessions
                centroids = {
                    assignment[subject_idx]: self._subject_centroid(
                        subject_seed, presets[assignment[subject_idx]]["subject_param_dist"]
                    )
                }
                mix = None
            else:
                # Stage 4b: per-session family switching from the subject's mixture.
                mixture, session_families = switching
                mix = mixture[subject_idx]
                per_session_preset_idx = session_families[subject_idx]
                centroids = {
                    pi: self._subject_centroid(
                        subject_seed + 1000003 * pi, presets[pi]["subject_param_dist"]
                    )
                    for pi in sorted(set(per_session_preset_idx))
                }

            sessions: list[dict[str, Any]] = []
            for session_idx in range(n_sessions):
                pi = per_session_preset_idx[session_idx]
                preset = presets[pi]
                centroid = centroids[pi]
                # Non-overlapping seed hierarchy (stride >> n_sessions), one
                # stream for the agent, one for the task.
                session_seed = self._session_seed(base_seed, subject_idx, session_idx)
                session_task_seed = self._session_seed(base_task_seed, subject_idx, session_idx)
                session_frac = 0.0 if n_sessions == 1 else session_idx / (n_sessions - 1)
                session_params = self._session_params(
                    centroid, preset["agent_params"], preset["drift"],
                    preset["session_noise"], session_frac, session_seed
                )
                # eval-session flag matches the actual split (interleaved OR tail);
                # _eval_idx_set was precomputed above to mirror the split site.
                is_eval = session_idx in _eval_idx_set
                session_id = f"{subject_id}_s{session_idx:03d}"
                sessions.append(
                    {
                        "session_id": session_id,
                        "session_params": session_params,
                        "session_seed": int(session_seed),
                        "session_task_seed": int(session_task_seed),
                        "is_eval": bool(is_eval),
                        # per-session agent (Stage-4b); Stage-3/4a repeats the subject's.
                        "agent_class_name": preset["agent_class"],
                        "agent_kwargs": preset["agent_kwargs"],
                    }
                )

                gt_row = {
                    "subject_index": subject_idx,
                    "subject_id": subject_id,
                    "preset_index": int(pi),
                    "preset_name": preset["name"],
                    "agent_class": preset["agent_class"],
                    "session_index_0based": session_idx,
                    "session_index_1based": session_idx + 1,
                    "session_id": session_id,
                    "is_eval": bool(is_eval),
                    "session_frac": float(session_frac),
                    "session_seed": int(session_seed),
                    "task_seed": int(session_task_seed),
                }
                if mix is not None:
                    for k, pr in enumerate(presets):
                        gt_row[f"mixweight_{pr['name']}"] = float(mix[k])
                for pname, pval in session_params.items():
                    gt_row[f"param_{pname}"] = float(pval)
                for pname, pval in centroid.items():
                    gt_row[f"centroid_{pname}"] = float(pval)
                groundtruth_rows.append(gt_row)

            work_items.append(
                {
                    "subject_idx": subject_idx,
                    "subject_id": subject_id,
                    "task_type_key": task_type_key,
                    "task_cfg": task_cfg,
                    # subject-level defaults (used when a session omits its own);
                    # Stage-4b sessions carry per-session agent_class_name/agent_kwargs.
                    "agent_class_name": presets[per_session_preset_idx[0]]["agent_class"],
                    "agent_kwargs": presets[per_session_preset_idx[0]]["agent_kwargs"],
                    "num_trials": self.num_trials,
                    "sessions": sessions,
                }
            )

        # --- Simulate subjects (embarrassingly parallel; deterministic per seed) ---
        n_workers = self._resolve_workers()
        if n_workers <= 1 or self.num_subjects <= 1:
            results = [_simulate_subject(w) for w in work_items]
        else:
            import multiprocessing as mp

            logger.info(
                "Simulating %d subjects with %d worker processes.",
                self.num_subjects, n_workers,
            )
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_workers) as pool:
                # chunksize=1 keeps memory bounded (each result carries a subject df)
                results = pool.map(_simulate_subject, work_items, chunksize=1)

        # Reassemble in deterministic subject order (results may return unordered).
        results.sort(key=lambda r: r["subject_idx"])
        prepared: list[dict[str, Any]] = []
        eval_choices_chunks: list[np.ndarray] = []
        eval_logits_chunks: list[np.ndarray] = []
        for res in results:
            prepared.append(
                {
                    "subject_id": res["subject_id"],
                    "df": res["df"],
                    "ordered_session_ids": res["ordered_session_ids"],
                }
            )
            eval_choices_chunks.extend(res["eval_choices"])
            eval_logits_chunks.extend(res["eval_logits"])

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
            # baseline_rl reads bundle.raw directly and requires a subject_id
            # column (GRU/disRNN build from the merged tensor, which already
            # carries Subject ID, so they don't need it -- but it's harmless there).
            subject_df["subject_id"] = normalize_subject_id(subject_id)

            dataset = dl.create_disrnn_dataset(
                subject_df,
                ignore_policy="exclude",
                batch_size=self.batch_size,
                batch_mode=("single" if self.batch_size is None else self.batch_mode),
            )
            sub_full_ids = item["ordered_session_ids"]
            if self.heldout_session_mode == "tail":
                dataset_train, dataset_eval = _split_dataset_tail(
                    dataset, self.heldout_frac
                )
                sub_train_ids, sub_eval_ids = _tail_train_eval_session_ids(
                    sub_full_ids, self.heldout_frac
                )
            else:
                dataset_train, dataset_eval = rnn_utils.split_dataset(
                    dataset, eval_every_n=self.eval_every_n
                )
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
                "presets": [
                    {k: p[k] for k in ("name", "agent_class", "agent_kwargs",
                                        "subject_param_dist", "drift", "session_noise")}
                    for p in presets
                ],
                "preset_assignment": [int(a) for a in assignment],
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
        # Thin wrapper over the module-level free function (single source of truth;
        # the free function is what the multiprocessing workers use).
        return _session_dataframe_from_forager(session_id, forager, task)

    def _write_groundtruth(
        self, groundtruth_df: pd.DataFrame, summary: dict[str, Any]
    ) -> dict[str, str]:
        out_dir = self.groundtruth_dir
        paths: dict[str, str] = {}
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as exc:  # pragma: no cover - best-effort artifact
            logger.warning("Could not create ground-truth dir %s: %s", out_dir, exc)
            return paths

        # CSV is the PRIMARY, dependency-free format the recovery analysis reads
        # (the training conda env may lack pyarrow/fastparquet). Always write it.
        csv_path = os.path.join(out_dir, "groundtruth_params.csv")
        try:
            groundtruth_df.to_csv(csv_path, index=False)
            paths["csv"] = csv_path
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not write ground-truth CSV to %s: %s", csv_path, exc)

        # Parquet is a bonus (smaller, typed) when an engine is available.
        parquet_path = os.path.join(out_dir, "groundtruth_params.parquet")
        try:
            groundtruth_df.to_parquet(parquet_path, index=False)
            paths["parquet"] = parquet_path
        except Exception as exc:  # pragma: no cover - optional
            logger.info("Parquet ground-truth not written (no engine); CSV is authoritative: %s", exc)

        json_path = os.path.join(out_dir, "groundtruth_summary.json")
        try:
            with open(json_path, "w") as f:
                json.dump(_jsonable(summary), f, indent=2)
            paths["json"] = json_path
        except Exception as exc:  # pragma: no cover
            logger.warning("Could not write ground-truth summary JSON to %s: %s", json_path, exc)

        if paths:
            logger.info("Wrote ground-truth table (%d rows): %s",
                        len(groundtruth_df), ", ".join(f"{k}={v}" for k, v in paths.items()))
        return paths


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
