"""Load mice behavioral data from the AIND dynamic-foraging parquet database.

Data comes from the public DuckDB/parquet database exposed by
:mod:`aind_dynamic_foraging_database`. Two layers of its API are used:

* :func:`select_sessions` returns the (small) session table, carrying the
  ``task`` and stage (``current_stage_actual``) metadata we select on.
* :func:`fetch_trials` pulls trial rows for the selected sessions (scoped to just
  those subjects' partitions, so the read stays fast) with the session metadata
  joined onto every trial row.

The session id (``session_id`` in the database, e.g. ``"739195_2024-10-03_92026"``)
is surfaced as ``ses_idx`` to match the column name the rest of the pipeline
expects; it still embeds the ``YYYY-MM-DD`` date, so downstream date parsing keeps
working. Subject ids are returned as strings (the database's native form).

Subject selection (the :func:`_partition_subjects` pipeline) is, in order:

1. Load all sessions.
2. Drop subjects with fewer than ``min_sessions`` sessions (counting ALL sessions,
   regardless of stage).
3. Assign each remaining subject its most-common ``task`` as its
   ``subject_curriculum``.
4. Rank subjects within each ``subject_curriculum`` by total session count, desc.
5. For each chosen curriculum, reserve a fixed ~20% heldout test set by dropping
   every ``heldout_every_n``-th subject in the ranked order (keep the first
   ``heldout_every_n - 1`` of every ``heldout_every_n``); the dropped subjects are
   the clean heldout set.
6. From the remaining (~80%) pool, draw a seeded random sample of a per-curriculum
   ``subject_ratio`` of subjects as the training set.

``split="train"`` returns the step-6 sample; ``split="heldout"`` returns the full
reserved set from step 5 (seed-independent, so it reconstructs identically across
runs). The ``mature_only`` flag does NOT affect selection — it only filters which
trials are returned (mature ``STAGE_FINAL``/``GRADUATED`` vs all stages).

    train_df, train_ids = load_mice_from_database(
        split="train",
        curricula=["Coupled Baiting", "Uncoupled Baiting", "Uncoupled Without Baiting"],
        subject_ratio={"Coupled Baiting": 0.5, "Uncoupled Baiting": 0.5},
        seed=0,
    )
    test_df, test_ids = load_mice_from_database(split="heldout", curricula=[...])

    # Direct selection (bypasses the pipeline entirely):
    df, ids = load_mice_from_database(subject_ids=[111, 222, 333])
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Literal, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from aind_dynamic_foraging_database import fetch_trials, select_sessions

logger = logging.getLogger(__name__)


# Stages that qualify a session as "mature".
MATURE_STAGES: Tuple[str, ...] = ("STAGE_FINAL", "GRADUATED")

# ``data.type`` values that use this database-backed selection pipeline (and thus
# have a derived heldout test set). Used to type-gate held-out evaluation so that
# non-database runs (e.g. synthetic) don't trigger a mice-DB query.
MICE_DATABASE_DATA_TYPES: Tuple[str, ...] = ("mice_snapshot",)

DEFAULT_CURRICULA: List[str] = [
    "Coupled Baiting",
    "Uncoupled Baiting",
    "Uncoupled Without Baiting",
]

DEFAULT_COLS: List[str] = [
    "trial",
    "subject_id",
    "ses_idx",
    "animal_response",
    "earned_reward",
    "curriculum_name",
]

# Session-metadata columns to carry from the session table onto every trial row.
# ``task`` drives selection; ``current_stage_actual`` drives the mature-only trial
# filter; ``curriculum_name`` is kept for backwards-compatible cols_to_retain.
_SESSION_METADATA_COLS: List[str] = ["task", "current_stage_actual", "curriculum_name"]

# Identity / session-level columns that are *not* read from the trial table: they
# come from the session selection (renamed or carried on) rather than being
# requested as trial columns from ``fetch_trials`` (which always emits ``trial``).
_NON_TRIAL_COLS = frozenset(
    {
        "subject_id",
        "ses_idx",
        "session_id",
        "_session_id",
        "session_date",
        "trial",
        *_SESSION_METADATA_COLS,
    }
)


# ---------------------------------------------------------------------------
# Subject-selection pipeline
# ---------------------------------------------------------------------------


def _resolve_subject_ratio(
    subject_ratio: Optional[Union[Mapping[str, float], float]],
    curricula: List[str],
) -> Dict[str, float]:
    """Coerce ``subject_ratio`` into a ``{curriculum: float}`` dict.

    ``None`` -> 1.0 for every curriculum (use the whole pool). A scalar broadcasts
    to every curriculum. A mapping is used as-is, with any missing curriculum
    defaulting to 1.0 (logged).
    """
    if subject_ratio is None:
        return {curriculum: 1.0 for curriculum in curricula}
    if isinstance(subject_ratio, (int, float)) and not isinstance(subject_ratio, bool):
        return {curriculum: float(subject_ratio) for curriculum in curricula}

    resolved: Dict[str, float] = {}
    for curriculum in curricula:
        if curriculum in subject_ratio:
            resolved[curriculum] = float(subject_ratio[curriculum])
        else:
            resolved[curriculum] = 1.0
            logger.warning(
                "subject_ratio missing curriculum %r; defaulting to 1.0 (use whole pool).",
                curriculum,
            )
    return resolved


def _partition_subjects(
    df_session: pd.DataFrame,
    *,
    curricula: List[str],
    min_sessions: int = 10,
    heldout_every_n: int = 5,
    subject_ratio: Optional[Union[Mapping[str, float], float]] = None,
    seed: Optional[int] = None,
) -> Dict[str, object]:
    """Partition subjects into train / heldout sets (see module docstring).

    ``df_session`` has one row per session and must contain ``subject_id`` and
    ``task``. All session counts use ALL sessions (stage is ignored here).

    Returns a dict with ``train_ids``, ``heldout_ids``, ``train_pool_ids`` (flat
    string lists), ``subject_curriculum`` (subject_id -> assigned task), and
    ``per_curriculum`` (per-curriculum ranked / heldout / pool / train lists).
    """
    curricula = [str(curriculum) for curriculum in curricula]
    ratio_map = _resolve_subject_ratio(subject_ratio, curricula)

    # Step 1 — load all sessions.
    logger.info(
        "[select 1/6] loaded all sessions: %d subjects, %d sessions",
        df_session["subject_id"].nunique(),
        len(df_session),
    )

    # Step 2 — drop subjects with fewer than min_sessions (counting ALL sessions).
    sessions_per_subject = df_session.groupby("subject_id").size()
    kept_subjects = sessions_per_subject[sessions_per_subject >= min_sessions].index
    df_kept = df_session[df_session["subject_id"].isin(kept_subjects)]
    logger.info(
        "[select 2/6] after >=%d-session filter: %d subjects, %d sessions",
        min_sessions,
        df_kept["subject_id"].nunique(),
        len(df_kept),
    )

    # Step 3 — assign each subject its most-common task (ignoring null tasks;
    # ties broken by task name ascending, for determinism).
    df_task = df_kept[df_kept["task"].notna()]
    task_counts = (
        df_task.groupby(["subject_id", "task"])
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["subject_id", "n", "task"], ascending=[True, False, True])
    )
    subject_curriculum: Dict[str, str] = (
        task_counts.groupby("subject_id", sort=False).first()["task"].to_dict()
    )
    total_sessions = sessions_per_subject.loc[list(kept_subjects)].to_dict()
    logger.info(
        "[select 3/6] assigned subject_curriculum for %d subjects; tasks present: %s",
        len(subject_curriculum),
        sorted(set(subject_curriculum.values())),
    )

    # Step 4 — rank subjects within each curriculum by total session count desc
    # (ties broken by subject_id ascending). Subjects whose assigned curriculum is
    # not requested are dropped.
    bucket: Dict[str, List[str]] = defaultdict(list)
    for subject_id, curriculum in subject_curriculum.items():
        bucket[curriculum].append(subject_id)
    ranked: Dict[str, List[str]] = {
        curriculum: sorted(
            subjects, key=lambda s: (-int(total_sessions[s]), str(s))
        )
        for curriculum, subjects in bucket.items()
    }
    logger.info(
        "[select 4/6] ranked subjects per chosen curriculum: %s",
        {curriculum: len(ranked.get(curriculum, [])) for curriculum in curricula},
    )

    # Step 5 — reserve a fixed heldout set: drop every heldout_every_n-th subject.
    per_curriculum: Dict[str, Dict[str, List[str]]] = {}
    heldout_ids: List[str] = []
    train_pool_ids: List[str] = []
    for curriculum in curricula:
        ranked_list = ranked.get(curriculum, [])
        heldout = [
            subject_id
            for index, subject_id in enumerate(ranked_list)
            if index % heldout_every_n == heldout_every_n - 1
        ]
        pool = [
            subject_id
            for index, subject_id in enumerate(ranked_list)
            if index % heldout_every_n != heldout_every_n - 1
        ]
        per_curriculum[curriculum] = {"ranked": ranked_list, "heldout": heldout, "pool": pool}
        heldout_ids.extend(heldout)
        train_pool_ids.extend(pool)
        logger.info(
            "[select 5/6] curriculum=%r: ranked=%d -> train_pool=%d, heldout=%d",
            curriculum,
            len(ranked_list),
            len(pool),
            len(heldout),
        )
    logger.info(
        "[select 5/6] heldout reservation totals: train_pool=%d, heldout=%d",
        len(train_pool_ids),
        len(heldout_ids),
    )

    # Step 6 — seeded ratio sample of the train pool, per curriculum.
    rng = np.random.default_rng(seed)
    train_ids: List[str] = []
    for curriculum in curricula:
        pool = per_curriculum[curriculum]["pool"]
        ratio = ratio_map[curriculum]
        n_take = max(0, min(len(pool), int(round(ratio * len(pool)))))
        if pool and n_take == 0:
            logger.warning(
                "[select 6/6] curriculum=%r: ratio=%.3f yielded 0 subjects from pool of %d.",
                curriculum,
                ratio,
                len(pool),
            )
        if n_take > 0:
            sampled_index = set(
                int(i) for i in rng.choice(len(pool), size=n_take, replace=False)
            )
            curr_train = [s for i, s in enumerate(pool) if i in sampled_index]
        else:
            curr_train = []
        per_curriculum[curriculum]["train"] = curr_train
        train_ids.extend(curr_train)
        logger.info(
            "[select 6/6] curriculum=%r: ratio=%.3f -> sampled %d/%d subjects",
            curriculum,
            ratio,
            len(curr_train),
            len(pool),
        )
    logger.info(
        "[select 6/6] final training selection: %d subjects: %s", len(train_ids), train_ids
    )

    return {
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "train_pool_ids": train_pool_ids,
        "subject_curriculum": subject_curriculum,
        "per_curriculum": per_curriculum,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_mice_from_database(
    *,
    split: Literal["train", "heldout"] = "train",
    curricula: Optional[List[str]] = None,
    subject_ratio: Optional[Union[Mapping[str, float], float]] = None,
    min_sessions: int = 10,
    heldout_every_n: int = 5,
    seed: Optional[int] = None,
    mature_only: bool = True,
    cols_to_retain: Optional[List[str]] = None,
    subject_ids: Optional[List] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load mice behavioral trials from the foraging database.

    Subjects are partitioned by the pipeline described in the module docstring;
    ``split`` chooses which partition's trials are returned.

    Parameters
    ----------
    split:
        ``"train"`` returns the seeded ratio-sampled training subjects;
        ``"heldout"`` returns the full reserved ~20% heldout test set. Ignored
        when ``subject_ids`` is provided.
    curricula:
        Chosen ``task`` names to include. Defaults to ``Coupled Baiting``,
        ``Uncoupled Baiting``, ``Uncoupled Without Baiting``.
    subject_ratio:
        Per-curriculum fraction of the post-heldout (~80%) pool to sample for
        training (``{task: float}``; a scalar broadcasts; ``None`` -> 1.0 each).
        Only used for ``split="train"``.
    min_sessions:
        Minimum number of (all-stage) sessions a subject must have to be eligible.
    heldout_every_n:
        Reserve every ``heldout_every_n``-th ranked subject per curriculum as
        heldout (default 5 -> ~20%).
    seed:
        Seed for the training-set random sample. The heldout set is independent of
        the seed.
    mature_only:
        If ``True`` *(default)*, only ``STAGE_FINAL``/``GRADUATED`` trials are
        returned. This filters the returned trials only — it does NOT affect which
        subjects are selected.
    cols_to_retain:
        Columns to keep in the returned DataFrame (absent columns are skipped).
    subject_ids:
        If provided, these subjects are used directly and the selection pipeline
        is bypassed entirely (``split``/``curricula``/``subject_ratio`` ignored).

    Returns
    -------
    df : pandas.DataFrame
        Trial-level data for the selected subjects, projected to ``cols_to_retain``
        and ordered by ``subject_id, session_date, trial``. The database
        ``session_id`` is surfaced as ``ses_idx``.
    selected_ids : list[str]
        Subject ids that were selected (as strings).
    """
    if split not in ("train", "heldout"):
        raise ValueError(f"split must be 'train' or 'heldout', got {split!r}.")
    if curricula is None:
        curricula = DEFAULT_CURRICULA
    curricula = [str(curriculum) for curriculum in curricula]
    if cols_to_retain is None:
        cols_to_retain = DEFAULT_COLS

    # Read the session table. Direct selection scopes the read to the requested
    # subjects; the ranking pipeline needs the full population.
    if subject_ids is not None:
        selected_ids = [str(subject_id) for subject_id in subject_ids]
        logger.info(
            "Using directly specified subject_ids (%d): %s", len(selected_ids), selected_ids
        )
        df_session = select_sessions(subjects=selected_ids, columns=_SESSION_METADATA_COLS)
    else:
        logger.info(
            "Selecting %s subjects from database (curricula=%s, min_sessions=%d, "
            "heldout_every_n=%d, seed=%s) …",
            split,
            curricula,
            min_sessions,
            heldout_every_n,
            seed,
        )
        df_session = select_sessions(subjects=None, columns=_SESSION_METADATA_COLS)
        if df_session.empty:
            logger.warning("Session selection returned no rows.")
            return pd.DataFrame(columns=list(cols_to_retain)), []
        df_session["subject_id"] = df_session["subject_id"].astype(str)
        partition = _partition_subjects(
            df_session,
            curricula=curricula,
            min_sessions=min_sessions,
            heldout_every_n=heldout_every_n,
            subject_ratio=subject_ratio,
            seed=seed,
        )
        selected_ids = list(
            partition["train_ids"] if split == "train" else partition["heldout_ids"]
        )

    if df_session.empty:
        logger.warning("Session selection returned no rows.")
        return pd.DataFrame(columns=list(cols_to_retain)), selected_ids
    df_session["subject_id"] = df_session["subject_id"].astype(str)

    # Restrict to the selected subjects; mature_only filters the returned trials.
    selected_set = {str(subject_id) for subject_id in selected_ids}
    valid_mask = df_session["subject_id"].isin(selected_set)
    if mature_only:
        valid_mask &= df_session["current_stage_actual"].isin(MATURE_STAGES)
    valid_sessions = df_session[valid_mask].copy()
    if valid_sessions.empty:
        logger.warning("No sessions remain after subject selection / mature filtering.")
        return pd.DataFrame(columns=list(cols_to_retain)), selected_ids

    # Fetch trials for just those sessions (scoped read), with the session metadata
    # joined on, then surface session_id as ses_idx.
    trial_cols = [c for c in cols_to_retain if c not in _NON_TRIAL_COLS]
    logger.info(
        "Fetching %s trials for %d sessions across %d subjects (trial columns=%s) …",
        split,
        len(valid_sessions),
        valid_sessions["subject_id"].nunique(),
        trial_cols,
    )
    trials = fetch_trials(valid_sessions, columns=trial_cols or None)
    if trials.empty:
        logger.warning("Trial fetch returned no rows.")
        return pd.DataFrame(columns=list(cols_to_retain)), selected_ids

    trials = trials.rename(columns={"session_id": "ses_idx"})
    trials["subject_id"] = trials["subject_id"].astype(str)

    # Project to the requested columns (skip any absent), preserving order.
    existing_cols = [c for c in cols_to_retain if c in trials.columns]
    missing = [c for c in cols_to_retain if c not in trials.columns]
    if missing:
        logger.warning("Requested cols not found in data and will be skipped: %s", missing)
    df_out = trials[existing_cols].copy()

    logger.info(
        "df (%s): %d trials across %d subjects, %d sessions",
        split,
        len(df_out),
        df_out["subject_id"].nunique() if "subject_id" in df_out.columns else -1,
        trials["ses_idx"].nunique(),
    )
    return df_out, selected_ids
