"""Utilities for loading mice behavioral snapshot data from pickle files.

Primary entry point
-------------------
``load_mice_snapshot`` returns a single ``(df, subject_ids)`` pair.  Call it
twice with different selection parameters to produce independent train and eval
splits:

    # Rank-based slice (subjects ranked 0-2 → train, 3-5 → eval per curriculum)
    df_train, train_ids = load_mice_snapshot(
        subject_start=0, subject_end=3,
        mature_only=True,
    )
    df_eval, eval_ids = load_mice_snapshot(
        subject_start=3, subject_end=6,
        mature_only=True,
    )

    # Direct subject selection
    df_train, train_ids = load_mice_snapshot(
        subject_ids=[111, 222, 333],
    )
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


SNAPSHOT_PATHS = [
    '/data/mice_behavioral_data_20260309-0_200-185844.pkl',
    '/data/mice_behavioral_data_20260309-200_400-114605.pkl',
    '/data/mice_behavioral_data_20260309-400_600-213302.pkl',
    '/data/mice_behavioral_data_20260310-600_end-120215.pkl',
    '/data/mice_behavioral_data_20260309-misc_778149_778147_753618-215212.pkl'
]

# Stages that qualify a session as "mature"
MATURE_STAGES: Tuple[str, ...] = ("STAGE_FINAL", "GRADUATED")

DEFAULT_CURRICULA: List[str] = [
    "Uncoupled Without Baiting",
    "Uncoupled Baiting",
    "Coupled Baiting",
]

DEFAULT_COLS: List[str] = [
    "trial",
    "subject_id",
    "ses_idx",
    "animal_response",
    "earned_reward",
]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_snapshot_files() -> pd.DataFrame:
    """Load and concatenate one or more pickle snapshot files."""
    dfs = []
    for path in SNAPSHOT_PATHS:
        path = Path(path)
        logger.info("Loading snapshot file: %s", path)
        with open(path, "rb") as f:
            dfs.append(pickle.load(f))

    # Sanity-check column consistency
    ref_cols = set(dfs[0].columns)
    for i, df in enumerate(dfs[1:], start=1):
        if set(df.columns) != ref_cols:
            logger.warning("Column mismatch between file 0 and file %d", i)

    df_all = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Concatenated %d files → %d trials, %d subjects, %d sessions",
        len(SNAPSHOT_PATHS),
        len(df_all),
        df_all["subject_id"].nunique(),
        df_all["ses_idx"].nunique(),
    )
    return df_all


def _get_han_session_table() -> pd.DataFrame:
    """Fetch the HAN session table (lazily imported to avoid hard dependency)."""
    from aind_analysis_arch_result_access import han_pipeline  # noqa: PLC0415

    logger.info("Fetching HAN session table …")
    return han_pipeline.get_session_table(if_load_bpod=True)


def _build_session_metadata(df_all: pd.DataFrame, df_han: pd.DataFrame) -> pd.DataFrame:
    """Join session-level trial data with HAN metadata (stage, curriculum)."""
    df_session = (
        df_all[["subject_id", "ses_idx"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    df_session["session_date"] = df_session["ses_idx"].str.split("_").str[1]

    def _lookup(row: pd.Series) -> pd.Series:
        mask = (
            (df_han["subject_id"] == row["subject_id"])
            & (df_han["session_date"] == row["session_date"])
        )
        matches = df_han[mask]
        if len(matches) == 0:
            # logger.warning(
            #     "No HAN row for subject_id=%s, session_date=%s",
            #     row["subject_id"], row["session_date"],
            # )
            return pd.Series(
                {"nwb_suffix": None, "current_stage_actual": None, "curriculum_name": None}
            )
        if len(matches) > 1:
            logger.warning(
                "Multiple HAN rows for subject_id=%s, session_date=%s",
                row["subject_id"], row["session_date"],
            )
            return pd.Series(
                {"nwb_suffix": None, "current_stage_actual": None, "curriculum_name": None}
            )
        r = matches.iloc[0]
        return pd.Series(
            {
                "nwb_suffix": r["nwb_suffix"],
                "current_stage_actual": r["current_stage_actual"],
                "curriculum_name": r["curriculum_name"],
            }
        )

    df_session[["nwb_suffix", "current_stage_actual", "curriculum_name"]] = (
        df_session.apply(_lookup, axis=1)
    )

    n_before = len(df_session)
    df_session = df_session.dropna(subset=["nwb_suffix"]).reset_index(drop=True)
    logger.info(
        "Dropped %d sessions not found in HAN table. Remaining: %d",
        n_before - len(df_session),
        len(df_session),
    )
    return df_session


def _select_subjects(
    df_session: pd.DataFrame,
    mature_only: bool,
    curricula: List[str],
    subject_ids: Optional[List] = None,
    subject_start: Optional[int] = None,
    subject_end: Optional[int] = None,
) -> List:
    """Select subject IDs either directly or by ranking within curricula.

    Parameters
    ----------
    df_session:
        Session metadata table produced by :func:`_build_session_metadata`.
        Not used when ``subject_ids`` is provided.
    mature_only:
        If ``True``, rank by number of *mature* sessions only.
        If ``False``, rank by total session count.
        Ignored when ``subject_ids`` is provided.
    curricula:
        Which curricula to include. Ignored when ``subject_ids`` is provided.
    subject_ids:
        If provided, return these IDs directly without any ranking.
    subject_start:
        Start index (inclusive, 0-based) of the ranked slice per curriculum.
        ``None`` means start from the beginning.
    subject_end:
        End index (exclusive) of the ranked slice per curriculum.
        ``None`` means go to the end of the ranked list.

    Returns
    -------
    List of selected subject IDs.
    """
    if subject_ids is not None:
        logger.info(
            "Using directly specified subject_ids (%d): %s", len(subject_ids), subject_ids
        )
        return list(subject_ids)

    # Rank-based selection
    if mature_only:
        df_candidate = df_session[
            df_session["current_stage_actual"].isin(MATURE_STAGES)
        ].copy()
        count_col = "n_mature_sessions"
        logger.info("Ranking subjects by mature sessions (%s)", ", ".join(MATURE_STAGES))
    else:
        df_candidate = df_session.copy()
        count_col = "n_sessions"
        logger.info("Ranking subjects by all sessions (mature_only=False)")

    ranked_per_curriculum = (
        df_candidate[df_candidate["curriculum_name"].isin(curricula)]
        .groupby(["curriculum_name", "subject_id"])
        .size()
        .rename(count_col)
        .reset_index()
        .sort_values(["curriculum_name", count_col], ascending=[True, False])
        .groupby("curriculum_name")["subject_id"]
        .apply(list)
    )

    selected_ids: List = []
    for ranked_list in ranked_per_curriculum:
        selected_ids.extend(ranked_list[subject_start:subject_end])

    logger.info("Selected subjects (%d): %s", len(selected_ids), selected_ids)
    return selected_ids


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_mice_snapshot(
    subject_ids: Optional[List] = None,
    subject_start: Optional[int] = None,
    subject_end: Optional[int] = None,
    mature_only: bool = True,
    curricula: Optional[List[str]] = None,
    cols_to_retain: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List]:
    """Load mice behavioral snapshots and return a DataFrame and subject IDs.

    Subjects are selected either directly (via ``subject_ids``) or by ranking
    within each curriculum by session count and slicing with
    ``subject_start``/``subject_end``.

    To produce separate train and eval splits, call this function twice with
    different selection parameters (e.g. different slices or subject ID lists).

    Parameters
    ----------
    subject_ids:
        If provided, these subject IDs are used directly and no ranking is
        performed. Mutually exclusive with ``subject_start``/``subject_end``.
    subject_start:
        Start index (inclusive, 0-based) of the ranked slice per curriculum.
        ``None`` means start from the beginning of the ranked list.
    subject_end:
        End index (exclusive) of the ranked slice per curriculum.
        ``None`` means go to the end of the ranked list.
    mature_only:
        If ``True`` *(default)*, only sessions with stage ``STAGE_FINAL`` or
        ``GRADUATED`` are counted when ranking subjects and included in the
        returned DataFrame.
        If ``False``, all sessions are used for both ranking and inclusion.
        Ignored for session filtering when ``subject_ids`` is provided, but
        still applied to session-level filtering if ``mature_only=True``.
    curricula:
        Curricula to consider when ranking. Defaults to
        ``['Uncoupled Without Baiting', 'Uncoupled Baiting', 'Coupled Baiting']``.
        Ignored when ``subject_ids`` is provided.
    cols_to_retain:
        Columns to keep in the returned DataFrame. Columns that are absent
        from the raw data are silently skipped. Defaults to
        ``['trial', 'subject_id', 'ses_idx', 'animal_response', 'earned_reward']``.

    Returns
    -------
    df : pd.DataFrame
        Trial-level data for the selected subjects.
    selected_ids : list
        Subject IDs that were selected.
    """
    if subject_ids is not None and (subject_start is not None or subject_end is not None):
        raise ValueError(
            "Specify either subject_ids or subject_start/subject_end, not both."
        )
    if curricula is None:
        curricula = DEFAULT_CURRICULA
    if cols_to_retain is None:
        cols_to_retain = DEFAULT_COLS

    df_all = _load_snapshot_files()
    df_han = _get_han_session_table()
    df_session = _build_session_metadata(df_all, df_han)

    # Restrict df_all to sessions that were resolved in df_session
    if mature_only:
        valid_sessions = df_session[
            df_session["current_stage_actual"].isin(MATURE_STAGES)
        ][["subject_id", "ses_idx"]]
    else:
        valid_sessions = df_session[["subject_id", "ses_idx"]]

    df_filtered = df_all.merge(valid_sessions, on=["subject_id", "ses_idx"], how="inner")

    selected_ids = _select_subjects(
        df_session,
        mature_only=mature_only,
        curricula=curricula,
        subject_ids=subject_ids,
        subject_start=subject_start,
        subject_end=subject_end,
    )

    # Keep only columns that exist in the data
    existing_cols = [c for c in cols_to_retain if c in df_filtered.columns]
    missing = set(cols_to_retain) - set(existing_cols)
    if missing:
        logger.warning("Requested cols not found in data and will be skipped: %s", missing)

    df_out = df_filtered[df_filtered["subject_id"].isin(selected_ids)][existing_cols].copy()

    logger.info(
        "df: %d trials across %d subjects",
        len(df_out), df_out["subject_id"].nunique(),
    )
    return df_out, selected_ids