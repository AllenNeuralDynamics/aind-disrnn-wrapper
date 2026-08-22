"""Derive per-trial *timing* features (reaction time, post-go-cue lick counts).

These augment the standard disRNN inputs (previous choice + previous reward) with
two behavioral quantities that the raw choice/reward stream does not carry:

* ``reaction_time`` — go-cue -> first lick latency (s).
* ``n_lick_left`` / ``n_lick_right`` — number of left / right licks in the window
  ``[go_cue, go_cue + lick_window_s)``.

Why a dedicated module instead of extra ``cols_to_retain``
--------------------------------------------------------
The two NWB readers behind the parquet cache store timing under *different*
columns, and lick counts are only reliably present in the **event** table:

* Reaction time — ``bonsai_s3`` populates ``reaction_time`` directly, while
  ``co_asset`` leaves it NULL but populates ``choice_time_in_session`` and
  ``goCue_start_time_in_session`` (RT = choice - go-cue). A COALESCE across the
  two recovers ~100% of responded trials.
* Lick counts — ``co_asset`` trial-table lick-time arrays are unpopulated;
  ``bonsai_s3`` stores them as VARCHAR arrays. The **event** table
  (``left_lick_time`` / ``right_lick_time`` on the session clock) is populated
  for BOTH readers, and event-derived counts cross-validate exactly against the
  bonsai trial-table arrays. So licks are always counted from the event table.

The derived columns are computed once from the database (scoped to the already
selected subjects) and merged onto the trial dataframe on ``(ses_idx, trial)``.
Nothing here changes the disRNN architecture: the trainer infers ``obs_size``
from the input tensor width, and adding feature columns simply widens it.

Design decisions (see the calibration notebook / session notes)
---------------------------------------------------------------
* Feed **raw left and right** lick counts, not total/difference. {total, diff} is
  an invertible rotation of {L, R} carrying identical information, but total is
  strongly reward-collinear (consummatory licking) while difference is
  reward-orthogonal; passing L, R separately lets the disRNN form its own
  combination rather than pre-committing to a rotation.
* Reaction time is fed **log-transformed** (RT is ~log-normal, median ~0.14 s);
  see :func:`encode_timing_features`.
"""

from __future__ import annotations

import logging
from typing import List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Semantic feature labels (values of the ``features`` dict) for the timing inputs.
# Keys are the dataframe column names produced by :func:`attach_timing_features`.
TIMING_FEATURE_LABELS: Mapping[str, str] = {
    "prev_log_reaction_time": "prev log RT",
    "prev_n_lick_left": "prev n_lick_left",
    "prev_n_lick_right": "prev n_lick_right",
}

# Raw derived columns (before the previous-trial shift / encoding).
RAW_TIMING_COLUMNS: tuple[str, ...] = ("reaction_time", "n_lick_left", "n_lick_right")

DEFAULT_LICK_WINDOW_S: float = 2.0
# RT is clipped to this range before the log to tame a handful of extreme
# outliers (the p99 is <1 s; values >~5 s are almost always mis-scored).
RT_CLIP_S: tuple[float, float] = (1e-3, 10.0)


def compute_timing_features(
    subject_ids: Sequence[str],
    *,
    snapshot: Optional[str] = None,
    lick_window_s: float = DEFAULT_LICK_WINDOW_S,
) -> pd.DataFrame:
    """Compute per-trial RT and lick counts for ``subject_ids`` from the database.

    Returns a tidy frame keyed by ``(ses_idx, trial)`` with columns
    ``reaction_time``, ``n_lick_left``, ``n_lick_right``. ``ses_idx`` matches the
    database ``session_id`` (surfaced as ``ses_idx`` by the trial loader).

    The read is scoped to just ``subject_ids`` (their partition files), so it is
    fast even though it touches both the trial and event tables. Only mature
    sessions in the three standard curricula contribute — matching the trainer's
    selection — but this frame is merged by key, so any extra rows are harmless.
    """
    import aind_dynamic_foraging_database as db  # noqa: PLC0415

    subject_ids = [str(s) for s in subject_ids]
    if not subject_ids:
        return pd.DataFrame(
            columns=["ses_idx", "trial", *RAW_TIMING_COLUMNS]
        )

    import duckdb  # noqa: PLC0415

    tsrc = db.read_trials(subject_ids, snapshot=snapshot)
    esrc = db.read_events(subject_ids, snapshot=snapshot)

    query = f"""
      WITH t AS (
        SELECT
          tr.session_id AS ses_idx,
          tr.trial,
          COALESCE(tr.goCue_start_time_in_session, tr.goCue_start_time) AS gocue,
          COALESCE(
            tr.reaction_time,
            tr.choice_time_in_session - tr.goCue_start_time_in_session
          ) AS reaction_time
        FROM {tsrc} tr
      ),
      lk AS (
        SELECT session_id, event, timestamps
        FROM {esrc}
        WHERE event IN ('left_lick_time', 'right_lick_time')
      )
      SELECT
        t.ses_idx,
        t.trial,
        ANY_VALUE(t.reaction_time) AS reaction_time,
        COUNT(*) FILTER (WHERE lk.event = 'left_lick_time')  AS n_lick_left,
        COUNT(*) FILTER (WHERE lk.event = 'right_lick_time') AS n_lick_right
      FROM t
      LEFT JOIN lk
        ON lk.session_id = t.ses_idx
       AND lk.timestamps >= t.gocue
       AND lk.timestamps <  t.gocue + {float(lick_window_s)}
      GROUP BY t.ses_idx, t.trial
    """
    out = duckdb.sql(query).df()
    out["trial"] = out["trial"].astype(int)
    out["n_lick_left"] = out["n_lick_left"].astype(int)
    out["n_lick_right"] = out["n_lick_right"].astype(int)
    logger.info(
        "Computed timing features for %d subjects: %d (session, trial) rows "
        "(lick_window=%.1fs, snapshot=%s).",
        len(subject_ids),
        len(out),
        lick_window_s,
        snapshot,
    )
    return out


def attach_timing_features(
    df: pd.DataFrame,
    *,
    snapshot: Optional[str] = None,
    lick_window_s: float = DEFAULT_LICK_WINDOW_S,
    timing_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge raw timing features onto a trial dataframe (keyed by ses_idx, trial).

    ``df`` must contain ``subject_id``, ``ses_idx`` and ``trial``. When
    ``timing_df`` is given it is used directly (already computed); otherwise the
    features are computed from the database for the subjects present in ``df``.
    Missing values (rare unmatched rows) are left as NaN for RT and 0 for licks.
    """
    for col in ("subject_id", "ses_idx", "trial"):
        if col not in df.columns:
            raise ValueError(f"attach_timing_features requires a '{col}' column.")

    if timing_df is None:
        subject_ids = [str(s) for s in df["subject_id"].unique().tolist()]
        timing_df = compute_timing_features(
            subject_ids, snapshot=snapshot, lick_window_s=lick_window_s
        )

    merged = df.copy()
    merged["trial"] = merged["trial"].astype(int)
    merged = merged.merge(
        timing_df[["ses_idx", "trial", *RAW_TIMING_COLUMNS]],
        on=["ses_idx", "trial"],
        how="left",
    )
    merged["n_lick_left"] = merged["n_lick_left"].fillna(0).astype(int)
    merged["n_lick_right"] = merged["n_lick_right"].fillna(0).astype(int)
    n_missing_rt = int(merged["reaction_time"].isna().sum())
    if n_missing_rt:
        logger.info(
            "attach_timing_features: %d/%d rows have no matched reaction_time "
            "(kept as NaN; encoded to 0 after log).",
            n_missing_rt,
            len(merged),
        )
    return merged


def encode_timing_features(
    df: pd.DataFrame,
    *,
    rt_clip_s: tuple[float, float] = RT_CLIP_S,
) -> pd.DataFrame:
    """Add the *encoded* timing columns the disRNN consumes.

    Produces ``log_reaction_time`` from the raw ``reaction_time`` (clipped, log).
    Lick counts are passed through as raw integer counts (kept as float columns
    for the disRNN dataset). Missing RT (NaN) encodes to 0.0 — a neutral value on
    the standardized-ish log scale — mirroring how ignore trials carry no RT.

    The previous-trial shift itself is done downstream by
    ``create_disrnn_dataset`` (which builds ``xs`` from row ``t-1`` for the target
    at row ``t``), so we only need the *current-trial* encoded columns here.
    """
    out = df.copy()
    lo, hi = rt_clip_s
    rt = out["reaction_time"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        log_rt = np.log(np.clip(rt, lo, hi))
    log_rt = np.where(np.isfinite(log_rt), log_rt, 0.0)
    out["log_reaction_time"] = log_rt
    out["n_lick_left"] = out["n_lick_left"].astype(float)
    out["n_lick_right"] = out["n_lick_right"].astype(float)
    return out


# Column name (in the trial dataframe) -> semantic label, for the current-trial
# encoded features. ``create_disrnn_dataset`` shifts these by one trial to form
# the previous-trial inputs, so the labels describe the previous-trial meaning.
ENCODED_TIMING_FEATURES: Mapping[str, str] = {
    "log_reaction_time": "prev log RT",
    "n_lick_left": "prev n_lick_left",
    "n_lick_right": "prev n_lick_right",
}


def timing_feature_map(
    *,
    include_reaction_time: bool = True,
    include_lick_counts: bool = True,
) -> dict[str, str]:
    """Return the ``features`` sub-mapping for the requested timing inputs."""
    feats: dict[str, str] = {}
    if include_reaction_time:
        feats["log_reaction_time"] = "prev log RT"
    if include_lick_counts:
        feats["n_lick_left"] = "prev n_lick_left"
        feats["n_lick_right"] = "prev n_lick_right"
    return feats


def required_raw_columns(
    *,
    include_reaction_time: bool = True,
    include_lick_counts: bool = True,
) -> List[str]:
    """Raw derived columns needed before encoding, for the requested inputs."""
    cols: List[str] = []
    if include_reaction_time:
        cols.append("reaction_time")
    if include_lick_counts:
        cols.extend(["n_lick_left", "n_lick_right"])
    return cols


class TimingConfig:
    """Resolved timing-feature options (from the data-config ``timing_features`` block).

    Attributes
    ----------
    enabled : bool
        Master switch. When False, the loader behaves exactly as before.
    reaction_time / lick_counts : bool
        Which feature groups to include.
    lick_window_s : float
        Window (s) after go-cue for counting licks.
    """

    __slots__ = ("enabled", "reaction_time", "lick_counts", "lick_window_s")

    def __init__(
        self,
        *,
        enabled: bool = False,
        reaction_time: bool = True,
        lick_counts: bool = True,
        lick_window_s: float = DEFAULT_LICK_WINDOW_S,
    ) -> None:
        self.enabled = bool(enabled)
        self.reaction_time = bool(reaction_time)
        self.lick_counts = bool(lick_counts)
        self.lick_window_s = float(lick_window_s)

    def feature_map(self) -> dict[str, str]:
        if not self.enabled:
            return {}
        return timing_feature_map(
            include_reaction_time=self.reaction_time,
            include_lick_counts=self.lick_counts,
        )

    def raw_columns(self) -> List[str]:
        if not self.enabled:
            return []
        return required_raw_columns(
            include_reaction_time=self.reaction_time,
            include_lick_counts=self.lick_counts,
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"TimingConfig(enabled={self.enabled}, reaction_time={self.reaction_time}, "
            f"lick_counts={self.lick_counts}, lick_window_s={self.lick_window_s})"
        )


def resolve_timing_config(source: Optional[Mapping[str, object]]) -> TimingConfig:
    """Build a :class:`TimingConfig` from a ``timing_features`` mapping.

    Accepts the value of the data-config ``timing_features`` key (a dict / OmegaConf
    mapping), or ``None`` / a bare bool for convenience. Unknown keys are ignored.
    """
    if source is None:
        return TimingConfig(enabled=False)
    if isinstance(source, bool):
        return TimingConfig(enabled=source)
    if not isinstance(source, Mapping):
        raise TypeError(
            f"timing_features must be a mapping, bool, or None; got {type(source)!r}."
        )
    return TimingConfig(
        enabled=bool(source.get("enabled", False)),
        reaction_time=bool(source.get("reaction_time", True)),
        lick_counts=bool(source.get("lick_counts", True)),
        lick_window_s=float(source.get("lick_window_s", DEFAULT_LICK_WINDOW_S)),
    )
