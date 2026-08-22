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

  .. warning::
     The two readers' RT distributions are **not** interchangeable. Measured on a
     60-subject sample of the 20260603 snapshot: median 0.136 s (co_asset,
     n=500,867) vs 0.212 s (bonsai_s3, n=54,809) — a 1.56x ratio, with the gap
     widening in the tail (p99 0.70 s vs 1.40 s; two-sample KS = 0.196, p < 1e-300).
     Both were validated as *correct* computations of go-cue-to-first-lick, so this
     is a real difference in the underlying sessions and/or timestamp pipeline, not
     an extraction artifact. Consequences: (a) ``nwb_data_source`` is confounded
     with RT, so any RT effect should be checked for robustness to reader
     composition; (b) the fixed standardization constants below are pooled across
     both readers, so a reader-imbalanced subject sample will not be exactly
     zero-mean. Neither blocks use of the feature, but a finding that rests on RT
     magnitude needs a reader-stratified check.
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

# Raw derived columns (before the previous-trial shift / encoding).
RAW_TIMING_COLUMNS: tuple[str, ...] = ("reaction_time", "n_lick_left", "n_lick_right")

DEFAULT_LICK_WINDOW_S: float = 2.0
# RT is clipped to this range before the log, to bound extreme outliers without
# discarding trials. The upper bound is deliberately LOOSE relative to the
# distribution: the goal is to keep log() finite and bounded, not to police
# plausibility. Measured on a 60-subject / 555,676-responded-trial sample of the
# 20260603 snapshot: p99 = 0.79 s, p99.99 = 1.92 s, max = 3154 s (clearly a
# mis-scored session boundary). Only 7 trials (1.3e-5) exceed 10 s -- and the
# same 7 exceed 5 s, so the exact upper bound is immaterial; a much tighter bound
# would start reshaping the tail the model may legitimately use. The lower bound
# affects 326 trials (5.9e-4) with a recorded RT of ~0 and keeps log() finite.
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


# ── Standardization constants ────────────────────────────────────────────────
# FIXED, documented constants rather than per-run fitted statistics. Two reasons:
#
# 1. The train and held-out loaders are instantiated INDEPENDENTLY
#    (`instantiate(hydra_config.data, **heldout_kwargs)`), so a fitted transform
#    would have to be persisted and threaded between them; any mismatch would
#    silently apply a different transform to held-out data than to train.
#    Constants are identical across splits — and across post-hoc analysis — by
#    construction, with no plumbing and no leakage.
# 2. They are inspectable and reproducible: the numbers below are population
#    statistics measured once on a 60-subject / 545k-responded-trial sample of the
#    20260603 snapshot (see analysis/calibrate_timing_features.py).
#
# Standardization is GLOBAL, never per-subject or per-session. Per-subject
# z-scoring would erase between-mouse differences in reaction time and licking
# vigor — exactly the individual variation the subject embedding exists to
# capture — and would make a mouse's own baseline unrecoverable by the model.
LOG_RT_CENTER: float = -1.822   # mean of log(reaction_time), clipped
LOG_RT_SCALE: float = 0.599     # std  of log(reaction_time), clipped
LICK_CENTER: float = 3.139      # mean lick count per side, go-cue + 2 s
LICK_SCALE: float = 4.159       # std  lick count per side


def encode_timing_features(
    df: pd.DataFrame,
    *,
    rt_clip_s: tuple[float, float] = RT_CLIP_S,
    standardize: bool = True,
) -> pd.DataFrame:
    """Add the *encoded* timing columns the disRNN consumes.

    Produces ``log_reaction_time`` from the raw ``reaction_time`` (clipped, log).
    Missing RT (NaN) encodes to the channel's neutral value (0.0 after
    standardization, i.e. the population mean), mirroring how ignore trials carry
    no reaction time.

    ``standardize`` (default True) centers and scales both channels by the fixed
    population constants above, so every input channel reaches the disRNN's
    information bottleneck at comparable magnitude. This matters because the
    bottleneck's KL penalty is QUADRATIC in input magnitude
    (``elementwise_kl = mus**2 + sigma**2 - 1 - log(sigma**2)``): at multiplier 1
    the mean ``mus**2`` is ~0.5 for a binary choice/reward channel but ~27 for raw
    lick counts, a ~54x difference in KL cost for the same information content.
    The learned per-dimension multiplier can absorb scale in principle, but the
    early-training optimization path still sees the inflated penalty, and — more
    importantly — the per-channel sigma readouts (``update_net_obs`` openness /
    sparsity) are only comparable across channels when their inputs are on
    comparable scales.

    Set ``standardize=False`` to feed native units (seconds-log and raw counts).
    That is the right choice when a figure should read in native units, or to
    reproduce a run launched before this option existed. For a GRU (no
    bottleneck) the choice is largely cosmetic — Adam's per-parameter scaling
    adapts — but keep it consistent across model families so an architecture
    comparison isn't confounded by preprocessing.

    The previous-trial shift itself is done downstream by the dataset builder
    (row ``t`` of ``xs`` holds trial ``t-1``'s features), so we only produce the
    *current-trial* encoded columns here.
    """
    out = df.copy()
    lo, hi = rt_clip_s
    rt = out["reaction_time"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        log_rt = np.log(np.clip(rt, lo, hi))

    lick_l = out["n_lick_left"].to_numpy(dtype=float)
    lick_r = out["n_lick_right"].to_numpy(dtype=float)

    if standardize:
        log_rt = (log_rt - LOG_RT_CENTER) / LOG_RT_SCALE
        lick_l = (lick_l - LICK_CENTER) / LICK_SCALE
        lick_r = (lick_r - LICK_CENTER) / LICK_SCALE

    # Non-finite RT (no response / unmatched row) -> 0.0, which is the population
    # mean when standardized and a neutral log-scale value when not.
    out["log_reaction_time"] = np.where(np.isfinite(log_rt), log_rt, 0.0)
    out["n_lick_left"] = lick_l
    out["n_lick_right"] = lick_r
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


def create_disrnn_dataset_float(
    df_trials: pd.DataFrame,
    *,
    ignore_policy: str = "include",
    batch_size: Optional[int] = None,
    batch_mode: str = "random",
    features: Optional[Mapping[str, str]] = None,
):
    """Float-safe re-implementation of ``aind_disrnn_utils.create_disrnn_dataset``.

    WHY THIS EXISTS — upstream integer-truncation bug
    -------------------------------------------------
    The upstream builder allocates the input tensor with::

        xs = np.full((n_timesteps, n_sessions, n_features), -1)   # int64!

    ``np.full`` with an integer fill value yields an **int64** array, so assigning
    float feature columns into it TRUNCATES them toward zero; the later
    ``xs.astype(float)`` is too late — the precision is already gone. This is
    invisible for the stock features (``animal_response`` in {0,1,2} and
    ``rewarded`` in {0,1} are already integers) but destroys any CONTINUOUS
    feature: in testing, 24,149 distinct log-reaction-time values collapsed to 7
    integers (-6 … 0).

    Verified present in BOTH the SHA pinned by this repo's ``pyproject.toml``
    (``aind_disrnn_utils@74de874d``, ``src/aind_disrnn_utils/data_loader.py`` L76:
    ``xs = np.full((max_session_length, num_sessions, num_input_features), -1)``,
    with the too-late ``xs.astype(float)`` at L99) and in the latest release
    (aind-disrnn-utils 0.0.16), as of 2026-08. This function is
    a faithful copy of the upstream semantics with the tensors allocated as float
    from the start. It is used ONLY when continuous features are requested, so
    integer-only runs keep calling upstream and stay bit-for-bit reproducible.

    TODO: upstream this as a one-line dtype fix in aind_disrnn_utils
    (``np.full(..., -1, dtype=float)``) and drop this shim once the pin moves.

    Semantics preserved exactly
    ---------------------------
    * ``ignore_policy`` "exclude" drops ``animal_response == 2`` rows and yields
      2 classes; "include" keeps them and yields 3.
    * ``rewarded`` is derived from ``earned_reward`` as int.
    * Inputs are the PREVIOUS trial's feature values: row ``t`` of ``xs`` holds
      trial ``t-1``'s features, with row 0 left at the -1 fill.
    * Targets ``ys`` are the current trial's ``animal_response``.
    * Padding beyond a session's length stays at -1 (masked downstream via the
      negative-target rule).
    """
    from disentangled_rnns.library import rnn_utils  # noqa: PLC0415

    if "ses_idx" not in df_trials:
        raise ValueError("df_trials must contain index of sessions ses_idx")
    if ignore_policy not in ("include", "exclude"):
        raise ValueError('ignore_policy must be either "include" or "exclude"')

    df_trials = df_trials.copy()
    if ignore_policy == "include":
        n_classes = 3
    else:
        n_classes = 2
        df_trials = df_trials[df_trials["animal_response"] != 2]

    df_trials["rewarded"] = df_trials["earned_reward"].astype(int)

    if features is None:
        features = {"animal_response": "prev choice", "rewarded": "prev reward"}
    feature_cols = list(features.keys())
    feature_labels = [features[c] for c in feature_cols]
    for col in feature_cols:
        if col not in df_trials.columns:
            raise ValueError(f"input feature '{col}' not in df_trials")

    max_session_length = df_trials.groupby("ses_idx")["trial"].count().max()
    session_ids = df_trials["ses_idx"].unique()
    num_sessions = len(session_ids)

    # The fix: allocate as float so continuous features are not truncated.
    xs = np.full(
        (max_session_length, num_sessions, len(feature_cols)), -1.0, dtype=float
    )
    ys = np.full((max_session_length, num_sessions, 1), -1.0, dtype=float)

    for dex, ses_idx in enumerate(session_ids):
        temp = df_trials[df_trials["ses_idx"] == ses_idx]
        xs[1 : len(temp), dex, :] = temp[feature_cols].to_numpy(dtype=float)[:-1, :]
        ys[0 : len(temp), dex, :] = temp[["animal_response"]].to_numpy(dtype=float)

    return rnn_utils.DatasetRNN(
        ys=ys,
        xs=xs,
        y_type="categorical",
        n_classes=n_classes,
        x_names=feature_labels,
        y_names=["choice"],
        batch_size=batch_size,
        batch_mode=batch_mode,
    )


def has_continuous_features(features: Optional[Mapping[str, str]]) -> bool:
    """True if ``features`` includes any column that is not integer-valued.

    Used to route dataset construction: integer-only feature sets keep using the
    upstream builder (exact reproducibility of prior runs); anything continuous
    must use :func:`create_disrnn_dataset_float` to avoid the truncation bug.

    NOTE the lick-count columns are listed here even though raw counts are
    integers: with ``standardize=True`` (the default) they become continuous, and
    routing must not depend on a flag this predicate cannot see. Sending an
    integer-valued column through the float-safe builder is harmless — it
    produces identical values — whereas missing a continuous one truncates it.
    """
    if not features:
        return False
    continuous = {
        "log_reaction_time",
        "reaction_time",
        "n_lick_left",
        "n_lick_right",
    }
    return bool(set(features) & continuous)


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

    __slots__ = (
        "enabled", "reaction_time", "lick_counts", "lick_window_s", "standardize",
    )

    def __init__(
        self,
        *,
        enabled: bool = False,
        reaction_time: bool = True,
        lick_counts: bool = True,
        lick_window_s: float = DEFAULT_LICK_WINDOW_S,
        standardize: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.reaction_time = bool(reaction_time)
        self.lick_counts = bool(lick_counts)
        self.lick_window_s = float(lick_window_s)
        self.standardize = bool(standardize)

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
            f"lick_counts={self.lick_counts}, lick_window_s={self.lick_window_s}, "
            f"standardize={self.standardize})"
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
        standardize=bool(source.get("standardize", True)),
    )
