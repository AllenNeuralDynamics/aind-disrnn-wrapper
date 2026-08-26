"""Ignore-trial statistics for 3-way (L/R/ignore) generative rollouts.

Characterizes how ``ignore`` (no-response) trials are distributed in a set of
choice sequences, so a model's generated ignore behaviour can be compared to
real animal data on the same footing. All functions consume **raw choice
codes** (0=left, 1=right, 2=ignore) — NOT the NaN-collapsed ``choice_history``
that ``generative_analysis._normalize_choice_value`` produces (that maps 2->NaN
and would erase every real ignore trial). Pull the real-data side from the raw
``animal_response`` column (which keeps the 2s); the model rollout already
stores 2 verbatim in ``choice_history``.

Statistics (per side: model rollouts and real sessions):
  * overall ignore rate,
  * ignore rate vs within-session position (binned fraction through the session),
  * ignore run-length distribution (consecutive-ignore streak lengths),
  * lag-k ignore autocorrelation and P(ignore | prev ignore),
  * the 3x3 L/R/ignore first-order transition matrix.

Everything is stdlib + numpy so it unit-tests without jax/haiku/gym.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_IGNORE = 2
_N_CLASSES = 3  # 0=left, 1=right, 2=ignore


def _clean_sequence(seq: Iterable[Any]) -> np.ndarray:
    """Coerce one session's choices to an int array over {0,1,2}, dropping NaN/None."""
    out = []
    for v in seq:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f):
            continue
        i = int(round(f))
        if i in (0, 1, 2):
            out.append(i)
    return np.asarray(out, dtype=int)


def _iter_sequences(sessions: Iterable[Any]) -> list[np.ndarray]:
    """Accept a list of sequences, or a DataFrame-like with a choice column."""
    seqs: list[np.ndarray] = []
    # DataFrame-like with a per-session list column
    if hasattr(sessions, "columns"):
        for col in ("choice_history", "animal_response", "choice", "action"):
            if col in sessions.columns:
                for val in sessions[col].tolist():
                    if isinstance(val, (list, tuple, np.ndarray)):
                        seqs.append(_clean_sequence(val))
                    else:  # a flat column: treat whole df column as one sequence
                        seqs = [_clean_sequence(sessions[col].tolist())]
                        break
                return [s for s in seqs if s.size]
        raise ValueError(
            f"No choice column found in dataframe (columns={list(sessions.columns)})."
        )
    # plain iterable of sequences
    for val in sessions:
        seqs.append(_clean_sequence(val))
    return [s for s in seqs if s.size]


def overall_ignore_rate(seqs: Sequence[np.ndarray]) -> dict[str, float]:
    total = int(sum(s.size for s in seqs))
    n_ignore = int(sum(int((s == _IGNORE).sum()) for s in seqs))
    rate = (n_ignore / total) if total else math.nan
    # per-session rates -> mean +/- sem across sessions
    per_session = np.array([float((s == _IGNORE).mean()) for s in seqs]) if seqs else np.array([])
    return {
        "n_trials": total,
        "n_ignore": n_ignore,
        "ignore_rate": rate,
        "per_session_mean": float(per_session.mean()) if per_session.size else math.nan,
        "per_session_sem": (
            float(per_session.std(ddof=1) / math.sqrt(per_session.size))
            if per_session.size > 1
            else math.nan
        ),
        "n_sessions": len(seqs),
    }


def ignore_rate_by_position(seqs: Sequence[np.ndarray], n_bins: int = 10) -> dict[str, list]:
    """Ignore fraction in each fractional-position bin, pooled across sessions."""
    num = np.zeros(n_bins)
    den = np.zeros(n_bins)
    for s in seqs:
        if s.size == 0:
            continue
        # fractional position in [0,1)
        pos = np.linspace(0.0, 1.0, s.size, endpoint=False) if s.size > 1 else np.array([0.0])
        b = np.minimum((pos * n_bins).astype(int), n_bins - 1)
        is_ig = (s == _IGNORE).astype(float)
        for bi in range(n_bins):
            m = b == bi
            den[bi] += float(m.sum())
            num[bi] += float(is_ig[m].sum())
    rate = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
    centers = (np.arange(n_bins) + 0.5) / n_bins
    return {
        "bin_center": centers.tolist(),
        "ignore_rate": rate.tolist(),
        "n_trials": den.astype(int).tolist(),
    }


def ignore_run_lengths(seqs: Sequence[np.ndarray]) -> dict[str, Any]:
    """Distribution of consecutive-ignore streak lengths (across all sessions)."""
    counts: Counter[int] = Counter()
    for s in seqs:
        run = 0
        for v in s:
            if v == _IGNORE:
                run += 1
            elif run:
                counts[run] += 1
                run = 0
        if run:
            counts[run] += 1
    if not counts:
        return {"run_length": [], "count": [], "mean_run_length": math.nan, "max_run_length": 0}
    lengths = sorted(counts)
    total_runs = sum(counts.values())
    mean_rl = sum(l * c for l, c in counts.items()) / total_runs
    return {
        "run_length": lengths,
        "count": [counts[l] for l in lengths],
        "mean_run_length": float(mean_rl),
        "max_run_length": int(max(lengths)),
        "n_runs": int(total_runs),
    }


def ignore_autocorrelation(seqs: Sequence[np.ndarray], max_lag: int = 5) -> dict[str, Any]:
    """Autocorrelation of the binary ignore indicator at lags 1..max_lag,
    plus P(ignore|prev ignore) and P(ignore|prev not-ignore) (lag 1)."""
    # pool binary series with session breaks respected
    acf = []
    for lag in range(1, max_lag + 1):
        num = 0.0
        den = 0.0
        # correlation of centered indicator across all sessions at this lag
        xs = []
        ys = []
        for s in seqs:
            if s.size <= lag:
                continue
            b = (s == _IGNORE).astype(float)
            xs.append(b[:-lag])
            ys.append(b[lag:])
        if xs:
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            if x.std() > 0 and y.std() > 0:
                acf.append(float(np.corrcoef(x, y)[0, 1]))
            else:
                acf.append(math.nan)
        else:
            acf.append(math.nan)
    # lag-1 conditional probabilities
    n_ii = n_i = n_ni = n_n = 0
    for s in seqs:
        if s.size < 2:
            continue
        b = (s == _IGNORE).astype(int)
        prev, cur = b[:-1], b[1:]
        n_i += int(prev.sum())
        n_n += int((prev == 0).sum())
        n_ii += int(((prev == 1) & (cur == 1)).sum())
        n_ni += int(((prev == 0) & (cur == 1)).sum())
    return {
        "lag": list(range(1, max_lag + 1)),
        "autocorr": acf,
        "p_ignore_given_prev_ignore": (n_ii / n_i) if n_i else math.nan,
        "p_ignore_given_prev_not_ignore": (n_ni / n_n) if n_n else math.nan,
    }


def transition_matrix(seqs: Sequence[np.ndarray]) -> dict[str, Any]:
    """First-order 3x3 transition matrix over {0=L,1=R,2=ignore}, row-normalized."""
    counts = np.zeros((_N_CLASSES, _N_CLASSES), dtype=float)
    for s in seqs:
        if s.size < 2:
            continue
        for a, b in zip(s[:-1], s[1:]):
            counts[a, b] += 1.0
    row = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row, out=np.full_like(counts, np.nan), where=row > 0)
    return {
        "labels": ["left", "right", "ignore"],
        "counts": counts.astype(int).tolist(),
        "probs": probs.tolist(),
    }


def compute_ignore_statistics(
    sessions: Any,
    *,
    n_position_bins: int = 10,
    max_lag: int = 5,
) -> dict[str, Any]:
    """Full ignore-statistics bundle for one side (model or real).

    ``sessions`` may be a list of choice sequences (each over {0,1,2}) or a
    DataFrame-like with a ``choice_history`` list column (model rollout output)
    or a raw ``animal_response`` column.
    """
    seqs = _iter_sequences(sessions)
    return {
        "overall": overall_ignore_rate(seqs),
        "by_position": ignore_rate_by_position(seqs, n_bins=n_position_bins),
        "run_lengths": ignore_run_lengths(seqs),
        "autocorrelation": ignore_autocorrelation(seqs, max_lag=max_lag),
        "transition_matrix": transition_matrix(seqs),
    }


def compare_sides(model_stats: Mapping[str, Any], real_stats: Mapping[str, Any]) -> dict[str, Any]:
    """Scalar model-vs-real agreement metrics per statistic."""
    def _safe(a, b):
        if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (
            isinstance(b, float) and math.isnan(b)
        ):
            return math.nan
        return float(a) - float(b)

    # position-curve L1 distance (over bins where both defined)
    mp = np.array(model_stats["by_position"]["ignore_rate"], dtype=float)
    rp = np.array(real_stats["by_position"]["ignore_rate"], dtype=float)
    both = ~(np.isnan(mp) | np.isnan(rp))
    pos_l1 = float(np.abs(mp[both] - rp[both]).mean()) if both.any() else math.nan

    # transition-matrix Frobenius distance (over cells where both defined)
    mt = np.array(model_stats["transition_matrix"]["probs"], dtype=float)
    rt = np.array(real_stats["transition_matrix"]["probs"], dtype=float)
    cell = ~(np.isnan(mt) | np.isnan(rt))
    tm_fro = float(np.sqrt(((mt[cell] - rt[cell]) ** 2).sum())) if cell.any() else math.nan

    return {
        "ignore_rate_diff": _safe(
            model_stats["overall"]["ignore_rate"], real_stats["overall"]["ignore_rate"]
        ),
        "ignore_rate_model": model_stats["overall"]["ignore_rate"],
        "ignore_rate_real": real_stats["overall"]["ignore_rate"],
        "mean_run_length_diff": _safe(
            model_stats["run_lengths"]["mean_run_length"],
            real_stats["run_lengths"]["mean_run_length"],
        ),
        "p_ignore_given_prev_ignore_diff": _safe(
            model_stats["autocorrelation"]["p_ignore_given_prev_ignore"],
            real_stats["autocorrelation"]["p_ignore_given_prev_ignore"],
        ),
        "position_curve_L1": pos_l1,
        "transition_matrix_frobenius": tm_fro,
    }
