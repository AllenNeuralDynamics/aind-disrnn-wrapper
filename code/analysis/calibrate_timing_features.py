"""Calibrate the incremental value of timing inputs with a logistic probe.

This is the pre-registration-style justification for the RT + lick-count disRNN
variant: before spending GPU on the full model, quantify whether previous-trial
reaction time and lick counts carry choice-predictive information *beyond* a
strong choice+reward history baseline, using a cheap logistic regression with a
session-held-out split (mirroring the disRNN eval split, ``eval_every_n=2``).

It is a LINEAR LOWER BOUND: the disRNN can exploit nonlinear / temporal
structure a logistic model cannot, so a positive Δ here is a conservative signal
that the feature is worth adding. A near-zero Δ would argue against it.

Feature-encoding finding this script demonstrates
-------------------------------------------------
Lick {total, difference} is an invertible rotation of {left, right} and carries
identical information for a linear model, but total licks are strongly collinear
with previous reward (consummatory licking) while difference is orthogonal.
Feeding raw left/right lets the model form its own combination. This script
reports both encodings so the choice is inspectable.

Usage
-----
    python analysis/calibrate_timing_features.py \
        --n-subjects 60 --snapshot 20260603 --lick-window-s 2.0 \
        --output-dir /path/to/out

Outputs ``timing_calibration.csv`` (one row per feature set) and
``timing_calibration.json`` (metadata + provenance).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CURRICULA = ["Coupled Baiting", "Uncoupled Baiting", "Uncoupled Without Baiting"]
MATURE_STAGES = ("STAGE_FINAL", "GRADUATED")


# ---------------------------------------------------------------------------
# Subject selection (mirrors utils.load_mice_database._partition_subjects)
# ---------------------------------------------------------------------------
def select_cohort_subjects(
    session_df: pd.DataFrame,
    *,
    min_sessions: int = 10,
    heldout_every_n: int = 5,
) -> Dict[str, List[str]]:
    """Return {'train': [...], 'heldout': [...]} subject ids by the trainer's rule."""
    df = session_df.copy()
    df["subject_id"] = df["subject_id"].astype(str)
    n_per = df.groupby("subject_id").size()
    kept = n_per[n_per >= min_sessions].index
    df_kept = df[df["subject_id"].isin(kept)]
    df_task = df_kept[df_kept["task"].notna()]
    tc = (
        df_task.groupby(["subject_id", "task"]).size().rename("n").reset_index()
        .sort_values(["subject_id", "n", "task"], ascending=[True, False, True])
    )
    subj_cur = tc.groupby("subject_id", sort=False).first()["task"].to_dict()
    train, heldout = [], []
    for cur in CURRICULA:
        subs = [s for s, c in subj_cur.items() if c == cur]
        ranked = sorted(subs, key=lambda s: (-int(n_per[s]), s))
        heldout += [s for i, s in enumerate(ranked, 1) if i % heldout_every_n == 0]
        train += [s for i, s in enumerate(ranked, 1) if i % heldout_every_n != 0]
    return {"train": sorted(set(train)), "heldout": sorted(set(heldout))}


# ---------------------------------------------------------------------------
# Logistic fit + session-held-out normalized likelihood
# ---------------------------------------------------------------------------
def _fit_eval(design: Dict[str, np.ndarray], y: np.ndarray,
              train_mask: np.ndarray, eval_mask: np.ndarray) -> float:
    from scipy.optimize import minimize

    M = np.column_stack([design[k] for k in design])

    def nll(w, Mx, yx):
        z = Mx @ w
        return np.logaddexp(0, z).sum() - (yx * z).sum()

    w = minimize(nll, np.zeros(M.shape[1]), args=(M[train_mask], y[train_mask]),
                 method="L-BFGS-B").x
    ze = M[eval_mask] @ w
    ll = -(np.logaddexp(0, ze).sum() - (y[eval_mask] * ze).sum())
    return float(np.exp(ll / eval_mask.sum()))


def run_calibration(seq: pd.DataFrame, *, n_lags: int = 3) -> pd.DataFrame:
    """Fit nested logistic models on a prepared previous-trial dataframe.

    ``seq`` must contain (for responded trials only, ignore-excluded first, then
    a per-session previous-trial shift): p{1..n_lags}_animal_response,
    p{1..n_lags}_rewarded, p1_rt, p1_n_lick_left, p1_n_lick_right, animal_response.
    """
    ch1 = seq["p1_animal_response"].values * 2 - 1
    y = seq["animal_response"].values
    sids = seq["session_id"].astype("category").cat.codes.values
    tr, ev = sids % 2 == 0, sids % 2 == 1

    def z(a):
        a = np.asarray(a, float)
        s = a[tr].std()
        return (a - a[tr].mean()) / (s if s > 0 else 1.0)

    base = {"bias": np.ones(len(seq))}
    for L in range(1, n_lags + 1):
        ch = seq[f"p{L}_animal_response"].values * 2 - 1
        base[f"ch{L}"] = ch
        base[f"rewch{L}"] = ch * seq[f"p{L}_rewarded"].values

    Lc, Rc = seq["p1_n_lick_left"].values, seq["p1_n_lick_right"].values
    lrt = z(np.log(np.clip(seq["p1_rt"].values, 1e-3, 10)))
    enc_rt = {"lrt": lrt, "lrt_x_ch": lrt * ch1}
    enc_lr = {"nL": z(np.log1p(Lc)), "nL_x_ch": z(np.log1p(Lc)) * ch1,
              "nR": z(np.log1p(Rc)), "nR_x_ch": z(np.log1p(Rc)) * ch1}
    enc_td = {"tot": z(np.log1p(Lc + Rc)), "tot_x_ch": z(np.log1p(Lc + Rc)) * ch1,
              "dif": z(Rc - Lc), "dif_x_ch": z(Rc - Lc) * ch1}

    sets = {
        "baseline (choice+reward history)": {},
        "+ reaction time": enc_rt,
        "+ licks {left, right}": enc_lr,
        "+ licks {total, difference}": enc_td,
        "+ RT and licks {L, R}": {**enc_rt, **enc_lr},
    }
    base_lik = _fit_eval(base, y, tr, ev)
    rows = []
    for name, extra in sets.items():
        lik = _fit_eval({**base, **extra}, y, tr, ev)
        rows.append({"feature_set": name, "eval_norm_likelihood": lik,
                     "delta_vs_baseline": lik - base_lik,
                     "n_params": len(base) + len(extra)})
    return pd.DataFrame(rows)


def prepare_sequence(feat: pd.DataFrame, *, n_lags: int = 3) -> pd.DataFrame:
    """Ignore-exclude, then per-session previous-trial shift (disRNN ordering)."""
    seq = feat[feat["animal_response"] < 2].sort_values(["session_id", "trial"]).copy()
    seq["rewarded"] = seq["earned_reward"].astype(float)
    g = seq.groupby("session_id", sort=False)
    seq["p1_rt"] = g["rt"].shift(1) if "rt" in seq else g["reaction_time"].shift(1)
    seq["p1_n_lick_left"] = g["n_lick_left"].shift(1)
    seq["p1_n_lick_right"] = g["n_lick_right"].shift(1)
    for L in range(1, n_lags + 1):
        seq[f"p{L}_animal_response"] = g["animal_response"].shift(L)
        seq[f"p{L}_rewarded"] = g["rewarded"].shift(L)
    need = ["p1_rt", "p1_n_lick_left", "p1_n_lick_right",
            f"p{n_lags}_animal_response", f"p{n_lags}_rewarded"]
    return seq.dropna(subset=need).copy()


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-subjects", type=int, default=60)
    p.add_argument("--snapshot", default="20260603")
    p.add_argument("--lick-window-s", type=float, default=2.0)
    p.add_argument("--split", default="train", choices=["train", "heldout"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-lags", type=int, default=3)
    p.add_argument("--output-dir", type=Path, default=Path("."))
    args = p.parse_args(argv)

    import aind_dynamic_foraging_database as db  # noqa: PLC0415
    from utils.trial_timing_features import compute_timing_features  # noqa: PLC0415

    db.use_snapshot(args.snapshot)
    sdb = db.session_db()
    import duckdb  # noqa: PLC0415
    session_df = duckdb.sql(
        f"SELECT _session_id, subject_id, session_date, task, current_stage_actual "
        f"FROM read_parquet('{sdb}')"
    ).df().rename(columns={"subject_id": "subject_id"})

    cohort = select_cohort_subjects(session_df)[args.split]
    rng = np.random.default_rng(args.seed)
    subjects = sorted(rng.choice(sorted(cohort),
                                 min(args.n_subjects, len(cohort)), replace=False).tolist())
    logger.info("Calibrating on %d %s subjects", len(subjects), args.split)

    # Pull choice/reward + attach timing, restricted to mature curricula sessions.
    sel = db.select_sessions(
        subjects=subjects,
        columns=["_session_id", "subject_id", "task", "current_stage_actual"],
    )
    sel = sel[sel["current_stage_actual"].isin(MATURE_STAGES) & sel["task"].isin(CURRICULA)]
    trials = db.fetch_trials(sel, columns=["animal_response", "earned_reward"])
    trials = trials.rename(columns={"session_id": "session_id"})
    timing = compute_timing_features(subjects, snapshot=args.snapshot,
                                     lick_window_s=args.lick_window_s)
    timing = timing.rename(columns={"ses_idx": "session_id"})
    feat = trials.merge(timing, on=["session_id", "trial"], how="left")
    feat["n_lick_left"] = feat["n_lick_left"].fillna(0)
    feat["n_lick_right"] = feat["n_lick_right"].fillna(0)

    seq = prepare_sequence(feat, n_lags=args.n_lags)
    result = run_calibration(seq, n_lags=args.n_lags)
    logger.info("\n%s", result.to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_dir / "timing_calibration.csv", index=False)
    meta = {
        "n_subjects": len(subjects), "subjects": subjects, "split": args.split,
        "snapshot": args.snapshot, "lick_window_s": args.lick_window_s,
        "n_lags": args.n_lags, "n_modelling_trials": int(len(seq)),
    }
    (args.output_dir / "timing_calibration.json").write_text(json.dumps(meta, indent=1))
    logger.info("Wrote calibration to %s", args.output_dir)


if __name__ == "__main__":
    main()
