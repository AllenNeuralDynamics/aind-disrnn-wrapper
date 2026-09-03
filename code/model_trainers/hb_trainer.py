"""Hierarchical Bayesian cognitive-model trainer.

Fits ``HB-Hattori2019`` through the same ``ModelTrainer`` interface as the neural models
and ``baseline_rl``, so all four consume the identical ``DatasetBundle`` and are scored on
the identical trials. That shared path is what makes the comparison valid: reimplementing
the session filters and ignore policy here and hoping they matched would not.

Two estimators are available. ``two_stage`` fits each subject independently and then fits a
population distribution over the resulting posteriors; ``one_stage`` infers all three levels
jointly. The joint fit is the reference the approximation is judged against.

The model itself lives in ``aind-dynamic-foraging-models`` and knows nothing about this
stack; this module only marshals data, runs the fit, and reports.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from base.interfaces import ModelTrainer
from utils.multisubject import compute_train_eval_session_ids
from base.types import DatasetBundle

logger = logging.getLogger(__name__)

FEW_SHOT_K = (0, 1, 2, 4, 8)


def _extract_subject_sessions(
    raw_df: pd.DataFrame,
    session_ids: Optional[List[Any]] = None,
) -> Tuple[Dict[Any, List[np.ndarray]], Dict[Any, List[np.ndarray]]]:
    """Group a raw trial dataframe into per-subject lists of session arrays.

    Ignored trials (``animal_response`` outside {0, 1}) are dropped, matching
    ``baseline_rl`` and the wrapper's ``ignore_policy="exclude"``. The HB and the neural
    models must score the same trials or the held-out comparison is meaningless.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Trial-level frame with ``subject_id``, ``ses_idx``, ``trial``,
        ``animal_response`` and ``earned_reward``.
    session_ids : list, optional
        Restrict to these sessions, in this order.

    Returns
    -------
    tuple of dict
        Choices, rewards and session ids, each keyed by subject id. The session ids
        preserve dataframe order, so a split computed from them matches the one the neural
        models use.
    """
    required = {"subject_id", "ses_idx", "trial", "animal_response", "earned_reward"}
    missing = sorted(required - set(raw_df.columns))
    if missing:
        raise ValueError(f"Raw dataframe missing columns required by HBTrainer: {missing}")

    ordered = (
        list(session_ids)
        if session_ids is not None
        else list(dict.fromkeys(raw_df["ses_idx"].tolist()))
    )
    choices: Dict[Any, List[np.ndarray]] = {}
    rewards: Dict[Any, List[np.ndarray]] = {}
    session_ids_by_subject: Dict[Any, List[Any]] = {}

    for session_id in ordered:
        session_df = raw_df[raw_df["ses_idx"] == session_id].sort_values("trial")
        if session_df.empty:
            continue
        choice_arr = session_df["animal_response"].to_numpy(dtype=int)
        valid = (choice_arr == 0) | (choice_arr == 1)
        if not np.any(valid):
            continue
        subject = session_df["subject_id"].iloc[0]
        choices.setdefault(subject, []).append(choice_arr[valid].astype(int))
        rewards.setdefault(subject, []).append(
            session_df["earned_reward"].to_numpy(dtype=float)[valid]
        )
        session_ids_by_subject.setdefault(subject, []).append(session_id)
    return choices, rewards, session_ids_by_subject


def _pad_cohort(
    choices: Mapping[Any, List[np.ndarray]],
    rewards: Mapping[Any, List[np.ndarray]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Any]]:
    """Pad ragged per-subject sessions into dense arrays with masks.

    Sessions are never truncated: the neural models pad to the longest session and mask,
    so trimming here would change the trial set and void the comparison.

    Returns
    -------
    tuple
        ``(choices, rewards, valid_mask, session_mask, subject_ids)`` where the arrays are
        ``(n_subjects, n_sessions, n_trials)`` and ``session_mask`` is
        ``(n_subjects, n_sessions)``.
    """
    subject_ids = list(choices.keys())
    n_subjects = len(subject_ids)
    n_sessions = max(len(choices[s]) for s in subject_ids)
    n_trials = max(len(a) for s in subject_ids for a in choices[s])

    choice_arr = np.zeros((n_subjects, n_sessions, n_trials), dtype=int)
    reward_arr = np.zeros((n_subjects, n_sessions, n_trials), dtype=float)
    valid_mask = np.zeros((n_subjects, n_sessions, n_trials), dtype=bool)
    session_mask = np.zeros((n_subjects, n_sessions), dtype=bool)

    for i, subject in enumerate(subject_ids):
        for j, (c, r) in enumerate(zip(choices[subject], rewards[subject])):
            choice_arr[i, j, : len(c)] = c
            reward_arr[i, j, : len(r)] = r
            valid_mask[i, j, : len(c)] = True
            session_mask[i, j] = True

    return choice_arr, reward_arr, valid_mask, session_mask, subject_ids


def _pad_context(subjects, choices, rewards, context_indices):
    """Pad each subject's context sessions into one dense batch with masks.

    Subjects contribute different numbers of context sessions, and the batched adaptation
    needs a rectangular array. Padded slots are masked out rather than truncated.

    Returns
    -------
    tuple of np.ndarray
        ``(choices, rewards, session_mask, valid_mask)``, the first two shaped
        ``(n_subjects, max_context, max_trials)``.
    """
    n_subjects = len(subjects)
    max_context = max((len(idx) for idx in context_indices), default=0)
    max_trials = max(
        (len(choices[s][i]) for s, idx in zip(subjects, context_indices) for i in idx),
        default=1,
    )

    choice_arr = np.zeros((n_subjects, max_context, max_trials), dtype=int)
    reward_arr = np.zeros((n_subjects, max_context, max_trials), dtype=float)
    valid = np.zeros((n_subjects, max_context, max_trials), dtype=bool)
    session_mask = np.zeros((n_subjects, max_context), dtype=bool)

    for row, (subject, idx) in enumerate(zip(subjects, context_indices)):
        for slot, i in enumerate(idx):
            c = choices[subject][i]
            choice_arr[row, slot, : len(c)] = c
            reward_arr[row, slot, : len(c)] = rewards[subject][i][: len(c)]
            valid[row, slot, : len(c)] = True
            session_mask[row, slot] = True
    return choice_arr, reward_arr, session_mask, valid


def _log_fit_artifact(wandb_run, saved, estimator, n_subjects):
    """Upload the persisted posterior to W&B so it is retrievable off this filesystem.

    Local netCDF is fine until someone needs the draws from another machine, or after the
    scratch directory is cleaned. Versioning it against the run also ties the posterior to
    the exact config that produced it.
    """
    if wandb_run is None:
        return
    try:
        import wandb

        artifact = wandb.Artifact(
            name=f"hb-fit-{estimator}-D{n_subjects}",
            type="hb_posterior",
            metadata={"estimator": estimator, "n_subjects": n_subjects,
                      **saved.get("diagnostics", {})},
        )
        for key in ("netcdf", "sample_stats", "json"):
            path = saved.get(key)
            if path and Path(path).exists():
                artifact.add_file(path)
        wandb_run.log_artifact(artifact)
        logger.info("HBTrainer: logged posterior artifact %s", artifact.name)
    except Exception as error:  # pragma: no cover - never fail a fit over telemetry
        logger.warning("HBTrainer: could not log posterior artifact: %s", error)


def _log_per_subject_table(wandb_run, per_subject):
    """Publish per-subject held-out scores as a W&B table."""
    if wandb_run is None or not per_subject:
        return
    try:
        import wandb

        columns = ["subject_id", "likelihood", "n_context", "n_scored", "n_trials"]
        table = wandb.Table(columns=columns)
        for subject, row in sorted(per_subject.items()):
            table.add_data(
                subject, float(row["likelihood"]), int(row["n_context"]),
                int(row["n_scored"]), int(row["n_trials"]),
            )
        wandb_run.log({"heldout/per_subject_matched": table})
        logger.info("HBTrainer: logged per-subject table (%d rows)", len(per_subject))
    except Exception as error:  # pragma: no cover - never fail a fit over telemetry
        logger.warning("HBTrainer: could not log per-subject table: %s", error)


def _source_revisions():
    """Git SHAs of the repositories whose code produced a fit.

    The model lives in aind-dynamic-foraging-models and the orchestration here, so a
    dispatcher SHA alone pins none of the code that produced a number.
    """
    import subprocess

    out = {}
    for label, module in (("wrapper", None), ("models", "aind_dynamic_foraging_models")):
        try:
            if module is None:
                path = Path(__file__).resolve().parents[2]
            else:
                path = Path(__import__(module).__file__).resolve().parents[2]
            sha = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=10, check=True,
            ).stdout.strip()
            out[f"{label}_git_sha"] = sha
        except Exception:  # pragma: no cover - absent git or a non-repo install
            out[f"{label}_git_sha"] = None
    return out


def _flatten_for_scoring(subjects, choices, rewards, score_indices):
    """Flatten the sessions to be scored into one padded batch.

    Scoring every held-out session in a single vmapped pass needs a rectangular array and a
    map from each row back to the subject whose adapted posterior it should use.

    Returns
    -------
    dict
        ``choices`` and ``rewards`` of shape ``(n_rows, max_trials)``, a ``valid_mask``,
        ``subject_indices`` giving each row's position in the batched fit, and
        ``rows_by_subject`` for aggregating results back per subject.
    """
    rows, subject_indices, rows_by_subject = [], [], []
    for position, (subject, idx) in enumerate(zip(subjects, score_indices)):
        rows_by_subject.append([])
        for i in idx:
            rows_by_subject[position].append(len(rows))
            rows.append((subject, i))
            subject_indices.append(position)

    max_trials = max((len(choices[s][i]) for s, i in rows), default=1)
    choice_arr = np.zeros((len(rows), max_trials), dtype=int)
    reward_arr = np.zeros((len(rows), max_trials), dtype=float)
    valid = np.zeros((len(rows), max_trials), dtype=bool)
    for row, (subject, i) in enumerate(rows):
        c = choices[subject][i]
        choice_arr[row, : len(c)] = c
        reward_arr[row, : len(c)] = rewards[subject][i][: len(c)]
        valid[row, : len(c)] = True

    return {
        "choices": choice_arr,
        "rewards": reward_arr,
        "valid_mask": valid,
        "subject_indices": np.asarray(subject_indices, dtype=int),
        "rows_by_subject": [np.asarray(r, dtype=int) for r in rows_by_subject],
    }


def _normalized_likelihood(total_log_lik: float, total_trials: int) -> float:
    """Geometric-mean per-trial likelihood, the metric shared with the neural models."""
    if total_trials == 0:
        return 0.0
    return float(np.exp(total_log_lik / total_trials))


class HBTrainer(ModelTrainer):
    """Fit a hierarchical Bayesian cognitive model and score held-out subjects."""

    def __init__(
        self,
        estimator: str = "one_stage",
        num_warmup: int = 500,
        num_samples: int = 500,
        num_chains: int = 4,
        beta_max: float = 10.0,
        few_shot_k: Any = FEW_SHOT_K,
        eval_every_n: int = 2,
        artifact_dir: Optional[str] = None,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        save_session_sites: bool = True,
        architecture: Mapping[str, Any] | Any = {},
        output_dir: str = "/results/outputs",
        seed: Optional[int] = None,
        **_: Any,
    ) -> None:
        """Initialize the HBTrainer.

        Arguments arrive from Hydra's ``instantiate`` on the model config, so each is a
        named parameter rather than an opaque config object -- the same contract the other
        trainers use.

        Parameters
        ----------
        estimator : str
            ``"one_stage"`` for the joint three-level fit, or ``"two_stage"`` for empirical
            Bayes. One-stage is both the reference estimator and, on GPU, the cheaper one.
        num_warmup, num_samples, num_chains : int
            NUTS settings.
        beta_max : float
            Upper bound of ``softmax_inverse_temperature``.
        few_shot_k : sequence of int
            Context-session counts to score, alongside the matched rung.
        eval_every_n : int
            Per-subject train/eval session split, matching the neural models'.
        artifact_dir : str, optional
            Where to persist posterior draws and diagnostics.
        target_accept_prob : float
            NUTS target acceptance probability; NumPyro's default is 0.8 and Stan's
            ``adapt_delta`` is the same quantity. Raising it to 0.9-0.95 shortens the
            integrator's steps, the standard remedy for divergences that survive longer
            warmup, at the cost of a slower fit. Exposed because divergences are curvature
            the integrator cannot follow, so they are **not** fixed by more draws.
        max_tree_depth : int
            NUTS maximum trajectory depth (NumPyro default 10). Raise only if the sampler
            reports saturating it; a saturated tree means the step size is too small for
            the posterior's scale, which is a different failure from divergence.
        save_session_sites : bool
            Persist the session level too (``save_fit``'s ``include_session_sites``, which
            defaults to off). On the ``one_stage`` route that adds ``session_log_lik``,
            which is the bulk of the artifact -- but a site exists only while the sampler
            does, so a fit that ships without it can be recovered only by re-fitting, at
            hours per fit. Defaults on for that reason; turn it off for a throwaway fit.
        architecture : mapping
            Present for parity with the other trainers; unused.
        output_dir : str
            Run output directory.
        seed : int, optional
            Seed for the sampler.
        """
        super().__init__(seed=seed)
        self.estimator = str(estimator)
        self.num_warmup = int(num_warmup)
        self.num_samples = int(num_samples)
        self.num_chains = int(num_chains)
        self.beta_max = float(beta_max)
        self.few_shot_k = tuple(few_shot_k)
        self.eval_every_n = int(eval_every_n)
        self.artifact_dir = artifact_dir
        self.target_accept_prob = float(target_accept_prob)
        self.max_tree_depth = int(max_tree_depth)
        self.save_session_sites = bool(save_session_sites)
        self.output_dir = output_dir

    def fit(
        self,
        bundle: DatasetBundle,
        loggers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fit the model on the bundle's training split and score held-out subjects.

        Parameters
        ----------
        bundle : DatasetBundle
            Shared dataset material, identical to what the neural models receive.
        loggers : dict, optional
            May contain a ``"wandb"`` run.

        Returns
        -------
        dict
            Fitted population parameters, timings and held-out likelihoods by ``k``.
        """
        import jax

        from aind_dynamic_foraging_models.hierarchical_bayes.heldout import (
            batched_heldout_log_lik,
            fit_adaptation_batched,
        )

        started = time.time()
        wandb_run = (loggers or {}).get("wandb")
        estimator = self.estimator
        num_warmup, num_samples = self.num_warmup, self.num_samples
        num_chains, beta_max = self.num_chains, self.beta_max
        k_values, eval_every_n = self.few_shot_k, self.eval_every_n
        artifact_dir = self.artifact_dir

        if bundle.raw is None or len(bundle.raw) == 0:
            raise ValueError("HBTrainer requires bundle.raw with trial-level rows.")

        choices, rewards, _ = _extract_subject_sessions(bundle.raw)
        choice_arr, reward_arr, valid_mask, session_mask, subject_ids = _pad_cohort(
            choices, rewards
        )
        logger.info(
            "HBTrainer: %d subjects, up to %d sessions, up to %d trials (estimator=%s)",
            len(subject_ids), choice_arr.shape[1], choice_arr.shape[2], estimator,
        )

        population, fit_info, mcmc = self._fit_population(
            estimator, choice_arr, reward_arr, valid_mask, session_mask,
            num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
            beta_max=beta_max,
        )
        fit_seconds = time.time() - started

        # Persist the fit before scoring. A cohort fit costs hours and scoring adds more;
        # keeping only posterior means would force a refit for every later question.
        if artifact_dir and mcmc is not None:
            from aind_dynamic_foraging_models.hierarchical_bayes.artifacts import save_fit

            saved = save_fit(
                mcmc, artifact_dir, name=f"{estimator}_fit",
                include_session_sites=self.save_session_sites,
                meta={
                    "estimator": estimator,
                    "n_subjects": len(subject_ids),
                    "subject_ids": [str(s) for s in subject_ids],
                    "num_warmup": num_warmup,
                    "num_samples": num_samples,
                    "num_chains": num_chains,
                    # Needed to invert the softmax_inverse_temperature transform when
                    # per-session parameters are reconstructed offline from theta_raw
                    # (phi(theta[..., 3]) * beta_max). It is a config key, so a rung can
                    # override the published 10.0; without it recorded, an offline replay
                    # would silently rescale beta. See dispatcher #115.
                    "beta_max": beta_max,
                    # Recorded so a rung's convergence can be read against the settings
                    # that produced it, without going back to the launch config.
                    "target_accept_prob": self.target_accept_prob,
                    "max_tree_depth": self.max_tree_depth,
                    "seed": self.seed,
                    **_source_revisions(),
                },
            )
            fit_info["artifacts"] = saved
            logger.info("HBTrainer: wrote fit artifacts to %s", saved["netcdf"])
            _log_fit_artifact(wandb_run, saved, estimator, len(subject_ids))
        logger.info(
            "HBTrainer: population fitted in %.0fs: %s",
            fit_seconds, {k: np.round(np.asarray(v), 4).tolist() for k, v in population.items()},
        )
        # Optional hook so long runs can checkpoint the population before scoring starts.
        callback = getattr(self, "on_population_fitted", None)
        if callable(callback):
            callback({k: np.asarray(v).tolist() for k, v in population.items()}, dict(fit_info))

        # -- Held-out evaluation, one adaptation per (subject, k) --
        # The held-out cohort must be supplied explicitly. Falling back to bundle.raw
        # would adapt and score on the very subjects the population was fitted to, which
        # inflates every number silently; better to skip and say so.
        heldout_df = (bundle.extras or {}).get("heldout_raw")
        if heldout_df is None or len(heldout_df) == 0:
            logger.warning(
                "HBTrainer: no bundle.extras['heldout_raw']; skipping held-out evaluation. "
                "Scoring the training cohort would leak."
            )
            return {
                "estimator": estimator,
                "n_subjects": len(subject_ids),
                "fit_seconds": fit_seconds,
                "population": {
                    name: np.asarray(value).tolist() for name, value in population.items()
                },
                "heldout_likelihood": {},
                "heldout_skipped": True,
                **fit_info,
            }

        heldout_choices, heldout_rewards, heldout_session_ids = (
            _extract_subject_sessions(heldout_df)
        )
        scores: Dict[Any, float] = {}
        rng_key = jax.random.PRNGKey(int(self.seed or 0))
        # One batched adaptation per rung rather than one per subject. Subjects are
        # independent given the frozen population, and sequential fitting paid the scan
        # depth once per subject: about four hours per rung over this cohort.
        for k in k_values:
            eligible = [s for s, sess in heldout_choices.items() if len(sess) > k]
            if not eligible:
                scores[k] = 0.0
                continue
            key_fit, key_draw, rng_key = jax.random.split(rng_key, 3)
            context_c, context_r, ctx_session_mask, ctx_valid = _pad_context(
                eligible, heldout_choices, heldout_rewards, [list(range(k))] * len(eligible)
            )
            samples = fit_adaptation_batched(
                context_c, context_r, population, rng_key=key_fit,
                session_mask=ctx_session_mask, valid_mask=ctx_valid,
                num_warmup=num_warmup, num_samples=num_samples, beta_max=beta_max,
            )
            score_idx = [
                list(range(k, len(heldout_choices[subject]))) for subject in eligible
            ]
            flat = _flatten_for_scoring(
                eligible, heldout_choices, heldout_rewards, score_idx
            )
            session_log_lik, session_trials = batched_heldout_log_lik(
                samples, flat["subject_indices"], flat["choices"], flat["rewards"],
                valid_mask=flat["valid_mask"], rng_key=key_draw, beta_max=beta_max,
            )
            total_log_lik = float(np.sum(session_log_lik))
            total_trials = int(np.sum(session_trials))
            scores[k] = _normalized_likelihood(total_log_lik, total_trials)
            logger.info("HBTrainer: k=%d heldout likelihood %.5f", k, scores[k])

        # Matched conditioning: condition on exactly the sessions the per-mouse MLE
        # baseline fits, and score exactly the ones it scores. Without this the HB is
        # compared against MLE across different amounts of conditioning, which is not a
        # comparison. eval_every_n mirrors study 01's data config.
        matched_log_lik, matched_trials = 0.0, 0
        per_subject_matched: Dict[str, Any] = {}
        matched_subjects, context_indices, score_indices = [], [], []
        for subject, ids in heldout_session_ids.items():
            if len(ids) < 2:
                continue
            train_ids, eval_ids = compute_train_eval_session_ids(ids, eval_every_n)
            index_of = {sid: i for i, sid in enumerate(ids)}
            matched_subjects.append(subject)
            context_indices.append([index_of[sid] for sid in train_ids])
            score_indices.append([index_of[sid] for sid in eval_ids])

        key_fit, key_draw, rng_key = jax.random.split(rng_key, 3)
        context_c, context_r, ctx_session_mask, ctx_valid = _pad_context(
            matched_subjects, heldout_choices, heldout_rewards, context_indices
        )
        matched_samples = fit_adaptation_batched(
            context_c, context_r, population, rng_key=key_fit,
            session_mask=ctx_session_mask, valid_mask=ctx_valid,
            num_warmup=num_warmup, num_samples=num_samples, beta_max=beta_max,
        )

        flat = _flatten_for_scoring(
            matched_subjects, heldout_choices, heldout_rewards, score_indices
        )
        session_log_lik, session_trials = batched_heldout_log_lik(
            matched_samples, flat["subject_indices"], flat["choices"], flat["rewards"],
            valid_mask=flat["valid_mask"], rng_key=key_draw, beta_max=beta_max,
        )
        for position, subject in enumerate(matched_subjects):
            rows = flat["rows_by_subject"][position]
            subject_log_lik = float(np.sum(session_log_lik[rows]))
            subject_trials = int(np.sum(session_trials[rows]))
            per_subject_matched[str(subject)] = {
                "likelihood": _normalized_likelihood(subject_log_lik, subject_trials),
                "n_context": len(context_indices[position]),
                "n_scored": len(score_indices[position]),
                "n_trials": subject_trials,
            }
        matched_log_lik = float(np.sum(session_log_lik))
        matched_trials = int(np.sum(session_trials))
        scores["matched"] = _normalized_likelihood(matched_log_lik, matched_trials)
        logger.info(
            "HBTrainer: matched-conditioning heldout likelihood %.5f (eval_every_n=%d)",
            scores["matched"], eval_every_n,
        )

        output = {
            "estimator": estimator,
            "n_subjects": len(subject_ids),
            "fit_seconds": fit_seconds,
            "population": {
                name: np.asarray(value).tolist() for name, value in population.items()
            },
            "heldout_likelihood": scores,
            "heldout_per_subject_matched": per_subject_matched,
            **fit_info,
        }

        if wandb_run is not None:
            for k, value in scores.items():
                key = ("heldout/matched_likelihood" if k == "matched"
                       else f"heldout/few_shot_k{k}_likelihood")
                wandb_run.summary[key] = float(value)
            # Cross-model parity. The GRU's scaling y-axis is a held-out fine-tune on each
            # subject's train sessions scored on its eval sessions, and the per-mouse MLE
            # baseline uses the same split. Our matched rung is that same protocol, so it is
            # what belongs under the shared keys; the k sweep is reported separately.
            if "matched" in scores:
                wandb_run.summary["heldout/eval_likelihood"] = float(scores["matched"])
                wandb_run.summary["heldout/test_likelihood"] = float(scores["matched"])
                wandb_run.summary["heldout_test_likelihood"] = float(scores["matched"])
            wandb_run.summary["heldout/num_test_trials"] = int(matched_trials)
            wandb_run.summary["heldout/num_test_subjects"] = int(len(per_subject_matched))
            # Per-subject scores as a table: the existing GRU-vs-MLE claim is a paired
            # per-mouse test, which needs the per-subject numbers rather than the aggregate.
            _log_per_subject_table(wandb_run, per_subject_matched)
            wandb_run.summary["hb/fit_seconds"] = float(fit_seconds)
            wandb_run.summary["hb/estimator"] = estimator
            if fit_info.get("divergences") is not None:
                wandb_run.summary["hb/divergences"] = int(fit_info["divergences"])

        # Figures for the fit that just cost hours, built before this process exits. A run
        # that reports a likelihood and no diagnostics cannot be judged: divergences and
        # ESS decide whether its number may be quoted at all. Regenerating them later means
        # reloading the artifact somewhere the bayes extra is installed, so the cheap moment
        # is now, while the fit is already in hand. Comparator lines are deliberately not
        # passed -- those are study constants and belong to the study's analysis, not to
        # runtime code that would carry them stale.
        if fit_info.get("artifacts"):
            try:
                from utils.hb_figures import log_hb_figures

                log_hb_figures(
                    fit_info["artifacts"], scores=scores, wandb_run=wandb_run,
                    output_dir=artifact_dir, beta_max=beta_max,
                )
            except Exception:
                logger.exception("Could not generate HB figures; continuing.")

        return output

    def _fit_population(
        self, estimator, choice_arr, reward_arr, valid_mask, session_mask,
        *, num_warmup, num_samples, num_chains, beta_max,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], Any]:
        """Fit the population level by the requested estimator.

        Returns
        -------
        tuple
            Population point estimates, fit diagnostics, and the sampler itself so the
            caller can persist its draws.
        """
        import jax
        from numpyro.infer import MCMC, NUTS

        from aind_dynamic_foraging_models.hierarchical_bayes.model import (
            hattori2019_three_level,
        )
        from aind_dynamic_foraging_models.hierarchical_bayes.two_stage import (
            fit_two_stage,
        )

        key = jax.random.PRNGKey(int(self.seed or 0))

        if estimator == "one_stage":
            mcmc = MCMC(
                # Integrator geometry is configurable because the two diagnostic
                # failures have different cures: divergences want longer warmup or a
                # higher target acceptance, while low ESS on clean geometry is the only
                # one more draws actually fixes.
                NUTS(
                    hattori2019_three_level,
                    target_accept_prob=self.target_accept_prob,
                    max_tree_depth=self.max_tree_depth,
                ),
                num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method="vectorized", progress_bar=False,
            )
            mcmc.run(
                key, choice_arr, reward_arr, valid_mask, session_mask, beta_max=beta_max
            )
            samples = mcmc.get_samples()
            population = {
                name: np.asarray(samples[name]).mean(axis=0)
                for name in (
                    "population_mean", "population_scale",
                    "log_sigma_mean", "log_sigma_spread",
                )
            }
            divergences = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))
            return population, {"divergences": divergences}, mcmc

        if estimator != "two_stage":
            raise ValueError(f"Unknown estimator {estimator!r}; expected two_stage/one_stage.")

        n_params = 5
        subjects = [
            (choice_arr[i][session_mask[i]], reward_arr[i][session_mask[i]],
             valid_mask[i][session_mask[i]])
            for i in range(choice_arr.shape[0])
        ]
        result = fit_two_stage(
            subjects, rng_key=key,
            subject_kwargs=dict(num_warmup=num_warmup, num_samples=num_samples,
                                beta_max=beta_max),
            population_kwargs=dict(num_warmup=num_warmup, num_samples=num_samples),
        )
        samples = result["population_mcmc"].get_samples()
        mean = np.asarray(samples["population_mean"]).mean(axis=0)
        scale = np.asarray(samples["population_scale"]).mean(axis=0)
        population = {
            "population_mean": mean[:n_params],
            "population_scale": scale[:n_params],
            "log_sigma_mean": mean[n_params:],
            "log_sigma_spread": scale[n_params:],
        }
        return population, {"divergences": None}, result["population_mcmc"]
