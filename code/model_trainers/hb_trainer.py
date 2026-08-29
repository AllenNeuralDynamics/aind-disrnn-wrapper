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
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from base.interfaces import ModelTrainer
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
        Choices and rewards, each keyed by subject id, values a list of 1-D arrays.
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
    return choices, rewards


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


def _normalized_likelihood(total_log_lik: float, total_trials: int) -> float:
    """Geometric-mean per-trial likelihood, the metric shared with the neural models."""
    if total_trials == 0:
        return 0.0
    return float(np.exp(total_log_lik / total_trials))


class HBTrainer(ModelTrainer):
    """Fit a hierarchical Bayesian cognitive model and score held-out subjects."""

    def __init__(self, config: Any, seed: Optional[int] = None) -> None:
        """Store configuration and seed.

        Parameters
        ----------
        config : Any
            Model config. Recognised keys: ``estimator`` (``"two_stage"`` or
            ``"one_stage"``), ``num_warmup``, ``num_samples``, ``num_chains``,
            ``beta_max``, ``few_shot_k``.
        seed : int, optional
            Seed for the sampler.
        """
        super().__init__(seed=seed)
        self.config = config

    def _cfg(self, key: str, default: Any) -> Any:
        """Read a config value from either a mapping or an attribute-style object."""
        if isinstance(self.config, Mapping):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

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
            fit_adaptation,
            pointwise_log_predictive_density,
            posterior_predictive_choice_prob,
        )

        started = time.time()
        wandb_run = (loggers or {}).get("wandb")
        estimator = str(self._cfg("estimator", "two_stage"))
        num_warmup = int(self._cfg("num_warmup", 500))
        num_samples = int(self._cfg("num_samples", 500))
        num_chains = int(self._cfg("num_chains", 4))
        beta_max = float(self._cfg("beta_max", 10.0))
        k_values = tuple(self._cfg("few_shot_k", FEW_SHOT_K))

        if bundle.raw is None or len(bundle.raw) == 0:
            raise ValueError("HBTrainer requires bundle.raw with trial-level rows.")

        choices, rewards = _extract_subject_sessions(bundle.raw)
        choice_arr, reward_arr, valid_mask, session_mask, subject_ids = _pad_cohort(
            choices, rewards
        )
        logger.info(
            "HBTrainer: %d subjects, up to %d sessions, up to %d trials (estimator=%s)",
            len(subject_ids), choice_arr.shape[1], choice_arr.shape[2], estimator,
        )

        population, fit_info = self._fit_population(
            estimator, choice_arr, reward_arr, valid_mask, session_mask,
            num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
            beta_max=beta_max,
        )
        fit_seconds = time.time() - started

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

        heldout_choices, heldout_rewards = _extract_subject_sessions(heldout_df)
        scores: Dict[int, float] = {}
        rng_key = jax.random.PRNGKey(int(self.seed or 0))
        for k in k_values:
            total_log_lik, total_trials = 0.0, 0
            for subject, sessions in heldout_choices.items():
                if len(sessions) <= k:
                    continue
                key_fit, key_draw, rng_key = jax.random.split(rng_key, 3)
                context_c = np.stack(sessions[:k]) if k else np.zeros((0, 1), dtype=int)
                context_r = (
                    np.stack(heldout_rewards[subject][:k]) if k
                    else np.zeros((0, 1), dtype=float)
                )
                samples = fit_adaptation(
                    context_c, context_r, population, rng_key=key_fit,
                    num_warmup=num_warmup, num_samples=num_samples, beta_max=beta_max,
                )
                for session_idx in range(k, len(sessions)):
                    prob = posterior_predictive_choice_prob(
                        samples, sessions[session_idx],
                        heldout_rewards[subject][session_idx],
                        rng_key=key_draw, beta_max=beta_max,
                    )
                    log_lik, n = pointwise_log_predictive_density(
                        prob, sessions[session_idx]
                    )
                    total_log_lik += log_lik
                    total_trials += n
            scores[k] = _normalized_likelihood(total_log_lik, total_trials)
            logger.info("HBTrainer: k=%d heldout likelihood %.5f", k, scores[k])

        output = {
            "estimator": estimator,
            "n_subjects": len(subject_ids),
            "fit_seconds": fit_seconds,
            "population": {
                name: np.asarray(value).tolist() for name, value in population.items()
            },
            "heldout_likelihood": scores,
            **fit_info,
        }

        if wandb_run is not None:
            for k, value in scores.items():
                wandb_run.summary[f"heldout/few_shot_k{k}_likelihood"] = float(value)
            # Cross-model parity: the neural models publish this key, so the panels overlay.
            if k_values:
                wandb_run.summary["heldout_test_likelihood"] = float(scores[max(k_values)])
            wandb_run.summary["hb/fit_seconds"] = float(fit_seconds)
            wandb_run.summary["hb/estimator"] = estimator
            if fit_info.get("divergences") is not None:
                wandb_run.summary["hb/divergences"] = int(fit_info["divergences"])

        return output

    def _fit_population(
        self, estimator, choice_arr, reward_arr, valid_mask, session_mask,
        *, num_warmup, num_samples, num_chains, beta_max,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Fit the population level by the requested estimator."""
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
                NUTS(hattori2019_three_level),
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
            return population, {"divergences": divergences}

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
        return population, {"divergences": None}
