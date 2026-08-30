"""HBTrainer data marshalling and end-to-end fit."""

import unittest

import numpy as np
import pandas as pd

from base.types import DatasetBundle
from model_trainers.hb_trainer import (
    _extract_subject_sessions,
    _normalized_likelihood,
    _pad_cohort,
)

try:
    import jax  # noqa: F401
    import numpyro  # noqa: F401

    from model_trainers.hb_trainer import HBTrainer

    HAS_BAYES = True
except ImportError:  # pragma: no cover - exercised only without the bayes extra
    HAS_BAYES = False


def _make_frame(n_subjects=3, n_sessions=4, n_trials=60, ignore_every=None, seed=0):
    """Build a trial-level frame shaped like the wrapper's raw dataframe."""
    rng = np.random.default_rng(seed)
    rows = []
    for subject in range(n_subjects):
        for session in range(n_sessions):
            for trial in range(n_trials):
                response = int(rng.integers(0, 2))
                if ignore_every and trial % ignore_every == 0:
                    response = 2  # ignored trial
                rows.append({
                    "subject_id": f"m{subject}",
                    "ses_idx": f"m{subject}_s{session}",
                    "trial": trial,
                    "animal_response": response,
                    "earned_reward": float(rng.integers(0, 2)),
                })
    return pd.DataFrame(rows)


class TestExtraction(unittest.TestCase):
    """Grouping trials into per-subject sessions."""

    def test_groups_by_subject_and_session(self):
        """Every subject gets one array per session."""
        choices, rewards, _ = _extract_subject_sessions(_make_frame())
        self.assertEqual(sorted(choices.keys()), ["m0", "m1", "m2"])
        self.assertEqual(len(choices["m0"]), 4)
        self.assertEqual(len(choices["m0"][0]), 60)
        self.assertEqual(len(rewards["m0"][0]), 60)

    def test_drops_ignored_trials(self):
        """Trials with animal_response outside {0,1} are excluded, as baseline_rl does."""
        choices, _, _ = _extract_subject_sessions(_make_frame(ignore_every=3))
        session = choices["m0"][0]
        self.assertEqual(len(session), 40)  # 60 trials less every third
        self.assertTrue(np.all((session == 0) | (session == 1)))

    def test_returns_session_ids_in_frame_order(self):
        """Session ids come back ordered, so a split matches the neural models' own."""
        _, _, ids = _extract_subject_sessions(_make_frame(n_subjects=2, n_sessions=3))
        self.assertEqual(ids["m0"], ["m0_s0", "m0_s1", "m0_s2"])

    def test_missing_columns_raise(self):
        """A frame without the required columns fails loudly."""
        with self.assertRaises(ValueError):
            _extract_subject_sessions(pd.DataFrame({"subject_id": ["m0"]}))


class TestPadding(unittest.TestCase):
    """Padding ragged cohorts into dense arrays."""

    def test_shapes_and_masks(self):
        """Ragged subjects pad to a common shape with masks marking the real entries."""
        choices = {"a": [np.zeros(10, int), np.zeros(6, int)], "b": [np.zeros(8, int)]}
        rewards = {"a": [np.zeros(10), np.zeros(6)], "b": [np.zeros(8)]}
        c, r, valid, session_mask, ids = _pad_cohort(choices, rewards)

        self.assertEqual(c.shape, (2, 2, 10))
        self.assertEqual(session_mask.tolist(), [[True, True], [True, False]])
        self.assertEqual(int(valid[0, 0].sum()), 10)
        self.assertEqual(int(valid[0, 1].sum()), 6)
        self.assertEqual(int(valid[1, 0].sum()), 8)
        self.assertEqual(int(valid[1, 1].sum()), 0)  # padded session contributes nothing
        self.assertEqual(ids, ["a", "b"])

    def test_never_truncates(self):
        """The longest session survives intact; padding only ever adds."""
        choices = {"a": [np.ones(25, int), np.ones(5, int)]}
        rewards = {"a": [np.ones(25), np.ones(5)]}
        c, _, valid, _, _ = _pad_cohort(choices, rewards)
        self.assertEqual(c.shape[2], 25)
        self.assertEqual(int(valid[0, 0].sum()), 25)


class TestNormalizedLikelihood(unittest.TestCase):
    """The metric shared with the neural models."""

    def test_geometric_mean(self):
        """exp(sum log p / n) matches a hand-computed geometric mean."""
        probs = np.array([0.8, 0.5, 0.4])
        expected = float(np.exp(np.sum(np.log(probs)) / 3))
        self.assertAlmostEqual(
            _normalized_likelihood(float(np.sum(np.log(probs))), 3), expected, places=10
        )

    def test_empty_is_zero(self):
        """No scored trials yields zero rather than a division error."""
        self.assertEqual(_normalized_likelihood(0.0, 0), 0.0)


@unittest.skipUnless(HAS_BAYES, "requires the 'bayes' extra (jax, numpyro)")
class TestFit(unittest.TestCase):
    """End-to-end fit through the ModelTrainer interface."""

    def test_skips_heldout_without_explicit_cohort(self):
        """Absent a held-out frame the trainer skips scoring rather than leaking."""
        bundle = DatasetBundle(
            raw=_make_frame(n_subjects=3, n_sessions=3, n_trials=50),
            train_set=None, eval_set=None, metadata={},
        )
        trainer = HBTrainer(
            estimator="two_stage", num_warmup=30, num_samples=30, num_chains=1, seed=0,
        )
        output = trainer.fit(bundle)
        self.assertTrue(output["heldout_skipped"])
        self.assertEqual(output["heldout_likelihood"], {})
        self.assertEqual(len(output["population"]["population_mean"]), 5)

    def test_one_stage_scores_heldout(self):
        """With a held-out cohort supplied, every k yields a likelihood in (0, 1)."""
        bundle = DatasetBundle(
            raw=_make_frame(n_subjects=3, n_sessions=3, n_trials=50, seed=0),
            train_set=None, eval_set=None, metadata={},
            extras={"heldout_raw": _make_frame(n_subjects=2, n_sessions=4,
                                               n_trials=50, seed=7)},
        )
        trainer = HBTrainer(
            estimator="one_stage", num_warmup=30, num_samples=30, num_chains=1,
            few_shot_k=(0, 2), seed=0,
        )
        output = trainer.fit(bundle)
        scores = output["heldout_likelihood"]
        self.assertEqual(sorted(k for k in scores if isinstance(k, int)), [0, 2])
        for value in scores.values():
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0)

    def test_matched_conditioning_uses_the_baseline_split(self):
        """A 'matched' rung is reported alongside the k sweep.

        It conditions on exactly the sessions the per-mouse MLE baseline fits and scores
        exactly the ones it scores, so the two sit on the same footing rather than being
        compared across different amounts of conditioning.
        """
        bundle = DatasetBundle(
            raw=_make_frame(n_subjects=3, n_sessions=3, n_trials=50, seed=0),
            train_set=None, eval_set=None, metadata={},
            extras={"heldout_raw": _make_frame(n_subjects=2, n_sessions=4,
                                               n_trials=50, seed=7)},
        )
        trainer = HBTrainer(
            estimator="one_stage", num_warmup=30, num_samples=30, num_chains=1,
            few_shot_k=(0,), eval_every_n=2, seed=0,
        )
        scores = trainer.fit(bundle)["heldout_likelihood"]
        self.assertIn("matched", scores)
        self.assertGreater(scores["matched"], 0.0)
        self.assertLess(scores["matched"], 1.0)


if __name__ == "__main__":
    unittest.main()
