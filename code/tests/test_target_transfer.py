from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from evaluation.target_transfer import (
    assert_aligned_test_trial_keys,
    build_binary_trial_predictions,
    summarize_binary_trial_predictions,
)


class TestTargetTransferMetrics(unittest.TestCase):
    def test_session_manifest_scores_only_frozen_eval_sessions(self):
        trial_df = pd.DataFrame(
            {
                "subject_id": ["m1"] * 4,
                "ses_idx": ["m1__adapt", "m1__adapt", "m1__test", "m1__test"],
                "trial": [0, 1, 0, 1],
                "animal_response": [0, 1, 1, 0],
            }
        )
        predictions = build_binary_trial_predictions(
            trial_df,
            {"eval_session_ids": ["m1__test"]},
            probability_choice_1=np.array([0.1, 0.8, 0.75, 0.25]),
            model="gru",
        )

        self.assertEqual(predictions["trial"].tolist(), [0, 1])
        self.assertEqual(predictions["choice"].tolist(), [1, 0])
        np.testing.assert_allclose(predictions["probability_chosen"], [0.75, 0.75])
        summary = summarize_binary_trial_predictions(predictions)
        self.assertEqual(summary["n_trials"], 2)
        self.assertAlmostEqual(summary["normalized_likelihood"], 0.75)
        self.assertAlmostEqual(summary["brier_score"], 0.0625)
        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(
            summary["calibration_columns"],
            ["choice", "probability_choice_1"],
        )

    def test_prefix_manifest_replays_but_does_not_score_prefix(self):
        trial_df = pd.DataFrame(
            {
                "subject_id": ["h1"] * 4,
                "ses_idx": ["h1__main"] * 4,
                "trial": [0, 1, 2, 3],
                "animal_response": [0, 1, 1, 0],
                "external_split_partition": ["adapt", "adapt", "test", "test"],
            }
        )
        predictions = build_binary_trial_predictions(
            trial_df,
            {
                "split_strategy": "within_session_prefix_suffix",
                "trial_partition_column": "external_split_partition",
                "test_trial_partition": "test",
            },
            probability_choice_1=np.array([0.9, 0.1, 0.7, 0.2]),
            model="q_learning",
        )

        self.assertEqual(predictions["trial"].tolist(), [2, 3])
        self.assertEqual(predictions["model"].unique().tolist(), ["q_learning"])

    def test_parity_requires_identical_ordered_trial_keys(self):
        base = pd.DataFrame(
            {
                "subject_id": ["a", "a"],
                "ses_idx": ["s", "s"],
                "trial": [2, 3],
            }
        )
        assert_aligned_test_trial_keys(base, base.copy())
        with self.assertRaisesRegex(ValueError, "not identical and ordered"):
            assert_aligned_test_trial_keys(base, base.iloc[::-1].reset_index(drop=True))

    def test_gru_and_q_outputs_share_the_same_prefix_test_keys(self):
        trial_df = pd.DataFrame(
            {
                "subject_id": ["h1"] * 5,
                "ses_idx": ["h1__main"] * 5,
                "trial": np.arange(5),
                "animal_response": [0, 1, 0, 1, 1],
                "external_split_partition": ["adapt", "adapt", "test", "test", "test"],
            }
        )
        metadata = {
            "split_strategy": "within_session_prefix_suffix",
            "trial_partition_column": "external_split_partition",
            "test_trial_partition": "test",
        }
        gru = build_binary_trial_predictions(
            trial_df,
            metadata,
            probability_choice_1=[0.2, 0.7, 0.4, 0.8, 0.6],
            model="gru",
        )
        q_learning = build_binary_trial_predictions(
            trial_df,
            metadata,
            probability_choice_1=[np.nan, np.nan, 0.3, 0.75, 0.55],
            model="q_learning",
        )
        assert_aligned_test_trial_keys(gru, q_learning)
        self.assertEqual(gru["trial"].tolist(), [2, 3, 4])


if __name__ == "__main__":
    unittest.main()
