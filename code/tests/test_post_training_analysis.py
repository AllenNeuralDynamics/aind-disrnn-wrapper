"""Unit tests for standalone post-training analysis helpers."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from post_training_analysis.generative_analysis import (
    _parse_simple_yaml,
    compute_switch_stats,
    resolve_model_run,
)


class TestPostTrainingAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.example_run_dir = (
            Path(__file__).resolve().parents[1]
            / "ex_model_dir-train10_test3-disrnn-260324"
            / "9"
        )

    def test_parse_simple_yaml_handles_saved_hydra_inputs(self):
        parsed = _parse_simple_yaml(
            """
data:
  subject_start: 0
  subject_end: 10
  mature_only: true
  curricula:
  - Uncoupled Without Baiting
  - Uncoupled Baiting
  - Coupled Baiting
model:
  type: disrnn
  training:
    lr: 0.001
    n_steps: 30000
"""
        )

        self.assertEqual(parsed["data"]["subject_start"], 0)
        self.assertEqual(parsed["data"]["subject_end"], 10)
        self.assertTrue(parsed["data"]["mature_only"])
        self.assertEqual(
            parsed["data"]["curricula"],
            [
                "Uncoupled Without Baiting",
                "Uncoupled Baiting",
                "Coupled Baiting",
            ],
        )
        self.assertEqual(parsed["model"]["type"], "disrnn")
        self.assertEqual(parsed["model"]["training"]["n_steps"], 30000)

    def test_resolve_model_run_best_eval_uses_checkpoint_index(self):
        resolved = resolve_model_run(
            self.example_run_dir,
            split="train",
            checkpoint_policy="best_eval",
        )

        self.assertEqual(resolved.model_type, "disrnn")
        self.assertEqual(resolved.split, "train")
        self.assertEqual(resolved.checkpoint_step, 24000)
        self.assertEqual(resolved.selection["subject_start"], 0)
        self.assertEqual(resolved.selection["subject_end"], 10)
        self.assertTrue(resolved.params_path.endswith("step_24000/params.json"))
        self.assertIsNone(resolved.fallback_reason)

    def test_resolve_model_run_best_heldout_uses_output_summary(self):
        resolved = resolve_model_run(
            self.example_run_dir,
            split="train",
            checkpoint_policy="best_heldout",
        )

        self.assertEqual(resolved.checkpoint_step, 6000)
        self.assertTrue(resolved.params_path.endswith("step_6000/params.json"))
        self.assertIsNone(resolved.fallback_reason)

    def test_resolve_model_run_final_uses_top_level_params(self):
        resolved = resolve_model_run(
            self.example_run_dir,
            split="train",
            checkpoint_policy="final",
        )

        self.assertEqual(resolved.checkpoint_step, 30000)
        self.assertTrue(resolved.params_path.endswith("outputs/params.json"))
        self.assertEqual(resolved.checkpoint_label, "final")

    def test_compute_switch_stats_matches_expected_reward_conditioning(self):
        animal_sessions = [
            {
                "subject_id": "m1",
                "ses_idx": "m1_2024-01-01_00-00-00",
                "session_date": "2024-01-01",
                "curriculum_name": "Uncoupled Baiting",
                "choice_history": [0, 0, 1, 0, 0],
                "reward_history": [0, 1, 1, 0, 1],
                "n_trials": 5,
            }
        ]
        simulated_sessions = [
            {
                "subject_id": "m1",
                "ses_idx": "m1_2024-01-01_00-00-00",
                "session_date": "2024-01-01",
                "curriculum_name": "Uncoupled Baiting",
                "choice_history": [0, 1, 1, 0, 0],
                "reward_history": [0, 0, 1, 1, 0],
                "n_trials": 5,
            }
        ]

        stats = compute_switch_stats(
            animal_sessions=animal_sessions,
            simulated_sessions=simulated_sessions,
            window_size=1,
        )

        animal_rewarded = stats["animal"]["post_switch_by_reward"]["rewarded"]
        animal_unrewarded = stats["animal"]["post_switch_by_reward"]["unrewarded"]
        simulated_rewarded = stats["simulated"]["post_switch_by_reward"]["rewarded"]
        simulated_unrewarded = stats["simulated"]["post_switch_by_reward"]["unrewarded"]

        self.assertEqual(animal_rewarded["n"], 1)
        self.assertEqual(animal_rewarded["probability"], 1.0)
        self.assertEqual(animal_unrewarded["n"], 1)
        self.assertEqual(animal_unrewarded["probability"], 0.0)
        self.assertEqual(simulated_unrewarded["n"], 1)
        self.assertEqual(simulated_unrewarded["probability"], 0.0)
        self.assertEqual(simulated_rewarded["n"], 1)
        self.assertEqual(simulated_rewarded["probability"], 0.0)

    def test_compute_switch_stats_handles_sessions_without_switches(self):
        animal_sessions = [
            {
                "subject_id": "m1",
                "ses_idx": "m1_2024-01-01_00-00-00",
                "session_date": "2024-01-01",
                "curriculum_name": "Uncoupled Baiting",
                "choice_history": [0, 0, 0],
                "reward_history": [1, 0, 1],
                "n_trials": 3,
            }
        ]
        simulated_sessions = [
            {
                "subject_id": "m1",
                "ses_idx": "m1_2024-01-01_00-00-00",
                "session_date": "2024-01-01",
                "curriculum_name": "Uncoupled Baiting",
                "choice_history": [1, 1, 1],
                "reward_history": [0, 0, 0],
                "n_trials": 3,
            }
        ]

        stats = compute_switch_stats(
            animal_sessions=animal_sessions,
            simulated_sessions=simulated_sessions,
            window_size=1,
        )

        self.assertEqual(stats["animal"]["pooled_switch_probability"], [])
        self.assertEqual(stats["simulated"]["pooled_switch_probability"], [])
        self.assertEqual(
            stats["comparison"]["post_switch_by_reward"]["rewarded"]["animal_probability"],
            None,
        )


if __name__ == "__main__":
    unittest.main()
