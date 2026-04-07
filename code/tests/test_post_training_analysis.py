"""Unit tests for standalone post-training analysis helpers."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from post_training_analysis import generative_analysis
from post_training_analysis.generative_analysis import (
    _parse_simple_yaml,
    compute_switch_stats,
    load_animal_session_history,
    resolve_model_run,
)


class TestPostTrainingAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.example_run_dir = (
            Path(__file__).resolve().parents[1]
            / "ex_model_dir-train10_test3-disrnn-260324"
            / "9"
        )

    def _session(
        self,
        *,
        subject_id: str,
        ses_idx: str,
        choice_history: list[int],
        reward_history: list[int],
        source_ses_idx: str | None = None,
    ) -> dict[str, object]:
        session = {
            "subject_id": subject_id,
            "ses_idx": ses_idx,
            "session_date": "2024-01-01",
            "curriculum_name": "Uncoupled Baiting",
            "choice_history": choice_history,
            "reward_history": reward_history,
            "n_trials": len(choice_history),
        }
        if source_ses_idx is not None:
            session["source_ses_idx"] = source_ses_idx
        return session

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

    def test_load_animal_session_history_uses_snapshot_without_raw_nwb_loading(self):
        resolved = resolve_model_run(
            self.example_run_dir,
            split="train",
            checkpoint_policy="best_eval",
        )

        class _FakeFrameSeries:
            def __init__(self, values):
                self._values = list(values)

            def map(self, func):
                return [func(value) for value in self._values]

        class _FakeSnapshotFrame:
            def __init__(self, data):
                self._data = {key: list(values) for key, values in data.items()}

            def copy(self):
                return _FakeSnapshotFrame(self._data)

            def __len__(self):
                if not self._data:
                    return 0
                first_key = next(iter(self._data))
                return len(self._data[first_key])

            def __getitem__(self, key):
                return _FakeFrameSeries(self._data[key])

            def __setitem__(self, key, value):
                self._data[key] = list(value)

        snapshot_df = _FakeSnapshotFrame(
            {
                "subject_id": ["m1", "m1", "m1"],
                "ses_idx": ["m1_2024-01-01_00-00-00"] * 3,
                "trial": [0, 1, 2],
                "animal_response": [0, 1, 1],
                "earned_reward": [1, 0, 1],
                "curriculum_name": ["Uncoupled Baiting"] * 3,
                "current_stage_actual": ["GRADUATED"] * 3,
            }
        )

        class _FakeResultSeries:
            def __init__(self, values):
                self._values = list(values)

            def tolist(self):
                return list(self._values)

        class _FakeSessionHistory:
            def __len__(self):
                return 1

            def __getitem__(self, key):
                if key != "ses_idx":
                    raise KeyError(key)
                return _FakeResultSeries(["m1_2024-01-01_00-00-00"])

        built_history = _FakeSessionHistory()

        fake_snapshot_module = mock.Mock()
        fake_snapshot_module.load_mice_snapshot.return_value = (snapshot_df, ["m1"])

        def _fake_import_module(name):
            if name == "utils.load_mice_snapshot":
                return fake_snapshot_module
            if name == "load_mice_data":
                raise AssertionError("load_mice_data should not be imported")
            return __import__(name)

        with mock.patch.object(
            generative_analysis, "_build_session_history_dataframe", return_value=built_history
        ) as build_history_mock, mock.patch.object(
            generative_analysis.importlib,
            "import_module",
            side_effect=_fake_import_module,
        ):
            session_history = load_animal_session_history(resolved)

        fake_snapshot_module.load_mice_snapshot.assert_called_once()
        build_history_mock.assert_called_once()
        snapshot_arg = build_history_mock.call_args.args[0]
        self.assertIsInstance(snapshot_arg, _FakeSnapshotFrame)
        self.assertIs(session_history, built_history)
        self.assertEqual(resolved.resolved_subject_ids, ["m1"])
        self.assertEqual(resolved.resolved_session_ids, ["m1_2024-01-01_00-00-00"])

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

    def test_compute_switch_stats_subject_level_points_match_subjects(self):
        animal_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1",
                choice_history=[0, 1, 0, 1, 0, 0, 1],
                reward_history=[0, 1, 0, 1, 0, 0, 1],
            ),
            self._session(
                subject_id="m2",
                ses_idx="m2_s1",
                choice_history=[0, 0, 1, 1, 0, 0, 1, 1],
                reward_history=[0, 0, 1, 0, 0, 0, 1, 0],
            ),
            self._session(
                subject_id="m3",
                ses_idx="m3_s1",
                choice_history=[0, 1, 0, 1, 0],
                reward_history=[0, 1, 1, 1, 1],
            ),
        ]
        simulated_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_sim_s1",
                choice_history=[0, 1, 0, 0, 1, 0, 0],
                reward_history=[0, 1, 0, 0, 1, 0, 0],
                source_ses_idx="m1_s1",
            ),
            self._session(
                subject_id="m2",
                ses_idx="m2_sim_s1",
                choice_history=[0, 0, 1, 0, 0, 1, 1, 0],
                reward_history=[0, 0, 1, 0, 0, 1, 0, 0],
                source_ses_idx="m2_s1",
            ),
        ]

        stats = compute_switch_stats(
            animal_sessions=animal_sessions,
            simulated_sessions=simulated_sessions,
            window_size=1,
        )

        rewarded_points = {
            point["subject_id"]: point
            for point in stats["subject_level"]["post_switch_by_reward"]["rewarded"][
                "points"
            ]
        }
        self.assertEqual(set(rewarded_points), {"m1", "m2"})
        self.assertEqual(rewarded_points["m1"]["animal_probability"], 1.0)
        self.assertEqual(rewarded_points["m1"]["animal_n"], 2)
        self.assertEqual(rewarded_points["m1"]["animal_n_sessions"], 1)
        self.assertEqual(rewarded_points["m1"]["simulated_probability"], 1.0)
        self.assertEqual(rewarded_points["m1"]["simulated_effective_n"], 2)
        self.assertEqual(rewarded_points["m1"]["simulated_n_source_sessions"], 1)
        self.assertEqual(rewarded_points["m2"]["animal_probability"], 0.0)
        self.assertEqual(rewarded_points["m2"]["simulated_probability"], 0.5)

        rewarded_run1_points = {
            point["subject_id"]: point
            for point in stats["subject_level"]["post_switch_by_reward_and_run_length"][
                "rewarded"
            ]["run_length_1"]["points"]
        }
        self.assertEqual(rewarded_run1_points["m1"]["animal_probability"], 1.0)
        self.assertEqual(rewarded_run1_points["m1"]["animal_n"], 2)
        self.assertEqual(rewarded_run1_points["m1"]["simulated_probability"], 1.0)
        self.assertEqual(rewarded_run1_points["m2"]["animal_probability"], None)
        self.assertEqual(rewarded_run1_points["m2"]["animal_n"], 0)

        rewarded_rungt1_points = {
            point["subject_id"]: point
            for point in stats["subject_level"]["post_switch_by_reward_and_run_length"][
                "rewarded"
            ]["run_length_gt1"]["points"]
        }
        self.assertEqual(rewarded_rungt1_points["m1"]["animal_probability"], None)
        self.assertEqual(rewarded_rungt1_points["m2"]["animal_probability"], 0.0)
        self.assertEqual(rewarded_rungt1_points["m2"]["simulated_probability"], 0.5)

        rewarded_summary = stats["subject_level"]["post_switch_by_reward"]["rewarded"][
            "summary"
        ]
        self.assertEqual(rewarded_summary["n_subjects"], 0)
        self.assertIsNone(rewarded_summary["correlation"])

    def test_compute_switch_stats_subject_level_rollouts_are_normalized_by_source_session(self):
        animal_sessions = [
            self._session(
                subject_id="m4",
                ses_idx="m4_s1",
                choice_history=[0, 1, 0, 1, 0, 1, 0, 1],
                reward_history=[0, 1, 1, 1, 1, 1, 1, 0],
            )
        ]
        simulated_sessions = [
            self._session(
                subject_id="m4",
                ses_idx="m4_s1__rollout_0",
                source_ses_idx="m4_s1",
                choice_history=[0, 1, 0],
                reward_history=[0, 1, 0],
            ),
            self._session(
                subject_id="m4",
                ses_idx="m4_s1__rollout_1",
                source_ses_idx="m4_s1",
                choice_history=[0, 1, 1],
                reward_history=[0, 1, 0],
            ),
            self._session(
                subject_id="m4",
                ses_idx="m4_s2__rollout_0",
                source_ses_idx="m4_s2",
                choice_history=[0, 1, 0],
                reward_history=[0, 1, 0],
            ),
        ]

        stats = compute_switch_stats(
            animal_sessions=animal_sessions,
            simulated_sessions=simulated_sessions,
            window_size=1,
        )

        rewarded_point = stats["subject_level"]["post_switch_by_reward"]["rewarded"][
            "points"
        ][0]
        self.assertEqual(rewarded_point["subject_id"], "m4")
        self.assertEqual(rewarded_point["animal_n"], 6)
        self.assertEqual(rewarded_point["simulated_effective_n"], 2)
        self.assertEqual(rewarded_point["simulated_n_source_sessions"], 2)
        self.assertAlmostEqual(rewarded_point["simulated_probability"], 0.75)

        rewarded_summary = stats["subject_level"]["post_switch_by_reward"]["rewarded"][
            "summary"
        ]
        self.assertEqual(rewarded_summary["n_subjects"], 1)
        self.assertIsNone(rewarded_summary["correlation"])
        self.assertAlmostEqual(rewarded_summary["rmse"], 0.25)
        self.assertAlmostEqual(rewarded_summary["bias"], -0.25)

    def test_save_switch_figures_includes_subject_level_scatter_plots(self):
        try:
            import matplotlib.pyplot  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed")

        animal_sessions = [
            self._session(
                subject_id="m4",
                ses_idx="m4_s1",
                choice_history=[0, 1, 0, 1, 0, 1, 0, 1],
                reward_history=[0, 1, 1, 1, 1, 1, 1, 0],
            )
        ]
        simulated_sessions = [
            self._session(
                subject_id="m4",
                ses_idx="m4_s1__rollout_0",
                source_ses_idx="m4_s1",
                choice_history=[0, 1, 0],
                reward_history=[0, 1, 0],
            )
        ]
        stats = compute_switch_stats(
            animal_sessions=animal_sessions,
            simulated_sessions=simulated_sessions,
            window_size=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            figure_paths = generative_analysis._save_switch_figures(
                switch_stats=stats,
                output_dir=Path(tmpdir),
            )
            self.assertIn("post_switch_by_reward_subject_scatter", figure_paths)
            self.assertIn(
                "post_switch_by_reward_and_run_length_subject_scatter",
                figure_paths,
            )
            self.assertTrue(
                figure_paths["post_switch_by_reward_subject_scatter"]
                .name.endswith(".png")
            )
            self.assertTrue(
                figure_paths[
                    "post_switch_by_reward_and_run_length_subject_scatter"
                ].name.endswith(".png")
            )
            self.assertTrue(
                figure_paths["post_switch_by_reward_subject_scatter"].exists()
            )
            self.assertTrue(
                figure_paths[
                    "post_switch_by_reward_and_run_length_subject_scatter"
                ].exists()
            )


if __name__ == "__main__":
    unittest.main()
