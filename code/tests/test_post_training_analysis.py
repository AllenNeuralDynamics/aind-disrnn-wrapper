"""Unit tests for standalone post-training analysis helpers."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from post_training_analysis import generative_analysis
from post_training_analysis.generative_analysis import (
    _parse_simple_yaml,
    compute_history_dependent_switch_stats,
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

    def test_compute_history_dependent_switch_stats_encodes_detailed_and_abstract_patterns(self):
        stats = compute_history_dependent_switch_stats(
            animal_sessions=[
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1",
                    choice_history=[0, 0, 1, 1, 0],
                    reward_history=[1, 0, 1, 0, 1],
                )
            ],
            simulated_sessions=[],
        )

        animal_detailed = stats["animal"]["detailed"]
        animal_abstract = stats["animal"]["abstract"]

        self.assertEqual(animal_detailed[1]["L"]["total"], 1)
        self.assertEqual(animal_detailed[1]["L"]["switch_probability"], 0.0)
        self.assertEqual(animal_detailed[1]["l"]["total"], 1)
        self.assertEqual(animal_detailed[1]["l"]["switch_probability"], 1.0)
        self.assertEqual(animal_detailed[1]["R"]["total"], 1)
        self.assertEqual(animal_detailed[1]["R"]["switch_probability"], 0.0)
        self.assertEqual(animal_detailed[1]["r"]["total"], 1)
        self.assertEqual(animal_detailed[1]["r"]["switch_probability"], 1.0)

        self.assertEqual(animal_abstract[1]["A"]["total"], 2)
        self.assertEqual(animal_abstract[1]["A"]["switch_probability"], 0.0)
        self.assertEqual(animal_abstract[1]["a"]["total"], 2)
        self.assertEqual(animal_abstract[1]["a"]["switch_probability"], 1.0)
        self.assertEqual(animal_abstract[2]["Aa"]["total"], 2)
        self.assertEqual(animal_abstract[2]["Aa"]["switch_probability"], 1.0)
        self.assertEqual(animal_abstract[2]["aB"]["total"], 1)
        self.assertEqual(animal_abstract[2]["aB"]["switch_probability"], 0.0)
        self.assertEqual(animal_abstract[3]["AaB"]["total"], 1)
        self.assertEqual(animal_abstract[3]["AaB"]["switch_probability"], 0.0)
        self.assertEqual(animal_abstract[3]["aBb"]["total"], 1)
        self.assertEqual(animal_abstract[3]["aBb"]["switch_probability"], 1.0)

    def test_compute_history_dependent_switch_stats_builds_aggregate_comparison_rows(self):
        stats = compute_history_dependent_switch_stats(
            animal_sessions=[
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1",
                    choice_history=[0, 0, 1, 1, 0],
                    reward_history=[1, 0, 1, 0, 1],
                )
            ],
            simulated_sessions=[
                self._session(
                    subject_id="m1",
                    ses_idx="m1_sim_s1",
                    source_ses_idx="m1_s1",
                    choice_history=[0, 0, 1, 0, 0],
                    reward_history=[1, 0, 1, 0, 1],
                )
            ],
            aggregate_min_trials=2,
        )

        rows = {
            row["pattern"]: row
            for row in stats["comparison"]["abstract"][1]["rows"]
        }
        self.assertEqual(set(rows), {"A", "a"})
        self.assertEqual(rows["A"]["animal_total"], 2)
        self.assertEqual(rows["A"]["simulated_total_effective"], 2)
        self.assertEqual(rows["A"]["animal_probability"], 0.0)
        self.assertEqual(rows["A"]["simulated_probability"], 0.5)
        self.assertEqual(rows["A"]["delta_probability"], 0.5)
        self.assertEqual(rows["a"]["animal_probability"], 1.0)
        self.assertEqual(rows["a"]["simulated_probability"], 0.5)
        self.assertEqual(
            stats["comparison"]["abstract"][1]["summary"]["n_patterns"],
            2,
        )

        filtered = compute_history_dependent_switch_stats(
            animal_sessions=[
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1",
                    choice_history=[0, 0, 1, 1, 0],
                    reward_history=[1, 0, 1, 0, 1],
                )
            ],
            simulated_sessions=[
                self._session(
                    subject_id="m1",
                    ses_idx="m1_sim_s1",
                    source_ses_idx="m1_s1",
                    choice_history=[0, 0, 1, 0, 0],
                    reward_history=[1, 0, 1, 0, 1],
                )
            ],
            aggregate_min_trials=3,
        )
        self.assertEqual(
            filtered["comparison"]["abstract"][1]["summary"]["n_patterns"],
            0,
        )
        self.assertIsNone(
            filtered["comparison"]["abstract"][1]["summary"]["correlation"]
        )

    def test_compute_history_dependent_switch_stats_normalizes_rollouts_by_source_session(self):
        stats = compute_history_dependent_switch_stats(
            animal_sessions=[],
            simulated_sessions=[
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1__rollout_0",
                    source_ses_idx="m1_s1",
                    choice_history=[0, 0, 1],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1__rollout_1",
                    source_ses_idx="m1_s1",
                    choice_history=[0, 0, 0],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s2__rollout_0",
                    source_ses_idx="m1_s2",
                    choice_history=[0, 0, 1],
                    reward_history=[1, 0, 0],
                ),
            ],
        )

        abstract_a = stats["simulated"]["abstract"][1]["a"]
        self.assertEqual(abstract_a["total"], 2)
        self.assertEqual(abstract_a["switches"], 1.5)
        self.assertEqual(abstract_a["stays"], 0.5)
        self.assertAlmostEqual(abstract_a["switch_probability"], 0.75)
        self.assertAlmostEqual(
            abstract_a["switch_probability_sem"],
            math.sqrt(0.75 * 0.25 / 2.0),
        )

    def test_compute_history_dependent_switch_stats_builds_subject_level_points(self):
        stats = compute_history_dependent_switch_stats(
            animal_sessions=[
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1",
                    choice_history=[0, 0, 1],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s2",
                    choice_history=[0, 0, 1],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m2",
                    ses_idx="m2_s1",
                    choice_history=[0, 0, 0],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m2",
                    ses_idx="m2_s2",
                    choice_history=[0, 0, 0],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m3",
                    ses_idx="m3_s1",
                    choice_history=[0, 0, 1],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m3",
                    ses_idx="m3_s2",
                    choice_history=[0, 0, 1],
                    reward_history=[1, 0, 0],
                ),
            ],
            simulated_sessions=[
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1__rollout_0",
                    source_ses_idx="m1_s1",
                    choice_history=[0, 0, 1],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1__rollout_1",
                    source_ses_idx="m1_s1",
                    choice_history=[0, 0, 0],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s2__rollout_0",
                    source_ses_idx="m1_s2",
                    choice_history=[0, 0, 1],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m2",
                    ses_idx="m2_s1__rollout_0",
                    source_ses_idx="m2_s1",
                    choice_history=[0, 0, 0],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m2",
                    ses_idx="m2_s2__rollout_0",
                    source_ses_idx="m2_s2",
                    choice_history=[0, 0, 0],
                    reward_history=[1, 0, 0],
                ),
            ],
            subject_min_trials=2,
        )

        points = {
            point["subject_id"]: point
            for point in stats["subject_level"]["abstract"][1]["a"]["points"]
        }
        self.assertEqual(set(points), {"m1", "m2"})
        self.assertEqual(points["m1"]["animal_probability"], 1.0)
        self.assertEqual(points["m1"]["animal_total"], 2)
        self.assertEqual(points["m1"]["animal_n_sessions"], 2)
        self.assertAlmostEqual(points["m1"]["simulated_probability"], 0.75)
        self.assertEqual(points["m1"]["simulated_total_effective"], 2)
        self.assertEqual(points["m1"]["simulated_n_source_sessions"], 2)
        self.assertEqual(points["m2"]["animal_probability"], 0.0)
        self.assertEqual(points["m2"]["simulated_probability"], 0.0)

        summary = stats["subject_level"]["abstract"][1]["a"]["summary"]
        self.assertEqual(summary["n_subjects"], 2)
        self.assertAlmostEqual(summary["correlation"], 1.0)
        self.assertAlmostEqual(summary["rmse"], math.sqrt(0.03125))

    def test_save_history_dependent_switch_figures_creates_expected_outputs(self):
        try:
            import matplotlib.pyplot  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed")

        stats = compute_history_dependent_switch_stats(
            animal_sessions=[
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1",
                    choice_history=[0, 0, 1],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s2",
                    choice_history=[0, 0, 1],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m2",
                    ses_idx="m2_s1",
                    choice_history=[0, 0, 0],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m2",
                    ses_idx="m2_s2",
                    choice_history=[0, 0, 0],
                    reward_history=[1, 0, 0],
                ),
            ],
            simulated_sessions=[
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1__rollout_0",
                    source_ses_idx="m1_s1",
                    choice_history=[0, 0, 1],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1__rollout_1",
                    source_ses_idx="m1_s1",
                    choice_history=[0, 0, 0],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s2__rollout_0",
                    source_ses_idx="m1_s2",
                    choice_history=[0, 0, 1],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m2",
                    ses_idx="m2_s1__rollout_0",
                    source_ses_idx="m2_s1",
                    choice_history=[0, 0, 0],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m2",
                    ses_idx="m2_s2__rollout_0",
                    source_ses_idx="m2_s2",
                    choice_history=[0, 0, 0],
                    reward_history=[1, 0, 0],
                ),
            ],
            aggregate_min_trials=2,
            subject_min_trials=2,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            figure_paths = generative_analysis._save_history_dependent_switch_figures(
                history_stats=stats,
                output_dir=Path(tmpdir),
            )
            self.assertIn("history_pattern_comparison_abstract", figure_paths)
            self.assertIn("history_pattern_subject_level_abstract_nback_1", figure_paths)
            self.assertIn("history_pattern_subject_level_abstract_nback_2", figure_paths)
            self.assertIn("history_pattern_subject_level_abstract_nback_3", figure_paths)
            for path in figure_paths.values():
                self.assertTrue(path.name.endswith(".png"))
                self.assertTrue(path.exists())

    def test_run_post_training_analysis_saves_history_outputs(self):
        resolved_run = generative_analysis.ResolvedModelRun(
            model_dir="/tmp/model",
            inputs_path="/tmp/model/inputs.yaml",
            outputs_dir="/tmp/model/outputs",
            model_type="disrnn",
            split="train",
            checkpoint_policy="best_eval",
            checkpoint_step=100,
            checkpoint_label="step_100",
            params_path="/tmp/model/outputs/checkpoints/step_100/params.json",
            config_path="/tmp/model/outputs/disrnn_config.json",
            seed=123,
            multisubject=True,
            mature_only=True,
            ignore_policy="none",
            curricula=["Uncoupled Baiting"],
            features=None,
            selection={"subject_start": 0, "subject_end": 1},
        )
        animal_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1",
                choice_history=[0, 0, 1],
                reward_history=[1, 0, 0],
            )
        ]
        simulated_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1__rollout_0",
                source_ses_idx="m1_s1",
                choice_history=[0, 0, 1],
                reward_history=[1, 0, 0],
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            switch_figure = Path(tmpdir) / "switch.png"
            history_figure = Path(tmpdir) / "history.png"
            switch_figure.write_text("switch")
            history_figure.write_text("history")

            with mock.patch.object(
                generative_analysis,
                "resolve_model_run",
                return_value=resolved_run,
            ), mock.patch.object(
                generative_analysis,
                "load_animal_session_history",
                return_value=animal_sessions,
            ), mock.patch.object(
                generative_analysis,
                "simulate_model_sessions",
                return_value=simulated_sessions,
            ), mock.patch.object(
                generative_analysis,
                "_save_switch_figures",
                return_value={"pooled_switch_probability": switch_figure},
            ), mock.patch.object(
                generative_analysis,
                "_save_history_dependent_switch_figures",
                return_value={"history_pattern_comparison_abstract": history_figure},
            ):
                result = generative_analysis.run_post_training_analysis(
                    model_dir="/tmp/model",
                    output_dir=Path(tmpdir) / "analysis",
                )

            self.assertIn("history_dependent_switch_stats", result)
            self.assertIn(
                "history_pattern_comparison_abstract",
                result["figure_paths"],
            )
            self.assertTrue(Path(result["history_dependent_switch_stats"]).exists())
            self.assertTrue(Path(result["switch_stats"]).exists())

            history_payload = json.loads(
                Path(result["history_dependent_switch_stats"]).read_text()
            )
            self.assertIn("figure_paths", history_payload)
            self.assertEqual(
                history_payload["figure_paths"]["history_pattern_comparison_abstract"],
                str(history_figure),
            )


if __name__ == "__main__":
    unittest.main()
