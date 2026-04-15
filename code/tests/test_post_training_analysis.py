"""Unit tests for standalone post-training analysis helpers."""

from __future__ import annotations

import json
import math
import pickle
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

    def _write_multisubject_run_dir(
        self,
        root: Path,
        *,
        model_type: str = "gru",
        with_subject_map: bool = True,
    ) -> Path:
        model_dir = root / f"{model_type}_multisubject_run"
        outputs_dir = model_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        (model_dir / "inputs.yaml").write_text(
            """
data:
  subject_ids: null
  subject_start: 0
  subject_end: 2
  test_subject_start: 2
  test_subject_end: 3
  mature_only: true
  curricula:
  - Uncoupled Baiting
  multisubject: true
  ignore_policy: exclude
model:
  type: """
            + model_type
            + """
  architecture:
    multisubject: true
seed: 7
"""
        )
        if model_type == "gru":
            (outputs_dir / "gru_config.json").write_text(
                json.dumps(
                    {
                        "architecture": {
                            "multisubject": True,
                            "hidden_size": 8,
                            "subject_embedding_size": 3,
                            "subject_embedding_init": "zeros",
                        },
                        "output_size": 2,
                    }
                )
            )
        else:
            (outputs_dir / "disrnn_config.json").write_text(
                json.dumps(
                    {
                        "obs_size": 2,
                        "output_size": 2,
                        "latent_size": 4,
                        "update_net_n_units_per_layer": 8,
                        "update_net_n_layers": 2,
                        "choice_net_n_units_per_layer": 4,
                        "choice_net_n_layers": 1,
                        "activation": "leaky_relu",
                        "noiseless_mode": False,
                        "latent_penalty": 1e-3,
                        "choice_net_latent_penalty": 1e-3,
                        "update_net_obs_penalty": 1e-3,
                        "update_net_latent_penalty": 1e-3,
                        "max_n_subjects": 2,
                        "subject_embedding_size": 3,
                        "subject_embedding_init": "zeros",
                        "use_global_subject_bottleneck": True,
                        "subj_penalty": 1e-3,
                        "update_net_subj_penalty": 1e-3,
                        "choice_net_subj_penalty": 1e-3,
                    }
                )
            )
        (outputs_dir / "params.json").write_text(json.dumps({}))
        if with_subject_map:
            (outputs_dir / "subject_index_map.json").write_text(
                json.dumps(
                    {
                        "subject_id_to_index": {"m1": 0, "m2": 1},
                        "index_to_subject_id": {"0": "m1", "1": "m2"},
                    }
                )
            )
        return model_dir

    def _session(
        self,
        *,
        subject_id: str,
        ses_idx: str,
        choice_history: list[int],
        reward_history: list[int],
        source_ses_idx: str | None = None,
        curriculum_name: str = "Uncoupled Baiting",
    ) -> dict[str, object]:
        session = {
            "subject_id": subject_id,
            "ses_idx": ses_idx,
            "session_date": "2024-01-01",
            "curriculum_name": curriculum_name,
            "choice_history": choice_history,
            "reward_history": reward_history,
            "n_trials": len(choice_history),
        }
        if source_ses_idx is not None:
            session["source_ses_idx"] = source_ses_idx
        return session

    def _assert_multisubject_simulation_uses_subject_indices(self, model_type: str) -> None:
        resolved_run = generative_analysis.ResolvedModelRun(
            model_dir="/tmp/model",
            inputs_path="/tmp/model/inputs.yaml",
            outputs_dir="/tmp/model/outputs",
            model_type=model_type,
            split="train",
            checkpoint_policy="final",
            checkpoint_step=20,
            checkpoint_label="final",
            params_path="/tmp/model/outputs/params.json",
            config_path=f"/tmp/model/outputs/{model_type}_config.json",
            seed=11,
            multisubject=True,
            mature_only=True,
            ignore_policy="exclude",
            curricula=["Uncoupled Baiting"],
            features=None,
            selection={"subject_start": 0, "subject_end": 2},
            subject_index_map_path="/tmp/model/outputs/subject_index_map.json",
            trained_subject_ids=["m1", "m2"],
        )
        animal_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1",
                choice_history=[0, 0],
                reward_history=[1, 1],
            ),
            self._session(
                subject_id="m2",
                ses_idx="m2_s1",
                choice_history=[0, 0],
                reward_history=[1, 1],
            ),
        ]

        class _FakeRunner:
            def __init__(self) -> None:
                self.initial_state = {"hidden": 0}
                self.n_actions = 2
                self.calls = []
                self._subject_id_to_index = {"m1": 0, "m2": 1}

            def validate_subject_ids(self, subject_ids):
                if list(subject_ids) != ["m1", "m2"]:
                    raise ValueError("unexpected subjects")

            def encode_inputs(self, subject_id, inputs):
                encoded = [float(self._subject_id_to_index[subject_id]), *list(inputs)]
                self.calls.append((subject_id, encoded))
                return encoded

            def step(self, inputs, prev_state):
                self.calls[-1] = (self.calls[-1][0], list(inputs))
                return [50.0, -50.0], prev_state

        fake_runner = _FakeRunner()

        class _FakeRng:
            def choice(self, n_actions, p):
                return 0

        class _FakeNumpy:
            class random:
                @staticmethod
                def default_rng(seed):
                    return _FakeRng()

        class _FakePandas:
            class DataFrame:
                @staticmethod
                def from_records(records):
                    return list(records)

        with mock.patch.object(
            generative_analysis,
            "_restore_model_runner",
            return_value=fake_runner,
        ), mock.patch.object(
            generative_analysis,
            "_build_curriculum_matched_task",
            return_value=mock.Mock(reset=mock.Mock()),
        ), mock.patch.object(
            generative_analysis,
            "_step_task_reward",
            return_value=1,
        ), mock.patch.object(
            generative_analysis,
            "_import_dependency",
            side_effect=lambda module_name: (
                _FakeNumpy
                if module_name == "numpy"
                else _FakePandas
                if module_name == "pandas"
                else getattr(generative_analysis, module_name)
            ),
        ):
            simulated = generative_analysis.simulate_model_sessions(
                resolved_run=resolved_run,
                animal_sessions=animal_sessions,
                n_rollouts_per_session=1,
            )

        self.assertEqual(len(simulated), 2)
        self.assertEqual(
            fake_runner.calls,
            [
                ("m1", [0.0, -1.0, -1.0]),
                ("m1", [0.0, 0.0, 1.0]),
                ("m2", [1.0, -1.0, -1.0]),
                ("m2", [1.0, 0.0, 1.0]),
            ],
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

    def test_resolve_model_run_multisubject_loads_subject_index_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_multisubject_run_dir(Path(tmpdir), model_type="gru")

            resolved = resolve_model_run(
                model_dir,
                split="train",
                checkpoint_policy="final",
            )

        self.assertTrue(resolved.multisubject)
        self.assertTrue(resolved.subject_index_map_path.endswith("subject_index_map.json"))
        self.assertEqual(resolved.trained_subject_ids, ["m1", "m2"])

    def test_resolve_model_run_multisubject_requires_subject_index_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_multisubject_run_dir(
                Path(tmpdir),
                model_type="gru",
                with_subject_map=False,
            )

            with self.assertRaisesRegex(FileNotFoundError, "subject index map"):
                resolve_model_run(
                    model_dir,
                    split="train",
                    checkpoint_policy="final",
                )

    def test_resolve_model_run_multisubject_rejects_heldout_split(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_multisubject_run_dir(Path(tmpdir), model_type="gru")

            with self.assertRaisesRegex(
                NotImplementedError,
                "seen-subject personalization only",
            ):
                resolve_model_run(
                    model_dir,
                    split="heldout",
                    checkpoint_policy="final",
                )

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
            generative_analysis,
            "_align_snapshot_df_with_ignore_policy",
            side_effect=lambda snapshot_df, ignore_policy: snapshot_df,
        ), mock.patch.object(
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

    def test_load_animal_session_history_multisubject_uses_trained_subject_ids(self):
        resolved = generative_analysis.ResolvedModelRun(
            model_dir="/tmp/model",
            inputs_path="/tmp/model/inputs.yaml",
            outputs_dir="/tmp/model/outputs",
            model_type="gru",
            split="train",
            checkpoint_policy="final",
            checkpoint_step=20,
            checkpoint_label="final",
            params_path="/tmp/model/outputs/params.json",
            config_path="/tmp/model/outputs/gru_config.json",
            seed=7,
            multisubject=True,
            mature_only=True,
            ignore_policy="exclude",
            curricula=["Uncoupled Baiting"],
            features=None,
            selection={"subject_start": 0, "subject_end": 2},
            subject_index_map_path="/tmp/model/outputs/subject_index_map.json",
            trained_subject_ids=["m1", "m2"],
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
                return len(self._data[next(iter(self._data))])

            def __getitem__(self, key):
                return _FakeFrameSeries(self._data[key])

            def __setitem__(self, key, value):
                self._data[key] = list(value)

        class _FakeResultSeries:
            def __init__(self, values):
                self._values = list(values)

            def tolist(self):
                return list(self._values)

            def __iter__(self):
                return iter(self._values)

        class _FakeSessionHistory:
            def __len__(self):
                return 2

            def __getitem__(self, key):
                if key == "subject_id":
                    return _FakeResultSeries(["m1", "m2"])
                if key == "ses_idx":
                    return _FakeResultSeries(["m1_s1", "m2_s1"])
                raise KeyError(key)

        snapshot_df = _FakeSnapshotFrame(
            {
                "subject_id": ["m1", "m1", "m2", "m2"],
                "ses_idx": ["m1_s1", "m1_s1", "m2_s1", "m2_s1"],
                "trial": [0, 1, 0, 1],
                "animal_response": [0, 1, 0, 1],
                "earned_reward": [1, 0, 0, 1],
                "curriculum_name": ["Uncoupled Baiting"] * 4,
                "current_stage_actual": ["GRADUATED"] * 4,
            }
        )
        built_history = _FakeSessionHistory()
        fake_snapshot_module = mock.Mock()
        fake_snapshot_module.load_mice_snapshot.return_value = (snapshot_df, ["m1", "m2"])

        def _fake_import_module(name):
            if name == "utils.load_mice_snapshot":
                return fake_snapshot_module
            return __import__(name)

        with mock.patch.object(
            generative_analysis, "_build_session_history_dataframe", return_value=built_history
        ) as build_history_mock, mock.patch.object(
            generative_analysis,
            "_align_snapshot_df_with_ignore_policy",
            side_effect=lambda snapshot_df, ignore_policy: snapshot_df,
        ), mock.patch.object(
            generative_analysis,
            "importlib",
        ) as importlib_mock:
            importlib_mock.import_module.side_effect = _fake_import_module
            session_history = load_animal_session_history(resolved)

        fake_snapshot_module.load_mice_snapshot.assert_called_once_with(
            subject_ids=["m1", "m2"],
            subject_start=None,
            subject_end=None,
            mature_only=True,
            curricula=["Uncoupled Baiting"],
            cols_to_retain=[
                "trial",
                "subject_id",
                "ses_idx",
                "animal_response",
                "earned_reward",
                "curriculum_name",
                "current_stage_actual",
            ],
        )
        build_history_mock.assert_called_once()
        self.assertIs(session_history, built_history)
        self.assertEqual(resolved.resolved_subject_ids, ["m1", "m2"])
        self.assertEqual(resolved.resolved_session_ids, ["m1_s1", "m2_s1"])

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
        self.assertEqual(rewarded_point["animal_n_source_sessions"], 1)
        self.assertEqual(rewarded_point["simulated_effective_n"], 2)
        self.assertEqual(rewarded_point["simulated_n_source_sessions"], 2)
        self.assertAlmostEqual(rewarded_point["simulated_probability"], 0.75)
        self.assertAlmostEqual(rewarded_point["delta_probability"], -0.25)
        self.assertEqual(rewarded_point["curriculum_name"], "Uncoupled Baiting")
        self.assertIsNone(rewarded_point["animal_ci_low"])
        self.assertIsNone(rewarded_point["animal_ci_high"])
        self.assertIsNotNone(rewarded_point["simulated_ci_low"])
        self.assertIsNotNone(rewarded_point["simulated_ci_high"])

        rewarded_summary = stats["subject_level"]["post_switch_by_reward"]["rewarded"][
            "summary"
        ]
        self.assertEqual(rewarded_summary["n_subjects"], 1)
        self.assertIsNone(rewarded_summary["correlation"])
        self.assertAlmostEqual(rewarded_summary["rmse"], 0.25)
        self.assertAlmostEqual(rewarded_summary["bias"], -0.25)

        rewarded_subject_aggregate = stats["subject_aggregate"]["post_switch_by_reward"][
            "rewarded"
        ]
        self.assertEqual(rewarded_subject_aggregate["n_subjects"], 1)
        self.assertAlmostEqual(rewarded_subject_aggregate["animal_mean"], 1.0)
        self.assertAlmostEqual(rewarded_subject_aggregate["simulated_mean"], 0.75)
        self.assertAlmostEqual(rewarded_subject_aggregate["delta_mean"], -0.25)
        self.assertEqual(
            stats["quantitative_summary"]["subject_mean"]["post_switch_by_reward"][
                "n_rows"
            ],
            1,
        )

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
            self.assertIn("post_switch_by_reward_pooled", figure_paths)
            self.assertIn("post_switch_by_reward_subject_scatter", figure_paths)
            self.assertIn("post_switch_delta_by_reward", figure_paths)
            self.assertIn("post_switch_delta_by_reward_and_run_length", figure_paths)
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
        self.assertEqual(points["m1"]["animal_n_source_sessions"], 2)
        self.assertAlmostEqual(points["m1"]["simulated_probability"], 0.75)
        self.assertEqual(points["m1"]["simulated_total_effective"], 2)
        self.assertEqual(points["m1"]["simulated_n_source_sessions"], 2)
        self.assertAlmostEqual(points["m1"]["delta_probability"], -0.25)
        self.assertEqual(points["m1"]["curriculum_name"], "Uncoupled Baiting")
        self.assertIsNotNone(points["m1"]["animal_ci_low"])
        self.assertIsNotNone(points["m1"]["animal_ci_high"])
        self.assertIsNotNone(points["m1"]["simulated_ci_low"])
        self.assertIsNotNone(points["m1"]["simulated_ci_high"])
        self.assertEqual(points["m2"]["animal_probability"], 0.0)
        self.assertEqual(points["m2"]["simulated_probability"], 0.0)

        summary = stats["subject_level"]["abstract"][1]["a"]["summary"]
        self.assertEqual(summary["n_subjects"], 2)
        self.assertAlmostEqual(summary["correlation"], 1.0)
        self.assertAlmostEqual(summary["rmse"], math.sqrt(0.03125))

        aggregate_rows = {
            row["pattern"]: row
            for row in stats["subject_aggregate"]["abstract"][1]["rows"]
        }
        self.assertEqual(aggregate_rows["a"]["n_subjects"], 2)
        self.assertAlmostEqual(aggregate_rows["a"]["animal_mean"], 0.5)
        self.assertAlmostEqual(aggregate_rows["a"]["simulated_mean"], 0.375)
        self.assertAlmostEqual(aggregate_rows["a"]["delta_mean"], -0.125)
        self.assertEqual(
            stats["quantitative_summary"]["subject_mean"]["abstract"][1]["n_rows"],
            2,
        )

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
            self.assertIn("history_pattern_comparison_abstract_pooled", figure_paths)
            self.assertIn("history_pattern_delta_abstract", figure_paths)
            self.assertIn("history_pattern_subject_level_abstract_nback_1", figure_paths)
            self.assertIn("history_pattern_subject_level_abstract_nback_2", figure_paths)
            self.assertIn("history_pattern_subject_level_abstract_nback_3", figure_paths)
            for path in figure_paths.values():
                self.assertTrue(path.name.endswith(".png"))
                self.assertTrue(path.exists())

    def test_build_sorted_history_delta_rows_orders_by_descending_delta_mean(self):
        panel_group = {
            "high": {
                "points": [
                    {
                        "animal_probability": 0.8,
                        "simulated_probability": 0.9,
                        "delta_probability": 0.1,
                        "animal_total": 3,
                        "simulated_total_effective": 3,
                    }
                ]
            },
            "low": {
                "points": [
                    {
                        "animal_probability": 0.2,
                        "simulated_probability": 0.0,
                        "delta_probability": -0.2,
                        "animal_total": 3,
                        "simulated_total_effective": 3,
                    }
                ]
            },
            "mid": {
                "points": [
                    {
                        "animal_probability": 0.5,
                        "simulated_probability": 0.55,
                        "delta_probability": 0.05,
                        "animal_total": 3,
                        "simulated_total_effective": 3,
                    }
                ]
            },
        }

        rows = generative_analysis._build_sorted_history_delta_rows(
            panel_group,
            min_trials=2,
        )

        self.assertEqual([row["label"] for row in rows], ["high", "mid", "low"])

    def test_simulate_model_sessions_multisubject_gru_uses_subject_indices(self):
        self._assert_multisubject_simulation_uses_subject_indices("gru")

    def test_simulate_model_sessions_multisubject_disrnn_uses_subject_indices(self):
        self._assert_multisubject_simulation_uses_subject_indices("disrnn")

    def test_simulate_model_sessions_multisubject_rejects_subjects_missing_from_map(self):
        resolved_run = generative_analysis.ResolvedModelRun(
            model_dir="/tmp/model",
            inputs_path="/tmp/model/inputs.yaml",
            outputs_dir="/tmp/model/outputs",
            model_type="gru",
            split="train",
            checkpoint_policy="final",
            checkpoint_step=20,
            checkpoint_label="final",
            params_path="/tmp/model/outputs/params.json",
            config_path="/tmp/model/outputs/gru_config.json",
            seed=11,
            multisubject=True,
            mature_only=True,
            ignore_policy="exclude",
            curricula=["Uncoupled Baiting"],
            features=None,
            selection={"subject_start": 0, "subject_end": 2},
            subject_index_map_path="/tmp/model/outputs/subject_index_map.json",
            trained_subject_ids=["m1"],
        )
        animal_sessions = [
            self._session(
                subject_id="m3",
                ses_idx="m3_s1",
                choice_history=[0],
                reward_history=[1],
            )
        ]

        class _FakeRunner:
            initial_state = {"hidden": 0}
            n_actions = 2

            def validate_subject_ids(self, subject_ids):
                raise ValueError(
                    "Multisubject post-training analysis encountered subject ids "
                    "that are not present in subject_index_map.json: ['m3']"
                )

        class _FakeRng:
            def choice(self, n_actions, p):
                return 0

        class _FakeNumpy:
            class random:
                @staticmethod
                def default_rng(seed):
                    return _FakeRng()

        class _FakePandas:
            class DataFrame:
                @staticmethod
                def from_records(records):
                    return list(records)

        with mock.patch.object(
            generative_analysis,
            "_restore_model_runner",
            return_value=_FakeRunner(),
        ), mock.patch.object(
            generative_analysis,
            "_import_dependency",
            side_effect=lambda module_name: (
                _FakeNumpy if module_name == "numpy" else _FakePandas
            ),
        ):
            with self.assertRaisesRegex(ValueError, "subject_index_map.json"):
                generative_analysis.simulate_model_sessions(
                    resolved_run=resolved_run,
                    animal_sessions=animal_sessions,
                    n_rollouts_per_session=1,
                )

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
            subject_index_map_path="/tmp/model/outputs/subject_index_map.json",
            trained_subject_ids=["m1"],
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
            resolved_payload = json.loads(Path(result["resolved_run"]).read_text())
            self.assertEqual(
                resolved_payload["subject_index_map_path"],
                "/tmp/model/outputs/subject_index_map.json",
            )
            self.assertEqual(resolved_payload["trained_subject_ids"], ["m1"])

    def test_run_post_training_analysis_from_histories_writes_quantitative_summary(self):
        animal_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1",
                choice_history=[0, 1, 0, 1, 0, 1],
                reward_history=[0, 1, 0, 1, 0, 1],
                curriculum_name="Coupled Baiting",
            )
        ]
        simulated_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1__rollout_0",
                source_ses_idx="m1_s1",
                choice_history=[0, 1, 0, 1, 0, 0],
                reward_history=[0, 1, 0, 1, 0, 0],
                curriculum_name="Coupled Baiting",
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generative_analysis.run_post_training_analysis_from_histories(
                animal_sessions=animal_sessions,
                simulated_sessions=simulated_sessions,
                output_dir=Path(tmpdir),
            )

            self.assertIn("model_vs_animal_quantitative_summary", result)
            summary_payload = json.loads(
                Path(result["model_vs_animal_quantitative_summary"]).read_text()
            )
            self.assertIn("switch_triggered", summary_payload)
            self.assertIn("history_dependent", summary_payload)

            switch_payload = json.loads(Path(result["switch_stats"]).read_text())
            self.assertIn("subject_aggregate", switch_payload)
            self.assertIn("quantitative_summary", switch_payload)

            history_payload = json.loads(
                Path(result["history_dependent_switch_stats"]).read_text()
            )
            self.assertIn("subject_aggregate", history_payload)
            self.assertIn("quantitative_summary", history_payload)

    def test_run_post_training_analysis_from_saved_histories_matches_direct_reanalysis(self):
        animal_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1",
                choice_history=[0, 0, 1, 1, 0],
                reward_history=[1, 0, 1, 0, 1],
            )
        ]
        simulated_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1__rollout_0",
                source_ses_idx="m1_s1",
                choice_history=[0, 0, 1, 0, 0],
                reward_history=[1, 0, 1, 0, 1],
            )
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            direct_result = generative_analysis.run_post_training_analysis_from_histories(
                animal_sessions=animal_sessions,
                simulated_sessions=simulated_sessions,
                output_dir=root / "direct",
            )

            animal_path = root / "animal.pkl"
            simulated_path = root / "simulated.pkl"
            with animal_path.open("wb") as f:
                pickle.dump(animal_sessions, f)
            with simulated_path.open("wb") as f:
                pickle.dump(simulated_sessions, f)

            saved_result = generative_analysis.run_post_training_analysis_from_saved_histories(
                simulated_session_history_path=simulated_path,
                animal_session_history_path=animal_path,
                output_dir=root / "saved",
            )

            direct_switch_payload = json.loads(Path(direct_result["switch_stats"]).read_text())
            saved_switch_payload = json.loads(Path(saved_result["switch_stats"]).read_text())
            direct_switch_payload.pop("figure_paths", None)
            saved_switch_payload.pop("figure_paths", None)
            self.assertEqual(
                direct_switch_payload,
                saved_switch_payload,
            )
            direct_history_payload = json.loads(
                Path(direct_result["history_dependent_switch_stats"]).read_text()
            )
            saved_history_payload = json.loads(
                Path(saved_result["history_dependent_switch_stats"]).read_text()
            )
            direct_history_payload.pop("figure_paths", None)
            saved_history_payload.pop("figure_paths", None)
            self.assertEqual(
                direct_history_payload,
                saved_history_payload,
            )
            self.assertEqual(
                json.loads(
                    Path(direct_result["model_vs_animal_quantitative_summary"]).read_text()
                ),
                json.loads(
                    Path(saved_result["model_vs_animal_quantitative_summary"]).read_text()
                ),
            )

    def test_run_post_training_analysis_from_saved_histories_loads_animal_from_resolved_run(self):
        simulated_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1__rollout_0",
                source_ses_idx="m1_s1",
                choice_history=[0, 0, 1],
                reward_history=[1, 0, 0],
            )
        ]
        resolved_run = generative_analysis.ResolvedModelRun(
            model_dir="/tmp/model",
            inputs_path="/tmp/model/inputs.yaml",
            outputs_dir="/tmp/model/outputs",
            model_type="gru",
            split="train",
            checkpoint_policy="final",
            checkpoint_step=10,
            checkpoint_label="final",
            params_path="/tmp/model/outputs/params.json",
            config_path="/tmp/model/outputs/gru_config.json",
            seed=1,
            multisubject=False,
            mature_only=True,
            ignore_policy="exclude",
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

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            simulated_path = root / "simulated.pkl"
            resolved_run_path = root / "resolved_run.json"
            with simulated_path.open("wb") as f:
                pickle.dump(simulated_sessions, f)
            resolved_run_path.write_text(json.dumps(resolved_run.to_dict(), indent=2))

            with mock.patch.object(
                generative_analysis,
                "load_animal_session_history",
                return_value=animal_sessions,
            ) as mocked_load:
                result = generative_analysis.run_post_training_analysis_from_saved_histories(
                    simulated_session_history_path=simulated_path,
                    resolved_run_path=resolved_run_path,
                    output_dir=root / "saved",
                )

            mocked_load.assert_called_once()
            self.assertTrue(Path(result["switch_stats"]).exists())
            self.assertTrue(Path(result["history_dependent_switch_stats"]).exists())


if __name__ == "__main__":
    unittest.main()
