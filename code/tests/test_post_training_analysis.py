"""Unit tests for standalone post-training analysis helpers."""

from __future__ import annotations

import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

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
        session_conditioning: bool = False,
        with_session_context_map: bool = True,
    ) -> Path:
        model_dir = root / f"{model_type}_multisubject_run"
        outputs_dir = model_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        architecture_yaml_lines = ["    multisubject: true"]
        if session_conditioning:
            architecture_yaml_lines.extend(
                [
                    "    session_encoding_type: scalar",
                    "    session_integration_type: direct",
                ]
            )

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
"""
            + "\n".join(architecture_yaml_lines)
            + """
seed: 7
"""
        )
        if model_type == "gru":
            gru_architecture = {
                "multisubject": True,
                "hidden_size": 8,
                "subject_embedding_size": 3,
                "subject_embedding_init": "zeros",
            }
            if session_conditioning:
                gru_architecture.update(
                    {
                        "session_encoding_type": "scalar",
                        "session_integration_type": "direct",
                    }
                )
            (outputs_dir / "gru_config.json").write_text(
                json.dumps(
                    {
                        "architecture": gru_architecture,
                        "output_size": 2,
                    }
                )
            )
        else:
            disrnn_config = {
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
            if session_conditioning:
                disrnn_config.update(
                    {
                        "session_encoding_type": "scalar",
                        "session_integration_type": "direct",
                    }
                )
            (outputs_dir / "disrnn_config.json").write_text(
                json.dumps(disrnn_config)
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
        if with_subject_map and with_session_context_map:
            (outputs_dir / "session_context_map.json").write_text(
                json.dumps(
                    {
                        "indexing": "1_based",
                        "per_subject": [
                            {
                                "subject_id": "m1",
                                "subject_index": 0,
                                "ordered_session_ids": ["m1__s1"],
                                "ordered_source_session_ids": ["m1_s1"],
                            },
                            {
                                "subject_id": "m2",
                                "subject_index": 1,
                                "ordered_session_ids": ["m2__s1"],
                                "ordered_source_session_ids": ["m2_s1"],
                            },
                        ],
                    }
                )
            )
        return model_dir

    def _write_baseline_run_dir(
        self,
        root: Path,
        *,
        multisubject: bool = False,
        include_baseline_output: bool = True,
    ) -> Path:
        model_dir = root / (
            "baseline_rl_multisubject_run" if multisubject else "baseline_rl_single_run"
        )
        outputs_dir = model_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        subject_ids_yaml = "  - m1\n  - m2" if multisubject else "  - m1"
        (model_dir / "inputs.yaml").write_text(
            """
data:
  subject_ids:
"""
            + subject_ids_yaml
            + """
  subject_start: null
  subject_end: null
  test_subject_ids:
  - m3
  test_subject_start: null
  test_subject_end: null
  mature_only: true
  curricula:
  - Uncoupled Baiting
  ignore_policy: exclude
  eval_every_n: 2
"""
            + ("  multisubject: true\n" if multisubject else "")
            + """
model:
  type: baseline_rl
  architecture:
    multisubject: """
            + ("true" if multisubject else "false")
            + """
  agent_class: ForagerQLearning
  agent_kwargs:
    number_of_learning_rate: 1
    number_of_forget_rate: 1
    choice_kernel: one_step
    action_selection: softmax
seed: 7
"""
        )

        if include_baseline_output:
            baseline_output = {
                "multisubject": bool(multisubject),
                "fit_strategy": "per_subject" if multisubject else "single_subject",
                "agent_class": "ForagerQLearning",
                "agent_kwargs": {
                    "number_of_learning_rate": 1,
                    "number_of_forget_rate": 1,
                    "choice_kernel": "one_step",
                    "action_selection": "softmax",
                },
            }
            if multisubject:
                baseline_output["fitted_params_per_subject"] = {
                    "m1": {
                        "subject_id": "m1",
                        "subject_index": 0,
                        "train_session_ids": ["m1_s1"],
                        "eval_session_ids": ["m1_s2"],
                        "fitted_params": {"biasL": 0.0},
                    },
                    "m2": {
                        "subject_id": "m2",
                        "subject_index": 1,
                        "train_session_ids": ["m2_s1"],
                        "eval_session_ids": ["m2_s2"],
                        "fitted_params": {"biasL": 1.0},
                    },
                }
                (outputs_dir / "subject_index_map.json").write_text(
                    json.dumps(
                        {
                            "subject_id_to_index": {"m1": 0, "m2": 1},
                            "index_to_subject_id": {"0": "m1", "1": "m2"},
                        }
                    )
                )
            else:
                baseline_output["fitted_params"] = {"biasL": 0.25}

            (outputs_dir / "baseline_rl_output.json").write_text(
                json.dumps(baseline_output, indent=2)
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

            def encode_inputs(
                self,
                subject_id,
                session_id,
                inputs,
                *,
                source_session_id=None,
            ):
                encoded = [float(self._subject_id_to_index[subject_id]), *list(inputs)]
                self.calls.append((subject_id, encoded))
                return encoded

            def initial_state_batch(self, batch_size):
                return {"hidden": [0] * int(batch_size)}

            def step_batch(self, inputs_2d, prev_state):
                if not hasattr(self, "batched_inputs"):
                    self.batched_inputs = []
                arr = np.asarray(inputs_2d, dtype=float)
                self.batched_inputs.append(arr.tolist())
                return np.array([[50.0, -50.0]] * arr.shape[0]), prev_state

        fake_runner = _FakeRunner()

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
                np
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
        # encode_inputs is called once per lane to build the constant prefix
        # (subject index), in lane order.
        self.assertEqual([subject for subject, _ in fake_runner.calls], ["m1", "m2"])
        # The model step is batched across lanes, one call per trial. Each row
        # carries [subject_index, prev_choice, prev_reward]; row b == the old
        # per-session step for lane b.
        self.assertEqual(
            fake_runner.batched_inputs,
            [
                [[0.0, -1.0, -1.0], [1.0, -1.0, -1.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            ],
        )

    def _assert_multisubject_simulation_uses_session_conditioning_indices(
        self,
        model_type: str,
    ) -> None:
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
            session_conditioning_enabled=True,
            session_conditioning_encoding_type="scalar",
            subject_index_map_path="/tmp/model/outputs/subject_index_map.json",
            session_context_map_path="/tmp/model/outputs/session_context_map.json",
            trained_subject_ids=["m1", "m2"],
        )
        animal_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1__s1",
                source_ses_idx="m1_s1",
                choice_history=[0, 0],
                reward_history=[1, 1],
            ),
            self._session(
                subject_id="m2",
                ses_idx="m2__s1",
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
                self._merged_session_index_lookup = {
                    ("m1", "m1__s1"): 1,
                    ("m2", "m2__s1"): 1,
                }
                self._source_session_index_lookup = {
                    ("m1", "m1_s1"): 1,
                    ("m2", "m2_s1"): 1,
                }

            def validate_subject_ids(self, subject_ids):
                if list(subject_ids) != ["m1", "m2"]:
                    raise ValueError("unexpected subjects")

            def encode_inputs(
                self,
                subject_id,
                session_id,
                inputs,
                *,
                source_session_id=None,
            ):
                session_index = generative_analysis._resolve_session_index_for_subject(
                    subject_id=subject_id,
                    session_id=session_id,
                    source_session_id=source_session_id,
                    merged_session_index_lookup=self._merged_session_index_lookup,
                    source_session_index_lookup=self._source_session_index_lookup,
                )
                encoded = [
                    float(self._subject_id_to_index[subject_id]),
                    float(session_index),
                    *list(inputs),
                ]
                self.calls.append((subject_id, encoded))
                return encoded

            def initial_state_batch(self, batch_size):
                return {"hidden": [0] * int(batch_size)}

            def step_batch(self, inputs_2d, prev_state):
                if not hasattr(self, "batched_inputs"):
                    self.batched_inputs = []
                arr = np.asarray(inputs_2d, dtype=float)
                self.batched_inputs.append(arr.tolist())
                return np.array([[50.0, -50.0]] * arr.shape[0]), prev_state

        fake_runner = _FakeRunner()

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
                np
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
        self.assertEqual([subject for subject, _ in fake_runner.calls], ["m1", "m2"])
        # Batched step inputs carry [subject_index, session_index, prev_choice,
        # prev_reward] for each lane, one batched call per trial.
        self.assertEqual(
            fake_runner.batched_inputs,
            [
                [[0.0, 1.0, -1.0, -1.0], [1.0, 1.0, -1.0, -1.0]],
                [[0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]],
            ],
        )

    def test_run_batched_rollout_matches_per_lane_and_respects_lengths(self):
        # The only non-trivial part of batching is variable session lengths:
        # a batched rollout must (a) produce, for each lane, exactly what a
        # solo (batch-1) rollout of that lane would, and (b) not over-run a
        # short lane just because a longer lane shares the batch.
        class _Runner:
            n_actions = 2

            def encode_inputs(self, subject_id, session_id, inputs, *, source_session_id=None):
                return [float(subject_id), *list(inputs)]

            def initial_state_batch(self, batch_size):
                return np.zeros((int(batch_size), 1), dtype=float)

            def step_batch(self, inputs_2d, prev_state):
                arr = np.asarray(inputs_2d, dtype=float)
                # Row-independent recurrence: state[b] depends only on input[b].
                state = np.asarray(prev_state, dtype=float) + arr[:, :1]
                logits = np.stack([state[:, 0], -state[:, 0]], axis=1)
                return logits, state

        lanes = [
            {"subject_id": 0, "model_ses_idx": "a", "source_ses_idx": "a",
             "curriculum_name": "Uncoupled Baiting", "n_trials": 3, "seed": 7},
            {"subject_id": 1, "model_ses_idx": "b", "source_ses_idx": "b",
             "curriculum_name": "Uncoupled Baiting", "n_trials": 1, "seed": 9},
        ]

        class _Task:
            def __init__(self, seed):
                self.rng = np.random.default_rng(seed)

            def reset(self):
                pass

        with mock.patch.object(
            generative_analysis,
            "_build_curriculum_matched_task",
            side_effect=lambda **kwargs: _Task(kwargs["seed"]),
        ), mock.patch.object(
            generative_analysis,
            "_step_task_reward",
            side_effect=lambda task, action: float(task.rng.integers(0, 2)),
        ):
            runner = _Runner()
            ch_both, rw_both = generative_analysis._run_batched_rollout(runner, lanes, n_actions=2)
            ch0, rw0 = generative_analysis._run_batched_rollout(runner, [lanes[0]], n_actions=2)
            ch1, rw1 = generative_analysis._run_batched_rollout(runner, [lanes[1]], n_actions=2)

        # Lengths respected: short lane not over-run despite batch max_trials=3.
        self.assertEqual(len(ch_both[0]), 3)
        self.assertEqual(len(ch_both[1]), 1)
        # Batched == per-lane, bit-for-bit (batch-independence + correct masking).
        self.assertEqual(ch_both[0], ch0[0])
        self.assertEqual(rw_both[0], rw0[0])
        self.assertEqual(ch_both[1], ch1[0])
        self.assertEqual(rw_both[1], rw1[0])

    def test_build_curriculum_matched_task_maps_family_and_version_variants(self):
        # Family-level mapping (coupled/uncoupled + baiting), robust to curriculum
        # *version* suffixes like "2p3". NOTE: deliberately family-only (default
        # task params, stage ignored) — see the TODO in the function.
        calls = []

        class _FakeTaskMod:
            class UncoupledBlockTask:
                def __init__(self, **kwargs):
                    calls.append(("Uncoupled", kwargs))

            class CoupledBlockTask:
                def __init__(self, **kwargs):
                    calls.append(("Coupled", kwargs))

        real_import = generative_analysis.importlib.import_module
        with mock.patch.object(
            generative_analysis.importlib,
            "import_module",
            side_effect=lambda name, *a, **k: (
                _FakeTaskMod if "dynamic_foraging.task" in name else real_import(name, *a, **k)
            ),
        ):
            cases = {
                "Uncoupled Baiting": ("Uncoupled", True),
                "Uncoupled Without Baiting": ("Uncoupled", False),
                "Coupled Baiting": ("Coupled", True),
                "None": ("Uncoupled", True),
                None: ("Uncoupled", True),
                # the high-D failure that motivated the fix:
                "UnCoupledBaiting2p3Curriculum": ("Uncoupled", True),
            }
            for name, (family, baiting) in cases.items():
                calls.clear()
                generative_analysis._build_curriculum_matched_task(
                    curriculum_name=name, n_trials=5, seed=1
                )
                self.assertEqual(calls[-1][0], family, msg=f"{name!r} family")
                self.assertEqual(calls[-1][1].get("reward_baiting"), baiting, msg=f"{name!r} baiting")
            # A non-block-task family still raises (don't silently mis-map it).
            with self.assertRaises(ValueError):
                generative_analysis._build_curriculum_matched_task(
                    curriculum_name="RandomWalkFooCurriculum", n_trials=5, seed=1
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
        self.assertEqual(resolved.selection["min_sessions"], 10)
        self.assertEqual(resolved.selection["heldout_every_n"], 5)
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

    def test_resolve_model_run_baseline_best_eval_uses_final_fit_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_baseline_run_dir(Path(tmpdir), multisubject=False)

            resolved = resolve_model_run(
                model_dir,
                split="train",
                checkpoint_policy="best_eval",
            )

        self.assertEqual(resolved.model_type, "baseline_rl")
        self.assertEqual(resolved.checkpoint_policy, "best_eval")
        self.assertIsNone(resolved.checkpoint_step)
        self.assertEqual(resolved.checkpoint_label, "final_fit")
        self.assertTrue(resolved.params_path.endswith("baseline_rl_output.json"))
        self.assertTrue(resolved.config_path.endswith("baseline_rl_output.json"))
        self.assertTrue(resolved.baseline_output_path.endswith("baseline_rl_output.json"))
        self.assertEqual(resolved.model_config["fit_strategy"], "single_subject")
        self.assertIn("final fitted parameters", resolved.checkpoint_selection_reason)

    def test_resolve_model_run_baseline_multisubject_loads_subject_index_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_baseline_run_dir(Path(tmpdir), multisubject=True)

            resolved = resolve_model_run(
                model_dir,
                split="train",
                checkpoint_policy="final",
            )

        self.assertEqual(resolved.model_type, "baseline_rl")
        self.assertTrue(resolved.multisubject)
        self.assertTrue(resolved.subject_index_map_path.endswith("subject_index_map.json"))
        self.assertEqual(resolved.trained_subject_ids, ["m1", "m2"])
        self.assertEqual(
            sorted(resolved.model_config["fitted_params_per_subject"].keys()),
            ["m1", "m2"],
        )

    def test_resolve_model_run_baseline_multisubject_requires_subject_index_map(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_baseline_run_dir(Path(tmpdir), multisubject=True)
            (model_dir / "outputs" / "subject_index_map.json").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "subject index map"):
                resolve_model_run(
                    model_dir,
                    split="train",
                    checkpoint_policy="final",
                )

    def test_resolve_model_run_baseline_requires_baseline_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_baseline_run_dir(
                Path(tmpdir),
                multisubject=False,
                include_baseline_output=False,
            )

            with self.assertRaisesRegex(FileNotFoundError, "baseline RL output"):
                resolve_model_run(
                    model_dir,
                    split="train",
                    checkpoint_policy="final",
                )

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
        self.assertTrue(resolved.session_context_map_path.endswith("session_context_map.json"))
        self.assertEqual(resolved.trained_subject_ids, ["m1", "m2"])
        self.assertFalse(resolved.session_conditioning_enabled)
        self.assertEqual(resolved.session_conditioning_encoding_type, "none")

    def test_resolve_model_run_session_conditioned_multisubject_records_artifact_contract(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_multisubject_run_dir(
                Path(tmpdir),
                model_type="gru",
                session_conditioning=True,
            )

            resolved = resolve_model_run(
                model_dir,
                split="train",
                checkpoint_policy="final",
            )

        self.assertTrue(resolved.session_conditioning_enabled)
        self.assertEqual(resolved.session_conditioning_encoding_type, "scalar")
        self.assertEqual(
            resolved.required_session_conditioning_artifacts,
            ["subject_index_map.json", "session_context_map.json"],
        )
        self.assertEqual(resolved.missing_session_conditioning_artifacts, [])

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

    def test_load_multisubject_analysis_context_requires_session_context_artifact_for_session_conditioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_multisubject_run_dir(
                Path(tmpdir),
                model_type="gru",
                session_conditioning=True,
                with_session_context_map=False,
            )
            resolved = resolve_model_run(
                model_dir,
                split="train",
                checkpoint_policy="final",
            )

            with self.assertRaisesRegex(FileNotFoundError, "session_context_map.json"):
                generative_analysis._load_multisubject_analysis_context(resolved)

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
        fake_snapshot_module.load_mice_from_database.return_value = (snapshot_df, ["m1"])

        def _fake_import_module(name):
            if name == "utils.load_mice_database":
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

        fake_snapshot_module.load_mice_from_database.assert_called_once()
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
        fake_snapshot_module.load_mice_from_database.return_value = (snapshot_df, ["m1", "m2"])

        def _fake_import_module(name):
            if name == "utils.load_mice_database":
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

        fake_snapshot_module.load_mice_from_database.assert_called_once_with(
            split="train",
            subject_ids=["m1", "m2"],
            curricula=["Uncoupled Baiting"],
            subject_ratio=None,
            min_sessions=10,
            heldout_every_n=5,
            seed=None,
            mature_only=True,
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

    def test_load_animal_session_history_multisubject_aligns_session_ids_to_training_namespace(self):
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
            session_context_map_path="/tmp/model/outputs/session_context_map.json",
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
        built_history = [
            {
                "subject_id": "m1",
                "ses_idx": "m1_s1",
                "choice_history": [0, 1],
                "reward_history": [1, 0],
                "n_trials": 2,
            },
            {
                "subject_id": "m2",
                "ses_idx": "m2_s1",
                "choice_history": [0, 1],
                "reward_history": [0, 1],
                "n_trials": 2,
            },
        ]
        fake_snapshot_module = mock.Mock()
        fake_snapshot_module.load_mice_from_database.return_value = (snapshot_df, ["m1", "m2"])

        fake_multisubject_module = mock.Mock()
        fake_multisubject_module.load_session_context_map.return_value = {
            "indexing": "1_based",
            "per_subject": [
                {
                    "subject_id": "m1",
                    "subject_index": 0,
                    "ordered_session_ids": ["m1__m1_s1"],
                    "ordered_source_session_ids": ["m1_s1"],
                },
                {
                    "subject_id": "m2",
                    "subject_index": 1,
                    "ordered_session_ids": ["m2__m2_s1"],
                    "ordered_source_session_ids": ["m2_s1"],
                },
            ],
        }
        fake_multisubject_module.ordered_session_context_rows.return_value = [
            {
                "subject_id": "m1",
                "subject_index": 0,
                "ordered_session_ids": ["m1__m1_s1"],
                "ordered_source_session_ids": ["m1_s1"],
            },
            {
                "subject_id": "m2",
                "subject_index": 1,
                "ordered_session_ids": ["m2__m2_s1"],
                "ordered_source_session_ids": ["m2_s1"],
            },
        ]

        def _fake_import_module(name):
            if name == "utils.load_mice_database":
                return fake_snapshot_module
            if name == "utils.multisubject":
                return fake_multisubject_module
            return __import__(name)

        with mock.patch.object(
            generative_analysis, "_build_session_history_dataframe", return_value=built_history
        ), mock.patch.object(
            generative_analysis,
            "_align_snapshot_df_with_ignore_policy",
            side_effect=lambda snapshot_df, ignore_policy: snapshot_df,
        ), mock.patch.object(
            generative_analysis,
            "importlib",
        ) as importlib_mock:
            importlib_mock.import_module.side_effect = _fake_import_module
            session_history = load_animal_session_history(resolved)

        self.assertEqual(
            session_history,
            [
                {
                    "subject_id": "m1",
                    "ses_idx": "m1__m1_s1",
                    "source_ses_idx": "m1_s1",
                    "choice_history": [0, 1],
                    "reward_history": [1, 0],
                    "n_trials": 2,
                },
                {
                    "subject_id": "m2",
                    "ses_idx": "m2__m2_s1",
                    "source_ses_idx": "m2_s1",
                    "choice_history": [0, 1],
                    "reward_history": [0, 1],
                    "n_trials": 2,
                },
            ],
        )
        self.assertEqual(resolved.resolved_session_ids, ["m1__m1_s1", "m2__m2_s1"])

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
        session_point = stats["session_level"]["post_switch_by_reward"]["rewarded"][
            "points"
        ][0]
        self.assertEqual(session_point["subject_id"], "m4")
        self.assertEqual(session_point["session_id"], "m4_s1")
        self.assertEqual(session_point["source_ses_idx"], "m4_s1")
        self.assertEqual(session_point["animal_n"], 6)
        self.assertEqual(session_point["simulated_effective_n"], 2)
        self.assertAlmostEqual(session_point["simulated_probability"], 0.75)
        self.assertAlmostEqual(session_point["delta_probability"], -0.25)

        rewarded_session_aggregate = stats["session_aggregate"]["post_switch_by_reward"][
            "rewarded"
        ]
        self.assertEqual(rewarded_session_aggregate["n_sessions"], 1)
        self.assertAlmostEqual(rewarded_session_aggregate["animal_mean"], 1.0)
        self.assertAlmostEqual(rewarded_session_aggregate["simulated_mean"], 0.75)
        self.assertAlmostEqual(rewarded_session_aggregate["delta_mean"], -0.25)
        self.assertEqual(
            stats["quantitative_summary"]["session_mean"]["post_switch_by_reward"][
                "n_rows"
            ],
            1,
        )

    def test_compute_switch_stats_session_level_matches_source_sessions_only(self):
        stats = compute_switch_stats(
            animal_sessions=[
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1",
                    choice_history=[0, 1, 0, 1, 0, 1],
                    reward_history=[0, 1, 0, 1, 0, 1],
                ),
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s2",
                    choice_history=[0, 1, 0, 1, 0, 1],
                    reward_history=[0, 1, 0, 1, 0, 1],
                ),
            ],
            simulated_sessions=[
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1__rollout_0",
                    source_ses_idx="m1_s1",
                    choice_history=[0, 1, 0, 1, 0, 1],
                    reward_history=[0, 1, 0, 1, 0, 1],
                ),
                self._session(
                    subject_id="m1",
                    ses_idx="m1_extra__rollout_0",
                    source_ses_idx="m1_extra",
                    choice_history=[0, 1, 0, 1, 0, 1],
                    reward_history=[0, 1, 0, 1, 0, 1],
                ),
            ],
            window_size=1,
        )

        rewarded_points = stats["session_level"]["post_switch_by_reward"]["rewarded"][
            "points"
        ]
        self.assertEqual(
            [point["session_id"] for point in rewarded_points],
            ["m1_s1"],
        )
        self.assertEqual(
            stats["session_level"]["post_switch_by_reward"]["rewarded"]["summary"][
                "n_sessions"
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
                subject_id="m4/a",
                ses_idx="m4_s1",
                choice_history=[0, 1, 0, 1, 0, 1, 0, 1],
                reward_history=[0, 1, 1, 1, 1, 1, 1, 0],
            )
        ]
        simulated_sessions = [
            self._session(
                subject_id="m4/a",
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
            self.assertIn("post_switch_by_reward_session_scatter", figure_paths)
            self.assertIn("post_switch_delta_by_reward", figure_paths)
            self.assertIn("post_switch_delta_by_reward_no_stats", figure_paths)
            self.assertIn("post_switch_delta_by_reward_and_run_length", figure_paths)
            self.assertIn(
                "post_switch_delta_by_reward_and_run_length_no_stats",
                figure_paths,
            )
            self.assertIn(
                "post_switch_by_reward_and_run_length_subject_scatter",
                figure_paths,
            )
            self.assertIn(
                "post_switch_by_reward_and_run_length_session_scatter",
                figure_paths,
            )
            self.assertIn(
                "post_switch_by_reward_session_scatter__subject_m4_a",
                figure_paths,
            )
            self.assertIn(
                "post_switch_by_reward_and_run_length_session_scatter__subject_m4_a",
                figure_paths,
            )
            self.assertTrue(
                figure_paths["post_switch_by_reward_subject_scatter"]
                .name.endswith(".png")
            )
            self.assertTrue(
                figure_paths["post_switch_by_reward_session_scatter"].exists()
            )
            self.assertTrue(
                figure_paths[
                    "post_switch_by_reward_and_run_length_subject_scatter"
                ].name.endswith(".png")
            )

    def test_save_switch_figures_handles_missing_probability_bars(self):
        try:
            import matplotlib.pyplot  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed")

        animal_sessions = [
            self._session(
                subject_id="m5",
                ses_idx="m5_s1",
                choice_history=[0, 1, 0, 1, 0, 1, 0, 1],
                reward_history=[0, 1, 1, 1, 1, 1, 1, 0],
            )
        ]
        simulated_sessions = [
            self._session(
                subject_id="m5",
                ses_idx="m5_s1__rollout_0",
                source_ses_idx="m5_s1",
                choice_history=[0, 1, 0],
                reward_history=[0, 1, 0],
            )
        ]
        stats = compute_switch_stats(
            animal_sessions=animal_sessions,
            simulated_sessions=simulated_sessions,
            window_size=1,
        )
        stats["animal"]["post_switch_by_reward"]["rewarded"]["probability"] = None
        stats["animal"]["post_switch_by_reward"]["rewarded"]["sem"] = None
        stats["subject_aggregate"]["post_switch_by_reward"]["rewarded"]["animal_mean"] = None
        stats["subject_aggregate"]["post_switch_by_reward"]["rewarded"]["animal_sem"] = None
        stats["animal"]["post_switch_by_reward_and_run_length"]["rewarded"]["run_length_1"][
            "probability"
        ] = None
        stats["animal"]["post_switch_by_reward_and_run_length"]["rewarded"]["run_length_1"][
            "sem"
        ] = None
        stats["subject_aggregate"]["post_switch_by_reward_and_run_length"]["rewarded"][
            "run_length_1"
        ]["animal_mean"] = None
        stats["subject_aggregate"]["post_switch_by_reward_and_run_length"]["rewarded"][
            "run_length_1"
        ]["animal_sem"] = None

        with tempfile.TemporaryDirectory() as tmpdir:
            figure_paths = generative_analysis._save_switch_figures(
                switch_stats=stats,
                output_dir=Path(tmpdir),
            )
            self.assertTrue(figure_paths["post_switch_by_reward_pooled"].exists())
            self.assertTrue(figure_paths["post_switch_by_reward"].exists())
            self.assertTrue(
                figure_paths["post_switch_by_reward_and_run_length_pooled"].exists()
            )
            self.assertTrue(figure_paths["post_switch_by_reward_and_run_length"].exists())
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
        session_points = {
            point["session_id"]: point
            for point in stats["session_level"]["abstract"][1]["a"]["points"]
        }
        self.assertEqual(set(session_points), {"m1_s1", "m1_s2", "m2_s1", "m2_s2"})
        self.assertEqual(session_points["m1_s1"]["subject_id"], "m1")
        self.assertEqual(session_points["m1_s1"]["animal_total"], 1)
        self.assertAlmostEqual(session_points["m1_s1"]["simulated_probability"], 0.5)
        self.assertEqual(session_points["m1_s1"]["simulated_total_effective"], 1)
        self.assertAlmostEqual(session_points["m1_s1"]["delta_probability"], -0.5)
        session_aggregate_rows = {
            row["pattern"]: row
            for row in stats["session_aggregate"]["abstract"][1]["rows"]
        }
        self.assertEqual(
            session_aggregate_rows["a"]["n_sessions"],
            4,
        )
        self.assertEqual(
            stats["quantitative_summary"]["session_mean"]["abstract"][1]["n_rows"],
            2,
        )

    def test_compute_history_dependent_switch_stats_ignores_nan_source_session_ids(self):
        stats = compute_history_dependent_switch_stats(
            animal_sessions=[
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s1",
                    source_ses_idx=float("nan"),
                    choice_history=[0, 0, 1],
                    reward_history=[1, 0, 0],
                ),
                self._session(
                    subject_id="m1",
                    ses_idx="m1_s2",
                    source_ses_idx=float("nan"),
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
                    ses_idx="m1_s2__rollout_0",
                    source_ses_idx="m1_s2",
                    choice_history=[0, 0, 0],
                    reward_history=[1, 0, 0],
                ),
            ],
        )

        session_points = {
            point["session_id"]: point
            for point in stats["session_level"]["abstract"][1]["a"]["points"]
        }
        self.assertEqual(set(session_points), {"m1_s1", "m1_s2"})
        self.assertEqual(session_points["m1_s1"]["source_ses_idx"], "m1_s1")
        self.assertEqual(session_points["m1_s2"]["source_ses_idx"], "m1_s2")

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
            self.assertIn("history_pattern_delta_abstract_no_stats", figure_paths)
            self.assertIn("history_pattern_subject_level_abstract_nback_1", figure_paths)
            self.assertIn("history_pattern_subject_level_abstract_nback_2", figure_paths)
            self.assertIn("history_pattern_subject_level_abstract_nback_3", figure_paths)
            self.assertIn("history_pattern_session_level_abstract_nback_1", figure_paths)
            self.assertIn("history_pattern_session_level_abstract_nback_2", figure_paths)
            self.assertIn("history_pattern_session_level_abstract_nback_3", figure_paths)
            self.assertIn(
                "history_pattern_session_level_abstract_nback_1__subject_m1",
                figure_paths,
            )
            for path in figure_paths.values():
                self.assertTrue(path.name.endswith(".png"))
                self.assertTrue(path.exists())

    def test_build_sorted_history_delta_rows_orders_by_descending_delta_median(self):
        panel_group = {
            "median_high": {
                "points": [
                    {
                        "animal_probability": 0.2,
                        "simulated_probability": 1.1,
                        "delta_probability": 0.9,
                        "animal_total": 3,
                        "simulated_total_effective": 3,
                    },
                    {
                        "animal_probability": 0.2,
                        "simulated_probability": 1.1,
                        "delta_probability": 0.9,
                        "animal_total": 3,
                        "simulated_total_effective": 3,
                    },
                    {
                        "animal_probability": 0.8,
                        "simulated_probability": 0.0,
                        "delta_probability": -0.8,
                        "animal_total": 3,
                        "simulated_total_effective": 3,
                    }
                ]
            },
            "median_mid": {
                "points": [
                    {
                        "animal_probability": 0.2,
                        "simulated_probability": 0.7,
                        "delta_probability": 0.5,
                        "animal_total": 3,
                        "simulated_total_effective": 3,
                    },
                    {
                        "animal_probability": 0.2,
                        "simulated_probability": 0.7,
                        "delta_probability": 0.5,
                        "animal_total": 3,
                        "simulated_total_effective": 3,
                    },
                    {
                        "animal_probability": 0.2,
                        "simulated_probability": 0.7,
                        "delta_probability": 0.5,
                        "animal_total": 3,
                        "simulated_total_effective": 3,
                    },
                ]
            },
            "median_low": {
                "points": [
                    {
                        "animal_probability": 0.6,
                        "simulated_probability": 0.3,
                        "delta_probability": -0.3,
                        "animal_total": 3,
                        "simulated_total_effective": 3,
                    },
                    {
                        "animal_probability": 0.6,
                        "simulated_probability": 0.4,
                        "delta_probability": -0.2,
                        "animal_total": 3,
                        "simulated_total_effective": 3,
                    },
                    {
                        "animal_probability": 0.6,
                        "simulated_probability": 0.5,
                        "delta_probability": -0.1,
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

        self.assertEqual(
            [row["label"] for row in rows],
            ["median_high", "median_mid", "median_low"],
        )

    def test_wilcoxon_signed_rank_against_zero_and_label(self):
        result = generative_analysis._wilcoxon_signed_rank_against_zero(
            [0.2, 0.1, 0.3, 0.4, 0.5, 0.6]
        )

        self.assertEqual(result["n_nonzero"], 6)
        self.assertEqual(result["statistic"], 0.0)
        self.assertAlmostEqual(result["p_value"], 0.03125)
        self.assertEqual(
            generative_analysis._format_significance_label(result["p_value"]),
            "*",
        )
        self.assertEqual(
            generative_analysis._format_significance_label(None),
            "",
        )

    def test_build_delta_condition_summary_summarizes_subject_condition_errors(self):
        rows = [
            {
                "label": "A",
                "points": [
                    {"subject_id": subject_id, "delta_probability": value, "animal_n": 2}
                    for subject_id, value in zip(
                        ["s1", "s2", "s3", "s4", "s5", "s6"],
                        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    )
                ],
            },
            {
                "label": "B",
                "points": [
                    {"subject_id": subject_id, "delta_probability": value, "animal_n": 1}
                    for subject_id, value in zip(
                        ["s1", "s2", "s3", "s4", "s5", "s6"],
                        [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
                    )
                ],
            },
            {
                "label": "C",
                "points": [
                    {"subject_id": subject_id, "delta_probability": value, "animal_n": 5}
                    for subject_id, value in zip(
                        ["s1", "s2"],
                        [-0.1, 0.1],
                    )
                ],
            },
        ]

        summary = generative_analysis._build_delta_condition_summary(
            rows,
            animal_count_key="animal_n",
        )

        error_summary = summary["subject_condition_error_summary"]
        subject_balanced_summary = summary["subject_balanced_error_summary"]
        condition_balanced_summary = summary["condition_balanced_error_summary"]
        significant_summary = summary["significant_conditions_summary"]
        self.assertEqual(significant_summary["n_significant_conditions"], 2)
        self.assertAlmostEqual(
            error_summary["mean_signed_error"],
            0.2357142857142857,
        )
        self.assertAlmostEqual(
            error_summary["mean_signed_error_sem"],
            0.04472950774347691,
        )
        self.assertAlmostEqual(
            error_summary["p_value"],
            0.0014412255000055883,
        )
        self.assertAlmostEqual(
            error_summary["mean_absolute_error"],
            0.25,
        )
        self.assertAlmostEqual(
            error_summary["mean_squared_error"],
            0.08357142857142857,
        )
        self.assertEqual(error_summary["n_subject_condition_pairs"], 14)
        self.assertEqual(error_summary["n_nonzero_subject_condition_pairs"], 14)
        self.assertAlmostEqual(
            subject_balanced_summary["mean_signed_error"],
            0.25555555555555554,
        )
        self.assertAlmostEqual(
            subject_balanced_summary["mean_signed_error_sem"],
            0.04575610778002162,
        )
        self.assertAlmostEqual(
            subject_balanced_summary["p_value"],
            0.03125,
        )
        self.assertAlmostEqual(
            subject_balanced_summary["mean_absolute_error"],
            0.25555555555555554,
        )
        self.assertAlmostEqual(
            subject_balanced_summary["mean_squared_error"],
            0.07787037037037038,
        )
        self.assertEqual(subject_balanced_summary["n_subjects"], 6)
        self.assertEqual(subject_balanced_summary["n_nonzero_subjects"], 6)
        self.assertAlmostEqual(
            condition_balanced_summary["mean_signed_error"],
            0.18333333333333335,
        )
        self.assertAlmostEqual(
            condition_balanced_summary["mean_signed_error_sem"],
            0.08277591347639632,
        )
        self.assertAlmostEqual(
            condition_balanced_summary["p_value"],
            0.5,
        )
        self.assertAlmostEqual(
            condition_balanced_summary["mean_absolute_error"],
            0.18333333333333335,
        )
        self.assertAlmostEqual(
            condition_balanced_summary["mean_squared_error"],
            0.05416666666666666,
        )
        self.assertEqual(condition_balanced_summary["n_conditions"], 3)
        self.assertEqual(condition_balanced_summary["n_nonzero_conditions"], 2)
        self.assertEqual(significant_summary["condition_labels"], ["A", "B"])

    def test_simulate_model_sessions_multisubject_gru_uses_subject_indices(self):
        self._assert_multisubject_simulation_uses_subject_indices("gru")

    def test_simulate_model_sessions_multisubject_disrnn_uses_subject_indices(self):
        self._assert_multisubject_simulation_uses_subject_indices("disrnn")

    def test_simulate_model_sessions_session_conditioned_gru_uses_subject_and_session_indices(self):
        self._assert_multisubject_simulation_uses_session_conditioning_indices("gru")

    def test_simulate_model_sessions_session_conditioned_disrnn_uses_subject_and_session_indices(self):
        self._assert_multisubject_simulation_uses_session_conditioning_indices("disrnn")

    def test_simulate_model_sessions_baseline_multisubject_uses_subject_specific_fitted_params(self):
        resolved_run = generative_analysis.ResolvedModelRun(
            model_dir="/tmp/model",
            inputs_path="/tmp/model/inputs.yaml",
            outputs_dir="/tmp/model/outputs",
            model_type="baseline_rl",
            split="train",
            checkpoint_policy="best_eval",
            checkpoint_step=None,
            checkpoint_label="final_fit",
            params_path="/tmp/model/outputs/baseline_rl_output.json",
            config_path="/tmp/model/outputs/baseline_rl_output.json",
            baseline_output_path="/tmp/model/outputs/baseline_rl_output.json",
            seed=11,
            multisubject=True,
            mature_only=True,
            ignore_policy="exclude",
            curricula=["Uncoupled Baiting"],
            features=None,
            selection={"subject_ids": ["m1", "m2"]},
            subject_index_map_path="/tmp/model/outputs/subject_index_map.json",
            trained_subject_ids=["m1", "m2"],
            model_config={
                "multisubject": True,
                "fit_strategy": "per_subject",
                "agent_class": "ForagerQLearning",
                "agent_kwargs": {"choice_kernel": "one_step"},
                "fitted_params_per_subject": {
                    "m1": {"fitted_params": {"biasL": 0.0}},
                    "m2": {"fitted_params": {"biasL": 1.0}},
                },
            },
            run_config={
                "model": {
                    "type": "baseline_rl",
                    "agent_class": "ForagerQLearning",
                    "agent_kwargs": {"choice_kernel": "one_step"},
                }
            },
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

        class _FakeTask:
            def __init__(self, n_trials):
                self.n_trials = int(n_trials)

            def reset(self):
                return None

        class _FakeBaselineAgent:
            def __init__(self, **kwargs):
                self.params = None

            def set_params(self, **params):
                self.params = dict(params)

            def perform(self, task):
                bias = float(self.params["biasL"])
                choice_value = 1.0 if bias > 0.5 else 0.0
                reward_value = 1.0 if bias > 0.5 else 0.0
                self._choice_history = [choice_value] * int(task.n_trials)
                self._reward_history = [reward_value] * int(task.n_trials)

            def get_choice_history(self):
                return list(self._choice_history)

            def get_reward_history(self):
                return list(self._reward_history)

        class _FakeGenerativeModelModule:
            ForagerQLearning = _FakeBaselineAgent

        class _FakePandas:
            class DataFrame:
                @staticmethod
                def from_records(records):
                    return list(records)

        with mock.patch.object(
            generative_analysis,
            "_build_curriculum_matched_task",
            side_effect=lambda **kwargs: _FakeTask(kwargs["n_trials"]),
        ), mock.patch.object(
            generative_analysis,
            "_import_dependency",
            side_effect=lambda module_name: (
                _FakePandas
                if module_name == "pandas"
                else _FakeGenerativeModelModule
                if module_name == "aind_dynamic_foraging_models.generative_model"
                else None
            ),
        ):
            simulated = generative_analysis.simulate_model_sessions(
                resolved_run=resolved_run,
                animal_sessions=animal_sessions,
                n_rollouts_per_session=1,
            )

        self.assertEqual(len(simulated), 2)
        self.assertEqual(simulated[0]["source_ses_idx"], "m1_s1")
        self.assertEqual(simulated[0]["choice_history"], [0.0, 0.0])
        self.assertEqual(simulated[1]["source_ses_idx"], "m2_s1")
        self.assertEqual(simulated[1]["choice_history"], [1.0, 1.0])

    def test_resolve_session_index_for_subject_prefers_source_session_id(self):
        resolved_index = generative_analysis._resolve_session_index_for_subject(
            subject_id="m1",
            session_id="m1__s1",
            source_session_id="m1_s1",
            merged_session_index_lookup={("m1", "m1__s1"): 7},
            source_session_index_lookup={("m1", "m1_s1"): 3},
        )

        self.assertEqual(resolved_index, 3)

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

    def test_run_post_training_analysis_baseline_single_subject_resolves_and_saves_outputs(self):
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
            root = Path(tmpdir)
            model_dir = self._write_baseline_run_dir(root, multisubject=False)
            switch_figure = root / "switch.png"
            history_figure = root / "history.png"
            switch_figure.write_text("switch")
            history_figure.write_text("history")

            with mock.patch.object(
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
                    model_dir=model_dir,
                    output_dir=root / "analysis",
                )

            resolved_payload = json.loads(Path(result["resolved_run"]).read_text())
            self.assertEqual(resolved_payload["model_type"], "baseline_rl")
            self.assertTrue(
                resolved_payload["baseline_output_path"].endswith("baseline_rl_output.json")
            )
            self.assertTrue(Path(result["switch_stats"]).exists())
            self.assertTrue(Path(result["history_dependent_switch_stats"]).exists())

    def test_run_post_training_analysis_baseline_multisubject_partitions_use_saved_manifest(self):
        animal_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1",
                choice_history=[0, 0, 1],
                reward_history=[1, 0, 0],
            ),
            self._session(
                subject_id="m1",
                ses_idx="m1_s2",
                choice_history=[1, 1, 0],
                reward_history=[0, 1, 0],
            ),
            self._session(
                subject_id="m2",
                ses_idx="m2_s1",
                choice_history=[0, 1, 0],
                reward_history=[1, 1, 0],
            ),
            self._session(
                subject_id="m2",
                ses_idx="m2_s2",
                choice_history=[1, 0, 1],
                reward_history=[0, 1, 1],
            ),
        ]
        simulated_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1__rollout_0",
                source_ses_idx="m1_s1",
                choice_history=[0, 0, 1],
                reward_history=[1, 0, 0],
            ),
            self._session(
                subject_id="m1",
                ses_idx="m1_s2__rollout_0",
                source_ses_idx="m1_s2",
                choice_history=[1, 1, 0],
                reward_history=[0, 1, 0],
            ),
            self._session(
                subject_id="m2",
                ses_idx="m2_s1__rollout_0",
                source_ses_idx="m2_s1",
                choice_history=[0, 1, 0],
                reward_history=[1, 1, 0],
            ),
            self._session(
                subject_id="m2",
                ses_idx="m2_s2__rollout_0",
                source_ses_idx="m2_s2",
                choice_history=[1, 0, 1],
                reward_history=[0, 1, 1],
            ),
        ]
        captured_calls: dict[str, dict[str, object]] = {}

        def _fake_compute(
            *,
            animal_sessions,
            simulated_sessions,
            output_dir,
            resolved_run,
            window_size,
            save_animal_session_history,
        ):
            partition = Path(output_dir).name
            captured_calls[partition] = {
                "animal_session_ids": [row["ses_idx"] for row in animal_sessions],
                "simulated_source_session_ids": [
                    row.get("source_ses_idx", row["ses_idx"]) for row in simulated_sessions
                ],
                "resolved_run_has_manifest": resolved_run.session_split_manifest is not None,
            }
            output_dir_path = Path(output_dir)
            return {
                "output_dir": str(output_dir_path),
                "simulated_session_history": str(output_dir_path / "simulated.pkl"),
                "switch_stats": str(output_dir_path / "switch.json"),
                "history_dependent_switch_stats": str(output_dir_path / "history.json"),
                "model_vs_animal_quantitative_summary": str(output_dir_path / "summary.json"),
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            model_dir = self._write_baseline_run_dir(root, multisubject=True)

            with mock.patch.object(
                generative_analysis,
                "load_animal_session_history",
                return_value=animal_sessions,
            ), mock.patch.object(
                generative_analysis,
                "simulate_model_sessions",
                return_value=simulated_sessions,
            ), mock.patch.object(
                generative_analysis,
                "_compute_and_save_post_training_outputs",
                side_effect=_fake_compute,
            ) as mocked_compute, mock.patch.object(
                generative_analysis.importlib,
                "import_module",
                side_effect=AssertionError("manifest reconstruction should not run"),
            ):
                result = generative_analysis.run_post_training_analysis(
                    model_dir=model_dir,
                    output_dir=root / "analysis",
                    session_partitions=("train", "eval", "combined"),
                )

            self.assertEqual(mocked_compute.call_count, 3)
            self.assertEqual(
                captured_calls["train"]["animal_session_ids"],
                ["m1_s1", "m2_s1"],
            )
            self.assertEqual(
                captured_calls["train"]["simulated_source_session_ids"],
                ["m1_s1", "m2_s1"],
            )
            self.assertEqual(
                captured_calls["eval"]["animal_session_ids"],
                ["m1_s2", "m2_s2"],
            )
            self.assertEqual(
                captured_calls["eval"]["simulated_source_session_ids"],
                ["m1_s2", "m2_s2"],
            )
            resolved_payload = json.loads(Path(result["resolved_run"]).read_text())
            self.assertEqual(
                resolved_payload["session_split_manifest"]["train_session_ids"],
                ["m1_s1", "m2_s1"],
            )

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
            self.assertIn(
                "delta_significance_summary",
                summary_payload["switch_triggered"],
            )
            self.assertIn(
                "delta_significance_summary",
                summary_payload["history_dependent"],
            )
            self.assertIn(
                "session_mean",
                summary_payload["switch_triggered"]["quantitative_summary"],
            )
            self.assertIn(
                "session_mean",
                summary_payload["history_dependent"]["quantitative_summary"],
            )

            switch_payload = json.loads(Path(result["switch_stats"]).read_text())
            self.assertIn("subject_aggregate", switch_payload)
            self.assertIn("session_level", switch_payload)
            self.assertIn("session_aggregate", switch_payload)
            self.assertIn("quantitative_summary", switch_payload)
            self.assertIn("delta_significance_summary", switch_payload)

            history_payload = json.loads(
                Path(result["history_dependent_switch_stats"]).read_text()
            )
            self.assertIn("subject_aggregate", history_payload)
            self.assertIn("session_level", history_payload)
            self.assertIn("session_aggregate", history_payload)
            self.assertIn("quantitative_summary", history_payload)
            self.assertIn("delta_significance_summary", history_payload)

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

    def test_run_post_training_analysis_from_histories_partitions_train_eval_and_combined(self):
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
            session_split_manifest={
                "source": "persisted",
                "multisubject": False,
                "eval_every_n": 2,
                "selected_subject_ids": ["m1"],
                "full_session_ids": ["m1_s1", "m1_s2"],
                "train_session_ids": ["m1_s1"],
                "eval_session_ids": ["m1_s2"],
                "per_subject": [
                    {
                        "subject_id": "m1",
                        "full_session_ids": ["m1_s1", "m1_s2"],
                        "train_session_ids": ["m1_s1"],
                        "eval_session_ids": ["m1_s2"],
                    }
                ],
            },
        )
        animal_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1",
                choice_history=[0, 0, 1],
                reward_history=[1, 0, 1],
            ),
            self._session(
                subject_id="m1",
                ses_idx="m1_s2",
                choice_history=[1, 1, 0],
                reward_history=[0, 1, 0],
            ),
        ]
        simulated_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1__rollout_0",
                source_ses_idx="m1_s1",
                choice_history=[0, 0, 1],
                reward_history=[1, 0, 1],
            ),
            self._session(
                subject_id="m1",
                ses_idx="m1_s1__rollout_1",
                source_ses_idx="m1_s1",
                choice_history=[0, 1, 1],
                reward_history=[1, 1, 0],
            ),
            self._session(
                subject_id="m1",
                ses_idx="m1_s2__rollout_0",
                source_ses_idx="m1_s2",
                choice_history=[1, 1, 0],
                reward_history=[0, 1, 0],
            ),
        ]

        captured_calls: dict[str, dict[str, object]] = {}

        def _fake_compute(
            *,
            animal_sessions,
            simulated_sessions,
            output_dir,
            resolved_run,
            window_size,
            save_animal_session_history,
        ):
            partition = Path(output_dir).name
            captured_calls[partition] = {
                "animal_session_ids": [row["ses_idx"] for row in animal_sessions],
                "simulated_source_session_ids": [
                    row.get("source_ses_idx", row["ses_idx"]) for row in simulated_sessions
                ],
                "resolved_run_has_manifest": resolved_run.session_split_manifest is not None,
                "window_size": window_size,
                "save_animal_session_history": save_animal_session_history,
            }
            output_dir_path = Path(output_dir)
            return {
                "output_dir": str(output_dir_path),
                "simulated_session_history": str(output_dir_path / "simulated.pkl"),
                "switch_stats": str(output_dir_path / "switch.json"),
                "history_dependent_switch_stats": str(output_dir_path / "history.json"),
                "model_vs_animal_quantitative_summary": str(output_dir_path / "summary.json"),
            }

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            generative_analysis,
            "_compute_and_save_post_training_outputs",
            side_effect=_fake_compute,
        ) as mocked_compute, mock.patch.object(
            generative_analysis.importlib,
            "import_module",
            side_effect=AssertionError("manifest reconstruction should not run"),
        ):
            result = generative_analysis.run_post_training_analysis_from_histories(
                animal_sessions=animal_sessions,
                simulated_sessions=simulated_sessions,
                output_dir=Path(tmpdir),
                resolved_run=resolved_run,
                session_partitions=("train", "eval", "combined"),
            )

            self.assertEqual(mocked_compute.call_count, 3)
            self.assertEqual(
                captured_calls["train"]["animal_session_ids"],
                ["m1_s1"],
            )
            self.assertEqual(
                captured_calls["train"]["simulated_source_session_ids"],
                ["m1_s1", "m1_s1"],
            )
            self.assertEqual(
                captured_calls["eval"]["animal_session_ids"],
                ["m1_s2"],
            )
            self.assertEqual(
                captured_calls["eval"]["simulated_source_session_ids"],
                ["m1_s2"],
            )
            self.assertEqual(
                captured_calls["combined"]["animal_session_ids"],
                ["m1_s1", "m1_s2"],
            )
            self.assertEqual(
                captured_calls["combined"]["simulated_source_session_ids"],
                ["m1_s1", "m1_s1", "m1_s2"],
            )
            self.assertTrue(Path(result["resolved_run"]).exists())
            self.assertTrue(Path(result["session_partition_summary"]).exists())
            self.assertEqual(
                set(result["partition_results"].keys()),
                {"train", "eval", "combined"},
            )

    def test_run_post_training_analysis_from_saved_histories_reconstructs_missing_manifest(self):
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
            run_config={"data": {"type": "mice_snapshot", "eval_every_n": 2}},
        )
        animal_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1",
                choice_history=[0, 0, 1],
                reward_history=[1, 0, 1],
            ),
            self._session(
                subject_id="m1",
                ses_idx="m1_s2",
                choice_history=[1, 1, 0],
                reward_history=[0, 1, 0],
            ),
        ]
        simulated_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1__rollout_0",
                source_ses_idx="m1_s1",
                choice_history=[0, 0, 1],
                reward_history=[1, 0, 1],
            ),
            self._session(
                subject_id="m1",
                ses_idx="m1_s2__rollout_0",
                source_ses_idx="m1_s2",
                choice_history=[1, 1, 0],
                reward_history=[0, 1, 0],
            ),
        ]
        reconstructed_manifest = {
            "source": "snapshot_reconstruction",
            "multisubject": False,
            "eval_every_n": 2,
            "selected_subject_ids": ["m1"],
            "full_session_ids": ["m1_s1", "m1_s2"],
            "train_session_ids": ["m1_s1"],
            "eval_session_ids": ["m1_s2"],
            "per_subject": [
                {
                    "subject_id": "m1",
                    "full_session_ids": ["m1_s1", "m1_s2"],
                    "train_session_ids": ["m1_s1"],
                    "eval_session_ids": ["m1_s2"],
                }
            ],
        }

        class _FakeMiceLoaderModule:
            @staticmethod
            def resolve_mice_snapshot_session_split_manifest(**_kwargs):
                return reconstructed_manifest

        captured_calls: dict[str, dict[str, object]] = {}

        def _fake_compute(
            *,
            animal_sessions,
            simulated_sessions,
            output_dir,
            resolved_run,
            window_size,
            save_animal_session_history,
        ):
            partition = Path(output_dir).name
            captured_calls[partition] = {
                "animal_session_ids": [row["ses_idx"] for row in animal_sessions],
                "simulated_source_session_ids": [
                    row.get("source_ses_idx", row["ses_idx"]) for row in simulated_sessions
                ],
                "resolved_run_has_manifest": resolved_run.session_split_manifest is not None,
            }
            output_dir_path = Path(output_dir)
            return {
                "output_dir": str(output_dir_path),
                "simulated_session_history": str(output_dir_path / "simulated.pkl"),
                "switch_stats": str(output_dir_path / "switch.json"),
                "history_dependent_switch_stats": str(output_dir_path / "history.json"),
                "model_vs_animal_quantitative_summary": str(output_dir_path / "summary.json"),
            }

        def _fake_import_module(name):
            if name == "data_loaders.mice":
                return _FakeMiceLoaderModule()
            return __import__(name)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            animal_path = root / "animal.pkl"
            simulated_path = root / "simulated.pkl"
            resolved_run_path = root / "resolved_run.json"
            with animal_path.open("wb") as f:
                pickle.dump(animal_sessions, f)
            with simulated_path.open("wb") as f:
                pickle.dump(simulated_sessions, f)
            resolved_run_path.write_text(json.dumps(resolved_run.to_dict(), indent=2))

            with mock.patch.object(
                generative_analysis,
                "_compute_and_save_post_training_outputs",
                side_effect=_fake_compute,
            ) as mocked_compute, mock.patch.object(
                generative_analysis.importlib,
                "import_module",
                side_effect=_fake_import_module,
            ):
                result = generative_analysis.run_post_training_analysis_from_saved_histories(
                    simulated_session_history_path=simulated_path,
                    animal_session_history_path=animal_path,
                    resolved_run_path=resolved_run_path,
                    output_dir=root / "saved",
                    session_partitions=("train", "eval", "combined"),
                )

            self.assertEqual(mocked_compute.call_count, 3)
            saved_resolved_payload = json.loads(Path(result["resolved_run"]).read_text())
            self.assertEqual(
                saved_resolved_payload["session_split_manifest"]["train_session_ids"],
                ["m1_s1"],
            )
            self.assertEqual(
                captured_calls["train"]["simulated_source_session_ids"],
                ["m1_s1"],
            )
            self.assertEqual(
                captured_calls["eval"]["simulated_source_session_ids"],
                ["m1_s2"],
            )
            self.assertTrue(captured_calls["combined"]["resolved_run_has_manifest"])
            self.assertTrue(Path(result["session_partition_summary"]).exists())

    def test_run_post_training_analysis_from_saved_histories_uses_baseline_saved_manifest(self):
        resolved_run = generative_analysis.ResolvedModelRun(
            model_dir="/tmp/model",
            inputs_path="/tmp/model/inputs.yaml",
            outputs_dir="/tmp/model/outputs",
            model_type="baseline_rl",
            split="train",
            checkpoint_policy="best_eval",
            checkpoint_step=None,
            checkpoint_label="final_fit",
            params_path="/tmp/model/outputs/baseline_rl_output.json",
            config_path="/tmp/model/outputs/baseline_rl_output.json",
            baseline_output_path="/tmp/model/outputs/baseline_rl_output.json",
            seed=1,
            multisubject=True,
            mature_only=True,
            ignore_policy="exclude",
            curricula=["Uncoupled Baiting"],
            features=None,
            selection={"subject_ids": ["m1", "m2"]},
            subject_index_map_path="/tmp/model/outputs/subject_index_map.json",
            trained_subject_ids=["m1", "m2"],
            model_config={
                "multisubject": True,
                "fit_strategy": "per_subject",
                "agent_class": "ForagerQLearning",
                "fitted_params_per_subject": {
                    "m1": {
                        "subject_id": "m1",
                        "train_session_ids": ["m1_s1"],
                        "eval_session_ids": ["m1_s2"],
                        "fitted_params": {"biasL": 0.0},
                    },
                    "m2": {
                        "subject_id": "m2",
                        "train_session_ids": ["m2_s1"],
                        "eval_session_ids": ["m2_s2"],
                        "fitted_params": {"biasL": 1.0},
                    },
                },
            },
            run_config={
                "data": {"type": "mice_snapshot", "eval_every_n": 2},
                "model": {"type": "baseline_rl", "architecture": {"multisubject": True}},
            },
        )
        animal_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1",
                choice_history=[0, 0, 1],
                reward_history=[1, 0, 1],
            ),
            self._session(
                subject_id="m1",
                ses_idx="m1_s2",
                choice_history=[1, 1, 0],
                reward_history=[0, 1, 0],
            ),
            self._session(
                subject_id="m2",
                ses_idx="m2_s1",
                choice_history=[0, 1, 0],
                reward_history=[1, 0, 1],
            ),
            self._session(
                subject_id="m2",
                ses_idx="m2_s2",
                choice_history=[1, 0, 1],
                reward_history=[0, 1, 1],
            ),
        ]
        simulated_sessions = [
            self._session(
                subject_id="m1",
                ses_idx="m1_s1__rollout_0",
                source_ses_idx="m1_s1",
                choice_history=[0, 0, 1],
                reward_history=[1, 0, 1],
            ),
            self._session(
                subject_id="m1",
                ses_idx="m1_s2__rollout_0",
                source_ses_idx="m1_s2",
                choice_history=[1, 1, 0],
                reward_history=[0, 1, 0],
            ),
            self._session(
                subject_id="m2",
                ses_idx="m2_s1__rollout_0",
                source_ses_idx="m2_s1",
                choice_history=[0, 1, 0],
                reward_history=[1, 0, 1],
            ),
            self._session(
                subject_id="m2",
                ses_idx="m2_s2__rollout_0",
                source_ses_idx="m2_s2",
                choice_history=[1, 0, 1],
                reward_history=[0, 1, 1],
            ),
        ]
        captured_calls: dict[str, dict[str, object]] = {}

        def _fake_compute(
            *,
            animal_sessions,
            simulated_sessions,
            output_dir,
            resolved_run,
            window_size,
            save_animal_session_history,
        ):
            partition = Path(output_dir).name
            captured_calls[partition] = {
                "animal_session_ids": [row["ses_idx"] for row in animal_sessions],
                "simulated_source_session_ids": [
                    row.get("source_ses_idx", row["ses_idx"]) for row in simulated_sessions
                ],
                "resolved_run_has_manifest": resolved_run.session_split_manifest is not None,
            }
            output_dir_path = Path(output_dir)
            return {
                "output_dir": str(output_dir_path),
                "simulated_session_history": str(output_dir_path / "simulated.pkl"),
                "switch_stats": str(output_dir_path / "switch.json"),
                "history_dependent_switch_stats": str(output_dir_path / "history.json"),
                "model_vs_animal_quantitative_summary": str(output_dir_path / "summary.json"),
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            animal_path = root / "animal.pkl"
            simulated_path = root / "simulated.pkl"
            resolved_run_path = root / "resolved_run.json"
            with animal_path.open("wb") as f:
                pickle.dump(animal_sessions, f)
            with simulated_path.open("wb") as f:
                pickle.dump(simulated_sessions, f)
            resolved_run_path.write_text(json.dumps(resolved_run.to_dict(), indent=2))

            with mock.patch.object(
                generative_analysis,
                "_compute_and_save_post_training_outputs",
                side_effect=_fake_compute,
            ) as mocked_compute, mock.patch.object(
                generative_analysis.importlib,
                "import_module",
                side_effect=AssertionError("manifest reconstruction should not run"),
            ):
                result = generative_analysis.run_post_training_analysis_from_saved_histories(
                    simulated_session_history_path=simulated_path,
                    animal_session_history_path=animal_path,
                    resolved_run_path=resolved_run_path,
                    output_dir=root / "saved",
                    session_partitions=("train", "eval", "combined"),
                )

            self.assertEqual(mocked_compute.call_count, 3)
            saved_resolved_payload = json.loads(Path(result["resolved_run"]).read_text())
            self.assertEqual(
                saved_resolved_payload["session_split_manifest"]["train_session_ids"],
                ["m1_s1", "m2_s1"],
            )
            self.assertEqual(
                captured_calls["eval"]["simulated_source_session_ids"],
                ["m1_s2", "m2_s2"],
            )
            self.assertTrue(captured_calls["combined"]["resolved_run_has_manifest"])


if __name__ == "__main__":
    unittest.main()
