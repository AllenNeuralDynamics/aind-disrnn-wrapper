"""Unit tests for standalone likelihood advantage analysis helpers."""

from __future__ import annotations

import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - local desktop Python may be minimal
    np = None

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - local desktop Python may be minimal
    pd = None

import post_training_analysis.likelihood_advantage_analysis as likelihood_advantage_analysis
from post_training_analysis.likelihood_comparison import ResolvedLikelihoodRun


@unittest.skipIf(pd is None or np is None, "pandas or numpy is not installed")
class TestLikelihoodAdvantageAnalysis(unittest.TestCase):
    def _make_resolved_run(
        self,
        *,
        model_type: str,
        model_label: str,
        model_index: int,
    ) -> ResolvedLikelihoodRun:
        return ResolvedLikelihoodRun(
            model_dir=f"/tmp/{model_label}",
            inputs_path=f"/tmp/{model_label}/inputs.yaml",
            outputs_dir=f"/tmp/{model_label}/outputs",
            model_type=model_type,
            model_label=model_label,
            model_index=int(model_index),
            multisubject=False,
            seed=17,
            checkpoint_policy="best_eval" if model_type != "baseline_rl" else "final_fit",
            checkpoint_step=20 if model_type != "baseline_rl" else None,
            checkpoint_label="step_20" if model_type != "baseline_rl" else "final_fit",
            params_path=(
                f"/tmp/{model_label}/outputs/params.json"
                if model_type != "baseline_rl"
                else None
            ),
            baseline_output_path=(
                f"/tmp/{model_label}/outputs/baseline_rl_output.json"
                if model_type == "baseline_rl"
                else None
            ),
            artifact_selection_reason="selected",
            run_config={},
            model_config={},
        )

    def _raw_trial_rows(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "subject_id": "m1",
                    "ses_idx": "m1__s1",
                    "source_ses_idx": "s1",
                    "trial": 0,
                    "animal_response": 0,
                    "earned_reward": 1.0,
                    "curriculum_name": "A",
                },
                {
                    "subject_id": "m1",
                    "ses_idx": "m1__s1",
                    "source_ses_idx": "s1",
                    "trial": 1,
                    "animal_response": 2,
                    "earned_reward": 0.0,
                    "curriculum_name": "A",
                },
                {
                    "subject_id": "m1",
                    "ses_idx": "m1__s1",
                    "source_ses_idx": "s1",
                    "trial": 2,
                    "animal_response": 1,
                    "earned_reward": 0.0,
                    "curriculum_name": "A",
                },
                {
                    "subject_id": "m1",
                    "ses_idx": "m1__s2",
                    "source_ses_idx": "s2",
                    "trial": 0,
                    "animal_response": 1,
                    "earned_reward": 1.0,
                    "curriculum_name": "A",
                },
                {
                    "subject_id": "m2",
                    "ses_idx": "m2__t1",
                    "source_ses_idx": "t1",
                    "trial": 0,
                    "animal_response": 0,
                    "earned_reward": 0.0,
                    "curriculum_name": "B",
                },
            ]
        )

    def _model_output_rows(self) -> pd.DataFrame:
        output_df = self._raw_trial_rows().copy()
        output_df["choice_prob_0"] = [0.8, 0.2, 0.3, 0.1, 0.7]
        output_df["choice_prob_1"] = [0.2, 0.3, 0.7, 0.9, 0.3]
        output_df["choice_prob_2"] = [0.0, 0.5, 0.0, 0.0, 0.0]
        output_df["latent_0"] = [1.0, 2.0, 3.0, 4.0, 5.0]
        output_df["latent_1"] = [10.0, 20.0, 30.0, 40.0, 50.0]
        return output_df

    def _single_session_trial_df(self) -> pd.DataFrame:
        actions = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]
        rewards = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        return pd.DataFrame.from_records(
            [
                {
                    "subject_id": "m1",
                    "ses_idx": "m1__s1",
                    "session_id": "s1",
                    "session_idx": 1,
                    "trial_idx": trial_idx,
                    "action": action,
                    "reward": reward,
                    "curriculum_name": "A",
                    "advantage": float(trial_idx) / 10.0,
                }
                for trial_idx, (action, reward) in enumerate(
                    zip(actions, rewards),
                    start=1,
                )
            ]
        )

    def _smoke_raw_df(self) -> pd.DataFrame:
        records = []
        curricula = {"m1": "A", "m2": "B", "m3": "A"}
        for subject_offset, subject_id in enumerate(["m1", "m2", "m3"]):
            for session_idx in range(1, 4):
                ses_idx = f"{subject_id}__s{session_idx}"
                for trial in range(12):
                    action = int((trial + session_idx + subject_offset) % 2)
                    reward = float(((trial + subject_offset) % 3) == 0)
                    records.append(
                        {
                            "subject_id": subject_id,
                            "ses_idx": ses_idx,
                            "source_ses_idx": f"s{session_idx}",
                            "trial": trial,
                            "animal_response": action,
                            "earned_reward": reward,
                            "curriculum_name": curricula[subject_id],
                        }
                    )
        return pd.DataFrame.from_records(records)

    def _smoke_output_df(self) -> pd.DataFrame:
        output_df = self._smoke_raw_df().copy()
        p_chosen = []
        p_other = []
        for action, trial in zip(
            output_df["animal_response"].tolist(),
            output_df["trial"].tolist(),
        ):
            chosen = max(0.55, 0.82 - 0.01 * int(trial))
            other = 1.0 - chosen
            if int(action) == 0:
                p_chosen.append(chosen)
                p_other.append(other)
            else:
                p_chosen.append(other)
                p_other.append(chosen)
        output_df["choice_prob_0"] = p_chosen
        output_df["choice_prob_1"] = p_other
        output_df["latent_0"] = np.linspace(0.0, 1.0, len(output_df))
        output_df["latent_1"] = np.linspace(1.0, 2.0, len(output_df))
        output_df["latent_2"] = np.linspace(2.0, 3.0, len(output_df))
        output_df["latent_3"] = np.linspace(3.0, 4.0, len(output_df))
        return output_df

    def test_build_canonical_trial_dataframe_uses_one_based_indices(self):
        trial_df = likelihood_advantage_analysis._build_canonical_trial_dataframe(
            self._raw_trial_rows(),
            metadata={},
        )

        self.assertEqual(len(trial_df), 4)
        self.assertEqual(trial_df["session_id"].tolist(), ["s1", "s1", "s2", "t1"])
        self.assertEqual(trial_df["trial_idx"].tolist(), [1, 2, 1, 1])
        self.assertEqual(trial_df["session_idx"].tolist(), [1, 1, 2, 1])
        self.assertEqual(trial_df["action"].tolist(), [0, 1, 1, 0])
        self.assertEqual(trial_df["curriculum_name"].tolist(), ["A", "A", "A", "B"])

    def test_extract_model1_probability_frame_skips_non_binary_trials(self):
        prob_df = likelihood_advantage_analysis._extract_model1_probability_frame(
            self._model_output_rows(),
            n_action_logits=2,
        )

        self.assertEqual(prob_df["trial_idx"].tolist(), [1, 2, 1, 1])
        self.assertEqual(prob_df["p_model1"].tolist(), [0.8, 0.7, 0.9, 0.7])
        self.assertEqual(prob_df["p_model1_left"].tolist(), [0.8, 0.3, 0.1, 0.7])
        self.assertEqual(prob_df["p_model1_right"].tolist(), [0.2, 0.7, 0.9, 0.3])

    def test_extract_model1_rnn_state_frame_sorts_and_renames_latents(self):
        output_df = self._model_output_rows().copy()
        output_df["latent_10"] = [100.0, 200.0, 300.0, 400.0, 500.0]

        state_df = likelihood_advantage_analysis._extract_model1_rnn_state_frame(
            output_df,
        )

        self.assertEqual(
            [column for column in state_df.columns if column.startswith("rnn_state_")],
            ["rnn_state_0", "rnn_state_1", "rnn_state_2"],
        )
        self.assertEqual(state_df["trial_idx"].tolist(), [1, 2, 1, 1])
        self.assertEqual(state_df["rnn_state_0"].tolist(), [1.0, 3.0, 4.0, 5.0])
        self.assertEqual(state_df["rnn_state_2"].tolist(), [100.0, 300.0, 400.0, 500.0])

    def test_merge_rnn_state_frame_rejects_action_mismatch(self):
        canonical_df = likelihood_advantage_analysis._build_canonical_trial_dataframe(
            self._raw_trial_rows(),
            metadata={},
        )
        state_df = likelihood_advantage_analysis._extract_model1_rnn_state_frame(
            self._model_output_rows(),
        )
        state_df.loc[0, "action"] = 1

        with self.assertRaisesRegex(ValueError, "Action mismatch"):
            likelihood_advantage_analysis._merge_rnn_state_frame(
                canonical_df,
                state_df,
                source_label="gru",
            )

    def test_merge_and_advantage_computation_matches_expected_logs(self):
        canonical_df = likelihood_advantage_analysis._build_canonical_trial_dataframe(
            self._raw_trial_rows(),
            metadata={},
        )
        model1_prob_df = likelihood_advantage_analysis._extract_model1_probability_frame(
            self._model_output_rows(),
            n_action_logits=2,
        )
        baseline_prob_df = pd.DataFrame.from_records(
            [
                {"subject_id": "m1", "ses_idx": "m1__s1", "trial_idx": 1, "action": 0, "p_rl": 0.4},
                {"subject_id": "m1", "ses_idx": "m1__s1", "trial_idx": 2, "action": 1, "p_rl": 0.6},
                {"subject_id": "m1", "ses_idx": "m1__s2", "trial_idx": 1, "action": 1, "p_rl": 0.5},
                {"subject_id": "m2", "ses_idx": "m2__t1", "trial_idx": 1, "action": 0, "p_rl": 0.3},
            ]
        )

        merged_df = likelihood_advantage_analysis._merge_probability_frame(
            canonical_df,
            model1_prob_df,
            prob_column="p_model1",
            source_label="gru",
        )
        self.assertIn("p_model1_left", merged_df.columns)
        self.assertIn("p_model1_right", merged_df.columns)
        merged_df = likelihood_advantage_analysis._merge_probability_frame(
            merged_df,
            baseline_prob_df,
            prob_column="p_rl",
            source_label="baseline_rl",
        )
        merged_df["p_model1"] = np.clip(merged_df["p_model1"], 1e-6, 1 - 1e-6)
        merged_df["p_rl"] = np.clip(merged_df["p_rl"], 1e-6, 1 - 1e-6)
        merged_df["advantage"] = np.log(merged_df["p_model1"]) - np.log(merged_df["p_rl"])

        expected = [
            math.log(0.8) - math.log(0.4),
            math.log(0.7) - math.log(0.6),
            math.log(0.9) - math.log(0.5),
            math.log(0.7) - math.log(0.3),
        ]
        self.assertTrue(np.allclose(merged_df["advantage"].to_numpy(dtype=float), expected))

    def test_probability_frame_from_rollout_sessions_includes_baseline_probabilities_and_q_values(self):
        session_payloads = [
            {
                "subject_id": "m1",
                "ses_idx": "m1__s1",
                "choices": np.array([0, 1, 0]),
                "rewards": np.array([1.0, 0.0, 1.0]),
                "trial_indices": np.array([1, 2, 3]),
            }
        ]
        choice_prob_sessions = [
            np.array(
                [
                    [0.7, 0.2, 0.4],
                    [0.3, 0.8, 0.6],
                ]
            )
        ]
        q_value_sessions = [
            np.array(
                [
                    [1.0, 2.0, 3.0, 99.0],
                    [4.0, 5.0, 6.0, 99.0],
                ]
            )
        ]

        baseline_df = likelihood_advantage_analysis._probability_frame_from_rollout_sessions(
            session_payloads,
            choice_prob_sessions=choice_prob_sessions,
            q_value_sessions=q_value_sessions,
        )

        self.assertEqual(
            baseline_df.columns.tolist(),
            likelihood_advantage_analysis._BASELINE_PROBABILITY_FRAME_COLUMNS,
        )
        self.assertEqual(baseline_df["p_rl"].tolist(), [0.7, 0.8, 0.4])
        self.assertEqual(baseline_df["p_rl_left"].tolist(), [0.7, 0.2, 0.4])
        self.assertEqual(baseline_df["p_rl_right"].tolist(), [0.3, 0.8, 0.6])
        self.assertEqual(baseline_df["q_rl_left"].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(baseline_df["q_rl_right"].tolist(), [4.0, 5.0, 6.0])

    def test_align_policy_time_q_session_supports_trial_and_prepost_histories(self):
        trial_aligned, trial_mode = likelihood_advantage_analysis._align_policy_time_q_session(
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            n_trials=2,
        )
        prepost_aligned, prepost_mode = likelihood_advantage_analysis._align_policy_time_q_session(
            np.array([[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]]),
            n_trials=2,
        )

        self.assertEqual(trial_mode, "trial_aligned")
        self.assertEqual(prepost_mode, "prepost_first_n")
        self.assertTrue(np.allclose(trial_aligned, [[1.0, 2.0], [3.0, 4.0]]))
        self.assertTrue(np.allclose(prepost_aligned, [[1.0, 3.0], [2.0, 4.0]]))

    def test_perform_baseline_agent_rollout_with_q_histories_uses_mock_agent(self):
        class FakeBaselineAgent:
            def __init__(self, **kwargs):
                del kwargs
                self.q_value_history = None
                self.params = {}

            def set_params(self, **kwargs):
                self.params = kwargs

            def perform_closed_loop_multi_session(self, choice_sessions, reward_sessions):
                del reward_sessions
                self.q_value_history = [
                    np.array(
                        [
                            [1.0, 2.0, 99.0],
                            [3.0, 4.0, 99.0],
                        ]
                    )
                    for _ in choice_sessions
                ]
                return [
                    np.array(
                        [
                            [0.8, 0.25],
                            [0.2, 0.75],
                        ]
                    )
                    for _ in choice_sessions
                ]

        package = types.ModuleType("aind_dynamic_foraging_models")
        package.generative_model = types.SimpleNamespace(FakeBaselineAgent=FakeBaselineAgent)
        with mock.patch.dict(sys.modules, {"aind_dynamic_foraging_models": package}):
            choice_prob_sessions, q_value_sessions, q_alignment = (
                likelihood_advantage_analysis._perform_baseline_agent_rollout_with_q_histories(
                    agent_class_name="FakeBaselineAgent",
                    agent_kwargs={},
                    fitted_params={"alpha": 0.2},
                    choice_sessions=[np.array([0, 1])],
                    reward_sessions=[np.array([1.0, 0.0])],
                    seed=0,
                    require_q_values=True,
                )
            )

        self.assertEqual(len(choice_prob_sessions), 1)
        self.assertEqual(len(q_value_sessions), 1)
        self.assertEqual(q_alignment["q_source"], "agent_exposed_history")
        self.assertEqual(q_alignment["alignment"], "prepost_first_n")
        self.assertTrue(np.allclose(q_value_sessions[0], [[1.0, 2.0], [3.0, 4.0]]))

    def test_exposed_q_history_wins_over_manual_forager_fallback(self):
        class ForagerQLearning:
            def __init__(self, **kwargs):
                del kwargs
                self.q_value_history = None

            def set_params(self, **kwargs):
                del kwargs

            def perform_closed_loop_multi_session(self, choice_sessions, reward_sessions):
                del reward_sessions
                self.q_value_history = [
                    np.array([[0.5, 0.6], [0.5, 0.4]])
                    for _ in choice_sessions
                ]
                return [
                    np.array([[0.5, 0.6], [0.5, 0.4]])
                    for _ in choice_sessions
                ]

        package = types.ModuleType("aind_dynamic_foraging_models")
        package.generative_model = types.SimpleNamespace(ForagerQLearning=ForagerQLearning)
        with mock.patch.dict(sys.modules, {"aind_dynamic_foraging_models": package}), mock.patch.object(
            likelihood_advantage_analysis,
            "_manual_forager_q_learning_q_histories",
            side_effect=AssertionError("manual fallback should not run"),
        ):
            _, q_value_sessions, q_alignment = (
                likelihood_advantage_analysis._perform_baseline_agent_rollout_with_q_histories(
                    agent_class_name="ForagerQLearning",
                    agent_kwargs={
                        "action_selection": "softmax",
                        "choice_kernel": "none",
                    },
                    fitted_params={"learn_rate": 0.1},
                    choice_sessions=[np.array([0, 1])],
                    reward_sessions=[np.array([1.0, 0.0])],
                    seed=0,
                    require_q_values=True,
                )
            )

        self.assertEqual(q_alignment["q_source"], "agent_exposed_history")
        self.assertTrue(np.allclose(q_value_sessions[0], [[0.5, 0.6], [0.5, 0.4]]))

    def test_manual_forager_q_learning_fallback_produces_policy_time_q_values(self):
        agent_kwargs = {
            "action_selection": "softmax",
            "choice_kernel": "none",
            "number_of_learning_rate": 2,
            "number_of_forget_rate": 1,
        }
        fitted_params = {
            "learn_rate_rew": 0.5,
            "learn_rate_unrew": 0.25,
            "forget_rate_unchosen": 0.1,
            "softmax_inverse_temperature": 2.0,
            "biasL": 0.0,
        }

        class ForagerQLearning:
            def __init__(self, **kwargs):
                del kwargs
                self.params = {}

            def set_params(self, **kwargs):
                self.params = kwargs

            def perform_closed_loop_multi_session(self, choice_sessions, reward_sessions):
                probabilities = []
                for choices, rewards in zip(choice_sessions, reward_sessions):
                    q_values = (
                        likelihood_advantage_analysis._manual_forager_q_learning_session_q_values(
                            choices=choices,
                            rewards=rewards,
                            fitted_params=self.params,
                            number_of_learning_rate=2,
                            number_of_forget_rate=1,
                            forget_rate=self.params["forget_rate_unchosen"],
                            initial_q=np.array([0.5, 0.5]),
                        )
                    )
                    probabilities.append(
                        likelihood_advantage_analysis._manual_forager_q_learning_softmax_probabilities(
                            q_values,
                            fitted_params=self.params,
                        )
                    )
                return probabilities

        package = types.ModuleType("aind_dynamic_foraging_models")
        package.generative_model = types.SimpleNamespace(ForagerQLearning=ForagerQLearning)
        with mock.patch.dict(sys.modules, {"aind_dynamic_foraging_models": package}):
            choice_prob_sessions, q_value_sessions, q_alignment = (
                likelihood_advantage_analysis._perform_baseline_agent_rollout_with_q_histories(
                    agent_class_name="ForagerQLearning",
                    agent_kwargs=agent_kwargs,
                    fitted_params=fitted_params,
                    choice_sessions=[np.array([0, 1, 0])],
                    reward_sessions=[np.array([1.0, 0.0, 1.0])],
                    seed=0,
                    require_q_values=True,
                )
            )

        self.assertEqual(q_alignment["q_source"], "manual_forager_q_learning")
        self.assertEqual(q_alignment["validation"], "manual_validation_passed")
        self.assertEqual(len(choice_prob_sessions), 1)
        self.assertTrue(
            np.allclose(
                q_value_sessions[0],
                np.array([[0.5, 0.75, 0.725], [0.5, 0.5, 0.375]]),
            )
        )

    def test_manual_forager_q_learning_preserves_live_agent_probabilities_for_one_step(self):
        class ForagerQLearning:
            def __init__(self, **kwargs):
                del kwargs

            def set_params(self, **kwargs):
                self.params = kwargs

            def perform_closed_loop_multi_session(self, choice_sessions, reward_sessions):
                del reward_sessions
                return [
                    np.array([[0.9, 0.1], [0.1, 0.9]])
                    for _ in choice_sessions
                ]

        session_payloads = [
            {
                "subject_id": "m1",
                "ses_idx": "s1",
                "choices": np.array([0, 1]),
                "rewards": np.array([1.0, 0.0]),
                "trial_indices": np.array([1, 2]),
            }
        ]
        package = types.ModuleType("aind_dynamic_foraging_models")
        package.generative_model = types.SimpleNamespace(ForagerQLearning=ForagerQLearning)
        with mock.patch.dict(sys.modules, {"aind_dynamic_foraging_models": package}):
            choice_prob_sessions, q_value_sessions, q_alignment = (
                likelihood_advantage_analysis._perform_baseline_agent_rollout_with_q_histories(
                    agent_class_name="ForagerQLearning",
                    agent_kwargs={
                        "action_selection": "softmax",
                        "choice_kernel": "one_step",
                        "number_of_learning_rate": 1,
                        "number_of_forget_rate": 0,
                    },
                    fitted_params={"learn_rate": 0.5},
                    choice_sessions=[session_payloads[0]["choices"]],
                    reward_sessions=[session_payloads[0]["rewards"]],
                    seed=0,
                    require_q_values=True,
                )
            )

        baseline_df = likelihood_advantage_analysis._probability_frame_from_rollout_sessions(
            session_payloads,
            choice_prob_sessions=choice_prob_sessions,
            q_value_sessions=q_value_sessions,
        )
        self.assertEqual(q_alignment["validation"], "manual_validation_skipped_choice_kernel")
        self.assertEqual(baseline_df["p_rl"].tolist(), [0.9, 0.9])
        self.assertEqual(baseline_df["p_rl_left"].tolist(), [0.9, 0.1])
        self.assertEqual(baseline_df["p_rl_right"].tolist(), [0.1, 0.9])

    def test_manual_forager_q_learning_validates_softmax_probability_mismatch(self):
        with self.assertRaisesRegex(ValueError, "did not reproduce live-agent"):
            likelihood_advantage_analysis._manual_forager_q_learning_q_histories(
                agent_kwargs={
                    "action_selection": "softmax",
                    "choice_kernel": "none",
                    "number_of_learning_rate": 1,
                    "number_of_forget_rate": 0,
                },
                fitted_params={
                    "learn_rate": 0.5,
                    "softmax_inverse_temperature": 2.0,
                    "biasL": 0.0,
                },
                choice_sessions=[np.array([0, 1])],
                reward_sessions=[np.array([1.0, 0.0])],
                choice_prob_sessions=[np.array([[0.5, 0.5], [0.5, 0.5]])],
            )

    def test_manual_forager_q_learning_skips_validation_for_one_step(self):
        q_value_sessions, q_alignment = (
            likelihood_advantage_analysis._manual_forager_q_learning_q_histories(
                agent_kwargs={
                    "action_selection": "softmax",
                    "choice_kernel": "one_step",
                    "number_of_learning_rate": 1,
                    "number_of_forget_rate": 0,
                },
                fitted_params={"learn_rate": 0.5},
                choice_sessions=[np.array([0, 1])],
                reward_sessions=[np.array([1.0, 0.0])],
                choice_prob_sessions=[np.array([[0.99, 0.01], [0.01, 0.99]])],
            )
        )

        self.assertEqual(q_alignment["q_source"], "manual_forager_q_learning")
        self.assertEqual(q_alignment["validation"], "manual_validation_skipped_choice_kernel")
        self.assertTrue(np.allclose(q_value_sessions[0], [[0.5, 0.75], [0.5, 0.5]]))

    def test_unsupported_baseline_agent_fails_clearly_when_q_space_requested(self):
        class ForagerCompareThreshold:
            def __init__(self, **kwargs):
                del kwargs

            def set_params(self, **kwargs):
                del kwargs

            def perform_closed_loop_multi_session(self, choice_sessions, reward_sessions):
                del reward_sessions
                return [np.ones((2, len(session))) * 0.5 for session in choice_sessions]

        package = types.ModuleType("aind_dynamic_foraging_models")
        package.generative_model = types.SimpleNamespace(
            ForagerCompareThreshold=ForagerCompareThreshold
        )
        with mock.patch.dict(sys.modules, {"aind_dynamic_foraging_models": package}):
            with self.assertRaisesRegex(ValueError, "include_baseline_q_space=False"):
                likelihood_advantage_analysis._perform_baseline_agent_rollout_with_q_histories(
                    agent_class_name="ForagerCompareThreshold",
                    agent_kwargs={},
                    fitted_params={"biasL": 0.0},
                    choice_sessions=[np.array([0, 1])],
                    reward_sessions=[np.array([1.0, 0.0])],
                    seed=0,
                    require_q_values=True,
                )

    def test_q_space_not_requested_skips_q_recovery_for_unsupported_agent(self):
        class ForagerCompareThreshold:
            def __init__(self, **kwargs):
                del kwargs

            def set_params(self, **kwargs):
                del kwargs

            def perform_closed_loop_multi_session(self, choice_sessions, reward_sessions):
                del reward_sessions
                return [np.ones((2, len(session))) * 0.5 for session in choice_sessions]

        package = types.ModuleType("aind_dynamic_foraging_models")
        package.generative_model = types.SimpleNamespace(
            ForagerCompareThreshold=ForagerCompareThreshold
        )
        with mock.patch.dict(sys.modules, {"aind_dynamic_foraging_models": package}):
            choice_prob_sessions, q_value_sessions, q_alignment = (
                likelihood_advantage_analysis._perform_baseline_agent_rollout_with_q_histories(
                    agent_class_name="ForagerCompareThreshold",
                    agent_kwargs={},
                    fitted_params={"biasL": 0.0},
                    choice_sessions=[np.array([0, 1])],
                    reward_sessions=[np.array([1.0, 0.0])],
                    seed=0,
                    require_q_values=False,
                )
            )

        self.assertEqual(len(choice_prob_sessions), 1)
        self.assertIsNone(q_value_sessions)
        self.assertEqual(q_alignment["q_source"], "not_requested")

    def test_extract_policy_time_q_histories_rejects_missing_q_history(self):
        class NoQAgent:
            pass

        with self.assertRaisesRegex(ValueError, "Could not recover policy-time"):
            likelihood_advantage_analysis._extract_policy_time_q_histories(
                NoQAgent(),
                choice_prob_sessions=[np.ones((2, 3)) * 0.5],
                expected_trials_per_session=[3],
            )

    def test_add_candidate_variables_applies_warmup_and_history_features(self):
        enriched_df = likelihood_advantage_analysis.add_candidate_variables(
            self._single_session_trial_df(),
            history_warmup=10,
        )

        self.assertTrue(enriched_df.loc[:9, "prev_outcome"].isna().all())
        self.assertEqual(float(enriched_df.loc[10, "prev_outcome"]), 0.0)
        self.assertEqual(float(enriched_df.loc[10, "prev_action"]), 1.0)
        self.assertEqual(float(enriched_df.loc[10, "switch"]), 1.0)
        self.assertEqual(enriched_df.loc[10, "history_pattern_1"], "r")
        self.assertEqual(enriched_df.loc[10, "history_pattern_2"], "Rr")
        self.assertEqual(enriched_df.loc[10, "history_pattern_3"], "rRr")
        self.assertAlmostEqual(float(enriched_df.loc[10, "recent_reward_rate_3"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(enriched_df.loc[10, "recent_reward_rate_5"]), 0.4)
        self.assertAlmostEqual(float(enriched_df.loc[10, "recent_reward_rate_10"]), 0.5)
        self.assertEqual(float(enriched_df.loc[10, "trials_since_reward"]), 1.0)
        self.assertAlmostEqual(float(enriched_df.loc[10, "recent_switch_rate_5"]), 0.6)
        self.assertEqual(str(enriched_df.loc[10, "switch_x_prev_outcome"]), "unrewarded-switch")
        self.assertAlmostEqual(float(enriched_df.loc[0, "trial_position"]), 0.0)
        self.assertAlmostEqual(float(enriched_df.loc[11, "trial_position"]), 1.0)

    def test_trial_position_is_zero_for_single_trial_session(self):
        trial_df = pd.DataFrame.from_records(
            [
                {
                    "subject_id": "m1",
                    "ses_idx": "m1__s1",
                    "session_id": "s1",
                    "session_idx": 1,
                    "trial_idx": 1,
                    "action": 0,
                    "reward": 1.0,
                    "curriculum_name": "A",
                    "advantage": 0.0,
                }
            ]
        )
        enriched_df = likelihood_advantage_analysis.add_candidate_variables(
            trial_df,
            history_warmup=0,
        )
        self.assertAlmostEqual(float(enriched_df.loc[0, "trial_position"]), 0.0)

    def test_analyze_variable_bins_continuous_uses_quantile_bins(self):
        trial_df = pd.DataFrame(
            {
                "trial_position": np.linspace(0.0, 0.9, 10),
                "advantage": np.arange(10, dtype=float),
            }
        )

        summary_df = likelihood_advantage_analysis.analyze_variable_bins(
            trial_df,
            "trial_position",
        )

        self.assertEqual(len(summary_df), 5)
        self.assertEqual(int(summary_df["n_trials"].sum()), 10)
        self.assertTrue((summary_df["n_trials"] == 2).all())

    def test_analyze_variable_bins_handles_tied_values_and_grouping(self):
        trial_df = pd.DataFrame(
            {
                "trial_position": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                "advantage": [0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0],
                "session_stage": ["early"] * 4 + ["late"] * 4,
            }
        )

        summary_df = likelihood_advantage_analysis.analyze_variable_bins(
            trial_df,
            "trial_position",
            groupby_cols=["session_stage"],
        )

        self.assertIn("session_stage", summary_df.columns)
        self.assertLessEqual(int(summary_df["bin"].nunique()), 2)
        self.assertEqual(set(summary_df["session_stage"].astype(str)), {"early", "late"})

    def test_add_session_stage_matches_expected_boundaries(self):
        trial_df = pd.DataFrame.from_records(
            [
                {"subject_id": "one", "ses_idx": "one__s1", "session_idx": 1, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "two", "ses_idx": "two__s1", "session_idx": 1, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "two", "ses_idx": "two__s2", "session_idx": 2, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "three", "ses_idx": "three__s1", "session_idx": 1, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "three", "ses_idx": "three__s2", "session_idx": 2, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "three", "ses_idx": "three__s3", "session_idx": 3, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "four", "ses_idx": "four__s1", "session_idx": 1, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "four", "ses_idx": "four__s2", "session_idx": 2, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "four", "ses_idx": "four__s3", "session_idx": 3, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "four", "ses_idx": "four__s4", "session_idx": 4, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
            ]
        )

        staged_df = likelihood_advantage_analysis._add_session_stage(trial_df)
        stage_map = {
            (row["subject_id"], row["ses_idx"]): str(row["session_stage"])
            for row in staged_df.loc[:, ["subject_id", "ses_idx", "session_stage"]]
            .drop_duplicates()
            .to_dict(orient="records")
        }

        self.assertEqual(stage_map[("one", "one__s1")], "early")
        self.assertEqual(stage_map[("two", "two__s1")], "early")
        self.assertEqual(stage_map[("two", "two__s2")], "late")
        self.assertEqual(stage_map[("three", "three__s1")], "early")
        self.assertEqual(stage_map[("three", "three__s2")], "mid")
        self.assertEqual(stage_map[("three", "three__s3")], "late")
        self.assertEqual(stage_map[("four", "four__s2")], "mid")
        self.assertEqual(stage_map[("four", "four__s3")], "late")

    def test_build_subject_mean_advantage_df_is_seed_stable(self):
        trial_df = pd.DataFrame.from_records(
            [
                {"subject_id": "m1", "advantage": 0.2, "curriculum_name": "A"},
                {"subject_id": "m1", "advantage": 0.4, "curriculum_name": "A"},
                {"subject_id": "m2", "advantage": 0.3, "curriculum_name": "Mixed"},
                {"subject_id": "m2", "advantage": 0.1, "curriculum_name": "Mixed"},
                {"subject_id": "m3", "advantage": 0.0, "curriculum_name": "Unknown"},
            ]
        )

        with mock.patch.object(
            likelihood_advantage_analysis.lc,
            "_build_curriculum_palette",
            return_value={
                "A": "#111111",
                "Mixed": "#222222",
                "Unknown": "#bdbdbd",
            },
        ):
            first_df = likelihood_advantage_analysis._build_subject_mean_advantage_df(
                trial_df,
                jitter_seed=0,
            )
            second_df = likelihood_advantage_analysis._build_subject_mean_advantage_df(
                trial_df,
                jitter_seed=0,
            )
            third_df = likelihood_advantage_analysis._build_subject_mean_advantage_df(
                trial_df,
                jitter_seed=1,
            )

        self.assertEqual(len(first_df), 3)
        self.assertTrue(
            np.allclose(
                first_df["x_jitter"].to_numpy(dtype=float),
                second_df["x_jitter"].to_numpy(dtype=float),
            )
        )
        self.assertFalse(
            np.allclose(
                first_df["x_jitter"].to_numpy(dtype=float),
                third_df["x_jitter"].to_numpy(dtype=float),
            )
        )
        color_map = dict(zip(first_df["curriculum_name"], first_df["color"]))
        self.assertEqual(
            color_map["Unknown"],
            likelihood_advantage_analysis.lc._DEFAULT_CURRICULUM_COLORS["Unknown"],
        )
        self.assertTrue(str(color_map["Mixed"]).startswith("#"))

    def test_fit_and_project_rnn_state_pca_is_seed_stable(self):
        trial_df = pd.DataFrame(
            {
                "advantage": np.linspace(-1.0, 1.0, 8),
                "rnn_state_0": np.arange(8, dtype=float),
                "rnn_state_1": np.arange(8, dtype=float) * 2.0,
                "rnn_state_2": np.arange(8, dtype=float) % 3,
            }
        )

        first = likelihood_advantage_analysis._fit_and_project_rnn_state_pca(
            trial_df,
            state_columns=["rnn_state_0", "rnn_state_1", "rnn_state_2"],
            pca_seed=0,
            pca_fit_fraction=0.5,
        )
        second = likelihood_advantage_analysis._fit_and_project_rnn_state_pca(
            trial_df,
            state_columns=["rnn_state_0", "rnn_state_1", "rnn_state_2"],
            pca_seed=0,
            pca_fit_fraction=0.5,
        )

        self.assertEqual(first["fit_n_trials"], 4)
        self.assertEqual(first["scores"].shape[0], len(trial_df))
        self.assertTrue(np.allclose(first["scores"], second["scores"]))
        self.assertLessEqual(
            float(first["cumulative_explained_variance_ratio"][-1]),
            1.0 + 1e-9,
        )

    def test_build_state_condition_specs_quantile_bins_continuous_columns(self):
        trial_df = pd.DataFrame(
            {
                "trial_position": np.linspace(0.0, 0.9, 10),
                "switch": [0, 1] * 5,
            }
        )

        condition_specs = likelihood_advantage_analysis._build_state_condition_specs(
            trial_df,
            condition_columns=["trial_position", "switch"],
            condition_values_by_column={},
        )

        self.assertEqual(len(condition_specs["trial_position"]), 5)
        self.assertEqual(
            {entry["label"] for entry in condition_specs["switch"]},
            {"0", "1"},
        )

    def test_run_rnn_state_space_condition_analysis_from_pickle(self):
        trial_df = pd.DataFrame(
            {
                "advantage": np.linspace(-0.5, 0.5, 12),
                "trial_position": np.linspace(0.0, 1.0, 12),
                "switch": [0, 1] * 6,
                "rnn_state_0": np.linspace(0.0, 1.0, 12),
                "rnn_state_1": np.linspace(1.0, 0.0, 12),
                "rnn_state_2": np.sin(np.linspace(0.0, 1.0, 12)),
                "rnn_state_3": np.cos(np.linspace(0.0, 1.0, 12)),
            }
        )

        def _write_placeholder_plot(*args, output_path: Path, **kwargs):
            del args
            del kwargs
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("plot")

        with tempfile.TemporaryDirectory(prefix="rnn_state_space_") as tmpdir, mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_rnn_state_pca_variance",
            side_effect=_write_placeholder_plot,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_rnn_state_condition_figure",
            side_effect=_write_placeholder_plot,
        ):
            pickle_path = Path(tmpdir) / "trial_advantage.pkl"
            trial_df.to_pickle(pickle_path)
            result = likelihood_advantage_analysis.run_rnn_state_space_condition_analysis(
                pickle_path,
                condition_columns=["switch", "trial_position"],
                output_dir=Path(tmpdir) / "state_plots",
                pca_seed=0,
            )

            self.assertTrue(Path(result["rnn_state_pca_variance"]).exists())
            self.assertTrue(Path(result["rnn_state_pca_variance_csv"]).exists())
            self.assertEqual(result["condition_columns"], ["switch", "trial_position"])
            self.assertIn("switch", result["rnn_state_condition_plots"])
            self.assertIn("trial_position", result["rnn_state_condition_plots"])

    def test_run_rnn_state_space_subject_analysis_from_pickle(self):
        trial_df = pd.DataFrame(
            {
                "subject_id": ["m1"] * 6 + ["m2"] * 6,
                "p_model1_left": np.linspace(0.1, 0.9, 12),
                "p_model1_right": np.linspace(0.9, 0.1, 12),
                "rnn_state_0": np.linspace(0.0, 1.0, 12),
                "rnn_state_1": np.linspace(1.0, 0.0, 12),
                "rnn_state_2": np.sin(np.linspace(0.0, 1.0, 12)),
                "rnn_state_3": np.cos(np.linspace(0.0, 1.0, 12)),
            }
        )

        def _write_placeholder_plot(*args, output_path: Path, **kwargs):
            del args
            del kwargs
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("plot")

        with tempfile.TemporaryDirectory(prefix="rnn_state_subjects_") as tmpdir, mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_rnn_state_pca_variance",
            side_effect=_write_placeholder_plot,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_rnn_state_subject_probability_figure",
            side_effect=_write_placeholder_plot,
        ):
            pickle_path = Path(tmpdir) / "trial_advantage.pkl"
            trial_df.to_pickle(pickle_path)
            result = likelihood_advantage_analysis.run_rnn_state_space_subject_analysis(
                pickle_path,
                probability_column="p_model1_right",
                subject_ids=["m2"],
                output_dir=Path(tmpdir) / "subject_plots",
                pca_seed=0,
            )

            self.assertEqual(result["probability_column"], "p_model1_right")
            self.assertEqual(result["subject_ids"], ["m2"])
            self.assertEqual(set(result["subject_probability_plots"].keys()), {"m2"})
            self.assertTrue(Path(result["rnn_state_pca_variance"]).exists())
            self.assertTrue(Path(result["rnn_state_pca_variance_csv"]).exists())
            self.assertTrue(Path(result["subject_probability_plots"]["m2"]).exists())

    def test_run_rnn_state_space_subject_analysis_validates_required_columns(self):
        base_df = pd.DataFrame(
            {
                "subject_id": ["m1", "m1"],
                "p_model1_left": [0.2, 0.8],
                "rnn_state_0": [0.0, 1.0],
                "rnn_state_1": [1.0, 0.0],
            }
        )

        with tempfile.TemporaryDirectory(prefix="rnn_state_subjects_missing_") as tmpdir:
            missing_probability_path = Path(tmpdir) / "missing_probability.pkl"
            base_df.drop(columns=["p_model1_left"]).to_pickle(missing_probability_path)
            with self.assertRaisesRegex(ValueError, "p_model1_left"):
                likelihood_advantage_analysis.run_rnn_state_space_subject_analysis(
                    missing_probability_path,
                    probability_column="p_model1_left",
                    output_dir=Path(tmpdir) / "missing_probability",
                )

            missing_subject_path = Path(tmpdir) / "missing_subject.pkl"
            base_df.drop(columns=["subject_id"]).to_pickle(missing_subject_path)
            with self.assertRaisesRegex(ValueError, "subject_id"):
                likelihood_advantage_analysis.run_rnn_state_space_subject_analysis(
                    missing_subject_path,
                    probability_column="p_model1_left",
                    output_dir=Path(tmpdir) / "missing_subject",
                )

            missing_state_path = Path(tmpdir) / "missing_state.pkl"
            base_df.drop(columns=["rnn_state_0", "rnn_state_1"]).to_pickle(
                missing_state_path
            )
            with self.assertRaisesRegex(ValueError, "rnn_state_"):
                likelihood_advantage_analysis.run_rnn_state_space_subject_analysis(
                    missing_state_path,
                    probability_column="p_model1_left",
                    output_dir=Path(tmpdir) / "missing_state",
                )

    def test_run_baseline_q_space_condition_analysis_from_pickle(self):
        trial_df = pd.DataFrame(
            {
                "advantage": np.linspace(-0.5, 0.5, 12),
                "trial_position": np.linspace(0.0, 1.0, 12),
                "switch": [0, 1] * 6,
                "q_rl_left": np.linspace(0.0, 1.0, 12),
                "q_rl_right": np.linspace(1.0, 0.0, 12),
            }
        )

        def _write_placeholder_plot(*args, output_path: Path, **kwargs):
            del args
            del kwargs
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("plot")

        with tempfile.TemporaryDirectory(prefix="baseline_q_space_") as tmpdir, mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_baseline_q_space_condition_figure",
            side_effect=_write_placeholder_plot,
        ):
            pickle_path = Path(tmpdir) / "trial_advantage.pkl"
            trial_df.to_pickle(pickle_path)
            result = likelihood_advantage_analysis.run_baseline_q_space_condition_analysis(
                pickle_path,
                condition_columns=["switch", "trial_position"],
                output_dir=Path(tmpdir) / "q_plots",
            )

            self.assertTrue(Path(result["summary"]).exists())
            self.assertEqual(result["condition_columns"], ["switch", "trial_position"])
            self.assertIn("switch", result["baseline_q_condition_plots"])
            self.assertIn("trial_position", result["baseline_q_condition_plots"])
            self.assertGreaterEqual(result["n_trials_projected"], 1)

    def test_run_baseline_q_space_subject_analysis_from_pickle(self):
        trial_df = pd.DataFrame(
            {
                "subject_id": ["m1"] * 6 + ["m2"] * 6,
                "p_rl_left": np.linspace(0.1, 0.9, 12),
                "p_rl_right": np.linspace(0.9, 0.1, 12),
                "q_rl_left": np.linspace(0.0, 1.0, 12),
                "q_rl_right": np.linspace(1.0, 0.0, 12),
            }
        )

        def _write_placeholder_plot(*args, output_path: Path, **kwargs):
            del args
            del kwargs
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("plot")

        with tempfile.TemporaryDirectory(prefix="baseline_q_subjects_") as tmpdir, mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_baseline_q_space_subject_probability_figure",
            side_effect=_write_placeholder_plot,
        ):
            pickle_path = Path(tmpdir) / "trial_advantage.pkl"
            trial_df.to_pickle(pickle_path)
            result = likelihood_advantage_analysis.run_baseline_q_space_subject_analysis(
                pickle_path,
                probability_column="p_rl_right",
                subject_ids=["m2"],
                output_dir=Path(tmpdir) / "subject_q_plots",
            )

            self.assertEqual(result["probability_column"], "p_rl_right")
            self.assertEqual(result["subject_ids"], ["m2"])
            self.assertEqual(set(result["subject_probability_plots"].keys()), {"m2"})
            self.assertTrue(Path(result["summary"]).exists())
            self.assertTrue(Path(result["subject_probability_plots"]["m2"]).exists())

    def test_run_baseline_q_space_subject_analysis_validates_required_columns(self):
        base_df = pd.DataFrame(
            {
                "subject_id": ["m1", "m1"],
                "p_rl_left": [0.2, 0.8],
                "q_rl_left": [0.0, 1.0],
                "q_rl_right": [1.0, 0.0],
            }
        )

        with tempfile.TemporaryDirectory(prefix="baseline_q_subjects_missing_") as tmpdir:
            missing_probability_path = Path(tmpdir) / "missing_probability.pkl"
            base_df.drop(columns=["p_rl_left"]).to_pickle(missing_probability_path)
            with self.assertRaisesRegex(ValueError, "p_rl_left"):
                likelihood_advantage_analysis.run_baseline_q_space_subject_analysis(
                    missing_probability_path,
                    probability_column="p_rl_left",
                    output_dir=Path(tmpdir) / "missing_probability",
                )

            missing_state_path = Path(tmpdir) / "missing_q.pkl"
            base_df.drop(columns=["q_rl_left"]).to_pickle(missing_state_path)
            with self.assertRaisesRegex(ValueError, "q_rl_left"):
                likelihood_advantage_analysis.run_baseline_q_space_subject_analysis(
                    missing_state_path,
                    probability_column="p_rl_left",
                    output_dir=Path(tmpdir) / "missing_q",
                )

    def test_validate_model_pair_rejects_invalid_model_types(self):
        with self.assertRaisesRegex(ValueError, "model1_dir must resolve"):
            likelihood_advantage_analysis._validate_model_pair(
                self._make_resolved_run(
                    model_type="baseline_rl",
                    model_label="baseline",
                    model_index=0,
                ),
                self._make_resolved_run(
                    model_type="baseline_rl",
                    model_label="baseline_2",
                    model_index=1,
                ),
            )

        with self.assertRaisesRegex(ValueError, "model2_dir must resolve"):
            likelihood_advantage_analysis._validate_model_pair(
                self._make_resolved_run(
                    model_type="gru",
                    model_label="gru",
                    model_index=0,
                ),
                self._make_resolved_run(
                    model_type="gru",
                    model_label="gru_2",
                    model_index=1,
                ),
            )

    def test_run_likelihood_advantage_analysis_smoke_with_mocked_model_outputs(self):
        model1_run = self._make_resolved_run(
            model_type="gru",
            model_label="gru_model",
            model_index=0,
        )
        model2_run = self._make_resolved_run(
            model_type="baseline_rl",
            model_label="baseline_model",
            model_index=1,
        )
        raw_df = self._smoke_raw_df()
        output_df = self._smoke_output_df()
        canonical_df = likelihood_advantage_analysis._build_canonical_trial_dataframe(
            raw_df,
            metadata={},
        )
        baseline_prob_df = canonical_df.loc[:, ["subject_id", "ses_idx", "trial_idx", "action"]].copy()
        baseline_prob_df["p_rl"] = np.where(
            baseline_prob_df["action"].to_numpy(dtype=int) == 0,
            0.62,
            0.58,
        )
        baseline_prob_df["p_rl_left"] = 0.62
        baseline_prob_df["p_rl_right"] = 0.58
        baseline_prob_df["q_rl_left"] = np.linspace(0.0, 1.0, len(baseline_prob_df))
        baseline_prob_df["q_rl_right"] = np.linspace(1.0, 0.0, len(baseline_prob_df))

        def _write_placeholder_plot(*args, output_path: Path, **kwargs):
            del args
            del kwargs
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text("plot")

        subject_summary_df = pd.DataFrame.from_records(
            [
                {
                    "subject_id": "m1",
                    "mean_advantage": 0.1,
                    "curriculum_name": "A",
                    "x_jitter": -0.1,
                    "color": "#111111",
                },
                {
                    "subject_id": "m2",
                    "mean_advantage": 0.2,
                    "curriculum_name": "B",
                    "x_jitter": 0.0,
                    "color": "#222222",
                },
                {
                    "subject_id": "m3",
                    "mean_advantage": 0.3,
                    "curriculum_name": "A",
                    "x_jitter": 0.1,
                    "color": "#111111",
                },
            ]
        )

        with tempfile.TemporaryDirectory(prefix="likelihood_advantage_") as tmpdir, mock.patch.object(
            likelihood_advantage_analysis.lc,
            "_resolve_likelihood_run",
            side_effect=[model1_run, model2_run],
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_load_model1_evaluation_for_split",
            return_value={
                "raw_df": raw_df,
                "output_df": output_df,
                "metadata": {},
                "n_action_logits": 2,
            },
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_build_baseline_probability_frame",
            return_value=baseline_prob_df,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_build_subject_mean_advantage_df",
            return_value=subject_summary_df,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_advantage_histogram",
            side_effect=_write_placeholder_plot,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_subject_mean_advantage_scatter",
            side_effect=_write_placeholder_plot,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_variable_summary",
            side_effect=_write_placeholder_plot,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_summary_figure",
            side_effect=_write_placeholder_plot,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_session_stage_summary",
            side_effect=_write_placeholder_plot,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_rnn_state_pca_variance",
            side_effect=_write_placeholder_plot,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_rnn_state_condition_figure",
            side_effect=_write_placeholder_plot,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_baseline_q_space_condition_figure",
            side_effect=_write_placeholder_plot,
        ):
            result = likelihood_advantage_analysis.run_likelihood_advantage_analysis(
                model1_dir="/tmp/gru_model",
                model2_dir="/tmp/baseline_model",
                split="combined",
                output_dir=tmpdir,
                history_warmup=10,
                top_k_variables=3,
                jitter_seed=0,
                pca_seed=0,
            )

            expected_keys = {
                "output_dir",
                "summary",
                "trial_advantage_pickle",
                "advantage_histogram",
                "subject_mean_advantage_scatter",
                "bin_summary_long_csv",
                "bin_summary_wide_csv",
                "summary_figure",
                "variable_ranking_csv",
                "session_stage_top3_figure",
                "subject_bin_summary_csv",
                "rnn_state_pca_variance",
                "rnn_state_pca_variance_csv",
                "rnn_state_condition_plots",
                "baseline_q_space_summary",
                "baseline_q_condition_plots",
            }
            self.assertEqual(set(result.keys()), expected_keys)
            for key, path in result.items():
                if key == "output_dir":
                    self.assertTrue(Path(path).exists())
                elif key == "rnn_state_condition_plots":
                    self.assertEqual(
                        set(path.keys()),
                        {spec.name for spec in likelihood_advantage_analysis._VARIABLE_SPECS},
                    )
                elif key == "baseline_q_condition_plots":
                    self.assertEqual(
                        set(path.keys()),
                        {spec.name for spec in likelihood_advantage_analysis._VARIABLE_SPECS},
                    )
                else:
                    self.assertTrue(Path(path).exists(), msg=key)

            self.assertFalse(Path(tmpdir, "trial_advantage.csv").exists())
            trial_df = pd.read_pickle(result["trial_advantage_pickle"])
            self.assertIn("advantage", trial_df.columns)
            self.assertIn("session_stage", trial_df.columns)
            self.assertIn("p_model1_left", trial_df.columns)
            self.assertIn("p_model1_right", trial_df.columns)
            self.assertIn("p_rl_left", trial_df.columns)
            self.assertIn("p_rl_right", trial_df.columns)
            self.assertIn("q_rl_left", trial_df.columns)
            self.assertIn("q_rl_right", trial_df.columns)
            self.assertIn("rnn_state_0", trial_df.columns)
            self.assertIn("rnn_state_3", trial_df.columns)

            long_summary_df = pd.read_csv(result["bin_summary_long_csv"])
            self.assertIn("variable", long_summary_df.columns)
            self.assertIn("mean_advantage", long_summary_df.columns)


if __name__ == "__main__":
    unittest.main()
