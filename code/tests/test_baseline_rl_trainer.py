"""Unit tests for BaselineRLTrainer using aind-dynamic-foraging-models."""

import json
import os
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from aind_dynamic_foraging_models import generative_model

from base.types import DatasetBundle
from model_trainers.baseline_rl_trainer import BaselineRLTrainer
from data_loaders.synthetic import SyntheticCognitiveAgents
from utils.baseline_rl_evaluation import evaluate_baseline_rl_on_heldout_subjects


class TestBaselineRLTrainer(unittest.TestCase):
    """Test suite for BaselineRLTrainer."""

    def setUp(self):
        """Set up test fixtures."""
        self.loader = SyntheticCognitiveAgents(
            task={
                "type": "random_walk",
                "reward_baiting": False,
                "p_min": 0.0,
                "p_max": 1.0,
                "sigma": 0.15,
                "mean": 0,
                "num_trials": 100,
                "seed": 42,
            },
            agent={
                "agent_class": "ForagerQLearning",
                "agent_kwargs": {
                    "number_of_learning_rate": 2,
                    "number_of_forget_rate": 1,
                    "choice_kernel": "none",
                    "action_selection": "softmax",
                },
                "agent_params": {
                    "learn_rate_rew": 0.5,
                    "learn_rate_unrew": 0.1,
                    "forget_rate_unchosen": 0.05,
                    "softmax_inverse_temperature": 5.0,
                    "biasL": 0.0,
                },
                "agent_params_session_var": {
                    "biasL": {
                        "type": "gaussian",
                        "mean": 0,
                        "std": 1,
                    }
                },
                "seed": 42,
            },
            num_trials=100,
            num_sessions=6,
            eval_every_n=2,
        )
        self.bundle = self.loader.load()
        self.multisubject_bundle = self._make_multisubject_bundle()

    def _make_multisubject_bundle(self) -> DatasetBundle:
        raw_df = self.bundle.raw.copy()
        session_ids = list(dict.fromkeys(raw_df["ses_idx"].tolist()))
        subject_specs = [
            (101, "Curriculum A", session_ids[:3]),
            (202, "Curriculum B", session_ids[3:6]),
        ]

        session_frames = []
        train_session_ids = []
        eval_session_ids = []
        subject_id_to_index = {}
        index_to_subject_id = {}
        subject_curricula = {}

        for subject_index, (subject_id, curriculum_name, subject_session_ids) in enumerate(subject_specs):
            subject_id_to_index[subject_id] = subject_index
            index_to_subject_id[subject_index] = subject_id
            subject_curricula[subject_id] = curriculum_name

            for local_session_index, session_id in enumerate(subject_session_ids):
                session_df = raw_df[raw_df["ses_idx"] == session_id].copy()
                session_df["subject_id"] = subject_id
                session_df["curriculum_name"] = curriculum_name
                session_df["source_ses_idx"] = session_df["ses_idx"]
                unique_session_id = f"{subject_id}__{local_session_index}"
                session_df["ses_idx"] = unique_session_id
                session_frames.append(session_df)

                if local_session_index == 1:
                    eval_session_ids.append(unique_session_id)
                else:
                    train_session_ids.append(unique_session_id)

        multisubject_raw = (
            pd.concat(session_frames, ignore_index=True)
            .sort_values(["ses_idx", "trial"])
            .reset_index(drop=True)
        )
        metadata = dict(self.bundle.metadata)
        metadata.update(
            {
                "multisubject": True,
                "subject_ids": [spec[0] for spec in subject_specs],
                "subject_id_to_index": subject_id_to_index,
                "index_to_subject_id": index_to_subject_id,
                "num_subjects": len(subject_specs),
                "subject_curricula": subject_curricula,
                "train_session_ids": train_session_ids,
                "eval_session_ids": eval_session_ids,
                "num_trials": int(len(multisubject_raw)),
                "num_sessions": int(multisubject_raw["ses_idx"].nunique()),
            }
        )
        return DatasetBundle(
            raw=multisubject_raw,
            train_set=None,
            eval_set=None,
            metadata=metadata,
            extras={},
        )

    def test_instantiation(self):
        """Test that BaselineRLTrainer can be instantiated."""
        trainer = BaselineRLTrainer(
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 2,
                "number_of_forget_rate": 1,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            seed=42,
        )
        self.assertEqual(trainer.agent_class, "ForagerQLearning")
        self.assertEqual(trainer.agent_kwargs["number_of_learning_rate"], 2)
        self.assertEqual(trainer.agent_kwargs["number_of_forget_rate"], 1)
        self.assertEqual(trainer.seed, 42)

    def test_fit_basic(self):
        """Test that fit method runs and returns expected output structure."""
        trainer = BaselineRLTrainer(
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 2,
                "number_of_forget_rate": 1,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            DE_kwargs={"workers": 1, "maxiter": 3, "popsize": 5},
            output_dir="/tmp/baseline_rl_test",
            seed=42,
        )

        output = trainer.fit(self.bundle)

        # Check required output keys
        self.assertIn("likelihood", output)
        self.assertIn("likelihood_train", output)
        self.assertIn("fitted_params", output)
        self.assertIn("agent_class", output)
        self.assertIn("agent_kwargs", output)
        self.assertIn("n_free_params", output)
        self.assertIn("num_train_sessions", output)
        self.assertIn("num_eval_sessions", output)
        self.assertIn("num_train_trials", output)
        self.assertIn("num_eval_trials", output)
        self.assertIn("elapsed_seconds", output)

        # Check types
        self.assertIsInstance(output["likelihood"], float)
        self.assertIsInstance(output["likelihood_train"], float)
        self.assertIsInstance(output["fitted_params"], dict)
        self.assertIsInstance(output["n_free_params"], int)
        self.assertIsInstance(output["elapsed_seconds"], float)
        self.assertEqual(output["mean_subject_train_likelihood"], output["likelihood_train"])
        self.assertEqual(output["mean_subject_eval_likelihood"], output["likelihood"])
        self.assertEqual(output["pooled_train_trial_likelihood"], output["likelihood_train"])
        self.assertEqual(output["pooled_eval_trial_likelihood"], output["likelihood"])

    def test_fitted_parameters(self):
        """Test that fitted parameters are reasonable."""
        trainer = BaselineRLTrainer(
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 2,
                "number_of_forget_rate": 1,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            DE_kwargs={"workers": 1, "maxiter": 5, "popsize": 10},
            output_dir="/tmp/baseline_rl_test",
            seed=42,
        )

        output = trainer.fit(self.bundle)
        fitted_params = output["fitted_params"]

        # Check that all expected parameters are fitted
        expected_params = [
            "learn_rate_rew",
            "learn_rate_unrew",
            "forget_rate_unchosen",
            "softmax_inverse_temperature",
            "biasL",
        ]
        for param in expected_params:
            self.assertIn(param, fitted_params, f"Missing fitted parameter: {param}")

        # Check parameter ranges (should be positive and reasonable)
        self.assertGreater(fitted_params["learn_rate_rew"], 0.0)
        self.assertLess(fitted_params["learn_rate_rew"], 1.0)
        self.assertGreater(fitted_params["learn_rate_unrew"], 0.0)
        self.assertLess(fitted_params["learn_rate_unrew"], 1.0)
        self.assertGreater(fitted_params["forget_rate_unchosen"], 0.0)
        self.assertLess(fitted_params["forget_rate_unchosen"], 1.0)
        self.assertGreater(fitted_params["softmax_inverse_temperature"], 0.0)

    def test_likelihood_values(self):
        """Test that likelihood values are valid (between 0 and 1)."""
        trainer = BaselineRLTrainer(
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 2,
                "number_of_forget_rate": 1,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            DE_kwargs={"workers": 1, "maxiter": 3, "popsize": 5},
            output_dir="/tmp/baseline_rl_test",
            seed=42,
        )

        output = trainer.fit(self.bundle)

        # Likelihood should be between 0 and 1 (random guessing would be ~0.5)
        self.assertGreater(output["likelihood"], 0.0)
        self.assertLess(output["likelihood"], 1.0)
        self.assertGreater(output["likelihood_train"], 0.0)
        self.assertLess(output["likelihood_train"], 1.0)

    def test_groundtruth_comparison(self):
        """Test that groundtruth likelihood is included in output when available."""
        trainer = BaselineRLTrainer(
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 2,
                "number_of_forget_rate": 1,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            DE_kwargs={"workers": 1, "maxiter": 3, "popsize": 5},
            output_dir="/tmp/baseline_rl_test",
            seed=42,
        )

        output = trainer.fit(self.bundle)

        # Groundtruth likelihood should be present for synthetic data
        self.assertIn("groundtruth_likelihood", output)
        self.assertIsInstance(output["groundtruth_likelihood"], float)
        self.assertGreater(output["groundtruth_likelihood"], 0.0)
        self.assertLess(output["groundtruth_likelihood"], 1.0)

        # Relative to groundtruth should be present
        self.assertIn("likelihood_relative_to_groundtruth", output)
        self.assertIsInstance(output["likelihood_relative_to_groundtruth"], float)

    def test_session_data_extraction(self):
        """Test that session data is correctly extracted from bundle."""
        trainer = BaselineRLTrainer(
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 2,
                "number_of_forget_rate": 1,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            DE_kwargs={"workers": 1, "maxiter": 3, "popsize": 5},
            output_dir="/tmp/baseline_rl_test",
            seed=42,
        )

        # Extract session data
        (
            train_choices,
            train_rewards,
            eval_choices,
            eval_rewards,
        ) = trainer._extract_session_data(self.bundle)

        # Check that we have correct number of sessions
        n_train_sessions = len(train_choices)
        n_eval_sessions = len(eval_choices)
        n_total_sessions = n_train_sessions + n_eval_sessions

        # With eval_every_n=2 and eval_offset=1, sessions 1,3,5 are eval
        # So train: 0,2,4 (3 sessions) and eval: 1,3,5 (3 sessions)
        self.assertEqual(n_train_sessions, 3)
        self.assertEqual(n_eval_sessions, 3)
        self.assertEqual(n_total_sessions, 6)

        # Check that each session has correct number of trials
        for i in range(n_train_sessions):
            self.assertEqual(len(train_choices[i]), 100)
            self.assertEqual(len(train_rewards[i]), 100)

        for i in range(n_eval_sessions):
            self.assertEqual(len(eval_choices[i]), 100)
            self.assertEqual(len(eval_rewards[i]), 100)

    def test_normalization_computation(self):
        """Test that normalized likelihood computation is correct."""
        trainer = BaselineRLTrainer(
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 2,
                "number_of_forget_rate": 1,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            seed=42,
        )

        # Create simple test case
        n_trials = 10
        choices = np.array([0] * 5 + [1] * 5)
        # Perfect predictions: always predict correct choice with probability 1.0
        choice_prob_perfect = np.array(
            [[1.0] * 5 + [0.0] * 5, [0.0] * 5 + [1.0] * 5]
        )
        # Random predictions: always 0.5 for both choices
        choice_prob_random = np.ones((2, n_trials)) * 0.5

        # Perfect predictions should have likelihood close to 1.0
        likelihood_perfect = trainer._compute_normalized_likelihood(
            [choices], [choice_prob_perfect]
        )
        self.assertGreater(likelihood_perfect, 0.99)

        # Random predictions should have likelihood close to 0.5
        likelihood_random = trainer._compute_normalized_likelihood(
            [choices], [choice_prob_random]
        )
        self.assertGreater(likelihood_random, 0.4)
        self.assertLess(likelihood_random, 0.6)

    def test_multiple_agent_types(self):
        """Test that trainer works with different agent types."""
        # Test ForagerQLearning
        trainer_ql = BaselineRLTrainer(
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 1,
                "number_of_forget_rate": 0,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            seed=42,
        )
        self.assertEqual(trainer_ql.agent_class, "ForagerQLearning")

        # Test ForagerLossCounting
        trainer_lc = BaselineRLTrainer(
            agent_class="ForagerLossCounting",
            agent_kwargs={
                "win_stay_lose_switch": True,
                "choice_kernel": "none",
            },
            seed=42,
        )
        self.assertEqual(trainer_lc.agent_class, "ForagerLossCounting")

        # Test ForagerCompareThreshold
        trainer_ct = BaselineRLTrainer(
            agent_class="ForagerCompareThreshold",
            agent_kwargs={"choice_kernel": "none"},
            seed=42,
        )
        self.assertEqual(trainer_ct.agent_class, "ForagerCompareThreshold")

    def test_fit_bounds_override(self):
        """Test that fit_bounds_override is passed correctly."""
        trainer = BaselineRLTrainer(
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 2,
                "number_of_forget_rate": 1,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            fit_bounds_override={"learn_rate_rew": [0.3, 0.7]},
            DE_kwargs={"workers": 1, "maxiter": 3, "popsize": 5},
            output_dir="/tmp/baseline_rl_test",
            seed=42,
        )

        output = trainer.fit(self.bundle)
        fitted_params = output["fitted_params"]

        # learn_rate_rew should be within overridden bounds
        self.assertGreaterEqual(fitted_params["learn_rate_rew"], 0.3)
        self.assertLessEqual(fitted_params["learn_rate_rew"], 0.7)

    def test_clamp_params(self):
        """Test that clamp_params fixes parameters during fitting."""
        trainer = BaselineRLTrainer(
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 2,
                "number_of_forget_rate": 1,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            clamp_params={"biasL": 0.5},
            DE_kwargs={"workers": 1, "maxiter": 3, "popsize": 5},
            output_dir="/tmp/baseline_rl_test",
            seed=42,
        )

        output = trainer.fit(self.bundle)
        fitted_params = output["fitted_params"]

        # biasL should be clamped to 0.5
        self.assertAlmostEqual(fitted_params["biasL"], 0.5, places=5)

    def test_parameter_recovery(self):
        """Test that fitted parameters recover true parameters reasonably."""
        trainer = BaselineRLTrainer(
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 2,
                "number_of_forget_rate": 1,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            DE_kwargs={"workers": 1, "maxiter": 10, "popsize": 10},
            output_dir="/tmp/baseline_rl_test",
            seed=42,
        )

        output = trainer.fit(self.bundle)
        fitted_params = output["fitted_params"]

        # Get true parameters from first session
        # Note: biasL varies by session due to agent_params_session_var,
        # so we use the mean true value (0.0) for comparison
        true_params = {
            "learn_rate_rew": 0.5,
            "learn_rate_unrew": 0.1,
            "forget_rate_unchosen": 0.05,
            "softmax_inverse_temperature": 5.0,
            "biasL": 0.0,  # Mean true value (varies per session)
        }

        # Check that fitted parameters are reasonably close to true values
        # With limited iterations, allow some error
        max_allowed_error = 0.5  # 50% max deviation allowed

        for param in ["learn_rate_rew", "learn_rate_unrew", "forget_rate_unchosen",
                      "softmax_inverse_temperature"]:
            fitted_val = float(fitted_params[param])
            true_val = true_params[param]
            error = abs(fitted_val - true_val)
            relative_error = error / true_val if true_val != 0 else error

            # Check that error is reasonable (less than max_allowed_error)
            # Note: with limited DE iterations, we don't expect perfect recovery
            self.assertLess(relative_error, max_allowed_error,
                          f"Parameter {param}: fitted={fitted_val:.4f}, "
                          f"true={true_val:.4f}, error={relative_error:.4f}")

        # biasL recovery is harder due to session-to-session variation
        # Just check it's recovered to a reasonable value (not too far from 0.0)
        self.assertLess(abs(float(fitted_params["biasL"])), 2.0,
                      "biasL should be recovered within +/-2.0")

    def test_parameter_recovery_plot(self):
        """Test that parameter recovery plot is generated correctly."""
        output_dir = "/tmp/baseline_rl_test_plot"
        trainer = BaselineRLTrainer(
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 2,
                "number_of_forget_rate": 1,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            DE_kwargs={"workers": 1, "maxiter": 3, "popsize": 5},
            output_dir=output_dir,
            seed=42,
        )

        output = trainer.fit(self.bundle)

        # Check that plot path is in output
        self.assertIn("parameter_recovery_plot_path", output)

        # Check that plot file exists
        plot_path = output["parameter_recovery_plot_path"]
        self.assertTrue(os.path.exists(plot_path), f"Plot file not found: {plot_path}")

        # Check that plot file has reasonable size (not empty)
        file_size = os.path.getsize(plot_path)
        self.assertGreater(file_size, 1000, "Plot file seems too small (possibly empty)")

        print(f"\n=== Parameter Recovery Plot Test ===")
        print(f"Plot saved to: {plot_path}")
        print(f"File size: {file_size} bytes")

        # Print fitted vs true parameters for visual verification
        fitted_params = output["fitted_params"]
        true_params = {
            "learn_rate_rew": 0.5,
            "learn_rate_unrew": 0.1,
            "forget_rate_unchosen": 0.05,
            "softmax_inverse_temperature": 5.0,
            "biasL": 0.0,
        }

        print("\nParameter Comparison:")
        print(f"{'Parameter':<30} {'True':>10} {'Fitted':>10} {'Error':>10}")
        print("-" * 62)
        for param in fitted_params:
            true_val = true_params.get(param, float("nan"))
            fitted_val = float(fitted_params[param])
            error = fitted_val - true_val
            print(f"{param:<30} {true_val:>10.4f} {fitted_val:>10.4f} {error:>+10.4f}")

        print(f"\nLikelihood: {output['likelihood']:.4f}")
        print(f"Groundtruth: {output['groundtruth_likelihood']:.4f}")
        print(f"Ratio: {output['likelihood_relative_to_groundtruth']:.4f}")

    def test_multisubject_fit_exports_subject_artifacts_and_metrics(self):
        """Test that multisubject baseline RL saves per-subject artifacts and metrics."""
        with tempfile.TemporaryDirectory(prefix="baseline_rl_multisubject_") as tmpdir:
            trainer = BaselineRLTrainer(
                architecture={"multisubject": True},
                multisubject_subject_workers=2,
                agent_class="ForagerQLearning",
                agent_kwargs={
                    "number_of_learning_rate": 2,
                    "number_of_forget_rate": 1,
                    "choice_kernel": "none",
                    "action_selection": "softmax",
                },
                DE_kwargs={
                    "workers": 4,
                    "maxiter": 2,
                    "popsize": 5,
                    "mutation": (0.5, 0.9),
                    "recombination": 0.6,
                    "polish": False,
                },
                output_dir=tmpdir,
                seed=42,
            )

            output = trainer.fit(self.multisubject_bundle)

            self.assertTrue(output["multisubject"])
            self.assertEqual(output["fit_strategy"], "per_subject")
            self.assertIn("mean_subject_train_likelihood", output)
            self.assertIn("mean_subject_eval_likelihood", output)
            self.assertIn("pooled_train_trial_likelihood", output)
            self.assertIn("pooled_eval_trial_likelihood", output)
            self.assertEqual(output["likelihood_train"], output["pooled_train_trial_likelihood"])
            self.assertEqual(output["likelihood"], output["pooled_eval_trial_likelihood"])
            self.assertEqual(output["multisubject_subject_workers"], 2)
            self.assertEqual(output["effective_de_workers_per_subject"], 1)

            subject_artifacts = output["subject_artifacts"]
            subject_index_map_path = Path(subject_artifacts["subject_index_map"])
            subject_metrics_csv_path = Path(subject_artifacts["subject_fit_metrics_csv"])
            subject_metrics_pickle_path = Path(subject_artifacts["subject_fit_metrics_pickle"])
            parameter_space_path = Path(output["subject_parameter_state_space_path"])
            likelihood_scatter_path = Path(output["subject_likelihood_scatter_path"])

            self.assertTrue(subject_index_map_path.exists())
            self.assertTrue(subject_metrics_csv_path.exists())
            self.assertTrue(subject_metrics_pickle_path.exists())
            self.assertTrue(parameter_space_path.exists())
            self.assertTrue(likelihood_scatter_path.exists())

            subject_metrics_df = pd.read_csv(subject_metrics_csv_path)
            self.assertEqual(len(subject_metrics_df), 2)
            self.assertEqual(subject_metrics_df["subject_index"].tolist(), [0, 1])
            self.assertCountEqual(
                subject_metrics_df["curriculum_name"].tolist(),
                ["Curriculum A", "Curriculum B"],
            )
            self.assertAlmostEqual(
                output["mean_subject_train_likelihood"],
                float(subject_metrics_df["train_likelihood"].mean()),
                places=6,
            )
            self.assertAlmostEqual(
                output["mean_subject_eval_likelihood"],
                float(subject_metrics_df["eval_likelihood"].mean()),
                places=6,
            )

            pooled_train_choices = []
            pooled_train_probs = []
            pooled_eval_choices = []
            pooled_eval_probs = []
            agent_class_obj = generative_model.ForagerQLearning

            for subject_id in subject_metrics_df["subject_id"].tolist():
                subject_summary_path = (
                    Path(tmpdir)
                    / "subjects"
                    / str(subject_id)
                    / "fit_summary.json"
                )
                self.assertTrue(subject_summary_path.exists())
                with subject_summary_path.open("r") as f:
                    subject_summary = json.load(f)

                subject_df = self.multisubject_bundle.raw[
                    self.multisubject_bundle.raw["subject_id"] == subject_summary["subject_id"]
                ].copy()
                train_choices, train_rewards, _ = trainer._extract_sessions_from_raw_df(
                    subject_df,
                    session_ids=subject_summary["train_session_ids"],
                )
                eval_choices, eval_rewards, _ = trainer._extract_sessions_from_raw_df(
                    subject_df,
                    session_ids=subject_summary["eval_session_ids"],
                )

                agent = agent_class_obj(**trainer.agent_kwargs, seed=trainer.seed)
                agent.set_params(**subject_summary["fitted_params"])
                pooled_train_choices.extend(train_choices)
                pooled_train_probs.extend(
                    [
                        np.asarray(arr)
                        for arr in agent.perform_closed_loop_multi_session(
                            train_choices,
                            train_rewards,
                        )
                    ]
                )

                eval_agent = agent_class_obj(**trainer.agent_kwargs, seed=trainer.seed)
                eval_agent.set_params(**subject_summary["fitted_params"])
                pooled_eval_choices.extend(eval_choices)
                pooled_eval_probs.extend(
                    [
                        np.asarray(arr)
                        for arr in eval_agent.perform_closed_loop_multi_session(
                            eval_choices,
                            eval_rewards,
                        )
                    ]
                )

            self.assertAlmostEqual(
                output["pooled_train_trial_likelihood"],
                trainer._compute_normalized_likelihood(pooled_train_choices, pooled_train_probs),
                places=6,
            )
            self.assertAlmostEqual(
                output["pooled_eval_trial_likelihood"],
                trainer._compute_normalized_likelihood(pooled_eval_choices, pooled_eval_probs),
                places=6,
            )

    def test_multisubject_effective_de_kwargs_force_one_worker_and_preserve_other_options(self):
        """Test that multisubject mode forces one DE worker per subject."""
        trainer = BaselineRLTrainer(
            architecture={"multisubject": True},
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 2,
                "number_of_forget_rate": 1,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            DE_kwargs={
                "workers": 8,
                "popsize": 11,
                "mutation": (0.4, 0.95),
                "recombination": 0.55,
                "polish": False,
            },
            seed=42,
        )

        effective_kwargs = trainer._effective_multisubject_de_kwargs()
        self.assertEqual(effective_kwargs["workers"], 1)
        self.assertEqual(effective_kwargs["popsize"], 11)
        self.assertEqual(tuple(effective_kwargs["mutation"]), (0.4, 0.95))
        self.assertEqual(effective_kwargs["recombination"], 0.55)
        self.assertFalse(effective_kwargs["polish"])

    def test_multisubject_parameter_space_plot_skips_when_fewer_than_two_params_vary(self):
        """Test that the subject parameter-space plot is skipped cleanly when needed."""
        trainer = BaselineRLTrainer(
            architecture={"multisubject": True},
            agent_class="ForagerQLearning",
            agent_kwargs={
                "number_of_learning_rate": 2,
                "number_of_forget_rate": 1,
                "choice_kernel": "none",
                "action_selection": "softmax",
            },
            seed=42,
        )

        subject_metrics_df = pd.DataFrame(
            [
                {
                    "subject_index": 0,
                    "subject_id": 101,
                    "curriculum_name": "Curriculum A",
                    "learn_rate_rew": 0.2,
                    "biasL": 0.0,
                },
                {
                    "subject_index": 1,
                    "subject_id": 202,
                    "curriculum_name": "Curriculum B",
                    "learn_rate_rew": 0.4,
                    "biasL": 0.0,
                },
            ]
        )
        fig = trainer._plot_subject_parameter_state_space(
            subject_metrics_df,
            parameter_columns=["learn_rate_rew", "biasL"],
        )
        self.assertIsNone(fig)

    def test_heldout_eval_skips_multisubject_per_subject_outputs(self):
        """Test that held-out evaluation skips multisubject per-subject runs."""
        with tempfile.TemporaryDirectory(prefix="baseline_rl_heldout_skip_") as tmpdir:
            output_path = Path(tmpdir) / "baseline_rl_output.json"
            with output_path.open("w") as f:
                json.dump(
                    {
                        "multisubject": True,
                        "fit_strategy": "per_subject",
                    },
                    f,
                )

            hydra_config = types.SimpleNamespace(
                data=types.SimpleNamespace(),
                model=types.SimpleNamespace(
                    type="baseline_rl",
                    output_dir=tmpdir,
                ),
            )
            summary = evaluate_baseline_rl_on_heldout_subjects(hydra_config)
            self.assertIsNotNone(summary)
            self.assertTrue(summary["skipped"])
            self.assertIn("multisubject baseline RL", summary["reason"])


if __name__ == "__main__":
    unittest.main()
