"""Unit tests for BaselineRLTrainer using aind-dynamic-foraging-models."""

import os
import numpy as np
import unittest

from model_trainers.baseline_rl_trainer import BaselineRLTrainer
from data_loaders.synthetic import SyntheticCognitiveAgents


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


if __name__ == "__main__":
    unittest.main()
