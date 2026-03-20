"""Smoke tests for DisrnnTrainer."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from data_loaders.synthetic import SyntheticCognitiveAgents
    from model_trainers.disrnn_trainer import DisrnnTrainer

    DISRNN_DEPS_AVAILABLE = True
    DISRNN_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    DISRNN_DEPS_AVAILABLE = False
    DISRNN_IMPORT_ERROR = exc


@unittest.skipUnless(
    DISRNN_DEPS_AVAILABLE,
    f"disRNN trainer dependencies unavailable: {DISRNN_IMPORT_ERROR}",
)
class TestDisrnnTrainer(unittest.TestCase):
    """Test suite for DisrnnTrainer."""

    def setUp(self):
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
        self.output_dir = Path(tempfile.mkdtemp(prefix="disrnn_trainer_test_"))

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_checkpoint_training(self):
        trainer = DisrnnTrainer(
            architecture={
                "latent_size": 4,
                "update_net_n_units_per_layer": 8,
                "update_net_n_layers": 2,
                "choice_net_n_units_per_layer": 4,
                "choice_net_n_layers": 1,
                "activation": "leaky_relu",
            },
            penalties={
                "latent_penalty": 1e-3,
                "choice_net_latent_penalty": 1e-3,
                "update_net_obs_penalty": 1e-3,
                "update_net_latent_penalty": 1e-3,
            },
            training={
                "lr": 1e-3,
                "n_steps": 4,
                "n_warmup_steps": 1,
                "loss": "penalized_categorical",
                "loss_param": 1.0,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 2,
                "checkpoint_plot_split_examples_every_n": 0,
                "checkpoint_save_output_df_every_n": 0,
                "checkpoint_log_eval_to_wandb": False,
                "checkpoint_log_train_to_wandb": False,
                "checkpoint_log_split_examples_to_wandb": False,
                "checkpoint_run_heldout_eval": False,
                "checkpoint_plot_choice_rule": False,
                "checkpoint_plot_update_rules": False,
                "plot_choice_rule": False,
                "plot_update_rules": False,
                "save_output_df": False,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        output = trainer.fit(self.bundle)

        self.assertIn("checkpoints", output)
        self.assertEqual(len(output["checkpoints"]), 2)
        for checkpoint in output["checkpoints"]:
            self.assertIn("train_likelihood", checkpoint)
            self.assertIsInstance(checkpoint["train_likelihood"], float)
            self.assertGreaterEqual(checkpoint["train_likelihood"], 0.0)
            self.assertLessEqual(checkpoint["train_likelihood"], 1.0)
        self.assertTrue((self.output_dir / "checkpoints" / "index.json").exists())


if __name__ == "__main__":
    unittest.main()
