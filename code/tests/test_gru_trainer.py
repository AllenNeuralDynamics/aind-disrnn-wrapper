"""Smoke tests for GruTrainer."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

try:
    from data_loaders.synthetic import SyntheticCognitiveAgents
    from model_trainers.gru_trainer import GruTrainer

    GRU_DEPS_AVAILABLE = True
    GRU_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    GRU_DEPS_AVAILABLE = False
    GRU_IMPORT_ERROR = exc


@unittest.skipUnless(
    GRU_DEPS_AVAILABLE,
    f"GRU trainer dependencies unavailable: {GRU_IMPORT_ERROR}",
)
class TestGruTrainer(unittest.TestCase):
    """Test suite for GruTrainer."""

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
        self.output_dir = Path(tempfile.mkdtemp(prefix="gru_trainer_test_"))

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def test_instantiation(self):
        trainer = GruTrainer(
            architecture={"hidden_size": 8, "num_layers": 1},
            training={
                "lr": 1e-3,
                "n_steps": 1,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
            },
            seed=42,
        )
        self.assertEqual(trainer.architecture["hidden_size"], 8)
        self.assertEqual(trainer.architecture["num_layers"], 1)
        self.assertEqual(trainer.seed, 42)

    def test_fit_basic(self):
        trainer = GruTrainer(
            architecture={"hidden_size": 8, "num_layers": 1},
            training={
                "lr": 1e-3,
                "n_steps": 5,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 0,
                "save_output_df": True,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        output = trainer.fit(self.bundle)

        self.assertIn("likelihood", output)
        self.assertIn("training_time", output)
        self.assertIn("split_examples", output)
        self.assertIn("random_key", output)
        self.assertIsInstance(output["likelihood"], float)
        self.assertGreaterEqual(output["likelihood"], 0.0)
        self.assertLessEqual(output["likelihood"], 1.0)

        self.assertTrue((self.output_dir / "params.json").exists())
        self.assertTrue((self.output_dir / "output_summary.json").exists())
        self.assertTrue((self.output_dir / "gru_config.json").exists())
        self.assertTrue((self.output_dir / "output_df.csv").exists())

    def test_checkpoint_training(self):
        trainer = GruTrainer(
            architecture={"hidden_size": 8, "num_layers": 1},
            training={
                "lr": 1e-3,
                "n_steps": 4,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 2,
                "checkpoint_plot_split_examples_every_n": 0,
                "checkpoint_save_output_df_every_n": 0,
                "checkpoint_log_eval_to_wandb": False,
                "checkpoint_log_split_examples_to_wandb": False,
                "checkpoint_run_heldout_eval": False,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        output = trainer.fit(self.bundle)

        self.assertIn("checkpoints", output)
        self.assertEqual(len(output["checkpoints"]), 2)
        self.assertTrue((self.output_dir / "checkpoints" / "index.json").exists())


if __name__ == "__main__":
    unittest.main()
