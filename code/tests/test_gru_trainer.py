"""Smoke tests for GruTrainer."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import aind_disrnn_utils.data_loader as dl
    import numpy as np
    import pandas as pd
    from disentangled_rnns.library import rnn_utils
    from omegaconf import OmegaConf

    from base.types import DatasetBundle
    from data_loaders.synthetic import SyntheticCognitiveAgents
    from model_trainers.gru_trainer import (
        GruTrainer,
        _validate_multisubject_dataset_inputs,
    )
    from utils.gru_evaluation import evaluate_gru_on_heldout_subjects
    from utils.multisubject import (
        build_subject_index_maps,
        compute_train_eval_session_ids,
        merge_datasets_with_subject_index,
    )

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
        self.multisubject_bundle = self._make_multisubject_bundle()
        self.output_dir = Path(tempfile.mkdtemp(prefix="gru_trainer_test_"))

    def tearDown(self):
        shutil.rmtree(self.output_dir, ignore_errors=True)

    def _make_multisubject_bundle(self) -> DatasetBundle:
        raw_df = self.bundle.raw.copy()
        raw_df["subject_id"] = np.where(raw_df["ses_idx"].astype(int) < 3, 711041, 793446)
        raw_df["source_ses_idx"] = raw_df["ses_idx"]
        raw_df["ses_idx"] = raw_df.apply(
            lambda row: f"{int(row['subject_id'])}__{int(row['source_ses_idx'])}",
            axis=1,
        )

        ordered_subject_ids, subject_id_to_index, index_to_subject_id = build_subject_index_maps(
            [711041, 793446]
        )

        full_datasets = []
        train_datasets = []
        eval_datasets = []
        ordered_frames = []
        subject_indices = []
        full_session_ids = []
        train_session_ids = []
        eval_session_ids = []

        for subject_id in ordered_subject_ids:
            subject_df = raw_df[raw_df["subject_id"] == subject_id].copy()
            subject_df = subject_df.sort_values(["ses_idx", "trial"]).reset_index(drop=True)
            session_ids = list(dict.fromkeys(subject_df["ses_idx"].tolist()))
            train_ids, eval_ids = compute_train_eval_session_ids(session_ids, eval_every_n=2)
            subject_df["subject_index"] = subject_id_to_index[subject_id]

            dataset = dl.create_disrnn_dataset(
                subject_df,
                ignore_policy="exclude",
                batch_size=None,
                batch_mode="random",
            )
            dataset_train, dataset_eval = rnn_utils.split_dataset(dataset, eval_every_n=2)

            full_datasets.append(dataset)
            train_datasets.append(dataset_train)
            eval_datasets.append(dataset_eval)
            ordered_frames.append(subject_df)
            subject_indices.append(subject_id_to_index[subject_id])
            full_session_ids.extend(session_ids)
            train_session_ids.extend(train_ids)
            eval_session_ids.extend(eval_ids)

        merged_dataset = merge_datasets_with_subject_index(full_datasets, subject_indices)
        merged_train = merge_datasets_with_subject_index(train_datasets, subject_indices)
        merged_eval = merge_datasets_with_subject_index(eval_datasets, subject_indices)
        merged_raw_df = pd.concat(ordered_frames, ignore_index=True)
        merged_raw_df["ses_idx"] = pd.Categorical(
            merged_raw_df["ses_idx"],
            categories=full_session_ids,
            ordered=True,
        )
        merged_raw_df = merged_raw_df.sort_values(["ses_idx", "trial"]).reset_index(drop=True)
        merged_raw_df["ses_idx"] = merged_raw_df["ses_idx"].astype(str)

        return DatasetBundle(
            raw=merged_raw_df,
            train_set=merged_train,
            eval_set=merged_eval,
            metadata={
                "ignore_policy": "exclude",
                "eval_every_n": 2,
                "multisubject": True,
                "subject_ids": ordered_subject_ids,
                "subject_id_to_index": subject_id_to_index,
                "index_to_subject_id": index_to_subject_id,
                "num_subjects": len(ordered_subject_ids),
                "num_trials": len(merged_raw_df),
                "num_sessions": len(full_session_ids),
                "train_session_ids": train_session_ids,
                "eval_session_ids": eval_session_ids,
            },
            extras={"dataset": merged_dataset},
        )

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

        self.assertIn("initial_evaluations", output)
        self.assertIn("before_training", output["initial_evaluations"])
        self.assertIn("likelihood", output)
        self.assertIn("likelihood_train", output)
        self.assertIn("training_time", output)
        self.assertIn("split_examples", output)
        self.assertIn("random_key", output)
        before_training = output["initial_evaluations"]["before_training"]
        self.assertTrue(Path(before_training["params_path"]).exists())
        self.assertNotIn("output_df_path", before_training)
        self.assertIn("split_examples", before_training)
        self.assertIsInstance(output["likelihood"], float)
        self.assertIsInstance(output["likelihood_train"], float)
        self.assertGreaterEqual(output["likelihood"], 0.0)
        self.assertLessEqual(output["likelihood"], 1.0)
        self.assertGreaterEqual(output["likelihood_train"], 0.0)
        self.assertLessEqual(output["likelihood_train"], 1.0)

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
        self.assertIn("initial_evaluations", output)
        self.assertIn("before_training", output["initial_evaluations"])
        self.assertEqual(len(output["checkpoints"]), 2)
        for checkpoint in output["checkpoints"]:
            self.assertIn("train_likelihood", checkpoint)
            self.assertIsInstance(checkpoint["train_likelihood"], float)
            self.assertGreaterEqual(checkpoint["train_likelihood"], 0.0)
            self.assertLessEqual(checkpoint["train_likelihood"], 1.0)
        self.assertTrue((self.output_dir / "checkpoints" / "index.json").exists())

    def test_multisubject_training_exports_subject_artifacts(self):
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
            },
            training={
                "lr": 1e-3,
                "n_steps": 2,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 0,
                "checkpoint_run_heldout_eval": False,
                "save_output_df": False,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        output = trainer.fit(self.multisubject_bundle)

        self.assertTrue(output["multisubject"])
        self.assertIn("subject_artifacts", output)
        self.assertTrue((self.output_dir / "subject_index_map.json").exists())
        self.assertTrue((self.output_dir / "subject_embeddings.pkl").exists())
        self.assertTrue((self.output_dir / "subject_embedding_state_space.png").exists())
        before_training = output["initial_evaluations"]["before_training"]
        self.assertIn("plot_paths", before_training)
        self.assertTrue(before_training["plot_paths"]["subject_embedding_state_space"])
        self.assertTrue(Path(before_training["plot_paths"]["subject_embedding_state_space"]).exists())

        subject_embeddings_df = pd.read_pickle(self.output_dir / "subject_embeddings.pkl")
        self.assertEqual(subject_embeddings_df["subject_id"].tolist(), [711041, 793446])

        params = json.loads((self.output_dir / "params.json").read_text())
        self.assertIn("multisubject_gru", params)
        self.assertIn("subject_embeddings", params["multisubject_gru"])

    def test_multisubject_checkpoint_training_plots_subject_embeddings(self):
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
            },
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
                "checkpoint_keep_media_files": True,
                "checkpoint_run_heldout_eval": False,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        output = trainer.fit(self.multisubject_bundle)

        self.assertEqual(len(output["checkpoints"]), 2)
        for checkpoint in output["checkpoints"]:
            plot_paths = checkpoint.get("plot_paths", {})
            self.assertTrue(plot_paths["subject_embedding_state_space"])
            self.assertTrue(Path(plot_paths["subject_embedding_state_space"]).exists())

    def test_multisubject_requires_subject_embedding_size(self):
        trainer = GruTrainer(
            architecture={"multisubject": True, "hidden_size": 8, "num_layers": 1},
            training={
                "lr": 1e-3,
                "n_steps": 1,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        with self.assertRaisesRegex(
            ValueError,
            "architecture.subject_embedding_size > 0",
        ):
            trainer.fit(self.multisubject_bundle)

    def test_multisubject_accepts_all_subject_embedding_init_modes(self):
        for init_mode in ("zeros", "small_random", "subject_count_scaled_random"):
            trainer = GruTrainer(
                architecture={
                    "multisubject": True,
                    "hidden_size": 8,
                    "num_layers": 1,
                    "subject_embedding_size": 3,
                    "subject_embedding_init": init_mode,
                },
                training={
                    "lr": 1e-3,
                    "n_steps": 0,
                    "loss": "categorical",
                    "loss_param": 1,
                    "max_grad_norm": 1.0,
                    "checkpoint_run_heldout_eval": False,
                    "save_output_df": False,
                },
                output_dir=str(self.output_dir / init_mode),
                seed=42,
            )

            output = trainer.fit(self.multisubject_bundle)
            self.assertTrue(output["multisubject"])

    def test_multisubject_rejects_unknown_subject_embedding_init_mode(self):
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
                "subject_embedding_init": "not_a_real_mode",
            },
            training={
                "lr": 1e-3,
                "n_steps": 0,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported subject_embedding_init",
        ):
            trainer.fit(self.multisubject_bundle)

    def test_multisubject_requires_subject_count_metadata(self):
        missing_metadata_bundle = DatasetBundle(
            raw=self.multisubject_bundle.raw.copy(),
            train_set=self.multisubject_bundle.train_set,
            eval_set=self.multisubject_bundle.eval_set,
            metadata={
                "ignore_policy": "exclude",
                "eval_every_n": 2,
                "multisubject": True,
            },
            extras=self.multisubject_bundle.extras,
        )
        trainer = GruTrainer(
            architecture={
                "multisubject": True,
                "hidden_size": 8,
                "num_layers": 1,
                "subject_embedding_size": 3,
            },
            training={
                "lr": 1e-3,
                "n_steps": 1,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        with self.assertRaisesRegex(
            ValueError,
            "metadata.num_subjects or metadata.subject_ids",
        ):
            trainer.fit(missing_metadata_bundle)

    def test_validate_multisubject_dataset_inputs_allows_padding(self):
        class _DummyDataset:
            def __init__(self, xs):
                self._xs = xs

            def get_all(self):
                return self._xs, np.zeros((self._xs.shape[0], self._xs.shape[1], 1))

        xs = np.array(
            [
                [[0.0, 1.0, 0.0]],
                [[-1.0, -1.0, -1.0]],
            ]
        )

        _validate_multisubject_dataset_inputs(
            _DummyDataset(xs),
            max_n_subjects=1,
            context="test",
        )

    def test_validate_multisubject_dataset_inputs_rejects_out_of_range_ids(self):
        class _DummyDataset:
            def __init__(self, xs):
                self._xs = xs

            def get_all(self):
                return self._xs, np.zeros((self._xs.shape[0], self._xs.shape[1], 1))

        xs = np.array([[[2.0, 1.0, 0.0]]])

        with self.assertRaisesRegex(ValueError, "out-of-range subject ids"):
            _validate_multisubject_dataset_inputs(
                _DummyDataset(xs),
                max_n_subjects=2,
                context="test",
            )

    def test_heldout_eval_rejects_multisubject_gru(self):
        hydra_config = OmegaConf.create(
            {
                "data": {
                    "test_subject_ids": [711041],
                    "ignore_policy": "exclude",
                    "heldout_example_sessions_per_subject": 0,
                    "example_max_subjects": 1,
                },
                "model": {
                    "architecture": {
                        "multisubject": True,
                        "hidden_size": 8,
                        "output_size": 2,
                    },
                    "output_dir": str(self.output_dir),
                },
            }
        )

        with self.assertRaisesRegex(
            ValueError,
            "not supported for multisubject GRU",
        ):
            evaluate_gru_on_heldout_subjects(hydra_config)


if __name__ == "__main__":
    unittest.main()
