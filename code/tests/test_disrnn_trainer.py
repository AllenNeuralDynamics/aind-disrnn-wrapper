"""Smoke tests for DisrnnTrainer."""

from __future__ import annotations

import shutil
import tempfile
import unittest
import json
import types
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import aind_disrnn_utils.data_loader as dl
    from disentangled_rnns.library import rnn_utils

    from base.types import DatasetBundle
    from data_loaders.synthetic import SyntheticCognitiveAgents
    from model_trainers.disrnn_trainer import DisrnnTrainer
    from utils.disrnn_evaluation import _aligned_action_probabilities_from_output_df
    from utils.multisubject import (
        build_subject_index_maps,
        compute_train_eval_session_ids,
        merge_datasets_with_subject_index,
    )

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
        self.assertIn("initial_evaluations", output)
        self.assertIn("before_warmup", output["initial_evaluations"])
        self.assertIn("after_warmup", output["initial_evaluations"])
        self.assertIn("likelihood", output)
        self.assertIn("likelihood_train", output)
        before_warmup = output["initial_evaluations"]["before_warmup"]
        after_warmup = output["initial_evaluations"]["after_warmup"]
        self.assertTrue(Path(before_warmup["params_path"]).exists())
        self.assertTrue(Path(before_warmup["output_df_path"]).exists())
        self.assertIn("split_examples", before_warmup)
        self.assertTrue(Path(after_warmup["params_path"]).exists())
        self.assertTrue(Path(after_warmup["output_df_path"]).exists())
        self.assertIn("split_examples", after_warmup)
        self.assertEqual(len(output["checkpoints"]), 2)
        for checkpoint in output["checkpoints"]:
            self.assertIn("train_likelihood", checkpoint)
            self.assertIsInstance(checkpoint["train_likelihood"], float)
            self.assertGreaterEqual(checkpoint["train_likelihood"], 0.0)
            self.assertLessEqual(checkpoint["train_likelihood"], 1.0)
        self.assertIsInstance(output["likelihood"], float)
        self.assertIsInstance(output["likelihood_train"], float)
        self.assertGreaterEqual(output["likelihood"], 0.0)
        self.assertLessEqual(output["likelihood"], 1.0)
        self.assertGreaterEqual(output["likelihood_train"], 0.0)
        self.assertLessEqual(output["likelihood_train"], 1.0)
        self.assertTrue((self.output_dir / "checkpoints" / "index.json").exists())

    def test_multisubject_training_exports_subject_artifacts(self):
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

        multisubject_bundle = DatasetBundle(
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

        trainer = DisrnnTrainer(
            architecture={
                "multisubject": True,
                "latent_size": 4,
                "update_net_n_units_per_layer": 8,
                "update_net_n_layers": 2,
                "choice_net_n_units_per_layer": 4,
                "choice_net_n_layers": 1,
                "activation": "leaky_relu",
                "subject_embedding_size": 3,
            },
            penalties={
                "latent_penalty": 1e-3,
                "choice_net_latent_penalty": 1e-3,
                "update_net_obs_penalty": 1e-3,
                "update_net_latent_penalty": 1e-3,
                "subject_penalty": 1e-3,
                "update_net_subject_penalty": 1e-3,
                "choice_net_subject_penalty": 1e-3,
            },
            training={
                "lr": 1e-3,
                "n_steps": 2,
                "n_warmup_steps": 1,
                "loss": "penalized_categorical",
                "loss_param": 1.0,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 0,
                "checkpoint_run_heldout_eval": False,
                "plot_choice_rule": False,
                "plot_update_rules": False,
                "plot_subject_index": 0,
                "save_output_df": False,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        output = trainer.fit(multisubject_bundle)

        self.assertTrue(output["multisubject"])
        self.assertIn("subject_artifacts", output)
        self.assertTrue((self.output_dir / "subject_index_map.json").exists())
        self.assertTrue((self.output_dir / "subject_embeddings.pkl").exists())
        subject_embeddings_df = pd.read_pickle(self.output_dir / "subject_embeddings.pkl")
        self.assertEqual(subject_embeddings_df["subject_id"].tolist(), [711041, 793446])

        params = json.loads((self.output_dir / "params.json").read_text())
        self.assertIn("multisubject_dis_rnn", params)
        self.assertIn("subject_embeddings", params["multisubject_dis_rnn"])

    def test_aligned_action_probabilities_preserve_dataframe_rows(self):
        session_df = pd.DataFrame(
            {
                "trial": [1, 2, 3, 4],
                "logit(left)": [0.0, np.nan, 2.0, -1.0],
                "logit(right)": [1.0, np.nan, 0.0, -1.0],
            }
        )

        probs = _aligned_action_probabilities_from_output_df(
            session_df,
            n_action_logits=2,
        )

        self.assertEqual(probs.shape, (4, 2))
        self.assertTrue(np.isnan(probs[1]).all())

        expected_row0 = np.exp([0.0, 1.0])
        expected_row0 = expected_row0 / expected_row0.sum()
        np.testing.assert_allclose(probs[0], expected_row0)

        expected_row2 = np.exp([2.0, 0.0])
        expected_row2 = expected_row2 / expected_row2.sum()
        np.testing.assert_allclose(probs[2], expected_row2)

    def test_multisubject_config_can_disable_global_subject_bottleneck(self):
        trainer = DisrnnTrainer(
            architecture={
                "multisubject": True,
                "latent_size": 4,
                "update_net_n_units_per_layer": 8,
                "update_net_n_layers": 2,
                "choice_net_n_units_per_layer": 4,
                "choice_net_n_layers": 1,
                "activation": "leaky_relu",
                "subject_embedding_size": 3,
                "use_global_subject_bottleneck": False,
            },
            penalties={
                "latent_penalty": 1e-3,
                "choice_net_latent_penalty": 1e-3,
                "update_net_obs_penalty": 1e-3,
                "update_net_latent_penalty": 1e-3,
                "subject_penalty": 1e-3,
                "update_net_subject_penalty": 1e-3,
                "choice_net_subject_penalty": 1e-3,
            },
            training={
                "lr": 1e-3,
                "n_steps": 2,
                "n_warmup_steps": 1,
                "loss": "penalized_categorical",
                "loss_param": 1.0,
                "max_grad_norm": 1.0,
                "plot_subject_index": 0,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        dummy_dataset = types.SimpleNamespace(
            _xs=np.zeros((5, 2, 3), dtype=float),
            x_names=["Subject ID", "prev choice", "prev reward"],
            y_names=["choice"],
        )
        config, _ = trainer._build_network_configs(
            dataset=dummy_dataset,
            ignore_policy="exclude",
            metadata={"multisubject": True, "num_subjects": 2},
        )

        self.assertFalse(config.use_global_subject_bottleneck)


if __name__ == "__main__":
    unittest.main()
