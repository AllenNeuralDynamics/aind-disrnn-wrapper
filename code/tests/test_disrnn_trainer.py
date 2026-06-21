"""Smoke tests for DisrnnTrainer."""

from __future__ import annotations

import shutil
import tempfile
import unittest
import json
import types
from pathlib import Path
from unittest.mock import patch

try:
    import numpy as np
    import pandas as pd
    import aind_disrnn_utils.data_loader as dl
    from disentangled_rnns.library import rnn_utils

    from base.types import DatasetBundle
    from data_loaders.synthetic import SyntheticCognitiveAgents
    from model_trainers.disrnn_trainer import DisrnnTrainer, _require_n_action_logits
    from model_trainers.base_multisubject_trainer import BaseMultisubjectTrainer
    from models import MultisubjectDisRnn, MultisubjectDisRnnConfig
    from models.session_conditioning import compute_session_curriculum_lambda
    from models.subject_embedding_initialization import (
        make_subject_embedding_initializer,
    )
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
        session_max_index_by_subject_index = []
        session_context_rows = []

        for subject_id in ordered_subject_ids:
            subject_df = raw_df[raw_df["subject_id"] == subject_id].copy()
            subject_df = subject_df.sort_values(["ses_idx", "trial"]).reset_index(drop=True)
            session_ids = list(dict.fromkeys(subject_df["ses_idx"].tolist()))
            source_session_ids = list(dict.fromkeys(subject_df["source_ses_idx"].tolist()))
            session_index_by_session_id = {
                str(session_id): int(index)
                for index, session_id in enumerate(session_ids, start=1)
            }
            train_ids, eval_ids = compute_train_eval_session_ids(session_ids, eval_every_n=2)
            subject_df["subject_index"] = subject_id_to_index[subject_id]
            subject_df["subject_session_index"] = subject_df["ses_idx"].map(
                lambda session_id: session_index_by_session_id[str(session_id)]
            )
            subject_df["subject_max_session_index"] = int(len(session_ids))

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
            session_max_index_by_subject_index.append(int(len(session_ids)))
            session_context_rows.append(
                {
                    "subject_id": subject_id,
                    "subject_index": int(subject_id_to_index[subject_id]),
                    "ordered_session_ids": [str(session_id) for session_id in session_ids],
                    "ordered_source_session_ids": [
                        str(session_id) for session_id in source_session_ids
                    ],
                }
            )

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
                "session_max_index_by_subject_index": session_max_index_by_subject_index,
                "session_context": {
                    "indexing": "1_based",
                    "per_subject": session_context_rows,
                },
                "num_trials": len(merged_raw_df),
                "num_sessions": len(full_session_ids),
                "train_session_ids": train_session_ids,
                "eval_session_ids": eval_session_ids,
            },
            extras={"dataset": merged_dataset},
        )

    def test_constructor_resolves_penalty_multiplier(self):
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
                "beta": 1e-3,
                "latent_penalty": 1e-3,
                "choice_net_latent_penalty": 1e-3,
                "update_net_obs_penalty": 1e-3,
                "update_net_latent_penalty_multiplier": 10,
            },
            training={},
            output_dir=str(self.output_dir),
            seed=42,
        )

        self.assertEqual(trainer.penalties["update_net_latent_penalty"], 1e-2)
        self.assertNotIn("update_net_latent_penalty_multiplier", trainer.penalties)

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
        self.assertNotIn("output_df_path", before_warmup)
        self.assertIn("split_examples", before_warmup)
        self.assertTrue(Path(after_warmup["params_path"]).exists())
        self.assertNotIn("output_df_path", after_warmup)
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

    def test_checkpointed_session_curriculum_uses_last_applied_step_for_snapshots(self):
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
                "session_encoding_type": "scalar",
                "session_n_pretrain_steps": 0,
                "session_n_warmup_steps": 5,
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
                "n_steps": 3,
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
                "plot_session_context_state_space": False,
                "save_output_df": False,
            },
            output_dir=str(self.output_dir / "session_curriculum_ckpt"),
            seed=42,
        )

        output = trainer.fit(self._make_multisubject_bundle())

        self.assertEqual(output["session_curriculum"]["total_training_steps"], 4)
        self.assertAlmostEqual(
            output["initial_evaluations"]["before_warmup"]["session_curriculum_lambda"],
            0.0,
        )
        self.assertAlmostEqual(
            output["initial_evaluations"]["after_warmup"]["session_curriculum_lambda"],
            0.0,
        )
        self.assertEqual(len(output["checkpoints"]), 2)
        self.assertAlmostEqual(output["checkpoints"][0]["session_curriculum_lambda"], 0.4)
        self.assertAlmostEqual(output["checkpoints"][1]["session_curriculum_lambda"], 0.6)
        self.assertAlmostEqual(output["session_curriculum"]["final_lambda"], 0.6)

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
        session_max_index_by_subject_index = []
        session_context_rows = []

        for subject_id in ordered_subject_ids:
            subject_df = raw_df[raw_df["subject_id"] == subject_id].copy()
            subject_df = subject_df.sort_values(["ses_idx", "trial"]).reset_index(drop=True)
            session_ids = list(dict.fromkeys(subject_df["ses_idx"].tolist()))
            source_session_ids = list(dict.fromkeys(subject_df["source_ses_idx"].tolist()))
            session_index_by_session_id = {
                str(session_id): int(index)
                for index, session_id in enumerate(session_ids, start=1)
            }
            train_ids, eval_ids = compute_train_eval_session_ids(session_ids, eval_every_n=2)
            subject_df["subject_index"] = subject_id_to_index[subject_id]
            subject_df["subject_session_index"] = subject_df["ses_idx"].map(
                lambda session_id: session_index_by_session_id[str(session_id)]
            )
            subject_df["subject_max_session_index"] = int(len(session_ids))

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
            session_max_index_by_subject_index.append(int(len(session_ids)))
            session_context_rows.append(
                {
                    "subject_id": subject_id,
                    "subject_index": int(subject_id_to_index[subject_id]),
                    "ordered_session_ids": [str(session_id) for session_id in session_ids],
                    "ordered_source_session_ids": [
                        str(session_id) for session_id in source_session_ids
                    ],
                }
            )

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
                "session_max_index_by_subject_index": session_max_index_by_subject_index,
                "session_context": {
                    "indexing": "1_based",
                    "per_subject": session_context_rows,
                },
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

    def test_multisubject_session_conditioning_training_saves_session_context_artifact(self):
        multisubject_bundle = self._make_multisubject_bundle()
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
                "session_encoding_type": "scalar",
                "session_integration_type": "direct",
                "session_delta_n_layers": 2,
                "session_delta_hidden_size": 11,
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
                "n_steps": 0,
                "n_warmup_steps": 0,
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
            output_dir=str(self.output_dir / "scalar_session"),
            seed=42,
        )

        output = trainer.fit(multisubject_bundle)

        self.assertIn("session_context_map", output["subject_artifacts"])
        self.assertTrue(Path(output["subject_artifacts"]["session_context_map"]).exists())
        self.assertIn("subject_session_context_state_space_path", output)
        self.assertTrue(Path(output["subject_session_context_state_space_path"]).exists())
        before_warmup = output["initial_evaluations"]["before_warmup"]
        self.assertTrue(before_warmup["plot_paths"]["subject_session_context_state_space"])

    def test_multisubject_session_conditioning_reduces_obs_size(self):
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
                "session_encoding_type": "fourier",
                "session_integration_type": "pre_mlp",
                "session_fourier_k": 3,
                "session_delta_n_layers": 2,
                "session_delta_hidden_size": 11,
                "session_n_pretrain_steps": 4,
                "session_n_warmup_steps": 3,
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
                "n_steps": 0,
                "n_warmup_steps": 0,
                "loss": "penalized_categorical",
                "loss_param": 1.0,
                "max_grad_norm": 1.0,
                "plot_subject_index": 0,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        dummy_dataset = types.SimpleNamespace(
            _xs=np.zeros((5, 2, 4), dtype=float),
            x_names=["Subject ID", "Session Index", "prev choice", "prev reward"],
            y_names=["choice"],
        )
        config, _ = trainer._build_network_configs(
            dataset=dummy_dataset,
            ignore_policy="exclude",
            metadata={
                "multisubject": True,
                "num_subjects": 2,
                "session_max_index_by_subject_index": [3, 3],
            },
        )

        self.assertEqual(config.obs_size, 2)
        self.assertEqual(config.x_names, ["prev choice", "prev reward"])
        self.assertEqual(config.session_encoding_type, "fourier")
        self.assertEqual(config.session_integration_type, "pre_mlp")
        self.assertEqual(config.session_fourier_k, 3)
        self.assertEqual(config.session_delta_n_layers, 2)
        self.assertEqual(config.session_delta_hidden_size, 11)
        self.assertEqual(config.session_n_pretrain_steps, 4)
        self.assertEqual(config.session_n_warmup_steps, 3)

    def test_build_network_configs_tolerates_none_curriculum_steps(self):
        # Held-out fine-tuning rebuilds the trainer from the saved source config,
        # which ships session_n_pretrain_steps / session_n_warmup_steps as explicit
        # null, and calls _build_network_configs directly (without the
        # resolve_session_curriculum_steps write-back that fit() does). A None here
        # must not reach int() and raise; it should fall back to 0 (lambda=1.0).
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
                "session_encoding_type": "scalar",
                "session_n_pretrain_steps": None,
                "session_n_warmup_steps": None,
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
                "n_steps": 0,
                "n_warmup_steps": 0,
                "loss": "penalized_categorical",
                "loss_param": 1.0,
                "max_grad_norm": 1.0,
                "plot_subject_index": 0,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        dummy_dataset = types.SimpleNamespace(
            _xs=np.zeros((5, 2, 4), dtype=float),
            x_names=["Subject ID", "Session Index", "prev choice", "prev reward"],
            y_names=["choice"],
        )
        config, _ = trainer._build_network_configs(
            dataset=dummy_dataset,
            ignore_policy="exclude",
            metadata={
                "multisubject": True,
                "num_subjects": 2,
                "session_max_index_by_subject_index": [3, 3],
            },
        )

        self.assertEqual(config.session_n_pretrain_steps, 0)
        self.assertEqual(config.session_n_warmup_steps, 0)

    def test_multisubject_session_conditioning_uses_trailing_observation_names(self):
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
                "session_encoding_type": "scalar",
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
                "n_steps": 0,
                "n_warmup_steps": 0,
                "loss": "penalized_categorical",
                "loss_param": 1.0,
                "max_grad_norm": 1.0,
                "plot_subject_index": 0,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        dummy_dataset = types.SimpleNamespace(
            _xs=np.zeros((5, 2, 4), dtype=float),
            x_names=["Session Index", "prev choice", "prev reward"],
            y_names=["choice"],
        )
        config, _ = trainer._build_network_configs(
            dataset=dummy_dataset,
            ignore_policy="exclude",
            metadata={
                "multisubject": True,
                "num_subjects": 2,
                "session_max_index_by_subject_index": [3, 3],
            },
        )

        self.assertEqual(config.obs_size, 2)
        self.assertEqual(config.x_names, ["prev choice", "prev reward"])

    def test_multisubject_session_conditioning_resolves_default_curriculum_steps(self):
        multisubject_bundle = self._make_multisubject_bundle()
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
                "session_encoding_type": "scalar",
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
                "n_warmup_steps": 2,
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
            output_dir=str(self.output_dir / "resolved_curriculum"),
            seed=42,
        )

        trainer.fit(multisubject_bundle)

        saved_cfg = json.loads(
            (self.output_dir / "resolved_curriculum" / "disrnn_config.json").read_text()
        )
        self.assertEqual(saved_cfg["session_n_pretrain_steps"], 1)
        self.assertEqual(saved_cfg["session_n_warmup_steps"], 1)

    def test_session_conditioning_requires_multisubject_mode(self):
        trainer = DisrnnTrainer(
            architecture={
                "latent_size": 4,
                "update_net_n_units_per_layer": 8,
                "update_net_n_layers": 2,
                "choice_net_n_units_per_layer": 4,
                "choice_net_n_layers": 1,
                "activation": "leaky_relu",
                "session_encoding_type": "scalar",
            },
            penalties={
                "latent_penalty": 1e-3,
                "choice_net_latent_penalty": 1e-3,
                "update_net_obs_penalty": 1e-3,
                "update_net_latent_penalty": 1e-3,
            },
            training={
                "lr": 1e-3,
                "n_steps": 0,
                "n_warmup_steps": 0,
                "loss": "penalized_categorical",
                "loss_param": 1.0,
                "max_grad_norm": 1.0,
            },
            output_dir=str(self.output_dir),
            seed=42,
        )

        dummy_dataset = types.SimpleNamespace(
            _xs=np.zeros((5, 2, 2), dtype=float),
            x_names=["prev choice", "prev reward"],
            y_names=["choice"],
        )
        with self.assertRaisesRegex(ValueError, "requires multisubject mode"):
            trainer._build_network_configs(
                dataset=dummy_dataset,
                ignore_policy="exclude",
                metadata={"multisubject": False},
            )

    def test_session_delta_regularization_skipped_when_conditioning_disabled(self):
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
            },
            training={
                "lr": 1e-3,
                "n_steps": 0,
                "n_warmup_steps": 0,
                "loss": "penalized_categorical",
                "loss_param": 1.0,
                "lambda_reg_session": 1e-3,
                "max_grad_norm": 1.0,
            },
            output_dir=str(self.output_dir / "invalid_session_reg"),
            seed=42,
        )

        # Conditioning is disabled (session_encoding_type=none by default), so a
        # positive lambda_reg_session is auto-disabled with a warning rather than
        # raising — this is what lets a sweep cross none x scalar arms safely.
        with self.assertLogs(level="WARNING") as captured:
            trainer.fit(self._make_multisubject_bundle())
        self.assertTrue(
            any(
                "skipping session-delta regularization" in message
                for message in captured.output
            ),
            captured.output,
        )

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

    def test_multisubject_config_defaults_subject_embedding_init_to_zeros(self):
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

        self.assertEqual(config.subject_embedding_init, "zeros")

    def test_multisubject_config_accepts_subject_embedding_init_modes(self):
        dummy_dataset = types.SimpleNamespace(
            _xs=np.zeros((5, 2, 3), dtype=float),
            x_names=["Subject ID", "prev choice", "prev reward"],
            y_names=["choice"],
        )

        for init_mode in ("zeros", "small_random", "subject_count_scaled_random"):
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
                    "subject_embedding_init": init_mode,
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

            config, _ = trainer._build_network_configs(
                dataset=dummy_dataset,
                ignore_policy="exclude",
                metadata={"multisubject": True, "num_subjects": 2},
            )
            self.assertEqual(config.subject_embedding_init, init_mode)

    def test_multisubject_model_rejects_unknown_subject_embedding_init(self):
        config = MultisubjectDisRnnConfig(
            obs_size=2,
            output_size=2,
            x_names=["prev choice", "prev reward"],
            y_names=["choice"],
            latent_size=4,
            update_net_n_units_per_layer=8,
            update_net_n_layers=2,
            choice_net_n_units_per_layer=4,
            choice_net_n_layers=1,
            activation="leaky_relu",
            noiseless_mode=False,
            latent_penalty=1e-3,
            choice_net_latent_penalty=1e-3,
            update_net_obs_penalty=1e-3,
            update_net_latent_penalty=1e-3,
            max_n_subjects=2,
            subject_embedding_size=3,
            subject_embedding_init="not_a_real_mode",
            subj_penalty=1e-3,
            update_net_subj_penalty=1e-3,
            choice_net_subj_penalty=1e-3,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported subject_embedding_init",
        ):
            MultisubjectDisRnn(config)

    # --- Phase A regression guards (bug fixes) --------------------------------

    def _minimal_trainer(self, training_overrides: dict) -> DisrnnTrainer:
        """A small single-subject trainer for fail-fast validation tests."""
        training = {
            "lr": 1e-3,
            "n_steps": 2,
            "n_warmup_steps": 1,
            "loss": "penalized_categorical",
            "loss_param": 1.0,
            "max_grad_norm": 1.0,
            "checkpoint_every_n_steps": 0,
            "initialization_eval_before_warmup": False,
            "initialization_eval_after_warmup": False,
            "save_output_df": False,
        }
        training.update(training_overrides)
        return DisrnnTrainer(
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
            training=training,
            output_dir=str(self.output_dir),
            seed=42,
        )

    def test_rejects_negative_n_steps(self):
        """Bug #1: negative n_steps fails fast with a clear ValueError."""
        trainer = self._minimal_trainer({"n_steps": -1})
        with self.assertRaisesRegex(ValueError, "n_steps must be >= 0"):
            trainer.fit(self.bundle)

    def test_rejects_negative_n_warmup_steps(self):
        """Bug #1: negative n_warmup_steps fails fast with a clear ValueError."""
        trainer = self._minimal_trainer({"n_warmup_steps": -1})
        with self.assertRaisesRegex(ValueError, "n_warmup_steps must be >= 0"):
            trainer.fit(self.bundle)

    def test_plot_losses_handles_empty_losses(self):
        """Bug #2: _plot_losses must not IndexError on empty loss arrays."""
        trainer = self._minimal_trainer({})
        path = trainer._plot_losses(
            {"training_loss": [], "validation_loss": []},
            title="empty",
            output_name="empty_losses.png",
        )
        self.assertTrue(Path(path).exists())

    def test_require_n_action_logits_requires_positive_n_classes(self):
        """Bug #5: missing/zero n_classes raises rather than silently inferring."""

        class _Stub:
            n_classes = 0

        yhat = np.zeros((2, 5, 3))
        with self.assertRaisesRegex(ValueError, "requires dataset.n_classes"):
            _require_n_action_logits(_Stub(), yhat, context="unit test")

    def test_require_n_action_logits_rejects_shape_mismatch(self):
        """Bug #5: output width must be n_classes + 1 (trailing penalty channel)."""

        class _Stub:
            n_classes = 2

        # width 4 != n_classes (2) + 1 -> mismatch
        yhat_bad = np.zeros((2, 5, 4))
        with self.assertRaisesRegex(ValueError, "logits shape mismatch"):
            _require_n_action_logits(_Stub(), yhat_bad, context="unit test")

        # width 3 == n_classes (2) + 1 -> accepted, returns n_classes
        yhat_ok = np.zeros((2, 5, 3))
        self.assertEqual(
            _require_n_action_logits(_Stub(), yhat_ok, context="unit test"), 2
        )

    # --- Phase B/C/D refactor guards + correctness-gap tests ------------------

    def test_inherits_base_and_sets_model_label(self):
        # Pins the parameterization used by the shared base class (T6).
        self.assertTrue(issubclass(DisrnnTrainer, BaseMultisubjectTrainer))
        self.assertEqual(DisrnnTrainer._MODEL_LABEL, "disRNN")
        self.assertEqual(DisrnnTrainer._TRAINER_CONTEXT_NAME, "DisrnnTrainer")

    def test_plot_examples_for_split_dispatches_to_disrnn_and_forwards_params(self):
        # T7: disRNN's split-example hook must call the disRNN plotter and forward params.
        trainer = self._minimal_trainer({})
        sentinel = {"plotting_skipped": True}
        with patch(
            "model_trainers.disrnn_trainer.plot_disrnn_examples_for_split",
            return_value=sentinel,
        ) as mock_plot:
            result = trainer._plot_examples_for_split(
                split_name="training",
                output_dir=self.output_dir,
                output_df=None,
                network_states=None,
                yhat_logits=None,
                params={"p": 1},
                sessions_per_subject=1,
                max_subjects_to_plot=1,
                n_action_logits=2,
                wandb_run=None,
                log_scope="Test",
            )
        self.assertIs(result, sentinel)
        mock_plot.assert_called_once()
        self.assertEqual(mock_plot.call_args.kwargs["params"], {"p": 1})

    def test_warmup_uses_penalty_free_noiseless_network(self):
        # T8: warmup trains on a noiseless config with every penalty zeroed, so the
        # "penalty-free warmup" contract holds even with loss=penalized_categorical.
        trainer = self._minimal_trainer({})
        dataset = self.bundle.extras.get("dataset")
        _config, noiseless = trainer._build_network_configs(
            dataset=dataset,
            ignore_policy="exclude",
            metadata=dict(self.bundle.metadata),
        )
        self.assertEqual(noiseless.latent_penalty, 0)
        self.assertEqual(noiseless.choice_net_latent_penalty, 0)
        self.assertEqual(noiseless.update_net_obs_penalty, 0)
        self.assertEqual(noiseless.update_net_latent_penalty, 0)
        self.assertTrue(noiseless.noiseless_mode)

    def test_session_curriculum_lambda_schedule(self):
        # T10: lambda is 0 through pretrain, ramps linearly over warmup, then 1.0 —
        # the same schedule is used for warmup, training, and eval-network builds.
        def f(step):
            return compute_session_curriculum_lambda(
                current_step=step,
                session_n_pretrain_steps=2,
                session_n_warmup_steps=4,
            )

        self.assertEqual(f(0), 0.0)
        self.assertEqual(f(2), 0.0)
        self.assertAlmostEqual(f(3), 0.25)
        self.assertAlmostEqual(f(4), 0.5)
        self.assertEqual(f(6), 1.0)
        self.assertEqual(f(100), 1.0)
        values = [f(step) for step in range(12)]
        self.assertEqual(values, sorted(values))  # monotonic non-decreasing

    def test_subject_embedding_initializer_modes(self):
        # T11: documents the differing stddev semantics of the init modes.
        import haiku as hk

        zeros = make_subject_embedding_initializer(
            subject_embedding_init="zeros", max_n_subjects=10, subject_embedding_size=4
        )
        self.assertIsInstance(zeros, hk.initializers.Constant)

        small = make_subject_embedding_initializer(
            subject_embedding_init="small_random",
            max_n_subjects=10,
            subject_embedding_size=4,
        )
        self.assertAlmostEqual(float(small.stddev), 1.0 / (4 ** 0.5))

        scaled = make_subject_embedding_initializer(
            subject_embedding_init="subject_count_scaled_random",
            max_n_subjects=16,
            subject_embedding_size=4,
        )
        self.assertAlmostEqual(float(scaled.stddev), 1.0 / (16 ** 0.5))

        with self.assertRaisesRegex(ValueError, "Unsupported subject_embedding_init"):
            make_subject_embedding_initializer(
                subject_embedding_init="nope",
                max_n_subjects=10,
                subject_embedding_size=4,
            )


if __name__ == "__main__":
    unittest.main()
