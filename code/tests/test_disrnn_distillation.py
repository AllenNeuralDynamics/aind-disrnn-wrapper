"""Tests for GRU -> disRNN distillation support."""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import aind_disrnn_utils.data_loader as dl
    import numpy as np
    import pandas as pd
    from disentangled_rnns.library import rnn_utils

    from base.types import DatasetBundle
    from data_loaders.synthetic import SyntheticCognitiveAgents
    from model_trainers.disrnn_trainer import DisrnnTrainer
    from model_trainers.gru_trainer import GruTrainer
    from utils.disrnn_distillation import (
        _load_teacher_summary,
        aggregate_teacher_probabilities,
        build_teacher_ensemble,
        remap_multisubject_teacher_inputs,
        resolve_distillation_config,
    )
    from utils.multisubject import (
        build_subject_index_maps,
        compute_train_eval_session_ids,
        merge_datasets_with_subject_index,
    )

    DISTILLATION_DEPS_AVAILABLE = True
    DISTILLATION_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    DISTILLATION_DEPS_AVAILABLE = False
    DISTILLATION_IMPORT_ERROR = exc


@unittest.skipUnless(
    DISTILLATION_DEPS_AVAILABLE,
    f"distillation dependencies unavailable: {DISTILLATION_IMPORT_ERROR}",
)
class TestDisrnnDistillation(unittest.TestCase):
    """Coverage for the distillation utilities and trainer path."""

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
        self.root_dir = Path(tempfile.mkdtemp(prefix="disrnn_distillation_test_"))

    def tearDown(self):
        shutil.rmtree(self.root_dir, ignore_errors=True)

    def _base_penalties(self) -> dict[str, float]:
        return {
            "latent_penalty": 1e-3,
            "choice_net_latent_penalty": 1e-3,
            "update_net_obs_penalty": 1e-3,
            "update_net_latent_penalty": 1e-3,
        }

    def _base_disrnn_training(self) -> dict[str, object]:
        return {
            "lr": 1e-3,
            "n_steps": 2,
            "n_warmup_steps": 1,
            "loss": "penalized_categorical",
            "loss_param": 1.0,
            "max_grad_norm": 1.0,
            "checkpoint_every_n_steps": 1,
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
        }

    def _train_gru_teacher(
        self,
        *,
        output_dir: Path,
        bundle: DatasetBundle | None = None,
        multisubject: bool = False,
        seed: int = 42,
    ) -> None:
        trainer = GruTrainer(
            architecture={
                "hidden_size": 8,
                "num_layers": 1,
                **(
                    {
                        "multisubject": True,
                        "subject_embedding_size": 3,
                        "subject_embedding_init": "zeros",
                    }
                    if multisubject
                    else {}
                ),
            },
            training={
                "lr": 1e-3,
                "n_steps": 2,
                "loss": "categorical",
                "loss_param": 1,
                "max_grad_norm": 1.0,
                "checkpoint_every_n_steps": 0,
                "checkpoint_plot_split_examples_every_n": 0,
                "checkpoint_save_output_df_every_n": 0,
                "checkpoint_log_eval_to_wandb": False,
                "checkpoint_log_train_to_wandb": False,
                "checkpoint_log_split_examples_to_wandb": False,
                "checkpoint_run_heldout_eval": False,
                "checkpoint_keep_media_files": True,
                "initialization_eval_before_training": False,
                "save_output_df": False,
            },
            output_dir=str(output_dir),
            seed=seed,
        )
        trainer.fit(bundle or self.bundle)

    def _make_multisubject_bundle(self, subject_order: list[int]) -> DatasetBundle:
        raw_df = self.bundle.raw.copy()
        raw_df["subject_id"] = np.where(raw_df["ses_idx"].astype(int) < 3, 711041, 793446)
        raw_df["source_ses_idx"] = raw_df["ses_idx"]
        raw_df["ses_idx"] = raw_df.apply(
            lambda row: f"{int(row['subject_id'])}__{int(row['source_ses_idx'])}",
            axis=1,
        )

        ordered_subject_ids, subject_id_to_index, index_to_subject_id = build_subject_index_maps(
            subject_order
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

    def test_aggregate_teacher_probabilities_supports_single_and_multi_teacher(self):
        teacher_a = np.array([[[0.8, 0.2], [0.1, 0.9]]], dtype=np.float32)
        teacher_b = np.array([[[0.6, 0.4], [0.3, 0.7]]], dtype=np.float32)

        single = aggregate_teacher_probabilities([teacher_a])
        np.testing.assert_allclose(single, teacher_a)

        aggregated = aggregate_teacher_probabilities([teacher_a, teacher_b])
        expected = np.array([[[0.7, 0.3], [0.2, 0.8]]], dtype=np.float32)
        np.testing.assert_allclose(aggregated, expected)

    def test_remap_multisubject_teacher_inputs_reorders_subject_indices(self):
        xs = np.array(
            [
                [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]],
                [[-1.0, -1.0, -1.0], [0.0, 1.0, 1.0]],
            ],
            dtype=np.float32,
        )

        remapped = remap_multisubject_teacher_inputs(
            xs,
            student_index_to_subject_id={0: 793446, 1: 711041},
            teacher_subject_id_to_index={711041: 0, 793446: 1},
        )

        np.testing.assert_array_equal(remapped[..., 0], np.array([[1.0, 0.0], [-1.0, 1.0]]))
        np.testing.assert_allclose(remapped[..., 1:], xs[..., 1:])

    def test_distilled_training_writes_standard_outputs_and_diagnostics(self):
        teacher_dir = self.root_dir / "teacher_gru"
        self._train_gru_teacher(output_dir=teacher_dir, seed=42)

        student_dir = self.root_dir / "student_disrnn"
        trainer = DisrnnTrainer(
            architecture={
                "latent_size": 4,
                "update_net_n_units_per_layer": 8,
                "update_net_n_layers": 2,
                "choice_net_n_units_per_layer": 4,
                "choice_net_n_layers": 1,
                "activation": "leaky_relu",
            },
            penalties=self._base_penalties(),
            distillation={
                "enabled": True,
                "teacher_model_dirs": [str(teacher_dir)],
                "temperature": 2.0,
                "aggregation": "mean_probs",
                "loss": "kl",
                "use_hard_labels": False,
                "evaluate_against_hard_labels": True,
                "save_manifest": True,
            },
            training=self._base_disrnn_training(),
            output_dir=str(student_dir),
            seed=7,
        )

        output = trainer.fit(self.bundle)

        self.assertIn("distillation", output)
        self.assertIn("train_distillation_loss", output["distillation"])
        self.assertIn("eval_distillation_loss", output["distillation"])
        self.assertEqual(output["distillation"]["teacher_count"], 1)
        self.assertEqual(len(output["checkpoints"]), 2)
        for checkpoint in output["checkpoints"]:
            self.assertIn("train_distillation_loss", checkpoint)
            self.assertIn("eval_distillation_loss", checkpoint)

        self.assertTrue((student_dir / "distillation_manifest.json").exists())
        self.assertTrue((student_dir / "output_summary.json").exists())
        self.assertTrue((student_dir / "disrnn_config.json").exists())
        self.assertTrue((student_dir / "checkpoints" / "index.json").exists())

    def test_build_teacher_ensemble_remaps_multisubject_indices_across_different_orders(self):
        teacher_bundle = self._make_multisubject_bundle([711041, 793446])
        teacher_dir = self.root_dir / "teacher_multisubject_gru"
        self._train_gru_teacher(
            output_dir=teacher_dir,
            bundle=teacher_bundle,
            multisubject=True,
            seed=11,
        )

        student_bundle = self._make_multisubject_bundle([793446, 711041])
        distillation_config = resolve_distillation_config(
            {
                "enabled": True,
                "teacher_model_dirs": [str(teacher_dir)],
                "temperature": 2.0,
                "aggregation": "mean_probs",
                "loss": "kl",
                "use_hard_labels": False,
            }
        )

        ensemble = build_teacher_ensemble(
            distillation=distillation_config,
            dataset=student_bundle.extras["dataset"],
            dataset_train=student_bundle.train_set,
            dataset_eval=student_bundle.eval_set,
            metadata=student_bundle.metadata,
            output_dir=self.root_dir / "multisubject_ensemble",
            expected_output_size=2,
        )

        xs_train, _ = student_bundle.train_set.get_all()
        xs_eval, _ = student_bundle.eval_set.get_all()
        self.assertEqual(ensemble.train_probs.shape[:2], xs_train.shape[:2])
        self.assertEqual(ensemble.eval_probs.shape[:2], xs_eval.shape[:2])

    def test_distillation_requires_teacher_paths(self):
        with self.assertRaisesRegex(ValueError, "teacher_model_dirs"):
            DisrnnTrainer(
                architecture={
                    "latent_size": 4,
                    "update_net_n_units_per_layer": 8,
                    "update_net_n_layers": 2,
                    "choice_net_n_units_per_layer": 4,
                    "choice_net_n_layers": 1,
                    "activation": "leaky_relu",
                },
                penalties=self._base_penalties(),
                distillation={"enabled": True, "teacher_model_dirs": []},
                training=self._base_disrnn_training(),
                output_dir=str(self.root_dir / "bad_student"),
                seed=1,
            )

    def test_distillation_rejects_missing_teacher_artifacts(self):
        trainer = DisrnnTrainer(
            architecture={
                "latent_size": 4,
                "update_net_n_units_per_layer": 8,
                "update_net_n_layers": 2,
                "choice_net_n_units_per_layer": 4,
                "choice_net_n_layers": 1,
                "activation": "leaky_relu",
            },
            penalties=self._base_penalties(),
            distillation={
                "enabled": True,
                "teacher_model_dirs": [str(self.root_dir / "missing_teacher")],
            },
            training=self._base_disrnn_training(),
            output_dir=str(self.root_dir / "missing_teacher_student"),
            seed=3,
        )

        with self.assertRaises(FileNotFoundError):
            trainer.fit(self.bundle)

    def test_distillation_rejects_incompatible_teacher_output_size(self):
        teacher_dir = self.root_dir / "bad_output_teacher"
        teacher_dir.mkdir(parents=True, exist_ok=True)
        (teacher_dir / "params.json").write_text("{}")
        (teacher_dir / "gru_config.json").write_text(
            json.dumps(
                {
                    "architecture": {"hidden_size": 8, "num_layers": 1, "multisubject": False},
                    "output_size": 3,
                }
            )
        )

        trainer = DisrnnTrainer(
            architecture={
                "latent_size": 4,
                "update_net_n_units_per_layer": 8,
                "update_net_n_layers": 2,
                "choice_net_n_units_per_layer": 4,
                "choice_net_n_layers": 1,
                "activation": "leaky_relu",
            },
            penalties=self._base_penalties(),
            distillation={
                "enabled": True,
                "teacher_model_dirs": [str(teacher_dir)],
            },
            training=self._base_disrnn_training(),
            output_dir=str(self.root_dir / "bad_output_student"),
            seed=5,
        )

        with self.assertRaisesRegex(ValueError, "output_size mismatch"):
            trainer.fit(self.bundle)

    def test_distillation_rejects_incompatible_teacher_multisubject_mode(self):
        teacher_dir = self.root_dir / "bad_multisubject_teacher"
        teacher_dir.mkdir(parents=True, exist_ok=True)
        (teacher_dir / "params.json").write_text("{}")
        (teacher_dir / "gru_config.json").write_text(
            json.dumps(
                {
                    "architecture": {"hidden_size": 8, "num_layers": 1, "multisubject": True},
                    "output_size": 2,
                }
            )
        )

        trainer = DisrnnTrainer(
            architecture={
                "latent_size": 4,
                "update_net_n_units_per_layer": 8,
                "update_net_n_layers": 2,
                "choice_net_n_units_per_layer": 4,
                "choice_net_n_layers": 1,
                "activation": "leaky_relu",
            },
            penalties=self._base_penalties(),
            distillation={
                "enabled": True,
                "teacher_model_dirs": [str(teacher_dir)],
            },
            training=self._base_disrnn_training(),
            output_dir=str(self.root_dir / "bad_multisubject_student"),
            seed=9,
        )

        with self.assertRaisesRegex(ValueError, "multisubject mode mismatch"):
            trainer.fit(self.bundle)

    def test_load_teacher_summary_resolves_relative_dir_from_data_path(self):
        data_root = self.root_dir / "mounted_data"
        teacher_dir = data_root / "jobs" / "job_a" / "teacher_model_a"
        teacher_dir.mkdir(parents=True, exist_ok=True)
        (teacher_dir / "params.json").write_text("{}")
        (teacher_dir / "gru_config.json").write_text(
            json.dumps(
                {
                    "architecture": {"hidden_size": 8, "num_layers": 1, "multisubject": False},
                    "output_size": 2,
                }
            )
        )

        with patch("utils.disrnn_distillation._CANDIDATE_DATA_DIRS", (data_root,)):
            summary, architecture, _ = _load_teacher_summary(
                Path("teacher_model_a"),
                student_is_multisubject=False,
                expected_output_size=2,
            )

        self.assertEqual(summary.output_size, 2)
        self.assertEqual(summary.multisubject, False)
        self.assertEqual(summary.model_dir, str(teacher_dir))
        self.assertEqual(architecture["hidden_size"], 8)

    def test_load_teacher_summary_resolves_nextflow_staged_absolute_data_path(self):
        data_root = self.root_dir / "mounted_data"
        teacher_dir = data_root / "jobs" / "job_a" / "teacher_model_a"
        teacher_dir.mkdir(parents=True, exist_ok=True)
        (teacher_dir / "params.json").write_text("{}")
        (teacher_dir / "gru_config.json").write_text(
            json.dumps(
                {
                    "architecture": {"hidden_size": 8, "num_layers": 1, "multisubject": False},
                    "output_size": 2,
                }
            )
        )

        staged_teacher_dir = (
            self.root_dir
            / "tmp"
            / "nxf.zIVPQIe7nb"
            / "capsule"
            / "data"
            / "jobs"
            / "job_a"
            / "teacher_model_a"
        )

        with patch("utils.disrnn_distillation._CANDIDATE_DATA_DIRS", (data_root,)):
            summary, architecture, _ = _load_teacher_summary(
                staged_teacher_dir,
                student_is_multisubject=False,
                expected_output_size=2,
            )

        self.assertEqual(summary.output_size, 2)
        self.assertEqual(summary.multisubject, False)
        self.assertEqual(summary.model_dir, str(teacher_dir))
        self.assertEqual(architecture["hidden_size"], 8)

    def test_load_teacher_summary_resolves_prefixed_dataset_segment_in_staged_path(self):
        data_root = self.root_dir / "mounted_data"
        teacher_dir = (
            data_root
            / "14"
            / "outputs"
            / "checkpoints"
            / "step_100000"
        )
        teacher_dir.mkdir(parents=True, exist_ok=True)
        (teacher_dir / "params.json").write_text("{}")
        (teacher_dir.parent.parent / "gru_config.json").write_text(
            json.dumps(
                {
                    "architecture": {
                        "hidden_size": 8,
                        "num_layers": 1,
                        "multisubject": False,
                    },
                    "output_size": 2,
                }
            )
        )

        staged_teacher_dir = (
            self.root_dir
            / "tmp"
            / "nxf.zIVPQIe7nb"
            / "capsule"
            / "data"
            / "mice_multisubject_train10-gru-260323"
            / "14"
            / "outputs"
            / "checkpoints"
            / "step_100000"
        )

        with patch("utils.disrnn_distillation._CANDIDATE_DATA_DIRS", (data_root,)):
            summary, architecture, _ = _load_teacher_summary(
                staged_teacher_dir,
                student_is_multisubject=False,
                expected_output_size=2,
            )

        self.assertEqual(summary.output_size, 2)
        self.assertEqual(summary.multisubject, False)
        self.assertEqual(summary.model_dir, str(teacher_dir))
        self.assertEqual(architecture["hidden_size"], 8)


if __name__ == "__main__":
    unittest.main()
