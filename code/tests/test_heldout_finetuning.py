from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest import mock

try:
    import jax
    import optax
    import numpy as np
    import pandas as pd
    from omegaconf import OmegaConf

    from base.types import DatasetBundle
    from data_loaders import mice as mice_loader
    from data_loaders.synthetic import SyntheticCognitiveAgents
    from disentangled_rnns.library import rnn_utils
    from model_trainers.disrnn_trainer import DisrnnTrainer
    from model_trainers.gru_trainer import GruTrainer
    from models.gru_network import make_gru_network
    from models.session_conditioning import resolve_session_conditioning_from_architecture
    from post_training_analysis.heldout_finetuning import (
        run_heldout_subject_finetuning_from_config,
    )
    from utils.multisubject import (
        expand_local_multisubject_params,
        extract_subject_embeddings_from_params,
        prepend_session_index_to_multisubject_split_datasets,
    )
    from utils.session_regularized_training import (
        train_network_with_session_regularization,
    )

    HELDOUT_FINETUNE_DEPS_AVAILABLE = True
    HELDOUT_FINETUNE_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    HELDOUT_FINETUNE_DEPS_AVAILABLE = False
    HELDOUT_FINETUNE_IMPORT_ERROR = exc


@unittest.skipUnless(
    HELDOUT_FINETUNE_DEPS_AVAILABLE,
    f"Held-out fine-tuning dependencies unavailable: {HELDOUT_FINETUNE_IMPORT_ERROR}",
)
class TestHeldoutSubjectFinetuning(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp(prefix="heldout_finetune_test_"))
        self.output_root = self.tmpdir / "results"
        loader = SyntheticCognitiveAgents(
            task={
                "type": "random_walk",
                "reward_baiting": False,
                "p_min": 0.0,
                "p_max": 1.0,
                "sigma": 0.15,
                "mean": 0,
                "num_trials": 24,
                "seed": 11,
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
                    "learn_rate_rew": 0.4,
                    "learn_rate_unrew": 0.1,
                    "forget_rate_unchosen": 0.03,
                    "softmax_inverse_temperature": 4.0,
                    "biasL": 0.0,
                },
                "agent_params_session_var": {
                    "biasL": {
                        "type": "gaussian",
                        "mean": 0.0,
                        "std": 0.5,
                    }
                },
                "seed": 13,
            },
            num_trials=24,
            num_sessions=9,
            eval_every_n=2,
        )
        synthetic_bundle = loader.load()
        self.subject_ids = [711041, 793446, 812345]
        self.train_subject_ids = self.subject_ids[:2]
        self.heldout_subject_ids = self.subject_ids[2:]
        self.full_snapshot_df = self._assign_subject_ids_to_sessions(
            synthetic_bundle.raw.copy(),
            subject_ids=self.subject_ids,
        )
        self.train_snapshot_df = self.full_snapshot_df[
            self.full_snapshot_df["subject_id"].isin(self.train_subject_ids)
        ].copy()
        self.heldout_snapshot_df = self.full_snapshot_df[
            self.full_snapshot_df["subject_id"].isin(self.heldout_subject_ids)
        ].copy()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _assign_subject_ids_to_sessions(
        self,
        raw_df: pd.DataFrame,
        *,
        subject_ids: list[int],
    ) -> pd.DataFrame:
        session_ids = list(dict.fromkeys(raw_df["ses_idx"].tolist()))
        chunk_size = len(session_ids) // len(subject_ids)
        subject_by_session: dict[str, int] = {}
        for subject_index, subject_id in enumerate(subject_ids):
            start = subject_index * chunk_size
            stop = start + chunk_size
            for session_id in session_ids[start:stop]:
                subject_by_session[str(session_id)] = int(subject_id)

        labeled = raw_df.copy()
        labeled["source_ses_idx"] = labeled["ses_idx"].astype(str)
        labeled["subject_id"] = labeled["source_ses_idx"].map(subject_by_session)
        labeled["ses_idx"] = labeled.apply(
            lambda row: f"{int(row['subject_id'])}__{row['source_ses_idx']}",
            axis=1,
        )
        return labeled

    def _make_training_bundle(
        self,
        *,
        session_conditioning: bool,
    ) -> DatasetBundle:
        metadata = {
            "subject_ids": list(self.train_subject_ids),
            "ignore_policy": "exclude",
            "features": None,
            "eval_every_n": 2,
        }
        bundle = mice_loader._build_multisubject_bundle(
            df=self.train_snapshot_df,
            resolved_subject_ids=self.train_subject_ids,
            ignore_policy="exclude",
            features=None,
            eval_every_n=2,
            batch_size=None,
            batch_mode="single",
            metadata=metadata,
        )
        if not session_conditioning:
            return bundle

        dataset_full, dataset_train, dataset_eval = prepend_session_index_to_multisubject_split_datasets(
            dataset=bundle.extras["dataset"],
            dataset_train=bundle.train_set,
            dataset_eval=bundle.eval_set,
            metadata=bundle.metadata,
        )
        return DatasetBundle(
            raw=bundle.raw,
            train_set=dataset_train,
            eval_set=dataset_eval,
            metadata=dict(bundle.metadata),
            extras={"dataset": dataset_full},
        )

    def _write_source_inputs(
        self,
        model_dir: Path,
        *,
        model_type: str,
        architecture: dict[str, object],
        training: dict[str, object],
        penalties: dict[str, object] | None = None,
    ) -> None:
        cfg = {
            "data": {
                "type": "mice_snapshot",
                "subject_ids": list(self.train_subject_ids),
                "test_subject_ids": list(self.heldout_subject_ids),
                "multisubject": True,
                "mature_only": True,
                "curricula": ["Uncoupled Baiting"],
                "ignore_policy": "exclude",
                "features": None,
                "eval_every_n": 2,
                "batch_size": None,
                "batch_mode": "single",
            },
            "model": {
                "type": model_type,
                "architecture": architecture,
                "training": training,
            },
            "seed": 7,
        }
        if penalties is not None:
            cfg["model"]["penalties"] = penalties
        OmegaConf.save(config=OmegaConf.create(cfg), f=str(model_dir / "inputs.yaml"))

    def _write_common_multisubject_artifacts(
        self,
        outputs_dir: Path,
        *,
        bundle: DatasetBundle,
    ) -> None:
        subject_index_map = {
            "subject_id_to_index": {
                str(subject_id): int(index)
                for subject_id, index in bundle.metadata["subject_id_to_index"].items()
            },
            "index_to_subject_id": {
                str(index): subject_id
                for index, subject_id in bundle.metadata["index_to_subject_id"].items()
            },
        }
        (outputs_dir / "subject_index_map.json").write_text(json.dumps(subject_index_map, indent=2))
        if bundle.metadata.get("session_context") is not None:
            (outputs_dir / "session_context_map.json").write_text(
                json.dumps(bundle.metadata["session_context"], indent=2)
            )

    def _create_gru_source_run(self, *, session_conditioning: bool) -> tuple[Path, DatasetBundle, Any]:
        model_dir = self.tmpdir / ("gru_session" if session_conditioning else "gru_plain")
        outputs_dir = model_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        bundle = self._make_training_bundle(session_conditioning=session_conditioning)

        architecture = {
            "multisubject": True,
            "hidden_size": 6,
            "num_layers": 1,
            "subject_embedding_size": 3,
            "subject_embedding_init": "zeros",
            "session_encoding_type": "scalar" if session_conditioning else "none",
            "session_integration_type": "direct",
            "session_fourier_k": 4,
            "session_delta_n_layers": 2,
            "session_delta_hidden_size": 8,
        }
        training = {
            "lr": 1e-3,
            "n_steps": 5,
            "loss": "categorical",
            "loss_param": 1,
            "max_grad_norm": 1.0,
            "plot_session_context_state_space": True,
            "session_context_plot_max_subjects": 2,
        }
        self._write_source_inputs(
            model_dir,
            model_type="gru",
            architecture=architecture,
            training=training,
        )

        session_cfg = resolve_session_conditioning_from_architecture(
            architecture=architecture,
            metadata=bundle.metadata,
            multisubject=True,
            max_n_subjects=int(bundle.metadata["num_subjects"]),
            subject_embedding_size=int(architecture["subject_embedding_size"]),
            context="GRU heldout source test",
        )
        gru_make_network = make_gru_network(
            hidden_size=int(architecture["hidden_size"]),
            output_size=2,
            multisubject=True,
            max_n_subjects=int(bundle.metadata["num_subjects"]),
            subject_embedding_size=int(architecture["subject_embedding_size"]),
            subject_embedding_init=str(architecture["subject_embedding_init"]),
            session_encoding_type=str(session_cfg["session_encoding_type"]),
            session_integration_type=str(session_cfg["session_integration_type"]),
            session_fourier_k=int(session_cfg["session_fourier_k"]),
            session_delta_n_layers=int(session_cfg["session_delta_n_layers"]),
            session_delta_hidden_size=int(session_cfg["session_delta_hidden_size"]),
            session_max_index_by_subject_index=list(
                bundle.metadata.get("session_max_index_by_subject_index") or []
            ),
        )
        params, _, _ = train_network_with_session_regularization(
            lambda session_curriculum_lambda=1.0: gru_make_network(session_curriculum_lambda),
            bundle.train_set,
            bundle.eval_set,
            loss="categorical",
            loss_param=1,
            n_action_logits=2,
            session_regularization_apply=None,
            session_regularization_scale=0.0,
            opt=optax.adam(1e-3),
            random_key=jax.random.PRNGKey(0),
            n_steps=0,
        )
        (outputs_dir / "params.json").write_text(
            json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder)
        )
        (outputs_dir / "gru_config.json").write_text(
            json.dumps(
                {
                    "architecture": architecture,
                    "training": training,
                    "output_size": 2,
                },
                indent=2,
            )
        )
        self._write_common_multisubject_artifacts(outputs_dir, bundle=bundle)
        return model_dir, bundle, params

    def _create_disrnn_source_run(self, *, session_conditioning: bool) -> tuple[Path, DatasetBundle, Any]:
        model_dir = self.tmpdir / ("disrnn_session" if session_conditioning else "disrnn_plain")
        outputs_dir = model_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        bundle = self._make_training_bundle(session_conditioning=session_conditioning)

        architecture = {
            "multisubject": True,
            "latent_size": 4,
            "update_net_n_units_per_layer": 6,
            "update_net_n_layers": 2,
            "choice_net_n_units_per_layer": 4,
            "choice_net_n_layers": 1,
            "activation": "leaky_relu",
            "subject_embedding_size": 3,
            "subject_embedding_init": "zeros",
            "session_encoding_type": "scalar" if session_conditioning else "none",
            "session_integration_type": "direct",
            "session_fourier_k": 4,
            "session_delta_n_layers": 2,
            "session_delta_hidden_size": 8,
            "use_global_subject_bottleneck": True,
        }
        penalties = {
            "latent_penalty": 1e-3,
            "choice_net_latent_penalty": 1e-3,
            "update_net_obs_penalty": 1e-3,
            "update_net_latent_penalty": 1e-3,
            "subject_penalty": 1e-3,
            "update_net_subject_penalty": 1e-3,
            "choice_net_subject_penalty": 1e-3,
        }
        training = {
            "lr": 1e-3,
            "n_steps": 5,
            "loss": "penalized_categorical",
            "loss_param": 1,
            "max_grad_norm": 1.0,
            "plot_session_context_state_space": True,
            "session_context_plot_max_subjects": 2,
        }
        self._write_source_inputs(
            model_dir,
            model_type="disrnn",
            architecture=architecture,
            training=training,
            penalties=penalties,
        )

        trainer = DisrnnTrainer(
            architecture=architecture,
            penalties=penalties,
            training=training,
            output_dir=str(outputs_dir),
            seed=7,
        )
        disrnn_config, _ = trainer._build_network_configs(
            dataset=bundle.extras["dataset"],
            ignore_policy="exclude",
            metadata=bundle.metadata,
        )
        make_network = trainer._make_network_factory(disrnn_config, multisubject=True)
        params, _, _ = train_network_with_session_regularization(
            lambda session_curriculum_lambda=1.0: make_network(session_curriculum_lambda),
            bundle.train_set,
            bundle.eval_set,
            loss="penalized_categorical",
            loss_param=1,
            n_action_logits=2,
            session_regularization_apply=None,
            session_regularization_scale=0.0,
            opt=optax.adam(1e-3),
            random_key=jax.random.PRNGKey(0),
            n_steps=0,
        )
        (outputs_dir / "params.json").write_text(
            json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder)
        )
        (outputs_dir / "disrnn_config.json").write_text(
            json.dumps(asdict(disrnn_config), indent=2)
        )
        self._write_common_multisubject_artifacts(outputs_dir, bundle=bundle)
        return model_dir, bundle, params

    def _make_runner_config(self, *, model_dir: Path) -> dict[str, object]:
        return {
            "source_run": {
                "model_dir": str(model_dir),
                "checkpoint_policy": "best_eval",
            },
            "heldout_subjects": {
                "test_subject_ids": None,
                "test_subject_start": None,
                "test_subject_end": None,
                "mature_only": None,
                "curricula": None,
                "cols_to_retain": None,
            },
            "heldout_finetuning": {
                "n_steps": 2,
                "lr": 1e-2,
                "checkpoint_every_n_steps": 1,
                "batch_size": None,
                "batch_mode": "single",
                "checkpoint_plot_split_examples_every_n": 1,
                "checkpoint_save_output_df_every_n": 1,
                "train_example_sessions_per_subject": 1,
                "eval_example_sessions_per_subject": 1,
                "example_max_subjects": 2,
                "keep_media_files": True,
            },
            "output": {
                "output_root": str(self.output_root),
                "run_name_suffix": None,
            },
            "seed": 17,
        }

    def test_gru_runner_finetunes_only_new_subject_rows(self) -> None:
        model_dir, _, source_params = self._create_gru_source_run(session_conditioning=False)
        config = self._make_runner_config(model_dir=model_dir)

        with mock.patch(
            "post_training_analysis.heldout_finetuning._load_heldout_snapshot_selection",
            return_value=(self.heldout_snapshot_df.copy(), list(self.heldout_subject_ids)),
        ):
            result = run_heldout_subject_finetuning_from_config(config)

        checkpoint_metrics = json.loads(Path(result["checkpoint_metrics_path"]).read_text())
        self.assertEqual([record["step"] for record in checkpoint_metrics], [0, 1, 2])
        self.assertIn("train_loss", checkpoint_metrics[0])
        self.assertIn("eval_likelihood", checkpoint_metrics[-1])
        self.assertTrue(Path(result["loss_curve_path"]).exists())
        self.assertTrue(Path(result["likelihood_curve_path"]).exists())

        final_params = rnn_utils.to_np(json.loads(Path(result["params_path"]).read_text()))
        initial_expanded_params = expand_local_multisubject_params(
            source_params,
            n_new_subjects=1,
            init="mean",
        )
        initial_embeddings = extract_subject_embeddings_from_params(initial_expanded_params)
        final_embeddings = extract_subject_embeddings_from_params(final_params)
        self.assertEqual(final_embeddings.shape[0], 3)
        self.assertTrue(np.allclose(final_embeddings[:2], initial_embeddings[:2]))
        self.assertFalse(np.allclose(final_embeddings[2], initial_embeddings[2]))

        subject_map_path = Path(result["subject_index_map_path"])
        self.assertTrue(subject_map_path.exists())
        subject_map = json.loads(subject_map_path.read_text())
        self.assertIn(str(self.heldout_subject_ids[0]), subject_map["subject_id_to_index"])
        self.assertFalse(
            (Path(result["outputs_dir"]) / "session_context_map.json").exists()
        )

    def test_disrnn_runner_supports_session_conditioning(self) -> None:
        model_dir, _, source_params = self._create_disrnn_source_run(session_conditioning=True)
        config = self._make_runner_config(model_dir=model_dir)

        with mock.patch(
            "post_training_analysis.heldout_finetuning._load_heldout_snapshot_selection",
            return_value=(self.heldout_snapshot_df.copy(), list(self.heldout_subject_ids)),
        ):
            result = run_heldout_subject_finetuning_from_config(config)

        outputs_dir = Path(result["outputs_dir"])
        checkpoint_metrics = json.loads(Path(result["checkpoint_metrics_path"]).read_text())
        self.assertEqual(len(checkpoint_metrics), 3)
        self.assertTrue((outputs_dir / "session_context_map.json").exists())
        self.assertTrue((outputs_dir / "checkpoints" / "step_0" / "params.json").exists())
        self.assertTrue((outputs_dir / "checkpoints" / "step_2" / "output_df.csv").exists())

        disrnn_config = json.loads(Path(result["model_config_path"]).read_text())
        self.assertEqual(int(disrnn_config["max_n_subjects"]), 3)
        self.assertEqual(len(disrnn_config["session_max_index_by_subject_index"]), 3)

        final_params = rnn_utils.to_np(json.loads(Path(result["params_path"]).read_text()))
        initial_expanded_params = expand_local_multisubject_params(
            source_params,
            n_new_subjects=1,
            init="mean",
        )
        initial_embeddings = extract_subject_embeddings_from_params(initial_expanded_params)
        final_embeddings = extract_subject_embeddings_from_params(final_params)
        self.assertTrue(np.allclose(final_embeddings[:2], initial_embeddings[:2]))
        self.assertFalse(np.allclose(final_embeddings[2], initial_embeddings[2]))

    def test_runner_falls_back_to_source_run_heldout_selectors(self) -> None:
        # With no explicit selectors, the heldout selector is derived from the
        # source run's pipeline params (the reserved every-Nth subjects) — no error.
        from post_training_analysis.heldout_finetuning import _resolve_heldout_selector

        selector = _resolve_heldout_selector(
            config={"heldout_subjects": {}},
            source_data_cfg={
                "curricula": ["Coupled Baiting"],
                "min_sessions": 12,
                "heldout_every_n": 4,
                "mature_only": True,
            },
        )
        self.assertIsNone(selector["test_subject_ids"])
        self.assertEqual(selector["curricula"], ["Coupled Baiting"])
        self.assertEqual(selector["min_sessions"], 12)
        self.assertEqual(selector["heldout_every_n"], 4)

    def test_runner_allows_config_override_for_heldout_selectors(self) -> None:
        model_dir, _, _ = self._create_gru_source_run(session_conditioning=False)
        cfg_path = model_dir / "inputs.yaml"
        cfg = OmegaConf.load(cfg_path)
        cfg.data.test_subject_ids = None
        OmegaConf.save(cfg, f=str(cfg_path))

        config = self._make_runner_config(model_dir=model_dir)
        config["heldout_subjects"] = {
            "test_subject_ids": list(self.heldout_subject_ids),
            "test_subject_start": None,
            "test_subject_end": None,
            "mature_only": True,
            "curricula": ["Uncoupled Baiting"],
            "cols_to_retain": None,
        }

        captured_selector: dict[str, object] = {}

        def _fake_load_selector(*, heldout_selector):
            captured_selector.update(dict(heldout_selector))
            return self.heldout_snapshot_df.copy(), list(self.heldout_subject_ids)

        with mock.patch(
            "post_training_analysis.heldout_finetuning._load_heldout_snapshot_selection",
            side_effect=_fake_load_selector,
        ):
            run_heldout_subject_finetuning_from_config(config, output_root=self.output_root)

        self.assertEqual(
            captured_selector["test_subject_ids"],
            list(self.heldout_subject_ids),
        )
