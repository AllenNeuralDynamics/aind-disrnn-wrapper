"""Unit tests for predictive-likelihood comparison helpers."""

from __future__ import annotations

import json
import math
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

from post_training_analysis.likelihood_comparison import (
    ResolvedLikelihoodRun,
    _baseline_session_metrics_from_probabilities,
    _build_plot_title_session_counts,
    _build_curriculum_palette,
    _deduplicate_model_labels,
    _evaluate_baseline_global_sessions,
    _evaluate_baseline_heldout_split,
    _evaluate_baseline_multisubject_sessions,
    _make_completed_split_result,
    _make_dataframe,
    _plot_pooled_likelihood_bars,
    _plot_subject_comparison_scatter,
    _plot_subject_likelihood_violins,
    _pooled_metric_from_session_metrics,
    _resolve_likelihood_run,
    _resolve_reference_lines,
    _resolve_splits_to_plot,
    _session_metrics_from_output_df,
    _session_sem,
    run_prediction_likelihood_comparison,
)
try:
    from utils.gru_evaluation import add_gru_model_results
except ModuleNotFoundError:  # pragma: no cover - local desktop Python may be minimal
    add_gru_model_results = None


@unittest.skipIf(
    pd is None or np is None or add_gru_model_results is None,
    "pandas, numpy, or GRU evaluation dependencies are not installed",
)
class TestLikelihoodComparison(unittest.TestCase):
    def _write_gru_run_dir(self, root: Path, *, label: str) -> Path:
        model_dir = root / label
        outputs_dir = model_dir / "outputs"
        checkpoints_dir = outputs_dir / "checkpoints"
        (checkpoints_dir / "step_10").mkdir(parents=True, exist_ok=True)
        (checkpoints_dir / "step_20").mkdir(parents=True, exist_ok=True)

        (model_dir / "inputs.yaml").write_text(
            """
data:
  type: mice_snapshot
  _target_: data_loaders.mice.MiceSnapshotDatasetLoader
  subject_ids: [101]
  mature_only: true
  ignore_policy: exclude
  eval_every_n: 2
  test_subject_ids: [202]
model:
  type: gru
  architecture:
    hidden_size: 8
    num_layers: 1
    multisubject: false
seed: 7
"""
        )
        (outputs_dir / "gru_config.json").write_text(
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
        (outputs_dir / "params.json").write_text(json.dumps({"final": True}))
        (checkpoints_dir / "step_10" / "params.json").write_text(json.dumps({"step": 10}))
        (checkpoints_dir / "step_20" / "params.json").write_text(json.dumps({"step": 20}))
        (checkpoints_dir / "index.json").write_text(
            json.dumps(
                {
                    "checkpoints": [
                        {
                            "step": 10,
                            "eval_likelihood": 0.61,
                            "params_path": "/results/outputs/checkpoints/step_10/params.json",
                        },
                        {
                            "step": 20,
                            "eval_likelihood": 0.78,
                            "params_path": "/results/outputs/checkpoints/step_20/params.json",
                        },
                    ]
                }
            )
        )
        (outputs_dir / "output_summary.json").write_text(json.dumps({}))
        return model_dir

    def _write_disrnn_run_dir(self, root: Path, *, label: str) -> Path:
        model_dir = root / label
        outputs_dir = model_dir / "outputs"
        checkpoints_dir = outputs_dir / "checkpoints"
        (checkpoints_dir / "step_5").mkdir(parents=True, exist_ok=True)
        (checkpoints_dir / "step_15").mkdir(parents=True, exist_ok=True)

        (model_dir / "inputs.yaml").write_text(
            """
data:
  type: mice_snapshot
  _target_: data_loaders.mice.MiceSnapshotDatasetLoader
  subject_ids: [111]
  mature_only: true
  ignore_policy: exclude
  eval_every_n: 2
  test_subject_ids: [333]
model:
  type: disrnn
  architecture:
    multisubject: false
    latent_size: 4
    update_net_n_units_per_layer: 8
    update_net_n_layers: 2
    choice_net_n_units_per_layer: 4
    choice_net_n_layers: 1
    activation: leaky_relu
  penalties:
    latent_penalty: 0.001
    choice_net_latent_penalty: 0.001
    update_net_obs_penalty: 0.001
    update_net_latent_penalty: 0.001
seed: 11
"""
        )
        (outputs_dir / "disrnn_config.json").write_text(
            json.dumps(
                {
                    "obs_size": 2,
                    "output_size": 2,
                    "latent_size": 4,
                    "update_net_n_units_per_layer": 8,
                    "update_net_n_layers": 2,
                    "choice_net_n_units_per_layer": 4,
                    "choice_net_n_layers": 1,
                    "activation": "leaky_relu",
                }
            )
        )
        (outputs_dir / "params.json").write_text(json.dumps({"final": True}))
        (checkpoints_dir / "step_5" / "params.json").write_text(json.dumps({"step": 5}))
        (checkpoints_dir / "step_15" / "params.json").write_text(json.dumps({"step": 15}))
        (checkpoints_dir / "index.json").write_text(
            json.dumps(
                {
                    "checkpoints": [
                        {
                            "step": 5,
                            "eval_likelihood": 0.42,
                            "params_path": "/results/outputs/checkpoints/step_5/params.json",
                        },
                        {
                            "step": 15,
                            "eval_likelihood": 0.67,
                            "params_path": "/results/outputs/checkpoints/step_15/params.json",
                        },
                    ]
                }
            )
        )
        (outputs_dir / "output_summary.json").write_text(json.dumps({}))
        return model_dir

    def _write_baseline_run_dir(
        self,
        root: Path,
        *,
        label: str,
        multisubject: bool = False,
    ) -> Path:
        model_dir = root / label
        outputs_dir = model_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "inputs.yaml").write_text(
            f"""
data:
  type: mice_snapshot
  _target_: data_loaders.mice.MiceSnapshotDatasetLoader
  subject_ids: [222]
  mature_only: true
  ignore_policy: exclude
  eval_every_n: 2
  test_subject_ids: [444]
model:
  type: baseline_rl
  architecture:
    multisubject: {"true" if multisubject else "false"}
  agent_class: ForagerQLearning
  agent_kwargs:
    number_of_learning_rate: 1
    number_of_forget_rate: 1
    choice_kernel: one_step
    action_selection: softmax
seed: 13
"""
        )
        baseline_output = {
            "multisubject": bool(multisubject),
            "fit_strategy": "per_subject" if multisubject else "single_subject",
            "agent_class": "ForagerQLearning",
            "agent_kwargs": {
                "number_of_learning_rate": 1,
                "number_of_forget_rate": 1,
                "choice_kernel": "one_step",
                "action_selection": "softmax",
            },
        }
        if multisubject:
            baseline_output["fitted_params_per_subject"] = {
                "101": {"fitted_params": {"biasL": 0.1}},
                "202": {"fitted_params": {"biasL": -0.2}},
            }
        else:
            baseline_output["fitted_params"] = {"biasL": 0.1}
        (outputs_dir / "baseline_rl_output.json").write_text(
            json.dumps(baseline_output, indent=2)
        )
        return model_dir

    def _make_run(
        self,
        *,
        model_label: str = "model_a",
        model_index: int = 0,
        model_type: str = "gru",
        multisubject: bool = False,
        baseline_output_path: str | None = None,
    ) -> ResolvedLikelihoodRun:
        return ResolvedLikelihoodRun(
            model_dir=f"/tmp/{model_label}",
            inputs_path=f"/tmp/{model_label}/inputs.yaml",
            outputs_dir=f"/tmp/{model_label}/outputs",
            model_type=model_type,
            model_label=model_label,
            model_index=model_index,
            multisubject=multisubject,
            seed=7,
            checkpoint_policy="best_eval",
            checkpoint_step=10,
            checkpoint_label="step_10",
            params_path=f"/tmp/{model_label}/outputs/params.json"
            if model_type != "baseline_rl"
            else None,
            baseline_output_path=baseline_output_path,
            artifact_selection_reason=None,
            run_config={"data": {"ignore_policy": "exclude"}, "model": {"type": model_type}},
            model_config={},
        )

    def _make_rnn_output_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "ses_idx": "101__s1",
                    "trial": 0,
                    "subject_id": 101,
                    "curriculum_name": "raw_a",
                    "animal_response": 0,
                    "choice_prob_0": 0.9,
                    "choice_prob_1": 0.1,
                },
                {
                    "ses_idx": "101__s1",
                    "trial": 1,
                    "subject_id": 101,
                    "curriculum_name": "raw_a",
                    "animal_response": 1,
                    "choice_prob_0": 0.2,
                    "choice_prob_1": 0.8,
                },
                {
                    "ses_idx": "202__s2",
                    "trial": 0,
                    "subject_id": 202,
                    "curriculum_name": "raw_b",
                    "animal_response": 1,
                    "choice_prob_0": 0.4,
                    "choice_prob_1": 0.6,
                },
                {
                    "ses_idx": "202__s2",
                    "trial": 1,
                    "subject_id": 202,
                    "curriculum_name": "raw_b",
                    "animal_response": 0,
                    "choice_prob_0": 0.75,
                    "choice_prob_1": 0.25,
                },
            ]
        )

    def _make_baseline_raw_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "ses_idx": "session_1",
                    "trial": 0,
                    "subject_id": 101,
                    "curriculum_name": "Curriculum A",
                    "animal_response": 0,
                    "earned_reward": 1.0,
                },
                {
                    "ses_idx": "session_1",
                    "trial": 1,
                    "subject_id": 101,
                    "curriculum_name": "Curriculum A",
                    "animal_response": 1,
                    "earned_reward": 0.0,
                },
                {
                    "ses_idx": "session_2",
                    "trial": 0,
                    "subject_id": 202,
                    "curriculum_name": "Curriculum B",
                    "animal_response": 1,
                    "earned_reward": 1.0,
                },
                {
                    "ses_idx": "session_2",
                    "trial": 1,
                    "subject_id": 202,
                    "curriculum_name": "Curriculum B",
                    "animal_response": 0,
                    "earned_reward": 0.0,
                },
            ]
        )

    def test_deduplicate_model_labels_uses_input_order(self):
        labels = _deduplicate_model_labels(
            ["/tmp/foo", "/tmp/foo", "/tmp/bar"],
            model_labels=["same", "same", "third"],
        )
        self.assertEqual(labels, ["same", "same_2", "third"])

    def test_resolve_likelihood_run_uses_best_eval_for_rnns_and_final_fit_for_baseline(self):
        with tempfile.TemporaryDirectory(prefix="likelihood_resolve_") as tmpdir:
            root = Path(tmpdir)
            gru_dir = self._write_gru_run_dir(root, label="gru_run")
            disrnn_dir = self._write_disrnn_run_dir(root, label="disrnn_run")
            baseline_dir = self._write_baseline_run_dir(root, label="baseline_run")

            resolved_gru = _resolve_likelihood_run(
                model_dir=gru_dir,
                model_label="GRU",
                model_index=0,
                checkpoint_policy="best_eval",
            )
            resolved_disrnn = _resolve_likelihood_run(
                model_dir=disrnn_dir,
                model_label="disRNN",
                model_index=1,
                checkpoint_policy="best_eval",
            )
            resolved_baseline = _resolve_likelihood_run(
                model_dir=baseline_dir,
                model_label="Baseline",
                model_index=2,
                checkpoint_policy="best_eval",
            )

            self.assertEqual(resolved_gru.checkpoint_step, 20)
            self.assertTrue(str(resolved_gru.params_path).endswith("step_20/params.json"))
            self.assertEqual(resolved_disrnn.checkpoint_step, 15)
            self.assertTrue(str(resolved_disrnn.params_path).endswith("step_15/params.json"))
            self.assertEqual(resolved_baseline.checkpoint_policy, "final_fit")
            self.assertIsNone(resolved_baseline.params_path)
            self.assertTrue(
                str(resolved_baseline.baseline_output_path).endswith("baseline_rl_output.json")
            )

    def test_rnn_session_metrics_and_pooled_likelihood_use_metadata_curricula(self):
        run = self._make_run(model_type="gru")
        output_df = self._make_rnn_output_df()
        raw_df = output_df.copy()
        metadata = {
            "subject_curricula": {
                101: "Curriculum A",
                202: "Curriculum B",
            }
        }
        session_metrics_df = _session_metrics_from_output_df(
            run,
            split_name="combined",
            output_df=output_df,
            raw_df=raw_df,
            metadata=metadata,
            n_action_logits=2,
        )

        self.assertEqual(
            session_metrics_df["curriculum_name"].tolist(),
            ["Curriculum A", "Curriculum B"],
        )
        self.assertEqual(session_metrics_df["session_id"].tolist(), ["101__s1", "202__s2"])

        pooled_metric = _pooled_metric_from_session_metrics(
            run,
            split_name="combined",
            session_metrics_df=session_metrics_df,
            subject_metrics_df=_make_dataframe([], ["subject_id"]),
        )
        expected_total_log_likelihood = math.log(0.9) + math.log(0.8) + math.log(0.6) + math.log(0.75)
        expected_likelihood = math.exp(expected_total_log_likelihood / 4.0)
        self.assertAlmostEqual(
            float(pooled_metric["pooled_trial_likelihood"]),
            expected_likelihood,
            places=6,
        )

    def test_gru_results_alignment_preserves_non_lexicographic_session_order(self):
        run = self._make_run(model_type="gru", multisubject=True)
        raw_df = pd.DataFrame.from_records(
            [
                {
                    "ses_idx": "subject_b__session_1",
                    "trial": 0,
                    "subject_id": "subject_b",
                    "animal_response": 1,
                    "earned_reward": 1,
                },
                {
                    "ses_idx": "subject_b__session_1",
                    "trial": 1,
                    "subject_id": "subject_b",
                    "animal_response": 1,
                    "earned_reward": 0,
                },
                {
                    "ses_idx": "subject_a__session_1",
                    "trial": 0,
                    "subject_id": "subject_a",
                    "animal_response": 0,
                    "earned_reward": 1,
                },
                {
                    "ses_idx": "subject_a__session_1",
                    "trial": 1,
                    "subject_id": "subject_a",
                    "animal_response": 0,
                    "earned_reward": 0,
                },
            ]
        )

        network_states = np.zeros((2, 2, 3), dtype=float)
        yhat = np.array(
            [
                [[-5.0, 5.0], [5.0, -5.0]],
                [[-4.0, 4.0], [4.0, -4.0]],
            ],
            dtype=float,
        )

        output_df = add_gru_model_results(
            raw_df,
            network_states,
            yhat,
            ignore_policy="exclude",
        )
        session_metrics_df = _session_metrics_from_output_df(
            run,
            split_name="eval",
            output_df=output_df,
            raw_df=raw_df,
            metadata={},
            n_action_logits=2,
        )

        pooled_likelihood = math.exp(
            float(session_metrics_df["total_log_likelihood"].sum())
            / float(session_metrics_df["total_trials"].sum())
        )
        self.assertGreater(pooled_likelihood, 0.99)

    def test_pooled_metric_session_sem_matches_session_likelihood_sem(self):
        run = self._make_run(model_type="gru")
        session_metrics_df = pd.DataFrame.from_records(
            [
                {
                    "model_index": 0,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 101,
                    "session_id": "s1",
                    "curriculum_name": "Curriculum A",
                    "total_log_likelihood": math.log(0.81),
                    "total_trials": 2,
                    "likelihood": 0.9,
                },
                {
                    "model_index": 0,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 202,
                    "session_id": "s2",
                    "curriculum_name": "Curriculum B",
                    "total_log_likelihood": math.log(0.49),
                    "total_trials": 2,
                    "likelihood": 0.7,
                },
            ]
        )
        pooled_metric = _pooled_metric_from_session_metrics(
            run,
            split_name="train",
            session_metrics_df=session_metrics_df,
            subject_metrics_df=_make_dataframe([], ["subject_id"]),
        )
        expected_sem = _session_sem([0.9, 0.7])
        self.assertAlmostEqual(float(pooled_metric["session_sem"]), expected_sem, places=6)

    def test_evaluate_baseline_global_sessions_uses_one_global_fit(self):
        run = self._make_run(
            model_type="baseline_rl",
            baseline_output_path="/tmp/baseline/global.json",
        )
        raw_df = self._make_baseline_raw_df()
        baseline_output = {
            "multisubject": False,
            "fit_strategy": "single_subject",
            "agent_class": "DummyAgent",
            "agent_kwargs": {},
            "fitted_params": {"biasL": 0.1},
        }
        with mock.patch(
            "post_training_analysis.likelihood_comparison._perform_baseline_agent_rollout",
            return_value=[
                [[0.8, 0.2], [0.2, 0.8]],
                [[0.4, 0.75], [0.6, 0.25]],
            ],
        ) as mocked_rollout:
            session_metrics_df = _evaluate_baseline_global_sessions(
                run,
                split_name="combined",
                raw_df=raw_df,
                baseline_output=baseline_output,
            )

        self.assertEqual(mocked_rollout.call_count, 1)
        self.assertEqual(session_metrics_df["subject_id"].tolist(), [101, 202])
        self.assertEqual(session_metrics_df["curriculum_name"].tolist(), ["Curriculum A", "Curriculum B"])

    def test_evaluate_baseline_multisubject_sessions_uses_per_subject_fits(self):
        run = self._make_run(
            model_type="baseline_rl",
            multisubject=True,
            baseline_output_path="/tmp/baseline/multisubject.json",
        )
        raw_df = self._make_baseline_raw_df()
        baseline_output = {
            "multisubject": True,
            "fit_strategy": "per_subject",
            "agent_class": "DummyAgent",
            "agent_kwargs": {},
            "fitted_params_per_subject": {
                "101": {"fitted_params": {"biasL": 0.1}},
                "202": {"fitted_params": {"biasL": -0.2}},
            },
        }
        with mock.patch(
            "post_training_analysis.likelihood_comparison._perform_baseline_agent_rollout",
            side_effect=[
                [[[0.8, 0.2], [0.2, 0.8]]],
                [[[0.4, 0.75], [0.6, 0.25]]],
            ],
        ) as mocked_rollout:
            session_metrics_df = _evaluate_baseline_multisubject_sessions(
                run,
                split_name="combined",
                raw_df=raw_df,
                baseline_output=baseline_output,
            )

        self.assertEqual(mocked_rollout.call_count, 2)
        self.assertEqual(session_metrics_df["subject_id"].tolist(), [101, 202])
        self.assertEqual(session_metrics_df["session_id"].tolist(), ["session_1", "session_2"])

    def test_baseline_heldout_split_skips_multisubject_outputs(self):
        with tempfile.TemporaryDirectory(prefix="baseline_skip_heldout_") as tmpdir:
            root = Path(tmpdir)
            baseline_dir = self._write_baseline_run_dir(
                root,
                label="baseline_multisubject",
                multisubject=True,
            )
            run = _resolve_likelihood_run(
                model_dir=baseline_dir,
                model_label="baseline_ms",
                model_index=0,
                checkpoint_policy="best_eval",
            )

            split_result = _evaluate_baseline_heldout_split(run)

            self.assertEqual(split_result["pooled_metric"]["status"], "skipped")
            self.assertIn("multisubject baseline RL", split_result["pooled_metric"]["skip_reason"])

    def test_reference_lines_are_split_specific(self):
        pooled_metrics_df = pd.DataFrame.from_records(
            [
                {
                    "model_index": 0,
                    "model_label": "first",
                    "model_dir": "/tmp/first",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 2,
                    "num_subjects": 2,
                    "total_log_likelihood": -1.0,
                    "total_trials": 4,
                    "pooled_trial_likelihood": 0.82,
                    "session_sem": 0.03,
                },
                {
                    "model_index": 0,
                    "model_label": "first",
                    "model_dir": "/tmp/first",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "eval",
                    "status": "skipped",
                    "skip_reason": "unsupported",
                    "num_sessions": 0,
                    "num_subjects": 0,
                    "total_log_likelihood": None,
                    "total_trials": 0,
                    "pooled_trial_likelihood": None,
                    "session_sem": None,
                },
            ]
        )

        reference_lines, omissions = _resolve_reference_lines(
            pooled_metrics_df,
            model_label="first",
            splits_to_plot=["train", "eval"],
        )

        self.assertEqual(reference_lines, {"train": 0.82})
        self.assertIn("eval", omissions)

    def test_resolve_splits_to_plot_excludes_combined_panel(self):
        pooled_metrics_df = pd.DataFrame.from_records(
            [
                {
                    "model_index": 0,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 2,
                    "num_subjects": 2,
                    "total_log_likelihood": -1.0,
                    "total_trials": 4,
                    "pooled_trial_likelihood": 0.80,
                    "session_sem": 0.04,
                },
                {
                    "model_index": 0,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "combined",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 3,
                    "num_subjects": 2,
                    "total_log_likelihood": -1.2,
                    "total_trials": 6,
                    "pooled_trial_likelihood": 0.79,
                    "session_sem": 0.03,
                },
                {
                    "model_index": 0,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "eval",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 1,
                    "num_subjects": 1,
                    "total_log_likelihood": -0.4,
                    "total_trials": 2,
                    "pooled_trial_likelihood": 0.81,
                    "session_sem": 0.0,
                },
            ]
        )

        self.assertEqual(_resolve_splits_to_plot(pooled_metrics_df), ["train", "eval"])

    def test_build_plot_title_session_counts_handles_per_model_counts(self):
        pooled_metrics_df = pd.DataFrame.from_records(
            [
                {
                    "model_index": 0,
                    "model_label": "model_b",
                    "model_dir": "/tmp/model_b",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 2,
                    "num_subjects": 2,
                    "total_log_likelihood": -1.0,
                    "total_trials": 4,
                    "pooled_trial_likelihood": 0.80,
                    "session_sem": 0.04,
                },
                {
                    "model_index": 1,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "baseline_rl",
                    "multisubject": False,
                    "split": "train",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 3,
                    "num_subjects": 2,
                    "total_log_likelihood": -1.2,
                    "total_trials": 6,
                    "pooled_trial_likelihood": 0.74,
                    "session_sem": 0.02,
                },
                {
                    "model_index": 2,
                    "model_label": "model_c",
                    "model_dir": "/tmp/model_c",
                    "model_type": "disrnn",
                    "multisubject": False,
                    "split": "train",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 4,
                    "num_subjects": 2,
                    "total_log_likelihood": -1.1,
                    "total_trials": 8,
                    "pooled_trial_likelihood": 0.77,
                    "session_sem": 0.02,
                },
                {
                    "model_index": 3,
                    "model_label": "model_d",
                    "model_dir": "/tmp/model_d",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 5,
                    "num_subjects": 2,
                    "total_log_likelihood": -1.3,
                    "total_trials": 10,
                    "pooled_trial_likelihood": 0.73,
                    "session_sem": 0.03,
                },
                {
                    "model_index": 0,
                    "model_label": "model_b",
                    "model_dir": "/tmp/model_b",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "eval",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 1,
                    "num_subjects": 1,
                    "total_log_likelihood": -0.5,
                    "total_trials": 2,
                    "pooled_trial_likelihood": 0.82,
                    "session_sem": 0.0,
                },
                {
                    "model_index": 1,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "baseline_rl",
                    "multisubject": False,
                    "split": "eval",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 1,
                    "num_subjects": 1,
                    "total_log_likelihood": -0.6,
                    "total_trials": 2,
                    "pooled_trial_likelihood": 0.76,
                    "session_sem": 0.0,
                },
                {
                    "model_index": 2,
                    "model_label": "model_c",
                    "model_dir": "/tmp/model_c",
                    "model_type": "disrnn",
                    "multisubject": False,
                    "split": "eval",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 1,
                    "num_subjects": 1,
                    "total_log_likelihood": -0.55,
                    "total_trials": 2,
                    "pooled_trial_likelihood": 0.79,
                    "session_sem": 0.0,
                },
                {
                    "model_index": 3,
                    "model_label": "model_d",
                    "model_dir": "/tmp/model_d",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "eval",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 1,
                    "num_subjects": 1,
                    "total_log_likelihood": -0.65,
                    "total_trials": 2,
                    "pooled_trial_likelihood": 0.74,
                    "session_sem": 0.0,
                },
            ]
        )

        title = _build_plot_title_session_counts(
            pooled_metrics_df,
            model_order=["model_b", "model_a", "model_c", "model_d"],
        )

        self.assertEqual(
            title,
            "Train sessions:\nmodel_b=2, model_a=3, model_c=4\nmodel_d=5\nEval sessions:\nall models=1",
        )

    def test_plot_helpers_preserve_model_order_and_reference_lines(self):
        try:
            import matplotlib.colors as mcolors
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed")

        pooled_metrics_df = pd.DataFrame.from_records(
            [
                {
                    "model_index": 0,
                    "model_label": "model_b",
                    "model_dir": "/tmp/model_b",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 2,
                    "num_subjects": 2,
                    "total_log_likelihood": -1.0,
                    "total_trials": 4,
                    "pooled_trial_likelihood": 0.80,
                    "session_sem": 0.04,
                },
                {
                    "model_index": 1,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "baseline_rl",
                    "multisubject": False,
                    "split": "train",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 2,
                    "num_subjects": 2,
                    "total_log_likelihood": -1.2,
                    "total_trials": 4,
                    "pooled_trial_likelihood": 0.74,
                    "session_sem": 0.02,
                },
                {
                    "model_index": 0,
                    "model_label": "model_b",
                    "model_dir": "/tmp/model_b",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "eval",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 1,
                    "num_subjects": 1,
                    "total_log_likelihood": -0.6,
                    "total_trials": 2,
                    "pooled_trial_likelihood": 0.78,
                    "session_sem": 0.0,
                },
                {
                    "model_index": 1,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "baseline_rl",
                    "multisubject": False,
                    "split": "eval",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 1,
                    "num_subjects": 1,
                    "total_log_likelihood": -0.7,
                    "total_trials": 2,
                    "pooled_trial_likelihood": 0.71,
                    "session_sem": 0.0,
                },
            ]
        )
        subject_metrics_df = pd.DataFrame.from_records(
            [
                {
                    "model_index": 0,
                    "model_label": "model_b",
                    "model_dir": "/tmp/model_b",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 101,
                    "subject_index": 0,
                    "curriculum_name": "Curriculum A",
                    "num_sessions": 1,
                    "total_log_likelihood": -0.2,
                    "total_trials": 2,
                    "likelihood": 0.9,
                },
                {
                    "model_index": 1,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "baseline_rl",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 101,
                    "subject_index": 0,
                    "curriculum_name": "Curriculum A",
                    "num_sessions": 1,
                    "total_log_likelihood": -0.5,
                    "total_trials": 2,
                    "likelihood": 0.75,
                },
            ]
        )
        paired_t_tests_df = pd.DataFrame.from_records(
            [
                {
                    "split": "train",
                    "reference_model_label": "model_b",
                    "comparison_model_label": "model_a",
                    "n_subjects": 1,
                    "statistic": None,
                    "p_value": None,
                    "significance_label": "",
                    "mean_reference": 0.9,
                    "mean_comparison": 0.75,
                    "mean_difference": -0.15,
                    "status": "skipped",
                    "method": None,
                    "note": "At least two paired subjects are required.",
                }
            ]
        )
        palette = _build_curriculum_palette(subject_metrics_df)
        plot_dir = Path(tempfile.mkdtemp(prefix="likelihood_plot_test_"))
        bar_path = plot_dir / "bar.png"
        violin_path = plot_dir / "violin.png"

        with mock.patch("matplotlib.pyplot.close") as mocked_close:
            _plot_pooled_likelihood_bars(
                pooled_metrics_df=pooled_metrics_df,
                model_order=["model_b", "model_a"],
                splits_to_plot=["train"],
                reference_lines={"train": 0.80},
                figure_title_session_counts=_build_plot_title_session_counts(
                    pooled_metrics_df,
                    model_order=["model_b", "model_a"],
                ),
                output_path=bar_path,
            )
            bar_fig = plt.gcf()
            self.assertEqual(
                [tick.get_text() for tick in bar_fig.axes[0].get_xticklabels()],
                ["model_b", "model_a"],
            )
            self.assertEqual(len(bar_fig.axes[0].lines), 1)
            self.assertAlmostEqual(bar_fig.axes[0].lines[0].get_ydata()[0], 0.80, places=6)
            self.assertEqual(
                [text.get_text() for text in bar_fig.axes[0].texts],
                ["0.800", "0.740"],
            )
            self.assertEqual(tuple(bar_fig.axes[0].get_ylim()), (0.6, 0.9))
            self.assertEqual(
                bar_fig.axes[0].patches[0].get_facecolor(),
                mcolors.to_rgba("#4e79a7"),
            )
            self.assertEqual(
                bar_fig.axes[0].patches[1].get_facecolor(),
                mcolors.to_rgba("#59a14f"),
            )
            self.assertIn("Train sessions", bar_fig._suptitle.get_text())
            self.assertIn("Eval sessions", bar_fig._suptitle.get_text())
            plt.close(bar_fig)

        with mock.patch("matplotlib.pyplot.close") as mocked_close:
            _plot_subject_likelihood_violins(
                pooled_metrics_df=pooled_metrics_df,
                subject_metrics_df=subject_metrics_df,
                model_order=["model_b", "model_a"],
                splits_to_plot=["train"],
                reference_lines={"train": 0.80},
                curriculum_palette=palette,
                figure_title_session_counts=_build_plot_title_session_counts(
                    pooled_metrics_df,
                    model_order=["model_b", "model_a"],
                ),
                paired_t_tests_df=paired_t_tests_df,
                output_path=violin_path,
            )
            violin_fig = plt.gcf()
            self.assertEqual(
                [tick.get_text() for tick in violin_fig.axes[0].get_xticklabels()],
                ["model_b", "model_a"],
            )
            self.assertEqual(len(violin_fig.axes[0].lines) >= 2, True)
            self.assertEqual(tuple(violin_fig.axes[0].get_ylim()), (0.5, 0.9))
            self.assertTrue(
                all(label.get_visible() for label in violin_fig.axes[0].get_yticklabels())
            )
            scatter_collections = [
                collection
                for collection in violin_fig.axes[0].collections
                if hasattr(collection, "get_offsets")
                and len(collection.get_offsets()) > 0
            ]
            self.assertTrue(scatter_collections)
            self.assertAlmostEqual(scatter_collections[0].get_alpha(), 0.45, places=6)
            self.assertTrue(
                any("p=n/a" in text.get_text() for text in violin_fig.axes[0].texts)
            )
            self.assertIn("Train sessions", violin_fig._suptitle.get_text())
            plt.close(violin_fig)

        self.assertTrue(bar_path.exists())
        self.assertTrue(violin_path.exists())
        self.assertEqual(palette["Mixed"], "#7f7f7f")
        self.assertEqual(palette["Unknown"], "#bdbdbd")

    def test_subject_comparison_scatter_plot_includes_diagonal_reference(self):
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed")

        session_metrics_df = pd.DataFrame.from_records(
            [
                {
                    "model_index": 0,
                    "model_label": "model_b",
                    "model_dir": "/tmp/model_b",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 101,
                    "session_id": "m1_s1",
                    "curriculum_name": "Curriculum A",
                    "total_log_likelihood": math.log(0.82) * 2,
                    "total_trials": 2,
                    "likelihood": 0.82,
                },
                {
                    "model_index": 0,
                    "model_label": "model_b",
                    "model_dir": "/tmp/model_b",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 101,
                    "session_id": "m1_s2",
                    "curriculum_name": "Curriculum A",
                    "total_log_likelihood": math.log(0.78) * 2,
                    "total_trials": 2,
                    "likelihood": 0.78,
                },
                {
                    "model_index": 1,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "baseline_rl",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 101,
                    "session_id": "m1_s1",
                    "curriculum_name": "Curriculum A",
                    "total_log_likelihood": math.log(0.75) * 2,
                    "total_trials": 2,
                    "likelihood": 0.75,
                },
                {
                    "model_index": 1,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "baseline_rl",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 101,
                    "session_id": "m1_s2",
                    "curriculum_name": "Curriculum A",
                    "total_log_likelihood": math.log(0.72) * 2,
                    "total_trials": 2,
                    "likelihood": 0.72,
                },
                {
                    "model_index": 0,
                    "model_label": "model_b",
                    "model_dir": "/tmp/model_b",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 202,
                    "session_id": "m2_s1",
                    "curriculum_name": "Curriculum B",
                    "total_log_likelihood": math.log(0.70) * 2,
                    "total_trials": 2,
                    "likelihood": 0.70,
                },
                {
                    "model_index": 0,
                    "model_label": "model_b",
                    "model_dir": "/tmp/model_b",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 202,
                    "session_id": "m2_s2",
                    "curriculum_name": "Curriculum B",
                    "total_log_likelihood": math.log(0.68) * 2,
                    "total_trials": 2,
                    "likelihood": 0.68,
                },
                {
                    "model_index": 1,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "baseline_rl",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 202,
                    "session_id": "m2_s1",
                    "curriculum_name": "Curriculum B",
                    "total_log_likelihood": math.log(0.76) * 2,
                    "total_trials": 2,
                    "likelihood": 0.76,
                },
                {
                    "model_index": 1,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "baseline_rl",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 202,
                    "session_id": "m2_s2",
                    "curriculum_name": "Curriculum B",
                    "total_log_likelihood": math.log(0.74) * 2,
                    "total_trials": 2,
                    "likelihood": 0.74,
                },
            ]
        )
        subject_metrics_df = pd.DataFrame.from_records(
            [
                {
                    "model_index": 0,
                    "model_label": "model_b",
                    "model_dir": "/tmp/model_b",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 101,
                    "subject_index": 0,
                    "curriculum_name": "Curriculum A",
                    "num_sessions": 2,
                    "total_log_likelihood": math.log(0.82) * 2 + math.log(0.78) * 2,
                    "total_trials": 4,
                    "likelihood": math.exp((math.log(0.82) * 2 + math.log(0.78) * 2) / 4),
                },
                {
                    "model_index": 1,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "baseline_rl",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 101,
                    "subject_index": 0,
                    "curriculum_name": "Curriculum A",
                    "num_sessions": 2,
                    "total_log_likelihood": math.log(0.75) * 2 + math.log(0.72) * 2,
                    "total_trials": 4,
                    "likelihood": math.exp((math.log(0.75) * 2 + math.log(0.72) * 2) / 4),
                },
                {
                    "model_index": 0,
                    "model_label": "model_b",
                    "model_dir": "/tmp/model_b",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 202,
                    "subject_index": 1,
                    "curriculum_name": "Curriculum B",
                    "num_sessions": 2,
                    "total_log_likelihood": math.log(0.70) * 2 + math.log(0.68) * 2,
                    "total_trials": 4,
                    "likelihood": math.exp((math.log(0.70) * 2 + math.log(0.68) * 2) / 4),
                },
                {
                    "model_index": 1,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "baseline_rl",
                    "multisubject": False,
                    "split": "train",
                    "subject_id": 202,
                    "subject_index": 1,
                    "curriculum_name": "Curriculum B",
                    "num_sessions": 2,
                    "total_log_likelihood": math.log(0.76) * 2 + math.log(0.74) * 2,
                    "total_trials": 4,
                    "likelihood": math.exp((math.log(0.76) * 2 + math.log(0.74) * 2) / 4),
                },
            ]
        )
        palette = _build_curriculum_palette(subject_metrics_df)
        pooled_metrics_df = pd.DataFrame.from_records(
            [
                {
                    "model_index": 0,
                    "model_label": "model_b",
                    "model_dir": "/tmp/model_b",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "train",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 2,
                    "num_subjects": 1,
                    "total_log_likelihood": -1.0,
                    "total_trials": 4,
                    "pooled_trial_likelihood": 0.80,
                    "session_sem": 0.04,
                },
                {
                    "model_index": 1,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "baseline_rl",
                    "multisubject": False,
                    "split": "train",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 2,
                    "num_subjects": 1,
                    "total_log_likelihood": -1.2,
                    "total_trials": 4,
                    "pooled_trial_likelihood": 0.74,
                    "session_sem": 0.02,
                },
                {
                    "model_index": 0,
                    "model_label": "model_b",
                    "model_dir": "/tmp/model_b",
                    "model_type": "gru",
                    "multisubject": False,
                    "split": "eval",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 1,
                    "num_subjects": 1,
                    "total_log_likelihood": -0.3,
                    "total_trials": 2,
                    "pooled_trial_likelihood": 0.86,
                    "session_sem": 0.0,
                },
                {
                    "model_index": 1,
                    "model_label": "model_a",
                    "model_dir": "/tmp/model_a",
                    "model_type": "baseline_rl",
                    "multisubject": False,
                    "split": "eval",
                    "status": "completed",
                    "skip_reason": None,
                    "num_sessions": 1,
                    "num_subjects": 1,
                    "total_log_likelihood": -0.4,
                    "total_trials": 2,
                    "pooled_trial_likelihood": 0.81,
                    "session_sem": 0.0,
                },
            ]
        )
        scatter_path = Path(tempfile.mkdtemp(prefix="likelihood_scatter_test_")) / "scatter.png"

        with mock.patch("matplotlib.pyplot.close"):
            _plot_subject_comparison_scatter(
                session_metrics_df=session_metrics_df,
                subject_metrics_df=subject_metrics_df,
                model_order=["model_b", "model_a"],
                splits_to_plot=["train"],
                curriculum_palette=palette,
                figure_title_session_counts=_build_plot_title_session_counts(
                    pooled_metrics_df,
                    model_order=["model_b", "model_a"],
                ),
                output_path=scatter_path,
                n_bootstrap=200,
            )
            scatter_fig = plt.gcf()
            self.assertEqual(len(scatter_fig.axes), 1)
            self.assertEqual(scatter_fig.axes[0].get_xlabel(), "model_b")
            self.assertEqual(scatter_fig.axes[0].get_ylabel(), "model_a")
            self.assertTrue(scatter_fig.axes[0].lines)
            self.assertEqual(
                list(scatter_fig.axes[0].lines[0].get_xdata()),
                list(scatter_fig.axes[0].lines[0].get_ydata()),
            )
            self.assertEqual(
                list(scatter_fig.axes[0].get_xticks()),
                list(scatter_fig.axes[0].get_yticks()),
            )
            self.assertTrue(
                all(label.get_visible() for label in scatter_fig.axes[0].get_xticklabels())
            )
            self.assertTrue(
                all(label.get_visible() for label in scatter_fig.axes[0].get_yticklabels())
            )
            scatter_collections = [
                collection
                for collection in scatter_fig.axes[0].collections
                if hasattr(collection, "get_sizes") and len(collection.get_sizes()) > 0
            ]
            self.assertTrue(scatter_collections)
            marker_sizes = sorted(float(collection.get_sizes()[0]) for collection in scatter_collections)
            self.assertIn(26.0, marker_sizes)
            self.assertIn(64.0, marker_sizes)
            self.assertIn("Train sessions", scatter_fig._suptitle.get_text())
            self.assertIn("\nEval sessions", scatter_fig._suptitle.get_text())
            plt.close(scatter_fig)

        self.assertTrue(scatter_path.exists())

    def test_run_prediction_likelihood_comparison_reuses_precomputed_session_metrics(self):
        try:
            import matplotlib.pyplot  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory(prefix="likelihood_cache_") as tmpdir:
            root = Path(tmpdir)
            gru_dir = self._write_gru_run_dir(root, label="gru_run")
            baseline_dir = self._write_baseline_run_dir(root, label="baseline_run")
            cache_path = root / "session_metrics.pkl"
            pd.DataFrame.from_records(
                [
                    {
                        "model_index": 9,
                        "model_label": "old_label",
                        "model_dir": str(gru_dir.resolve()),
                        "model_type": "gru",
                        "multisubject": False,
                        "split": "train",
                        "subject_id": 101,
                        "session_id": "train_s1",
                        "curriculum_name": "Curriculum A",
                        "total_log_likelihood": math.log(0.82) * 2,
                        "total_trials": 2,
                        "likelihood": 0.82,
                    },
                    {
                        "model_index": 9,
                        "model_label": "old_label",
                        "model_dir": str(gru_dir.resolve()),
                        "model_type": "gru",
                        "multisubject": False,
                        "split": "eval",
                        "subject_id": 101,
                        "session_id": "eval_s1",
                        "curriculum_name": "Curriculum A",
                        "total_log_likelihood": math.log(0.79) * 2,
                        "total_trials": 2,
                        "likelihood": 0.79,
                    },
                    {
                        "model_index": 9,
                        "model_label": "old_label",
                        "model_dir": str(gru_dir.resolve()),
                        "model_type": "gru",
                        "multisubject": False,
                        "split": "combined",
                        "subject_id": 101,
                        "session_id": "combined_s1",
                        "curriculum_name": "Curriculum A",
                        "total_log_likelihood": math.log(0.81) * 2,
                        "total_trials": 2,
                        "likelihood": 0.81,
                    },
                ]
            ).to_pickle(cache_path)

            def fake_split_result(run: ResolvedLikelihoodRun, split_name: str, base_value: float):
                session_rows = [
                    {
                        "model_index": run.model_index,
                        "model_label": run.model_label,
                        "model_dir": run.model_dir,
                        "model_type": run.model_type,
                        "multisubject": run.multisubject,
                        "split": split_name,
                        "subject_id": 202,
                        "session_id": f"{split_name}_s2",
                        "curriculum_name": "Curriculum B",
                        "total_log_likelihood": math.log(base_value) * 2,
                        "total_trials": 2,
                        "likelihood": base_value,
                    },
                ]
                subject_rows = [
                    {
                        "model_index": run.model_index,
                        "model_label": run.model_label,
                        "model_dir": run.model_dir,
                        "model_type": run.model_type,
                        "multisubject": run.multisubject,
                        "split": split_name,
                        "subject_id": 202,
                        "subject_index": 0,
                        "curriculum_name": "Curriculum B",
                        "num_sessions": 1,
                        "total_log_likelihood": math.log(base_value) * 2,
                        "total_trials": 2,
                        "likelihood": base_value,
                    },
                ]
                return _make_completed_split_result(
                    run,
                    split_name=split_name,
                    session_metrics_df=pd.DataFrame.from_records(session_rows),
                    subject_metrics_df=pd.DataFrame.from_records(subject_rows),
                )

            def fake_evaluate(run: ResolvedLikelihoodRun, bundle, *, hydra_config, include_heldout):
                self.assertEqual(Path(run.model_dir).name, "baseline_run")
                return {
                    "train": fake_split_result(run, "train", 0.74),
                    "eval": fake_split_result(run, "eval", 0.72),
                    "combined": fake_split_result(run, "combined", 0.73),
                }

            with mock.patch(
                "post_training_analysis.likelihood_comparison._load_training_bundle_for_run",
                return_value=(types.SimpleNamespace(), types.SimpleNamespace()),
            ), mock.patch(
                "post_training_analysis.likelihood_comparison._evaluate_resolved_run_splits",
                side_effect=fake_evaluate,
            ) as mocked_evaluate:
                result = run_prediction_likelihood_comparison(
                    [gru_dir, baseline_dir],
                    model_labels=["GRU cached", "Baseline fresh"],
                    precomputed_session_metrics_path=cache_path,
                    include_heldout=False,
                )

            self.assertEqual(mocked_evaluate.call_count, 1)
            session_metrics_df = pd.read_pickle(result["session_metrics_pickle"])
            cached_rows = session_metrics_df[session_metrics_df["model_dir"] == str(gru_dir.resolve())]
            self.assertFalse(cached_rows.empty)
            self.assertEqual(set(cached_rows["model_label"].tolist()), {"GRU cached"})

            summary_payload = json.loads(Path(result["summary"]).read_text())
            self.assertEqual(
                summary_payload["models"][0]["result_source"],
                "precomputed_session_metrics",
            )
            self.assertEqual(
                summary_payload["models"][1]["result_source"],
                "evaluated",
            )

    def test_run_prediction_likelihood_comparison_smoke_writes_artifacts(self):
        try:
            import matplotlib.pyplot  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("matplotlib is not installed")

        with tempfile.TemporaryDirectory(prefix="likelihood_smoke_") as tmpdir:
            root = Path(tmpdir)
            gru_dir = self._write_gru_run_dir(root, label="gru_run")
            disrnn_dir = self._write_disrnn_run_dir(root, label="disrnn_run")
            baseline_dir = self._write_baseline_run_dir(root, label="baseline_run")

            def fake_split_result(run: ResolvedLikelihoodRun, split_name: str, base_value: float):
                session_rows = [
                    {
                        "model_index": run.model_index,
                        "model_label": run.model_label,
                        "model_dir": run.model_dir,
                        "model_type": run.model_type,
                        "multisubject": run.multisubject,
                        "split": split_name,
                        "subject_id": 101,
                        "session_id": f"{split_name}_s1",
                        "curriculum_name": "Curriculum A",
                        "total_log_likelihood": math.log(base_value) * 2,
                        "total_trials": 2,
                        "likelihood": base_value,
                    },
                    {
                        "model_index": run.model_index,
                        "model_label": run.model_label,
                        "model_dir": run.model_dir,
                        "model_type": run.model_type,
                        "multisubject": run.multisubject,
                        "split": split_name,
                        "subject_id": 202,
                        "session_id": f"{split_name}_s2",
                        "curriculum_name": "Curriculum B",
                        "total_log_likelihood": math.log(base_value - 0.05) * 2,
                        "total_trials": 2,
                        "likelihood": base_value - 0.05,
                    },
                ]
                subject_rows = [
                    {
                        "model_index": run.model_index,
                        "model_label": run.model_label,
                        "model_dir": run.model_dir,
                        "model_type": run.model_type,
                        "multisubject": run.multisubject,
                        "split": split_name,
                        "subject_id": 101,
                        "subject_index": 0,
                        "curriculum_name": "Curriculum A",
                        "num_sessions": 1,
                        "total_log_likelihood": math.log(base_value) * 2,
                        "total_trials": 2,
                        "likelihood": base_value,
                    },
                    {
                        "model_index": run.model_index,
                        "model_label": run.model_label,
                        "model_dir": run.model_dir,
                        "model_type": run.model_type,
                        "multisubject": run.multisubject,
                        "split": split_name,
                        "subject_id": 202,
                        "subject_index": 1,
                        "curriculum_name": "Curriculum B",
                        "num_sessions": 1,
                        "total_log_likelihood": math.log(base_value - 0.05) * 2,
                        "total_trials": 2,
                        "likelihood": base_value - 0.05,
                    },
                ]
                return _make_completed_split_result(
                    run,
                    split_name=split_name,
                    session_metrics_df=pd.DataFrame.from_records(session_rows),
                    subject_metrics_df=pd.DataFrame.from_records(subject_rows),
                )

            def fake_evaluate(run: ResolvedLikelihoodRun, bundle, *, hydra_config, include_heldout):
                base_value = {
                    "gru_run": 0.82,
                    "disrnn_run": 0.78,
                    "baseline_run": 0.74,
                }[Path(run.model_dir).name]
                result = {
                    "train": fake_split_result(run, "train", base_value),
                    "eval": fake_split_result(run, "eval", base_value - 0.03),
                    "combined": fake_split_result(run, "combined", base_value - 0.01),
                }
                if include_heldout:
                    result["heldout_test"] = fake_split_result(
                        run,
                        "heldout_test",
                        base_value - 0.06,
                    )
                return result

            with mock.patch(
                "post_training_analysis.likelihood_comparison._load_training_bundle_for_run",
                return_value=(types.SimpleNamespace(), types.SimpleNamespace()),
            ), mock.patch(
                "post_training_analysis.likelihood_comparison._evaluate_resolved_run_splits",
                side_effect=fake_evaluate,
            ):
                result = run_prediction_likelihood_comparison(
                    [gru_dir, disrnn_dir, baseline_dir],
                )

            for key in (
                "summary",
                "resolved_runs",
                "pooled_metrics_csv",
                "pooled_metrics_json",
                "session_metrics_csv",
                "session_metrics_pickle",
                "subject_metrics_csv",
                "subject_metrics_pickle",
                "bar_plot",
                "violin_plot",
                "subject_comparison_plot",
                "paired_t_tests_csv",
                "paired_t_tests_json",
            ):
                self.assertTrue(Path(result[key]).exists(), key)

            summary_payload = json.loads(Path(result["summary"]).read_text())
            self.assertEqual(
                summary_payload["model_labels"],
                ["gru_run", "disrnn_run", "baseline_run"],
            )
            self.assertEqual(
                summary_payload["splits_plotted"],
                ["train", "eval", "heldout_test"],
            )


if __name__ == "__main__":
    unittest.main()
