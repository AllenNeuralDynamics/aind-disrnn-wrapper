"""Unit tests for baseline RL post-training analysis helpers."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - local desktop Python may be minimal
    pd = None

import post_training_analysis.baseline_rl_analysis as baseline_rl_analysis
from post_training_analysis.baseline_rl_analysis import (
    _compute_fit_likelihood_summary,
    _simulate_alias_sessions,
    run_baseline_rl_post_training_analysis,
)
from post_training_analysis.generative_analysis import ResolvedModelRun


@unittest.skipIf(pd is None, "pandas is not installed")
class TestBaselineRlPostTrainingAnalysis(unittest.TestCase):
    def _resolved_run_payload(self, resolved_session_ids: list[str]) -> dict[str, object]:
        return {
            "model_dir": "/tmp/model",
            "inputs_path": "/tmp/model/inputs.yaml",
            "outputs_dir": "/tmp/model/outputs",
            "model_type": "gru",
            "split": "train",
            "checkpoint_policy": "final",
            "checkpoint_step": 20,
            "checkpoint_label": "final",
            "params_path": "/tmp/model/outputs/params.json",
            "config_path": "/tmp/model/outputs/gru_config.json",
            "seed": 17,
            "multisubject": False,
            "mature_only": True,
            "ignore_policy": "exclude",
            "curricula": ["Uncoupled Baiting"],
            "features": None,
            "selection": {"subject_ids": None, "subject_start": 0, "subject_end": 2},
            "resolved_session_ids": resolved_session_ids,
            "resolved_subject_ids": ["m1", "m2"],
        }

    def _animal_sessions(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "subject_id": "m1",
                    "ses_idx": "internal_session_1",
                    "session_date": "2024-01-01",
                    "curriculum_name": "Uncoupled Baiting",
                    "current_stage_actual": "GRADUATED",
                    "n_trials": 3,
                    "choice_history": [0, 0, 1],
                    "reward_history": [1, 0, 1],
                    "nwb_suffix": "13-00-00",
                    "nwb_name": "m1_2024-01-01_13-00-00.nwb",
                },
                {
                    "subject_id": "m2",
                    "ses_idx": "internal_session_2",
                    "session_date": "2024-01-02",
                    "curriculum_name": "Uncoupled Baiting",
                    "current_stage_actual": "GRADUATED",
                    "n_trials": 4,
                    "choice_history": [1, 1, 0, 0],
                    "reward_history": [0, 1, 1, 0],
                    "nwb_suffix": "14-00-00",
                    "nwb_name": "m2_2024-01-02_14-00-00.nwb",
                },
            ]
        )

    def _fitting_df(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "_id": "row_1",
                    "nwb_name": "m1_2024-01-01_13-00-00.nwb",
                    "session_date": "2024-01-01",
                    "subject_id": "m1",
                    "agent_alias": "QLearning_L1F1_CK1_softmax",
                    "log_likelihood": -1.2,
                    "LPT": -0.4,
                    "AIC": 1.0,
                    "BIC": 2.0,
                    "n_trials": 3,
                    "params": {"biasL": 0.1},
                    "curriculum_name": "Uncoupled Baiting",
                    "current_stage_actual": "GRADUATED",
                },
                {
                    "_id": "row_2",
                    "nwb_name": "m2_2024-01-02_14-00-00.nwb",
                    "session_date": "2024-01-02",
                    "subject_id": "m2",
                    "agent_alias": "QLearning_L1F1_CK1_softmax",
                    "log_likelihood": -2.0,
                    "LPT": -0.5,
                    "AIC": 1.5,
                    "BIC": 2.5,
                    "n_trials": 4,
                    "params": "{\"biasL\": 0.2}",
                    "curriculum_name": "Uncoupled Baiting",
                    "current_stage_actual": "GRADUATED",
                },
                {
                    "_id": "row_3",
                    "nwb_name": "m1_2024-01-01_13-00-00.nwb",
                    "session_date": "2024-01-01",
                    "subject_id": "m1",
                    "agent_alias": "QLearning_L2F1_softmax",
                    "log_likelihood": -1.8,
                    "LPT": -0.6,
                    "AIC": 3.0,
                    "BIC": 4.0,
                    "n_trials": 3,
                    "params": {"biasL": 0.3},
                    "curriculum_name": "Uncoupled Baiting",
                    "current_stage_actual": "GRADUATED",
                },
            ]
        )

    def _write_resolved_run_json(
        self,
        root: Path,
        *,
        resolved_session_ids: list[str],
    ) -> Path:
        payload = self._resolved_run_payload(resolved_session_ids)
        path = root / "resolved_run.json"
        path.write_text(json.dumps(payload, indent=2))
        return path

    def _write_fitting_pickle(self, root: Path) -> Path:
        path = root / "df_baseline_rl_fitting.pkl"
        self._fitting_df().to_pickle(path)
        return path

    def _mock_simulated_sessions(self, *, agent_alias: str, fit_rows) -> pd.DataFrame:
        records = []
        for record in fit_rows.to_dict(orient="records"):
            n_trials = int(record["n_trials"])
            records.append(
                {
                    "subject_id": record["subject_id"],
                    "ses_idx": record["ses_idx"],
                    "source_ses_idx": record["ses_idx"],
                    "session_date": record["session_date"],
                    "curriculum_name": record["curriculum_name"],
                    "current_stage_actual": record["current_stage_actual"],
                    "n_trials": n_trials,
                    "choice_history": [float(idx % 2) for idx in range(n_trials)],
                    "reward_history": [float((idx + 1) % 2) for idx in range(n_trials)],
                    "nwb_suffix": record["nwb_suffix"],
                    "nwb_name": record["nwb_name"],
                    "random_seed": 101,
                    "model_dir": "/tmp/model",
                    "checkpoint_step": 20,
                    "rollout_mode": "curriculum_matched",
                    "agent_alias": agent_alias,
                    "requested_session_id": record["requested_session_id"],
                    "resolved_session_id_mode": "nwb_name",
                    "fit_gap_policy": "per_model_skip",
                    "fit_row_id": record["_id"],
                    "fit_log_likelihood": record["log_likelihood"],
                    "fit_lpt": record["LPT"],
                    "fit_aic": record.get("AIC"),
                    "fit_bic": record.get("BIC"),
                    "fit_n_trials": record.get("n_trials_fit", record["n_trials"]),
                    "fit_params": {"biasL": 0.1},
                }
            )
        return pd.DataFrame.from_records(records)

    def _mock_save_switch_figures(self, *, switch_stats, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "pooled_switch_probability.png"
        path.write_text("switch")
        return {"pooled_switch_probability": path}

    def _mock_save_history_figures(self, *, history_stats, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "history_pattern_comparison_abstract.png"
        path.write_text("history")
        return {"history_pattern_comparison_abstract": path}

    def test_run_analysis_auto_detects_nwb_name_and_logs_missing_sessions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            resolved_run_path = self._write_resolved_run_json(
                root,
                resolved_session_ids=[
                    "m1_2024-01-01_13-00-00",
                    "m2_2024-01-02_14-00-00",
                ],
            )
            fitting_df_path = self._write_fitting_pickle(root)

            with self.assertLogs("post_training_analysis.baseline_rl_analysis", level="WARNING") as cm, mock.patch.object(
                baseline_rl_analysis,
                "load_animal_session_history",
                return_value=self._animal_sessions(),
            ), mock.patch.object(
                baseline_rl_analysis,
                "_simulate_alias_sessions",
                side_effect=lambda **kwargs: self._mock_simulated_sessions(
                    agent_alias=kwargs["agent_alias"],
                    fit_rows=kwargs["fit_rows"],
                ),
            ), mock.patch.object(
                baseline_rl_analysis,
                "compute_switch_stats",
                return_value={"window_size": 10},
            ), mock.patch.object(
                baseline_rl_analysis,
                "compute_history_dependent_switch_stats",
                return_value={"config": {"max_trials_back": 3}},
            ), mock.patch.object(
                baseline_rl_analysis,
                "save_switch_figures",
                side_effect=self._mock_save_switch_figures,
            ), mock.patch.object(
                baseline_rl_analysis,
                "save_history_dependent_switch_figures",
                side_effect=self._mock_save_history_figures,
            ):
                result = run_baseline_rl_post_training_analysis(
                    resolved_run_path,
                    fitting_df_path,
                )

            self.assertEqual(result["resolved_session_id_mode"], "nwb_name")
            self.assertEqual(result["models"]["QLearning_L1F1_CK1_softmax"]["status"], "completed")
            self.assertEqual(result["models"]["QLearning_L2F1_softmax"]["missing_session_ids"], ["m2_2024-01-02_14-00-00"])
            self.assertEqual(result["models"]["ForagingCompareThreshold"]["status"], "skipped")
            self.assertIn("m2_2024-01-02_14-00-00", "\n".join(cm.output))

            fit_coverage_summary = json.loads(Path(result["fit_coverage_summary"]).read_text())
            self.assertEqual(fit_coverage_summary["resolved_session_id_mode"], "nwb_name")
            self.assertEqual(
                fit_coverage_summary["models"]["QLearning_L2F1_softmax"]["missing_session_ids"],
                ["m2_2024-01-02_14-00-00"],
            )

            ql2_fit_coverage_path = Path(
                result["models"]["QLearning_L2F1_softmax"]["fit_coverage_path"]
            )
            ql2_fit_coverage = json.loads(ql2_fit_coverage_path.read_text())
            self.assertEqual(ql2_fit_coverage["missing_session_ids"], ["m2_2024-01-02_14-00-00"])
            self.assertTrue(Path(result["animal_session_history"]).exists())
            self.assertTrue(Path(result["model_summary_csv"]).exists())
            self.assertTrue(Path(result["model_summary_json"]).exists())

    def test_run_analysis_auto_detects_ses_idx_and_bridges_to_nwb_fit_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            resolved_run_path = self._write_resolved_run_json(
                root,
                resolved_session_ids=["internal_session_1", "internal_session_2"],
            )
            fitting_df_path = self._write_fitting_pickle(root)

            with mock.patch.object(
                baseline_rl_analysis,
                "load_animal_session_history",
                return_value=self._animal_sessions(),
            ), mock.patch.object(
                baseline_rl_analysis,
                "_simulate_alias_sessions",
                side_effect=lambda **kwargs: self._mock_simulated_sessions(
                    agent_alias=kwargs["agent_alias"],
                    fit_rows=kwargs["fit_rows"],
                ),
            ), mock.patch.object(
                baseline_rl_analysis,
                "compute_switch_stats",
                return_value={"window_size": 10},
            ), mock.patch.object(
                baseline_rl_analysis,
                "compute_history_dependent_switch_stats",
                return_value={"config": {"max_trials_back": 3}},
            ), mock.patch.object(
                baseline_rl_analysis,
                "save_switch_figures",
                side_effect=self._mock_save_switch_figures,
            ), mock.patch.object(
                baseline_rl_analysis,
                "save_history_dependent_switch_figures",
                side_effect=self._mock_save_history_figures,
            ):
                result = run_baseline_rl_post_training_analysis(
                    resolved_run_path,
                    fitting_df_path,
                    model_aliases=["QLearning_L1F1_CK1_softmax"],
                )

            self.assertEqual(result["resolved_session_id_mode"], "ses_idx")
            self.assertEqual(
                result["models"]["QLearning_L1F1_CK1_softmax"]["available_session_count"],
                2,
            )
            fit_coverage_path = Path(
                result["models"]["QLearning_L1F1_CK1_softmax"]["fit_coverage_path"]
            )
            fit_coverage = json.loads(fit_coverage_path.read_text())
            self.assertEqual(
                fit_coverage["available_session_ids"],
                ["internal_session_1", "internal_session_2"],
            )

    def test_compute_fit_likelihood_summary_uses_weighted_log_likelihood(self):
        fit_rows = pd.DataFrame.from_records(
            [
                {"log_likelihood": -0.1, "LPT": -0.1, "n_trials": 1},
                {"log_likelihood": -9.0, "LPT": -1.0, "n_trials": 9},
            ]
        )

        summary = _compute_fit_likelihood_summary(fit_rows)
        arithmetic_mean_lpt = (-0.1 + -1.0) / 2.0

        self.assertAlmostEqual(summary["pooled_total_log_likelihood"], -9.1)
        self.assertEqual(summary["pooled_total_trials"], 10)
        self.assertAlmostEqual(
            summary["pooled_trial_likelihood"],
            math.exp(-9.1 / 10.0),
        )
        self.assertAlmostEqual(summary["weighted_average_lpt_secondary"], -9.1 / 10.0)
        self.assertNotAlmostEqual(
            summary["weighted_average_lpt_secondary"],
            arithmetic_mean_lpt,
        )

    def test_simulate_alias_sessions_uses_seeded_rollouts_and_source_session_ids(self):
        resolved_run = ResolvedModelRun(**self._resolved_run_payload(["internal_session_1"]))
        fit_rows = pd.DataFrame.from_records(
            [
                {
                    "_id": "row_1",
                    "requested_session_id": "internal_session_1",
                    "subject_id": "m1",
                    "ses_idx": "internal_session_1",
                    "session_date": "2024-01-01",
                    "curriculum_name": "Uncoupled Baiting",
                    "current_stage_actual": "GRADUATED",
                    "n_trials": 3,
                    "n_trials_fit": 3,
                    "nwb_suffix": "13-00-00",
                    "nwb_name": "m1_2024-01-01_13-00-00.nwb",
                    "params": {"biasL": 0.25},
                    "log_likelihood": -1.2,
                    "LPT": -0.4,
                    "AIC": 1.0,
                    "BIC": 2.0,
                }
            ]
        )

        instantiated = []

        class _FakeForager:
            def __init__(self, *, seed: int) -> None:
                self.seed = seed
                self.params = None
                instantiated.append(self)

            def set_params(self, **params) -> None:
                self.params = dict(params)

            def perform(self, task) -> None:
                self.task = task

            def get_choice_history(self):
                return [0, 1, 0]

            def get_reward_history(self):
                return [1, 0, 1]

        with mock.patch.object(
            baseline_rl_analysis,
            "derive_session_seed",
            side_effect=[101, 202],
        ), mock.patch.object(
            baseline_rl_analysis,
            "build_curriculum_matched_task",
            side_effect=lambda **kwargs: {"seed": kwargs["seed"]},
        ), mock.patch.object(
            baseline_rl_analysis,
            "_instantiate_agent_for_alias",
            side_effect=lambda agent_alias, seed: _FakeForager(seed=seed),
        ):
            simulated = _simulate_alias_sessions(
                resolved_run=resolved_run,
                agent_alias="QLearning_L1F1_CK1_softmax",
                fit_rows=fit_rows,
                session_id_mode="ses_idx",
                fit_gap_policy="per_model_skip",
                n_rollouts_per_session=2,
            )

        self.assertEqual(
            simulated["ses_idx"].tolist(),
            ["internal_session_1__rollout_0", "internal_session_1__rollout_1"],
        )
        self.assertEqual(simulated["source_ses_idx"].tolist(), ["internal_session_1", "internal_session_1"])
        self.assertEqual(simulated["random_seed"].tolist(), [101, 202])
        self.assertEqual(simulated["agent_alias"].tolist(), ["QLearning_L1F1_CK1_softmax"] * 2)
        self.assertEqual(simulated["requested_session_id"].tolist(), ["internal_session_1", "internal_session_1"])
        self.assertEqual(simulated["fit_n_trials"].tolist(), [3, 3])
        self.assertEqual([forager.params for forager in instantiated], [{"biasL": 0.25}, {"biasL": 0.25}])


if __name__ == "__main__":
    unittest.main()
