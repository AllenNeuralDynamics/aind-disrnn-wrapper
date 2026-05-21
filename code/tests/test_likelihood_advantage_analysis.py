"""Unit tests for standalone likelihood advantage analysis helpers."""

from __future__ import annotations

import math
import tempfile
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

import post_training_analysis.likelihood_advantage_analysis as likelihood_advantage_analysis
from post_training_analysis.likelihood_comparison import ResolvedLikelihoodRun


@unittest.skipIf(pd is None or np is None, "pandas or numpy is not installed")
class TestLikelihoodAdvantageAnalysis(unittest.TestCase):
    def _make_resolved_run(
        self,
        *,
        model_type: str,
        model_label: str,
        model_index: int,
    ) -> ResolvedLikelihoodRun:
        return ResolvedLikelihoodRun(
            model_dir=f"/tmp/{model_label}",
            inputs_path=f"/tmp/{model_label}/inputs.yaml",
            outputs_dir=f"/tmp/{model_label}/outputs",
            model_type=model_type,
            model_label=model_label,
            model_index=int(model_index),
            multisubject=False,
            seed=17,
            checkpoint_policy="best_eval" if model_type != "baseline_rl" else "final_fit",
            checkpoint_step=20 if model_type != "baseline_rl" else None,
            checkpoint_label="step_20" if model_type != "baseline_rl" else "final_fit",
            params_path=(
                f"/tmp/{model_label}/outputs/params.json"
                if model_type != "baseline_rl"
                else None
            ),
            baseline_output_path=(
                f"/tmp/{model_label}/outputs/baseline_rl_output.json"
                if model_type == "baseline_rl"
                else None
            ),
            artifact_selection_reason="selected",
            run_config={},
            model_config={},
        )

    def _raw_trial_rows(self) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    "subject_id": "m1",
                    "ses_idx": "m1__s1",
                    "source_ses_idx": "s1",
                    "trial": 0,
                    "animal_response": 0,
                    "earned_reward": 1.0,
                    "curriculum_name": "A",
                },
                {
                    "subject_id": "m1",
                    "ses_idx": "m1__s1",
                    "source_ses_idx": "s1",
                    "trial": 1,
                    "animal_response": 2,
                    "earned_reward": 0.0,
                    "curriculum_name": "A",
                },
                {
                    "subject_id": "m1",
                    "ses_idx": "m1__s1",
                    "source_ses_idx": "s1",
                    "trial": 2,
                    "animal_response": 1,
                    "earned_reward": 0.0,
                    "curriculum_name": "A",
                },
                {
                    "subject_id": "m1",
                    "ses_idx": "m1__s2",
                    "source_ses_idx": "s2",
                    "trial": 0,
                    "animal_response": 1,
                    "earned_reward": 1.0,
                    "curriculum_name": "A",
                },
                {
                    "subject_id": "m2",
                    "ses_idx": "m2__t1",
                    "source_ses_idx": "t1",
                    "trial": 0,
                    "animal_response": 0,
                    "earned_reward": 0.0,
                    "curriculum_name": "B",
                },
            ]
        )

    def _model_output_rows(self) -> pd.DataFrame:
        output_df = self._raw_trial_rows().copy()
        output_df["choice_prob_0"] = [0.8, 0.2, 0.3, 0.1, 0.7]
        output_df["choice_prob_1"] = [0.2, 0.3, 0.7, 0.9, 0.3]
        output_df["choice_prob_2"] = [0.0, 0.5, 0.0, 0.0, 0.0]
        return output_df

    def _single_session_trial_df(self) -> pd.DataFrame:
        actions = [0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]
        rewards = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
        return pd.DataFrame.from_records(
            [
                {
                    "subject_id": "m1",
                    "ses_idx": "m1__s1",
                    "session_id": "s1",
                    "session_idx": 1,
                    "trial_idx": trial_idx,
                    "action": action,
                    "reward": reward,
                    "curriculum_name": "A",
                    "advantage": float(trial_idx) / 10.0,
                }
                for trial_idx, (action, reward) in enumerate(
                    zip(actions, rewards),
                    start=1,
                )
            ]
        )

    def _smoke_raw_df(self) -> pd.DataFrame:
        records = []
        curricula = {"m1": "A", "m2": "B", "m3": "A"}
        for subject_offset, subject_id in enumerate(["m1", "m2", "m3"]):
            for session_idx in range(1, 4):
                ses_idx = f"{subject_id}__s{session_idx}"
                for trial in range(12):
                    action = int((trial + session_idx + subject_offset) % 2)
                    reward = float(((trial + subject_offset) % 3) == 0)
                    records.append(
                        {
                            "subject_id": subject_id,
                            "ses_idx": ses_idx,
                            "source_ses_idx": f"s{session_idx}",
                            "trial": trial,
                            "animal_response": action,
                            "earned_reward": reward,
                            "curriculum_name": curricula[subject_id],
                        }
                    )
        return pd.DataFrame.from_records(records)

    def _smoke_output_df(self) -> pd.DataFrame:
        output_df = self._smoke_raw_df().copy()
        p_chosen = []
        p_other = []
        for action, trial in zip(
            output_df["animal_response"].tolist(),
            output_df["trial"].tolist(),
        ):
            chosen = max(0.55, 0.82 - 0.01 * int(trial))
            other = 1.0 - chosen
            if int(action) == 0:
                p_chosen.append(chosen)
                p_other.append(other)
            else:
                p_chosen.append(other)
                p_other.append(chosen)
        output_df["choice_prob_0"] = p_chosen
        output_df["choice_prob_1"] = p_other
        return output_df

    def test_build_canonical_trial_dataframe_uses_one_based_indices(self):
        trial_df = likelihood_advantage_analysis._build_canonical_trial_dataframe(
            self._raw_trial_rows(),
            metadata={},
        )

        self.assertEqual(len(trial_df), 4)
        self.assertEqual(trial_df["session_id"].tolist(), ["s1", "s1", "s2", "t1"])
        self.assertEqual(trial_df["trial_idx"].tolist(), [1, 2, 1, 1])
        self.assertEqual(trial_df["session_idx"].tolist(), [1, 1, 2, 1])
        self.assertEqual(trial_df["action"].tolist(), [0, 1, 1, 0])
        self.assertEqual(trial_df["curriculum_name"].tolist(), ["A", "A", "A", "B"])

    def test_extract_model1_probability_frame_skips_non_binary_trials(self):
        prob_df = likelihood_advantage_analysis._extract_model1_probability_frame(
            self._model_output_rows(),
            n_action_logits=2,
        )

        self.assertEqual(prob_df["trial_idx"].tolist(), [1, 2, 1, 1])
        self.assertEqual(prob_df["p_model1"].tolist(), [0.8, 0.7, 0.9, 0.7])

    def test_merge_and_advantage_computation_matches_expected_logs(self):
        canonical_df = likelihood_advantage_analysis._build_canonical_trial_dataframe(
            self._raw_trial_rows(),
            metadata={},
        )
        model1_prob_df = likelihood_advantage_analysis._extract_model1_probability_frame(
            self._model_output_rows(),
            n_action_logits=2,
        )
        baseline_prob_df = pd.DataFrame.from_records(
            [
                {"subject_id": "m1", "ses_idx": "m1__s1", "trial_idx": 1, "action": 0, "p_rl": 0.4},
                {"subject_id": "m1", "ses_idx": "m1__s1", "trial_idx": 2, "action": 1, "p_rl": 0.6},
                {"subject_id": "m1", "ses_idx": "m1__s2", "trial_idx": 1, "action": 1, "p_rl": 0.5},
                {"subject_id": "m2", "ses_idx": "m2__t1", "trial_idx": 1, "action": 0, "p_rl": 0.3},
            ]
        )

        merged_df = likelihood_advantage_analysis._merge_probability_frame(
            canonical_df,
            model1_prob_df,
            prob_column="p_model1",
            source_label="gru",
        )
        merged_df = likelihood_advantage_analysis._merge_probability_frame(
            merged_df,
            baseline_prob_df,
            prob_column="p_rl",
            source_label="baseline_rl",
        )
        merged_df["p_model1"] = np.clip(merged_df["p_model1"], 1e-6, 1 - 1e-6)
        merged_df["p_rl"] = np.clip(merged_df["p_rl"], 1e-6, 1 - 1e-6)
        merged_df["advantage"] = np.log(merged_df["p_model1"]) - np.log(merged_df["p_rl"])

        expected = [
            math.log(0.8) - math.log(0.4),
            math.log(0.7) - math.log(0.6),
            math.log(0.9) - math.log(0.5),
            math.log(0.7) - math.log(0.3),
        ]
        self.assertTrue(np.allclose(merged_df["advantage"].to_numpy(dtype=float), expected))

    def test_add_candidate_variables_applies_warmup_and_history_features(self):
        enriched_df = likelihood_advantage_analysis.add_candidate_variables(
            self._single_session_trial_df(),
            history_warmup=10,
        )

        self.assertTrue(enriched_df.loc[:9, "prev_outcome"].isna().all())
        self.assertEqual(float(enriched_df.loc[10, "prev_outcome"]), 0.0)
        self.assertEqual(float(enriched_df.loc[10, "prev_action"]), 1.0)
        self.assertEqual(float(enriched_df.loc[10, "switch"]), 1.0)
        self.assertEqual(enriched_df.loc[10, "history_pattern_1"], "r")
        self.assertEqual(enriched_df.loc[10, "history_pattern_2"], "Rr")
        self.assertEqual(enriched_df.loc[10, "history_pattern_3"], "rRr")
        self.assertAlmostEqual(float(enriched_df.loc[10, "recent_reward_rate_3"]), 1.0 / 3.0)
        self.assertAlmostEqual(float(enriched_df.loc[10, "recent_reward_rate_5"]), 0.4)
        self.assertAlmostEqual(float(enriched_df.loc[10, "recent_reward_rate_10"]), 0.5)
        self.assertEqual(float(enriched_df.loc[10, "trials_since_reward"]), 1.0)
        self.assertAlmostEqual(float(enriched_df.loc[10, "recent_switch_rate_5"]), 0.6)
        self.assertEqual(str(enriched_df.loc[10, "switch_x_prev_outcome"]), "unrewarded-switch")
        self.assertAlmostEqual(float(enriched_df.loc[0, "trial_position"]), 0.0)
        self.assertAlmostEqual(float(enriched_df.loc[11, "trial_position"]), 1.0)

    def test_trial_position_is_zero_for_single_trial_session(self):
        trial_df = pd.DataFrame.from_records(
            [
                {
                    "subject_id": "m1",
                    "ses_idx": "m1__s1",
                    "session_id": "s1",
                    "session_idx": 1,
                    "trial_idx": 1,
                    "action": 0,
                    "reward": 1.0,
                    "curriculum_name": "A",
                    "advantage": 0.0,
                }
            ]
        )
        enriched_df = likelihood_advantage_analysis.add_candidate_variables(
            trial_df,
            history_warmup=0,
        )
        self.assertAlmostEqual(float(enriched_df.loc[0, "trial_position"]), 0.0)

    def test_analyze_variable_bins_continuous_uses_quantile_bins(self):
        trial_df = pd.DataFrame(
            {
                "trial_position": np.linspace(0.0, 0.9, 10),
                "advantage": np.arange(10, dtype=float),
            }
        )

        summary_df = likelihood_advantage_analysis.analyze_variable_bins(
            trial_df,
            "trial_position",
        )

        self.assertEqual(len(summary_df), 5)
        self.assertEqual(int(summary_df["n_trials"].sum()), 10)
        self.assertTrue((summary_df["n_trials"] == 2).all())

    def test_analyze_variable_bins_handles_tied_values_and_grouping(self):
        trial_df = pd.DataFrame(
            {
                "trial_position": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                "advantage": [0.0, 1.0, 2.0, 3.0, 5.0, 6.0, 7.0, 8.0],
                "session_stage": ["early"] * 4 + ["late"] * 4,
            }
        )

        summary_df = likelihood_advantage_analysis.analyze_variable_bins(
            trial_df,
            "trial_position",
            groupby_cols=["session_stage"],
        )

        self.assertIn("session_stage", summary_df.columns)
        self.assertLessEqual(int(summary_df["bin"].nunique()), 2)
        self.assertEqual(set(summary_df["session_stage"].astype(str)), {"early", "late"})

    def test_add_session_stage_matches_expected_boundaries(self):
        trial_df = pd.DataFrame.from_records(
            [
                {"subject_id": "one", "ses_idx": "one__s1", "session_idx": 1, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "two", "ses_idx": "two__s1", "session_idx": 1, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "two", "ses_idx": "two__s2", "session_idx": 2, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "three", "ses_idx": "three__s1", "session_idx": 1, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "three", "ses_idx": "three__s2", "session_idx": 2, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "three", "ses_idx": "three__s3", "session_idx": 3, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "four", "ses_idx": "four__s1", "session_idx": 1, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "four", "ses_idx": "four__s2", "session_idx": 2, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "four", "ses_idx": "four__s3", "session_idx": 3, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
                {"subject_id": "four", "ses_idx": "four__s4", "session_idx": 4, "trial_idx": 1, "action": 0, "reward": 0.0, "curriculum_name": "A", "advantage": 0.0},
            ]
        )

        staged_df = likelihood_advantage_analysis._add_session_stage(trial_df)
        stage_map = {
            (row["subject_id"], row["ses_idx"]): str(row["session_stage"])
            for row in staged_df.loc[:, ["subject_id", "ses_idx", "session_stage"]]
            .drop_duplicates()
            .to_dict(orient="records")
        }

        self.assertEqual(stage_map[("one", "one__s1")], "early")
        self.assertEqual(stage_map[("two", "two__s1")], "early")
        self.assertEqual(stage_map[("two", "two__s2")], "late")
        self.assertEqual(stage_map[("three", "three__s1")], "early")
        self.assertEqual(stage_map[("three", "three__s2")], "mid")
        self.assertEqual(stage_map[("three", "three__s3")], "late")
        self.assertEqual(stage_map[("four", "four__s2")], "mid")
        self.assertEqual(stage_map[("four", "four__s3")], "late")

    def test_build_subject_mean_advantage_df_is_seed_stable(self):
        trial_df = pd.DataFrame.from_records(
            [
                {"subject_id": "m1", "advantage": 0.2, "curriculum_name": "A"},
                {"subject_id": "m1", "advantage": 0.4, "curriculum_name": "A"},
                {"subject_id": "m2", "advantage": 0.3, "curriculum_name": "Mixed"},
                {"subject_id": "m2", "advantage": 0.1, "curriculum_name": "Mixed"},
                {"subject_id": "m3", "advantage": 0.0, "curriculum_name": "Unknown"},
            ]
        )

        with mock.patch.object(
            likelihood_advantage_analysis.lc,
            "_build_curriculum_palette",
            return_value={
                "A": "#111111",
                "Mixed": "#222222",
                "Unknown": "#bdbdbd",
            },
        ):
            first_df = likelihood_advantage_analysis._build_subject_mean_advantage_df(
                trial_df,
                jitter_seed=0,
            )
            second_df = likelihood_advantage_analysis._build_subject_mean_advantage_df(
                trial_df,
                jitter_seed=0,
            )
            third_df = likelihood_advantage_analysis._build_subject_mean_advantage_df(
                trial_df,
                jitter_seed=1,
            )

        self.assertEqual(len(first_df), 3)
        self.assertTrue(
            np.allclose(
                first_df["x_jitter"].to_numpy(dtype=float),
                second_df["x_jitter"].to_numpy(dtype=float),
            )
        )
        self.assertFalse(
            np.allclose(
                first_df["x_jitter"].to_numpy(dtype=float),
                third_df["x_jitter"].to_numpy(dtype=float),
            )
        )
        color_map = dict(zip(first_df["curriculum_name"], first_df["color"]))
        self.assertEqual(
            color_map["Unknown"],
            likelihood_advantage_analysis.lc._DEFAULT_CURRICULUM_COLORS["Unknown"],
        )
        self.assertTrue(str(color_map["Mixed"]).startswith("#"))

    def test_validate_model_pair_rejects_invalid_model_types(self):
        with self.assertRaisesRegex(ValueError, "model1_dir must resolve"):
            likelihood_advantage_analysis._validate_model_pair(
                self._make_resolved_run(
                    model_type="baseline_rl",
                    model_label="baseline",
                    model_index=0,
                ),
                self._make_resolved_run(
                    model_type="baseline_rl",
                    model_label="baseline_2",
                    model_index=1,
                ),
            )

        with self.assertRaisesRegex(ValueError, "model2_dir must resolve"):
            likelihood_advantage_analysis._validate_model_pair(
                self._make_resolved_run(
                    model_type="gru",
                    model_label="gru",
                    model_index=0,
                ),
                self._make_resolved_run(
                    model_type="gru",
                    model_label="gru_2",
                    model_index=1,
                ),
            )

    def test_run_likelihood_advantage_analysis_smoke_with_mocked_model_outputs(self):
        model1_run = self._make_resolved_run(
            model_type="gru",
            model_label="gru_model",
            model_index=0,
        )
        model2_run = self._make_resolved_run(
            model_type="baseline_rl",
            model_label="baseline_model",
            model_index=1,
        )
        raw_df = self._smoke_raw_df()
        output_df = self._smoke_output_df()
        canonical_df = likelihood_advantage_analysis._build_canonical_trial_dataframe(
            raw_df,
            metadata={},
        )
        baseline_prob_df = canonical_df.loc[:, ["subject_id", "ses_idx", "trial_idx", "action"]].copy()
        baseline_prob_df["p_rl"] = np.where(
            baseline_prob_df["action"].to_numpy(dtype=int) == 0,
            0.62,
            0.58,
        )

        def _write_placeholder_plot(*args, output_path: Path, **kwargs):
            del args
            del kwargs
            output_path.write_text("plot")

        subject_summary_df = pd.DataFrame.from_records(
            [
                {
                    "subject_id": "m1",
                    "mean_advantage": 0.1,
                    "curriculum_name": "A",
                    "x_jitter": -0.1,
                    "color": "#111111",
                },
                {
                    "subject_id": "m2",
                    "mean_advantage": 0.2,
                    "curriculum_name": "B",
                    "x_jitter": 0.0,
                    "color": "#222222",
                },
                {
                    "subject_id": "m3",
                    "mean_advantage": 0.3,
                    "curriculum_name": "A",
                    "x_jitter": 0.1,
                    "color": "#111111",
                },
            ]
        )

        with tempfile.TemporaryDirectory(prefix="likelihood_advantage_") as tmpdir, mock.patch.object(
            likelihood_advantage_analysis.lc,
            "_resolve_likelihood_run",
            side_effect=[model1_run, model2_run],
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_load_model1_evaluation_for_split",
            return_value={
                "raw_df": raw_df,
                "output_df": output_df,
                "metadata": {},
                "n_action_logits": 2,
            },
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_build_baseline_probability_frame",
            return_value=baseline_prob_df,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_build_subject_mean_advantage_df",
            return_value=subject_summary_df,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_advantage_histogram",
            side_effect=_write_placeholder_plot,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_subject_mean_advantage_scatter",
            side_effect=_write_placeholder_plot,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_variable_summary",
            side_effect=_write_placeholder_plot,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_summary_figure",
            side_effect=_write_placeholder_plot,
        ), mock.patch.object(
            likelihood_advantage_analysis,
            "_plot_session_stage_summary",
            side_effect=_write_placeholder_plot,
        ):
            result = likelihood_advantage_analysis.run_likelihood_advantage_analysis(
                model1_dir="/tmp/gru_model",
                model2_dir="/tmp/baseline_model",
                split="combined",
                output_dir=tmpdir,
                history_warmup=10,
                top_k_variables=3,
                jitter_seed=0,
            )

            expected_keys = {
                "output_dir",
                "summary",
                "trial_advantage_pickle",
                "advantage_histogram",
                "subject_mean_advantage_scatter",
                "bin_summary_long_csv",
                "bin_summary_wide_csv",
                "summary_figure",
                "variable_ranking_csv",
                "session_stage_top3_figure",
                "subject_bin_summary_csv",
            }
            self.assertEqual(set(result.keys()), expected_keys)
            for key, path in result.items():
                if key == "output_dir":
                    self.assertTrue(Path(path).exists())
                else:
                    self.assertTrue(Path(path).exists(), msg=key)

            self.assertFalse(Path(tmpdir, "trial_advantage.csv").exists())
            trial_df = pd.read_pickle(result["trial_advantage_pickle"])
            self.assertIn("advantage", trial_df.columns)
            self.assertIn("session_stage", trial_df.columns)

            long_summary_df = pd.read_csv(result["bin_summary_long_csv"])
            self.assertIn("variable", long_summary_df.columns)
            self.assertIn("mean_advantage", long_summary_df.columns)


if __name__ == "__main__":
    unittest.main()
