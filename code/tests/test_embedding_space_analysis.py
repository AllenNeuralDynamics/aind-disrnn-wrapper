from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd


def _install_fake_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class _FakeArtist:
        pass

    class _FakeAxes:
        def scatter(self, *args, **kwargs):
            return _FakeArtist()

        def text(self, *args, **kwargs):
            return None

        def axhline(self, *args, **kwargs):
            return None

        def axvline(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def axis(self, *args, **kwargs):
            return None

        def plot(self, *args, **kwargs):
            return None

    class _FakeFigure:
        def legend(self, *args, **kwargs):
            return None

        def suptitle(self, *args, **kwargs):
            return None

        def tight_layout(self, *args, **kwargs):
            return None

        def colorbar(self, *args, **kwargs):
            return None

        def add_axes(self, *args, **kwargs):
            return _FakeAxes()

        def savefig(self, path, *args, **kwargs):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b"fake-plot")

    class _FakeCmap:
        N = 20

        def __call__(self, index):
            return (float(index) / 20.0, 0.5, 0.5, 1.0)

    class _FakeLine2D:
        def __init__(self, *args, **kwargs) -> None:
            pass

    matplotlib_mod = types.ModuleType("matplotlib")
    pyplot_mod = types.ModuleType("matplotlib.pyplot")
    lines_mod = types.ModuleType("matplotlib.lines")

    def _subplots(nrows, ncols, figsize=None, squeeze=False):
        axes = np.asarray(
            [[_FakeAxes() for _ in range(ncols)] for _ in range(nrows)],
            dtype=object,
        )
        return _FakeFigure(), axes

    pyplot_mod.subplots = _subplots
    pyplot_mod.get_cmap = lambda name: _FakeCmap()
    pyplot_mod.close = lambda fig=None: None
    lines_mod.Line2D = _FakeLine2D
    matplotlib_mod.use = lambda backend: None

    sys.modules["matplotlib"] = matplotlib_mod
    sys.modules["matplotlib.pyplot"] = pyplot_mod
    sys.modules["matplotlib.lines"] = lines_mod


_install_fake_matplotlib()

import post_training_analysis.embedding_space_analysis as embedding_space_analysis


class TestEmbeddingSpaceAnalysis(unittest.TestCase):
    def _write_model_dir(
        self,
        root: Path,
        *,
        session_conditioning: bool = False,
        write_multisubject_metadata: bool = True,
    ) -> Path:
        model_dir = root / ("gru_session_conditioned" if session_conditioning else "gru_standard")
        outputs_dir = model_dir / "outputs"
        checkpoints_dir = outputs_dir / "checkpoints"
        step_low_dir = checkpoints_dir / "step_100"
        step_high_dir = checkpoints_dir / "step_200"
        step_low_dir.mkdir(parents=True, exist_ok=True)
        step_high_dir.mkdir(parents=True, exist_ok=True)

        inputs_lines = [
            "data:",
            "  subject_ids: null",
            "  subject_start: 0",
            "  subject_end: 2",
            "  test_subject_start: 2",
            "  test_subject_end: 3",
            "  mature_only: true",
            "  curricula:",
            "  - Uncoupled Baiting",
            "  multisubject: true",
            "  ignore_policy: exclude",
            "model:",
            "  type: gru",
            "  architecture:",
            "    multisubject: true",
            "    hidden_size: 8",
            "    subject_embedding_size: 3",
            "    subject_embedding_init: zeros",
        ]
        if session_conditioning:
            inputs_lines.extend(
                [
                    "    session_encoding_type: scalar",
                    "    session_integration_type: direct",
                    "    session_delta_n_layers: 1",
                    "    session_delta_hidden_size: 2",
                ]
            )
        inputs_lines.extend(["seed: 7", ""])
        (model_dir / "inputs.yaml").write_text("\n".join(inputs_lines))

        architecture = {
            "multisubject": True,
            "hidden_size": 8,
            "subject_embedding_size": 3,
            "subject_embedding_init": "zeros",
        }
        if session_conditioning:
            architecture.update(
                {
                    "session_encoding_type": "scalar",
                    "session_integration_type": "direct",
                    "session_delta_n_layers": 1,
                    "session_delta_hidden_size": 2,
                }
            )
        (outputs_dir / "gru_config.json").write_text(
            json.dumps({"architecture": architecture, "output_size": 2}, indent=2)
        )

        low_params = self._make_params(session_conditioning=session_conditioning, offset=0.0)
        high_params = self._make_params(session_conditioning=session_conditioning, offset=1.0)
        (step_low_dir / "params.json").write_text(json.dumps(low_params, indent=2))
        (step_high_dir / "params.json").write_text(json.dumps(high_params, indent=2))
        (outputs_dir / "params.json").write_text(json.dumps(low_params, indent=2))
        (checkpoints_dir / "index.json").write_text(
            json.dumps(
                {
                    "checkpoints": [
                        {
                            "step": 100,
                            "eval_likelihood": 0.1,
                            "params_path": "outputs/checkpoints/step_100/params.json",
                        },
                        {
                            "step": 200,
                            "eval_likelihood": 0.9,
                            "params_path": "outputs/checkpoints/step_200/params.json",
                        },
                    ]
                },
                indent=2,
            )
        )

        subject_index_map = {
            "subject_id_to_index": {"m1": 0, "m2": 1},
            "index_to_subject_id": {"0": "m1", "1": "m2"},
        }
        session_context = {
            "indexing": "1_based",
            "per_subject": [
                {
                    "subject_id": "m1",
                    "subject_index": 0,
                    "ordered_session_ids": [
                        "m1__m1_2024-01-02_00-00-00",
                        "m1__m1_2024-01-01_00-00-00",
                    ],
                    "ordered_source_session_ids": [
                        "m1_2024-01-02_00-00-00",
                        "m1_2024-01-01_00-00-00",
                    ],
                },
                {
                    "subject_id": "m2",
                    "subject_index": 1,
                    "ordered_session_ids": [
                        "m2__m2_2024-01-03_00-00-00",
                        "m2__m2_2024-01-04_00-00-00",
                    ],
                    "ordered_source_session_ids": [
                        "m2_2024-01-03_00-00-00",
                        "m2_2024-01-04_00-00-00",
                    ],
                },
            ],
        }
        (outputs_dir / "subject_index_map.json").write_text(
            json.dumps(subject_index_map, indent=2)
        )
        (outputs_dir / "session_context_map.json").write_text(
            json.dumps(session_context, indent=2)
        )
        if write_multisubject_metadata:
            (outputs_dir / "multisubject_metadata.json").write_text(
                json.dumps(
                    {
                        "subject_id_to_index": subject_index_map["subject_id_to_index"],
                        "index_to_subject_id": subject_index_map["index_to_subject_id"],
                        "num_subjects": 2,
                        "session_max_index_by_subject_index": [2, 2],
                        "session_context": session_context,
                        "train_session_ids": [
                            "m1__m1_2024-01-02_00-00-00",
                            "m2__m2_2024-01-03_00-00-00",
                        ],
                        "eval_session_ids": [
                            "m1__m1_2024-01-01_00-00-00",
                            "m2__m2_2024-01-04_00-00-00",
                        ],
                    },
                    indent=2,
                )
            )
        return model_dir

    def _make_params(
        self,
        *,
        session_conditioning: bool,
        offset: float,
    ) -> dict[str, object]:
        params: dict[str, object] = {
            "multisubject_gru": {
                "subject_embeddings": [
                    [1.0 + offset, 2.0 + offset, 3.0 + offset],
                    [4.0 + offset, 5.0 + offset, 6.0 + offset],
                ]
            }
        }
        if session_conditioning:
            params["session_delta_modules"] = {
                "session_delta_hidden": {
                    "w": [
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ],
                    "b": [0.0, 0.0],
                },
                "session_delta_out": {
                    "w": [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ],
                    "b": [0.0, 0.0, 0.0],
                },
            }
        return params

    def _make_han_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "subject_id": "m1",
                    "session_date": "2024-01-02",
                    "rig": "rig_a",
                    "trainer": "Bowen Tan",
                    "room": "room_alpha",
                    "curriculum_name": "task_a",
                    "weekday": 2,
                    "current_stage_actual": "STAGE_FINAL",
                    "foraging_eff_random_seed": 1.0,
                    "bias_naive": 0.1,
                    "reaction_time_median": 0.10,
                },
                {
                    "subject_id": "m1",
                    "session_date": "2024-01-01",
                    "rig": "rig_b",
                    "trainer": "bowen.tan",
                    "room": "room_alpha",
                    "curriculum_name": "task_b",
                    "weekday": 1,
                    "current_stage_actual": "GRADUATED",
                    "foraging_eff_random_seed": 3.0,
                    "bias_naive": 0.3,
                    "reaction_time_median": 0.30,
                },
                {
                    "subject_id": "m2",
                    "session_date": "2024-01-03",
                    "rig": "rig_c",
                    "trainer": "trainer_y",
                    "room": "room_beta",
                    "curriculum_name": "task_c",
                    "weekday": 3,
                    "current_stage_actual": "STAGE_3",
                    "foraging_eff_random_seed": 7.0,
                    "bias_naive": 0.7,
                    "reaction_time_median": 0.40,
                },
                {
                    "subject_id": "m2",
                    "session_date": "2024-01-03",
                    "rig": "rig_d",
                    "trainer": "trainer_z",
                    "room": "room_gamma",
                    "curriculum_name": "task_d",
                    "weekday": 3,
                    "current_stage_actual": "STAGE_3",
                    "foraging_eff_random_seed": 9.0,
                    "bias_naive": 0.9,
                    "reaction_time_median": 0.45,
                },
            ]
        )

    def test_run_embedding_space_analysis_uses_best_eval_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_model_dir(Path(tmpdir))
            with mock.patch.object(
                embedding_space_analysis,
                "_load_han_session_table",
                return_value=self._make_han_table(),
            ):
                summary = embedding_space_analysis.run_embedding_space_analysis(model_dir)
            self.assertEqual(summary["checkpoint_step"], 200)
            self.assertTrue(summary["params_path"].endswith("step_200/params.json"))
            self.assertTrue(Path(summary["summary_path"]).exists())

    def test_subject_session_metadata_preserves_context_order_and_join_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_model_dir(Path(tmpdir))
            with mock.patch.object(
                embedding_space_analysis,
                "_load_han_session_table",
                return_value=self._make_han_table(),
            ):
                summary = embedding_space_analysis.run_embedding_space_analysis(model_dir)
            session_df = pd.read_csv(summary["subject_session_metadata_path"])

        self.assertEqual(
            session_df["source_session_id"].tolist(),
            [
                "m1_2024-01-02_00-00-00",
                "m1_2024-01-01_00-00-00",
                "m2_2024-01-03_00-00-00",
                "m2_2024-01-04_00-00-00",
            ],
        )
        self.assertEqual(session_df["session_index"].tolist(), [1, 2, 1, 2])
        self.assertEqual(session_df["join_status"].tolist(), ["matched", "matched", "ambiguous", "missing"])
        self.assertEqual(session_df["session_split"].tolist(), ["train", "eval", "train", "eval"])

    def test_falls_back_to_session_context_when_multisubject_metadata_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_model_dir(
                Path(tmpdir),
                write_multisubject_metadata=False,
            )
            with mock.patch.object(
                embedding_space_analysis,
                "_load_han_session_table",
                return_value=self._make_han_table(),
            ):
                summary = embedding_space_analysis.run_embedding_space_analysis(model_dir)
            session_df = pd.read_csv(summary["subject_session_metadata_path"])

        self.assertEqual(summary["analysis_metadata_source_type"], "session_context_map_fallback")
        self.assertTrue(summary["analysis_metadata_path"].endswith("session_context_map.json"))
        self.assertEqual(session_df["session_split"].tolist(), ["train", "eval", "train", "eval"])

    def test_subject_metadata_aggregates_majority_and_means(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_model_dir(Path(tmpdir))
            with mock.patch.object(
                embedding_space_analysis,
                "_load_han_session_table",
                return_value=self._make_han_table(),
            ):
                summary = embedding_space_analysis.run_embedding_space_analysis(model_dir)
            subject_df = pd.read_csv(summary["subject_metadata_path"])

        row_m1 = subject_df[subject_df["subject_id"] == "m1"].iloc[0]
        self.assertEqual(row_m1["rig_majority"], "Mixed")
        self.assertEqual(row_m1["trainer_majority"], "Bowen Tan")
        self.assertEqual(row_m1["room_majority"], "room_alpha")
        self.assertEqual(row_m1["task_majority"], "Mixed")
        self.assertEqual(int(row_m1["n_sessions_matched_han"]), 2)
        self.assertAlmostEqual(float(row_m1["foraging_eff_random_seed_mean"]), 2.0)
        self.assertAlmostEqual(float(row_m1["bias_naive_mean"]), 0.2)
        self.assertAlmostEqual(float(row_m1["reaction_time_median_mean"]), 0.2)

        row_m2 = subject_df[subject_df["subject_id"] == "m2"].iloc[0]
        self.assertEqual(row_m2["rig_majority"], "Unknown")
        self.assertEqual(int(row_m2["n_sessions_missing_han"]), 1)
        self.assertEqual(int(row_m2["n_sessions_ambiguous_han"]), 1)

    def test_subject_embedding_plots_are_written_for_all_requested_colorings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_model_dir(Path(tmpdir))
            with mock.patch.object(
                embedding_space_analysis,
                "_load_han_session_table",
                return_value=self._make_han_table(),
            ):
                summary = embedding_space_analysis.run_embedding_space_analysis(model_dir)
            subject_plots = summary["plot_paths"]["subject_embeddings"]
            self.assertEqual(len(subject_plots), 7)
            for path in subject_plots.values():
                self.assertTrue(Path(path).exists(), path)

    def test_session_conditioned_plots_are_written_for_all_requested_colorings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_model_dir(Path(tmpdir), session_conditioning=True)
            with mock.patch.object(
                embedding_space_analysis,
                "_load_han_session_table",
                return_value=self._make_han_table(),
            ):
                summary = embedding_space_analysis.run_embedding_space_analysis(model_dir)
            self.assertTrue(summary["session_conditioning_enabled"])
            session_plots = summary["plot_paths"]["session_context"]
            self.assertEqual(set(session_plots.keys()), {"subject_0_m1", "subject_1_m2"})
            for subject_plot_map in session_plots.values():
                self.assertEqual(len(subject_plot_map), 9)
                for path in subject_plot_map.values():
                    self.assertTrue(Path(path).exists(), path)

    def test_non_session_conditioned_runs_skip_session_context_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_model_dir(Path(tmpdir), session_conditioning=False)
            with mock.patch.object(
                embedding_space_analysis,
                "_load_han_session_table",
                return_value=self._make_han_table(),
            ):
                summary = embedding_space_analysis.run_embedding_space_analysis(model_dir)

        self.assertEqual(summary["plot_paths"]["session_context"], {})
        self.assertIn(
            "session_conditioning_enabled=false",
            " ".join(summary["skipped"]["session_context"]),
        )

    def test_missing_han_columns_raise_helpful_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = self._write_model_dir(Path(tmpdir))
            df_han = self._make_han_table().drop(columns=["weekday"])
            with mock.patch.object(
                embedding_space_analysis,
                "_load_han_session_table",
                return_value=df_han,
            ):
                with self.assertRaisesRegex(ValueError, "--weekday-column"):
                    embedding_space_analysis.run_embedding_space_analysis(model_dir)


if __name__ == "__main__":
    unittest.main()
