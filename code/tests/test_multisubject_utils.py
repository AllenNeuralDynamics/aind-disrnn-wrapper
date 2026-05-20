"""Unit tests for multisubject disRNN helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from models.session_conditioning import compute_session_curriculum_lambda
from utils.multisubject import (
    GRU_SUBJECT_MODULE_KEY,
    SUBJECT_MODULE_KEY,
    SUBJECT_TABLE_KEY,
    append_subjects_to_index_maps,
    build_subject_index_maps,
    compute_session_conditioned_context_dataframe,
    convert_local_params_to_upstream_multisubject,
    convert_upstream_params_to_local_multisubject,
    expand_saved_multisubject_checkpoint,
    extract_subject_embeddings_from_params,
    merge_datasets_with_subject_index,
    ordered_session_context_rows,
    ordered_session_ids_from_session_context,
    prepend_session_index_to_multisubject_dataset,
    resolve_session_context_plot_subject_indices,
    save_multisubject_metadata,
    save_session_context_map,
    session_indices_for_split,
    session_regularization_index_arrays_from_session_context,
)


class _DummyDataset:
    def __init__(
        self,
        xs,
        ys,
        *,
        y_type="categorical",
        n_classes=2,
        x_names=None,
        y_names=None,
        batch_size=None,
        batch_mode="random",
    ):
        self._xs = np.asarray(xs)
        self._ys = np.asarray(ys)
        self.y_type = y_type
        self.n_classes = n_classes
        self.x_names = list(x_names or ["x1", "x2"])
        self.y_names = list(y_names or ["y"])
        self.batch_size = batch_size
        self.batch_mode = batch_mode

    def get_all(self):
        return self._xs, self._ys


class TestMultisubjectUtils(unittest.TestCase):
    def test_compute_session_curriculum_lambda_obeys_pretrain_and_warmup_steps(self):
        lambdas = [
            compute_session_curriculum_lambda(
                current_step=step,
                session_n_pretrain_steps=3,
                session_n_warmup_steps=2,
            )
            for step in range(7)
        ]

        self.assertEqual(lambdas, [0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0])

    def test_build_subject_index_maps_preserves_order(self):
        ordered_subject_ids, subject_id_to_index, index_to_subject_id = (
            build_subject_index_maps([711041, 793446, 711041, 727456])
        )

        self.assertEqual(ordered_subject_ids, [711041, 793446, 727456])
        self.assertEqual(subject_id_to_index[711041], 0)
        self.assertEqual(subject_id_to_index[793446], 1)
        self.assertEqual(index_to_subject_id[2], 727456)

    def test_merge_datasets_with_subject_index_prepends_subject_feature(self):
        dataset_a = _DummyDataset(
            xs=np.array(
                [
                    [[0.0, 1.0]],
                    [[1.0, 0.0]],
                ]
            ),
            ys=np.array(
                [
                    [[0.0]],
                    [[1.0]],
                ]
            ),
        )
        dataset_b = _DummyDataset(
            xs=np.array(
                [
                    [[2.0, 3.0]],
                    [[3.0, 2.0]],
                ]
            ),
            ys=np.array(
                [
                    [[1.0]],
                    [[0.0]],
                ]
            ),
        )

        merged = merge_datasets_with_subject_index(
            [dataset_a, dataset_b],
            [0, 1],
            batch_size=None,
            batch_mode="random",
        )
        merged_xs, _ = merged.get_all()

        self.assertEqual(merged.x_names[0], "Subject ID")
        self.assertTrue(np.allclose(merged_xs[:, 0, 0], 0.0))
        self.assertTrue(np.allclose(merged_xs[:, 1, 0], 1.0))
        self.assertTrue(np.allclose(merged_xs[:, 0, 1:], dataset_a.get_all()[0][:, 0, :]))
        self.assertTrue(np.allclose(merged_xs[:, 1, 1:], dataset_b.get_all()[0][:, 0, :]))

    def test_merge_datasets_with_subject_and_session_indices(self):
        dataset = _DummyDataset(
            xs=np.array(
                [
                    [[0.0, 1.0], [1.0, 0.0]],
                    [[1.0, 0.0], [0.0, 1.0]],
                ]
            ),
            ys=np.array(
                [
                    [[0.0], [1.0]],
                    [[1.0], [0.0]],
                ]
            ),
        )

        merged = merge_datasets_with_subject_index(
            [dataset],
            [3],
            session_indices_per_dataset=[[1, 2]],
            batch_size=None,
            batch_mode="random",
        )
        merged_xs, _ = merged.get_all()

        self.assertEqual(merged.x_names[:2], ["Subject ID", "Session Index"])
        self.assertTrue(np.allclose(merged_xs[:, :, 0], 3.0))
        self.assertTrue(np.allclose(merged_xs[:, 0, 1], 1.0))
        self.assertTrue(np.allclose(merged_xs[:, 1, 1], 2.0))

    def test_prepend_session_index_to_multisubject_dataset_marks_padding(self):
        dataset = _DummyDataset(
            xs=np.array(
                [
                    [[0.0, 1.0, 0.0], [1.0, 2.0, 3.0]],
                    [[-1.0, -1.0, -1.0], [1.0, 4.0, 5.0]],
                ]
            ),
            ys=np.zeros((2, 2, 1)),
            x_names=["Subject ID", "x1", "x2"],
        )

        conditioned = prepend_session_index_to_multisubject_dataset(
            dataset,
            session_indices=[1, 2],
        )
        conditioned_xs, _ = conditioned.get_all()

        self.assertEqual(conditioned.x_names[:2], ["Subject ID", "Session Index"])
        self.assertEqual(float(conditioned_xs[1, 0, 1]), -1.0)
        self.assertEqual(float(conditioned_xs[0, 1, 1]), 2.0)

    def test_session_indices_for_split_uses_full_history_order(self):
        session_context = {
            "indexing": "1_based",
            "per_subject": [
                {
                    "subject_id": "m1",
                    "subject_index": 0,
                    "ordered_session_ids": ["m1__s1", "m1__s2", "m1__s3"],
                    "ordered_source_session_ids": ["s1", "s2", "s3"],
                },
                {
                    "subject_id": "m2",
                    "subject_index": 1,
                    "ordered_session_ids": ["m2__s1", "m2__s2"],
                    "ordered_source_session_ids": ["s1", "s2"],
                },
            ],
        }
        metadata = {
            "session_context": session_context,
            "train_session_ids": ["m1__s1", "m1__s3", "m2__s2"],
            "eval_session_ids": ["m1__s2", "m2__s1"],
        }

        self.assertEqual(
            ordered_session_ids_from_session_context(session_context),
            ["m1__s1", "m1__s2", "m1__s3", "m2__s1", "m2__s2"],
        )
        self.assertEqual(session_indices_for_split(metadata, split_name="train"), [1, 3, 2])
        self.assertEqual(session_indices_for_split(metadata, split_name="eval"), [2, 1])

    def test_session_indices_for_split_prefers_explicit_full_session_ids(self):
        session_context = {
            "indexing": "1_based",
            "per_subject": [
                {
                    "subject_id": "train_subject",
                    "subject_index": 0,
                    "ordered_session_ids": ["train_subject__s1", "train_subject__s2"],
                    "ordered_source_session_ids": ["s1", "s2"],
                },
                {
                    "subject_id": "heldout_subject",
                    "subject_index": 1,
                    "ordered_session_ids": ["heldout_subject__s1", "heldout_subject__s2"],
                    "ordered_source_session_ids": ["s1", "s2"],
                },
            ],
        }
        metadata = {
            "session_context": session_context,
            "full_session_ids": ["heldout_subject__s1", "heldout_subject__s2"],
            "train_session_ids": ["heldout_subject__s1"],
            "eval_session_ids": ["heldout_subject__s2"],
        }

        self.assertEqual(session_indices_for_split(metadata, split_name="full"), [1, 2])

    def test_save_session_context_map_round_trips_json(self):
        with tempfile.TemporaryDirectory(prefix="session_context_map_") as tmpdir:
            path = Path(tmpdir) / "session_context_map.json"
            payload = {
                "indexing": "1_based",
                "per_subject": [
                    {
                        "subject_id": "m1",
                        "subject_index": 0,
                        "ordered_session_ids": ["m1__s1"],
                        "ordered_source_session_ids": ["s1"],
                    }
                ],
            }

            saved_path = save_session_context_map(path, session_context=payload)
            loaded_payload = json.loads(saved_path.read_text())

            self.assertEqual(loaded_payload, payload)

    def test_save_multisubject_metadata_normalizes_omegaconf_containers(self):
        with tempfile.TemporaryDirectory(prefix="multisubject_metadata_") as tmpdir:
            path = Path(tmpdir) / "multisubject_metadata.json"
            payload = {
                "curricula": OmegaConf.create(["coupled", "uncoupled"]),
                "features": OmegaConf.create({"include": ["reward", "choice"]}),
            }

            saved_path = save_multisubject_metadata(path, metadata=payload)
            loaded_payload = json.loads(saved_path.read_text())

            self.assertEqual(
                loaded_payload,
                {
                    "curricula": ["coupled", "uncoupled"],
                    "features": {"include": ["reward", "choice"]},
                },
            )

    def test_ordered_session_context_rows_sorts_by_subject_index(self):
        session_context = {
            "indexing": "1_based",
            "per_subject": [
                {
                    "subject_id": "m2",
                    "subject_index": 2,
                    "ordered_session_ids": ["m2__s1"],
                    "ordered_source_session_ids": ["s1"],
                },
                {
                    "subject_id": "m0",
                    "subject_index": 0,
                    "ordered_session_ids": ["m0__s1"],
                    "ordered_source_session_ids": ["s1"],
                },
            ],
        }

        ordered_rows = ordered_session_context_rows(session_context)

        self.assertEqual([row["subject_index"] for row in ordered_rows], [0, 2])
        self.assertEqual([row["subject_id"] for row in ordered_rows], ["m0", "m2"])

    def test_resolve_session_context_plot_subject_indices_honors_requested_order(self):
        session_context = {
            "indexing": "1_based",
            "per_subject": [
                {"subject_id": "m0", "subject_index": 0, "ordered_session_ids": ["a"]},
                {"subject_id": "m1", "subject_index": 1, "ordered_session_ids": ["b"]},
                {"subject_id": "m2", "subject_index": 2, "ordered_session_ids": ["c"]},
            ],
        }

        resolved = resolve_session_context_plot_subject_indices(
            session_context,
            requested_subject_indices=[2, 0, 2],
            max_subjects=2,
        )

        self.assertEqual(resolved, [2, 0])

    def test_resolve_session_context_plot_subject_indices_can_sample_random_subset(self):
        session_context = {
            "indexing": "1_based",
            "per_subject": [
                {"subject_id": "m0", "subject_index": 0, "ordered_session_ids": ["a"]},
                {"subject_id": "m1", "subject_index": 1, "ordered_session_ids": ["b"]},
                {"subject_id": "m2", "subject_index": 2, "ordered_session_ids": ["c"]},
                {"subject_id": "m3", "subject_index": 3, "ordered_session_ids": ["d"]},
            ],
        }

        resolved = resolve_session_context_plot_subject_indices(
            session_context,
            max_subjects=2,
            random_seed=43,
        )

        self.assertEqual(resolved, [2, 1])

    def test_session_regularization_index_arrays_cover_all_sessions(self):
        session_context = {
            "indexing": "1_based",
            "per_subject": [
                {
                    "subject_id": "m1",
                    "subject_index": 1,
                    "ordered_session_ids": ["m1__s1", "m1__s2"],
                },
                {
                    "subject_id": "m0",
                    "subject_index": 0,
                    "ordered_session_ids": ["m0__s1"],
                },
            ],
        }

        subject_indices, session_indices = (
            session_regularization_index_arrays_from_session_context(session_context)
        )

        self.assertTrue(np.array_equal(subject_indices, np.array([0, 1, 1], dtype=np.int32)))
        self.assertTrue(np.array_equal(session_indices, np.array([1, 1, 2], dtype=np.int32)))

    def test_compute_session_conditioned_context_dataframe_reconstructs_direct_scalar(self):
        params = {
            GRU_SUBJECT_MODULE_KEY: {
                SUBJECT_TABLE_KEY: np.array(
                    [
                        [0.5, -0.5],
                        [1.0, 0.25],
                    ]
                )
            },
            "multisubject_gru/~/session_delta_hidden": {
                "w": np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 2.0, 0.0, 0.0],
                    ]
                ),
                "b": np.zeros(4),
            },
            "multisubject_gru/~/session_delta_out": {
                "w": np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.0, 0.0],
                        [0.0, 0.0],
                    ]
                ),
                "b": np.zeros(2),
            },
        }
        session_context = {
            "indexing": "1_based",
            "per_subject": [
                {
                    "subject_id": "m0",
                    "subject_index": 0,
                    "ordered_session_ids": ["m0__s1", "m0__s2"],
                    "ordered_source_session_ids": ["s1", "s2"],
                },
                {
                    "subject_id": "m1",
                    "subject_index": 1,
                    "ordered_session_ids": ["m1__s1"],
                    "ordered_source_session_ids": ["s1"],
                },
            ],
        }

        plot_df = compute_session_conditioned_context_dataframe(
            params,
            session_context=session_context,
            session_encoding_type="scalar",
            session_integration_type="direct",
            session_fourier_k=4,
            session_delta_n_layers=1,
            session_delta_hidden_size=4,
            session_max_index_by_subject_index=[2, 1],
            train_session_ids=["m0__s1"],
            eval_session_ids=["m0__s2"],
            selected_subject_indices=[0],
        )

        self.assertEqual(plot_df["subject_index"].tolist(), [0, 0])
        self.assertEqual(plot_df["session_split"].tolist(), ["train", "eval"])
        self.assertTrue(np.allclose(plot_df["embedding_1"].to_numpy(), [1.0, 1.5]))
        self.assertTrue(np.allclose(plot_df["embedding_2"].to_numpy(), [0.5, 1.5]))

    def test_compute_session_conditioned_context_dataframe_reconstructs_deep_delta_mlp(self):
        params = {
            GRU_SUBJECT_MODULE_KEY: {
                SUBJECT_TABLE_KEY: np.array([[0.5, -0.5]])
            },
            "multisubject_gru/~/session_delta_hidden": {
                "w": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [2.0, 0.0, 0.0],
                    ]
                ),
                "b": np.zeros(3),
            },
            "multisubject_gru/~/session_delta_hidden_1": {
                "w": np.eye(3),
                "b": np.zeros(3),
            },
            "multisubject_gru/~/session_delta_out": {
                "w": np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [0.0, 0.0],
                    ]
                ),
                "b": np.zeros(2),
            },
        }
        session_context = {
            "indexing": "1_based",
            "per_subject": [
                {
                    "subject_id": "m0",
                    "subject_index": 0,
                    "ordered_session_ids": ["m0__s1", "m0__s2"],
                    "ordered_source_session_ids": ["s1", "s2"],
                }
            ],
        }

        plot_df = compute_session_conditioned_context_dataframe(
            params,
            session_context=session_context,
            session_encoding_type="scalar",
            session_integration_type="direct",
            session_fourier_k=4,
            session_delta_n_layers=2,
            session_delta_hidden_size=3,
            session_max_index_by_subject_index=[2],
            selected_subject_indices=[0],
        )

        self.assertTrue(np.allclose(plot_df["embedding_1"].to_numpy(), [1.5, 2.5]))
        self.assertTrue(np.allclose(plot_df["embedding_2"].to_numpy(), [-0.5, -0.5]))

    def test_compute_session_conditioned_context_dataframe_applies_curriculum_lambda(self):
        params = {
            GRU_SUBJECT_MODULE_KEY: {
                SUBJECT_TABLE_KEY: np.array([[1.0, 0.0]])
            },
            "multisubject_gru/~/session_delta_hidden": {
                "w": np.array(
                    [
                        [0.0, 0.0],
                        [0.0, 0.0],
                        [2.0, 0.0],
                    ]
                ),
                "b": np.zeros(2),
            },
            "multisubject_gru/~/session_delta_out": {
                "w": np.array(
                    [
                        [1.0, 0.0],
                        [0.0, 0.0],
                    ]
                ),
                "b": np.zeros(2),
            },
        }
        session_context = {
            "indexing": "1_based",
            "per_subject": [
                {
                    "subject_id": "m0",
                    "subject_index": 0,
                    "ordered_session_ids": ["m0__s1", "m0__s2"],
                    "ordered_source_session_ids": ["s1", "s2"],
                }
            ],
        }

        plot_df = compute_session_conditioned_context_dataframe(
            params,
            session_context=session_context,
            session_encoding_type="scalar",
            session_integration_type="direct",
            session_fourier_k=4,
            session_delta_n_layers=1,
            session_delta_hidden_size=2,
            session_curriculum_lambda=0.5,
            session_max_index_by_subject_index=[2],
            selected_subject_indices=[0],
        )

        self.assertTrue(np.allclose(plot_df["embedding_1"].to_numpy(), [1.5, 2.0]))
        self.assertTrue(np.allclose(plot_df["embedding_2"].to_numpy(), [0.0, 0.0]))

    def test_local_upstream_conversion_preserves_effective_embeddings(self):
        params_local = {
            SUBJECT_MODULE_KEY: {
                SUBJECT_TABLE_KEY: np.array(
                    [
                        [0.25, -0.12],
                        [1.55, 0.28],
                        [-0.65, 0.78],
                    ]
                )
            }
        }

        params_upstream = convert_local_params_to_upstream_multisubject(params_local)
        recovered_local = convert_upstream_params_to_local_multisubject(params_upstream)

        self.assertTrue(
            np.allclose(
                extract_subject_embeddings_from_params(params_upstream),
                extract_subject_embeddings_from_params(params_local),
            )
        )
        self.assertTrue(
            np.allclose(
                extract_subject_embeddings_from_params(recovered_local),
                extract_subject_embeddings_from_params(params_local),
            )
        )

    def test_extract_subject_embeddings_from_gru_params(self):
        params_gru = {
            GRU_SUBJECT_MODULE_KEY: {
                SUBJECT_TABLE_KEY: np.array(
                    [
                        [0.1, -0.2, 0.3],
                        [1.0, 0.5, -0.1],
                    ]
                )
            }
        }

        extracted = extract_subject_embeddings_from_params(params_gru)
        self.assertEqual(extracted.shape, (2, 3))
        self.assertTrue(np.allclose(extracted, params_gru[GRU_SUBJECT_MODULE_KEY][SUBJECT_TABLE_KEY]))

    def test_append_subjects_to_index_maps_appends_new_rows_only(self):
        subject_id_to_index = {"711041": 0, "793446": 1}
        index_to_subject_id = {0: "711041", 1: "793446"}

        updated_id_to_index, updated_index_to_id = append_subjects_to_index_maps(
            subject_id_to_index,
            index_to_subject_id,
            ["793446", "727456"],
        )

        self.assertEqual(updated_id_to_index["711041"], 0)
        self.assertEqual(updated_id_to_index["793446"], 1)
        self.assertEqual(updated_id_to_index["727456"], 2)
        self.assertEqual(updated_index_to_id[2], "727456")

    def test_expand_saved_multisubject_checkpoint_writes_updated_artifacts(self):
        with tempfile.TemporaryDirectory(prefix="multisubject_utils_") as tmpdir:
            root = Path(tmpdir)
            params_path = root / "params.json"
            subject_map_path = root / "subject_index_map.json"

            params = {
                SUBJECT_MODULE_KEY: {
                    SUBJECT_TABLE_KEY: np.array(
                        [
                            [1.0, 2.0],
                            [3.0, 4.0],
                        ]
                    ).tolist()
                }
            }
            with params_path.open("w") as f:
                json.dump(params, f, indent=2)
            with subject_map_path.open("w") as f:
                json.dump(
                    {
                        "subject_id_to_index": {"711041": 0, "793446": 1},
                        "index_to_subject_id": {"0": "711041", "1": "793446"},
                    },
                    f,
                    indent=2,
                )

            artifacts = expand_saved_multisubject_checkpoint(
                params_path=params_path,
                subject_index_map_path=subject_map_path,
                new_subject_ids=["727456"],
                output_dir=root / "expanded",
                init="mean",
            )

            expanded_params = json.loads(Path(artifacts["params_path"]).read_text())
            expanded_embeddings = np.asarray(
                expanded_params[SUBJECT_MODULE_KEY][SUBJECT_TABLE_KEY],
                dtype=float,
            )
            saved_df = pd.read_pickle(artifacts["subject_embeddings_path"])
            saved_map = json.loads(Path(artifacts["subject_index_map_path"]).read_text())

            self.assertEqual(expanded_embeddings.shape, (3, 2))
            self.assertTrue(np.allclose(expanded_embeddings[2], np.array([2.0, 3.0])))
            self.assertEqual(saved_map["subject_id_to_index"]["727456"], 2)
            self.assertEqual(saved_df["subject_id"].tolist(), ["711041", "793446", "727456"])


if __name__ == "__main__":
    unittest.main()
