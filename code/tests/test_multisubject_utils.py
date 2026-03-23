"""Unit tests for multisubject disRNN helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils.multisubject import (
    GRU_SUBJECT_MODULE_KEY,
    SUBJECT_MODULE_KEY,
    SUBJECT_TABLE_KEY,
    append_subjects_to_index_maps,
    build_subject_index_maps,
    convert_local_params_to_upstream_multisubject,
    convert_upstream_params_to_local_multisubject,
    expand_saved_multisubject_checkpoint,
    extract_subject_embeddings_from_params,
    merge_datasets_with_subject_index,
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
