from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from data_loaders.external_bandit import (
    ExternalBanditDatasetLoader,
    load_external_split_manifest,
    validate_canonical_bandit_table,
)


def _optional_dependency_stubs():
    modules = {}
    if importlib.util.find_spec("disentangled_rnns") is None:
        package = types.ModuleType("disentangled_rnns")
        library = types.ModuleType("disentangled_rnns.library")
        rnn_utils = types.ModuleType("disentangled_rnns.library.rnn_utils")

        class DatasetRNN:
            def __init__(
                self,
                xs,
                ys,
                *,
                y_type,
                n_classes,
                x_names,
                y_names,
                batch_size,
                batch_mode,
                rng=None,
            ):
                self._xs = np.asarray(xs)
                self._ys = np.asarray(ys)
                self.y_type = y_type
                self.n_classes = n_classes
                self.x_names = list(x_names)
                self.y_names = list(y_names)
                self.batch_size = batch_size
                self.batch_mode = batch_mode
                self.rng = rng

            def get_all(self):
                return {"xs": self._xs, "ys": self._ys}

        rnn_utils.DatasetRNN = DatasetRNN
        library.rnn_utils = rnn_utils
        package.library = library
        modules.update(
            {
                "disentangled_rnns": package,
                "disentangled_rnns.library": library,
                "disentangled_rnns.library.rnn_utils": rnn_utils,
            }
        )
    if importlib.util.find_spec("omegaconf") is None:
        omegaconf = types.ModuleType("omegaconf")
        omegaconf.DictConfig = type("DictConfig", (), {})
        omegaconf.ListConfig = type("ListConfig", (), {})
        omegaconf.OmegaConf = type("OmegaConf", (), {})
        modules["omegaconf"] = omegaconf
    return mock.patch.dict(sys.modules, modules)


def _canonical_trials() -> pd.DataFrame:
    rows = []
    for subject_id in ("rat-a", "rat-b"):
        for session_id in ("s1", "s2", "s3", "s4"):
            for trial in range(4):
                rows.append(
                    {
                        "dataset_id": "demo-rats",
                        "species": "rat",
                        "subject_id": subject_id,
                        "ses_idx": session_id,
                        "trial": trial,
                        "animal_response": trial % 2,
                        "earned_reward": (trial + 1) % 2,
                    }
                )
    return pd.DataFrame(rows)


def _manifest() -> dict:
    return {
        "schema_version": 1,
        "dataset_id": "demo-rats",
        "species": "rat",
        "subjects": [
            {
                "subject_id": subject_id,
                "adapt_session_ids": ["s1", "s3"],
                "test_session_ids": ["s2", "s4"],
            }
            for subject_id in ("rat-a", "rat-b")
        ],
    }


class TestCanonicalBanditValidation(unittest.TestCase):
    def test_accepts_binary_two_arm_table(self):
        validate_canonical_bandit_table(_canonical_trials())

    def test_rejects_nonbinary_choice(self):
        df = _canonical_trials()
        df.loc[0, "animal_response"] = 2
        with self.assertRaisesRegex(ValueError, "binary 0/1"):
            validate_canonical_bandit_table(df)

    def test_rejects_duplicate_trial_key(self):
        df = pd.concat([_canonical_trials(), _canonical_trials().iloc[[0]]])
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_canonical_bandit_table(df)


class TestExternalSplitManifest(unittest.TestCase):
    def test_rejects_unknown_schema_version(self):
        manifest = _manifest()
        manifest["schema_version"] = 2
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "split.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_external_split_manifest(path)

    def test_rejects_adapt_test_overlap(self):
        manifest = _manifest()
        manifest["subjects"][0]["test_session_ids"] = ["s1"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "split.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlapping"):
                load_external_split_manifest(path)


class TestExternalBanditDatasetLoader(unittest.TestCase):
    def test_rejects_conflicting_species_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            table_path = temp_path / "trials.pkl"
            manifest_path = temp_path / "split.json"
            _canonical_trials().to_pickle(table_path)
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
            loader = ExternalBanditDatasetLoader(
                file_path=table_path,
                split_manifest_path=manifest_path,
                species="mouse",
                batch_size=None,
                batch_mode="single",
            )
            with _optional_dependency_stubs():
                with self.assertRaisesRegex(ValueError, "species values must agree"):
                    loader.load()

    def test_explicit_manifest_controls_train_and_eval_sessions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            table_path = temp_path / "trials.pkl"
            manifest_path = temp_path / "split.json"
            _canonical_trials().to_pickle(table_path)
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

            with _optional_dependency_stubs():
                bundle = ExternalBanditDatasetLoader(
                    file_path=table_path,
                    split_manifest_path=manifest_path,
                    batch_size=None,
                    batch_mode="single",
                    seed=7,
                ).load()

        self.assertEqual(bundle.metadata["dataset_id"], "demo-rats")
        self.assertEqual(bundle.metadata["species"], "rat")
        self.assertEqual(bundle.metadata["split_strategy"], "explicit_manifest")
        self.assertEqual(bundle.train_set.get_all()["xs"].shape[1], 4)
        self.assertEqual(bundle.eval_set.get_all()["xs"].shape[1], 4)
        self.assertEqual(
            bundle.metadata["adapt_session_ids_by_subject"],
            {"rat-a": ["s1", "s3"], "rat-b": ["s1", "s3"]},
        )
        self.assertEqual(
            bundle.metadata["test_session_ids_by_subject"],
            {"rat-a": ["s2", "s4"], "rat-b": ["s2", "s4"]},
        )
        self.assertEqual(
            bundle.metadata["train_session_ids"],
            ["rat-a__s1", "rat-a__s3", "rat-b__s1", "rat-b__s3"],
        )
        self.assertEqual(
            bundle.metadata["eval_session_ids"],
            ["rat-a__s2", "rat-a__s4", "rat-b__s2", "rat-b__s4"],
        )

    def test_rejects_manifest_that_does_not_cover_all_sessions(self):
        manifest = _manifest()
        manifest["subjects"][0]["test_session_ids"] = ["s2"]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            table_path = temp_path / "trials.pkl"
            manifest_path = temp_path / "split.json"
            _canonical_trials().to_pickle(table_path)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            loader = ExternalBanditDatasetLoader(
                file_path=table_path,
                split_manifest_path=manifest_path,
                batch_size=None,
                batch_mode="single",
            )
            with _optional_dependency_stubs():
                with self.assertRaisesRegex(ValueError, "cover every retained session"):
                    loader.load()


if __name__ == "__main__":
    unittest.main()
