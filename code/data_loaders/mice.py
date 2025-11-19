from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping

import jax
import numpy as np
import pandas as pd

import aind_disrnn_utils.data_loader as dl
from disentangled_rnns.library import rnn_utils

from base.interfaces import DatasetLoader
from base.types import DatasetBundle

logger = logging.getLogger(__name__)


class MiceDatasetLoader(DatasetLoader):
    """Load the foraging dataset for mice experiments."""

    def __init__(
        self,
        subject_ids: Iterable[int],
        ignore_policy: str,
        features: Mapping[str, str],
        eval_every_n: int,
        multisubject: bool = False,
        data_root: str = "/data",
        seed: int | None = None,
        **extras: object,
    ) -> None:
        super().__init__(seed=seed)
        self.subject_ids = list(subject_ids)
        self.ignore_policy = ignore_policy
        self.features = dict(features)
        self.eval_every_n = eval_every_n
        self.multisubject = multisubject
        self.data_root = Path(data_root)
        self.extras = extras

    def load(self) -> DatasetBundle:
        if self.seed is None:
            raise ValueError("MiceDatasetLoader requires a deterministic seed.")

        if self.multisubject:
            raise NotImplementedError("Multisubject loading is not yet supported.")

        frames = []
        for subject in self.subject_ids:
            asset_path = (
                self.data_root / f"disrnn_dataset_{subject}" / "disrnn_dataset.csv"
            )
            if not asset_path.exists():
                raise FileNotFoundError(f"Missing dataset CSV: {asset_path}")
            logger.info("Loading dataset for subject %s from %s", subject, asset_path)
            frames.append(pd.read_csv(asset_path))

        df = pd.concat(frames, ignore_index=True)
        np.random.seed(self.seed)

        dataset = dl.create_disrnn_dataset(
            df, ignore_policy=self.ignore_policy, features=self.features
        )
        dataset_train, dataset_eval = rnn_utils.split_dataset(
            dataset, eval_every_n=self.eval_every_n
        )

        key = jax.random.PRNGKey(self.seed)
        k1, k2 = jax.random.split(key)
        metadata = {
            "subject_ids": self.subject_ids,
            "ignore_policy": self.ignore_policy,
            "features": self.features,
            "eval_every_n": self.eval_every_n,
            "num_trials": len(df),
            "num_sessions": int(df["ses_idx"].nunique()) if "ses_idx" in df else None,
            "multisubject": self.multisubject,
            "seed": self.seed,
        }
        metadata.update(self.extras)
        rng_keys = {"primary": key, "warmup": k1, "training": k2}

        extras = {"dataset": dataset}
        return DatasetBundle(
            raw=df,
            train=dataset_train,
            eval=dataset_eval,
            metadata=metadata,
            rng_keys=rng_keys,
            extras=extras,
        )
