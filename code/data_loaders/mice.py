from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

import aind_dynamic_foraging_data_utils.code_ocean_utils as co
import aind_dynamic_foraging_multisession_analysis.multisession_load as ms_load
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
        batch_mode: Literal["single", "rolling", "random"] = "random",
        seed: int | None = None,
        **extras: object,
    ) -> None:
        super().__init__(seed=seed)
        self.subject_ids = list(subject_ids)
        self.ignore_policy = ignore_policy
        self.features = dict(features)
        self.eval_every_n = eval_every_n
        self.multisubject = multisubject
        self.batch_mode = batch_mode
        self.extras = extras

    def load(self) -> DatasetBundle:
        
        # Fix numpy random seed (will affect batch_mode="random")
        if self.seed is not None:
            np.random.seed(self.seed)
        
        if self.multisubject:
            raise NotImplementedError("Multisubject loading is not yet supported.")

        results = []
        for subject in self.subject_ids:
            logger.info("Querying docDB for {}".format(subject))
            subject_results = co.get_subject_assets(
                subject, modality=["behavior"], stage=["STAGE_FINAL", "GRADUATED"]
            )  # TODO, should we expose these filters?
            subject_results = subject_results.sort_values(
                by="session_name"
            ).reset_index(drop=True)
            results.append(subject_results)
        results = pd.concat(results)
        if len(results) == 0:
            raise Exception("No data found for subject ids")

        # TODO, filter sessions by performance

        logger.info("Getting s3 location")
        results = co.add_s3_location(results)

        logger.info("Loading NWBs")
        nwbs, df = ms_load.make_multisession_trials_df(results["s3_nwb_location"])

        dataset = dl.create_disrnn_dataset(
            df,
            ignore_policy=self.ignore_policy,
            features=self.features,
            batch_mode=self.batch_mode,
        )
        dataset_train, dataset_eval = rnn_utils.split_dataset(
            dataset, eval_every_n=self.eval_every_n
        )
        metadata = {
            "subject_ids": self.subject_ids,
            "ignore_policy": self.ignore_policy,
            "features": self.features,
            "eval_every_n": self.eval_every_n,
            "num_trials": len(df),
            "num_sessions": int(df["ses_idx"].nunique()) if "ses_idx" in df else None,
            "multisubject": self.multisubject,
        }
        metadata.update(self.extras)

        extras = {"dataset": dataset}
        return DatasetBundle(
            raw=df,
            train_set=dataset_train,
            eval_set=dataset_eval,
            metadata=metadata,
            extras=extras,
        )
