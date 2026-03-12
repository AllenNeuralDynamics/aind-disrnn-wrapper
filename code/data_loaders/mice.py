from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Literal, Mapping, Optional

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
        batch_size: int | None = None,
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
        self.batch_size = batch_size
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
            batch_size=self.batch_size,
            batch_mode=self.batch_mode,
        )
        dataset_train, dataset_eval = rnn_utils.split_dataset(
            dataset, eval_every_n=self.eval_every_n
        )
        # Align raw df with sessions that survived dataset construction.
        # create_disrnn_dataset silently drops sessions whose every trial is
        # ignored (animal_response==2) when ignore_policy=="exclude".
        if self.ignore_policy == "exclude" and "animal_response" in df.columns:
            valid_sessions = df[df["animal_response"] != 2]["ses_idx"].unique()
            df = df[df["ses_idx"].isin(valid_sessions)]
        metadata = {
            "subject_ids": self.subject_ids,
            "ignore_policy": self.ignore_policy,
            "features": self.features,
            "eval_every_n": self.eval_every_n,
            "num_trials": len(df),
            "num_sessions": int(df["ses_idx"].nunique()) if "ses_idx" in df else None,
            "multisubject": self.multisubject,
            "batch_size": self.batch_size,
            "batch_mode": self.batch_mode,
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


class MiceDatasetLoaderFromFile(DatasetLoader):
    """Load the foraging dataset from a pre-saved dataframe file (pkl or parquet)."""

    def __init__(
        self,
        file_path: str | Path,
        subject_ids: Iterable[int] | None = None,
        stages: Iterable[str] | None = None,
        ignore_policy: str = "none",
        features: Mapping[str, str] | None = None,
        eval_every_n: int = 10,
        multisubject: bool = False,
        batch_size: int | None = None,
        batch_mode: Literal["single", "rolling", "random"] = "random",
        seed: int | None = None,
        **extras: object,
    ) -> None:
        super().__init__(seed=seed)
        self.file_path = Path(file_path)
        self.subject_ids = list(subject_ids) if subject_ids is not None else None
        self.stages = list(stages) if stages is not None else None
        self.ignore_policy = ignore_policy
        self.features = dict(features) if features is not None else {}
        self.eval_every_n = eval_every_n
        self.multisubject = multisubject
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.extras = extras

    def load(self) -> DatasetBundle:
        
        # Fix numpy random seed (will affect batch_mode="random")
        if self.seed is not None:
            np.random.seed(self.seed)
        
        if self.multisubject:
            raise NotImplementedError("Multisubject loading is not yet supported.")

        # Load dataframe from file
        logger.info(f"Loading dataframe from {self.file_path}")
        if self.file_path.suffix == ".pkl":
            df = pd.read_pickle(self.file_path)
        elif self.file_path.suffix == ".parquet":
            df = pd.read_parquet(self.file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {self.file_path.suffix}. "
                "Supported formats are .pkl and .parquet"
            )
        
        logger.info(f"Loaded dataframe with {len(df)} trials")
        
        # Filter by subject_ids if provided
        if self.subject_ids is not None:
            if "subject_id" in df.columns:
                df = df[df["subject_id"].isin(self.subject_ids)]
                logger.info(
                    f"Filtered to {len(df)} trials for subject_ids: {self.subject_ids}"
                )
            else:
                logger.warning(
                    "subject_ids filter requested but 'subject_id' column not found in dataframe"
                )
        
        # Filter by stages if provided
        if self.stages is not None:
            if "stage" in df.columns:
                df = df[df["stage"].isin(self.stages)]
                logger.info(f"Filtered to {len(df)} trials for stages: {self.stages}")
            else:
                logger.warning(
                    "stages filter requested but 'stage' column not found in dataframe"
                )
        
        if len(df) == 0:
            raise Exception("No data found after filtering")

        # Create dataset
        dataset = dl.create_disrnn_dataset(
            df,
            ignore_policy=self.ignore_policy,
            features=self.features,
            batch_size=self.batch_size,
            batch_mode=self.batch_mode,
        )
        dataset_train, dataset_eval = rnn_utils.split_dataset(
            dataset, eval_every_n=self.eval_every_n
        )
        # Align raw df with sessions that survived dataset construction.
        if self.ignore_policy == "exclude" and "animal_response" in df.columns:
            valid_sessions = df[df["animal_response"] != 2]["ses_idx"].unique()
            df = df[df["ses_idx"].isin(valid_sessions)]

        metadata = {
            "file_path": str(self.file_path),
            "subject_ids": self.subject_ids,
            "stages": self.stages,
            "ignore_policy": self.ignore_policy,
            "features": self.features,
            "eval_every_n": self.eval_every_n,
            "num_trials": len(df),
            "num_sessions": int(df["ses_idx"].nunique()) if "ses_idx" in df else None,
            "multisubject": self.multisubject,
            "batch_size": self.batch_size,
            "batch_mode": self.batch_mode,
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


class MiceSnapshotDatasetLoader(DatasetLoader):
    """Load mice behavioral data from pre-saved snapshot pickle files.

    Subjects are selected in one of two ways:

    * **Direct selection** – pass an explicit list via ``subject_ids``.
    * **Rank-based slice** – subjects are ranked per curriculum by session
      count (or mature-session count) and a contiguous slice
      ``[subject_start, subject_end)`` is taken.  Either bound can be
      ``None`` (meaning "start from beginning" or "go to end").

    The two mechanisms are mutually exclusive.

    Train/eval splitting is then done by trial index via
    ``rnn_utils.split_dataset`` (every ``eval_every_n``-th trial → eval).

    Parameters
    ----------
    snapshot_paths:
        Paths to the pickle snapshot files to load.
    subject_ids:
        Explicit list of subject IDs to use. When provided,
        ``subject_start`` and ``subject_end`` are ignored.
    subject_start:
        Start index (inclusive, 0-based) of the ranked slice per curriculum.
    subject_end:
        End index (exclusive) of the ranked slice per curriculum.
    ignore_policy:
        Ignore-policy string passed to :func:`dl.create_disrnn_dataset`.
    features:
        Feature mapping passed to :func:`dl.create_disrnn_dataset`.
    eval_every_n:
        Every ``eval_every_n``-th trial goes to the eval split.
    multisubject:
        Not yet supported; raises ``NotImplementedError`` if ``True``.
    mature_only:
        If ``True`` *(default)*, restrict ranking and returned trials to
        sessions in stage ``STAGE_FINAL`` or ``GRADUATED``.
        If ``False``, use all sessions.
    curricula:
        Curricula to include. Defaults to the three standard foraging
        curricula (Uncoupled Without Baiting, Uncoupled Baiting, Coupled
        Baiting).
    cols_to_retain:
        Trial-level columns to keep. Falls back to the module default when
        ``None``.
    batch_size:
        Batch size for the disRNN dataset iterator.
    batch_mode:
        One of ``"single"``, ``"rolling"``, or ``"random"``.
    seed:
        Random seed forwarded to the base class and NumPy.
    """

    def __init__(
        self,
        snapshot_paths: List[str],
        subject_ids: Optional[List] = None,
        subject_start: Optional[int] = None,
        subject_end: Optional[int] = None,
        ignore_policy: str = "exclude",
        features: Optional[Mapping[str, str]] = None,
        eval_every_n: int = 2,
        multisubject: bool = False,
        mature_only: bool = True,
        curricula: Optional[List[str]] = None,
        cols_to_retain: Optional[List[str]] = None,
        batch_size: Optional[int] = None,
        batch_mode: Literal["single", "rolling", "random"] = "random",
        seed: Optional[int] = None,
        **extras: object,
    ) -> None:
        super().__init__(seed=seed)
        self.snapshot_paths = [str(p) for p in snapshot_paths]
        self.subject_ids = list(subject_ids) if subject_ids is not None else None
        self.subject_start = subject_start
        self.subject_end = subject_end
        self.mature_only = mature_only
        self.curricula = curricula
        self.cols_to_retain = cols_to_retain
        self.ignore_policy = ignore_policy
        self.features = dict(features) if features is not None else {}
        self.eval_every_n = eval_every_n
        self.multisubject = multisubject
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.extras = extras

    def load(self) -> DatasetBundle:
        from utils.load_mice_snapshot import load_mice_snapshot  # noqa: PLC0415

        if self.seed is not None:
            np.random.seed(self.seed)

        if self.multisubject:
            raise NotImplementedError("Multisubject loading is not yet supported.")

        logger.info("Loading train dataset (mature_only=%s) …", self.mature_only)
        df, subject_ids = load_mice_snapshot(
            snapshot_paths=self.snapshot_paths,
            subject_ids=self.subject_ids,
            subject_start=self.subject_start,
            subject_end=self.subject_end,
            mature_only=self.mature_only,
            curricula=self.curricula,
            cols_to_retain=self.cols_to_retain,
        )

        logger.info("Building disRNN datasets …")
        dataset = dl.create_disrnn_dataset(
            df,
            ignore_policy=self.ignore_policy,
            features=self.features or None,  # pass None to use library defaults when empty
            batch_size=self.batch_size,
            batch_mode=self.batch_mode,
        )
        dataset_train, dataset_eval = rnn_utils.split_dataset(
            dataset, eval_every_n=self.eval_every_n
        )
        # Align raw df with sessions that survived dataset construction.
        if self.ignore_policy == "exclude" and "animal_response" in df.columns:
            valid_sessions = df[df["animal_response"] != 2]["ses_idx"].unique()
            df = df[df["ses_idx"].isin(valid_sessions)]

        metadata = {
            "snapshot_paths": self.snapshot_paths,
            "subject_ids": subject_ids,
            "subject_start": self.subject_start,
            "subject_end": self.subject_end,
            "mature_only": self.mature_only,
            "curricula": self.curricula,
            "ignore_policy": self.ignore_policy,
            "features": self.features,
            "num_trials": len(df),
            "num_sessions": (
                int(df["ses_idx"].nunique()) if "ses_idx" in df.columns else None
            ),
            "batch_size": self.batch_size,
            "batch_mode": self.batch_mode,
        }
        metadata.update(self.extras)

        return DatasetBundle(
            raw=df,
            train_set=dataset_train,
            eval_set=dataset_eval,
            metadata=metadata,
            extras={"dataset": dataset},
        )