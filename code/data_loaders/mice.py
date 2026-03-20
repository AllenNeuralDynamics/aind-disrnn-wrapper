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
from utils.multisubject import (
    build_subject_index_maps,
    compute_train_eval_session_ids,
    merge_datasets_with_subject_index,
    normalize_subject_id,
    subject_sort_key,
    unique_subject_ids_preserve_order,
)

logger = logging.getLogger(__name__)


def _align_raw_df_with_valid_sessions(
    df: pd.DataFrame,
    *,
    ignore_policy: str,
) -> pd.DataFrame:
    """Drop sessions that would be silently removed by create_disrnn_dataset."""
    aligned = df.copy()
    if ignore_policy == "exclude" and "animal_response" in aligned.columns:
        valid_sessions = aligned.loc[aligned["animal_response"] != 2, "ses_idx"].unique()
        aligned = aligned[aligned["ses_idx"].isin(valid_sessions)].copy()
    return aligned


def _resolve_multisubject_subject_order(
    df: pd.DataFrame,
    resolved_subject_ids: Iterable[int] | Iterable[str] | None = None,
) -> list:
    """Resolve a deterministic subject order for multisubject training."""
    available_subject_ids = unique_subject_ids_preserve_order(df["subject_id"].tolist())
    if resolved_subject_ids is None:
        return sorted(available_subject_ids, key=subject_sort_key)

    requested_subject_ids = unique_subject_ids_preserve_order(resolved_subject_ids)
    available_subject_ids_set = set(available_subject_ids)
    ordered_subject_ids = [
        normalize_subject_id(subject_id)
        for subject_id in requested_subject_ids
        if normalize_subject_id(subject_id) in available_subject_ids_set
    ]
    missing_subject_ids = [
        normalize_subject_id(subject_id)
        for subject_id in requested_subject_ids
        if normalize_subject_id(subject_id) not in available_subject_ids_set
    ]
    if missing_subject_ids:
        logger.warning(
            "Requested subject_ids were not found after filtering and will be skipped: %s",
            missing_subject_ids,
        )
    return ordered_subject_ids


def _make_multisubject_session_ids_unique(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure session ids are unique across subjects in merged datasets."""
    if "subject_id" not in df.columns:
        raise ValueError(
            "Multisubject loading requires a 'subject_id' column in the raw dataframe."
        )
    if "ses_idx" not in df.columns:
        raise ValueError(
            "Multisubject loading requires a 'ses_idx' column in the raw dataframe."
        )

    normalized_df = df.copy()
    normalized_df["subject_id"] = normalized_df["subject_id"].map(normalize_subject_id)
    if "source_ses_idx" not in normalized_df.columns:
        normalized_df["source_ses_idx"] = normalized_df["ses_idx"]
    normalized_df["ses_idx"] = [
        f"{normalize_subject_id(subject_id)}__{normalize_subject_id(session_id)}"
        for subject_id, session_id in zip(
            normalized_df["subject_id"],
            normalized_df["source_ses_idx"],
        )
    ]
    return normalized_df


def _build_multisubject_bundle(
    *,
    df: pd.DataFrame,
    resolved_subject_ids: Iterable[int] | Iterable[str] | None,
    ignore_policy: str,
    features: Mapping[str, str] | None,
    eval_every_n: int,
    batch_size: int | None,
    batch_mode: Literal["single", "rolling", "random"],
    metadata: Mapping[str, object],
) -> DatasetBundle:
    """Build a merged multisubject dataset bundle from a trial dataframe."""
    if "subject_id" not in df.columns:
        raise ValueError(
            "Multisubject loading requires a 'subject_id' column in the raw dataframe."
        )

    prepared_df = _make_multisubject_session_ids_unique(df)
    subject_order = _resolve_multisubject_subject_order(prepared_df, resolved_subject_ids)
    if not subject_order:
        raise ValueError("No subject data found for multisubject loading.")

    prepared_subjects: list[dict[str, object]] = []
    for subject_id in subject_order:
        subject_df = prepared_df[prepared_df["subject_id"] == subject_id].copy()
        if subject_df.empty:
            continue
        sort_columns = ["ses_idx"]
        if "trial" in subject_df.columns:
            sort_columns.append("trial")
        subject_df = subject_df.sort_values(sort_columns).reset_index(drop=True)
        subject_df = _align_raw_df_with_valid_sessions(
            subject_df,
            ignore_policy=ignore_policy,
        )
        if subject_df.empty:
            logger.warning(
                "Skipping subject_id=%s because no valid sessions remain after filtering.",
                subject_id,
            )
            continue

        session_ids = list(dict.fromkeys(subject_df["ses_idx"].tolist()))
        train_session_ids, eval_session_ids = compute_train_eval_session_ids(
            session_ids,
            eval_every_n=eval_every_n,
        )
        prepared_subjects.append(
            {
                "subject_id": subject_id,
                "df": subject_df,
                "full_session_ids": session_ids,
                "train_session_ids": train_session_ids,
                "eval_session_ids": eval_session_ids,
            }
        )

    if not prepared_subjects:
        raise ValueError("No multisubject data remains after filtering valid sessions.")

    ordered_subject_ids, subject_id_to_index, index_to_subject_id = build_subject_index_maps(
        [item["subject_id"] for item in prepared_subjects]
    )

    subject_indices: list[int] = []
    full_datasets = []
    train_datasets = []
    eval_datasets = []
    ordered_frames: list[pd.DataFrame] = []
    full_session_ids: list = []
    train_session_ids: list = []
    eval_session_ids: list = []

    for item in prepared_subjects:
        subject_id = item["subject_id"]
        subject_index = subject_id_to_index[subject_id]
        subject_df = pd.DataFrame(item["df"]).copy()
        subject_df["subject_index"] = int(subject_index)

        dataset = dl.create_disrnn_dataset(
            subject_df,
            ignore_policy=ignore_policy,
            features=features or None,
            batch_size=batch_size,
            batch_mode=batch_mode,
        )
        xs_subject, _ = dataset.get_all()
        expected_n_sessions = len(item["full_session_ids"])
        if int(xs_subject.shape[1]) != int(expected_n_sessions):
            raise ValueError(
                "Multisubject dataset construction changed the number of sessions for "
                f"subject {subject_id}. Expected {expected_n_sessions}, got {xs_subject.shape[1]}."
            )

        dataset_train, dataset_eval = rnn_utils.split_dataset(
            dataset,
            eval_every_n=eval_every_n,
        )
        full_datasets.append(dataset)
        train_datasets.append(dataset_train)
        eval_datasets.append(dataset_eval)
        subject_indices.append(subject_index)
        ordered_frames.append(subject_df)
        full_session_ids.extend(item["full_session_ids"])
        train_session_ids.extend(item["train_session_ids"])
        eval_session_ids.extend(item["eval_session_ids"])

    merged_dataset = merge_datasets_with_subject_index(
        full_datasets,
        subject_indices,
        batch_size=batch_size,
        batch_mode=batch_mode,
    )
    merged_train_dataset = merge_datasets_with_subject_index(
        train_datasets,
        subject_indices,
        batch_size=batch_size,
        batch_mode=batch_mode,
    )
    merged_eval_dataset = merge_datasets_with_subject_index(
        eval_datasets,
        subject_indices,
        batch_size=batch_size,
        batch_mode=batch_mode,
    )

    raw_df = pd.concat(ordered_frames, ignore_index=True)
    raw_df["ses_idx"] = pd.Categorical(
        raw_df["ses_idx"],
        categories=full_session_ids,
        ordered=True,
    )
    sort_columns = ["ses_idx"]
    if "trial" in raw_df.columns:
        sort_columns.append("trial")
    raw_df = raw_df.sort_values(sort_columns).reset_index(drop=True)
    raw_df["ses_idx"] = raw_df["ses_idx"].astype(str)

    metadata_dict = dict(metadata)
    metadata_dict.update(
        {
            "subject_ids": ordered_subject_ids,
            "subject_id_to_index": subject_id_to_index,
            "index_to_subject_id": index_to_subject_id,
            "num_subjects": int(len(ordered_subject_ids)),
            "multisubject": True,
            "num_trials": int(len(raw_df)),
            "num_sessions": int(len(full_session_ids)),
            "train_session_ids": train_session_ids,
            "eval_session_ids": eval_session_ids,
            "batch_size": batch_size,
            "batch_mode": batch_mode,
        }
    )

    return DatasetBundle(
        raw=raw_df,
        train_set=merged_train_dataset,
        eval_set=merged_eval_dataset,
        metadata=metadata_dict,
        extras={"dataset": merged_dataset},
    )


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

        if self.multisubject:
            metadata = {
                "ignore_policy": self.ignore_policy,
                "features": self.features,
                "eval_every_n": self.eval_every_n,
            }
            metadata.update(self.extras)
            return _build_multisubject_bundle(
                df=df,
                resolved_subject_ids=self.subject_ids,
                ignore_policy=self.ignore_policy,
                features=self.features,
                eval_every_n=self.eval_every_n,
                batch_size=self.batch_size,
                batch_mode=self.batch_mode,
                metadata=metadata,
            )

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
        df = _align_raw_df_with_valid_sessions(df, ignore_policy=self.ignore_policy)
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

        if self.multisubject:
            metadata = {
                "file_path": str(self.file_path),
                "subject_ids": self.subject_ids,
                "stages": self.stages,
                "ignore_policy": self.ignore_policy,
                "features": self.features,
                "eval_every_n": self.eval_every_n,
            }
            metadata.update(self.extras)
            return _build_multisubject_bundle(
                df=df,
                resolved_subject_ids=self.subject_ids,
                ignore_policy=self.ignore_policy,
                features=self.features,
                eval_every_n=self.eval_every_n,
                batch_size=self.batch_size,
                batch_mode=self.batch_mode,
                metadata=metadata,
            )

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
        df = _align_raw_df_with_valid_sessions(df, ignore_policy=self.ignore_policy)

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

    Train/eval splitting is done by session order. In single-subject mode this
    follows ``rnn_utils.split_dataset`` directly; in multisubject mode the same
    every-``eval_every_n`` rule is applied independently within each subject
    before merging.

    Parameters
    ----------
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
        If ``True``, build one dataset per subject, split sessions within each
        subject, and merge them with ``subject_index`` prepended as feature 0.
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

        logger.info("Loading train dataset (mature_only=%s) …", self.mature_only)
        df, subject_ids = load_mice_snapshot(
            subject_ids=self.subject_ids,
            subject_start=self.subject_start,
            subject_end=self.subject_end,
            mature_only=self.mature_only,
            curricula=self.curricula,
            cols_to_retain=self.cols_to_retain,
        )

        if self.multisubject:
            metadata = {
                "subject_ids": subject_ids,
                "subject_start": self.subject_start,
                "subject_end": self.subject_end,
                "mature_only": self.mature_only,
                "curricula": self.curricula,
                "ignore_policy": self.ignore_policy,
                "features": self.features,
            }
            metadata.update(self.extras)
            return _build_multisubject_bundle(
                df=df,
                resolved_subject_ids=subject_ids,
                ignore_policy=self.ignore_policy,
                features=self.features,
                eval_every_n=self.eval_every_n,
                batch_size=self.batch_size,
                batch_mode=self.batch_mode,
                metadata=metadata,
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
        df = _align_raw_df_with_valid_sessions(df, ignore_policy=self.ignore_policy)

        metadata = {
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
            "multisubject": self.multisubject,
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
