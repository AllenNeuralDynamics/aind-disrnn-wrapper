"""Canonical file loader for external binary two-arm bandit datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Literal, Mapping

import numpy as np
import pandas as pd

from base.interfaces import DatasetLoader
from base.types import DatasetBundle
logger = logging.getLogger(__name__)

CANONICAL_REQUIRED_COLUMNS = (
    "subject_id",
    "ses_idx",
    "trial",
    "animal_response",
    "rewarded",
    "earned_reward",
)
SESSION_SPLIT_MANIFEST_SCHEMA_VERSION = 1
TRIAL_SPLIT_MANIFEST_SCHEMA_VERSION = 2
SUPPORTED_SPLIT_MANIFEST_SCHEMA_VERSIONS = {
    SESSION_SPLIT_MANIFEST_SCHEMA_VERSION,
    TRIAL_SPLIT_MANIFEST_SCHEMA_VERSION,
}


def _normalize_identifier(value):
    return value.item() if isinstance(value, np.generic) else value


def _read_trial_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported external-bandit file format: {path.suffix}. "
        "Supported formats are .pkl and .parquet."
    )


def _validate_binary_column(df: pd.DataFrame, column: str) -> None:
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any() or not values.isin([0, 1]).all():
        raise ValueError(f"Canonical column {column!r} must contain only binary 0/1 values.")


def validate_canonical_bandit_table(df: pd.DataFrame) -> None:
    """Validate the minimal trial-level schema consumed by the GRU/RL stack."""
    missing = [column for column in CANONICAL_REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Canonical external-bandit table is missing columns: {missing}.")
    if df.empty:
        raise ValueError("Canonical external-bandit table contains no trials.")
    if df[list(CANONICAL_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Canonical external-bandit columns cannot contain missing values.")
    _validate_binary_column(df, "animal_response")
    _validate_binary_column(df, "rewarded")
    _validate_binary_column(df, "earned_reward")
    rewarded = pd.to_numeric(df["rewarded"])
    earned_reward = pd.to_numeric(df["earned_reward"])
    if (rewarded != earned_reward).any():
        raise ValueError(
            "Canonical reward aliases 'rewarded' and 'earned_reward' must agree row-wise."
        )
    if df.duplicated(["subject_id", "ses_idx", "trial"]).any():
        raise ValueError(
            "Canonical external-bandit rows must be unique by "
            "(subject_id, ses_idx, trial)."
        )
    trial_values = pd.to_numeric(df["trial"], errors="coerce")
    if (
        trial_values.isna().any()
        or not np.equal(trial_values, np.floor(trial_values)).all()
        or (trial_values < 0).any()
    ):
        raise ValueError(
            "Canonical column 'trial' must contain non-negative integer-valued indices."
        )


def load_external_split_manifest(path: str | Path) -> dict[str, object]:
    """Load an explicit session split (v1) or within-session prefix split (v2)."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("External split manifest must be a JSON object.")
    schema_version = manifest.get("schema_version")
    if schema_version not in SUPPORTED_SPLIT_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError(
            "External split manifest schema_version must be one of "
            f"{sorted(SUPPORTED_SPLIT_MANIFEST_SCHEMA_VERSIONS)}."
        )
    subjects = manifest.get("subjects")
    if not isinstance(subjects, list) or not subjects:
        raise ValueError("External split manifest must contain a non-empty 'subjects' list.")

    split_by_subject: dict[str, dict[str, object]] = {}
    for row in subjects:
        if not isinstance(row, dict) or "subject_id" not in row:
            raise ValueError("Every external split manifest subject needs a subject_id.")
        subject_key = str(_normalize_identifier(row["subject_id"]))
        if subject_key in split_by_subject:
            raise ValueError(f"Duplicate subject_id={subject_key!r} in split manifest.")
        if schema_version == SESSION_SPLIT_MANIFEST_SCHEMA_VERSION:
            adapt = row.get("adapt_session_ids")
            test = row.get("test_session_ids")
            if (
                not isinstance(adapt, list)
                or not adapt
                or not isinstance(test, list)
                or not test
            ):
                raise ValueError(
                    f"Subject {subject_key!r} needs non-empty adapt_session_ids and "
                    "test_session_ids lists."
                )
            normalized_adapt = [str(_normalize_identifier(value)) for value in adapt]
            normalized_test = [str(_normalize_identifier(value)) for value in test]
            all_ids = normalized_adapt + normalized_test
            if len(all_ids) != len(set(all_ids)):
                raise ValueError(
                    f"Subject {subject_key!r} has duplicate or overlapping session IDs."
                )
            split_by_subject[subject_key] = {
                "adapt_session_ids": normalized_adapt,
                "test_session_ids": normalized_test,
            }
        else:
            session_id = row.get("session_id")
            adapt_prefix_trials = row.get("adapt_prefix_trials")
            total_trials = row.get("total_trials")
            if session_id is None:
                raise ValueError(f"Subject {subject_key!r} needs a session_id.")
            if (
                not isinstance(adapt_prefix_trials, int)
                or not isinstance(total_trials, int)
                or adapt_prefix_trials <= 0
                or adapt_prefix_trials >= total_trials
            ):
                raise ValueError(
                    f"Subject {subject_key!r} needs 0 < adapt_prefix_trials < total_trials."
                )
            split_by_subject[subject_key] = {
                "session_id": str(_normalize_identifier(session_id)),
                "adapt_prefix_trials": adapt_prefix_trials,
                "total_trials": total_trials,
            }
    manifest["split_by_subject"] = split_by_subject
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def _resolve_scalar_metadata(
    df: pd.DataFrame,
    manifest: Mapping[str, object],
    *,
    field: str,
    configured_value: str | None,
) -> str | None:
    candidates: list[str] = []
    if configured_value is not None:
        candidates.append(str(configured_value))
    manifest_value = manifest.get(field)
    if manifest_value is not None:
        candidates.append(str(manifest_value))
    if field in df.columns:
        table_values = [str(value) for value in df[field].dropna().unique().tolist()]
        if len(table_values) > 1:
            raise ValueError(f"Canonical external-bandit table has multiple {field} values.")
        candidates.extend(table_values)
    if len(set(candidates)) > 1:
        raise ValueError(
            f"Configured, manifest, and table {field} values must agree: {candidates}."
        )
    return candidates[0] if candidates else None


class ExternalBanditDatasetLoader(DatasetLoader):
    """Load a canonical external two-arm bandit table with an explicit split."""

    def __init__(
        self,
        file_path: str | Path,
        split_manifest_path: str | Path,
        dataset_id: str | None = None,
        species: str | None = None,
        subject_ids: Iterable[str | int] | None = None,
        features: Mapping[str, str] | None = None,
        batch_size: int | None = None,
        batch_mode: Literal["single", "rolling", "random"] = "random",
        seed: int | None = None,
        **extras: object,
    ) -> None:
        super().__init__(seed=seed)
        self.file_path = Path(file_path)
        self.split_manifest_path = Path(split_manifest_path)
        self.dataset_id = dataset_id
        self.species = species
        self.subject_ids = list(subject_ids) if subject_ids is not None else None
        self.features = dict(features) if features is not None else {}
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.extras = extras

    def load(self) -> DatasetBundle:
        from data_loaders.mice import _build_multisubject_bundle

        if self.seed is not None:
            np.random.seed(self.seed)

        df = _read_trial_table(self.file_path)
        validate_canonical_bandit_table(df)
        manifest = load_external_split_manifest(self.split_manifest_path)
        split_by_subject = dict(manifest["split_by_subject"])
        schema_version = int(manifest["schema_version"])

        raw_subject_ids = df["subject_id"].unique().tolist()
        normalized_subject_keys = [
            str(_normalize_identifier(value)) for value in raw_subject_ids
        ]
        if len(set(normalized_subject_keys)) != len(normalized_subject_keys):
            raise ValueError(
                "Distinct table subject_id values collide after string normalization."
            )
        available_subject_keys = set(normalized_subject_keys)
        subject_id_by_key = {
            str(_normalize_identifier(value)): _normalize_identifier(value)
            for value in raw_subject_ids
        }
        manifest_subject_key_order = list(split_by_subject)
        manifest_subject_keys = set(manifest_subject_key_order)
        if self.subject_ids is None:
            selected_subject_key_order = manifest_subject_key_order
            selected_subject_keys = set(selected_subject_key_order)
            if manifest_subject_keys != available_subject_keys:
                raise ValueError(
                    "Without subject_ids filtering, the split manifest subjects must exactly "
                    "match the trial table; "
                    f"table_only={sorted(available_subject_keys - manifest_subject_keys)}, "
                    f"manifest_only={sorted(manifest_subject_keys - available_subject_keys)}."
                )
        else:
            selected_subject_key_order = [
                str(_normalize_identifier(value)) for value in self.subject_ids
            ]
            selected_subject_keys = set(selected_subject_key_order)
            if len(selected_subject_keys) != len(selected_subject_key_order):
                raise ValueError("subject_ids cannot contain duplicates.")
            missing_from_table = selected_subject_keys - available_subject_keys
            missing_from_manifest = selected_subject_keys - manifest_subject_keys
            if missing_from_table or missing_from_manifest:
                raise ValueError(
                    "Selected external-bandit subjects must exist in both table and manifest; "
                    f"missing_from_table={sorted(missing_from_table)}, "
                    f"missing_from_manifest={sorted(missing_from_manifest)}."
                )
            df = df[
                df["subject_id"].map(lambda value: str(_normalize_identifier(value))).isin(
                    selected_subject_keys
                )
            ].copy()
            split_by_subject = {
                key: value
                for key, value in split_by_subject.items()
                if key in selected_subject_keys
            }
        resolved_subject_ids = [
            subject_id_by_key[key]
            for key in selected_subject_key_order
        ]

        resolved_dataset_id = _resolve_scalar_metadata(
            df, manifest, field="dataset_id", configured_value=self.dataset_id
        )
        resolved_species = _resolve_scalar_metadata(
            df, manifest, field="species", configured_value=self.species
        )
        metadata = {
            "source": "external_bandit_file",
            "file_path": str(self.file_path),
            "split_manifest_path": str(self.split_manifest_path),
            "split_manifest_schema_version": schema_version,
            "split_strategy": "explicit_manifest",
            "dataset_id": resolved_dataset_id,
            "species": resolved_species,
            "features": self.features,
        }
        if schema_version == SESSION_SPLIT_MANIFEST_SCHEMA_VERSION:
            metadata.update(
                {
                    "adapt_session_ids_by_subject": {
                        key: value["adapt_session_ids"]
                        for key, value in split_by_subject.items()
                    },
                    "test_session_ids_by_subject": {
                        key: value["test_session_ids"]
                        for key, value in split_by_subject.items()
                    },
                }
            )
            session_split_by_subject = split_by_subject
            trial_split_by_subject = None
        else:
            metadata.update(
                {
                    "trial_split_by_subject": split_by_subject,
                    "split_strategy": "within_session_prefix_suffix",
                    "trial_partition_column": "external_split_partition",
                    "adapt_trial_partition": "adapt",
                    "test_trial_partition": "test",
                }
            )
            session_split_by_subject = None
            trial_split_by_subject = split_by_subject
        metadata.update(self.extras)
        logger.info(
            "Loading canonical external bandit dataset_id=%s species=%s with %d subjects",
            resolved_dataset_id,
            resolved_species,
            len(selected_subject_keys),
        )
        return _build_multisubject_bundle(
            df=df,
            resolved_subject_ids=resolved_subject_ids,
            ignore_policy="exclude",
            features=self.features,
            eval_every_n=2,
            batch_size=self.batch_size,
            batch_mode=self.batch_mode,
            metadata=metadata,
            session_split_by_subject=session_split_by_subject,
            trial_split_by_subject=trial_split_by_subject,
        )
