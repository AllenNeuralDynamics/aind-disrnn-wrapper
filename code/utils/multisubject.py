"""Helpers for multisubject disRNN datasets, params, and exports."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

SUBJECT_MODULE_KEY = "multisubject_dis_rnn"
SUBJECT_TABLE_KEY = "subject_embeddings"
UPSTREAM_SUBJECT_LINEAR_KEY = f"{SUBJECT_MODULE_KEY}/subject_embedding_weights"


def _json_default(value: Any) -> Any:
    """Convert numpy-backed values into JSON-serializable Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def normalize_subject_id(value: Any) -> Any:
    """Convert numpy scalars to plain Python scalars for stable mappings."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def unique_subject_ids_preserve_order(subject_ids: Iterable[Any]) -> list[Any]:
    """Return unique subject ids while preserving the first occurrence order."""
    ordered: list[Any] = []
    seen: set[Any] = set()
    for subject_id in subject_ids:
        normalized = normalize_subject_id(subject_id)
        if normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def subject_sort_key(value: Any) -> tuple[int, Any]:
    """Return a deterministic sort key across numeric and string ids."""
    normalized = normalize_subject_id(value)
    if isinstance(normalized, bool):
        return (0, int(normalized))
    if isinstance(normalized, (int, np.integer)):
        return (0, int(normalized))
    if isinstance(normalized, float):
        return (0, float(normalized))
    return (1, str(normalized))


def build_subject_index_maps(
    subject_ids: Sequence[Any],
) -> tuple[list[Any], dict[Any, int], dict[int, Any]]:
    """Return ordered subject ids and both mapping directions."""
    ordered_subject_ids = unique_subject_ids_preserve_order(subject_ids)
    subject_id_to_index = {
        normalize_subject_id(subject_id): index
        for index, subject_id in enumerate(ordered_subject_ids)
    }
    index_to_subject_id = {
        index: normalize_subject_id(subject_id)
        for index, subject_id in enumerate(ordered_subject_ids)
    }
    return ordered_subject_ids, subject_id_to_index, index_to_subject_id


def compute_train_eval_session_ids(
    session_ids: Sequence[Any],
    eval_every_n: int,
) -> tuple[list[Any], list[Any]]:
    """Return train/eval session ids for a per-subject split."""
    if eval_every_n <= 0:
        raise ValueError("eval_every_n must be a positive integer.")
    ordered_session_ids = list(session_ids)
    if not ordered_session_ids:
        raise ValueError("Cannot split an empty session list.")

    eval_indices = list(range(eval_every_n - 1, len(ordered_session_ids), eval_every_n))
    eval_session_ids = [ordered_session_ids[index] for index in eval_indices]
    train_session_ids = [
        session_id
        for index, session_id in enumerate(ordered_session_ids)
        if index not in set(eval_indices)
    ]

    if not train_session_ids:
        raise ValueError(
            "Per-subject splitting produced no training sessions. "
            "Increase the number of sessions or adjust eval_every_n."
        )
    if not eval_session_ids:
        raise ValueError(
            "Per-subject splitting produced no eval sessions. "
            "Increase the number of sessions or reduce eval_every_n."
        )
    return train_session_ids, eval_session_ids


def merge_datasets_with_subject_index(
    dataset_list: Sequence[Any],
    subject_indices: Sequence[int],
    *,
    batch_size: int | None = None,
    batch_mode: str | None = None,
    subject_feature_name: str = "Subject ID",
) -> Any:
    """Merge per-subject datasets and prepend a dense subject index feature."""
    if not dataset_list:
        raise ValueError("dataset_list must contain at least one dataset.")
    if len(dataset_list) != len(subject_indices):
        raise ValueError(
            "dataset_list and subject_indices must have the same length."
        )

    first_dataset = dataset_list[0]
    xs0, ys0 = first_dataset.get_all()
    base_x_names = list(first_dataset.x_names)
    base_y_names = list(first_dataset.y_names)
    base_y_type = getattr(first_dataset, "y_type", "categorical")
    base_n_classes = getattr(first_dataset, "n_classes", None)
    merged_batch_mode = batch_mode if batch_mode is not None else first_dataset.batch_mode

    max_n_timesteps = max(dataset.get_all()[0].shape[0] for dataset in dataset_list)
    merged_x_chunks: list[np.ndarray] = []
    merged_y_chunks: list[np.ndarray] = []

    for dataset, subject_index in zip(dataset_list, subject_indices):
        xs, ys = dataset.get_all()
        if list(dataset.x_names) != base_x_names:
            raise ValueError("All datasets must share the same x_names.")
        if list(dataset.y_names) != base_y_names:
            raise ValueError("All datasets must share the same y_names.")
        if getattr(dataset, "y_type", "categorical") != base_y_type:
            raise ValueError("All datasets must share the same y_type.")
        if getattr(dataset, "n_classes", None) != base_n_classes:
            raise ValueError("All datasets must share the same n_classes.")

        n_timesteps, n_sessions, _ = xs.shape
        subject_feature = np.full(
            (n_timesteps, n_sessions, 1),
            fill_value=int(subject_index),
            dtype=xs.dtype,
        )
        xs_with_subject = np.concatenate((subject_feature, xs), axis=2)

        if n_timesteps < max_n_timesteps:
            pad_timesteps = max_n_timesteps - n_timesteps
            xs_with_subject = np.concatenate(
                (
                    xs_with_subject,
                    -1 * np.ones(
                        (pad_timesteps, n_sessions, xs_with_subject.shape[2]),
                        dtype=xs_with_subject.dtype,
                    ),
                ),
                axis=0,
            )
            ys = np.concatenate(
                (
                    ys,
                    -1 * np.ones((pad_timesteps, n_sessions, ys.shape[2]), dtype=ys.dtype),
                ),
                axis=0,
            )

        merged_x_chunks.append(xs_with_subject)
        merged_y_chunks.append(ys)

    merged_xs = np.concatenate(merged_x_chunks, axis=1)
    merged_ys = np.concatenate(merged_y_chunks, axis=1)

    dataset_class = first_dataset.__class__
    return dataset_class(
        merged_xs,
        merged_ys,
        y_type=base_y_type,
        n_classes=base_n_classes,
        x_names=[subject_feature_name] + base_x_names,
        y_names=base_y_names,
        batch_size=batch_size,
        batch_mode=merged_batch_mode,
    )


def effective_subject_embeddings_from_upstream_linear(
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    """Collapse an upstream one-hot linear layer into an embedding table."""
    weights = np.asarray(weights, dtype=float)
    bias = np.asarray(bias, dtype=float)
    return weights + bias.reshape(1, -1)


def upstream_linear_from_effective_subject_embeddings(
    subject_embeddings: np.ndarray,
    *,
    center_rows: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose an embedding table into upstream linear weights plus bias."""
    subject_embeddings = np.asarray(subject_embeddings, dtype=float)
    if center_rows:
        bias = np.mean(subject_embeddings, axis=0)
        weights = subject_embeddings - bias.reshape(1, -1)
    else:
        bias = np.zeros(subject_embeddings.shape[1], dtype=float)
        weights = subject_embeddings.copy()
    return weights, bias


def convert_local_params_to_upstream_multisubject(
    params: dict[str, Any],
    *,
    center_rows: bool = True,
) -> dict[str, Any]:
    """Return a copy of local params with an upstream-style subject projection."""
    converted = copy.deepcopy(params)
    module_params = converted.get(SUBJECT_MODULE_KEY, {})
    if SUBJECT_TABLE_KEY not in module_params:
        return converted

    subject_embeddings = np.asarray(module_params.pop(SUBJECT_TABLE_KEY))
    weights, bias = upstream_linear_from_effective_subject_embeddings(
        subject_embeddings,
        center_rows=center_rows,
    )
    converted[UPSTREAM_SUBJECT_LINEAR_KEY] = {"w": weights, "b": bias}
    return converted


def convert_upstream_params_to_local_multisubject(
    params: dict[str, Any],
) -> dict[str, Any]:
    """Return a copy of upstream params with a local explicit embedding table."""
    converted = copy.deepcopy(params)
    if UPSTREAM_SUBJECT_LINEAR_KEY not in converted:
        return converted

    linear_params = converted.pop(UPSTREAM_SUBJECT_LINEAR_KEY)
    weights = np.asarray(linear_params["w"])
    bias = np.asarray(linear_params["b"])
    converted.setdefault(SUBJECT_MODULE_KEY, {})
    converted[SUBJECT_MODULE_KEY][SUBJECT_TABLE_KEY] = (
        effective_subject_embeddings_from_upstream_linear(weights, bias)
    )
    return converted


def extract_subject_embeddings_from_params(params: dict[str, Any]) -> np.ndarray:
    """Return effective subject embeddings from local or upstream params."""
    if SUBJECT_MODULE_KEY in params and SUBJECT_TABLE_KEY in params[SUBJECT_MODULE_KEY]:
        return np.asarray(params[SUBJECT_MODULE_KEY][SUBJECT_TABLE_KEY], dtype=float)

    if UPSTREAM_SUBJECT_LINEAR_KEY in params:
        linear_params = params[UPSTREAM_SUBJECT_LINEAR_KEY]
        return effective_subject_embeddings_from_upstream_linear(
            linear_params["w"],
            linear_params["b"],
        )

    raise KeyError("Could not locate subject embeddings in params.")


def subject_embeddings_to_dataframe(
    index_to_subject_id: dict[int, Any] | dict[str, Any],
    subject_embeddings: np.ndarray,
) -> pd.DataFrame:
    """Convert a subject embedding table to an inspection-friendly dataframe."""
    subject_embeddings = np.asarray(subject_embeddings, dtype=float)
    rows: list[dict[str, Any]] = []
    for subject_index, embedding in enumerate(subject_embeddings):
        subject_id = index_to_subject_id.get(subject_index)
        if subject_id is None:
            subject_id = index_to_subject_id.get(str(subject_index))
        row = {
            "subject_index": int(subject_index),
            "subject_id": normalize_subject_id(subject_id),
        }
        for dimension, value in enumerate(embedding, start=1):
            row[f"embedding_{dimension}"] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def expand_subject_embeddings_array(
    subject_embeddings: np.ndarray,
    n_new_subjects: int = 1,
    *,
    init: str = "mean",
) -> np.ndarray:
    """Append new subject rows to an embedding table."""
    if n_new_subjects <= 0:
        raise ValueError("n_new_subjects must be positive.")

    subject_embeddings = np.asarray(subject_embeddings, dtype=float)
    if subject_embeddings.ndim != 2:
        raise ValueError(
            "subject_embeddings must be a 2D array of shape [n_subjects, embedding_dim]."
        )

    if init == "mean":
        template = np.mean(subject_embeddings, axis=0, keepdims=True)
    elif init == "zeros":
        template = np.zeros((1, subject_embeddings.shape[1]), dtype=float)
    else:
        raise ValueError(f"Unsupported init mode: {init}")

    new_rows = np.repeat(template, repeats=n_new_subjects, axis=0)
    return np.concatenate((subject_embeddings, new_rows), axis=0)


def expand_local_multisubject_params(
    params: dict[str, Any],
    n_new_subjects: int = 1,
    *,
    init: str = "mean",
) -> dict[str, Any]:
    """Append new subject rows to local multisubject params."""
    expanded = convert_upstream_params_to_local_multisubject(params)
    expanded.setdefault(SUBJECT_MODULE_KEY, {})
    if SUBJECT_TABLE_KEY not in expanded[SUBJECT_MODULE_KEY]:
        raise KeyError("Local multisubject params do not contain subject embeddings.")

    subject_embeddings = np.asarray(expanded[SUBJECT_MODULE_KEY][SUBJECT_TABLE_KEY], dtype=float)
    expanded[SUBJECT_MODULE_KEY][SUBJECT_TABLE_KEY] = expand_subject_embeddings_array(
        subject_embeddings,
        n_new_subjects=n_new_subjects,
        init=init,
    )
    return expanded


def append_subjects_to_index_maps(
    subject_id_to_index: dict[Any, int],
    index_to_subject_id: dict[int, Any],
    new_subject_ids: Sequence[Any],
) -> tuple[dict[Any, int], dict[int, Any]]:
    """Return updated mappings with appended new subject ids."""
    updated_subject_id_to_index = {
        normalize_subject_id(subject_id): int(index)
        for subject_id, index in subject_id_to_index.items()
    }
    updated_index_to_subject_id = {
        int(index): normalize_subject_id(subject_id)
        for index, subject_id in index_to_subject_id.items()
    }

    next_index = len(updated_subject_id_to_index)
    for subject_id in new_subject_ids:
        normalized = normalize_subject_id(subject_id)
        if normalized in updated_subject_id_to_index:
            continue
        updated_subject_id_to_index[normalized] = next_index
        updated_index_to_subject_id[next_index] = normalized
        next_index += 1

    return updated_subject_id_to_index, updated_index_to_subject_id


def save_subject_index_map(
    path: str | Path,
    *,
    subject_id_to_index: dict[Any, int],
    index_to_subject_id: dict[int, Any],
) -> Path:
    """Persist subject-id mappings as a JSON artifact."""
    payload = {
        "subject_id_to_index": {
            str(normalize_subject_id(subject_id)): int(index)
            for subject_id, index in subject_id_to_index.items()
        },
        "index_to_subject_id": {
            str(int(index)): normalize_subject_id(subject_id)
            for index, subject_id in index_to_subject_id.items()
        },
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return output_path


def load_subject_index_map(
    path: str | Path,
) -> tuple[dict[Any, int], dict[int, Any]]:
    """Load subject-id mappings from a JSON artifact."""
    with Path(path).open("r") as f:
        payload = json.load(f)

    index_to_subject_id = {
        int(index): normalize_subject_id(subject_id)
        for index, subject_id in payload.get("index_to_subject_id", {}).items()
    }
    if index_to_subject_id:
        subject_id_to_index = {
            normalize_subject_id(subject_id): int(index)
            for index, subject_id in index_to_subject_id.items()
        }
        return subject_id_to_index, index_to_subject_id

    subject_id_to_index = {
        normalize_subject_id(subject_id): int(index)
        for subject_id, index in payload.get("subject_id_to_index", {}).items()
    }
    index_to_subject_id = {
        int(index): normalize_subject_id(subject_id)
        for subject_id, index in subject_id_to_index.items()
    }
    return subject_id_to_index, index_to_subject_id


def expand_saved_multisubject_checkpoint(
    *,
    params_path: str | Path,
    subject_index_map_path: str | Path,
    new_subject_ids: Sequence[Any],
    output_dir: str | Path | None = None,
    init: str = "mean",
) -> dict[str, Path]:
    """Expand saved params plus mapping artifacts for new subjects."""
    with Path(params_path).open("r") as f:
        params = json.load(f)

    subject_id_to_index, index_to_subject_id = load_subject_index_map(
        subject_index_map_path
    )
    updated_subject_id_to_index, updated_index_to_subject_id = append_subjects_to_index_maps(
        subject_id_to_index,
        index_to_subject_id,
        new_subject_ids,
    )
    n_added_subjects = len(updated_subject_id_to_index) - len(subject_id_to_index)
    if n_added_subjects <= 0:
        raise ValueError("No new subject ids were added to the mapping.")

    expanded_params = expand_local_multisubject_params(
        params,
        n_new_subjects=n_added_subjects,
        init=init,
    )
    expanded_embeddings = extract_subject_embeddings_from_params(expanded_params)
    embeddings_df = subject_embeddings_to_dataframe(
        updated_index_to_subject_id,
        expanded_embeddings,
    )

    destination_dir = Path(output_dir) if output_dir is not None else Path(params_path).parent
    destination_dir.mkdir(parents=True, exist_ok=True)

    expanded_params_path = destination_dir / "params.json"
    with expanded_params_path.open("w") as f:
        json.dump(expanded_params, f, indent=2, default=_json_default)

    subject_map_path = save_subject_index_map(
        destination_dir / "subject_index_map.json",
        subject_id_to_index=updated_subject_id_to_index,
        index_to_subject_id=updated_index_to_subject_id,
    )

    embeddings_path = destination_dir / "subject_embeddings.pkl"
    embeddings_df.to_pickle(embeddings_path)

    return {
        "params_path": expanded_params_path,
        "subject_index_map_path": subject_map_path,
        "subject_embeddings_path": embeddings_path,
    }
