"""Helpers for multisubject datasets, params, and exports."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig, ListConfig, OmegaConf

SUBJECT_MODULE_KEY = "multisubject_dis_rnn"
GRU_SUBJECT_MODULE_KEY = "multisubject_gru"
SUBJECT_TABLE_KEY = "subject_embeddings"
UPSTREAM_SUBJECT_LINEAR_KEY = f"{SUBJECT_MODULE_KEY}/subject_embedding_weights"


def _json_default(value: Any) -> Any:
    """Convert numpy-backed values into JSON-serializable Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _normalize_jsonable(value: Any) -> Any:
    """Recursively normalize metadata into JSON-friendly Python objects."""
    if isinstance(value, (DictConfig, ListConfig)):
        return _normalize_jsonable(OmegaConf.to_container(value, resolve=True))
    if isinstance(value, Mapping):
        return {str(key): _normalize_jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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
    session_indices_per_dataset: Sequence[Sequence[int] | np.ndarray] | None = None,
    batch_size: int | None = None,
    batch_mode: str | None = None,
    subject_feature_name: str = "Subject ID",
    session_feature_name: str = "Session Index",
) -> Any:
    """Merge per-subject datasets and prepend subject/session index features."""
    if not dataset_list:
        raise ValueError("dataset_list must contain at least one dataset.")
    if len(dataset_list) != len(subject_indices):
        raise ValueError(
            "dataset_list and subject_indices must have the same length."
        )
    if session_indices_per_dataset is not None and len(session_indices_per_dataset) != len(
        dataset_list
    ):
        raise ValueError(
            "session_indices_per_dataset must be None or have the same length as dataset_list."
        )

    first_dataset = dataset_list[0]
    _all = first_dataset.get_all()
    xs0, ys0 = _all["xs"], _all["ys"]
    base_x_names = list(first_dataset.x_names)
    base_y_names = list(first_dataset.y_names)
    base_y_type = getattr(first_dataset, "y_type", "categorical")
    base_n_classes = getattr(first_dataset, "n_classes", None)
    merged_batch_mode = batch_mode if batch_mode is not None else first_dataset.batch_mode

    max_n_timesteps = max(dataset.get_all()["xs"].shape[0] for dataset in dataset_list)
    merged_x_chunks: list[np.ndarray] = []
    merged_y_chunks: list[np.ndarray] = []

    for dataset_index, (dataset, subject_index) in enumerate(
        zip(dataset_list, subject_indices)
    ):
        _d = dataset.get_all()
        xs, ys = _d["xs"], _d["ys"]
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
        feature_chunks = [subject_feature]

        if session_indices_per_dataset is not None:
            session_indices = np.asarray(session_indices_per_dataset[dataset_index], dtype=xs.dtype)
            if session_indices.ndim != 1:
                raise ValueError(
                    "Each session_indices_per_dataset entry must be 1D with one value per session."
                )
            if int(session_indices.shape[0]) != int(n_sessions):
                raise ValueError(
                    "session_indices_per_dataset entry length must match the dataset session "
                    f"count. Expected {n_sessions}, got {session_indices.shape[0]}."
                )
            session_feature = np.broadcast_to(
                session_indices.reshape(1, n_sessions, 1),
                (n_timesteps, n_sessions, 1),
            ).astype(xs.dtype, copy=False)
            feature_chunks.append(session_feature)

        feature_chunks.append(xs)
        xs_with_subject = np.concatenate(feature_chunks, axis=2)

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
    merged_x_names = [subject_feature_name]
    if session_indices_per_dataset is not None:
        merged_x_names.append(session_feature_name)
    merged_x_names.extend(base_x_names)
    return dataset_class(
        merged_xs,
        merged_ys,
        y_type=base_y_type,
        n_classes=base_n_classes,
        x_names=merged_x_names,
        y_names=base_y_names,
        batch_size=batch_size,
        batch_mode=merged_batch_mode,
    )


def save_session_context_map(
    path: str | Path,
    *,
    session_context: dict[str, Any],
) -> Path:
    """Persist per-subject session ordering as a JSON artifact."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(session_context, f, indent=2, default=_json_default)
    return output_path


def load_session_context_map(path: str | Path) -> dict[str, Any]:
    """Load a persisted session-context artifact."""
    with Path(path).open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping-style session context in {path}")
    return payload


def _ordered_session_context_rows(session_context: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(session_context, Mapping):
        raise ValueError("session_context must be a mapping.")
    rows = session_context.get("per_subject", [])
    if not isinstance(rows, list):
        raise ValueError("session_context.per_subject must be a list.")
    normalized_rows: list[dict[str, Any]] = []
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("Each session_context.per_subject entry must be a mapping.")
        normalized_rows.append(dict(raw_row))
    return normalized_rows


def ordered_session_context_rows(
    session_context: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return validated per-subject session-context rows sorted by subject index."""
    normalized_rows: list[dict[str, Any]] = []
    for row in _ordered_session_context_rows(session_context):
        if row.get("subject_index") is None:
            raise ValueError("Each session_context.per_subject entry must include subject_index.")
        normalized_row = dict(row)
        normalized_row["subject_index"] = int(row["subject_index"])
        normalized_row["subject_id"] = normalize_subject_id(row.get("subject_id"))
        normalized_rows.append(normalized_row)
    return sorted(normalized_rows, key=lambda row: int(row["subject_index"]))


def ordered_session_ids_from_session_context(
    session_context: Mapping[str, Any],
) -> list[str]:
    """Return merged full-session order from session-context metadata."""
    ordered_session_ids: list[str] = []
    for row in ordered_session_context_rows(session_context):
        ordered_session_ids.extend(
            [str(session_id) for session_id in row.get("ordered_session_ids") or []]
        )
    return ordered_session_ids


def merged_session_index_lookup_from_session_context(
    session_context: Mapping[str, Any],
) -> dict[str, int]:
    """Return merged-session-id -> 1-based per-subject session index."""
    lookup: dict[str, int] = {}
    for row in ordered_session_context_rows(session_context):
        ordered_session_ids = [str(session_id) for session_id in row.get("ordered_session_ids") or []]
        for session_index, session_id in enumerate(ordered_session_ids, start=1):
            lookup[session_id] = int(session_index)
    return lookup


def subject_session_index_lookup_from_session_context(
    session_context: Mapping[str, Any],
) -> tuple[dict[tuple[Any, str], int], dict[tuple[Any, str], int]]:
    """Return subject/session lookups for merged and source session identifiers."""
    merged_lookup: dict[tuple[Any, str], int] = {}
    source_lookup: dict[tuple[Any, str], int] = {}
    for row in ordered_session_context_rows(session_context):
        subject_id = normalize_subject_id(row.get("subject_id"))
        ordered_session_ids = [str(session_id) for session_id in row.get("ordered_session_ids") or []]
        ordered_source_session_ids = [
            str(session_id)
            for session_id in (
                row.get("ordered_source_session_ids") or row.get("ordered_session_ids") or []
            )
        ]
        if len(ordered_source_session_ids) != len(ordered_session_ids):
            raise ValueError(
                "session_context ordered_source_session_ids must align 1:1 with "
                "ordered_session_ids."
            )
        for session_index, (merged_session_id, source_session_id) in enumerate(
            zip(ordered_session_ids, ordered_source_session_ids),
            start=1,
        ):
            merged_lookup[(subject_id, merged_session_id)] = int(session_index)
            source_lookup[(subject_id, source_session_id)] = int(session_index)
    return merged_lookup, source_lookup


def resolve_session_context_plot_subject_indices(
    session_context: Mapping[str, Any],
    *,
    requested_subject_indices: Sequence[int] | int | None = None,
    max_subjects: int = 3,
    random_seed: int | None = None,
) -> list[int]:
    """Resolve up to ``max_subjects`` subject indices for session-context plots."""
    if int(max_subjects) < 0:
        raise ValueError("max_subjects must be >= 0.")

    ordered_rows = ordered_session_context_rows(session_context)
    available_subject_indices = [int(row["subject_index"]) for row in ordered_rows]
    available_subject_index_set = set(available_subject_indices)
    if int(max_subjects) == 0 or not available_subject_indices:
        return []

    if requested_subject_indices is None:
        if random_seed is None:
            return available_subject_indices[: int(max_subjects)]
        rng = np.random.default_rng(int(random_seed))
        sampled = rng.choice(
            np.asarray(available_subject_indices, dtype=int),
            size=min(int(max_subjects), len(available_subject_indices)),
            replace=False,
        )
        return [int(value) for value in np.asarray(sampled, dtype=int).tolist()]

    if isinstance(requested_subject_indices, (int, np.integer)):
        requested_values = [int(requested_subject_indices)]
    else:
        requested_values = [int(value) for value in requested_subject_indices]

    resolved: list[int] = []
    for subject_index in requested_values:
        if subject_index not in available_subject_index_set:
            raise ValueError(
                "Requested session-context plot subject_index is not present in the "
                f"resolved session context: {subject_index}."
            )
        if subject_index not in resolved:
            resolved.append(int(subject_index))
    return resolved[: int(max_subjects)]


def session_regularization_index_arrays_from_session_context(
    session_context: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return flat subject/session arrays covering every session in context order."""
    subject_indices: list[int] = []
    session_indices: list[int] = []
    for row in ordered_session_context_rows(session_context):
        subject_index = int(row["subject_index"])
        ordered_session_ids = row.get("ordered_session_ids") or []
        for session_index, _ in enumerate(ordered_session_ids, start=1):
            subject_indices.append(subject_index)
            session_indices.append(int(session_index))
    return (
        np.asarray(subject_indices, dtype=np.int32),
        np.asarray(session_indices, dtype=np.int32),
    )


def _find_linear_module_params(
    params: Mapping[str, Any],
    *,
    module_leaf_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    matches: list[tuple[str, Mapping[str, Any]]] = []

    def _visit(mapping: Mapping[str, Any], prefix: str = "") -> None:
        for key, value in mapping.items():
            full_key = f"{prefix}/{key}" if prefix else str(key)
            if not isinstance(value, Mapping):
                continue
            if "w" in value and "b" in value:
                leaf_name = full_key.split("/")[-1]
                if leaf_name == module_leaf_name:
                    matches.append((full_key, value))
                continue
            _visit(value, full_key)

    _visit(params)
    if not matches:
        raise KeyError(
            f"Could not locate linear params for module leaf name {module_leaf_name!r}."
        )
    if len(matches) > 1:
        match_names = [name for name, _ in matches]
        raise ValueError(
            "Expected exactly one linear module named "
            f"{module_leaf_name!r}, found {match_names}."
        )

    _, module_params = matches[0]
    return (
        np.asarray(module_params["w"], dtype=float),
        np.asarray(module_params["b"], dtype=float),
    )


def _apply_linear_layer(
    inputs: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    return np.asarray(inputs, dtype=float) @ np.asarray(weights, dtype=float) + np.asarray(
        bias,
        dtype=float,
    )


def _build_session_feature_array(
    *,
    subject_indices: np.ndarray,
    session_indices: np.ndarray,
    session_max_index_by_subject_index: Sequence[int],
    encoding_type: str,
    fourier_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    safe_subject_indices = np.where(subject_indices >= 0, subject_indices, 0).astype(int)
    session_max = np.asarray(session_max_index_by_subject_index, dtype=int)
    subject_session_max = session_max[safe_subject_indices]
    valid_session_mask = np.logical_and(subject_indices >= 0, session_indices >= 1)
    bounded_session_max = np.maximum(subject_session_max.astype(float), 1.0)
    phase = session_indices.astype(float) / bounded_session_max

    if encoding_type == "scalar":
        session_feat = phase[:, None]
    elif encoding_type == "fourier":
        ks = np.arange(1, int(fourier_k) + 1, dtype=float)
        angles = 2.0 * np.pi * phase[:, None] * ks[None, :]
        session_feat = np.concatenate((np.sin(angles), np.cos(angles)), axis=1)
    else:
        raise ValueError(f"Unsupported session_encoding_type={encoding_type!r}.")

    session_feat = session_feat * valid_session_mask[:, None].astype(float)
    return session_feat, valid_session_mask


def compute_session_conditioned_context_dataframe(
    params: Mapping[str, Any],
    *,
    session_context: Mapping[str, Any],
    session_encoding_type: str,
    session_integration_type: str,
    session_fourier_k: int,
    session_delta_n_layers: int,
    session_delta_hidden_size: int,
    session_curriculum_lambda: float = 1.0,
    session_max_index_by_subject_index: Sequence[int],
    train_session_ids: Sequence[Any] | None = None,
    eval_session_ids: Sequence[Any] | None = None,
    selected_subject_indices: Sequence[int] | int | None = None,
) -> pd.DataFrame:
    """Reconstruct subject+delta context embeddings for selected subjects across sessions."""
    encoding_type = str(session_encoding_type).strip().lower()
    integration_type = str(session_integration_type).strip().lower()
    if encoding_type == "none":
        raise ValueError(
            "compute_session_conditioned_context_dataframe requires session conditioning to "
            "be enabled."
        )
    if integration_type not in {"direct", "pre_mlp"}:
        raise ValueError(
            "session_integration_type must be 'direct' or 'pre_mlp', got "
            f"{session_integration_type!r}."
        )
    if int(session_delta_n_layers) <= 0:
        raise ValueError("session_delta_n_layers must be > 0.")
    if int(session_delta_hidden_size) <= 0:
        raise ValueError("session_delta_hidden_size must be > 0.")

    ordered_rows = ordered_session_context_rows(session_context)
    if selected_subject_indices is None:
        requested_subject_count = len(ordered_rows)
    elif isinstance(selected_subject_indices, (int, np.integer)):
        requested_subject_count = 1
    else:
        requested_subject_count = max(1, len(selected_subject_indices))
    resolved_subject_indices = resolve_session_context_plot_subject_indices(
        session_context,
        requested_subject_indices=selected_subject_indices,
        max_subjects=requested_subject_count,
    )
    if not resolved_subject_indices:
        return pd.DataFrame()

    subject_embeddings = extract_subject_embeddings_from_params(dict(params))
    session_max = np.asarray(session_max_index_by_subject_index, dtype=int)
    if session_max.ndim != 1:
        raise ValueError("session_max_index_by_subject_index must be 1D.")
    if subject_embeddings.shape[0] != session_max.shape[0]:
        raise ValueError(
            "Subject embedding table and session_max_index_by_subject_index must agree on "
            f"the number of subjects, got {subject_embeddings.shape[0]} and "
            f"{session_max.shape[0]}."
        )

    hidden_layers: list[tuple[np.ndarray, np.ndarray]] = []
    for layer_index in range(int(session_delta_n_layers)):
        layer_name = (
            "session_delta_hidden"
            if layer_index == 0
            else f"session_delta_hidden_{layer_index}"
        )
        hidden_layers.append(
            _find_linear_module_params(
                params,
                module_leaf_name=layer_name,
            )
        )
    out_weights, out_bias = _find_linear_module_params(
        params,
        module_leaf_name="session_delta_out",
    )
    pre_weights: np.ndarray | None = None
    pre_bias: np.ndarray | None = None
    if integration_type == "pre_mlp":
        pre_weights, pre_bias = _find_linear_module_params(
            params,
            module_leaf_name="session_pre_mlp",
        )

    selected_subject_index_set = {int(value) for value in resolved_subject_indices}
    rows_by_subject_index = {
        int(row["subject_index"]): row
        for row in ordered_rows
        if int(row["subject_index"]) in selected_subject_index_set
    }
    train_session_id_set = {str(session_id) for session_id in train_session_ids or []}
    eval_session_id_set = {str(session_id) for session_id in eval_session_ids or []}

    records: list[dict[str, Any]] = []
    for subject_index in resolved_subject_indices:
        subject_row = rows_by_subject_index.get(int(subject_index))
        if subject_row is None:
            raise ValueError(
                f"Resolved session context does not contain subject_index={subject_index}."
            )

        ordered_session_ids = [
            str(session_id)
            for session_id in (subject_row.get("ordered_session_ids") or [])
        ]
        ordered_source_session_ids = [
            str(session_id)
            for session_id in (
                subject_row.get("ordered_source_session_ids")
                or subject_row.get("ordered_session_ids")
                or []
            )
        ]
        if len(ordered_source_session_ids) != len(ordered_session_ids):
            raise ValueError(
                "session_context ordered_source_session_ids must align 1:1 with "
                "ordered_session_ids."
            )
        if not ordered_session_ids:
            continue

        subject_index_array = np.full(len(ordered_session_ids), int(subject_index), dtype=int)
        session_index_array = np.arange(1, len(ordered_session_ids) + 1, dtype=int)
        session_feat, valid_session_mask = _build_session_feature_array(
            subject_indices=subject_index_array,
            session_indices=session_index_array,
            session_max_index_by_subject_index=session_max,
            encoding_type=encoding_type,
            fourier_k=int(session_fourier_k),
        )

        subject_embedding = np.asarray(subject_embeddings[int(subject_index)], dtype=float)
        subject_context = np.repeat(
            subject_embedding.reshape(1, -1),
            repeats=len(ordered_session_ids),
            axis=0,
        )
        conditioned_session_feat = session_feat
        if integration_type == "pre_mlp":
            assert pre_weights is not None and pre_bias is not None
            conditioned_session_feat = np.maximum(
                _apply_linear_layer(session_feat, pre_weights, pre_bias),
                0.0,
            )

        delta_inputs = np.concatenate((subject_context, conditioned_session_feat), axis=1)
        hidden = delta_inputs
        for hidden_weights, hidden_bias in hidden_layers:
            hidden = np.maximum(_apply_linear_layer(hidden, hidden_weights, hidden_bias), 0.0)
        delta = _apply_linear_layer(hidden, out_weights, out_bias)
        delta = delta * valid_session_mask[:, None].astype(float)
        delta = float(session_curriculum_lambda) * delta
        subject_context = subject_context + delta

        subject_id = normalize_subject_id(subject_row.get("subject_id"))
        for position, (session_id, source_session_id) in enumerate(
            zip(ordered_session_ids, ordered_source_session_ids),
            start=1,
        ):
            if session_id in train_session_id_set:
                session_split = "train"
            elif session_id in eval_session_id_set:
                session_split = "eval"
            else:
                session_split = "full"

            record = {
                "subject_index": int(subject_index),
                "subject_id": subject_id,
                "session_id": str(session_id),
                "source_session_id": str(source_session_id),
                "session_split": session_split,
                "session_index": int(position),
                "subject_max_session_index": int(len(ordered_session_ids)),
                "session_phase": float(position / max(len(ordered_session_ids), 1)),
            }
            for dimension, value in enumerate(subject_context[position - 1], start=1):
                record[f"embedding_{dimension}"] = float(value)
            records.append(record)

    return pd.DataFrame(records)


def session_indices_for_split(
    metadata: Mapping[str, Any],
    *,
    split_name: str,
) -> list[int]:
    """Return 1-based per-subject session indices aligned to a merged dataset split."""
    session_context = metadata.get("session_context")
    if not isinstance(session_context, Mapping):
        raise ValueError(
            "session_indices_for_split requires metadata.session_context for multisubject runs."
        )
    session_id_to_index = merged_session_index_lookup_from_session_context(session_context)
    if split_name == "full":
        explicit_full_session_ids = metadata.get("full_session_ids")
        if explicit_full_session_ids is not None:
            ordered_session_ids = [str(session_id) for session_id in explicit_full_session_ids]
        else:
            ordered_session_ids = ordered_session_ids_from_session_context(session_context)
    elif split_name == "train":
        ordered_session_ids = [str(session_id) for session_id in metadata.get("train_session_ids") or []]
    elif split_name == "eval":
        ordered_session_ids = [str(session_id) for session_id in metadata.get("eval_session_ids") or []]
    else:
        raise ValueError(f"Unsupported split_name={split_name!r}.")

    missing_session_ids = [
        session_id for session_id in ordered_session_ids if str(session_id) not in session_id_to_index
    ]
    if missing_session_ids:
        raise ValueError(
            "Session-context metadata is missing session ids required for split "
            f"{split_name!r}: {missing_session_ids}"
        )
    return [int(session_id_to_index[str(session_id)]) for session_id in ordered_session_ids]


def prepend_session_index_to_multisubject_dataset(
    dataset: Any,
    *,
    session_indices: Sequence[int],
    session_feature_name: str = "Session Index",
) -> Any:
    """Insert a session-index feature after the prepended subject index feature."""
    _all = dataset.get_all()
    xs, ys = _all["xs"], _all["ys"]
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if xs.ndim != 3:
        raise ValueError(f"Expected dataset xs to be 3D, got shape={xs.shape}.")
    if xs.shape[2] < 1:
        raise ValueError("Multisubject datasets must include a prepended subject-index feature.")

    resolved_session_indices = np.asarray(session_indices, dtype=xs.dtype)
    if resolved_session_indices.ndim != 1:
        raise ValueError("session_indices must be a 1D sequence with one value per session.")
    if int(resolved_session_indices.shape[0]) != int(xs.shape[1]):
        raise ValueError(
            "session_indices length must match the dataset session count. "
            f"Expected {xs.shape[1]}, got {resolved_session_indices.shape[0]}."
        )

    session_feature = np.broadcast_to(
        resolved_session_indices.reshape(1, xs.shape[1], 1),
        (xs.shape[0], xs.shape[1], 1),
    ).astype(xs.dtype, copy=False)
    pad_mask = np.asarray(xs[..., 0] < 0)
    session_feature = np.array(session_feature, copy=True)
    session_feature[pad_mask] = -1

    dataset_class = dataset.__class__
    return dataset_class(
        np.concatenate((xs[:, :, :1], session_feature, xs[:, :, 1:]), axis=2),
        ys,
        y_type=getattr(dataset, "y_type", "categorical"),
        n_classes=getattr(dataset, "n_classes", None),
        x_names=[list(dataset.x_names)[0], session_feature_name, *list(dataset.x_names)[1:]],
        y_names=list(dataset.y_names),
        batch_size=getattr(dataset, "batch_size", None),
        batch_mode=getattr(dataset, "batch_mode", "random"),
    )


def prepend_session_index_to_multisubject_split_datasets(
    *,
    dataset: Any,
    dataset_train: Any,
    dataset_eval: Any,
    metadata: Mapping[str, Any],
    session_feature_name: str = "Session Index",
) -> tuple[Any, Any, Any]:
    """Return full/train/eval datasets with a prepended session-index feature."""
    return (
        prepend_session_index_to_multisubject_dataset(
            dataset,
            session_indices=session_indices_for_split(metadata, split_name="full"),
            session_feature_name=session_feature_name,
        ),
        prepend_session_index_to_multisubject_dataset(
            dataset_train,
            session_indices=session_indices_for_split(metadata, split_name="train"),
            session_feature_name=session_feature_name,
        ),
        prepend_session_index_to_multisubject_dataset(
            dataset_eval,
            session_indices=session_indices_for_split(metadata, split_name="eval"),
            session_feature_name=session_feature_name,
        ),
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


def resolve_subject_embedding_module_key(params: Mapping[str, Any]) -> str:
    """Return the local module key that owns the subject-embedding table."""
    if SUBJECT_MODULE_KEY in params and SUBJECT_TABLE_KEY in params[SUBJECT_MODULE_KEY]:
        return SUBJECT_MODULE_KEY
    if (
        GRU_SUBJECT_MODULE_KEY in params
        and SUBJECT_TABLE_KEY in params[GRU_SUBJECT_MODULE_KEY]
    ):
        return GRU_SUBJECT_MODULE_KEY
    raise KeyError("Could not locate local subject embeddings in params.")


def extract_subject_embeddings_from_params(params: dict[str, Any]) -> np.ndarray:
    """Return effective subject embeddings from local or upstream params."""
    try:
        subject_module_key = resolve_subject_embedding_module_key(params)
    except KeyError:
        subject_module_key = None
    if subject_module_key is not None:
        return np.asarray(params[subject_module_key][SUBJECT_TABLE_KEY], dtype=float)

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
    subject_module_key = resolve_subject_embedding_module_key(expanded)
    subject_embeddings = np.asarray(
        expanded[subject_module_key][SUBJECT_TABLE_KEY],
        dtype=float,
    )
    expanded[subject_module_key][SUBJECT_TABLE_KEY] = expand_subject_embeddings_array(
        subject_embeddings,
        n_new_subjects=n_new_subjects,
        init=init,
    )
    return expanded


def build_subject_embedding_update_mask(
    params: Mapping[str, Any],
    *,
    trainable_subject_indices: Sequence[int],
) -> dict[str, Any]:
    """Return a params-shaped mask that updates only selected subject rows."""

    def _zeros_like_tree(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: _zeros_like_tree(child) for key, child in value.items()}
        return np.zeros_like(np.asarray(value))

    subject_module_key = resolve_subject_embedding_module_key(params)
    subject_embeddings = np.asarray(params[subject_module_key][SUBJECT_TABLE_KEY], dtype=float)
    if subject_embeddings.ndim != 2:
        raise ValueError(
            "Subject embeddings must be a 2D array to build a row-level update mask."
        )

    mask = _zeros_like_tree(params)
    embedding_mask = np.zeros_like(subject_embeddings, dtype=float)
    for raw_index in trainable_subject_indices:
        subject_index = int(raw_index)
        if subject_index < 0 or subject_index >= subject_embeddings.shape[0]:
            raise ValueError(
                "trainable_subject_indices contains an out-of-range subject index: "
                f"{subject_index} for table with {subject_embeddings.shape[0]} rows."
            )
        embedding_mask[subject_index, :] = 1.0
    mask[subject_module_key][SUBJECT_TABLE_KEY] = embedding_mask
    return mask


def extend_session_context(
    session_context: Mapping[str, Any],
    *,
    appended_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return a copy of session_context with appended per-subject rows."""
    extended = copy.deepcopy(dict(session_context))
    existing_rows = list(extended.get("per_subject", []))
    extended["per_subject"] = existing_rows + [dict(row) for row in appended_rows]
    return extended


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


def save_multisubject_metadata(
    path: str | Path,
    *,
    metadata: Mapping[str, Any],
) -> Path:
    """Persist dataset metadata as a JSON artifact for later debugging/inspection."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _normalize_jsonable(metadata)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2)
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
