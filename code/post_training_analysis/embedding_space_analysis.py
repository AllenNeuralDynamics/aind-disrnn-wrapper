"""Standalone embedding-space analysis for trained multisubject runs."""

from __future__ import annotations

import importlib
import json
import logging
import math
import re
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from models.session_conditioning import resolve_session_conditioning_from_architecture
from post_training_analysis.generative_analysis import resolve_model_run
from utils.multisubject import (
    compute_session_conditioned_context_dataframe,
    extract_subject_embeddings_from_params,
    load_session_context_map,
    load_subject_index_map,
    normalize_subject_id,
    ordered_session_context_rows,
    subject_embeddings_to_dataframe,
)

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT_POLICY = "best_eval"
_DEFAULT_TASK_COLUMN = "curriculum_name"
_DEFAULT_WEEKDAY_COLUMN = "weekday"
_DEFAULT_FORAGING_EFF_COLUMN = "foraging_eff_random_seed"
_DEFAULT_BIAS_NAIVE_COLUMN = "bias_naive"
_DEFAULT_REACTION_TIME_COLUMN = "reaction_time_median"
_UNKNOWN_LABEL = "Unknown"
_MIXED_LABEL = "Mixed"

_SUBJECT_PLOT_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("rig_majority", "subject_embeddings_by_rig.png", "Rig", "categorical"),
    ("trainer_majority", "subject_embeddings_by_trainer.png", "Trainer", "categorical"),
    ("task_majority", "subject_embeddings_by_task.png", "Task", "categorical"),
    (
        "foraging_eff_random_seed_mean",
        "subject_embeddings_by_foraging_eff_random_seed.png",
        "Average Foraging Eff Random Seed",
        "numeric",
    ),
    ("bias_naive_mean", "subject_embeddings_by_bias_naive.png", "Average Bias Naive", "numeric"),
    (
        "reaction_time_median_mean",
        "subject_embeddings_by_reaction_time_median.png",
        "Average Reaction Time Median",
        "numeric",
    ),
)

_SESSION_PLOT_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("session_index", "session_context_by_session_index.png", "Session Index", "session_index"),
    ("rig", "session_context_by_rig.png", "Rig", "categorical"),
    ("trainer", "session_context_by_trainer.png", "Trainer", "categorical"),
    (
        "current_stage_actual",
        "session_context_by_current_stage_actual.png",
        "Current Stage Actual",
        "categorical",
    ),
    ("task", "session_context_by_task.png", "Task", "categorical"),
    ("weekday", "session_context_by_weekday.png", "Weekday", "categorical"),
    (
        "foraging_eff_random_seed",
        "session_context_by_foraging_eff_random_seed.png",
        "Foraging Eff Random Seed",
        "numeric",
    ),
    ("bias_naive", "session_context_by_bias_naive.png", "Bias Naive", "numeric"),
    (
        "reaction_time_median",
        "session_context_by_reaction_time_median.png",
        "Reaction Time Median",
        "numeric",
    ),
)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _to_serializable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _safe_slug(value: Any) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    safe = safe.strip("_")
    return safe or "value"


def _normalize_subject_id_str(value: Any) -> str | None:
    normalized = normalize_subject_id(value)
    if normalized is None:
        return None
    if isinstance(normalized, float) and math.isnan(normalized):
        return None
    return str(normalized)


def _normalize_session_date(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = re.search(r"\d{4}-\d{2}-\d{2}", text)
    if match is not None:
        return match.group(0)
    return text


def _extract_session_date(source_session_id: Any) -> str | None:
    return _normalize_session_date(source_session_id)


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping-style JSON payload at {path}.")
    return payload


def _default_output_dir(model_dir: str | Path, checkpoint_label: str) -> Path:
    return (
        Path(model_dir).expanduser().resolve()
        / "outputs"
        / "embedding_space_analysis"
        / str(checkpoint_label)
    )


def _resolved_model_architecture(resolved_run: Any) -> dict[str, Any]:
    model_config = dict(getattr(resolved_run, "model_config", {}) or {})
    architecture = model_config.get("architecture")
    if isinstance(architecture, Mapping):
        return dict(architecture)
    return model_config


def _load_han_session_table() -> pd.DataFrame:
    han_pipeline = importlib.import_module("aind_analysis_arch_result_access.han_pipeline")
    return han_pipeline.get_session_table(if_load_bpod=True)


def _validate_resolved_run(resolved_run: Any) -> None:
    if not bool(getattr(resolved_run, "multisubject", False)):
        raise ValueError("Embedding-space analysis only supports multisubject runs.")
    if str(getattr(resolved_run, "model_type", "")).strip().lower() not in {"gru", "disrnn"}:
        raise ValueError(
            "Embedding-space analysis only supports multisubject GRU/disRNN runs."
        )
    if str(getattr(resolved_run, "split", "")).strip().lower() != "train":
        raise ValueError("Embedding-space analysis only supports split='train' runs.")
    if not getattr(resolved_run, "subject_index_map_path", None):
        raise FileNotFoundError(
            "Embedding-space analysis requires outputs/subject_index_map.json."
        )


def _load_analysis_metadata(
    *,
    resolved_run: Any,
    subject_id_to_index: Mapping[Any, int],
    index_to_subject_id: Mapping[int, Any],
) -> tuple[dict[str, Any], Path]:
    outputs_dir = Path(resolved_run.outputs_dir)
    metadata_path = outputs_dir / "multisubject_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Embedding-space analysis requires {metadata_path}."
        )

    metadata = _load_json_file(metadata_path)
    session_context = metadata.get("session_context")
    if not isinstance(session_context, Mapping):
        if getattr(resolved_run, "session_context_map_path", None):
            session_context = load_session_context_map(resolved_run.session_context_map_path)
        else:
            raise FileNotFoundError(
                "Embedding-space analysis requires either metadata.session_context in "
                f"{metadata_path} or outputs/session_context_map.json."
            )

    resolved_index_to_subject_id = {
        int(index): normalize_subject_id(subject_id)
        for index, subject_id in dict(index_to_subject_id).items()
    }
    resolved_subject_id_to_index = {
        normalize_subject_id(subject_id): int(index)
        for subject_id, index in dict(subject_id_to_index).items()
    }
    session_max_index_by_subject_index = list(
        metadata.get("session_max_index_by_subject_index") or []
    )
    if not session_max_index_by_subject_index:
        ordered_rows = ordered_session_context_rows(session_context)
        session_max_index_by_subject_index = [
            int(len(row.get("ordered_session_ids") or []))
            for row in ordered_rows
        ]

    metadata["session_context"] = dict(session_context)
    metadata["subject_id_to_index"] = resolved_subject_id_to_index
    metadata["index_to_subject_id"] = resolved_index_to_subject_id
    metadata["subject_ids"] = [
        resolved_index_to_subject_id[index]
        for index in sorted(resolved_index_to_subject_id.keys())
    ]
    metadata["num_subjects"] = int(len(resolved_index_to_subject_id))
    metadata["session_max_index_by_subject_index"] = [
        int(value) for value in session_max_index_by_subject_index
    ]
    metadata["train_session_ids"] = [
        str(session_id) for session_id in metadata.get("train_session_ids") or []
    ]
    metadata["eval_session_ids"] = [
        str(session_id) for session_id in metadata.get("eval_session_ids") or []
    ]
    return metadata, metadata_path


def _build_subject_session_metadata(
    *,
    session_context: Mapping[str, Any],
    train_session_ids: list[str],
    eval_session_ids: list[str],
) -> pd.DataFrame:
    train_session_id_set = {str(session_id) for session_id in train_session_ids}
    eval_session_id_set = {str(session_id) for session_id in eval_session_ids}
    records: list[dict[str, Any]] = []

    for row in ordered_session_context_rows(session_context):
        subject_index = int(row["subject_index"])
        subject_id = _normalize_subject_id_str(row.get("subject_id"))
        ordered_session_ids = [
            str(session_id) for session_id in (row.get("ordered_session_ids") or [])
        ]
        ordered_source_session_ids = [
            str(session_id)
            for session_id in (
                row.get("ordered_source_session_ids")
                or row.get("ordered_session_ids")
                or []
            )
        ]
        if len(ordered_session_ids) != len(ordered_source_session_ids):
            raise ValueError(
                "session_context ordered_source_session_ids must align 1:1 with "
                "ordered_session_ids."
            )

        for session_index, (session_id, source_session_id) in enumerate(
            zip(ordered_session_ids, ordered_source_session_ids),
            start=1,
        ):
            if session_id in train_session_id_set:
                session_split = "train"
            elif session_id in eval_session_id_set:
                session_split = "eval"
            else:
                session_split = "full"

            records.append(
                {
                    "subject_index": int(subject_index),
                    "subject_id": subject_id,
                    "session_id": str(session_id),
                    "source_session_id": str(source_session_id),
                    "session_index": int(session_index),
                    "subject_max_session_index": int(len(ordered_session_ids)),
                    "session_split": session_split,
                    "session_date": _extract_session_date(source_session_id),
                }
            )

    return pd.DataFrame.from_records(records).sort_values(
        ["subject_index", "session_index"]
    ).reset_index(drop=True)


def _resolve_han_column_map(
    df_han: pd.DataFrame,
    *,
    task_column: str,
    weekday_column: str,
    foraging_eff_column: str,
    bias_naive_column: str,
    reaction_time_column: str,
) -> dict[str, str]:
    column_map = {
        "rig": "rig",
        "trainer": "trainer",
        "current_stage_actual": "current_stage_actual",
        "task": str(task_column),
        "weekday": str(weekday_column),
        "foraging_eff_random_seed": str(foraging_eff_column),
        "bias_naive": str(bias_naive_column),
        "reaction_time_median": str(reaction_time_column),
    }
    required_columns = {"subject_id", "session_date", *column_map.values()}
    missing_columns = sorted(column for column in required_columns if column not in df_han.columns)
    if missing_columns:
        raise ValueError(
            "df_han is missing required columns: "
            f"{missing_columns}. Configurable columns can be overridden with "
            "--task-column, --weekday-column, --foraging-eff-column, "
            "--bias-naive-column, and --reaction-time-column."
        )
    return column_map


def _attach_han_metadata(
    session_df: pd.DataFrame,
    *,
    df_han: pd.DataFrame,
    column_map: Mapping[str, str],
) -> pd.DataFrame:
    join_keys = ["subject_id", "session_date"]
    actual_columns = list(dict.fromkeys(column_map.values()))
    han_subset = df_han[["subject_id", "session_date", *actual_columns]].copy()
    han_subset["subject_id"] = han_subset["subject_id"].map(_normalize_subject_id_str)
    han_subset["session_date"] = han_subset["session_date"].map(_normalize_session_date)

    counts = (
        han_subset.groupby(join_keys, dropna=False)
        .size()
        .reset_index(name="han_match_count")
    )
    matched_rows = han_subset.merge(counts, on=join_keys, how="left")
    matched_rows = matched_rows[matched_rows["han_match_count"] == 1].drop(
        columns=["han_match_count"]
    )
    matched_rows = matched_rows.drop_duplicates(subset=join_keys).reset_index(drop=True)

    joined = session_df.copy()
    joined["subject_id"] = joined["subject_id"].map(_normalize_subject_id_str)
    joined["session_date"] = joined["session_date"].map(_normalize_session_date)
    joined = joined.merge(counts, on=join_keys, how="left")
    joined = joined.merge(matched_rows, on=join_keys, how="left")
    joined["han_match_count"] = joined["han_match_count"].fillna(0).astype(int)
    joined["join_status"] = np.where(
        joined["han_match_count"] == 0,
        "missing",
        np.where(joined["han_match_count"] == 1, "matched", "ambiguous"),
    )

    for alias, actual_column in column_map.items():
        joined[alias] = joined[actual_column]

    for numeric_column in (
        "foraging_eff_random_seed",
        "bias_naive",
        "reaction_time_median",
    ):
        joined[numeric_column] = pd.to_numeric(joined[numeric_column], errors="coerce")

    return joined


def _clean_categorical_series(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    cleaned = cleaned.where(~cleaned.isna(), other=_UNKNOWN_LABEL)
    cleaned = cleaned.map(lambda value: _UNKNOWN_LABEL if str(value).strip() in {"", "nan", "None"} else str(value))
    return cleaned


def _majority_label(series: pd.Series) -> str:
    cleaned = _clean_categorical_series(series)
    cleaned = cleaned[cleaned != _UNKNOWN_LABEL]
    if cleaned.empty:
        return _UNKNOWN_LABEL
    counts = cleaned.value_counts()
    if len(counts) > 1 and int(counts.iloc[0]) == int(counts.iloc[1]):
        return _MIXED_LABEL
    return str(counts.index[0])


def _aggregate_subject_metadata(session_df: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouped = session_df.groupby("subject_index", sort=True, dropna=False)

    for subject_index, subject_rows in grouped:
        first_row = subject_rows.iloc[0]
        records.append(
            {
                "subject_index": int(subject_index),
                "subject_id": _normalize_subject_id_str(first_row["subject_id"]),
                "n_sessions_total": int(len(subject_rows)),
                "n_sessions_matched_han": int((subject_rows["join_status"] == "matched").sum()),
                "n_sessions_missing_han": int((subject_rows["join_status"] == "missing").sum()),
                "n_sessions_ambiguous_han": int((subject_rows["join_status"] == "ambiguous").sum()),
                "rig_majority": _majority_label(subject_rows["rig"]),
                "trainer_majority": _majority_label(subject_rows["trainer"]),
                "task_majority": _majority_label(subject_rows["task"]),
                "foraging_eff_random_seed_mean": float(
                    pd.to_numeric(
                        subject_rows["foraging_eff_random_seed"], errors="coerce"
                    ).mean()
                )
                if pd.to_numeric(
                    subject_rows["foraging_eff_random_seed"], errors="coerce"
                ).notna().any()
                else np.nan,
                "bias_naive_mean": float(
                    pd.to_numeric(subject_rows["bias_naive"], errors="coerce").mean()
                )
                if pd.to_numeric(subject_rows["bias_naive"], errors="coerce").notna().any()
                else np.nan,
                "reaction_time_median_mean": float(
                    pd.to_numeric(
                        subject_rows["reaction_time_median"], errors="coerce"
                    ).mean()
                )
                if pd.to_numeric(
                    subject_rows["reaction_time_median"], errors="coerce"
                ).notna().any()
                else np.nan,
            }
        )

    return pd.DataFrame.from_records(records).sort_values("subject_index").reset_index(drop=True)


def _embedding_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if str(column).startswith("embedding_")]


def _resolve_numeric_bounds(values: np.ndarray) -> tuple[float, float]:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite_values))
    vmax = float(np.max(finite_values))
    if vmin == vmax:
        return vmin - 0.5, vmax + 0.5
    return vmin, vmax


def _finalize_pairwise_grid(fig: Any, axes: np.ndarray, dim_pairs: list[tuple[str, str]]) -> list[Any]:
    axes_list = list(axes.flat)
    for ax in axes_list[len(dim_pairs):]:
        ax.axis("off")
    return axes_list[: len(dim_pairs)]


def _save_figure(fig: Any, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _make_subject_embedding_plot(
    plot_df: pd.DataFrame,
    *,
    color_column: str,
    color_kind: str,
    color_label: str,
) -> Any:
    embedding_columns = _embedding_columns(plot_df)
    dim_pairs = list(combinations(embedding_columns, 2))
    ncols = min(3, len(dim_pairs))
    nrows = int(math.ceil(len(dim_pairs) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5.5 * ncols, 5.0 * nrows),
        squeeze=False,
    )
    figure_axes = _finalize_pairwise_grid(fig, axes, dim_pairs)

    if color_kind == "categorical":
        categorical_values = _clean_categorical_series(plot_df[color_column])
        categories = list(dict.fromkeys(categorical_values.tolist()))
        cmap = plt.get_cmap("tab20")
        color_map = {
            category: cmap(index % max(1, cmap.N))
            for index, category in enumerate(categories)
        }
        for ax, (x_column, y_column) in zip(figure_axes, dim_pairs):
            for category in categories:
                category_rows = plot_df[categorical_values == category]
                ax.scatter(
                    category_rows[x_column],
                    category_rows[y_column],
                    color=color_map[category],
                    s=60,
                    alpha=0.9,
                    label=category,
                )
            for row in plot_df.itertuples(index=False):
                ax.text(
                    getattr(row, x_column),
                    getattr(row, y_column),
                    str(getattr(row, "subject_index")),
                    fontsize=8,
                    alpha=0.8,
                )
            ax.axhline(0, color="0.85", linewidth=1)
            ax.axvline(0, color="0.85", linewidth=1)
            ax.set_xlabel(x_column.replace("_", " ").title())
            ax.set_ylabel(y_column.replace("_", " ").title())
            ax.set_title(f"{x_column} vs {y_column}")

        legend_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markerfacecolor=color_map[category],
                markeredgecolor=color_map[category],
                label=category,
            )
            for category in categories
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=min(4, max(1, len(legend_handles))),
            title=color_label,
        )
    else:
        numeric_values = pd.to_numeric(plot_df[color_column], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(numeric_values)
        vmin, vmax = _resolve_numeric_bounds(numeric_values)
        scatter_artist = None
        for ax, (x_column, y_column) in zip(figure_axes, dim_pairs):
            ax.axhline(0, color="0.85", linewidth=1)
            ax.axvline(0, color="0.85", linewidth=1)
            if np.any(finite_mask):
                scatter_artist = ax.scatter(
                    plot_df.loc[finite_mask, x_column],
                    plot_df.loc[finite_mask, y_column],
                    c=numeric_values[finite_mask],
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                    s=60,
                    alpha=0.95,
                    edgecolors="black",
                    linewidths=0.35,
                )
            if np.any(~finite_mask):
                ax.scatter(
                    plot_df.loc[~finite_mask, x_column],
                    plot_df.loc[~finite_mask, y_column],
                    color="0.75",
                    s=60,
                    alpha=0.9,
                    edgecolors="black",
                    linewidths=0.35,
                )
            for row in plot_df.itertuples(index=False):
                ax.text(
                    getattr(row, x_column),
                    getattr(row, y_column),
                    str(getattr(row, "subject_index")),
                    fontsize=8,
                    alpha=0.8,
                )
            ax.set_xlabel(x_column.replace("_", " ").title())
            ax.set_ylabel(y_column.replace("_", " ").title())
            ax.set_title(f"{x_column} vs {y_column}")

        if scatter_artist is not None:
            fig.colorbar(scatter_artist, ax=figure_axes, label=color_label)

    fig.suptitle(f"Subject Embedding State Space - {color_label}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _generate_subject_embedding_plots(
    *,
    params: Mapping[str, Any],
    subject_metadata: pd.DataFrame,
    index_to_subject_id: Mapping[int, Any],
    output_dir: Path,
) -> tuple[dict[str, str], list[str]]:
    subject_embeddings = extract_subject_embeddings_from_params(dict(params))
    if subject_embeddings.shape[1] < 2:
        return {}, ["Skipped subject embedding plots because subject_embedding_size < 2."]

    plot_df = subject_embeddings_to_dataframe(dict(index_to_subject_id), subject_embeddings)
    plot_df["subject_id"] = plot_df["subject_id"].map(_normalize_subject_id_str)
    merged_df = plot_df.merge(
        subject_metadata,
        on=["subject_index", "subject_id"],
        how="left",
    )
    plot_paths: dict[str, str] = {}
    skipped: list[str] = []

    for color_column, filename, color_label, color_kind in _SUBJECT_PLOT_SPECS:
        fig = _make_subject_embedding_plot(
            merged_df,
            color_column=color_column,
            color_kind=color_kind,
            color_label=color_label,
        )
        plot_paths[color_column] = _save_figure(fig, output_dir / filename)

    return plot_paths, skipped


def _make_session_context_plot(
    subject_rows: pd.DataFrame,
    *,
    subject_embedding: np.ndarray,
    color_column: str,
    color_kind: str,
    color_label: str,
) -> Any:
    embedding_columns = _embedding_columns(subject_rows)
    dim_pairs = list(combinations(embedding_columns, 2))
    ncols = min(3, len(dim_pairs))
    nrows = int(math.ceil(len(dim_pairs) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6.0 * ncols, 5.0 * nrows),
        squeeze=False,
    )
    figure_axes = _finalize_pairwise_grid(fig, axes, dim_pairs)

    if color_kind == "categorical":
        categorical_values = _clean_categorical_series(subject_rows[color_column])
        categories = list(dict.fromkeys(categorical_values.tolist()))
        cmap = plt.get_cmap("tab20")
        color_map = {
            category: cmap(index % max(1, cmap.N))
            for index, category in enumerate(categories)
        }
        for ax, (x_column, y_column) in zip(figure_axes, dim_pairs):
            ax.plot(
                subject_rows[x_column],
                subject_rows[y_column],
                color="0.75",
                linewidth=1.1,
                alpha=0.9,
                zorder=1,
            )
            for category in categories:
                category_rows = subject_rows[categorical_values == category]
                ax.scatter(
                    category_rows[x_column],
                    category_rows[y_column],
                    color=color_map[category],
                    s=65,
                    alpha=0.95,
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=2,
                    label=category,
                )
            x_dim = int(x_column.split("_")[-1]) - 1
            y_dim = int(y_column.split("_")[-1]) - 1
            ax.scatter(
                float(subject_embedding[x_dim]),
                float(subject_embedding[y_dim]),
                marker="*",
                s=170,
                c="black",
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            ax.set_xlabel(x_column.replace("_", " ").title())
            ax.set_ylabel(y_column.replace("_", " ").title())
            ax.set_title(f"{x_column} vs {y_column}")

        legend_handles = [
            Line2D([0], [0], color="0.75", linewidth=1.1, label="Session trajectory"),
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="",
                markerfacecolor="black",
                markeredgecolor="white",
                markersize=11,
                label="Base subject embedding",
            ),
        ]
        legend_handles.extend(
            [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    linestyle="",
                    markerfacecolor=color_map[category],
                    markeredgecolor="black",
                    markersize=7,
                    label=category,
                )
                for category in categories
            ]
        )
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=min(4, max(1, len(legend_handles))),
            title=color_label,
        )
    else:
        numeric_values = pd.to_numeric(subject_rows[color_column], errors="coerce").to_numpy(dtype=float)
        finite_mask = np.isfinite(numeric_values)
        cmap_name = "plasma" if color_kind == "session_index" else "viridis"
        vmin, vmax = _resolve_numeric_bounds(numeric_values)
        scatter_artist = None

        for ax, (x_column, y_column) in zip(figure_axes, dim_pairs):
            ax.plot(
                subject_rows[x_column],
                subject_rows[y_column],
                color="0.75",
                linewidth=1.1,
                alpha=0.9,
                zorder=1,
            )
            if np.any(finite_mask):
                scatter_artist = ax.scatter(
                    subject_rows.loc[finite_mask, x_column],
                    subject_rows.loc[finite_mask, y_column],
                    c=numeric_values[finite_mask],
                    cmap=cmap_name,
                    vmin=vmin,
                    vmax=vmax,
                    s=65,
                    alpha=0.95,
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=2,
                )
            if np.any(~finite_mask):
                ax.scatter(
                    subject_rows.loc[~finite_mask, x_column],
                    subject_rows.loc[~finite_mask, y_column],
                    color="0.75",
                    s=65,
                    alpha=0.95,
                    edgecolors="black",
                    linewidths=0.4,
                    zorder=2,
                )
            x_dim = int(x_column.split("_")[-1]) - 1
            y_dim = int(y_column.split("_")[-1]) - 1
            ax.scatter(
                float(subject_embedding[x_dim]),
                float(subject_embedding[y_dim]),
                marker="*",
                s=170,
                c="black",
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )
            ax.set_xlabel(x_column.replace("_", " ").title())
            ax.set_ylabel(y_column.replace("_", " ").title())
            ax.set_title(f"{x_column} vs {y_column}")

        legend_handles = [
            Line2D([0], [0], color="0.75", linewidth=1.1, label="Session trajectory"),
            Line2D(
                [0],
                [0],
                marker="*",
                linestyle="",
                markerfacecolor="black",
                markeredgecolor="white",
                markersize=11,
                label="Base subject embedding",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            ncol=len(legend_handles),
            title=color_label,
        )
        if scatter_artist is not None:
            fig.colorbar(scatter_artist, ax=figure_axes, label=color_label)

    first_row = subject_rows.iloc[0]
    fig.suptitle(
        "Session-Conditioned Subject Context State Space "
        f"- Subject {int(first_row['subject_index'])} ({first_row['subject_id']}) - {color_label}",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def _generate_session_context_plots(
    *,
    params: Mapping[str, Any],
    metadata: Mapping[str, Any],
    model_architecture: Mapping[str, Any],
    session_metadata: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, dict[str, str]], list[str]]:
    session_conditioning_cfg = resolve_session_conditioning_from_architecture(
        architecture=model_architecture,
        metadata=metadata,
        multisubject=True,
        max_n_subjects=int(metadata["num_subjects"]),
        subject_embedding_size=int(model_architecture["subject_embedding_size"]),
        context="Embedding space analysis",
    )
    if not bool(session_conditioning_cfg["enabled"]):
        return {}, ["Skipped session-context plots because session_conditioning_enabled=false."]

    plot_df = compute_session_conditioned_context_dataframe(
        dict(params),
        session_context=metadata["session_context"],
        session_encoding_type=str(session_conditioning_cfg["session_encoding_type"]),
        session_integration_type=str(session_conditioning_cfg["session_integration_type"]),
        session_fourier_k=int(session_conditioning_cfg["session_fourier_k"]),
        session_delta_n_layers=int(session_conditioning_cfg["session_delta_n_layers"]),
        session_delta_hidden_size=int(session_conditioning_cfg["session_delta_hidden_size"]),
        session_max_index_by_subject_index=session_conditioning_cfg[
            "session_max_index_by_subject_index"
        ],
        train_session_ids=metadata.get("train_session_ids"),
        eval_session_ids=metadata.get("eval_session_ids"),
        selected_subject_indices=list(range(int(metadata["num_subjects"]))),
    )
    if plot_df.empty:
        return {}, ["Skipped session-context plots because no session-conditioned rows were produced."]

    plot_df["subject_id"] = plot_df["subject_id"].map(_normalize_subject_id_str)
    if len(_embedding_columns(plot_df)) < 2:
        return {}, ["Skipped session-context plots because subject_embedding_size < 2."]
    merged_df = plot_df.merge(
        session_metadata[
            [
                "subject_index",
                "subject_id",
                "session_id",
                "source_session_id",
                "session_index",
                "rig",
                "trainer",
                "current_stage_actual",
                "task",
                "weekday",
                "foraging_eff_random_seed",
                "bias_naive",
                "reaction_time_median",
                "join_status",
            ]
        ],
        on=["subject_index", "subject_id", "session_id", "source_session_id", "session_index"],
        how="left",
    )
    subject_embeddings = extract_subject_embeddings_from_params(dict(params))
    plot_paths: dict[str, dict[str, str]] = {}

    for subject_index, subject_rows in merged_df.groupby("subject_index", sort=True):
        subject_rows = subject_rows.sort_values("session_index").reset_index(drop=True)
        if subject_rows.empty:
            continue
        subject_id = _normalize_subject_id_str(subject_rows["subject_id"].iloc[0]) or f"subject_{subject_index}"
        subject_key = f"subject_{int(subject_index)}_{subject_id}"
        subject_output_dir = output_dir / subject_key
        plot_paths[subject_key] = {}
        subject_embedding = np.asarray(subject_embeddings[int(subject_index)], dtype=float)

        for color_column, filename, color_label, color_kind in _SESSION_PLOT_SPECS:
            fig = _make_session_context_plot(
                subject_rows,
                subject_embedding=subject_embedding,
                color_column=color_column,
                color_kind=color_kind,
                color_label=color_label,
            )
            plot_paths[subject_key][color_column] = _save_figure(fig, subject_output_dir / filename)

    return plot_paths, []


def run_embedding_space_analysis(
    model_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    checkpoint_policy: str = _DEFAULT_CHECKPOINT_POLICY,
    task_column: str = _DEFAULT_TASK_COLUMN,
    weekday_column: str = _DEFAULT_WEEKDAY_COLUMN,
    foraging_eff_column: str = _DEFAULT_FORAGING_EFF_COLUMN,
    bias_naive_column: str = _DEFAULT_BIAS_NAIVE_COLUMN,
    reaction_time_column: str = _DEFAULT_REACTION_TIME_COLUMN,
) -> dict[str, Any]:
    """Run standalone embedding-space analysis for a trained multisubject run."""
    resolved_run = resolve_model_run(
        model_dir,
        split="train",
        checkpoint_policy=checkpoint_policy,
    )
    _validate_resolved_run(resolved_run)

    subject_id_to_index, index_to_subject_id = load_subject_index_map(
        resolved_run.subject_index_map_path
    )
    analysis_metadata, multisubject_metadata_path = _load_analysis_metadata(
        resolved_run=resolved_run,
        subject_id_to_index=subject_id_to_index,
        index_to_subject_id=index_to_subject_id,
    )
    resolved_output_dir = (
        _default_output_dir(model_dir, resolved_run.checkpoint_label)
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    session_metadata = _build_subject_session_metadata(
        session_context=analysis_metadata["session_context"],
        train_session_ids=list(analysis_metadata.get("train_session_ids") or []),
        eval_session_ids=list(analysis_metadata.get("eval_session_ids") or []),
    )
    df_han = _load_han_session_table()
    han_column_map = _resolve_han_column_map(
        df_han,
        task_column=task_column,
        weekday_column=weekday_column,
        foraging_eff_column=foraging_eff_column,
        bias_naive_column=bias_naive_column,
        reaction_time_column=reaction_time_column,
    )
    session_metadata = _attach_han_metadata(
        session_metadata,
        df_han=df_han,
        column_map=han_column_map,
    )
    subject_metadata = _aggregate_subject_metadata(session_metadata)

    subject_session_metadata_path = resolved_output_dir / "subject_session_metadata.csv"
    subject_metadata_path = resolved_output_dir / "subject_metadata.csv"
    session_metadata.to_csv(subject_session_metadata_path, index=False)
    subject_metadata.to_csv(subject_metadata_path, index=False)

    params = _load_json_file(Path(resolved_run.params_path))
    subject_plot_paths, subject_skipped = _generate_subject_embedding_plots(
        params=params,
        subject_metadata=subject_metadata,
        index_to_subject_id=index_to_subject_id,
        output_dir=resolved_output_dir / "plots" / "subject_embeddings",
    )
    session_plot_paths, session_skipped = _generate_session_context_plots(
        params=params,
        metadata=analysis_metadata,
        model_architecture=_resolved_model_architecture(resolved_run),
        session_metadata=session_metadata,
        output_dir=resolved_output_dir / "plots" / "session_context",
    )

    han_join_counts = {
        str(key): int(value)
        for key, value in session_metadata["join_status"].value_counts(dropna=False).items()
    }
    summary = {
        "model_dir": str(Path(model_dir).expanduser().resolve()),
        "output_dir": str(resolved_output_dir),
        "summary_path": str(resolved_output_dir / "summary.json"),
        "subject_session_metadata_path": str(subject_session_metadata_path),
        "subject_metadata_path": str(subject_metadata_path),
        "multisubject_metadata_path": str(multisubject_metadata_path),
        "checkpoint_policy": str(resolved_run.checkpoint_policy),
        "checkpoint_step": resolved_run.checkpoint_step,
        "checkpoint_label": resolved_run.checkpoint_label,
        "params_path": str(resolved_run.params_path),
        "checkpoint_selection_reason": resolved_run.checkpoint_selection_reason,
        "session_conditioning_enabled": bool(resolved_run.session_conditioning_enabled),
        "session_conditioning_encoding_type": str(
            resolved_run.session_conditioning_encoding_type
        ),
        "counts": {
            "num_subjects": int(subject_metadata["subject_index"].nunique())
            if not subject_metadata.empty
            else 0,
            "num_sessions": int(len(session_metadata)),
        },
        "han_join_counts": han_join_counts,
        "plot_paths": {
            "subject_embeddings": subject_plot_paths,
            "session_context": session_plot_paths,
        },
        "skipped": {
            "subject_embeddings": list(subject_skipped),
            "session_context": list(session_skipped),
        },
    }
    summary_path = resolved_output_dir / "summary.json"
    summary_path.write_text(json.dumps(_to_serializable(summary), indent=2))
    summary["summary_path"] = str(summary_path)
    return summary
