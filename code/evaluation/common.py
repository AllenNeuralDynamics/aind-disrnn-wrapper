"""Model-agnostic evaluation helpers shared by disRNN, GRU, and baseline-RL evaluation.

These were historically private functions inside ``disrnn_evaluation`` that GRU evaluation
imported across modules. They are model-neutral (probability/logit math, identifier and
filename normalization, subject/session grouping, output-dir resolution, param loading),
so they live in a shared module to break the GRU->disRNN import coupling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from disentangled_rnns.library import rnn_utils


def _resolve_output_dir(model_cfg: Any) -> Path:
    output_dir = getattr(model_cfg, "output_dir", "/results/outputs")
    return Path(str(output_dir))


def _load_saved_params(params_path: Path) -> Any:
    with params_path.open("r") as f:
        params_dict = json.load(f)
    return rnn_utils.to_np(params_dict)


def _prob_from_logits(logits: np.ndarray, class_index: int) -> np.ndarray:
    logits = np.asarray(logits)
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    return probs[:, :, class_index]


def _probs_from_logits_2d(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(f"Expected 2D logits for one session, got shape={logits.shape}")
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def _normalize_identifier(value: Any) -> Any:
    """Convert numpy scalars to Python scalars and keep JSON-safe ID types."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_filename_component(value: Any) -> str:
    normalized = str(_normalize_identifier(value))
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in normalized)
    return safe.strip("_") or "unknown"


def _iter_subject_session_groups(output_df: Any) -> list[tuple[Any, list[Any]]]:
    """Return ordered (subject_id, [session_ids...]) groups for plotting.

    If ``subject_id`` is not available, all sessions are treated as one pseudo-subject.
    """
    if "ses_idx" not in output_df.columns:
        raise ValueError("Model outputs do not include required column: ses_idx")

    if "subject_id" in output_df.columns:
        subject_session_rows = (
            output_df[["subject_id", "ses_idx"]]
            .drop_duplicates()
            .sort_values(["subject_id", "ses_idx"])
        )
        groups: list[tuple[Any, list[Any]]] = []
        for subject_id, subject_rows in subject_session_rows.groupby("subject_id", sort=False):
            groups.append((subject_id, subject_rows["ses_idx"].tolist()))
        return groups

    session_ids = sorted(output_df["ses_idx"].drop_duplicates().tolist())
    return [("all_sessions", session_ids)]


def _compose_log_prefix(log_scope: str | None, label: str) -> str:
    if not log_scope:
        return label
    if log_scope.lower().startswith("checkpoint step"):
        return f"{log_scope}, {label}"
    return f"{log_scope} {label}"


def _aligned_action_probabilities_from_output_df(
    session_df: Any,
    *,
    n_action_logits: int,
) -> np.ndarray:
    """Return row-aligned action probabilities for one session dataframe.

    The trial example plots should be indexed by the same ``(ses_idx, trial)``
    rows for choices, rewards, latents, and probabilities. ``add_model_results``
    already aligns logits onto the original dataframe rows and leaves ignored
    trials as NaN, so we reconstruct probabilities from those merged columns
    instead of using the compact model tensor directly.
    """
    if n_action_logits <= 0:
        raise ValueError(f"Expected n_action_logits > 0, got {n_action_logits}")

    prob_cols = [f"choice_prob_{idx}" for idx in range(n_action_logits)]
    if all(col in session_df.columns for col in prob_cols):
        probs = session_df[prob_cols].to_numpy(dtype=float)
        if probs.ndim != 2 or probs.shape[1] != n_action_logits:
            raise ValueError(
                "Aligned probability columns have unexpected shape: "
                f"{probs.shape}, expected (_, {n_action_logits})"
            )
        return probs

    disrnn_logit_cols = ["logit(left)", "logit(right)", "logit(ignore)"][:n_action_logits]
    if all(col in session_df.columns for col in disrnn_logit_cols):
        logits = session_df[disrnn_logit_cols].to_numpy(dtype=float)
        return _probs_from_logits_2d(logits)

    gru_logit_cols = [f"choice_logit_{idx}" for idx in range(n_action_logits)]
    if all(col in session_df.columns for col in gru_logit_cols):
        logits = session_df[gru_logit_cols].to_numpy(dtype=float)
        return _probs_from_logits_2d(logits)

    available_cols = ", ".join(str(col) for col in session_df.columns)
    raise ValueError(
        "Could not find aligned action probability/logit columns for plotting. "
        f"Expected one of {prob_cols}, {disrnn_logit_cols}, or {gru_logit_cols}. "
        f"Available columns: {available_cols}"
    )
