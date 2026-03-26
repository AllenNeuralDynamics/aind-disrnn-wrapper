"""Held-out test-subject evaluation helpers for baseline RL runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from aind_dynamic_foraging_models import generative_model

from utils.load_mice_snapshot import load_mice_snapshot
from utils.multisubject import build_subject_index_maps, normalize_subject_id, save_subject_index_map

logger = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "__array__"):
        return np.asarray(value)
    return np.asarray(value)


def _normalize_identifier(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_filename_component(value: Any) -> str:
    normalized = str(_normalize_identifier(value))
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in normalized)
    return safe.strip("_") or "unknown"


def _resolve_output_dir(model_cfg: Any) -> Path:
    output_dir = getattr(model_cfg, "output_dir", "/results/outputs")
    return Path(str(output_dir))


def _load_baseline_output(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_baseline_rl_output(
    output_dir: str | Path,
    output: Mapping[str, Any],
    *,
    indent: int = 2,
) -> Path:
    output_path = Path(output_dir) / "baseline_rl_output.json"
    with output_path.open("w") as f:
        json.dump(dict(output), f, indent=indent, default=_json_default)
    return output_path


def _extract_sessions_from_df(df: Any) -> tuple[list[np.ndarray], list[np.ndarray], list[Any], list[Any]]:
    required_cols = {"ses_idx", "trial", "animal_response", "earned_reward"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Held-out dataframe missing required columns: {missing}")

    sessions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    session_ids: list[Any] = []
    subject_ids: list[Any] = []

    unique_sessions = list(dict.fromkeys(df["ses_idx"].tolist()))
    for session_id in unique_sessions:
        session_df = df[df["ses_idx"] == session_id].sort_values("trial")
        choice_arr = session_df["animal_response"].to_numpy().astype(int)
        valid_choice = (choice_arr == 0) | (choice_arr == 1)
        choice_arr = choice_arr[valid_choice]
        if len(choice_arr) == 0:
            continue

        reward_arr = session_df["earned_reward"].to_numpy().astype(float)[valid_choice]
        sessions.append(choice_arr)
        rewards.append(reward_arr)
        session_ids.append(session_id)
        if "subject_id" in session_df.columns:
            subject_ids.append(session_df["subject_id"].iloc[0])
        else:
            subject_ids.append("unknown")

    return sessions, rewards, session_ids, subject_ids


def _compute_normalized_likelihood(
    choice_sessions: list[np.ndarray],
    choice_prob_sessions: list[np.ndarray],
) -> float:
    total_log_lik = 0.0
    total_trials = 0

    for choices, choice_prob in zip(choice_sessions, choice_prob_sessions):
        choice_prob = _to_numpy(choice_prob)
        if choice_prob.ndim != 2:
            continue
        n_trials = min(len(choices), int(choice_prob.shape[1]))
        if n_trials == 0:
            continue
        choices_idx = choices[:n_trials].astype(int)
        valid = (choices_idx == 0) | (choices_idx == 1)
        if not np.any(valid):
            continue
        trial_idx = np.arange(n_trials)[valid]
        probs = choice_prob[choices_idx[valid], trial_idx]
        probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
        total_log_lik += float(np.sum(np.log(probs)))
        total_trials += int(np.sum(valid))

    if total_trials == 0:
        return 0.0
    return float(np.exp(total_log_lik / total_trials))


def _compute_likelihood_stats(
    choice_sessions: list[np.ndarray],
    choice_prob_sessions: list[np.ndarray],
) -> dict[str, float | int]:
    """Return aggregate held-out likelihood stats plus normalized likelihood."""
    total_log_lik = 0.0
    total_trials = 0

    for choices, choice_prob in zip(choice_sessions, choice_prob_sessions):
        choice_prob = _to_numpy(choice_prob)
        if choice_prob.ndim != 2:
            continue
        n_trials = min(len(choices), int(choice_prob.shape[1]))
        if n_trials == 0:
            continue
        choices_idx = choices[:n_trials].astype(int)
        valid = (choices_idx == 0) | (choices_idx == 1)
        if not np.any(valid):
            continue
        trial_idx = np.arange(n_trials)[valid]
        probs = choice_prob[choices_idx[valid], trial_idx]
        probs = np.clip(probs, 1e-10, 1.0 - 1e-10)
        total_log_lik += float(np.sum(np.log(probs)))
        total_trials += int(np.sum(valid))

    normalized_likelihood = (
        float(np.exp(total_log_lik / total_trials)) if total_trials > 0 else 0.0
    )
    return {
        "total_log_likelihood": float(total_log_lik),
        "total_trials": int(total_trials),
        "normalized_likelihood": float(normalized_likelihood),
    }


def _optional_normalized_likelihood_from_log_stats(
    total_log_likelihood: float,
    total_trials: int,
) -> float | None:
    """Return per-subject likelihood, or None when no trials were available."""
    if total_trials <= 0:
        return None
    return float(np.exp(float(total_log_likelihood) / float(total_trials)))


def _resolve_subject_curriculum_map(df: pd.DataFrame) -> dict[Any, str]:
    """Return subject_id -> curriculum_name using first-seen order."""
    if "subject_id" not in df.columns or "curriculum_name" not in df.columns:
        return {}

    subject_curriculum_map: dict[Any, str] = {}
    for subject_id, subject_rows in df.groupby("subject_id", sort=False):
        curricula = [
            str(value)
            for value in subject_rows["curriculum_name"].dropna().unique().tolist()
        ]
        normalized_subject_id = normalize_subject_id(subject_id)
        if not curricula:
            subject_curriculum_map[normalized_subject_id] = "Unknown"
        elif len(curricula) == 1:
            subject_curriculum_map[normalized_subject_id] = curricula[0]
        else:
            subject_curriculum_map[normalized_subject_id] = "Mixed"
    return subject_curriculum_map


def _plot_subject_likelihood_scatter(
    subject_metrics_df: pd.DataFrame,
    *,
    metric_specs: list[tuple[str, str, float | None]],
    title: str,
) -> plt.Figure:
    """Plot one or more per-subject likelihood panels with pooled references."""
    plot_df = subject_metrics_df.sort_values("subject_index").reset_index(drop=True).copy()
    if "curriculum_name" not in plot_df.columns:
        plot_df["curriculum_name"] = "Unknown"
    else:
        plot_df["curriculum_name"] = plot_df["curriculum_name"].fillna("Unknown")

    n_panels = max(1, len(metric_specs))
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6.6 * n_panels, 4.8),
        squeeze=False,
    )
    curriculum_values = list(dict.fromkeys(plot_df["curriculum_name"].astype(str).tolist()))
    cmap = plt.get_cmap("tab10")
    curriculum_to_color = {
        curriculum: cmap(index % max(1, cmap.N))
        for index, curriculum in enumerate(curriculum_values)
    }
    rng = np.random.default_rng(0)

    for panel_index, (metric_column, panel_title, pooled_value) in enumerate(metric_specs):
        ax = axes[0, panel_index]
        jitter = rng.uniform(-0.35, 0.35, size=len(plot_df))
        panel_df = plot_df[plot_df[metric_column].notna()].copy()
        if panel_df.empty:
            ax.text(
                0.5,
                0.5,
                "No subject-level data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(panel_title)
            ax.set_yticks([])
            continue

        for curriculum in curriculum_values:
            curriculum_rows = panel_df[panel_df["curriculum_name"].astype(str) == curriculum]
            if curriculum_rows.empty:
                continue
            curriculum_indices = curriculum_rows.index.to_numpy(dtype=int)
            ax.scatter(
                curriculum_rows[metric_column].astype(float),
                jitter[curriculum_indices],
                color=curriculum_to_color[curriculum],
                s=65,
                alpha=0.9,
                label=curriculum,
            )

        x_values = panel_df[metric_column].astype(float).to_numpy(dtype=float)
        x_range = float(np.max(x_values) - np.min(x_values)) if len(x_values) > 0 else 0.0
        x_pad = max(0.0025, x_range * 0.01)
        for row_index, row in panel_df.iterrows():
            label_y_offset = 0.035 if row_index % 2 == 0 else -0.045
            ax.text(
                float(row[metric_column]) + x_pad,
                float(jitter[row_index]) + label_y_offset,
                str(int(row["subject_index"])),
                fontsize=8,
                alpha=0.85,
            )

        if pooled_value is not None:
            ax.axvline(
                float(pooled_value),
                color="dimgray",
                linestyle=":",
                linewidth=2.0,
            )
        x_margin = max(0.01, x_range * 0.08 if x_range > 0 else 0.02)
        ax.set_xlim(float(np.min(x_values)) - x_margin, float(np.max(x_values)) + x_margin)
        ax.set_ylim(-0.55, 0.55)
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("Likelihood")
        ax.set_title(f"{panel_title} (n={len(panel_df)})")
        ax.grid(axis="x", color="0.9", linewidth=1)

    curriculum_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=curriculum_to_color[curriculum],
            markeredgecolor=curriculum_to_color[curriculum],
            label=curriculum,
        )
        for curriculum in curriculum_values
    ]
    handles = list(curriculum_handles)
    if any(pooled_value is not None for _, _, pooled_value in metric_specs):
        handles.append(
            Line2D(
                [0],
                [0],
                color="dimgray",
                linestyle=":",
                linewidth=2.0,
                label="Pooled trial",
            )
        )
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=min(len(handles), 5),
        )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def _build_heldout_subject_metrics_dataframe(
    *,
    df_test: pd.DataFrame,
    choice_sessions: list[np.ndarray],
    choice_prob_sessions: list[np.ndarray],
    session_subject_ids: list[Any],
) -> tuple[pd.DataFrame, dict[Any, int], dict[int, Any]]:
    """Aggregate held-out session likelihoods by subject."""
    if len(choice_sessions) != len(choice_prob_sessions) or len(choice_sessions) != len(
        session_subject_ids
    ):
        raise ValueError(
            "Held-out choice sessions, probabilities, and subject ids must have matching lengths."
        )

    ordered_subject_ids, subject_id_to_index, index_to_subject_id = build_subject_index_maps(
        [normalize_subject_id(subject_id) for subject_id in session_subject_ids]
    )
    subject_curriculum_map = _resolve_subject_curriculum_map(df_test)

    rows_by_subject: dict[Any, dict[str, Any]] = {
        subject_id: {
            "subject_id": subject_id,
            "subject_index": int(subject_id_to_index[subject_id]),
            "curriculum_name": str(subject_curriculum_map.get(subject_id, "Unknown")),
            "num_test_sessions": 0,
            "num_test_trials": 0,
            "heldout_total_log_likelihood": 0.0,
            "heldout_total_trials": 0,
        }
        for subject_id in ordered_subject_ids
    }

    for subject_id, choices, choice_prob in zip(
        session_subject_ids,
        choice_sessions,
        choice_prob_sessions,
    ):
        normalized_subject_id = normalize_subject_id(subject_id)
        row = rows_by_subject[normalized_subject_id]
        stats = _compute_likelihood_stats([choices], [choice_prob])
        row["num_test_sessions"] += 1
        row["num_test_trials"] += int(stats["total_trials"])
        row["heldout_total_log_likelihood"] += float(stats["total_log_likelihood"])
        row["heldout_total_trials"] += int(stats["total_trials"])

    rows: list[dict[str, Any]] = []
    for subject_id in ordered_subject_ids:
        row = rows_by_subject[subject_id]
        row["heldout_test_likelihood"] = _optional_normalized_likelihood_from_log_stats(
            float(row["heldout_total_log_likelihood"]),
            int(row["heldout_total_trials"]),
        )
        rows.append(row)

    return (
        pd.DataFrame(rows),
        subject_id_to_index,
        index_to_subject_id,
    )


def _extract_q_histories(
    eval_agent: Any,
    choice_prob_sessions: list[np.ndarray],
) -> list[np.ndarray] | None:
    candidate_attrs = [
        "q_value_history",
        "q_values_history",
        "Q_history",
        "Q_values_history",
        "Qs_history",
        "q_values",
    ]
    history = None
    for attr in candidate_attrs:
        if hasattr(eval_agent, attr):
            history = getattr(eval_agent, attr)
            if history is not None:
                break

    if history is None:
        return None

    if isinstance(history, dict):
        for key in ("q_values", "Q", "history"):
            if key in history:
                history = history[key]
                break

    if isinstance(history, np.ndarray):
        history_arr = _to_numpy(history)
        if history_arr.ndim == 3:
            return [history_arr[i] for i in range(history_arr.shape[0])]
        if history_arr.ndim == 2:
            return [history_arr]
        return None

    if isinstance(history, (list, tuple)):
        out: list[np.ndarray] = []
        for session_item in history:
            item = session_item
            if isinstance(item, dict):
                for key in ("q_values", "Q", "q_history", "values"):
                    if key in item:
                        item = item[key]
                        break
            arr = _to_numpy(item)
            if arr.ndim == 2:
                out.append(arr)
        if out:
            return out
    return None


def _align_two_action_history(
    session_history: np.ndarray,
    n_trials: int,
    *,
    value_name: str,
) -> np.ndarray:
    arr = _to_numpy(session_history)
    if arr.ndim != 2:
        raise ValueError(f"{value_name} session must be 2D, got shape={arr.shape}")

    if arr.shape[0] == n_trials and arr.shape[1] >= 2:
        return arr[:, :2].T
    if arr.shape[1] == n_trials and arr.shape[0] >= 2:
        return arr[:2, :]

    transposed = arr.T
    if transposed.shape[1] == n_trials and transposed.shape[0] >= 2:
        return transposed[:2, :]

    raise ValueError(
        f"Unable to align {value_name} with trials for plotting: "
        f"q_shape={arr.shape}, n_trials={n_trials}"
    )


def _align_q_session(q_session: np.ndarray, n_trials: int) -> np.ndarray:
    return _align_two_action_history(q_session, n_trials, value_name="Q values")


def _align_choice_prob_session(choice_prob_session: np.ndarray, n_trials: int) -> np.ndarray:
    return _align_two_action_history(
        choice_prob_session,
        n_trials,
        value_name="choice probabilities",
    )


def _plot_q_values_for_session(
    *,
    choices: np.ndarray,
    rewards: np.ndarray,
    q_values: np.ndarray | None = None,
    choice_probabilities: np.ndarray | None = None,
) -> plt.Figure:
    if (q_values is None) == (choice_probabilities is None):
        raise ValueError("Provide exactly one of q_values or choice_probabilities.")

    n_trials = len(choices)
    trial_idx = np.arange(n_trials)

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), dpi=100, sharex=True)

    ax0 = axs[0]
    choice_mask = (choices == 0) | (choices == 1)
    rewarded = (rewards > 0) & choice_mask
    unrewarded = (rewards <= 0) & choice_mask
    no_choice = ~choice_mask
    ax0.scatter(trial_idx[no_choice], choices[no_choice], marker="|", s=500, color="gray", alpha=0.7)
    ax0.scatter(trial_idx[unrewarded], choices[unrewarded], marker="|", s=300, color="black", alpha=0.7)
    ax0.scatter(trial_idx[rewarded], choices[rewarded], marker="|", s=600, color="black", alpha=0.85)
    ax0.set_ylim(-0.2, 1.2)
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels(["Left", "Right"])
    ax0.set_title("Choices / rewards", fontsize=14)

    ax1 = axs[1]
    if q_values is not None:
        ax1.plot(trial_idx, q_values[0], label="Q(left)", lw=1.8)
        ax1.plot(trial_idx, q_values[1], label="Q(right)", lw=1.8)
        ax1.set_ylabel("Q value")
        ax1.set_title("Q trajectories", fontsize=14)
        ax1.legend(loc="best")
    else:
        ax1.text(
            0.5,
            0.5,
            "Q trajectories unavailable.\nShowing model choice probabilities below.",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
        ax1.set_title("Q trajectories unavailable", fontsize=14)
        ax1.set_xticks([])
        ax1.set_yticks([])

    ax2 = axs[2]
    if q_values is not None:
        q_max = np.max(q_values, axis=0, keepdims=True)
        q_exp = np.exp(q_values - q_max)
        p = q_exp / np.sum(q_exp, axis=0, keepdims=True)
        probability_labels = {
            0: "Softmax(Q): P(left)",
            1: "Softmax(Q): P(right)",
        }
        ax2.set_title("Probabilities implied by Q values", fontsize=14)
    else:
        p = _to_numpy(choice_probabilities)
        denom = np.sum(p, axis=0, keepdims=True)
        denom = np.where(denom == 0, 1.0, denom)
        p = p / denom
        probability_labels = {
            0: "Model P(left)",
            1: "Model P(right)",
        }
        ax2.set_title("Model choice probabilities", fontsize=14)

    for action_idx in range(p.shape[0]):
        action_name = (
            probability_labels.get(action_idx)
            if p.shape[0] == 2
            else f"Model P(action_{action_idx})"
        )
        ax2.plot(
            trial_idx,
            p[action_idx],
            label=action_name,
            lw=1.8,
        )
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Probability")
    ax2.set_xlabel("Trial")
    ax2.legend(loc="best")

    fig.tight_layout()
    return fig


def evaluate_baseline_rl_on_heldout_subjects(
    hydra_config: Any,
    *,
    wandb_run: Any | None = None,
) -> dict[str, Any] | None:
    data_cfg = hydra_config.data
    model_cfg = hydra_config.model

    if getattr(model_cfg, "type", None) != "baseline_rl":
        logger.info("Skipping baseline RL held-out eval: model.type=%s", getattr(model_cfg, "type", None))
        return None

    output_dir = _resolve_output_dir(model_cfg)
    baseline_output_path = output_dir / "baseline_rl_output.json"
    if not baseline_output_path.exists():
        raise FileNotFoundError(f"Could not find baseline output at {baseline_output_path}")

    baseline_output = _load_baseline_output(baseline_output_path)
    if bool(baseline_output.get("multisubject")) or str(
        baseline_output.get("fit_strategy", "")
    ) == "per_subject":
        logger.info(
            "Skipping held-out baseline RL evaluation because multisubject runs do not "
            "produce a single global fitted parameter set."
        )
        return {
            "enabled": True,
            "skipped": True,
            "reason": (
                "Held-out evaluation is not supported for multisubject baseline RL "
                "per-subject fits."
            ),
        }

    fitted_params = baseline_output.get("fitted_params")
    if not isinstance(fitted_params, dict) or not fitted_params:
        raise ValueError("baseline_rl_output.json missing fitted_params for held-out evaluation")

    logger.info(
        "Loading held-out baseline RL test subjects with selectors: ids=%s, start=%s, end=%s",
        getattr(data_cfg, "test_subject_ids", None),
        getattr(data_cfg, "test_subject_start", None),
        getattr(data_cfg, "test_subject_end", None),
    )

    df_test, test_subject_ids = load_mice_snapshot(
        subject_ids=getattr(data_cfg, "test_subject_ids", None),
        subject_start=getattr(data_cfg, "test_subject_start", None),
        subject_end=getattr(data_cfg, "test_subject_end", None),
        mature_only=bool(getattr(data_cfg, "mature_only", True)),
        curricula=getattr(data_cfg, "curricula", None),
        cols_to_retain=getattr(data_cfg, "cols_to_retain", None),
    )
    if len(df_test) == 0:
        raise ValueError("Held-out baseline RL selection resulted in an empty dataset")

    choice_sessions, reward_sessions, session_ids, session_subject_ids = _extract_sessions_from_df(df_test)
    if len(choice_sessions) == 0:
        raise ValueError("No valid held-out sessions found for baseline RL evaluation")

    agent_class_name = str(getattr(model_cfg, "agent_class", baseline_output.get("agent_class")))
    agent_class_obj = getattr(generative_model, agent_class_name, None)
    if agent_class_obj is None:
        raise ValueError(
            f"Agent class '{agent_class_name}' not found in aind_dynamic_foraging_models.generative_model"
        )

    agent_kwargs = baseline_output.get("agent_kwargs")
    if not isinstance(agent_kwargs, dict):
        agent_kwargs = {}

    eval_agent = agent_class_obj(
        **agent_kwargs,
        seed=getattr(model_cfg, "seed", None),
    )
    eval_agent.set_params(**fitted_params)

    choice_prob_sessions = eval_agent.perform_closed_loop_multi_session(
        choice_sessions,
        reward_sessions,
    )
    choice_prob_sessions = [_to_numpy(arr) for arr in choice_prob_sessions]
    heldout_test_likelihood = _compute_normalized_likelihood(
        choice_sessions,
        choice_prob_sessions,
    )
    logger.info(
        "Held-out baseline RL test likelihood: %.6f",
        heldout_test_likelihood,
    )

    q_histories = _extract_q_histories(eval_agent, choice_prob_sessions)
    if q_histories is None:
        logger.warning(
            "Could not find explicit Q-value histories; plotting model choice probabilities directly."
        )

    sessions_per_subject = int(getattr(data_cfg, "heldout_example_sessions_per_subject", 1))
    if sessions_per_subject < 0:
        raise ValueError("data.heldout_example_sessions_per_subject must be >= 0")

    max_subjects_to_plot = int(
        getattr(
            data_cfg,
            "example_max_subjects",
            getattr(hydra_config, "example_max_subjects", 6),
        )
    )
    if max_subjects_to_plot < 0:
        raise ValueError("example_max_subjects must be >= 0")

    plot_dir = output_dir / "heldout_test"
    plot_dir.mkdir(parents=True, exist_ok=True)

    examples: list[dict[str, Any]] = []
    q_plot_paths: list[str] = []

    if sessions_per_subject > 0:
        sessions_by_subject: dict[Any, list[int]] = {}
        for idx, subject_id in enumerate(session_subject_ids):
            sessions_by_subject.setdefault(subject_id, []).append(idx)

        limited_subject_items = list(sessions_by_subject.items())[:max_subjects_to_plot]
        if len(sessions_by_subject) > max_subjects_to_plot:
            logger.info(
                "Limiting held-out example plotting to first %d subjects (of %d total)",
                max_subjects_to_plot,
                len(sessions_by_subject),
            )

        for subject_id, indices in limited_subject_items:
            for idx in indices[:sessions_per_subject]:
                choices = choice_sessions[idx]
                rewards = reward_sessions[idx]
                if q_histories is not None:
                    q_session = _align_q_session(q_histories[idx], len(choices))
                    fig = _plot_q_values_for_session(
                        choices=choices,
                        rewards=rewards,
                        q_values=q_session,
                    )
                else:
                    choice_prob_session = _align_choice_prob_session(
                        choice_prob_sessions[idx],
                        len(choices),
                    )
                    fig = _plot_q_values_for_session(
                        choices=choices,
                        rewards=rewards,
                        choice_probabilities=choice_prob_session,
                    )
                session_id = session_ids[idx]
                fig.suptitle(f"Session {_normalize_identifier(session_id)}", fontsize=14)
                fig.subplots_adjust(top=0.93)
                out_path = (
                    plot_dir
                    / (
                        f"q_values_over_trials_subject_{_safe_filename_component(subject_id)}"
                        f"_session_{_safe_filename_component(session_id)}.png"
                    )
                )
                fig.savefig(out_path)
                plt.close(fig)

                q_plot_paths.append(str(out_path))
                examples.append(
                    {
                        "subject_id": _normalize_identifier(subject_id),
                        "session_id": _normalize_identifier(session_id),
                        "q_values_over_trials": str(out_path),
                    }
                )

    subject_metrics_df, subject_id_to_index, index_to_subject_id = (
        _build_heldout_subject_metrics_dataframe(
            df_test=pd.DataFrame(df_test).copy(),
            choice_sessions=choice_sessions,
            choice_prob_sessions=choice_prob_sessions,
            session_subject_ids=session_subject_ids,
        )
    )
    subject_metrics_csv_path = plot_dir / "subject_fit_metrics.csv"
    subject_metrics_pickle_path = plot_dir / "subject_fit_metrics.pkl"
    subject_metrics_df.to_csv(subject_metrics_csv_path, index=False)
    subject_metrics_df.to_pickle(subject_metrics_pickle_path)
    subject_index_map_path = save_subject_index_map(
        plot_dir / "subject_index_map.json",
        subject_id_to_index=subject_id_to_index,
        index_to_subject_id=index_to_subject_id,
    )
    likelihood_scatter_fig = _plot_subject_likelihood_scatter(
        subject_metrics_df,
        metric_specs=[
            (
                "heldout_test_likelihood",
                "Heldout Test Likelihood",
                float(heldout_test_likelihood),
            ),
        ],
        title="Heldout Per-Subject Likelihood Scatter",
    )
    likelihood_scatter_path = plot_dir / "subject_likelihood_scatter.png"
    likelihood_scatter_fig.savefig(
        likelihood_scatter_path,
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(likelihood_scatter_fig)

    summary: dict[str, Any] = {
        "enabled": True,
        "test_subject_ids": [_normalize_identifier(s) for s in test_subject_ids],
        "num_test_trials": int(sum(len(x) for x in choice_sessions)),
        "num_test_sessions": int(len(choice_sessions)),
        "heldout_test_likelihood": float(heldout_test_likelihood),
        "heldout_example_sessions_per_subject": sessions_per_subject,
        "example_max_subjects": max_subjects_to_plot,
        "example_sessions": examples,
        "plots": {
            "q_values_over_trials_examples": q_plot_paths,
        },
        "subject_artifacts": {
            "subject_index_map": str(subject_index_map_path),
            "subject_fit_metrics_csv": str(subject_metrics_csv_path),
            "subject_fit_metrics_pickle": str(subject_metrics_pickle_path),
        },
        "subject_likelihood_scatter_path": str(likelihood_scatter_path),
    }

    summary_path = plot_dir / "heldout_baseline_rl_eval_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if wandb_run is not None:
        import wandb

        wandb_run.summary["heldout_test_likelihood"] = float(heldout_test_likelihood)
        wandb_run.summary["heldout/num_test_sessions"] = int(len(choice_sessions))
        wandb_run.summary["heldout/num_test_trials"] = int(sum(len(x) for x in choice_sessions))
        wandb_run.summary["heldout/example_sessions_per_subject"] = sessions_per_subject
        wandb_run.log(
            {
                "heldout/subject_fit_metrics": wandb.Table(dataframe=subject_metrics_df),
                "heldout/subject_likelihood_scatter": wandb.Image(
                    str(likelihood_scatter_path)
                ),
            }
        )
        if q_plot_paths:
            wandb_run.log(
                {
                    "heldout/q_values_over_trials_examples": [
                        wandb.Image(path) for path in q_plot_paths
                    ]
                }
            )

    return summary
