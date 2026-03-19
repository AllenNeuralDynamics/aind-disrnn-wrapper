"""Held-out test-subject evaluation helpers for baseline RL runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from aind_dynamic_foraging_models import generative_model

from utils.load_mice_snapshot import load_mice_snapshot

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

    q_from_probs: list[np.ndarray] = []
    for probs in choice_prob_sessions:
        probs_arr = _to_numpy(probs)
        if probs_arr.ndim != 2:
            return None
        q_from_probs.append(probs_arr)
    return q_from_probs if q_from_probs else None


def _align_q_session(q_session: np.ndarray, n_trials: int) -> np.ndarray:
    arr = _to_numpy(q_session)
    if arr.ndim != 2:
        raise ValueError(f"Q-session must be 2D, got shape={arr.shape}")

    if arr.shape[0] == n_trials and arr.shape[1] >= 2:
        return arr[:, :2].T
    if arr.shape[1] == n_trials and arr.shape[0] >= 2:
        return arr[:2, :]

    transposed = arr.T
    if transposed.shape[1] == n_trials and transposed.shape[0] >= 2:
        return transposed[:2, :]

    raise ValueError(
        "Unable to align Q values with trials for plotting: "
        f"q_shape={arr.shape}, n_trials={n_trials}"
    )


def _plot_q_values_for_session(
    *,
    choices: np.ndarray,
    rewards: np.ndarray,
    q_values: np.ndarray,
) -> plt.Figure:
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
    ax1.plot(trial_idx, q_values[0], label="Q(left)", lw=1.8)
    ax1.plot(trial_idx, q_values[1], label="Q(right)", lw=1.8)
    ax1.set_ylabel("Q value")
    ax1.set_title("Q trajectories", fontsize=14)
    ax1.legend(loc="best")

    ax2 = axs[2]
    q_max = np.max(q_values, axis=0, keepdims=True)
    q_exp = np.exp(q_values - q_max)
    p = q_exp / np.sum(q_exp, axis=0, keepdims=True)
    for action_idx in range(p.shape[0]):
        if p.shape[0] == 2:
            action_name = "left" if action_idx == 0 else "right"
        else:
            action_name = f"action_{action_idx}"
        ax2.plot(
            trial_idx,
            p[action_idx],
            label=f"Softmax(Q): P({action_name})",
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
    test_likelihood = _compute_normalized_likelihood(choice_sessions, choice_prob_sessions)
    logger.info("Held-out baseline RL test likelihood: %.6f", test_likelihood)

    q_histories = _extract_q_histories(eval_agent, choice_prob_sessions)
    if q_histories is None:
        logger.warning("Could not find explicit Q-value histories; using fallback from action values")
        q_histories = choice_prob_sessions

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
                q_session = _align_q_session(q_histories[idx], len(choices))

                fig = _plot_q_values_for_session(
                    choices=choices,
                    rewards=rewards,
                    q_values=q_session,
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

    summary: dict[str, Any] = {
        "enabled": True,
        "test_subject_ids": [_normalize_identifier(s) for s in test_subject_ids],
        "num_test_trials": int(sum(len(x) for x in choice_sessions)),
        "num_test_sessions": int(len(choice_sessions)),
        "test_likelihood": float(test_likelihood),
        "heldout_example_sessions_per_subject": sessions_per_subject,
        "example_max_subjects": max_subjects_to_plot,
        "example_sessions": examples,
        "plots": {
            "q_values_over_trials_examples": q_plot_paths,
        },
    }

    summary_path = plot_dir / "heldout_baseline_rl_eval_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if wandb_run is not None:
        import wandb

        wandb_run.summary["heldout_test_likelihood"] = float(test_likelihood)
        wandb_run.summary["heldout/num_test_sessions"] = int(len(choice_sessions))
        wandb_run.summary["heldout/num_test_trials"] = int(sum(len(x) for x in choice_sessions))
        wandb_run.summary["heldout/example_sessions_per_subject"] = sessions_per_subject
        if q_plot_paths:
            wandb_run.log(
                {
                    "heldout/q_values_over_trials_examples": [
                        wandb.Image(path) for path in q_plot_paths
                    ]
                }
            )

    return summary
