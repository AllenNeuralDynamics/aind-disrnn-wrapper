"""Held-out evaluation helpers for GRU runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

import aind_disrnn_utils.data_loader as dl
from disentangled_rnns.library import rnn_utils

from models.gru_network import make_gru_network
from utils.disrnn_evaluation import (
    HeldoutEvalConfig,
    _get_open_latents_from_params,
    _load_saved_params,
    _normalize_identifier,
    _prob_from_logits,
    _probs_from_logits_2d,
    _resolve_heldout_eval_config,
    _resolve_output_dir,
    _safe_filename_component,
    load_disrnn_heldout_subject_data,
)
from utils.disrnn_plotting import (
    plot_latents_in_space,
    plot_latents_over_trials,
    save_figure,
)

logger = logging.getLogger(__name__)

OPEN_HIDDEN_THRESHOLD = 0.03


def load_gru_heldout_subject_data(config_source: Any) -> dict[str, Any]:
    """Load held-out GRU data once for reuse across evaluation calls."""
    return load_disrnn_heldout_subject_data(config_source)


def evaluate_gru_on_heldout_subjects(
    hydra_config: Any,
    *,
    wandb_run: Any | None = None,
    params_path: Path | None = None,
    output_subdir: str = "heldout_test",
    log_to_wandb: bool = True,
    heldout_data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Evaluate a trained GRU model on held-out test subjects."""
    heldout_cfg = _resolve_heldout_eval_config(hydra_config)
    model_cfg = hydra_config.model

    if not heldout_cfg.enabled:
        logger.info("Held-out evaluation disabled (no test subject selectors configured).")
        return None
    heldout_cfg.validate()

    output_dir = _resolve_output_dir(model_cfg)
    resolved_params_path = params_path or (output_dir / "params.json")
    if not resolved_params_path.exists():
        raise FileNotFoundError(f"Could not find trained params at {resolved_params_path}")

    if heldout_data is None:
        heldout_data = load_gru_heldout_subject_data(hydra_config)

    df_test = heldout_data["df_test"]
    test_subject_ids = heldout_data["test_subject_ids"]
    dataset_test = heldout_data["dataset_test"]
    xs_test = heldout_data["xs_test"]
    ys_test = heldout_data["ys_test"]

    ignore_policy = heldout_cfg.ignore_policy
    output_size = 2 if ignore_policy == "exclude" else 3
    architecture = model_cfg.architecture
    network_output_size = int(getattr(architecture, "output_size", output_size))
    if network_output_size != output_size:
        raise ValueError(
            "Configured GRU output size does not match dataset ignore_policy: "
            f"configured={network_output_size} expected={output_size}"
        )

    make_network = make_gru_network(
        hidden_size=int(architecture.hidden_size),
        output_size=network_output_size,
    )
    params = _load_saved_params(resolved_params_path)

    yhat_test, network_states_test = rnn_utils.eval_network(
        make_network,
        params,
        xs_test,
    )

    n_action_logits = int(getattr(dataset_test, "n_classes", 0))
    if n_action_logits <= 0:
        n_action_logits = int(np.asarray(yhat_test).shape[2] - 1)
    if n_action_logits <= 0:
        raise ValueError(
            f"Invalid number of action logits inferred for held-out eval: {n_action_logits}"
        )

    test_likelihood = rnn_utils.normalized_likelihood(
        ys_test,
        np.asarray(yhat_test)[:, :, :n_action_logits],
    )
    test_likelihood = float(test_likelihood)
    logger.info("Held-out test likelihood: %.4f", test_likelihood)

    plot_dir = output_dir / output_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)

    output_df = dl.add_model_results(
        df_test.copy(),
        np.asarray(network_states_test),
        np.asarray(yhat_test),
        ignore_policy=ignore_policy,
    )

    open_hidden_threshold = OPEN_HIDDEN_THRESHOLD
    open_hidden_units = _get_open_latents_from_params(
        params,
        latent_size=int(np.asarray(network_states_test).shape[2]),
        threshold=open_hidden_threshold,
    )
    logger.info(
        "Held-out GRU hidden units selected with threshold %.4f: %s",
        open_hidden_threshold,
        open_hidden_units,
    )

    session_ids = output_df["ses_idx"].unique()
    if len(session_ids) == 0:
        raise ValueError("No sessions found after held-out model evaluation.")

    if "subject_id" not in output_df.columns:
        raise ValueError("Held-out model outputs do not include required column: subject_id")

    sessions_per_subject = heldout_cfg.heldout_example_sessions_per_subject
    max_subjects_to_plot = heldout_cfg.example_max_subjects

    if sessions_per_subject == 0:
        summary = {
            "enabled": True,
            "params_path": str(resolved_params_path),
            "test_subject_ids": [_normalize_identifier(s) for s in test_subject_ids],
            "num_test_trials": int(len(df_test)),
            "num_test_sessions": int(len(session_ids)),
            "test_likelihood": test_likelihood,
            "heldout_example_sessions_per_subject": 0,
            "plotting_skipped": True,
            "plots": {
                "latents_over_trials_examples": [],
                "latents_in_space_examples": [],
            },
        }
        summary_path = plot_dir / "heldout_eval_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        return summary

    try:
        subject_session_rows = (
            output_df[["subject_id", "ses_idx"]]
            .drop_duplicates()
            .sort_values(["subject_id", "ses_idx"])
        )
        subject_groups = list(subject_session_rows.groupby("subject_id", sort=False))
        if len(subject_groups) > max_subjects_to_plot:
            logger.info(
                "Held-out: Limiting example plotting to first %d subjects (of %d total)",
                max_subjects_to_plot,
                len(subject_groups),
            )
        selected_subject_groups = subject_groups[:max_subjects_to_plot]

        selected_examples: list[dict[str, Any]] = []
        trial_plot_paths: list[Path] = []
        latent_cols = sorted([c for c in output_df.columns if c.startswith("latent_")])
        if not latent_cols:
            raise ValueError("No latent columns found in held-out model outputs.")
        session_order_for_logits = list(dict.fromkeys(output_df["ses_idx"].tolist()))
        session_index_by_id_for_logits = {
            session_id: index for index, session_id in enumerate(session_order_for_logits)
        }
        yhat_test_np = np.asarray(yhat_test)

        for subject_id, subject_rows in selected_subject_groups:
            subject_sessions = subject_rows["ses_idx"].tolist()
            for session_id in subject_sessions[:sessions_per_subject]:
                session_df = output_df[output_df["ses_idx"] == session_id].sort_values("trial")
                latents = session_df[latent_cols].to_numpy()
                choices = session_df["animal_response"].to_numpy()
                rewards = session_df["earned_reward"].astype(int).to_numpy()
                if session_id not in session_index_by_id_for_logits:
                    continue

                session_index = session_index_by_id_for_logits[session_id]
                action_probabilities = _probs_from_logits_2d(
                    yhat_test_np[:, session_index, :n_action_logits]
                )
                n_trials = min(
                    latents.shape[0],
                    choices.shape[0],
                    rewards.shape[0],
                    action_probabilities.shape[0],
                )
                if n_trials <= 0:
                    continue

                fig_trials = plot_latents_over_trials(
                    choices=choices[:n_trials],
                    rewards=rewards[:n_trials],
                    latents=latents[:n_trials],
                    open_latents=open_hidden_units,
                    action_probabilities=action_probabilities[:n_trials],
                )
                fig_trials.suptitle(
                    f"Session {_normalize_identifier(session_id)}",
                    fontsize=14,
                )
                fig_trials.subplots_adjust(top=0.92)
                trials_plot_path = save_figure(
                    fig_trials,
                    plot_dir
                    / (
                        f"latents_over_trials_subject_{_safe_filename_component(subject_id)}"
                        f"_session_{_safe_filename_component(session_id)}.png"
                    ),
                )
                trial_plot_paths.append(trials_plot_path)
                selected_examples.append(
                    {
                        "subject_id": _normalize_identifier(subject_id),
                        "session_id": _normalize_identifier(session_id),
                        "latents_over_trials": str(trials_plot_path),
                    }
                )

        if not selected_examples:
            raise ValueError("Could not select any held-out sessions for example plotting.")

        states = np.asarray(network_states_test)
        session_order = list(dict.fromkeys(df_test["ses_idx"].tolist()))
        session_index_by_id = {session_id: index for index, session_id in enumerate(session_order)}
        prob_class_index = 1 if n_action_logits > 1 else 0
        probs_for_coloring = _prob_from_logits(
            np.asarray(yhat_test)[:, :, :n_action_logits],
            class_index=prob_class_index,
        )
        color_label = "P(right)" if n_action_logits == 2 else f"P(class_{prob_class_index})"
        selected_latents = [
            int(i) for i in open_hidden_units if 0 <= int(i) < states.shape[2]
        ][:4]
        if len(selected_latents) < 2:
            selected_latents = list(range(min(4, states.shape[2])))
        run_len = min(30, states.shape[0])

        space_plot_paths: list[Path] = []
        for subject_id, subject_rows in selected_subject_groups:
            subject_sessions = subject_rows["ses_idx"].tolist()
            session_indices = [
                session_index_by_id[s_id]
                for s_id in subject_sessions
                if s_id in session_index_by_id
            ]
            if not session_indices:
                continue

            subject_states = states[:, session_indices, :]
            subject_colors = probs_for_coloring[:, session_indices]
            example_session_id = subject_sessions[0]
            example_run = subject_states[:run_len, 0, :][:, selected_latents]
            subject_states_points = subject_states.reshape(-1, subject_states.shape[2])
            subject_color_points = subject_colors.reshape(-1)

            fig_space = plot_latents_in_space(
                latent_states=subject_states_points,
                color_values=subject_color_points,
                color_label=color_label,
                selected_latents=selected_latents,
                example_run=example_run,
            )
            fig_space.suptitle(
                str(_normalize_identifier(example_session_id)),
                fontsize=14,
            )
            fig_space.subplots_adjust(top=0.93)
            subject_space_plot_path = save_figure(
                fig_space,
                plot_dir
                / f"latents_in_space_subject_{_safe_filename_component(subject_id)}.png",
            )
            space_plot_paths.append(subject_space_plot_path)

        if not space_plot_paths:
            raise ValueError("Could not generate held-out latent-space plots for any subject.")

        summary = {
            "enabled": True,
            "params_path": str(resolved_params_path),
            "test_subject_ids": [_normalize_identifier(s) for s in test_subject_ids],
            "num_test_trials": int(len(df_test)),
            "num_test_sessions": int(len(session_ids)),
            "test_likelihood": test_likelihood,
            "heldout_example_sessions_per_subject": sessions_per_subject,
            "example_max_subjects": max_subjects_to_plot,
            "heldout_open_hidden_threshold": open_hidden_threshold,
            "open_hidden_units": open_hidden_units,
            "example_session": selected_examples[0]["session_id"],
            "example_subject": selected_examples[0]["subject_id"],
            "example_sessions": selected_examples,
            "plots": {
                "latents_over_trials_examples": [str(p) for p in trial_plot_paths],
                "latents_in_space_examples": [str(p) for p in space_plot_paths],
            },
        }
    except Exception as exc:
        logger.warning("Held-out plotting failed and will be skipped: %s", exc)
        summary = {
            "enabled": True,
            "params_path": str(resolved_params_path),
            "test_subject_ids": [_normalize_identifier(s) for s in test_subject_ids],
            "num_test_trials": int(len(df_test)),
            "num_test_sessions": int(len(session_ids)),
            "test_likelihood": test_likelihood,
            "heldout_example_sessions_per_subject": sessions_per_subject,
            "example_max_subjects": max_subjects_to_plot,
            "heldout_open_hidden_threshold": open_hidden_threshold,
            "open_hidden_units": open_hidden_units,
            "plotting_failed": True,
            "error": str(exc),
            "plots": {
                "latents_over_trials_examples": [],
                "latents_in_space_examples": [],
            },
        }

    summary_path = plot_dir / "heldout_eval_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if wandb_run is not None and log_to_wandb:
        import wandb

        wandb_run.summary["heldout/test_likelihood"] = test_likelihood
        wandb_run.summary["heldout/num_test_sessions"] = int(len(session_ids))
        wandb_run.summary["heldout/num_test_trials"] = int(len(df_test))
        wandb_run.summary["heldout/example_sessions_per_subject"] = sessions_per_subject
        if not summary.get("plotting_failed", False):
            wandb_trial_images = [
                wandb.Image(str(path))
                for path in summary["plots"]["latents_over_trials_examples"]
            ]
            wandb_space_images = [
                wandb.Image(str(path))
                for path in summary["plots"]["latents_in_space_examples"]
            ]
            wandb_run.log(
                {
                    "heldout/latents_over_trials_examples": wandb_trial_images,
                    "heldout/latents_in_space_examples": wandb_space_images,
                }
            )

    return summary
