"""Held-out test-subject evaluation helpers for disRNN runs."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import aind_disrnn_utils.data_loader as dl
from disentangled_rnns.library import disrnn, rnn_utils

from utils.disrnn_plotting import (
    plot_latents_in_space,
    plot_latents_over_trials,
    save_figure,
)
from utils.load_mice_snapshot import load_mice_snapshot

logger = logging.getLogger(__name__)

OPEN_LATENT_THRESHOLD = 0.03


@dataclass(frozen=True)
class HeldoutEvalConfig:
    test_subject_ids: list[Any] | None = None
    test_subject_start: int | None = None
    test_subject_end: int | None = None
    mature_only: bool = True
    curricula: list[str] | None = None
    cols_to_retain: list[str] | None = None
    ignore_policy: str = "exclude"
    features: Mapping[str, Any] | None = None
    batch_size: int | None = None
    batch_mode: str = "random"
    heldout_example_sessions_per_subject: int = 1
    example_max_subjects: int = 6

    @classmethod
    def from_data_cfg(
        cls,
        data_cfg: Any,
        *,
        default_example_max_subjects: int = 6,
    ) -> "HeldoutEvalConfig":
        test_subject_ids = _cfg_get(data_cfg, "test_subject_ids", None)
        if test_subject_ids is not None and not isinstance(test_subject_ids, list):
            if isinstance(test_subject_ids, Sequence) and not isinstance(test_subject_ids, str):
                test_subject_ids = list(test_subject_ids)
            else:
                test_subject_ids = [test_subject_ids]

        curricula = _cfg_get(data_cfg, "curricula", None)
        if curricula is not None and not isinstance(curricula, list):
            curricula = list(curricula)

        cols_to_retain = _cfg_get(data_cfg, "cols_to_retain", None)
        if cols_to_retain is not None and not isinstance(cols_to_retain, list):
            cols_to_retain = list(cols_to_retain)

        features = _cfg_get(data_cfg, "features", None)
        if features is not None and not isinstance(features, Mapping):
            features = None

        return cls(
            test_subject_ids=test_subject_ids,
            test_subject_start=_cfg_get(data_cfg, "test_subject_start", None),
            test_subject_end=_cfg_get(data_cfg, "test_subject_end", None),
            mature_only=bool(_cfg_get(data_cfg, "mature_only", True)),
            curricula=curricula,
            cols_to_retain=cols_to_retain,
            ignore_policy=str(_cfg_get(data_cfg, "ignore_policy", "exclude")),
            features=features,
            batch_size=_cfg_get(data_cfg, "batch_size", None),
            batch_mode=str(_cfg_get(data_cfg, "batch_mode", "random")),
            heldout_example_sessions_per_subject=int(
                _cfg_get(data_cfg, "heldout_example_sessions_per_subject", 1)
            ),
            example_max_subjects=int(
                _cfg_get(data_cfg, "example_max_subjects", default_example_max_subjects)
            ),
        )

    @property
    def enabled(self) -> bool:
        return any(
            v is not None
            for v in (self.test_subject_ids, self.test_subject_start, self.test_subject_end)
        )

    def validate(self) -> None:
        if self.test_subject_ids is not None and (
            self.test_subject_start is not None or self.test_subject_end is not None
        ):
            raise ValueError(
                "Specify either data.test_subject_ids or data.test_subject_start/end, not both."
            )
        if self.heldout_example_sessions_per_subject < 0:
            raise ValueError(
                "data.heldout_example_sessions_per_subject must be >= 0 for held-out plotting."
            )
        if self.example_max_subjects < 0:
            raise ValueError("example_max_subjects must be >= 0.")


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _resolve_heldout_eval_config(config_source: Any) -> HeldoutEvalConfig:
    if isinstance(config_source, HeldoutEvalConfig):
        return config_source

    if hasattr(config_source, "data"):
        data_cfg = config_source.data
        default_max_subjects = int(getattr(config_source, "example_max_subjects", 6))
    else:
        data_cfg = config_source
        default_max_subjects = 6

    return HeldoutEvalConfig.from_data_cfg(
        data_cfg,
        default_example_max_subjects=default_max_subjects,
    )


def should_run_heldout_eval(data_cfg: Any) -> bool:
    """Return True when held-out test subject selectors are configured."""
    return _resolve_heldout_eval_config(data_cfg).enabled


def _resolve_output_dir(model_cfg: Any) -> Path:
    output_dir = getattr(model_cfg, "output_dir", "/results/outputs")
    return Path(str(output_dir))


def _load_saved_params(params_path: Path) -> Any:
    with params_path.open("r") as f:
        params_dict = json.load(f)
    return rnn_utils.to_np(params_dict)


def _build_network_configs(model_cfg: Any, dataset: Any, ignore_policy: str) -> tuple[Any, Any]:
    architecture = model_cfg.architecture
    penalties = model_cfg.penalties
    if bool(getattr(architecture, "multisubject", False)):
        raise ValueError(
            "Held-out subject evaluation is not supported for multisubject disRNN. "
            "v1 only supports seen-subject personalization."
        )

    output_size = 2 if ignore_policy == "exclude" else 3
    disrnn_config = disrnn.DisRnnConfig(
        obs_size=dataset._xs.shape[2],
        output_size=output_size,
        x_names=dataset.x_names,
        y_names=dataset.y_names,
        latent_size=architecture.latent_size,
        update_net_n_units_per_layer=architecture.update_net_n_units_per_layer,
        update_net_n_layers=architecture.update_net_n_layers,
        choice_net_n_units_per_layer=architecture.choice_net_n_units_per_layer,
        choice_net_n_layers=architecture.choice_net_n_layers,
        activation=architecture.activation,
        noiseless_mode=False,
        latent_penalty=penalties.latent_penalty,
        choice_net_latent_penalty=penalties.choice_net_latent_penalty,
        update_net_obs_penalty=penalties.update_net_obs_penalty,
        update_net_latent_penalty=penalties.update_net_latent_penalty,
    )

    noiseless_network = copy.deepcopy(disrnn_config)
    noiseless_network.latent_penalty = 0
    noiseless_network.choice_net_latent_penalty = 0
    noiseless_network.update_net_obs_penalty = 0
    noiseless_network.update_net_latent_penalty = 0
    noiseless_network.l2_scale = 0
    noiseless_network.noiseless_mode = True
    return disrnn_config, noiseless_network


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


def _get_open_latents_from_params(
    params: Any,
    *,
    latent_size: int,
    threshold: float,
) -> list[int]:
    """Return latent indices with bottleneck value > threshold.

    Bottleneck is defined as ``1 - sigma`` where ``sigma`` is derived from
    ``latent_sigma_params``.
    """
    if latent_size <= 0:
        return []

    params_disrnn = None
    if isinstance(params, dict):
        if "hk_disentangled_rnn" in params:
            params_disrnn = params["hk_disentangled_rnn"]
        elif "multisubject_dis_rnn" in params:
            params_disrnn = params["multisubject_dis_rnn"]

    if not isinstance(params_disrnn, dict) or "latent_sigma_params" not in params_disrnn:
        logger.warning(
            "Could not locate latent_sigma_params in params; defaulting open latents to all."
        )
        return list(range(latent_size))

    latent_sigmas = np.asarray(
        disrnn.reparameterize_sigma(params_disrnn["latent_sigma_params"])
    ).reshape(-1)
    bottlenecks = 1.0 - latent_sigmas
    n = min(latent_size, bottlenecks.shape[0])
    return [i for i in range(n) if float(bottlenecks[i]) > threshold]


def plot_disrnn_examples_for_split(
    *,
    split_name: str,
    output_dir: Path,
    output_df: Any,
    network_states: np.ndarray,
    yhat_logits: np.ndarray,
    params: Any,
    sessions_per_subject: int,
    max_subjects_to_plot: int = 6,
    n_action_logits: int | None = None,
    open_latent_threshold: float = OPEN_LATENT_THRESHOLD,
    wandb_run: Any | None = None,
) -> dict[str, Any]:
    """Generate example plots for a split and return a summary dictionary."""
    if sessions_per_subject < 0:
        raise ValueError("sessions_per_subject must be >= 0.")
    if max_subjects_to_plot < 0:
        raise ValueError("max_subjects_to_plot must be >= 0.")

    output_df = output_df.copy()
    states = np.asarray(network_states)
    logits = np.asarray(yhat_logits)
    if states.ndim != 3:
        raise ValueError(f"Expected network_states 3D, got shape={states.shape}")
    if logits.ndim != 3:
        raise ValueError(f"Expected yhat_logits 3D, got shape={logits.shape}")

    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    if sessions_per_subject == 0:
        summary = {
            "split": split_name,
            "num_sessions": int(output_df["ses_idx"].nunique()) if "ses_idx" in output_df.columns else 0,
            "num_trials": int(len(output_df)),
            "plotting_skipped": True,
            "example_sessions_per_subject": 0,
            "example_max_subjects": max_subjects_to_plot,
            "plots": {
                "latents_over_trials_examples": [],
                "latents_in_space_examples": [],
            },
        }
        summary_path = split_dir / "split_eval_summary.json"
        with summary_path.open("w") as f:
            json.dump(summary, f, indent=2)
        return summary

    try:
        open_latents = _get_open_latents_from_params(
            params,
            latent_size=int(states.shape[2]),
            threshold=open_latent_threshold,
        )
        logger.info(
            "%s: open latents by bottleneck > %.4f: %s",
            split_name,
            open_latent_threshold,
            open_latents,
        )

        subject_groups = _iter_subject_session_groups(output_df)
        if not subject_groups:
            raise ValueError(f"No sessions available for split plotting: {split_name}")
        if len(subject_groups) > max_subjects_to_plot:
            logger.info(
                "%s: Limiting example plotting to first %d subjects (of %d total)",
                split_name,
                max_subjects_to_plot,
                len(subject_groups),
            )
        subject_groups = subject_groups[:max_subjects_to_plot]

        session_order = list(dict.fromkeys(output_df["ses_idx"].tolist()))
        session_index_by_id = {session_id: index for index, session_id in enumerate(session_order)}
        if len(session_index_by_id) != states.shape[1]:
            raise ValueError(
                "Split plotting requires output_df session order to match network states. "
                f"Found sessions={len(session_index_by_id)} states={states.shape[1]}"
            )

        if n_action_logits is None or n_action_logits <= 0:
            n_action_logits = int(logits.shape[2] - 1)
        if n_action_logits <= 0:
            raise ValueError(f"Invalid number of action logits inferred for split '{split_name}'")

        probs_for_coloring = _prob_from_logits(
            logits[:, :, :n_action_logits],
            class_index=1 if n_action_logits > 1 else 0,
        )
        color_label = "P(right)" if n_action_logits == 2 else "P(class_1)"
        selected_latents = [int(i) for i in open_latents if 0 <= int(i) < states.shape[2]][:4]
        if len(selected_latents) < 2:
            selected_latents = list(range(min(4, states.shape[2])))
        run_len = min(30, states.shape[0])

        latent_cols = sorted([c for c in output_df.columns if c.startswith("latent_")])
        if not latent_cols:
            raise ValueError(f"No latent columns found in model outputs for split: {split_name}")

        selected_examples: list[dict[str, Any]] = []
        trial_plot_paths: list[Path] = []
        space_plot_paths: list[Path] = []

        for subject_id, session_ids in subject_groups:
            selected_session_ids = session_ids[:sessions_per_subject]
            session_indices = [
                session_index_by_id[s_id]
                for s_id in session_ids
                if s_id in session_index_by_id
            ]
            if not session_indices:
                continue

            subject_states = states[:, session_indices, :]
            subject_colors = probs_for_coloring[:, session_indices]
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
            fig_space.suptitle(str(_normalize_identifier(session_ids[0])), fontsize=14)
            fig_space.subplots_adjust(top=0.93)
            subject_space_plot_path = save_figure(
                fig_space,
                split_dir / f"latents_in_space_subject_{_safe_filename_component(subject_id)}.png",
            )
            space_plot_paths.append(subject_space_plot_path)

            for session_id in selected_session_ids:
                session_df = output_df[output_df["ses_idx"] == session_id].sort_values("trial")
                latents = session_df[latent_cols].to_numpy()
                choices = session_df["animal_response"].to_numpy()
                rewards = session_df["earned_reward"].astype(int).to_numpy()
                session_index = session_index_by_id[session_id]
                action_probabilities = _probs_from_logits_2d(
                    logits[:, session_index, :n_action_logits]
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
                    open_latents=open_latents,
                    action_probabilities=action_probabilities[:n_trials],
                )
                fig_trials.suptitle(f"Session {_normalize_identifier(session_id)}", fontsize=14)
                fig_trials.subplots_adjust(top=0.92)
                trials_plot_path = save_figure(
                    fig_trials,
                    split_dir
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
            raise ValueError(f"Could not select any sessions for example plotting: {split_name}")
        if not space_plot_paths:
            raise ValueError(f"Could not generate latent-space plots for split: {split_name}")

        summary = {
            "split": split_name,
            "num_sessions": int(len(session_order)),
            "num_trials": int(len(output_df)),
            "example_sessions_per_subject": sessions_per_subject,
            "example_max_subjects": max_subjects_to_plot,
            "open_latent_threshold": open_latent_threshold,
            "open_latents": open_latents,
            "example_session": selected_examples[0]["session_id"],
            "example_subject": selected_examples[0]["subject_id"],
            "example_sessions": selected_examples,
            "plots": {
                "latents_over_trials_examples": [str(p) for p in trial_plot_paths],
                "latents_in_space_examples": [str(p) for p in space_plot_paths],
            },
        }
    except Exception as exc:
        logger.warning("Split plotting failed for %s: %s", split_name, exc)
        summary = {
            "split": split_name,
            "num_sessions": int(output_df["ses_idx"].nunique()) if "ses_idx" in output_df.columns else 0,
            "num_trials": int(len(output_df)),
            "plotting_failed": True,
            "error": str(exc),
            "example_sessions_per_subject": sessions_per_subject,
            "plots": {
                "latents_over_trials_examples": [],
                "latents_in_space_examples": [],
            },
        }

    summary_path = split_dir / "split_eval_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    if wandb_run is not None and not summary.get("plotting_failed", False):
        import wandb

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
                f"{split_name}/latents_over_trials_examples": wandb_trial_images,
                f"{split_name}/latents_in_space_examples": wandb_space_images,
            }
        )

    return summary


def evaluate_disrnn_on_heldout_subjects(
    hydra_config: Any,
    *,
    wandb_run: Any | None = None,
    params_path: Path | None = None,
    output_subdir: str = "heldout_test",
    log_to_wandb: bool = True,
    heldout_data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Evaluate trained disRNN model on held-out test subjects.

    This loads test subjects from snapshot data according to the test selection
    fields in ``hydra_config.data`` and evaluates the saved model parameters.
    """
    heldout_cfg = _resolve_heldout_eval_config(hydra_config)
    model_cfg = hydra_config.model

    if not heldout_cfg.enabled:
        logger.info("Held-out evaluation disabled (no test subject selectors configured).")
        return None
    if bool(getattr(model_cfg.architecture, "multisubject", False)):
        raise ValueError(
            "Held-out subject evaluation is not supported for multisubject disRNN. "
            "v1 only supports seen-subject personalization."
        )
    heldout_cfg.validate()

    output_dir = _resolve_output_dir(model_cfg)
    resolved_params_path = params_path or (output_dir / "params.json")
    if not resolved_params_path.exists():
        raise FileNotFoundError(f"Could not find trained params at {resolved_params_path}")

    if heldout_data is None:
        heldout_data = load_disrnn_heldout_subject_data(hydra_config)

    df_test = heldout_data["df_test"]
    test_subject_ids = heldout_data["test_subject_ids"]
    dataset_test = heldout_data["dataset_test"]

    _, noiseless_network = _build_network_configs(
        model_cfg,
        dataset_test,
        ignore_policy=heldout_cfg.ignore_policy,
    )
    params = _load_saved_params(resolved_params_path)

    xs_test = heldout_data["xs_test"]
    ys_test = heldout_data["ys_test"]
    yhat_test, network_states_test = rnn_utils.eval_network(
        lambda: disrnn.HkDisentangledRNN(noiseless_network),
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
        yhat_test[:, :, :n_action_logits],
    )
    test_likelihood = float(test_likelihood)
    logger.info("Held-out test likelihood: %.4f", test_likelihood)

    plot_dir = output_dir / output_subdir
    plot_dir.mkdir(parents=True, exist_ok=True)

    output_df = dl.add_model_results(
        df_test.copy(),
        np.asarray(network_states_test),
        np.asarray(yhat_test),
        ignore_policy=heldout_cfg.ignore_policy,
    )

    open_latent_threshold = OPEN_LATENT_THRESHOLD
    open_latents = _get_open_latents_from_params(
        params,
        latent_size=int(np.asarray(network_states_test).shape[2]),
        threshold=open_latent_threshold,
    )
    logger.info(
        "Held-out open latents by bottleneck > %.4f: %s",
        open_latent_threshold,
        open_latents,
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

        if wandb_run is not None and log_to_wandb:
            wandb_run.summary["heldout/test_likelihood"] = test_likelihood
            wandb_run.summary["heldout/num_test_sessions"] = int(len(session_ids))
            wandb_run.summary["heldout/num_test_trials"] = int(len(df_test))
            wandb_run.summary["heldout/example_sessions_per_subject"] = 0

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
                    open_latents=open_latents,
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
        selected_latents = [int(i) for i in open_latents if 0 <= int(i) < states.shape[2]][:4]
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
            "heldout_open_latent_threshold": open_latent_threshold,
            "open_latents": open_latents,
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
            "heldout_open_latent_threshold": open_latent_threshold,
            "open_latents": open_latents,
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


def load_disrnn_heldout_subject_data(config_source: Any) -> dict[str, Any]:
    """Load and construct held-out disRNN data once for reuse across eval calls."""
    heldout_cfg = _resolve_heldout_eval_config(config_source)

    if not heldout_cfg.enabled:
        raise ValueError("Held-out evaluation disabled (no test subject selectors configured).")
    heldout_cfg.validate()

    logger.info(
        "Loading held-out test subjects with selectors: ids=%s, start=%s, end=%s",
        heldout_cfg.test_subject_ids,
        heldout_cfg.test_subject_start,
        heldout_cfg.test_subject_end,
    )

    df_test, test_subject_ids = load_mice_snapshot(
        subject_ids=heldout_cfg.test_subject_ids,
        subject_start=heldout_cfg.test_subject_start,
        subject_end=heldout_cfg.test_subject_end,
        mature_only=heldout_cfg.mature_only,
        curricula=heldout_cfg.curricula,
        cols_to_retain=heldout_cfg.cols_to_retain,
    )
    if len(df_test) == 0:
        raise ValueError("Held-out test selection resulted in an empty dataset.")

    dataset_test = dl.create_disrnn_dataset(
        df_test,
        ignore_policy=heldout_cfg.ignore_policy,
        features=heldout_cfg.features,
        batch_size=heldout_cfg.batch_size,
        batch_mode=heldout_cfg.batch_mode,
    )
    # Keep held-out raw data aligned with sessions that survive dataset construction.
    # create_disrnn_dataset can silently drop sessions whose every trial is ignored
    # when ignore_policy == "exclude", which would otherwise break add_model_results.
    if heldout_cfg.ignore_policy == "exclude" and "animal_response" in df_test.columns:
        valid_sessions = df_test[df_test["animal_response"] != 2]["ses_idx"].unique()
        df_test = df_test[df_test["ses_idx"].isin(valid_sessions)].copy()
    xs_test, ys_test = dataset_test.get_all()

    return {
        "df_test": df_test,
        "test_subject_ids": test_subject_ids,
        "dataset_test": dataset_test,
        "xs_test": xs_test,
        "ys_test": ys_test,
    }
