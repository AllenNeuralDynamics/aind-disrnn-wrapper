"""Unified command-line entry point for standalone post-training evaluation.

This is the single, training-independent way to run any post-training analysis on a
*saved* trained-run directory (``model_dir`` containing ``inputs.yaml`` + ``outputs/``).
It dispatches to the existing public functions in :mod:`post_training_analysis`; no
training is invoked and ``model_trainers`` is never imported just to evaluate.

Examples
--------
    python run_analysis.py generative           --model-dir RUN [--split train --checkpoint-policy best_eval]
    python run_analysis.py from-histories       --simulated-history sim.pkl [--animal-history a.pkl --resolved-run resolved_run.json]
    python run_analysis.py likelihood-comparison --model-dirs A B C [--no-include-heldout]
    python run_analysis.py likelihood-advantage  --model1-dir A --model2-dir B [--split combined]
    python run_analysis.py embedding             --model-dir RUN [--checkpoint-policy best_eval]
    python run_analysis.py baseline-rl           --resolved-run resolved_run.json --fitting-df fits.pkl
    python run_analysis.py finetune              --config finetune.yaml [--output-root DIR]

State-space / Q-space re-plots (two-stage)
------------------------------------------
``likelihood-advantage`` is the expensive stage: it loads both models, evaluates them,
rolls out the baseline RL agent, extracts RNN hidden states, and writes a canonical
trial-level dataframe to ``<output-dir>/trial_advantage.pkl``.

The sub-commands below are the cheap stage: they read that pickle and re-render figures
**without loading any model**, so a projection/colouring can be changed without repeating
the evaluation. (``likelihood-advantage`` also runs them inline by default; these entry
points exist to re-plot afterwards.)

    python run_analysis.py state-space-condition --trial-advantage OUT/trial_advantage.pkl
    python run_analysis.py state-space-subject   --trial-advantage OUT/trial_advantage.pkl [--subject-ids S1 S2]
    python run_analysis.py state-space-overview  --trial-advantage OUT/trial_advantage.pkl [--color-columns COL ...]
    python run_analysis.py q-space-condition     --trial-advantage OUT/trial_advantage.pkl
    python run_analysis.py q-space-subject       --trial-advantage OUT/trial_advantage.pkl

``embedding-params`` is separate: it needs both run directories, and colours model1's
subject embeddings by model2's fitted baseline-RL parameters.

    python run_analysis.py embedding-params --model1-dir RNN_RUN --model2-dir BASELINE_RUN

Analysis functions are imported lazily inside each handler so that importing this module
(or running an unrelated sub-command) never pulls in heavy or training-adjacent code.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from utils.run_helpers import configure_sys_logger

logger = logging.getLogger(__name__)


def _opt_path(value: Any) -> Path | None:
    return None if value in (None, "") else Path(value).expanduser()


def _opt_json(value: Any) -> Any:
    """Parse an optional JSON-object argument (used for column -> values mappings)."""
    if value in (None, ""):
        return None
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            f"expected a JSON object (column -> values), got {type(parsed).__name__}"
        )
    return parsed


def _emit(result: Any) -> None:
    """Print a JSON summary of an analysis result (best-effort serialization)."""
    print(json.dumps(result, indent=2, default=str))


# --------------------------------------------------------------------------- #
# Sub-command handlers (each imports its analysis function lazily)
# --------------------------------------------------------------------------- #
def _cmd_generative(args: argparse.Namespace) -> dict[str, Any]:
    from post_training_analysis import run_post_training_analysis

    return run_post_training_analysis(
        args.model_dir,
        split=args.split,
        checkpoint_policy=args.checkpoint_policy,
        rollout_mode=args.rollout_mode,
        n_rollouts_per_session=args.n_rollouts_per_session,
        window_size=args.window_size,
        output_dir=_opt_path(args.output_dir),
        save_animal_session_history=args.save_animal_session_history,
        session_partitions=args.session_partitions,
    )


def _cmd_from_histories(args: argparse.Namespace) -> dict[str, Any]:
    from post_training_analysis import run_post_training_analysis_from_saved_histories

    return run_post_training_analysis_from_saved_histories(
        args.simulated_history,
        animal_session_history_path=args.animal_history,
        resolved_run_path=args.resolved_run,
        output_dir=_opt_path(args.output_dir),
        window_size=args.window_size,
        save_animal_session_history=args.save_animal_session_history,
        session_partitions=args.session_partitions,
    )


def _cmd_likelihood_comparison(args: argparse.Namespace) -> dict[str, Any]:
    from post_training_analysis import run_prediction_likelihood_comparison

    return run_prediction_likelihood_comparison(
        args.model_dirs,
        checkpoint_policy=args.checkpoint_policy,
        output_dir=_opt_path(args.output_dir),
        model_labels=args.model_labels,
        include_heldout=args.include_heldout,
        precomputed_session_metrics_path=args.precomputed_session_metrics,
    )


def _cmd_likelihood_advantage(args: argparse.Namespace) -> dict[str, Any]:
    from post_training_analysis import run_likelihood_advantage_analysis

    return run_likelihood_advantage_analysis(
        args.model1_dir,
        args.model2_dir,
        split=args.split,
        checkpoint_policy=args.checkpoint_policy,
        output_dir=_opt_path(args.output_dir),
        history_warmup=args.history_warmup,
        top_k_variables=args.top_k_variables,
        jitter_seed=args.jitter_seed,
        include_rnn_state_space=args.include_rnn_state_space,
        pca_seed=args.pca_seed,
        pca_fit_fraction=args.pca_fit_fraction,
        include_baseline_q_space=args.include_baseline_q_space,
    )


def _cmd_state_space_condition(args: argparse.Namespace) -> dict[str, Any]:
    from post_training_analysis import run_rnn_state_space_condition_analysis

    return run_rnn_state_space_condition_analysis(
        args.trial_advantage,
        condition_columns=args.condition_columns,
        condition_values_by_column=_opt_json(args.condition_values),
        probability_color_column=args.probability_color_column,
        output_dir=_opt_path(args.output_dir),
        pca_seed=args.pca_seed,
        pca_fit_fraction=args.pca_fit_fraction,
        n_variance_pcs=args.n_variance_pcs,
        n_plot_pcs=args.n_plot_pcs,
    )


def _cmd_state_space_subject(args: argparse.Namespace) -> dict[str, Any]:
    from post_training_analysis import run_rnn_state_space_subject_analysis

    return run_rnn_state_space_subject_analysis(
        args.trial_advantage,
        probability_column=args.probability_column,
        subject_ids=args.subject_ids,
        subject_embeddings_path=_opt_path(args.subject_embeddings),
        output_dir=_opt_path(args.output_dir),
        pca_seed=args.pca_seed,
        pca_fit_fraction=args.pca_fit_fraction,
        n_variance_pcs=args.n_variance_pcs,
        n_plot_pcs=args.n_plot_pcs,
    )


def _cmd_state_space_overview(args: argparse.Namespace) -> dict[str, Any]:
    from post_training_analysis import run_rnn_state_space_overview_analysis

    return run_rnn_state_space_overview_analysis(
        args.trial_advantage,
        color_columns=args.color_columns,
        output_dir=_opt_path(args.output_dir),
        pca_seed=args.pca_seed,
        pca_fit_fraction=args.pca_fit_fraction,
        n_variance_pcs=args.n_variance_pcs,
        n_plot_pcs=args.n_plot_pcs,
    )


def _cmd_q_space_condition(args: argparse.Namespace) -> dict[str, Any]:
    from post_training_analysis import run_baseline_q_space_condition_analysis

    return run_baseline_q_space_condition_analysis(
        args.trial_advantage,
        condition_columns=args.condition_columns,
        condition_values_by_column=_opt_json(args.condition_values),
        output_dir=_opt_path(args.output_dir),
    )


def _cmd_q_space_subject(args: argparse.Namespace) -> dict[str, Any]:
    from post_training_analysis import run_baseline_q_space_subject_analysis

    return run_baseline_q_space_subject_analysis(
        args.trial_advantage,
        probability_column=args.probability_column,
        subject_ids=args.subject_ids,
        output_dir=_opt_path(args.output_dir),
    )


def _cmd_embedding_params(args: argparse.Namespace) -> dict[str, Any]:
    from post_training_analysis import run_subject_embedding_baseline_parameter_analysis

    return run_subject_embedding_baseline_parameter_analysis(
        args.model1_dir,
        args.model2_dir,
        checkpoint_policy=args.checkpoint_policy,
        parameter_names=args.parameter_names,
        output_dir=_opt_path(args.output_dir),
        subject_embeddings_path=_opt_path(args.subject_embeddings),
    )


def _cmd_embedding(args: argparse.Namespace) -> dict[str, Any]:
    from post_training_analysis import run_embedding_space_analysis

    return run_embedding_space_analysis(
        args.model_dir,
        output_dir=_opt_path(args.output_dir),
        checkpoint_policy=args.checkpoint_policy,
        task_column=args.task_column,
        room_column=args.room_column,
        weekday_column=args.weekday_column,
        foraging_eff_column=args.foraging_eff_column,
        bias_naive_column=args.bias_naive_column,
        reaction_time_column=args.reaction_time_column,
    )


def _cmd_baseline_rl(args: argparse.Namespace) -> dict[str, Any]:
    from post_training_analysis import run_baseline_rl_post_training_analysis

    kwargs: dict[str, Any] = dict(
        output_dir=_opt_path(args.output_dir),
        n_rollouts_per_session=args.n_rollouts_per_session,
        session_id_policy=args.session_id_policy,
        fit_gap_policy=args.fit_gap_policy,
    )
    if args.model_aliases:
        kwargs["model_aliases"] = args.model_aliases
    return run_baseline_rl_post_training_analysis(
        args.resolved_run,
        args.fitting_df,
        **kwargs,
    )


def _cmd_finetune(args: argparse.Namespace) -> dict[str, Any]:
    from post_training_analysis import run_heldout_subject_finetuning_from_config

    return run_heldout_subject_finetuning_from_config(
        Path(args.config).expanduser(),
        output_root=_opt_path(args.output_root),
    )


# --------------------------------------------------------------------------- #
# Argument parser
# --------------------------------------------------------------------------- #
_CONDITION_VALUES_HELP = (
    'JSON object of column -> allowed values, e.g. \'{"session_stage": ["early", "late"]}\'.'
)


def _add_output_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: analysis-specific default, typically /results).",
    )


def _add_trial_advantage(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trial-advantage",
        required=True,
        help="Path to trial_advantage.pkl written by the `likelihood-advantage` sub-command.",
    )


def _add_pca_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pca-seed", type=int, default=0)
    parser.add_argument("--pca-fit-fraction", type=float, default=0.5)
    parser.add_argument("--n-variance-pcs", type=int, default=10)
    parser.add_argument("--n-plot-pcs", type=int, default=4)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_analysis.py",
        description="Standalone post-training evaluation runner (training-independent).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # generative
    p = sub.add_parser("generative", help="Generative rollout + switch/history statistics vs animal.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--split", default="train", help="train | eval | combined | heldout")
    p.add_argument("--checkpoint-policy", default="best_eval", help="best_eval | best_heldout | final")
    p.add_argument("--rollout-mode", default="curriculum_matched")
    p.add_argument("--n-rollouts-per-session", type=int, default=1)
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--save-animal-session-history", action="store_true")
    p.add_argument("--session-partitions", nargs="+", default=None, help="e.g. train eval combined")
    _add_output_dir(p)
    p.set_defaults(func=_cmd_generative)

    # from-histories
    p = sub.add_parser("from-histories", help="Switch statistics from saved simulated/animal histories (no model load).")
    p.add_argument("--simulated-history", required=True, help="Path to simulated session-history pickle.")
    p.add_argument("--animal-history", default=None, help="Optional animal session-history pickle.")
    p.add_argument("--resolved-run", default=None, help="Optional resolved_run.json for metadata.")
    p.add_argument("--window-size", type=int, default=10)
    p.add_argument("--save-animal-session-history", action="store_true")
    p.add_argument("--session-partitions", nargs="+", default=None)
    _add_output_dir(p)
    p.set_defaults(func=_cmd_from_histories)

    # likelihood-comparison
    p = sub.add_parser("likelihood-comparison", help="Cross-model next-trial prediction likelihood.")
    p.add_argument("--model-dirs", nargs="+", required=True)
    p.add_argument("--checkpoint-policy", default="best_eval")
    p.add_argument("--model-labels", nargs="+", default=None)
    p.add_argument("--include-heldout", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--precomputed-session-metrics", default=None)
    _add_output_dir(p)
    p.set_defaults(func=_cmd_likelihood_comparison)

    # likelihood-advantage
    p = sub.add_parser("likelihood-advantage", help="Log-likelihood advantage + RNN/Q state-space analysis between two models.")
    p.add_argument("--model1-dir", required=True)
    p.add_argument("--model2-dir", required=True)
    p.add_argument("--split", default="combined")
    p.add_argument("--checkpoint-policy", default="best_eval")
    p.add_argument("--history-warmup", type=int, default=10)
    p.add_argument("--top-k-variables", type=int, default=3)
    p.add_argument("--jitter-seed", type=int, default=0)
    p.add_argument("--include-rnn-state-space", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pca-seed", type=int, default=0)
    p.add_argument("--pca-fit-fraction", type=float, default=0.5)
    p.add_argument("--include-baseline-q-space", action=argparse.BooleanOptionalAction, default=True)
    _add_output_dir(p)
    p.set_defaults(func=_cmd_likelihood_advantage)

    # state-space-condition
    p = sub.add_parser("state-space-condition", help="RNN state-space PCA projections for condition-matched trials.")
    _add_trial_advantage(p)
    p.add_argument("--condition-columns", nargs="+", default=None)
    p.add_argument("--condition-values", default=None, help=_CONDITION_VALUES_HELP)
    p.add_argument("--probability-color-column", default="p_model1_right")
    _add_pca_options(p)
    _add_output_dir(p)
    p.set_defaults(func=_cmd_state_space_condition)

    # state-space-subject
    p = sub.add_parser("state-space-subject", help="Shared-PCA RNN states per subject, colored by model probability.")
    _add_trial_advantage(p)
    p.add_argument("--probability-column", default="p_model1_left")
    p.add_argument("--subject-ids", nargs="+", default=None)
    p.add_argument("--subject-embeddings", default=None)
    _add_pca_options(p)
    _add_output_dir(p)
    p.set_defaults(func=_cmd_state_space_subject)

    # state-space-overview
    p = sub.add_parser("state-space-overview", help="Full RNN state-space projections colored by diagnostic columns.")
    _add_trial_advantage(p)
    p.add_argument("--color-columns", nargs="+", default=None)
    _add_pca_options(p)
    _add_output_dir(p)
    p.set_defaults(func=_cmd_state_space_overview)

    # q-space-condition
    p = sub.add_parser("q-space-condition", help="Baseline-RL Q-space for condition-matched trials.")
    _add_trial_advantage(p)
    p.add_argument("--condition-columns", nargs="+", default=None)
    p.add_argument("--condition-values", default=None, help=_CONDITION_VALUES_HELP)
    _add_output_dir(p)
    p.set_defaults(func=_cmd_q_space_condition)

    # q-space-subject
    p = sub.add_parser("q-space-subject", help="Baseline-RL Q-space per subject, colored by baseline probability.")
    _add_trial_advantage(p)
    p.add_argument("--probability-column", default="p_rl_left")
    p.add_argument("--subject-ids", nargs="+", default=None)
    _add_output_dir(p)
    p.set_defaults(func=_cmd_q_space_subject)

    # embedding-params
    p = sub.add_parser(
        "embedding-params",
        help="Model1 subject embeddings colored by model2 baseline-RL fitted parameters.",
    )
    p.add_argument("--model1-dir", required=True, help="Multisubject RNN run (supplies the embeddings).")
    p.add_argument("--model2-dir", required=True, help="Baseline-RL run (supplies the fitted parameters).")
    p.add_argument("--checkpoint-policy", default="best_eval")
    p.add_argument("--parameter-names", nargs="+", default=None)
    p.add_argument("--subject-embeddings", default=None)
    _add_output_dir(p)
    p.set_defaults(func=_cmd_embedding_params)

    # embedding
    p = sub.add_parser("embedding", help="Subject embedding-space visualization for multisubject runs.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--checkpoint-policy", default="best_eval")
    p.add_argument("--task-column", default="curriculum_name")
    p.add_argument("--room-column", default="room")
    p.add_argument("--weekday-column", default="weekday")
    p.add_argument("--foraging-eff-column", default="foraging_eff_random_seed")
    p.add_argument("--bias-naive-column", default="bias_naive")
    p.add_argument("--reaction-time-column", default="reaction_time_median")
    _add_output_dir(p)
    p.set_defaults(func=_cmd_embedding)

    # baseline-rl
    p = sub.add_parser("baseline-rl", help="Baseline-RL post-training analysis from fitted parameters.")
    p.add_argument("--resolved-run", required=True, help="Path to resolved_run.json.")
    p.add_argument("--fitting-df", required=True, help="Path to fitted-parameter dataframe pickle.")
    p.add_argument("--model-aliases", nargs="+", default=None)
    p.add_argument("--n-rollouts-per-session", type=int, default=1)
    p.add_argument("--session-id-policy", default="auto")
    p.add_argument("--fit-gap-policy", default="per_model_skip")
    _add_output_dir(p)
    p.set_defaults(func=_cmd_baseline_rl)

    # finetune
    p = sub.add_parser("finetune", help="Held-out subject fine-tuning (training-adjacent; uses trainers).")
    p.add_argument("--config", required=True, help="Path to the standalone held-out fine-tuning YAML config.")
    p.add_argument("--output-root", default=None)
    p.set_defaults(func=_cmd_finetune)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_sys_logger()
    logger.info("Running post-training analysis: %s", args.command)
    result = args.func(args)
    _emit(result)


if __name__ == "__main__":
    main()
