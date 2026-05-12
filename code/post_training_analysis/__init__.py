"""Public entrypoints for standalone post-training analysis."""

from post_training_analysis.baseline_rl_analysis import (
    run_baseline_rl_post_training_analysis,
)
from post_training_analysis.generative_analysis import (
    ResolvedModelRun,
    compute_history_dependent_switch_stats,
    compute_switch_stats,
    load_animal_session_history,
    resolve_model_run,
    run_post_training_analysis,
    run_post_training_analysis_from_histories,
    run_post_training_analysis_from_saved_histories,
    simulate_model_sessions,
)
from post_training_analysis.heldout_finetuning import (
    run_heldout_subject_finetuning_from_config,
)
from post_training_analysis.likelihood_comparison import (
    run_prediction_likelihood_comparison,
)

__all__ = [
    "ResolvedModelRun",
    "run_baseline_rl_post_training_analysis",
    "compute_history_dependent_switch_stats",
    "compute_switch_stats",
    "load_animal_session_history",
    "resolve_model_run",
    "run_heldout_subject_finetuning_from_config",
    "run_post_training_analysis",
    "run_post_training_analysis_from_histories",
    "run_post_training_analysis_from_saved_histories",
    "run_prediction_likelihood_comparison",
    "simulate_model_sessions",
]
