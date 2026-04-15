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

__all__ = [
    "ResolvedModelRun",
    "run_baseline_rl_post_training_analysis",
    "compute_history_dependent_switch_stats",
    "compute_switch_stats",
    "load_animal_session_history",
    "resolve_model_run",
    "run_post_training_analysis",
    "run_post_training_analysis_from_histories",
    "run_post_training_analysis_from_saved_histories",
    "simulate_model_sessions",
]
