"""Public entrypoints for standalone post-training analysis.

This package is import-independent of model training: importing it (or running an analysis
via ``run_analysis.py``) does not import ``model_trainers``. Shared evaluation primitives live
in the ``evaluation`` package, and trained runs are loaded from saved artifacts via
``resolve_model_run`` (``model_dir/inputs.yaml`` + ``outputs/``).

Two functions are *deliberate, documented training-adjacent exceptions* that lazily import
the trainers at call time (never at import time):

* ``heldout_finetuning.run_heldout_subject_finetuning_from_config`` — genuinely *re-trains*
  held-out subject embeddings, so it legitimately uses ``DisrnnTrainer``/``GruTrainer``.
* ``likelihood_comparison._evaluate_disrnn_dataset`` — for the *multisubject* disRNN case
  only, it borrows ``DisrnnTrainer``'s network-construction helpers (single-subject eval
  uses ``evaluation.disrnn_evaluation``'s own builder). The clean fix is to extract disRNN
  network construction into ``models/disrnn_network.py`` (symmetric with
  ``models/gru_network.py``) and have both the trainer and eval call it; deferred to avoid
  touching training code.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "ResolvedModelRun": (
        "post_training_analysis.generative_analysis",
        "ResolvedModelRun",
    ),
    "run_baseline_rl_post_training_analysis": (
        "post_training_analysis.baseline_rl_analysis",
        "run_baseline_rl_post_training_analysis",
    ),
    "run_baseline_q_space_condition_analysis": (
        "post_training_analysis.likelihood_advantage_analysis",
        "run_baseline_q_space_condition_analysis",
    ),
    "run_baseline_q_space_subject_analysis": (
        "post_training_analysis.likelihood_advantage_analysis",
        "run_baseline_q_space_subject_analysis",
    ),
    "run_embedding_space_analysis": (
        "post_training_analysis.embedding_space_analysis",
        "run_embedding_space_analysis",
    ),
    "run_likelihood_advantage_analysis": (
        "post_training_analysis.likelihood_advantage_analysis",
        "run_likelihood_advantage_analysis",
    ),
    "run_rnn_state_space_condition_analysis": (
        "post_training_analysis.likelihood_advantage_analysis",
        "run_rnn_state_space_condition_analysis",
    ),
    "run_rnn_state_space_subject_analysis": (
        "post_training_analysis.likelihood_advantage_analysis",
        "run_rnn_state_space_subject_analysis",
    ),
    "compute_history_dependent_switch_stats": (
        "post_training_analysis.generative_analysis",
        "compute_history_dependent_switch_stats",
    ),
    "compute_switch_stats": (
        "post_training_analysis.generative_analysis",
        "compute_switch_stats",
    ),
    "load_animal_session_history": (
        "post_training_analysis.generative_analysis",
        "load_animal_session_history",
    ),
    "resolve_model_run": (
        "post_training_analysis.generative_analysis",
        "resolve_model_run",
    ),
    "run_heldout_subject_finetuning_from_config": (
        "post_training_analysis.heldout_finetuning",
        "run_heldout_subject_finetuning_from_config",
    ),
    "run_post_training_analysis": (
        "post_training_analysis.generative_analysis",
        "run_post_training_analysis",
    ),
    "run_post_training_analysis_from_histories": (
        "post_training_analysis.generative_analysis",
        "run_post_training_analysis_from_histories",
    ),
    "run_post_training_analysis_from_saved_histories": (
        "post_training_analysis.generative_analysis",
        "run_post_training_analysis_from_saved_histories",
    ),
    "run_prediction_likelihood_comparison": (
        "post_training_analysis.likelihood_comparison",
        "run_prediction_likelihood_comparison",
    ),
    "simulate_model_sessions": (
        "post_training_analysis.generative_analysis",
        "simulate_model_sessions",
    ),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
