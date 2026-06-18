"""Deprecated location. Canonical module: :mod:`evaluation.baseline_rl_evaluation`.

This shim re-exports every public and private name from
``evaluation.baseline_rl_evaluation`` so existing ``from utils.baseline_rl_evaluation
import ...`` call sites (baseline_rl_trainer, run_capsule, post_training_analysis) keep
working. Prefer importing from ``evaluation.baseline_rl_evaluation`` in new code.
"""

from __future__ import annotations

import evaluation.baseline_rl_evaluation as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})

del _impl
