"""Deprecated location. Canonical module: :mod:`evaluation.disrnn_evaluation`.

This shim re-exports every public and private name from ``evaluation.disrnn_evaluation``
so existing ``from utils.disrnn_evaluation import ...`` call sites (trainers, run_capsule,
post_training_analysis) keep working. Prefer importing from ``evaluation.disrnn_evaluation``
in new code.
"""

from __future__ import annotations

import evaluation.disrnn_evaluation as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})

del _impl
