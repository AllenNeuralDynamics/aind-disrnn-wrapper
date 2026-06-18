"""Deprecated location. Canonical module: :mod:`evaluation.gru_evaluation`.

This shim re-exports every public and private name from ``evaluation.gru_evaluation`` so
existing ``from utils.gru_evaluation import ...`` call sites keep working. Prefer importing
from ``evaluation.gru_evaluation`` in new code.
"""

from __future__ import annotations

import evaluation.gru_evaluation as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})

del _impl
