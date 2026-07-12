"""Deprecated location. Canonical module: :mod:`evaluation.plotting`.

This shim re-exports every public and private name from ``evaluation.plotting`` so
existing ``from utils.disrnn_plotting import ...`` call sites keep working. Prefer
importing from ``evaluation.plotting`` in new code.
"""

from __future__ import annotations

import evaluation.plotting as _impl

globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})

del _impl
