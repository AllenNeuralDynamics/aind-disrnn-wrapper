"""Environment-variable resolution for the ``BFM_*`` prefix.

The project renamed from ``disrnn`` to ``behavior-fm`` (ADR-0007 in the
dispatcher), and the env-var prefix ``DISRNN_*`` renames to ``BFM_*`` with it.
The prefix is a contract that crosses the repo boundary — the dispatcher writes
these variables, the wrapper reads them — so it migrates expand-contract rather
than as a synchronised two-repo deploy:

1. **expand** (this module): the wrapper accepts ``BFM_*`` and still honours
   ``DISRNN_*``;
2. **migrate**: the dispatcher switches to emitting ``BFM_*``;
3. **contract**: the wrapper drops the legacy read.

Until step 3, a job launched from an older dispatcher SHA keeps working: every
read goes through :func:`get_env`, which prefers ``BFM_*`` and falls back to the
matching ``DISRNN_*`` with a one-time deprecation warning naming the variable.
"""

from __future__ import annotations

import os
import warnings

PREFIX = "BFM_"
LEGACY_PREFIX = "DISRNN_"

# Variables whose legacy name has already been reported, so a read in a loop
# warns once rather than once per call.
_warned: set[str] = set()


def legacy_name(name: str) -> str:
    """Return the ``DISRNN_*`` spelling of a ``BFM_*`` variable name."""
    if not name.startswith(PREFIX):
        raise ValueError(f"expected a {PREFIX}* variable name, got {name!r}")
    return LEGACY_PREFIX + name[len(PREFIX) :]


def get_env(name: str, default: str | None = None) -> str | None:
    """Read ``name``, falling back to its legacy ``DISRNN_*`` spelling.

    ``BFM_*`` wins when both are set. Reading the legacy name emits a
    ``DeprecationWarning`` once per variable per process. An empty string is a
    set value, not an absent one — callers that treat "" as unset did so before
    this indirection too.
    """
    value = os.environ.get(name)
    if value is not None:
        return value

    legacy = legacy_name(name)
    value = os.environ.get(legacy)
    if value is not None:
        if legacy not in _warned:
            _warned.add(legacy)
            warnings.warn(
                f"{legacy} is deprecated and will be removed; use {name} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        return value

    return default


def _reset_deprecation_warnings() -> None:
    """Clear the once-per-variable warning memo. For tests."""
    _warned.clear()
