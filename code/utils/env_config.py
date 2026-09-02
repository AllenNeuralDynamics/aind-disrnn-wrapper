"""Environment-variable resolution for the ``BFM_*`` prefix.

The project renamed from ``disrnn`` to ``dynamic-foraging-bfm`` (ADR-0007 in the
dispatcher), and the env-var prefix ``DISRNN_*`` renamed to ``BFM_*`` with it.
The prefix is a contract that crosses the repo boundary — the dispatcher writes
these variables, the wrapper reads them — so it migrated expand-contract rather
than as a synchronised two-repo deploy:

1. **expand**: the wrapper accepted ``BFM_*`` and still honoured ``DISRNN_*``;
2. **migrate**: the dispatcher switched to emitting ``BFM_*``;
3. **contract** (this module's current state): the legacy read is gone.

Step 3 landed only after the migration was verified end to end on Beaker — all
``BFM_META_*`` values reaching W&B through this resolver, with zero deprecation
warnings — and after confirming no in-flight or resumable job was pinned to a
pre-migration dispatcher SHA. A job launched from a dispatcher older than that
migration now silently gets defaults instead of its provenance block and output
directory, which is why the fallback outlived the migration rather than being
dropped alongside it.

:func:`get_env` remains the single choke point for reading these variables, so
the guard test can keep asserting that no scattered ``os.environ`` lookup
reintroduces the legacy prefix.
"""

from __future__ import annotations

import os

PREFIX = "BFM_"


def get_env(name: str, default: str | None = None) -> str | None:
    """Read a ``BFM_*`` environment variable.

    An empty string is a set value, not an absent one — callers that treat ""
    as unset did so before this indirection too.
    """
    if not name.startswith(PREFIX):
        raise ValueError(f"expected a {PREFIX}* variable name, got {name!r}")

    value = os.environ.get(name)
    return default if value is None else value
