from __future__ import annotations

from typing import Callable

import haiku as hk


def make_gru_network(
    *,
    hidden_size: int,
    output_size: int,
) -> Callable[[], hk.DeepRNN]:
    """Build the single-layer GRU used by the upstream training notebook."""
    if int(hidden_size) <= 0:
        raise ValueError("architecture.hidden_size must be > 0 for GRU models.")
    if int(output_size) <= 0:
        raise ValueError("output_size must be > 0 for GRU models.")

    hidden_size = int(hidden_size)
    output_size = int(output_size)

    def _make_network() -> hk.DeepRNN:
        return hk.DeepRNN(
            [
                hk.GRU(hidden_size),
                hk.Linear(output_size=output_size),
            ]
        )

    return _make_network
