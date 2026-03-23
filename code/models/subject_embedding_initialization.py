from __future__ import annotations

import haiku as hk
import numpy as np


def make_subject_embedding_initializer(
    *,
    subject_embedding_init: str,
    max_n_subjects: int,
    subject_embedding_size: int,
):
    """Return the initializer for multisubject model embedding tables."""
    init_mode = str(subject_embedding_init).strip().lower()
    if init_mode == "zeros":
        return hk.initializers.Constant(0.0)
    if init_mode == "small_random":
        return hk.initializers.RandomNormal(
            stddev=1.0 / np.sqrt(max(1, int(subject_embedding_size)))
        )
    if init_mode in {"subject_count_scaled_random", "legacy_random"}:
        return hk.initializers.RandomNormal(
            stddev=1.0 / np.sqrt(max(1, int(max_n_subjects)))
        )
    raise ValueError(
        "Unsupported subject_embedding_init for multisubject models: "
        f"{subject_embedding_init!r}. "
        "Expected one of: 'zeros', 'small_random', 'subject_count_scaled_random'."
    )
