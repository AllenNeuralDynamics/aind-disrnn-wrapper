from __future__ import annotations

from typing import Callable

import haiku as hk
import jax.numpy as jnp
import numpy as np


def _subject_embedding_initializer(
    *,
    subject_embedding_init: str,
    max_n_subjects: int,
    subject_embedding_size: int,
):
    """Return the initializer for multisubject GRU subject embeddings."""
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
        "Unsupported subject_embedding_init for multisubject GRU: "
        f"{subject_embedding_init!r}. "
        "Expected one of: 'zeros', 'small_random', 'subject_count_scaled_random'."
    )


class MultisubjectGru(hk.RNNCore):
    """GRU core with a learned embedding row per subject."""

    def __init__(
        self,
        *,
        hidden_size: int,
        output_size: int,
        max_n_subjects: int,
        subject_embedding_size: int,
        subject_embedding_init: str,
        name: str = "multisubject_gru",
    ) -> None:
        super().__init__(name=name)
        self._hidden_size = int(hidden_size)
        self._output_size = int(output_size)
        self._max_n_subjects = int(max_n_subjects)
        self._subject_embedding_size = int(subject_embedding_size)
        self._subject_embedding_init = _subject_embedding_initializer(
            subject_embedding_init=subject_embedding_init,
            max_n_subjects=self._max_n_subjects,
            subject_embedding_size=self._subject_embedding_size,
        )
        self._gru = hk.GRU(self._hidden_size, name="gru")
        self._readout = hk.Linear(output_size=self._output_size, name="readout")

    def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
        subject_ids = jnp.asarray(inputs[..., 0], dtype=jnp.int32)
        observations = inputs[..., 1:]

        subject_embeddings_table = hk.get_parameter(
            "subject_embeddings",
            (self._max_n_subjects, self._subject_embedding_size),
            init=self._subject_embedding_init,
        )
        valid_subject_ids = jnp.logical_and(
            subject_ids >= 0,
            subject_ids < self._max_n_subjects,
        )
        safe_subject_ids = jnp.where(valid_subject_ids, subject_ids, 0)
        subject_embeddings = jnp.take(subject_embeddings_table, safe_subject_ids, axis=0)
        subject_embeddings = subject_embeddings * jnp.expand_dims(
            valid_subject_ids.astype(subject_embeddings.dtype),
            axis=-1,
        )

        gru_inputs = jnp.concatenate((observations, subject_embeddings), axis=-1)
        gru_output, next_state = self._gru(gru_inputs, prev_state)
        logits = self._readout(gru_output)
        return logits, next_state

    def initial_state(self, batch_size: int | None):
        return self._gru.initial_state(batch_size)


def make_gru_network(
    *,
    hidden_size: int,
    output_size: int,
    multisubject: bool = False,
    max_n_subjects: int | None = None,
    subject_embedding_size: int | None = None,
    subject_embedding_init: str = "zeros",
) -> Callable[[], hk.RNNCore]:
    """Build the single-layer GRU used by the upstream training notebook."""
    if int(hidden_size) <= 0:
        raise ValueError("architecture.hidden_size must be > 0 for GRU models.")
    if int(output_size) <= 0:
        raise ValueError("output_size must be > 0 for GRU models.")

    hidden_size = int(hidden_size)
    output_size = int(output_size)
    multisubject = bool(multisubject)

    if multisubject:
        if max_n_subjects is None or int(max_n_subjects) <= 0:
            raise ValueError("max_n_subjects must be > 0 for multisubject GRU models.")
        if subject_embedding_size is None or int(subject_embedding_size) <= 0:
            raise ValueError(
                "subject_embedding_size must be > 0 for multisubject GRU models."
            )
        max_n_subjects = int(max_n_subjects)
        subject_embedding_size = int(subject_embedding_size)
        subject_embedding_init = str(subject_embedding_init)

    def _make_network() -> hk.RNNCore:
        if multisubject:
            return MultisubjectGru(
                hidden_size=hidden_size,
                output_size=output_size,
                max_n_subjects=max_n_subjects,
                subject_embedding_size=subject_embedding_size,
                subject_embedding_init=subject_embedding_init,
            )
        return hk.DeepRNN(
            [
                hk.GRU(hidden_size),
                hk.Linear(output_size=output_size),
            ]
        )

    return _make_network
