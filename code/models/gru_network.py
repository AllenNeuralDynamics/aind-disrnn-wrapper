from __future__ import annotations

from typing import Callable, Sequence

import haiku as hk
import jax.numpy as jnp
from models.session_conditioning import (
    apply_session_conditioning,
    build_session_feat,
    resolve_session_conditioning_config,
)
from models.subject_embedding_initialization import make_subject_embedding_initializer


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
        session_encoding_type: str = "none",
        session_integration_type: str = "direct",
        session_fourier_k: int = 4,
        session_max_index_by_subject_index: Sequence[int] | None = None,
        name: str = "multisubject_gru",
    ) -> None:
        super().__init__(name=name)
        self._hidden_size = int(hidden_size)
        self._output_size = int(output_size)
        self._max_n_subjects = int(max_n_subjects)
        self._subject_embedding_size = int(subject_embedding_size)
        self._subject_embedding_init = make_subject_embedding_initializer(
            subject_embedding_init=subject_embedding_init,
            max_n_subjects=self._max_n_subjects,
            subject_embedding_size=self._subject_embedding_size,
        )
        session_cfg = resolve_session_conditioning_config(
            multisubject=True,
            session_encoding_type=session_encoding_type,
            session_integration_type=session_integration_type,
            session_fourier_k=session_fourier_k,
            session_max_index_by_subject_index=session_max_index_by_subject_index,
            max_n_subjects=self._max_n_subjects,
            context="Multisubject GRU",
        )
        self._session_conditioning_enabled = bool(session_cfg["enabled"])
        self._session_encoding_type = str(session_cfg["session_encoding_type"])
        self._session_integration_type = str(session_cfg["session_integration_type"])
        self._session_fourier_k = int(session_cfg["session_fourier_k"])
        self._session_max_index_by_subject_index = tuple(
            int(value) for value in session_cfg["session_max_index_by_subject_index"]
        )
        self._gru = hk.GRU(self._hidden_size, name="gru")
        self._readout = hk.Linear(output_size=self._output_size, name="readout")

    def __call__(self, inputs: jnp.ndarray, prev_state: jnp.ndarray):
        subject_ids = jnp.asarray(inputs[..., 0], dtype=jnp.int32)
        if self._session_conditioning_enabled:
            session_ids = jnp.asarray(inputs[..., 1], dtype=jnp.int32)
            observations = inputs[..., 2:]
        else:
            session_ids = None
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
        subject_context = subject_embeddings

        if self._session_conditioning_enabled:
            session_feat, valid_session_mask = build_session_feat(
                subject_idx=subject_ids,
                session_idx=session_ids,
                session_max_index_by_subject=jnp.asarray(
                    self._session_max_index_by_subject_index,
                    dtype=jnp.int32,
                ),
                encoding_type=self._session_encoding_type,
                fourier_k=self._session_fourier_k,
            )
            subject_context = apply_session_conditioning(
                subject_emb=subject_embeddings,
                session_feat=session_feat,
                valid_session_mask=valid_session_mask,
                d_subj=self._subject_embedding_size,
                integration_type=self._session_integration_type,
            )

        gru_inputs = jnp.concatenate((observations, subject_context), axis=-1)
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
    session_encoding_type: str = "none",
    session_integration_type: str = "direct",
    session_fourier_k: int = 4,
    session_max_index_by_subject_index: Sequence[int] | None = None,
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
        session_cfg = resolve_session_conditioning_config(
            multisubject=True,
            session_encoding_type=session_encoding_type,
            session_integration_type=session_integration_type,
            session_fourier_k=session_fourier_k,
            session_max_index_by_subject_index=session_max_index_by_subject_index,
            max_n_subjects=max_n_subjects,
            context="GRU network factory",
        )
    else:
        session_cfg = resolve_session_conditioning_config(
            multisubject=False,
            session_encoding_type=session_encoding_type,
            session_integration_type=session_integration_type,
            session_fourier_k=session_fourier_k,
            session_max_index_by_subject_index=session_max_index_by_subject_index,
            max_n_subjects=max_n_subjects,
            context="GRU network factory",
        )

    def _make_network() -> hk.RNNCore:
        if multisubject:
            return MultisubjectGru(
                hidden_size=hidden_size,
                output_size=output_size,
                max_n_subjects=max_n_subjects,
                subject_embedding_size=subject_embedding_size,
                subject_embedding_init=subject_embedding_init,
                session_encoding_type=str(session_cfg["session_encoding_type"]),
                session_integration_type=str(session_cfg["session_integration_type"]),
                session_fourier_k=int(session_cfg["session_fourier_k"]),
                session_max_index_by_subject_index=tuple(
                    int(value)
                    for value in session_cfg["session_max_index_by_subject_index"]
                ),
            )
        return hk.DeepRNN(
            [
                hk.GRU(hidden_size),
                hk.Linear(output_size=output_size),
            ]
        )

    return _make_network
