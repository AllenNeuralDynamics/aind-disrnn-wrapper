from __future__ import annotations

from typing import Any, Mapping, Sequence

import haiku as hk
import jax
import jax.numpy as jnp

_VALID_SESSION_ENCODING_TYPES = {"none", "scalar", "fourier"}
_VALID_SESSION_INTEGRATION_TYPES = {"direct", "pre_mlp"}


def is_session_conditioning_enabled(session_encoding_type: str | None) -> bool:
    """Return True when the encoding type enables session conditioning."""
    return str(session_encoding_type or "none").strip().lower() != "none"


def resolve_session_conditioning_config(
    *,
    multisubject: bool,
    session_encoding_type: str | None,
    session_integration_type: str | None,
    session_fourier_k: int | None,
    session_max_index_by_subject_index: Sequence[int] | None,
    max_n_subjects: int | None,
    context: str,
) -> dict[str, Any]:
    """Validate and normalize session-conditioning config values."""
    encoding_type = str(session_encoding_type or "none").strip().lower()
    integration_type = str(session_integration_type or "direct").strip().lower()
    fourier_k = int(session_fourier_k if session_fourier_k is not None else 4)
    enabled = is_session_conditioning_enabled(encoding_type)

    if encoding_type not in _VALID_SESSION_ENCODING_TYPES:
        raise ValueError(
            f"{context} session_encoding_type must be one of "
            f"{sorted(_VALID_SESSION_ENCODING_TYPES)}, got {encoding_type!r}."
        )
    if integration_type not in _VALID_SESSION_INTEGRATION_TYPES:
        raise ValueError(
            f"{context} session_integration_type must be one of "
            f"{sorted(_VALID_SESSION_INTEGRATION_TYPES)}, got {integration_type!r}."
        )
    if encoding_type == "fourier" and fourier_k <= 0:
        raise ValueError(f"{context} session_fourier_k must be > 0 for fourier encoding.")
    if enabled and not multisubject:
        raise ValueError(f"{context} session conditioning requires multisubject mode.")

    if not enabled:
        return {
            "enabled": False,
            "session_encoding_type": "none",
            "session_integration_type": integration_type,
            "session_fourier_k": fourier_k,
            "session_max_index_by_subject_index": (),
        }

    if max_n_subjects is None or int(max_n_subjects) <= 0:
        raise ValueError(
            f"{context} session conditioning requires a positive max_n_subjects value."
        )
    if session_max_index_by_subject_index is None:
        raise ValueError(
            f"{context} session conditioning requires session_max_index_by_subject_index."
        )

    resolved_session_max = tuple(
        int(value) for value in session_max_index_by_subject_index
    )
    if len(resolved_session_max) != int(max_n_subjects):
        raise ValueError(
            f"{context} session_max_index_by_subject_index must have length "
            f"{int(max_n_subjects)}, got {len(resolved_session_max)}."
        )
    if any(value <= 0 for value in resolved_session_max):
        raise ValueError(
            f"{context} session_max_index_by_subject_index must contain only positive integers."
        )

    return {
        "enabled": True,
        "session_encoding_type": encoding_type,
        "session_integration_type": integration_type,
        "session_fourier_k": fourier_k,
        "session_max_index_by_subject_index": resolved_session_max,
    }


def resolve_session_conditioning_from_architecture(
    *,
    architecture: Mapping[str, Any],
    metadata: Mapping[str, Any],
    multisubject: bool,
    max_n_subjects: int | None,
    context: str,
) -> dict[str, Any]:
    """Read and validate session-conditioning config from architecture plus metadata."""
    return resolve_session_conditioning_config(
        multisubject=multisubject,
        session_encoding_type=architecture.get("session_encoding_type", "none"),
        session_integration_type=architecture.get("session_integration_type", "direct"),
        session_fourier_k=architecture.get("session_fourier_k", 4),
        session_max_index_by_subject_index=metadata.get("session_max_index_by_subject_index"),
        max_n_subjects=max_n_subjects,
        context=context,
    )


def build_session_feat(
    *,
    subject_idx: jnp.ndarray,
    session_idx: jnp.ndarray,
    session_max_index_by_subject: jnp.ndarray,
    encoding_type: str,
    fourier_k: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return deterministic session features plus the valid-session mask."""
    safe_subject_idx = jnp.where(subject_idx >= 0, subject_idx, 0)
    subject_session_max = jnp.take(session_max_index_by_subject, safe_subject_idx, axis=0)
    valid_session_mask = jnp.logical_and(subject_idx >= 0, session_idx >= 1)
    bounded_session_max = jnp.maximum(subject_session_max.astype(jnp.float32), 1.0)
    phase = session_idx.astype(jnp.float32) / bounded_session_max

    if encoding_type == "scalar":
        session_feat = phase[..., None]
    elif encoding_type == "fourier":
        ks = jnp.arange(1, int(fourier_k) + 1, dtype=jnp.float32)
        angles = 2.0 * jnp.pi * phase[..., None] * ks
        session_feat = jnp.concatenate((jnp.sin(angles), jnp.cos(angles)), axis=-1)
    else:
        raise ValueError(f"Unsupported session_encoding_type={encoding_type!r}.")

    session_feat = session_feat * valid_session_mask[..., None].astype(session_feat.dtype)
    return session_feat, valid_session_mask


def apply_session_conditioning(
    *,
    subject_emb: jnp.ndarray,
    session_feat: jnp.ndarray,
    valid_session_mask: jnp.ndarray,
    d_subj: int,
    integration_type: str,
) -> jnp.ndarray:
    """Add a learned session-conditioned perturbation to the subject embedding."""
    conditioned_session_feat = session_feat
    if integration_type == "pre_mlp":
        conditioned_session_feat = jax.nn.relu(
            hk.Linear(int(d_subj), name="session_pre_mlp")(conditioned_session_feat)
        )
    elif integration_type != "direct":
        raise ValueError(f"Unsupported session_integration_type={integration_type!r}.")

    delta_inputs = jnp.concatenate((subject_emb, conditioned_session_feat), axis=-1)
    hidden = jax.nn.relu(
        hk.Linear(int(d_subj) * 2, name="session_delta_hidden")(delta_inputs)
    )
    delta = hk.Linear(
        int(d_subj),
        name="session_delta_out",
        w_init=hk.initializers.Constant(0.0),
        b_init=hk.initializers.Constant(0.0),
    )(hidden)
    delta = delta * valid_session_mask[..., None].astype(delta.dtype)
    return subject_emb + delta
