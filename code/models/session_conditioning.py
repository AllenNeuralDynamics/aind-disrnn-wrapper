from __future__ import annotations

from typing import Any, Mapping, Sequence

_VALID_SESSION_ENCODING_TYPES = {"none", "scalar", "fourier"}
_VALID_SESSION_INTEGRATION_TYPES = {"direct", "pre_mlp"}
_DEFAULT_SESSION_PRETRAIN_FRACTION = 0.3
_DEFAULT_SESSION_WARMUP_FRACTION = 0.2


def is_session_conditioning_enabled(session_encoding_type: str | None) -> bool:
    """Return True when the encoding type enables session conditioning."""
    return str(session_encoding_type or "none").strip().lower() != "none"


def resolve_session_conditioning_config(
    *,
    multisubject: bool,
    session_encoding_type: str | None,
    session_integration_type: str | None,
    session_fourier_k: int | None,
    session_delta_n_layers: int | None,
    session_delta_hidden_size: int | None,
    session_max_index_by_subject_index: Sequence[int] | None,
    max_n_subjects: int | None,
    context: str,
) -> dict[str, Any]:
    """Validate and normalize session-conditioning config values."""
    encoding_type = str(session_encoding_type or "none").strip().lower()
    integration_type = str(session_integration_type or "direct").strip().lower()
    fourier_k = int(session_fourier_k if session_fourier_k is not None else 4)
    delta_n_layers = int(session_delta_n_layers if session_delta_n_layers is not None else 3)
    delta_hidden_size = int(
        session_delta_hidden_size if session_delta_hidden_size is not None else 16
    )
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
    if delta_n_layers <= 0:
        raise ValueError(f"{context} session_delta_n_layers must be > 0.")
    if delta_hidden_size <= 0:
        raise ValueError(f"{context} session_delta_hidden_size must be > 0.")
    if enabled and not multisubject:
        raise ValueError(f"{context} session conditioning requires multisubject mode.")

    if not enabled:
        return {
            "enabled": False,
            "session_encoding_type": "none",
            "session_integration_type": integration_type,
            "session_fourier_k": fourier_k,
            "session_delta_n_layers": delta_n_layers,
            "session_delta_hidden_size": delta_hidden_size,
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
        "session_delta_n_layers": delta_n_layers,
        "session_delta_hidden_size": delta_hidden_size,
        "session_max_index_by_subject_index": resolved_session_max,
    }


def resolve_session_conditioning_from_architecture(
    *,
    architecture: Mapping[str, Any],
    metadata: Mapping[str, Any],
    multisubject: bool,
    max_n_subjects: int | None,
    subject_embedding_size: int | None = None,
    use_legacy_delta_defaults_when_missing: bool = False,
    context: str,
) -> dict[str, Any]:
    """Read and validate session-conditioning config from architecture plus metadata."""
    session_delta_n_layers = architecture.get("session_delta_n_layers")
    session_delta_hidden_size = architecture.get("session_delta_hidden_size")
    if session_delta_n_layers is None and use_legacy_delta_defaults_when_missing:
        session_delta_n_layers = 1
    if session_delta_hidden_size is None and use_legacy_delta_defaults_when_missing:
        if subject_embedding_size is not None and session_delta_n_layers in (None, 1):
            session_delta_hidden_size = int(subject_embedding_size) * 2
        else:
            session_delta_hidden_size = 16
    return resolve_session_conditioning_config(
        multisubject=multisubject,
        session_encoding_type=architecture.get("session_encoding_type", "none"),
        session_integration_type=architecture.get("session_integration_type", "direct"),
        session_fourier_k=architecture.get("session_fourier_k", 4),
        session_delta_n_layers=session_delta_n_layers,
        session_delta_hidden_size=session_delta_hidden_size,
        session_max_index_by_subject_index=metadata.get("session_max_index_by_subject_index"),
        max_n_subjects=max_n_subjects,
        context=context,
    )


def resolve_session_curriculum_steps(
    *,
    total_training_steps: int | None,
    session_n_pretrain_steps: int | None,
    session_n_warmup_steps: int | None,
    context: str,
) -> dict[str, int]:
    """Resolve the step schedule used to ramp in session conditioning."""
    resolved_total_training_steps = int(
        total_training_steps if total_training_steps is not None else 0
    )
    if resolved_total_training_steps < 0:
        raise ValueError(f"{context} total_training_steps must be >= 0.")

    if session_n_pretrain_steps is None:
        resolved_pretrain_steps = int(
            round(
                resolved_total_training_steps
                * _DEFAULT_SESSION_PRETRAIN_FRACTION
            )
        )
    else:
        resolved_pretrain_steps = int(session_n_pretrain_steps)
    if session_n_warmup_steps is None:
        resolved_warmup_steps = int(
            round(
                resolved_total_training_steps
                * _DEFAULT_SESSION_WARMUP_FRACTION
            )
        )
    else:
        resolved_warmup_steps = int(session_n_warmup_steps)

    if resolved_pretrain_steps < 0:
        raise ValueError(f"{context} session_n_pretrain_steps must be >= 0.")
    if resolved_warmup_steps < 0:
        raise ValueError(f"{context} session_n_warmup_steps must be >= 0.")

    return {
        "session_n_pretrain_steps": resolved_pretrain_steps,
        "session_n_warmup_steps": resolved_warmup_steps,
    }


def compute_session_curriculum_lambda(
    *,
    current_step: int,
    session_n_pretrain_steps: int,
    session_n_warmup_steps: int,
) -> float:
    """Return the current curriculum gate for session conditioning."""
    resolved_current_step = int(current_step)
    resolved_pretrain_steps = int(session_n_pretrain_steps)
    resolved_warmup_steps = int(session_n_warmup_steps)

    if resolved_current_step < 0:
        raise ValueError("current_step must be >= 0.")
    if resolved_pretrain_steps < 0:
        raise ValueError("session_n_pretrain_steps must be >= 0.")
    if resolved_warmup_steps < 0:
        raise ValueError("session_n_warmup_steps must be >= 0.")

    if resolved_current_step < resolved_pretrain_steps:
        return 0.0
    if resolved_warmup_steps <= 0:
        return 1.0
    if resolved_current_step < resolved_pretrain_steps + resolved_warmup_steps:
        return float(
            (resolved_current_step - resolved_pretrain_steps)
            / resolved_warmup_steps
        )
    return 1.0


def build_session_feat(
    *,
    subject_idx: jnp.ndarray,
    session_idx: jnp.ndarray,
    session_max_index_by_subject: jnp.ndarray,
    encoding_type: str,
    fourier_k: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return deterministic session features plus the valid-session mask."""
    import jax.numpy as jnp

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


def compute_session_delta(
    *,
    subject_emb: jnp.ndarray,
    session_feat: jnp.ndarray,
    valid_session_mask: jnp.ndarray,
    d_subj: int,
    integration_type: str,
    delta_n_layers: int,
    delta_hidden_size: int,
    curriculum_lambda: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Return the learned session-conditioned perturbation for a subject embedding."""
    import haiku as hk
    import jax
    import jax.numpy as jnp

    conditioned_session_feat = session_feat
    if integration_type == "pre_mlp":
        conditioned_session_feat = jax.nn.relu(
            hk.Linear(int(d_subj), name="session_pre_mlp")(conditioned_session_feat)
        )
    elif integration_type != "direct":
        raise ValueError(f"Unsupported session_integration_type={integration_type!r}.")

    delta_inputs = jnp.concatenate((subject_emb, conditioned_session_feat), axis=-1)
    hidden = delta_inputs
    for layer_index in range(int(delta_n_layers)):
        layer_name = (
            "session_delta_hidden"
            if layer_index == 0
            else f"session_delta_hidden_{layer_index}"
        )
        hidden = jax.nn.relu(
            hk.Linear(int(delta_hidden_size), name=layer_name)(hidden)
        )
    delta = hk.Linear(
        int(d_subj),
        name="session_delta_out",
        w_init=hk.initializers.Constant(0.0),
        b_init=hk.initializers.Constant(0.0),
    )(hidden)
    valid_delta = delta * valid_session_mask[..., None].astype(delta.dtype)
    curriculum_scale = jnp.asarray(curriculum_lambda, dtype=valid_delta.dtype)
    return curriculum_scale * valid_delta


def apply_session_conditioning(
    *,
    subject_emb: jnp.ndarray,
    session_feat: jnp.ndarray,
    valid_session_mask: jnp.ndarray,
    d_subj: int,
    integration_type: str,
    delta_n_layers: int,
    delta_hidden_size: int,
    curriculum_lambda: float | jnp.ndarray = 1.0,
) -> jnp.ndarray:
    """Add a learned session-conditioned perturbation to the subject embedding."""
    delta = compute_session_delta(
        subject_emb=subject_emb,
        session_feat=session_feat,
        valid_session_mask=valid_session_mask,
        d_subj=d_subj,
        integration_type=integration_type,
        delta_n_layers=delta_n_layers,
        delta_hidden_size=delta_hidden_size,
        curriculum_lambda=curriculum_lambda,
    )
    return subject_emb + delta
