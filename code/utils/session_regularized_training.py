"""Training helpers for exact zero-mean session-delta regularization."""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal, Mapping, Sequence

import haiku as hk
import jax
from jax.example_libraries import optimizers
import jax.numpy as jnp
import numpy as np
import optax

from disentangled_rnns.library import rnn_utils

logger = logging.getLogger(__name__)

ProgressMode = Literal["print", "log", "wandb", "none"]
SessionRegularizationApply = Callable[[Any, jax.Array, float | jax.Array], jnp.ndarray]
SessionCurriculumSchedule = Callable[[int], float]


def _extract_dataset_arrays(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    data = dataset.get_all()
    if isinstance(data, dict):
        return np.asarray(data["xs"]), np.asarray(data["ys"])
    xs, ys = data
    return np.asarray(xs), np.asarray(ys)


def _length_bucket_cache(xs_all: np.ndarray, grid: int) -> tuple[list[int], dict[int, np.ndarray], np.ndarray]:
    """Group session columns into grid-rounded length buckets.

    Padding timesteps are marked by feature 0 < 0 (same convention as
    ``prepend_session_index_to_multisubject_dataset``). Each session's real
    length is rounded UP to the next multiple of ``grid`` (capped at T_max);
    sessions sharing a rounded length form one bucket. Returns the sorted
    bucket lengths, the column indices per bucket, and per-bucket weights
    (proportional to #sessions, so sampling stays ~uniform over sessions).
    """
    t_max = int(xs_all.shape[0])
    grid = max(1, int(grid))
    valid = np.asarray(xs_all[:, :, 0]) >= 0
    lengths = valid.sum(axis=0).astype(int)
    rounded = np.minimum(((np.maximum(lengths, 1) + grid - 1) // grid) * grid, t_max)
    uniq = sorted(int(b) for b in set(rounded.tolist()))
    pools = {b: np.where(rounded == b)[0] for b in uniq}
    sizes = np.array([len(pools[b]) for b in uniq], dtype=float)
    return uniq, pools, sizes / sizes.sum()


def _sample_batch(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    xs_all, ys_all = _extract_dataset_arrays(dataset)
    n_episodes = int(xs_all.shape[1])
    batch_mode = str(getattr(dataset, "batch_mode", "single"))
    batch_size = getattr(dataset, "batch_size", None)

    if batch_mode == "single" or batch_size is None:
        return xs_all, ys_all

    batch_size = int(batch_size)
    if batch_size == 0:
        return xs_all[:, :0], ys_all[:, :0]

    rng = getattr(dataset, "rng", None)
    use_rng = rng if (rng is not None and hasattr(rng, "choice")) else np.random

    # Length-bucketed sampling (opt-in via dataset.length_bucketing): draw the
    # batch from a single length bucket and TRIM the unroll to that bucket's
    # grid-rounded length, so short sessions don't pay the global-T_max padding
    # cost. Trimming only drops all-padding rows (real length <= bucket length),
    # so the loss/mask are unaffected. Distinct trims = #buckets (~T_max/grid),
    # i.e. that many JAX recompiles, amortized over training.
    if batch_mode == "random" and bool(getattr(dataset, "length_bucketing", False)):
        grid = int(getattr(dataset, "length_bucket_grid", 128))
        cache = getattr(dataset, "_length_bucket_cache", None)
        if cache is None:
            cache = _length_bucket_cache(xs_all, grid)
            setattr(dataset, "_length_bucket_cache", cache)
        uniq, pools, weights = cache
        bucket = int(use_rng.choice(np.asarray(uniq), p=weights))
        indices = use_rng.choice(pools[bucket], size=batch_size)
        return xs_all[:bucket, indices], ys_all[:bucket, indices]

    if batch_mode == "rolling":
        current_start = int(getattr(dataset, "_current_start_index", 0))
        indices = np.arange(current_start, current_start + batch_size) % n_episodes
        setattr(dataset, "_current_start_index", (current_start + batch_size) % n_episodes)
    elif batch_mode == "random":
        indices = use_rng.choice(n_episodes, size=batch_size)
    else:
        raise ValueError(
            f"Unsupported dataset.batch_mode='{batch_mode}' for session-regularized training."
        )

    return xs_all[:, indices], ys_all[:, indices]


def _resolve_penalty_scale(loss_param: Mapping[str, Any] | float | int) -> float:
    if isinstance(loss_param, Mapping):
        if "penalty_scale" in loss_param:
            return float(loss_param["penalty_scale"])
        if "value" in loss_param:
            return float(loss_param["value"])
        return 1.0
    return float(loss_param)


def _categorical_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    if targets.ndim != 3 or targets.shape[-1] != 1:
        raise ValueError(
            "Session-regularized categorical training expects targets shaped "
            f"[timesteps, episodes, 1], got {targets.shape}."
        )
    raw_targets = jnp.squeeze(targets, axis=-1)
    valid_mask = jnp.logical_and(jnp.isfinite(raw_targets), raw_targets != -1)
    safe_targets = jnp.where(valid_mask, raw_targets, 0).astype(jnp.int32)
    per_sample_loss = optax.softmax_cross_entropy_with_integer_labels(logits, safe_targets)
    masked_loss = jnp.where(valid_mask, per_sample_loss, 0.0)
    n_valid = jnp.maximum(1, jnp.sum(valid_mask))
    return jnp.sum(masked_loss) / n_valid, valid_mask


def _average_penalty(
    model_output: jnp.ndarray,
    valid_mask: jnp.ndarray,
) -> jnp.ndarray:
    penalties = model_output[..., -1]
    masked_penalties = jnp.where(valid_mask, penalties, 0.0)
    n_valid = jnp.maximum(1, jnp.sum(valid_mask))
    return jnp.sum(masked_penalties) / n_valid


def build_zero_mean_session_delta_regularization_apply(
    *,
    make_network: Callable[[float], hk.RNNCore],
    subject_indices: Sequence[int] | np.ndarray,
    session_indices: Sequence[int] | np.ndarray,
    max_n_subjects: int,
) -> SessionRegularizationApply:
    """Build an exact regularizer over the full per-subject session set."""
    flat_subject_indices = np.asarray(subject_indices, dtype=np.int32)
    flat_session_indices = np.asarray(session_indices, dtype=np.int32)
    if flat_subject_indices.ndim != 1 or flat_session_indices.ndim != 1:
        raise ValueError("subject_indices and session_indices must be 1D.")
    if flat_subject_indices.shape[0] != flat_session_indices.shape[0]:
        raise ValueError("subject_indices and session_indices must have matching lengths.")
    if int(max_n_subjects) <= 0:
        raise ValueError("max_n_subjects must be > 0 for session regularization.")
    if flat_subject_indices.shape[0] == 0:
        raise ValueError("Session regularization requires at least one subject/session pair.")

    subject_indices_jnp = jnp.asarray(flat_subject_indices, dtype=jnp.int32)
    session_indices_jnp = jnp.asarray(flat_session_indices, dtype=jnp.int32)
    max_n_subjects = int(max_n_subjects)

    def _regularization_forward(
        regularization_subject_indices: jnp.ndarray,
        regularization_session_indices: jnp.ndarray,
        session_curriculum_lambda: jnp.ndarray,
    ) -> jnp.ndarray:
        core = make_network(session_curriculum_lambda)
        if not hasattr(core, "compute_session_delta"):
            raise ValueError(
                "Session regularization requires a network with compute_session_delta()."
            )
        session_deltas = core.compute_session_delta(
            subject_ids=regularization_subject_indices,
            session_ids=regularization_session_indices,
        )
        delta_sums = jnp.zeros(
            (max_n_subjects, session_deltas.shape[-1]),
            dtype=session_deltas.dtype,
        )
        delta_sums = delta_sums.at[regularization_subject_indices].add(session_deltas)
        counts = jnp.zeros((max_n_subjects,), dtype=session_deltas.dtype)
        counts = counts.at[regularization_subject_indices].add(1.0)
        mean_deltas = delta_sums / jnp.maximum(counts[:, None], 1.0)
        mean_deltas = mean_deltas * (counts > 0).astype(session_deltas.dtype)[:, None]
        return jnp.sum(jnp.square(mean_deltas))

    regularization_model = hk.transform(_regularization_forward)
    apply_regularization = jax.jit(regularization_model.apply)

    def _apply(
        current_params: Any,
        key: jax.Array,
        session_curriculum_lambda: float,
    ) -> jnp.ndarray:
        return apply_regularization(
            current_params,
            key,
            subject_indices_jnp,
            session_indices_jnp,
            jnp.asarray(session_curriculum_lambda, dtype=jnp.float32),
        )

    return _apply


def train_network_with_session_regularization(
    make_network: Callable[[float], hk.RNNCore],
    training_dataset: Any,
    validation_dataset: Any | None,
    *,
    loss: str,
    loss_param: Mapping[str, Any] | float | int,
    n_action_logits: int,
    session_regularization_apply: SessionRegularizationApply | None,
    session_regularization_scale: float,
    session_curriculum_lambda_schedule: SessionCurriculumSchedule | None = None,
    opt: optax.GradientTransformation = optax.adam(1e-3),
    parameter_update_mask: Any | None = None,
    random_key: jax.Array | None = None,
    opt_state: optax.OptState | None = None,
    params: Any | None = None,
    n_steps: int = 1000,
    max_grad_norm: float = 1.0,
    log_losses_every: int = 10,
    report_progress_by: ProgressMode = "print",
    wandb_run: Any | None = None,
    wandb_step_offset: int = 0,
) -> tuple[Any, optax.OptState, dict[str, np.ndarray]]:
    """Train an RNN with an exact zero-mean session-delta penalty."""
    resolved_loss = str(loss).strip().lower()
    if resolved_loss not in {"categorical", "penalized_categorical"}:
        raise ValueError(
            "Session-regularized training currently supports only "
            f"'categorical' and 'penalized_categorical', got {loss!r}."
        )
    if float(session_regularization_scale) < 0:
        raise ValueError("session_regularization_scale must be >= 0.")

    xs_all, _ = _extract_dataset_arrays(training_dataset)
    initial_session_curriculum_lambda = (
        1.0
        if session_curriculum_lambda_schedule is None
        else float(session_curriculum_lambda_schedule(0))
    )

    def unroll_network(
        inputs: jnp.ndarray,
        session_curriculum_lambda: jnp.ndarray,
    ) -> jnp.ndarray:
        core = make_network(session_curriculum_lambda)
        batch_size = jnp.shape(inputs)[1]
        state = core.initial_state(batch_size)
        outputs, _ = hk.dynamic_unroll(core, inputs, state)
        return outputs

    model = hk.transform(unroll_network)

    if random_key is None:
        random_key = jax.random.PRNGKey(0)
    if params is None:
        random_key, key1 = jax.random.split(random_key)
        params = model.init(
            key1,
            xs_all,
            jnp.asarray(initial_session_curriculum_lambda, dtype=jnp.float32),
        )
    if opt_state is None:
        opt_state = opt.init(params)
    if parameter_update_mask is not None:
        parameter_update_mask = jax.tree_util.tree_map(
            lambda leaf: jnp.asarray(leaf),
            parameter_update_mask,
        )

    xs_eval = ys_eval = None
    if validation_dataset is not None:
        xs_eval, ys_eval = _extract_dataset_arrays(validation_dataset)

    penalty_scale = _resolve_penalty_scale(loss_param)

    def compute_loss(
        current_params: Any,
        batch_xs: jnp.ndarray,
        batch_ys: jnp.ndarray,
        key: jax.Array,
        session_curriculum_lambda: jnp.ndarray,
    ) -> jnp.ndarray:
        model_key, regularization_key = jax.random.split(key)
        model_output = model.apply(
            current_params,
            model_key,
            batch_xs,
            session_curriculum_lambda,
        )
        logits = model_output[:, :, : int(n_action_logits)]
        supervised_loss, valid_mask = _categorical_loss(logits, batch_ys)
        total_loss = supervised_loss
        if resolved_loss == "penalized_categorical":
            total_loss = total_loss + float(penalty_scale) * _average_penalty(
                model_output,
                valid_mask,
            )
        if (
            session_regularization_apply is not None
            and float(session_regularization_scale) > 0
        ):
            total_loss = total_loss + float(
                session_regularization_scale
            ) * session_regularization_apply(
                current_params,
                regularization_key,
                session_curriculum_lambda,
            )
        return total_loss

    compute_loss_jit = jax.jit(compute_loss)

    @jax.jit
    def train_step(
        current_params: Any,
        current_opt_state: optax.OptState,
        batch_xs: jnp.ndarray,
        batch_ys: jnp.ndarray,
        key: jax.Array,
        session_curriculum_lambda: jnp.ndarray,
    ) -> tuple[jnp.ndarray, Any, optax.OptState]:
        loss_value, grads = jax.value_and_grad(compute_loss, argnums=0)(
            current_params,
            batch_xs,
            batch_ys,
            key,
            session_curriculum_lambda,
        )
        if parameter_update_mask is not None:
            grads = jax.tree_util.tree_map(
                lambda grad, mask: grad * mask,
                grads,
                parameter_update_mask,
            )
        grads, next_opt_state = opt.update(grads, current_opt_state)
        clipped_grads = optimizers.clip_grads(grads, max_grad_norm)
        next_params = optax.apply_updates(current_params, clipped_grads)
        return loss_value, next_params, next_opt_state

    training_loss: list[float] = []
    validation_loss: list[float] = []
    validation_value = np.nan

    for step in range(int(n_steps)):
        random_key, subkey_train, subkey_validation = jax.random.split(random_key, 3)
        current_session_curriculum_lambda = (
            1.0
            if session_curriculum_lambda_schedule is None
            else float(session_curriculum_lambda_schedule(step))
        )
        batch_xs, batch_ys = _sample_batch(training_dataset)
        loss_value, params, opt_state = train_step(
            params,
            opt_state,
            jnp.asarray(batch_xs),
            jnp.asarray(batch_ys),
            subkey_train,
            jnp.asarray(current_session_curriculum_lambda, dtype=jnp.float32),
        )

        if step % log_losses_every == 0 or step == int(n_steps) - 1:
            if rnn_utils.nan_in_dict(params):
                raise ValueError("NaN in params during session-regularized training.")
            if np.isnan(loss_value):
                raise ValueError("NaN in session-regularized training loss.")
            if loss_value > 1e50:
                raise ValueError("Session-regularized training loss exploded.")

            if validation_dataset is not None and xs_eval is not None and ys_eval is not None:
                validation_value = compute_loss_jit(
                    params,
                    jnp.asarray(xs_eval),
                    jnp.asarray(ys_eval),
                    subkey_validation,
                    jnp.asarray(current_session_curriculum_lambda, dtype=jnp.float32),
                )

            training_loss.append(float(loss_value))
            validation_loss.append(float(validation_value))
            log_str = (
                f"Step {step + 1} of {n_steps}. "
                f"Training Loss: {float(loss_value):.2e}. "
                f"Validation Loss: {float(validation_value):.2e}. "
                f"Session Curriculum Lambda: {current_session_curriculum_lambda:.3f}"
            )
            if report_progress_by == "print":
                print(log_str)
            elif report_progress_by == "log":
                logger.info(log_str)
            elif report_progress_by == "wandb" and wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": float(loss_value),
                        "valid/loss": float(validation_value),
                        "train/session_curriculum_lambda": float(
                            current_session_curriculum_lambda
                        ),
                    },
                    step=wandb_step_offset + step,
                )

    return params, opt_state, {
        "training_loss": np.asarray(training_loss, dtype=float),
        "validation_loss": np.asarray(validation_loss, dtype=float),
    }
