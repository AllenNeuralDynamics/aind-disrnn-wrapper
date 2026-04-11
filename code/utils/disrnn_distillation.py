"""Utilities for distilling GRU teachers into disRNN students."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import haiku as hk
import jax
from jax.example_libraries import optimizers
import jax.numpy as jnp
import numpy as np
import optax

from disentangled_rnns.library import rnn_utils

from models.gru_network import make_gru_network
from utils.multisubject import (
    extract_subject_embeddings_from_params,
    load_subject_index_map,
    normalize_subject_id,
)

logger = logging.getLogger(__name__)


def _candidate_parent(depth: int) -> Path | None:
    parents = Path(__file__).resolve().parents
    return parents[depth] / "data" if depth < len(parents) else None


_CANDIDATE_DATA_DIRS: tuple[Path, ...] = tuple(
    path
    for path in (
        Path("/data"),
        _candidate_parent(2),
        _candidate_parent(3),
        Path(os.environ["DATA_PATH"]) if "DATA_PATH" in os.environ else None,
    )
    if path is not None
)

DistillationProgressMode = Literal["print", "log", "wandb", "none"]


@dataclass(frozen=True)
class DistillationConfig:
    """Resolved distillation configuration."""

    enabled: bool = False
    teacher_model_dirs: tuple[str, ...] = ()
    temperature: float = 2.0
    aggregation: str = "mean_probs"
    loss: str = "kl"
    use_hard_labels: bool = False
    evaluate_against_hard_labels: bool = True
    save_manifest: bool = True

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "DistillationConfig":
        if not config:
            return cls()

        teacher_model_dirs = config.get("teacher_model_dirs", [])
        if teacher_model_dirs is None:
            teacher_model_dirs = []
        if isinstance(teacher_model_dirs, (str, Path)):
            teacher_model_dirs = [str(teacher_model_dirs)]
        teacher_dirs = tuple(str(Path(path).expanduser()) for path in teacher_model_dirs)

        return cls(
            enabled=bool(config.get("enabled", False)),
            teacher_model_dirs=teacher_dirs,
            temperature=float(config.get("temperature", 2.0)),
            aggregation=str(config.get("aggregation", "mean_probs")),
            loss=str(config.get("loss", "kl")),
            use_hard_labels=bool(config.get("use_hard_labels", False)),
            evaluate_against_hard_labels=bool(
                config.get("evaluate_against_hard_labels", True)
            ),
            save_manifest=bool(config.get("save_manifest", True)),
        )


@dataclass(frozen=True)
class TeacherModelSummary:
    """Serializable teacher metadata for saved manifests."""

    model_dir: str
    params_path: str
    config_path: str
    subject_index_map_path: str | None
    multisubject: bool
    output_size: int
    subject_count: int | None


@dataclass(frozen=True)
class DistillationEnsemble:
    """Teacher ensemble targets aligned to the student datasets."""

    config: DistillationConfig
    train_probs: np.ndarray
    eval_probs: np.ndarray
    full_probs: np.ndarray
    manifest: dict[str, Any]


def resolve_distillation_config(
    config: Mapping[str, Any] | None,
) -> DistillationConfig:
    """Normalize a distillation config mapping."""
    resolved = DistillationConfig.from_mapping(config)
    if not resolved.enabled:
        return resolved

    if not resolved.teacher_model_dirs:
        raise ValueError(
            "Distillation requires distillation.teacher_model_dirs to contain at least one GRU run."
        )
    if resolved.temperature <= 0:
        raise ValueError("distillation.temperature must be > 0.")
    if resolved.aggregation != "mean_probs":
        raise ValueError(
            f"Unsupported distillation.aggregation='{resolved.aggregation}'. "
            "Only 'mean_probs' is currently supported."
        )
    if resolved.loss != "kl":
        raise ValueError(
            f"Unsupported distillation.loss='{resolved.loss}'. Only 'kl' is supported."
        )
    if resolved.use_hard_labels:
        raise ValueError(
            "distillation.use_hard_labels=true is not supported in v1. "
            "Teacher-only optimization is the implemented mode."
        )
    return resolved


def resolve_penalty_scale(loss_param: dict[str, float] | float) -> float:
    """Resolve the penalty scale in the same way as upstream rnn_utils."""
    if isinstance(loss_param, Mapping):
        if "penalty_scale" in loss_param:
            return float(loss_param["penalty_scale"])
        if "value" in loss_param:
            return float(loss_param["value"])
        return 1.0
    return float(loss_param)


def aggregate_teacher_probabilities(probabilities: list[np.ndarray]) -> np.ndarray:
    """Aggregate per-teacher probabilities into one ensemble distribution."""
    if not probabilities:
        raise ValueError("probabilities must contain at least one teacher.")
    stacked = np.stack([np.asarray(prob, dtype=np.float32) for prob in probabilities], axis=0)
    aggregated = np.mean(stacked, axis=0, dtype=np.float32)
    totals = np.sum(aggregated, axis=-1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    return aggregated / totals


def remap_multisubject_teacher_inputs(
    xs: np.ndarray,
    *,
    student_index_to_subject_id: Mapping[int, Any] | Mapping[str, Any],
    teacher_subject_id_to_index: Mapping[Any, int],
) -> np.ndarray:
    """Rewrite the prepended student subject index feature into a teacher index space."""
    xs = np.asarray(xs)
    if xs.ndim != 3:
        raise ValueError(f"Expected xs to be 3D, got shape={xs.shape}.")
    if xs.shape[2] < 2:
        raise ValueError(
            "Multisubject teacher remapping requires a prepended subject-index feature."
        )

    remapped = np.array(xs, copy=True)
    subject_ids = np.asarray(np.rint(remapped[..., 0]), dtype=int)
    remapped_subject_ids = np.full_like(remapped[..., 0], fill_value=-1, dtype=remapped.dtype)

    unique_subject_indices = np.unique(subject_ids[subject_ids >= 0]).tolist()
    missing_subject_ids: list[Any] = []
    student_to_teacher: dict[int, int] = {}
    for student_subject_index in unique_subject_indices:
        subject_id = student_index_to_subject_id.get(student_subject_index)
        if subject_id is None:
            subject_id = student_index_to_subject_id.get(str(student_subject_index))
        normalized_subject_id = normalize_subject_id(subject_id)
        if normalized_subject_id not in teacher_subject_id_to_index:
            missing_subject_ids.append(normalized_subject_id)
            continue
        student_to_teacher[int(student_subject_index)] = int(
            teacher_subject_id_to_index[normalized_subject_id]
        )

    if missing_subject_ids:
        raise ValueError(
            "Teacher subject map is missing student subject ids required for distillation: "
            f"{missing_subject_ids}"
        )

    for student_subject_index, teacher_subject_index in student_to_teacher.items():
        remapped_subject_ids[subject_ids == student_subject_index] = teacher_subject_index

    remapped[..., 0] = remapped_subject_ids.astype(remapped.dtype)
    return remapped


def _extract_dataset_arrays(dataset: Any) -> tuple[np.ndarray, np.ndarray]:
    data = dataset.get_all()
    if isinstance(data, dict):
        return np.asarray(data["xs"]), np.asarray(data["ys"])
    xs, ys = data
    return np.asarray(xs), np.asarray(ys)


def _sample_batch(
    dataset: Any,
    *,
    teacher_probs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs_all, ys_all = _extract_dataset_arrays(dataset)
    n_episodes = int(xs_all.shape[1])
    batch_mode = str(getattr(dataset, "batch_mode", "single"))
    batch_size = getattr(dataset, "batch_size", None)

    if batch_mode == "single" or batch_size is None:
        return xs_all, ys_all, teacher_probs

    batch_size = int(batch_size)
    if batch_size == 0:
        return xs_all[:, :0], ys_all[:, :0], teacher_probs[:, :0]

    if batch_mode == "rolling":
        current_start = int(getattr(dataset, "_current_start_index", 0))
        indices = np.arange(current_start, current_start + batch_size) % n_episodes
        setattr(dataset, "_current_start_index", (current_start + batch_size) % n_episodes)
    elif batch_mode == "random":
        rng = getattr(dataset, "rng", None)
        if rng is not None and hasattr(rng, "choice"):
            indices = rng.choice(n_episodes, size=batch_size)
        else:
            indices = np.random.choice(n_episodes, size=batch_size)
    else:
        raise ValueError(
            f"Unsupported dataset.batch_mode='{batch_mode}' for distillation training."
        )

    return xs_all[:, indices], ys_all[:, indices], teacher_probs[:, indices]


def _mask_from_targets(targets: jnp.ndarray) -> jnp.ndarray:
    categorical_mask = jnp.logical_not(targets == -1)
    continuous_mask = jnp.logical_not(jnp.isnan(targets))
    combined_mask = jnp.logical_and(categorical_mask, continuous_mask)
    return jnp.any(combined_mask, axis=-1)


def _normalize_teacher_probs(teacher_probs: jnp.ndarray) -> jnp.ndarray:
    teacher_probs = jnp.asarray(teacher_probs, dtype=jnp.float32)
    teacher_probs = jnp.clip(teacher_probs, a_min=1e-8, a_max=1.0)
    totals = jnp.sum(teacher_probs, axis=-1, keepdims=True)
    totals = jnp.where(totals > 0, totals, 1.0)
    return teacher_probs / totals


def _distillation_kl_loss(
    *,
    teacher_probs: jnp.ndarray,
    student_logits: jnp.ndarray,
    targets: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    teacher_probs = _normalize_teacher_probs(teacher_probs)
    teacher_log_probs = jnp.log(teacher_probs)
    student_log_probs = jax.nn.log_softmax(student_logits / temperature, axis=-1)
    trialwise_kl = jnp.sum(
        teacher_probs * (teacher_log_probs - student_log_probs),
        axis=-1,
    )
    valid_mask = _mask_from_targets(targets)
    masked_kl = jnp.where(valid_mask, trialwise_kl, 0.0)
    n_valid = jnp.maximum(1, jnp.sum(valid_mask))
    return (temperature**2) * jnp.sum(masked_kl) / n_valid


def evaluate_distillation_loss(
    *,
    make_network: Callable[[], hk.RNNCore],
    params: Any,
    xs: np.ndarray,
    ys: np.ndarray,
    teacher_probs: np.ndarray,
    temperature: float,
    n_action_logits: int,
    include_penalty: bool,
    penalty_scale: float,
) -> float:
    """Compute the current distillation objective for a full dataset."""
    def unroll_network(inputs: jnp.ndarray) -> jnp.ndarray:
        core = make_network()
        batch_size = jnp.shape(inputs)[1]
        state = core.initial_state(batch_size)
        outputs, _ = hk.dynamic_unroll(core, inputs, state)
        return outputs

    model = hk.transform(unroll_network)
    params = rnn_utils.to_np(params) if isinstance(params, dict) else params
    apply = jax.jit(model.apply)
    random_key = jax.random.PRNGKey(0)
    model_output = apply(params, random_key, np.asarray(xs))
    model_output = jnp.asarray(model_output)
    logits = model_output[:, :, :n_action_logits]
    loss = _distillation_kl_loss(
        teacher_probs=jnp.asarray(teacher_probs),
        student_logits=logits,
        targets=jnp.asarray(ys),
        temperature=float(temperature),
    )
    if include_penalty:
        penalty, n_unmasked_samples = rnn_utils.compute_penalty(
            jnp.asarray(ys),
            model_output,
        )
        avg_penalty = penalty / jnp.maximum(1, n_unmasked_samples)
        loss = loss + float(penalty_scale) * avg_penalty
    return float(loss)


def train_network_with_distillation(
    make_network: Callable[[], hk.RNNCore],
    training_dataset: Any,
    validation_dataset: Any | None,
    *,
    training_teacher_probs: np.ndarray,
    validation_teacher_probs: np.ndarray | None,
    opt: optax.GradientTransformation = optax.adam(1e-3),
    random_key: jax.Array | None = None,
    opt_state: optax.OptState | None = None,
    params: Any | None = None,
    n_steps: int = 1000,
    max_grad_norm: float = 1.0,
    temperature: float = 2.0,
    n_action_logits: int,
    include_penalty: bool,
    penalty_scale: float,
    log_losses_every: int = 10,
    report_progress_by: DistillationProgressMode = "print",
    wandb_run: Any | None = None,
    wandb_step_offset: int = 0,
) -> tuple[Any, optax.OptState, dict[str, np.ndarray]]:
    """Train an RNN against soft teacher targets."""
    xs_all, _ = _extract_dataset_arrays(training_dataset)

    def unroll_network(inputs: jnp.ndarray) -> jnp.ndarray:
        core = make_network()
        batch_size = jnp.shape(inputs)[1]
        state = core.initial_state(batch_size)
        outputs, _ = hk.dynamic_unroll(core, inputs, state)
        return outputs

    model = hk.transform(unroll_network)

    if random_key is None:
        random_key = jax.random.PRNGKey(0)
    if params is None:
        random_key, key1 = jax.random.split(random_key)
        params = model.init(key1, xs_all)
    if opt_state is None:
        opt_state = opt.init(params)

    xs_eval = ys_eval = None
    if validation_dataset is not None:
        xs_eval, ys_eval = _extract_dataset_arrays(validation_dataset)

    def compute_loss(
        current_params: Any,
        batch_xs: jnp.ndarray,
        batch_ys: jnp.ndarray,
        batch_teacher_probs: jnp.ndarray,
        key: jax.Array,
    ) -> jnp.ndarray:
        model_output = model.apply(current_params, key, batch_xs)
        logits = model_output[:, :, :n_action_logits]
        loss = _distillation_kl_loss(
            teacher_probs=batch_teacher_probs,
            student_logits=logits,
            targets=batch_ys,
            temperature=float(temperature),
        )
        if include_penalty:
            penalty, n_unmasked_samples = rnn_utils.compute_penalty(batch_ys, model_output)
            avg_penalty = penalty / jnp.maximum(1, n_unmasked_samples)
            loss = loss + float(penalty_scale) * avg_penalty
        return loss

    compute_loss_jit = jax.jit(compute_loss)

    @jax.jit
    def train_step(
        current_params: Any,
        current_opt_state: optax.OptState,
        batch_xs: jnp.ndarray,
        batch_ys: jnp.ndarray,
        batch_teacher_probs: jnp.ndarray,
        key: jax.Array,
    ) -> tuple[jnp.ndarray, Any, optax.OptState]:
        loss_value, grads = jax.value_and_grad(compute_loss, argnums=0)(
            current_params,
            batch_xs,
            batch_ys,
            batch_teacher_probs,
            key,
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
        batch_xs, batch_ys, batch_teacher = _sample_batch(
            training_dataset,
            teacher_probs=np.asarray(training_teacher_probs),
        )
        loss_value, params, opt_state = train_step(
            params,
            opt_state,
            jnp.asarray(batch_xs),
            jnp.asarray(batch_ys),
            jnp.asarray(batch_teacher),
            subkey_train,
        )

        if step % log_losses_every == 0 or step == int(n_steps) - 1:
            if rnn_utils.nan_in_dict(params):
                raise ValueError("NaN in params during distillation training.")
            if np.isnan(loss_value):
                raise ValueError("NaN in distillation loss.")
            if loss_value > 1e50:
                raise ValueError("Distillation loss exploded.")

            if (
                validation_dataset is not None
                and validation_teacher_probs is not None
                and xs_eval is not None
                and ys_eval is not None
            ):
                validation_value = compute_loss_jit(
                    params,
                    jnp.asarray(xs_eval),
                    jnp.asarray(ys_eval),
                    jnp.asarray(validation_teacher_probs),
                    subkey_validation,
                )

            training_loss.append(float(loss_value))
            validation_loss.append(float(validation_value))
            log_str = (
                f"Step {step + 1} of {n_steps}. "
                f"Training Loss: {float(loss_value):.2e}. "
                f"Validation Loss: {float(validation_value):.2e}"
            )

            if report_progress_by == "wandb" and hasattr(wandb_run, "log"):
                wandb_run.log(
                    {"train/loss": float(loss_value), "valid/loss": float(validation_value)},
                    step=step + wandb_step_offset,
                )

            if report_progress_by in {"print", "wandb"}:
                print(log_str)
            elif report_progress_by == "log":
                logger.info(log_str)
            elif report_progress_by == "none":
                pass
            else:
                raise ValueError(
                    f"Unsupported report_progress_by='{report_progress_by}' during distillation."
                )

    return params, opt_state, {
        "training_loss": np.asarray(training_loss, dtype=float),
        "validation_loss": np.asarray(validation_loss, dtype=float),
    }


def _softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled_logits = np.asarray(logits, dtype=np.float32) / float(temperature)
    scaled_logits = scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True)
    exp_logits = np.exp(scaled_logits)
    totals = np.sum(exp_logits, axis=-1, keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    return exp_logits / totals


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return json.load(f)


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _path_tail_after_data_segment(path: Path) -> Path | None:
    parts = path.parts
    for index, part in enumerate(parts):
        if part == "data" and index + 1 < len(parts):
            return Path(*parts[index + 1 :])
    return None


def _candidate_layout_keys(teacher_dir: Path) -> list[Path]:
    if not teacher_dir.is_absolute():
        return [teacher_dir, Path(teacher_dir.name)]

    keys: list[Path] = [Path(teacher_dir.name)]
    tail_after_data = _path_tail_after_data_segment(teacher_dir)
    if tail_after_data is not None:
        keys.insert(0, tail_after_data)
        # Support staged data mounts where the top-level dataset root segment
        # is dropped (e.g. /data/dataset_name/<id>/... -> /data/<id>/...).
        if len(tail_after_data.parts) > 1:
            keys.append(Path(*tail_after_data.parts[1:]))
    return _dedupe_paths(keys)


def _iter_teacher_dir_candidates(teacher_dir: Path) -> list[Path]:
    def _with_useful_ancestors(path: Path, *, max_levels: int = 4) -> list[Path]:
        expanded = path.expanduser()
        out: list[Path] = [expanded]
        current = expanded
        for _ in range(max_levels):
            if current.parent == current:
                break
            current = current.parent
            # Never add filesystem roots / broad data roots as candidates.
            if current == Path("/") or current == Path("/data"):
                break
            out.append(current)
            # For checkpoint paths, resolving from outputs is the intended fallback.
            if current.name == "outputs":
                break
        return out

    teacher_dir = teacher_dir.expanduser()
    candidates: list[Path] = _with_useful_ancestors(teacher_dir)

    if not teacher_dir.is_absolute():
        candidates.append((Path.cwd() / teacher_dir).expanduser())

    layout_keys = _candidate_layout_keys(teacher_dir)

    search_roots = _dedupe_paths([*list(_CANDIDATE_DATA_DIRS), Path.cwd()])
    logger.debug(
        "Teacher candidate search for %s across roots: %s",
        teacher_dir,
        [str(root) for root in search_roots],
    )

    for data_dir in search_roots:
        # Layout-style lookup under each data root (similar to load_mice_snapshot):
        # 1) full relative tail if available (e.g. jobs/x/y/step_100000)
        # 2) basename fallback (e.g. step_100000)
        for key in layout_keys:
            mapped = (data_dir / key).expanduser()
            candidates.extend(_with_useful_ancestors(mapped))

        if not data_dir.exists():
            continue

        # Recursive lookup fallback for unknown nesting depth.
        # Match on params.json (artifact anchor), not only directory names,
        # because names like "step_100000" are common across many runs.
        params_files = sorted(
            path
            for path in data_dir.rglob("params.json")
            if path.is_file()
        )
        if not params_files:
            continue

        for key in layout_keys:
            key_suffix = key.as_posix()
            matched = False
            for params_path in params_files:
                parent = params_path.parent
                if len(key.parts) > 1:
                    if not parent.as_posix().endswith(key_suffix):
                        continue
                elif parent.name != key.name:
                    continue

                candidates.append(parent)
                candidates.append(params_path)
                logger.debug(
                    "Matched teacher candidate via params anchor: key=%s parent=%s params=%s",
                    key,
                    parent,
                    params_path,
                )
                matched = True
                break

            if matched:
                continue

    return _dedupe_paths(candidates)


def _find_upward(start_dir: Path, filename: str, max_depth: int = 6) -> Path | None:
    current = start_dir
    for _ in range(max_depth + 1):
        candidate = current / filename
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def _resolve_teacher_artifacts(teacher_dir: Path) -> tuple[Path, Path, Path | None] | None:
    if teacher_dir.is_file():
        if teacher_dir.name != "params.json":
            return None
        params_candidates = [teacher_dir]
    elif teacher_dir.is_dir():
        params_candidates: list[Path] = []
        direct_params = teacher_dir / "params.json"
        if direct_params.exists():
            params_candidates.append(direct_params)
        if not params_candidates:
            params_candidates.extend(sorted(teacher_dir.rglob("params.json")))
    else:
        return None

    for params_path in params_candidates:
        config_path = _find_upward(params_path.parent, "gru_config.json")
        if config_path is None:
            continue
        subject_index_map_path = _find_upward(config_path.parent, "subject_index_map.json")
        return params_path, config_path, subject_index_map_path

    return None


def _load_teacher_summary(
    teacher_dir: Path,
    *,
    student_is_multisubject: bool,
    expected_output_size: int,
) -> tuple[TeacherModelSummary, dict[str, Any], Any]:
    artifact_paths: tuple[Path, Path, Path | None] | None = None
    resolved_teacher_dir: Path | None = None
    searched_candidates: list[Path] = []
    for candidate in _iter_teacher_dir_candidates(teacher_dir):
        searched_candidates.append(candidate)
        resolved = _resolve_teacher_artifacts(candidate)
        if resolved is not None:
            artifact_paths = resolved
            resolved_teacher_dir = candidate
            logger.info(
                "Resolved teacher candidate %s from input %s",
                candidate,
                teacher_dir,
            )
            break

    if artifact_paths is None:
        searched = ", ".join(str(path) for path in searched_candidates)
        raise FileNotFoundError(
            "Teacher artifacts could not be resolved from teacher_dir="
            f"{teacher_dir}. Tried candidates: {searched}"
        )

    params_path, config_path, subject_index_map_path = artifact_paths
    logger.info(
        "Resolved teacher artifacts for %s -> params=%s config=%s",
        teacher_dir,
        params_path,
        config_path,
    )

    config_payload = _load_json(config_path)
    architecture = dict(config_payload.get("architecture", {}))
    output_size = int(config_payload.get("output_size", 0))
    if output_size != int(expected_output_size):
        raise ValueError(
            f"Teacher output_size mismatch for {teacher_dir}: "
            f"teacher={output_size} student={expected_output_size}"
        )

    teacher_is_multisubject = bool(architecture.get("multisubject", False))
    if teacher_is_multisubject != bool(student_is_multisubject):
        raise ValueError(
            f"Teacher multisubject mode mismatch for {teacher_dir}: "
            f"teacher={teacher_is_multisubject} student={student_is_multisubject}"
        )

    params = rnn_utils.to_np(_load_json(params_path))
    subject_count: int | None = None
    if teacher_is_multisubject:
        if subject_index_map_path is None or not subject_index_map_path.exists():
            raise FileNotFoundError(
                "Multisubject GRU teacher requires subject_index_map.json: "
                f"{subject_index_map_path}"
            )
        subject_embeddings = extract_subject_embeddings_from_params(params)
        subject_count = int(subject_embeddings.shape[0])
    summary = TeacherModelSummary(
        model_dir=str(resolved_teacher_dir or teacher_dir),
        params_path=str(params_path),
        config_path=str(config_path),
        subject_index_map_path=(
            str(subject_index_map_path)
            if subject_index_map_path is not None and subject_index_map_path.exists()
            else None
        ),
        multisubject=teacher_is_multisubject,
        output_size=output_size,
        subject_count=subject_count,
    )
    return summary, architecture, params


def _make_teacher_network(
    *,
    architecture: Mapping[str, Any],
    output_size: int,
    params: Any,
) -> Callable[[], hk.RNNCore]:
    multisubject = bool(architecture.get("multisubject", False))
    max_n_subjects = None
    subject_embedding_size = None
    if multisubject:
        subject_embeddings = extract_subject_embeddings_from_params(params)
        max_n_subjects = int(subject_embeddings.shape[0])
        subject_embedding_size = int(subject_embeddings.shape[1])

    return make_gru_network(
        hidden_size=int(architecture["hidden_size"]),
        output_size=int(output_size),
        multisubject=multisubject,
        max_n_subjects=max_n_subjects,
        subject_embedding_size=subject_embedding_size,
        subject_embedding_init=str(architecture.get("subject_embedding_init", "zeros")),
    )


def build_teacher_ensemble(
    *,
    distillation: DistillationConfig,
    dataset: Any,
    dataset_train: Any,
    dataset_eval: Any,
    metadata: Mapping[str, Any],
    output_dir: str | Path,
    expected_output_size: int,
) -> DistillationEnsemble:
    """Load GRU teachers, validate compatibility, and precompute soft targets."""
    if not distillation.enabled:
        raise ValueError("build_teacher_ensemble requires distillation.enabled=true.")

    xs_full, _ = _extract_dataset_arrays(dataset)
    xs_train, _ = _extract_dataset_arrays(dataset_train)
    xs_eval, _ = _extract_dataset_arrays(dataset_eval)
    student_is_multisubject = bool(metadata.get("multisubject", False))

    if student_is_multisubject:
        student_index_to_subject_id = metadata.get("index_to_subject_id")
        if not isinstance(student_index_to_subject_id, dict):
            raise ValueError(
                "Multisubject distillation requires metadata.index_to_subject_id."
            )
    else:
        student_index_to_subject_id = {}

    teacher_train_probs: list[np.ndarray] = []
    teacher_eval_probs: list[np.ndarray] = []
    teacher_full_probs: list[np.ndarray] = []
    teacher_summaries: list[TeacherModelSummary] = []

    for teacher_dir_str in distillation.teacher_model_dirs:
        teacher_dir = Path(teacher_dir_str).expanduser().resolve()
        summary, architecture, params = _load_teacher_summary(
            teacher_dir,
            student_is_multisubject=student_is_multisubject,
            expected_output_size=expected_output_size,
        )
        teacher_summaries.append(summary)

        if summary.multisubject:
            assert summary.subject_index_map_path is not None
            teacher_subject_id_to_index, _ = load_subject_index_map(summary.subject_index_map_path)
            teacher_subject_id_to_index = {
                normalize_subject_id(subject_id): int(index)
                for subject_id, index in teacher_subject_id_to_index.items()
            }
            teacher_xs_full = remap_multisubject_teacher_inputs(
                xs_full,
                student_index_to_subject_id=student_index_to_subject_id,
                teacher_subject_id_to_index=teacher_subject_id_to_index,
            )
            teacher_xs_train = remap_multisubject_teacher_inputs(
                xs_train,
                student_index_to_subject_id=student_index_to_subject_id,
                teacher_subject_id_to_index=teacher_subject_id_to_index,
            )
            teacher_xs_eval = remap_multisubject_teacher_inputs(
                xs_eval,
                student_index_to_subject_id=student_index_to_subject_id,
                teacher_subject_id_to_index=teacher_subject_id_to_index,
            )
        else:
            teacher_xs_full = np.asarray(xs_full, dtype=np.float32)
            teacher_xs_train = np.asarray(xs_train, dtype=np.float32)
            teacher_xs_eval = np.asarray(xs_eval, dtype=np.float32)

        make_network = _make_teacher_network(
            architecture=architecture,
            output_size=summary.output_size,
            params=params,
        )
        full_logits, _ = rnn_utils.eval_network(make_network, params, teacher_xs_full)
        train_logits, _ = rnn_utils.eval_network(make_network, params, teacher_xs_train)
        eval_logits, _ = rnn_utils.eval_network(make_network, params, teacher_xs_eval)

        full_logits = np.asarray(full_logits)[:, :, :expected_output_size]
        train_logits = np.asarray(train_logits)[:, :, :expected_output_size]
        eval_logits = np.asarray(eval_logits)[:, :, :expected_output_size]

        teacher_full_probs.append(
            _softmax_with_temperature(full_logits, distillation.temperature)
        )
        teacher_train_probs.append(
            _softmax_with_temperature(train_logits, distillation.temperature)
        )
        teacher_eval_probs.append(
            _softmax_with_temperature(eval_logits, distillation.temperature)
        )

    manifest = {
        "enabled": True,
        "teacher_count": len(teacher_summaries),
        "temperature": float(distillation.temperature),
        "aggregation": distillation.aggregation,
        "loss": distillation.loss,
        "use_hard_labels": distillation.use_hard_labels,
        "evaluate_against_hard_labels": distillation.evaluate_against_hard_labels,
        "teacher_model_dirs": list(distillation.teacher_model_dirs),
        "teachers": [
            {
                "model_dir": summary.model_dir,
                "params_path": summary.params_path,
                "config_path": summary.config_path,
                "subject_index_map_path": summary.subject_index_map_path,
                "multisubject": summary.multisubject,
                "output_size": summary.output_size,
                "subject_count": summary.subject_count,
            }
            for summary in teacher_summaries
        ],
    }
    if distillation.save_manifest:
        manifest_path = Path(output_dir) / "distillation_manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)
        manifest["manifest_path"] = str(manifest_path)

    return DistillationEnsemble(
        config=distillation,
        train_probs=aggregate_teacher_probabilities(teacher_train_probs),
        eval_probs=aggregate_teacher_probabilities(teacher_eval_probs),
        full_probs=aggregate_teacher_probabilities(teacher_full_probs),
        manifest=manifest,
    )
