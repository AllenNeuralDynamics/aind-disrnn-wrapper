"""Full-state checkpointing + resume for preemptible training.

The chunked checkpoint loops in :mod:`disrnn_trainer` / :mod:`gru_trainer`
already write ``params.json`` per checkpoint (consumed by downstream analysis).
That is enough to *evaluate* a model but not to *continue* training it: the
optimizer state, the evolving PRNG key and the completed-step counter are not
captured, so a re-launched job would restart from scratch.

For preemption recovery this module persists the *full* training state -
parameters, optimizer state, PRNG key, completed-step counter and the
accumulated loss history - as a single pickle sidecar (``train_state.pkl``)
written atomically alongside ``params.json`` in each ``step_<N>`` checkpoint
directory. On (re)start, :func:`find_latest_resumable_state` scans the
checkpoint root for the highest step with a loadable sidecar so training can
continue from there.

Pickling JAX/NumPy arrays is sufficient here: checkpoints are self-produced and
read back within the same pinned environment (same optax/Haiku versions, so the
optimizer-state pytree classes are stable). Restored leaves are coerced back to
device arrays via :func:`jax.tree_util.tree_map` so the resumed loop sees the
same dtypes it wrote.
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Any, NamedTuple

logger = logging.getLogger(__name__)

# Sidecar filename written next to ``params.json`` in every ``step_<N>`` dir.
TRAIN_STATE_FILENAME = "train_state.pkl"
_STEP_PREFIX = "step_"


class ResumeState(NamedTuple):
    """Full training state restored from a checkpoint sidecar."""

    steps_completed: int
    params: Any
    opt_state: Any
    random_key: Any
    training_losses: list[float]
    validation_losses: list[float]
    checkpoint_dir: Path


def _to_device_arrays(tree: Any) -> Any:
    """Coerce pickled (NumPy) leaves back to JAX arrays, if JAX is importable."""
    try:
        import jax
        import jax.numpy as jnp
    except Exception:  # pragma: no cover - JAX always present in training env
        return tree
    return jax.tree_util.tree_map(lambda leaf: jnp.asarray(leaf), tree)


def save_train_state(
    checkpoint_dir: str | os.PathLike[str],
    *,
    steps_completed: int,
    params: Any,
    opt_state: Any,
    random_key: Any,
    training_losses: Any,
    validation_losses: Any,
) -> Path:
    """Atomically write the full training state into ``checkpoint_dir``.

    The payload is written to a ``.tmp`` file, fsync'd and then ``os.replace``'d
    onto the final path, so a checkpoint interrupted mid-write never leaves a
    half-written ``train_state.pkl`` that resume would trip over.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "steps_completed": int(steps_completed),
        "params": params,
        "opt_state": opt_state,
        "random_key": random_key,
        "training_losses": list(training_losses),
        "validation_losses": list(validation_losses),
    }
    final_path = checkpoint_dir / TRAIN_STATE_FILENAME
    tmp_path = checkpoint_dir / (TRAIN_STATE_FILENAME + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, final_path)
    return final_path


def find_latest_resumable_state(
    checkpoint_root: str | os.PathLike[str],
) -> ResumeState | None:
    """Return the highest-step loadable training state, or ``None``.

    Scans ``checkpoint_root`` for ``step_<N>`` directories and tries them from
    the highest step downward, skipping any whose sidecar is missing or
    unreadable (e.g. truncated by a preemption mid-write that os.replace never
    completed). Returns ``None`` when nothing resumable is found.
    """
    checkpoint_root = Path(checkpoint_root)
    if not checkpoint_root.is_dir():
        return None

    candidates: list[tuple[int, Path]] = []
    for child in checkpoint_root.iterdir():
        if not child.is_dir() or not child.name.startswith(_STEP_PREFIX):
            continue
        try:
            step = int(child.name[len(_STEP_PREFIX):])
        except ValueError:
            continue
        if (child / TRAIN_STATE_FILENAME).is_file():
            candidates.append((step, child))

    for step, child in sorted(candidates, reverse=True):
        state_path = child / TRAIN_STATE_FILENAME
        try:
            with state_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception as exc:
            logger.warning("Skipping unreadable checkpoint state %s: %s", state_path, exc)
            continue
        logger.info("Resuming from checkpoint %s (step %s)", state_path, step)
        return ResumeState(
            steps_completed=int(payload["steps_completed"]),
            params=_to_device_arrays(payload["params"]),
            opt_state=_to_device_arrays(payload["opt_state"]),
            random_key=_to_device_arrays(payload["random_key"]),
            training_losses=list(payload.get("training_losses", [])),
            validation_losses=list(payload.get("validation_losses", [])),
            checkpoint_dir=child,
        )
    return None
