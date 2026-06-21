"""Unit tests for full-state checkpoint save/resume helpers."""

from __future__ import annotations

import pickle
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import optax

from model_trainers.checkpoint_resume import (
    TRAIN_STATE_FILENAME,
    find_latest_resumable_state,
    save_train_state,
)


def _make_state(seed: int):
    """A params pytree + matching optax adam state + PRNG key, as in training."""
    params = {"layer": {"w": jnp.ones((3, 2)) * seed, "b": jnp.zeros((2,))}}
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    # Advance the optimizer once so the state holds non-trivial moments + count.
    grads = jax.tree_util.tree_map(jnp.ones_like, params)
    _, opt_state = optimizer.update(grads, opt_state, params)
    random_key = jax.random.PRNGKey(seed)
    return params, opt_state, random_key


class SaveTrainStateTest(unittest.TestCase):
    def test_round_trip_restores_full_state(self) -> None:
        params, opt_state, random_key = _make_state(seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoints"
            save_train_state(
                root / "step_10",
                steps_completed=10,
                params=params,
                opt_state=opt_state,
                random_key=random_key,
                training_losses=[0.5, 0.4],
                validation_losses=[0.6, 0.55],
            )

            restored = find_latest_resumable_state(root)

        self.assertIsNotNone(restored)
        self.assertEqual(restored.steps_completed, 10)
        self.assertEqual(restored.training_losses, [0.5, 0.4])
        self.assertEqual(restored.validation_losses, [0.6, 0.55])
        # Params match exactly.
        self.assertTrue(
            jnp.allclose(restored.params["layer"]["w"], params["layer"]["w"])
        )
        # Optimizer-state pytree structure is preserved (same classes/leaves).
        self.assertEqual(
            jax.tree_util.tree_structure(restored.opt_state),
            jax.tree_util.tree_structure(opt_state),
        )
        # Restored state can drive another optimizer step without error.
        optimizer = optax.adam(1e-3)
        grads = jax.tree_util.tree_map(jnp.ones_like, restored.params)
        optimizer.update(grads, restored.opt_state, restored.params)
        # PRNG key round-trips so the resumed stream is deterministic.
        self.assertTrue(jnp.array_equal(restored.random_key, random_key))

    def test_picks_highest_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoints"
            for step in (5, 25, 15):
                params, opt_state, key = _make_state(seed=step)
                save_train_state(
                    root / f"step_{step}",
                    steps_completed=step,
                    params=params,
                    opt_state=opt_state,
                    random_key=key,
                    training_losses=[],
                    validation_losses=[],
                )
            restored = find_latest_resumable_state(root)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.steps_completed, 25)

    def test_skips_corrupt_latest_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "checkpoints"
            params, opt_state, key = _make_state(seed=1)
            save_train_state(
                root / "step_10",
                steps_completed=10,
                params=params,
                opt_state=opt_state,
                random_key=key,
                training_losses=[],
                validation_losses=[],
            )
            # A newer-but-corrupt checkpoint (e.g. preempted mid-write).
            corrupt_dir = root / "step_20"
            corrupt_dir.mkdir(parents=True)
            (corrupt_dir / TRAIN_STATE_FILENAME).write_bytes(b"not a pickle")

            restored = find_latest_resumable_state(root)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.steps_completed, 10)

    def test_no_checkpoints_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(find_latest_resumable_state(Path(tmp) / "missing"))
            empty = Path(tmp) / "checkpoints"
            empty.mkdir()
            self.assertIsNone(find_latest_resumable_state(empty))

    def test_write_is_atomic_no_tmp_left(self) -> None:
        params, opt_state, key = _make_state(seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "step_1"
            save_train_state(
                ckpt,
                steps_completed=1,
                params=params,
                opt_state=opt_state,
                random_key=key,
                training_losses=[],
                validation_losses=[],
            )
            files = {p.name for p in ckpt.iterdir()}
        self.assertIn(TRAIN_STATE_FILENAME, files)
        self.assertNotIn(TRAIN_STATE_FILENAME + ".tmp", files)


if __name__ == "__main__":
    unittest.main()
