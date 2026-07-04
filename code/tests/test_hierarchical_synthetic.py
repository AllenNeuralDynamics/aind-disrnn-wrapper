"""Tests for the hierarchical synthetic generator.

Two tiers:
  * ``test_pure_logic_*``: exercise the seed hierarchy, drift, and clamp helpers
    with NO heavy deps (jax / foragers / disrnn) -- runnable anywhere numpy exists.
  * ``test_end_to_end_*``: full generation + merged multisubject dataset +
    ground-truth table + determinism. Requires the wrapper stack (jax,
    disentangled_rnns, aind_dynamic_foraging_models, aind_behavior_gym,
    aind_disrnn_utils). Skipped automatically if those imports are unavailable.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest


# --------------------------------------------------------------------------- #
# Tier 1: pure-logic helpers (no heavy deps)
# --------------------------------------------------------------------------- #
def _load_helpers():
    """Import only the stack-free helpers by loading the module's functions.

    We import the functions directly to avoid importing the heavy top-level
    dependencies at collection time; the helpers live in the same module but do
    not use jax/foragers.
    """
    from data_loaders import hierarchical_synthetic as hs

    return hs


HAS_STACK = all(
    importlib.util.find_spec(mod) is not None
    for mod in (
        "jax",
        "disentangled_rnns",
        "aind_dynamic_foraging_models",
        "aind_behavior_gym",
        "aind_disrnn_utils",
    )
)
requires_stack = pytest.mark.skipif(not HAS_STACK, reason="wrapper training stack not installed")


@pytest.mark.skipif(not HAS_STACK, reason="module import needs stack for top-level imports")
def test_pure_logic_clamp_and_drift():
    hs = _load_helpers()
    # clamp respects known bounds
    assert hs._clamp("learn_rate", 1.7) == 1.0
    assert hs._clamp("learn_rate", -0.3) == 0.0
    assert hs._clamp("biasL", 9.0) == 5.0
    assert hs._clamp("softmax_inverse_temperature", -1.0) == 0.0
    # linear drift
    assert hs._apply_drift("learn_rate", 0.2, {"mode": "linear", "delta": 0.4}, 0.0) == 0.2
    assert abs(hs._apply_drift("learn_rate", 0.2, {"mode": "linear", "delta": 0.4}, 1.0) - 0.6) < 1e-9
    # toward_zero drift (|bias| shrinks)
    assert abs(hs._apply_drift("biasL", 2.0, {"mode": "toward_zero", "frac": 1.0}, 1.0)) < 1e-9
    assert abs(hs._apply_drift("biasL", 2.0, {"mode": "toward_zero", "frac": 0.5}, 1.0) - 1.0) < 1e-9


# --------------------------------------------------------------------------- #
# Tier 2: end-to-end generation (requires the stack)
# --------------------------------------------------------------------------- #
def _tiny_config():
    task = {"type": "random_walk", "reward_baiting": False, "p_min": 0.0, "p_max": 1.0,
            "sigma": 0.15, "mean": 0, "seed": 42}
    agent = {
        "agent_class": "ForagerQLearning",
        "agent_kwargs": {
            "number_of_learning_rate": 1,
            "number_of_forget_rate": 0,
            "choice_kernel": "none",
            "action_selection": "softmax",
        },
        "agent_params": {},
        "subject_param_dist": {
            "learn_rate": {"type": "uniform", "min": 0.2, "max": 0.8},
            "biasL": {"type": "uniform", "min": -1.5, "max": 1.5},
            "softmax_inverse_temperature": {"type": "uniform", "min": 3.0, "max": 12.0},
        },
        "seed": 0,
    }
    return task, agent


@requires_stack
def test_seed_hierarchy_no_collisions_realcode():
    """Exercise the ACTUAL _subject_seed/_session_seed methods, not a re-derived formula."""
    from data_loaders.hierarchical_synthetic import HierarchicalCognitiveAgents

    task, agent = _tiny_config()
    loader = HierarchicalCognitiveAgents(
        task=task, agent=agent, num_trials=10, num_subjects=300,
        num_sessions_per_subject=50, eval_every_n=2, subject_seed_stride=100_000,
    )
    # Collect seeds from the real methods for a mice-scale grid.
    agent_seeds, task_seeds = set(), set()
    for si in range(300):
        for se in range(50):
            agent_seeds.add(loader._session_seed(0, si, se))
            task_seeds.add(loader._session_seed(7, si, se))  # different base stream
    assert len(agent_seeds) == 300 * 50, "agent seed collisions from real _session_seed"
    assert len(task_seeds) == 300 * 50, "task seed collisions from real _session_seed"
    # Sessions stay strictly inside their subject's stride block.
    assert loader._session_seed(0, 0, 49) < loader._subject_seed(0, 1)


@requires_stack
def test_end_to_end_static_stage1(tmp_path):
    from data_loaders.hierarchical_synthetic import HierarchicalCognitiveAgents

    task, agent = _tiny_config()
    loader = HierarchicalCognitiveAgents(
        task=task, agent=agent, num_trials=80, num_subjects=4,
        num_sessions_per_subject=6, eval_every_n=2, batch_size=None,
        groundtruth_dir=str(tmp_path),
    )
    bundle = loader.load()
    md = bundle.metadata

    # merged multisubject shape: feature 0 is Subject ID
    xs = bundle.extras["dataset"].get_all()["xs"]
    assert xs.shape[1] == 4 * 6  # subjects * sessions
    assert bundle.extras["dataset"].x_names[0] == "Subject ID"
    # subject ids are dense [0, num_subjects)
    subj_col = xs[..., 0]
    present = np.unique(subj_col[subj_col >= 0]).astype(int)
    assert set(present.tolist()) == set(range(4))
    assert md["num_subjects"] == 4
    # ground-truth likelihood populated (finite, in (0, 1])
    gt = md["avg_eval_likelihood_groundtruth"]
    assert np.isfinite(gt) and 0.0 < gt <= 1.0
    # ground-truth table complete: one row per (subject, session)
    gtab = bundle.extras["groundtruth_table"]
    assert len(gtab) == 4 * 6
    assert {"param_learn_rate", "param_biasL", "param_softmax_inverse_temperature"} <= set(gtab.columns)
    assert "session_seed" in gtab.columns
    # Stage 1 (no drift): params constant across a subject's sessions
    for _, sub_rows in gtab.groupby("subject_index"):
        assert sub_rows["param_learn_rate"].nunique() == 1
        assert sub_rows["param_biasL"].nunique() == 1
    # session_context present with per-subject 1-based ordering
    assert md["session_context"]["indexing"] == "1_based"
    assert len(md["session_context"]["per_subject"]) == 4
    # Seed hierarchy is collision-free -- asserted on the ACTUAL seeds the
    # generator emitted (not a re-derived formula): every (subject, session)
    # got a unique agent seed and a unique task seed.
    assert gtab["session_seed"].is_unique, "agent session_seed collisions in generated data"
    assert gtab["task_seed"].is_unique, "task_seed collisions in generated data"


@requires_stack
def test_end_to_end_drift_stage2(tmp_path):
    from data_loaders.hierarchical_synthetic import HierarchicalCognitiveAgents

    task, agent = _tiny_config()
    agent["drift"] = {
        "learn_rate": {"mode": "linear", "delta": 0.3},
        "biasL": {"mode": "toward_zero", "frac": 0.8},
        "softmax_inverse_temperature": {"mode": "multiplicative", "rel": 0.5},
    }
    loader = HierarchicalCognitiveAgents(
        task=task, agent=agent, num_trials=80, num_subjects=3,
        num_sessions_per_subject=8, eval_every_n=2, batch_size=None,
        groundtruth_dir=str(tmp_path),
    )
    gtab = loader.load().extras["groundtruth_table"]
    # Stage 2: learn_rate strictly increases across sessions within each subject
    for _, sub_rows in gtab.sort_values("session_index_0based").groupby("subject_index"):
        lr = sub_rows["param_learn_rate"].to_numpy()
        assert np.all(np.diff(lr) >= -1e-9) and lr[-1] > lr[0]
        # |biasL| shrinks
        ab = np.abs(sub_rows["param_biasL"].to_numpy())
        assert ab[-1] <= ab[0] + 1e-9


@requires_stack
def test_end_to_end_determinism(tmp_path):
    """Same seed + config => byte-identical merged tensors + ground-truth table."""
    from data_loaders.hierarchical_synthetic import HierarchicalCognitiveAgents

    task, agent = _tiny_config()
    kw = dict(task=task, agent=agent, num_trials=80, num_subjects=3,
              num_sessions_per_subject=6, eval_every_n=2, batch_size=None)
    b1 = HierarchicalCognitiveAgents(groundtruth_dir=str(tmp_path / "a"), **kw).load()
    b2 = HierarchicalCognitiveAgents(groundtruth_dir=str(tmp_path / "b"), **kw).load()
    xs1 = b1.extras["dataset"].get_all()["xs"]
    xs2 = b2.extras["dataset"].get_all()["xs"]
    ys1 = b1.extras["dataset"].get_all()["ys"]
    ys2 = b2.extras["dataset"].get_all()["ys"]
    assert np.array_equal(xs1, xs2), "merged xs differ across identical-seed runs"
    assert np.array_equal(ys1, ys2), "merged ys differ across identical-seed runs"
    # ground-truth param table identical
    g1 = b1.extras["groundtruth_table"].reset_index(drop=True)
    g2 = b2.extras["groundtruth_table"].reset_index(drop=True)
    import pandas.testing as pdt
    pdt.assert_frame_equal(g1, g2)
    assert b1.metadata["avg_eval_likelihood_groundtruth"] == b2.metadata["avg_eval_likelihood_groundtruth"]


@requires_stack
def test_end_to_end_serial_equals_parallel(tmp_path):
    """Parallel generation (workers>1) is byte-identical to serial (workers=1)."""
    from data_loaders.hierarchical_synthetic import HierarchicalCognitiveAgents

    task, agent = _tiny_config()
    kw = dict(task=task, agent=agent, num_trials=80, num_subjects=6,
              num_sessions_per_subject=6, eval_every_n=2, batch_size=None)
    serial = HierarchicalCognitiveAgents(
        groundtruth_dir=str(tmp_path / "s"), generation_workers=1, **kw
    ).load()
    parallel = HierarchicalCognitiveAgents(
        groundtruth_dir=str(tmp_path / "p"), generation_workers=3, **kw
    ).load()
    xs_s = serial.extras["dataset"].get_all()["xs"]
    xs_p = parallel.extras["dataset"].get_all()["xs"]
    ys_s = serial.extras["dataset"].get_all()["ys"]
    ys_p = parallel.extras["dataset"].get_all()["ys"]
    assert np.array_equal(xs_s, xs_p), "parallel xs differ from serial"
    assert np.array_equal(ys_s, ys_p), "parallel ys differ from serial"
    import pandas.testing as pdt
    pdt.assert_frame_equal(
        serial.extras["groundtruth_table"].reset_index(drop=True),
        parallel.extras["groundtruth_table"].reset_index(drop=True),
    )
    assert (serial.metadata["avg_eval_likelihood_groundtruth"]
            == parallel.metadata["avg_eval_likelihood_groundtruth"])


def test_resolve_workers_logic(tmp_path):
    """_resolve_workers honors explicit counts and caps at num_subjects."""
    import importlib.util
    if importlib.util.find_spec("jax") is None:
        import pytest as _pytest
        _pytest.skip("module import needs stack")
    from data_loaders.hierarchical_synthetic import HierarchicalCognitiveAgents

    task, agent = _tiny_config()
    mk = lambda w, n: HierarchicalCognitiveAgents(
        task=task, agent=agent, num_trials=10, num_subjects=n,
        num_sessions_per_subject=2, generation_workers=w,
        groundtruth_dir=str(tmp_path),
    )
    assert mk(1, 10)._resolve_workers() == 1          # forced serial
    assert mk(4, 10)._resolve_workers() == 4          # explicit
    assert mk(50, 10)._resolve_workers() == 10        # capped at num_subjects
