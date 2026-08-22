"""Unit tests for utils.trial_timing_features.

These cover the pure-Python transformation logic (config resolution, encoding,
feature-map construction, merge/fill behavior) without touching the database —
so they run anywhere numpy/pandas import, no network or DuckDB needed.
"""

from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd

    from utils.trial_timing_features import (
        DEFAULT_LICK_WINDOW_S,
        attach_timing_features,
        encode_timing_features,
        required_raw_columns,
        resolve_timing_config,
        timing_feature_map,
    )

    DEPS = True
    IMPORT_ERR = None
except ModuleNotFoundError as exc:  # pragma: no cover
    DEPS = False
    IMPORT_ERR = exc


@unittest.skipUnless(DEPS, f"deps unavailable: {IMPORT_ERR}")
class TestResolveTimingConfig(unittest.TestCase):
    def test_none_is_disabled(self):
        cfg = resolve_timing_config(None)
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.feature_map(), {})
        self.assertEqual(cfg.raw_columns(), [])

    def test_bool_shortcut(self):
        self.assertTrue(resolve_timing_config(True).enabled)
        self.assertFalse(resolve_timing_config(False).enabled)

    def test_mapping_defaults(self):
        cfg = resolve_timing_config({"enabled": True})
        self.assertTrue(cfg.enabled)
        self.assertTrue(cfg.reaction_time)
        self.assertTrue(cfg.lick_counts)
        self.assertEqual(cfg.lick_window_s, DEFAULT_LICK_WINDOW_S)

    def test_mapping_partial_selection(self):
        cfg = resolve_timing_config(
            {"enabled": True, "reaction_time": False, "lick_counts": True,
             "lick_window_s": 1.5}
        )
        self.assertEqual(cfg.lick_window_s, 1.5)
        self.assertEqual(set(cfg.feature_map()), {"n_lick_left", "n_lick_right"})
        self.assertEqual(cfg.raw_columns(), ["n_lick_left", "n_lick_right"])

    def test_bad_type_raises(self):
        with self.assertRaises(TypeError):
            resolve_timing_config(42)


@unittest.skipUnless(DEPS, f"deps unavailable: {IMPORT_ERR}")
class TestFeatureMap(unittest.TestCase):
    def test_full_map_order_and_labels(self):
        fm = timing_feature_map()
        self.assertEqual(
            fm,
            {"log_reaction_time": "prev log RT",
             "n_lick_left": "prev n_lick_left",
             "n_lick_right": "prev n_lick_right"},
        )

    def test_rt_only(self):
        self.assertEqual(
            timing_feature_map(include_lick_counts=False),
            {"log_reaction_time": "prev log RT"},
        )

    def test_required_raw_columns(self):
        self.assertEqual(
            required_raw_columns(),
            ["reaction_time", "n_lick_left", "n_lick_right"],
        )


@unittest.skipUnless(DEPS, f"deps unavailable: {IMPORT_ERR}")
class TestEncode(unittest.TestCase):
    """Raw (native-unit) encoding contract — standardize explicitly disabled."""

    def test_log_rt_and_nan_to_zero(self):
        df = pd.DataFrame({
            "reaction_time": [1.0, np.e, np.nan, 0.0],
            "n_lick_left": [0, 3, 1, 2],
            "n_lick_right": [5, 0, 2, 4],
        })
        out = encode_timing_features(df, standardize=False)
        # log(1)=0, log(e)=1, NaN->0, log(clip(0,1e-3))=log(1e-3)
        self.assertAlmostEqual(out["log_reaction_time"].iloc[0], 0.0)
        self.assertAlmostEqual(out["log_reaction_time"].iloc[1], 1.0)
        self.assertEqual(out["log_reaction_time"].iloc[2], 0.0)  # NaN -> neutral 0
        self.assertAlmostEqual(out["log_reaction_time"].iloc[3], np.log(1e-3))
        self.assertTrue(np.isfinite(out["log_reaction_time"]).all())

    def test_lick_counts_kept_as_float(self):
        df = pd.DataFrame({"reaction_time": [0.2], "n_lick_left": [3], "n_lick_right": [4]})
        out = encode_timing_features(df, standardize=False)
        self.assertEqual(out["n_lick_left"].iloc[0], 3.0)
        self.assertEqual(out["n_lick_right"].iloc[0], 4.0)


@unittest.skipUnless(DEPS, f"deps unavailable: {IMPORT_ERR}")
class TestAttachMerge(unittest.TestCase):
    def test_merge_keys_and_fill(self):
        df = pd.DataFrame({
            "subject_id": ["1", "1", "1"],
            "ses_idx": ["s0", "s0", "s0"],
            "trial": [0, 1, 2],
            "animal_response": [0, 1, 0],
        })
        timing = pd.DataFrame({
            "ses_idx": ["s0", "s0"],       # trial 2 intentionally missing
            "trial": [0, 1],
            "reaction_time": [0.15, 0.30],
            "n_lick_left": [2, 0],
            "n_lick_right": [1, 4],
        })
        out = attach_timing_features(df, timing_df=timing)
        # trial 2 unmatched: licks fill 0, RT NaN
        self.assertEqual(out.loc[out.trial == 2, "n_lick_left"].iloc[0], 0)
        self.assertEqual(out.loc[out.trial == 2, "n_lick_right"].iloc[0], 0)
        self.assertTrue(np.isnan(out.loc[out.trial == 2, "reaction_time"].iloc[0]))
        # matched rows carry their values
        self.assertEqual(out.loc[out.trial == 1, "n_lick_right"].iloc[0], 4)
        self.assertEqual(len(out), 3)

    def test_missing_key_raises(self):
        df = pd.DataFrame({"subject_id": ["1"], "trial": [0]})  # no ses_idx
        with self.assertRaises(ValueError):
            attach_timing_features(df, timing_df=pd.DataFrame())


@unittest.skipUnless(DEPS, f"deps unavailable: {IMPORT_ERR}")
class TestStandardization(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame({
            "reaction_time": [0.14, 0.30, 1.0, np.nan],
            "n_lick_left": [0, 5, 12, 3],
            "n_lick_right": [4, 0, 8, 2],
        })

    def test_standardize_default_on(self):
        from utils.trial_timing_features import (
            LICK_CENTER, LICK_SCALE, LOG_RT_CENTER, LOG_RT_SCALE,
        )
        out = encode_timing_features(self._frame())
        expect_rt0 = (np.log(0.14) - LOG_RT_CENTER) / LOG_RT_SCALE
        self.assertAlmostEqual(out["log_reaction_time"].iloc[0], expect_rt0, places=6)
        self.assertAlmostEqual(out["n_lick_left"].iloc[1],
                               (5 - LICK_CENTER) / LICK_SCALE, places=6)

    def test_standardize_off_keeps_native_units(self):
        out = encode_timing_features(self._frame(), standardize=False)
        self.assertAlmostEqual(out["log_reaction_time"].iloc[0], float(np.log(0.14)), places=6)
        self.assertEqual(out["n_lick_left"].iloc[1], 5.0)

    def test_missing_rt_is_neutral_zero_in_both_modes(self):
        for std in (True, False):
            out = encode_timing_features(self._frame(), standardize=std)
            self.assertEqual(out["log_reaction_time"].iloc[3], 0.0)
            self.assertTrue(np.isfinite(out["log_reaction_time"]).all())

    def test_standardize_flows_through_config(self):
        self.assertTrue(resolve_timing_config({"enabled": True}).standardize)
        self.assertFalse(
            resolve_timing_config({"enabled": True, "standardize": False}).standardize
        )

    def test_lick_columns_route_to_float_builder(self):
        """Standardized lick counts are continuous, so routing must catch them."""
        from utils.trial_timing_features import has_continuous_features

        self.assertTrue(has_continuous_features({"n_lick_left": "prev n_lick_left"}))


@unittest.skipUnless(DEPS, f"deps unavailable: {IMPORT_ERR}")
class TestFloatSafeDatasetBuilder(unittest.TestCase):
    """Regression tests for the upstream int64 truncation bug.

    Upstream ``create_disrnn_dataset`` allocates ``xs`` via ``np.full(..., -1)``,
    which is int64, so continuous feature columns are truncated toward zero
    before the later ``astype(float)``. These tests pin the float-safe behavior.
    """

    def _frame(self):
        # Two sessions, continuous log-RT values that would truncate to the same
        # integer if the tensor were allocated as int.
        rows = []
        for ses, rts in (("s0", [0.11, 0.19, 0.87, 0.42]), ("s1", [0.55, 0.62, 0.71])):
            for i, rt in enumerate(rts):
                rows.append({
                    "ses_idx": ses, "trial": i,
                    "animal_response": i % 2, "earned_reward": bool(i % 2),
                    "log_reaction_time": float(np.log(rt)),
                    "n_lick_left": float(i), "n_lick_right": float(i + 1),
                })
        return pd.DataFrame(rows)

    def test_continuous_values_survive(self):
        from utils.trial_timing_features import create_disrnn_dataset_float

        df = self._frame()
        feats = {"animal_response": "prev choice", "rewarded": "prev reward",
                 "log_reaction_time": "prev log RT"}
        ds = create_disrnn_dataset_float(
            df, ignore_policy="exclude", features=feats, batch_size=None,
            batch_mode="single",
        )
        xs = ds.get_all()["xs"]
        self.assertEqual(xs.dtype.kind, "f")
        col = xs[:, :, 2]
        non_sentinel = col[col != -1]
        # All log-RT values are negative fractions in (-2.3, 0); if truncated they
        # would all collapse to 0 or -1.
        self.assertTrue(np.any(~np.isclose(non_sentinel, np.round(non_sentinel))),
                        "continuous values were truncated to integers")

    def test_prev_trial_shift_and_padding(self):
        from utils.trial_timing_features import create_disrnn_dataset_float

        df = self._frame()
        feats = {"animal_response": "prev choice", "rewarded": "prev reward",
                 "log_reaction_time": "prev log RT"}
        ds = create_disrnn_dataset_float(
            df, ignore_policy="include", features=feats, batch_size=None,
            batch_mode="single",
        )
        xs, ys = ds.get_all()["xs"], ds.get_all()["ys"]
        # Row 0 of every session is the -1 fill (no previous trial).
        self.assertTrue(np.all(xs[0, :, :] == -1))
        # Session 0 row 1 must carry session 0 trial 0's log RT.
        s0 = df[df.ses_idx == "s0"].sort_values("trial")
        self.assertAlmostEqual(float(xs[1, 0, 2]), float(s0["log_reaction_time"].iloc[0]))
        # Targets are the CURRENT trial's response.
        self.assertAlmostEqual(float(ys[0, 0, 0]), float(s0["animal_response"].iloc[0]))
        # Shorter session (s1, 3 trials) is padded at the tail.
        self.assertEqual(xs.shape[0], 4)
        self.assertTrue(np.all(xs[3, 1, :] == -1))

    def test_has_continuous_features_routing(self):
        from utils.trial_timing_features import has_continuous_features

        self.assertFalse(has_continuous_features(None))
        self.assertFalse(has_continuous_features({"animal_response": "prev choice",
                                                  "rewarded": "prev reward"}))
        self.assertTrue(has_continuous_features({"log_reaction_time": "prev log RT"}))
        # Lick columns count as continuous: with standardize=True (the default)
        # they are no longer integers, and the routing predicate cannot see that
        # flag — so it must treat them as continuous unconditionally.
        self.assertTrue(has_continuous_features({"n_lick_left": "prev n_lick_left"}))


if __name__ == "__main__":
    unittest.main()


class ShuffleControlArmTests(unittest.TestCase):
    """The shuffled-response control arm.

    The arm's whole scientific value rests on preserving everything EXCEPT
    trial alignment, so each invariant is asserted separately: if any one of them
    silently breaks, `real - shuffled` stops isolating information and starts
    measuring some second, unintended manipulation.
    """

    def _frame(self, n_per=40, n_ses=3, seed=0):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(seed)
        n = n_per * n_ses
        return pd.DataFrame({
            "subject_id": ["m1"] * n,
            "ses_idx": np.repeat(list(range(n_ses)), n_per).astype(str),
            "trial": list(range(n_per)) * n_ses,
            "reaction_time": rng.lognormal(-1.8, 0.6, n),
            "n_lick_left": rng.poisson(3, n),
            "n_lick_right": rng.poisson(3, n),
        })

    def test_per_session_marginals_are_preserved(self):
        import numpy as np
        import pandas as pd

        from utils import trial_timing_features as tf
        df = self._frame()
        out = tf.shuffle_raw_response_columns(df, seed=0)
        for ses, g in df.groupby("ses_idx"):
            h = out[out.ses_idx == ses]
            for col in tf.RAW_TIMING_COLUMNS:
                np.testing.assert_allclose(
                    np.sort(g[col].to_numpy().astype(float)),
                    np.sort(h[col].to_numpy().astype(float)),
                    err_msg=f"marginal changed for {col} in session {ses}",
                )

    def test_trial_alignment_is_destroyed(self):
        import numpy as np
        import pandas as pd

        from utils import trial_timing_features as tf
        df = self._frame()
        out = tf.shuffle_raw_response_columns(df, seed=0)
        moved = (df.reaction_time.to_numpy() != out.reaction_time.to_numpy()).mean()
        self.assertGreater(moved, 0.5)

    def test_columns_are_permuted_jointly(self):
        import numpy as np
        import pandas as pd

        from utils import trial_timing_features as tf
        # Independent per-column permutation would ALSO break the within-trial
        # RT/lick coupling, making the contrast ambiguous. The set of (rt, L, R)
        # triples must therefore survive intact.
        df = self._frame()
        out = tf.shuffle_raw_response_columns(df, seed=0)
        cols = list(tf.RAW_TIMING_COLUMNS)
        self.assertEqual(
            set(map(tuple, df[cols].to_numpy().round(9))),
            set(map(tuple, out[cols].to_numpy().round(9))),
        )

    def test_values_never_cross_sessions(self):
        import numpy as np
        import pandas as pd

        from utils import trial_timing_features as tf
        df = self._frame()
        out = tf.shuffle_raw_response_columns(df, seed=0)
        for ses, g in df.groupby("ses_idx"):
            self.assertEqual(
                set(g.reaction_time.round(9)),
                set(out[out.ses_idx == ses].reaction_time.round(9)),
            )

    def test_deterministic_and_seed_dependent(self):
        import numpy as np
        import pandas as pd

        from utils import trial_timing_features as tf
        df = self._frame()
        a = tf.shuffle_raw_response_columns(df, seed=0).reaction_time.to_numpy()
        b = tf.shuffle_raw_response_columns(df, seed=0).reaction_time.to_numpy()
        c = tf.shuffle_raw_response_columns(df, seed=1).reaction_time.to_numpy()
        np.testing.assert_allclose(a, b)
        self.assertFalse(np.allclose(a, c))

    def test_row_order_and_length_preserved(self):
        import numpy as np
        import pandas as pd

        from utils import trial_timing_features as tf
        df = self._frame()
        out = tf.shuffle_raw_response_columns(df, seed=0)
        self.assertEqual(list(out.index), list(df.index))
        self.assertEqual(len(out), len(df))
        pd.testing.assert_series_equal(out.trial, df.trial)

    def test_requires_raw_columns_present(self):
        import numpy as np
        import pandas as pd

        from utils import trial_timing_features as tf
        bad = pd.DataFrame({"subject_id": ["m1"], "ses_idx": ["0"], "trial": [0]})
        with self.assertRaises(ValueError):
            tf.shuffle_raw_response_columns(bad)

    def test_config_parses_shuffle_flags(self):
        import numpy as np
        import pandas as pd

        from utils import trial_timing_features as tf
        cfg = tf.resolve_timing_config(
            {"enabled": True, "shuffle": True, "shuffle_seed": 7}
        )
        self.assertTrue(cfg.shuffle)
        self.assertEqual(cfg.shuffle_seed, 7)
        # default must stay OFF so no existing arm silently becomes a control
        self.assertFalse(tf.resolve_timing_config({"enabled": True}).shuffle)

    def test_shuffled_arm_has_same_observation_width(self):
        import numpy as np
        import pandas as pd

        from utils import trial_timing_features as tf
        # Parameter-matching is the point: identical feature map to the real arm.
        real = tf.resolve_timing_config({"enabled": True})
        shuf = tf.resolve_timing_config({"enabled": True, "shuffle": True})
        self.assertEqual(real.feature_map(), shuf.feature_map())
        self.assertEqual(real.raw_columns(), shuf.raw_columns())
