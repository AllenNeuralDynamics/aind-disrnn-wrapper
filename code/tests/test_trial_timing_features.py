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
    def test_log_rt_and_nan_to_zero(self):
        df = pd.DataFrame({
            "reaction_time": [1.0, np.e, np.nan, 0.0],
            "n_lick_left": [0, 3, 1, 2],
            "n_lick_right": [5, 0, 2, 4],
        })
        out = encode_timing_features(df)
        # log(1)=0, log(e)=1, NaN->0, log(clip(0,1e-3))=log(1e-3)
        self.assertAlmostEqual(out["log_reaction_time"].iloc[0], 0.0)
        self.assertAlmostEqual(out["log_reaction_time"].iloc[1], 1.0)
        self.assertEqual(out["log_reaction_time"].iloc[2], 0.0)  # NaN -> neutral 0
        self.assertAlmostEqual(out["log_reaction_time"].iloc[3], np.log(1e-3))
        self.assertTrue(np.isfinite(out["log_reaction_time"]).all())

    def test_lick_counts_kept_as_float(self):
        df = pd.DataFrame({"reaction_time": [0.2], "n_lick_left": [3], "n_lick_right": [4]})
        out = encode_timing_features(df)
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
        self.assertFalse(has_continuous_features({"n_lick_left": "prev n_lick_left"}))
        self.assertTrue(has_continuous_features({"log_reaction_time": "prev log RT"}))


if __name__ == "__main__":
    unittest.main()
