"""Unit tests for the database-backed mice loader and selection pipeline.

The pure selection logic (``_partition_subjects``) is tested directly on synthetic
session frames; ``load_mice_from_database`` is tested with the DuckDB query helpers
(``select_sessions`` / ``fetch_trials``) stubbed out, so nothing touches the network.
"""

from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from utils import load_mice_database
from utils.load_mice_database import (
    DEFAULT_COLS,
    _partition_subjects,
    load_mice_from_database,
)


def _make_session_df(specs, stage="STAGE_FINAL"):
    """Build a one-row-per-session frame from ``(subject_id, task, n_sessions)`` specs.

    ``stage`` may be a single value (applied to all) or a callable ``(subject, k) -> stage``.
    """
    rows = []
    for subject_id, task, n_sessions in specs:
        for k in range(n_sessions):
            stage_value = stage(subject_id, k) if callable(stage) else stage
            rows.append(
                {
                    "_session_id": f"{subject_id}_2024-{(k % 12) + 1:02d}-{(k % 28) + 1:02d}_{task[:3]}{k}",
                    "subject_id": str(subject_id),
                    "session_date": f"2024-{(k % 12) + 1:02d}-{(k % 28) + 1:02d}",
                    "task": task,
                    "current_stage_actual": stage_value,
                    "curriculum_name": task,
                }
            )
    return pd.DataFrame(rows)


# 24 Coupled-Baiting subjects c00..c23 with strictly-descending session counts
# (c00=33 ... c23=10), plus a small Uncoupled-Baiting group and an ineligible one.
_RANKING_SPECS = (
    [(f"c{i:02d}", "Coupled Baiting", 33 - i) for i in range(24)]
    + [("u0", "Uncoupled Baiting", 15), ("u1", "Uncoupled Baiting", 14), ("u2", "Uncoupled Baiting", 13)]
    + [("x0", "Coupled Baiting", 5)]  # < min_sessions, dropped
)
_RANKING_DF = _make_session_df(_RANKING_SPECS)


def _fake_fetch_trials(sessions, columns=None, **kwargs):
    if len(sessions) == 0:
        return pd.DataFrame()
    meta_cols = [
        c for c in sessions.columns
        if c not in ("_session_id", "subject_id", "session_date", "session_id")
    ]
    rows = []
    for _, session in sessions.iterrows():
        for trial_index in range(2):
            row = {
                "subject_id": session["subject_id"],
                "session_date": session["session_date"],
                "session_id": session["_session_id"],
                "trial": float(trial_index),
                "animal_response": float(trial_index % 3),
                "earned_reward": bool(trial_index % 2),
            }
            for col in meta_cols:
                row[col] = session[col]
            rows.append(row)
    return pd.DataFrame(rows)


class TestPartitionSubjects(unittest.TestCase):
    def _partition(self, **overrides):
        kwargs = dict(
            curricula=["Coupled Baiting", "Uncoupled Baiting"],
            min_sessions=10,
            heldout_every_n=5,
            subject_ratio=1.0,
            seed=0,
        )
        kwargs.update(overrides)
        return _partition_subjects(_RANKING_DF, **kwargs)

    def test_min_sessions_filter_drops_low_session_subjects(self):
        part = self._partition()
        all_selected = set(part["train_ids"]) | set(part["heldout_ids"])
        self.assertNotIn("x0", all_selected)  # only 5 sessions

    def test_every_fifth_ranked_subject_is_heldout(self):
        part = self._partition()
        coupled = part["per_curriculum"]["Coupled Baiting"]
        # c00..c23 ranked by descending session count -> heldout at idx 4,9,14,19.
        self.assertEqual(coupled["ranked"], [f"c{i:02d}" for i in range(24)])
        self.assertEqual(coupled["heldout"], ["c04", "c09", "c14", "c19"])
        self.assertEqual(len(coupled["pool"]), 20)
        self.assertEqual(set(coupled["heldout"]) & set(coupled["pool"]), set())

    def test_full_ratio_train_pool_equals_pool(self):
        part = self._partition(subject_ratio=1.0)
        # ratio 1.0 -> the whole pool is the train set; train and heldout are disjoint.
        self.assertEqual(set(part["train_ids"]), set(part["train_pool_ids"]))
        self.assertEqual(set(part["train_ids"]) & set(part["heldout_ids"]), set())

    def test_per_curriculum_ratio_sampling(self):
        part = self._partition(subject_ratio={"Coupled Baiting": 0.5, "Uncoupled Baiting": 1.0})
        coupled = part["per_curriculum"]["Coupled Baiting"]
        uncoupled = part["per_curriculum"]["Uncoupled Baiting"]
        self.assertEqual(len(coupled["train"]), 10)  # round(0.5 * 20)
        self.assertTrue(set(coupled["train"]).issubset(set(coupled["pool"])))
        self.assertEqual(len(uncoupled["train"]), len(uncoupled["pool"]))  # ratio 1.0

    def test_training_sample_is_seed_reproducible_heldout_is_not(self):
        a = self._partition(subject_ratio=0.5, seed=0)
        b = self._partition(subject_ratio=0.5, seed=0)
        c = self._partition(subject_ratio=0.5, seed=1)
        self.assertEqual(a["train_ids"], b["train_ids"])  # same seed -> identical
        self.assertNotEqual(a["train_ids"], c["train_ids"])  # different seed -> different
        self.assertEqual(a["heldout_ids"], c["heldout_ids"])  # heldout is seed-independent

    def test_most_common_task_assignment_with_tiebreak(self):
        # m0: 6 Coupled + 6 Uncoupled Baiting -> tie, broken by task name asc -> Coupled.
        df = _make_session_df(
            [("m0", "Coupled Baiting", 6), ("m0", "Uncoupled Baiting", 6)]
        )
        part = _partition_subjects(
            df,
            curricula=["Coupled Baiting", "Uncoupled Baiting"],
            min_sessions=10,
            heldout_every_n=5,
            subject_ratio=1.0,
            seed=0,
        )
        self.assertEqual(part["subject_curriculum"]["m0"], "Coupled Baiting")

    def test_unchosen_curriculum_subjects_are_dropped(self):
        part = self._partition(curricula=["Coupled Baiting"])  # exclude Uncoupled
        all_selected = set(part["train_ids"]) | set(part["heldout_ids"])
        self.assertEqual({s for s in all_selected if s.startswith("u")}, set())


class TestLoadMiceFromDatabase(unittest.TestCase):
    def _patches(self, sessions=_RANKING_DF):
        return (
            mock.patch.object(
                load_mice_database, "select_sessions", return_value=sessions.copy()
            ),
            mock.patch.object(
                load_mice_database, "fetch_trials", side_effect=_fake_fetch_trials
            ),
        )

    def test_train_and_heldout_splits_are_disjoint(self):
        sel, fetch = self._patches()
        with sel, fetch:
            _train_df, train_ids = load_mice_from_database(
                split="train", curricula=["Coupled Baiting"], subject_ratio=1.0, seed=0
            )
            _held_df, held_ids = load_mice_from_database(
                split="heldout", curricula=["Coupled Baiting"]
            )
        self.assertEqual(set(train_ids) & set(held_ids), set())
        self.assertEqual(held_ids, ["c04", "c09", "c14", "c19"])
        self.assertEqual(len(train_ids), 20)

    def test_output_contract_columns_and_ses_idx(self):
        sel, fetch = self._patches()
        with sel, fetch:
            df, _ = load_mice_from_database(split="heldout", curricula=["Coupled Baiting"])
        self.assertEqual(list(df.columns), DEFAULT_COLS)
        self.assertIn("ses_idx", df.columns)
        self.assertNotIn("session_id", df.columns)
        self.assertEqual(df["subject_id"].dtype, object)

    def test_mature_only_filters_returned_trials_not_selection(self):
        # c00 has alternating mature / non-mature sessions; selection is unaffected,
        # but mature_only controls how many of its sessions' trials are returned.
        df_mixed = _make_session_df(
            [(f"c{i:02d}", "Coupled Baiting", 12) for i in range(6)],
            stage=lambda subj, k: "STAGE_FINAL" if k % 2 == 0 else "STAGE_2",
        )
        sel, fetch = self._patches(sessions=df_mixed)
        with sel, fetch:
            df_mature, ids_mature = load_mice_from_database(
                split="train", curricula=["Coupled Baiting"], subject_ratio=1.0,
                seed=0, mature_only=True,
            )
        sel2, fetch2 = self._patches(sessions=df_mixed)
        with sel2, fetch2:
            df_all, ids_all = load_mice_from_database(
                split="train", curricula=["Coupled Baiting"], subject_ratio=1.0,
                seed=0, mature_only=False,
            )
        # Same subjects selected regardless of mature_only ...
        self.assertEqual(set(ids_mature), set(ids_all))
        # ... but fewer trials returned when restricted to mature sessions.
        self.assertLess(len(df_mature), len(df_all))

    def test_subject_ids_override_bypasses_pipeline(self):
        captured = {}

        def _capturing_select(subjects=None, columns=None, **kwargs):
            captured["subjects"] = subjects
            df = _RANKING_DF
            if subjects is not None:
                df = df[df["subject_id"].isin([str(s) for s in subjects])]
            return df.copy()

        with mock.patch.object(
            load_mice_database, "select_sessions", side_effect=_capturing_select
        ), mock.patch.object(
            load_mice_database, "fetch_trials", side_effect=_fake_fetch_trials
        ):
            df, ids = load_mice_from_database(subject_ids=["c00", "c01"])
        self.assertEqual(ids, ["c00", "c01"])
        self.assertEqual(set(df["subject_id"]), {"c00", "c01"})
        # Direct selection scopes the session read to those subjects.
        self.assertEqual(set(captured["subjects"]), {"c00", "c01"})

    def test_invalid_split_raises(self):
        with self.assertRaises(ValueError):
            load_mice_from_database(split="bogus")


if __name__ == "__main__":
    unittest.main()
