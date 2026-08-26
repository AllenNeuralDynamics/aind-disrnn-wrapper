"""Off-curriculum mice must not break the generative rollout.

Some mice were trained OFF-CURRICULUM and carry no ``curriculum_name``. The data split already
handles this — ``load_mice_database._partition_subjects`` step 3 assigns each subject its
most-common ``task`` as its ``subject_curriculum``, and every training cohort was built from THAT.
The generative rollout instead read the per-session ``curriculum_name`` and hard-raised on the NaN:

    ValueError: Unsupported curriculum_name=nan (no coupled/uncoupled family match)

making ``run_analysis generative`` unusable on any cohort containing an off-curriculum mouse
(13/15 runs of studies/05-disrnn-scaling-law).
"""
from __future__ import annotations

import unittest

import pandas as pd

from post_training_analysis.generative_analysis import _fill_offcurriculum_curriculum_name


class TestOffCurriculumCurriculumName(unittest.TestCase):
    def test_offcurriculum_session_gets_its_own_task(self):
        """Each off-curriculum session is matched to the task the animal actually ran.

        A per-subject label would be wrong here: this is a per-session, curriculum-MATCHED rollout,
        and a subject's task changes across stages.
        """
        df = pd.DataFrame({
            "subject_id": ["B"] * 5,
            "curriculum_name": [None] * 5,
            "task": ["Uncoupled Baiting"] * 3 + ["Coupled Baiting"] * 2,
        })
        out = _fill_offcurriculum_curriculum_name(df)
        self.assertEqual(list(out["curriculum_name"]),
                         ["Uncoupled Baiting"] * 3 + ["Coupled Baiting"] * 2)

    def test_oncurriculum_subject_is_untouched_even_when_its_task_varies(self):
        """curriculum_name stays authoritative where present.

        This is the case that makes the modal task a FALLBACK and not a replacement: one
        curriculum can involve DIFFERENT TASKS AT DIFFERENT STAGES, so a subject's per-session
        `task` may disagree with its curriculum. Overwriting curriculum_name from the task would
        silently rewrite on-curriculum sessions -- and change prior results (study 01's r9).
        """
        df = pd.DataFrame({
            "subject_id": ["A"] * 4,
            "curriculum_name": ["Coupled Baiting"] * 4,
            "task": ["Coupled Baiting"] * 2 + ["Uncoupled Baiting"] * 2,  # stage change
        })
        out = _fill_offcurriculum_curriculum_name(df)
        self.assertEqual(list(out["curriculum_name"].unique()), ["Coupled Baiting"])

    def test_mixed_cohort(self):
        """On- and off-curriculum subjects in one frame: each resolved on its own terms."""
        df = pd.DataFrame({
            "subject_id": ["A", "A", "B", "B", "B"],
            "curriculum_name": ["Coupled Baiting", "Coupled Baiting", None, None, None],
            "task": ["Coupled Baiting", "Uncoupled Baiting",
                     "Uncoupled Baiting", "Uncoupled Baiting", "Coupled Baiting"],
        })
        out = _fill_offcurriculum_curriculum_name(df)
        self.assertTrue(out["curriculum_name"].notna().all())
        # A is ON-curriculum: authoritative curriculum_name kept, even though its per-session
        # task varies across stages.
        self.assertEqual(set(out.loc[out.subject_id == "A", "curriculum_name"]), {"Coupled Baiting"})
        # B is OFF-curriculum: each session takes the task the animal ACTUALLY ran, so its third
        # (coupled) session is no longer collapsed onto the subject's modal uncoupled task.
        self.assertEqual(list(out.loc[out.subject_id == "B", "curriculum_name"]),
                         ["Uncoupled Baiting", "Uncoupled Baiting", "Coupled Baiting"])

    def test_literal_none_string_is_treated_as_missing_not_defaulted(self):
        """The DB stores off-curriculum sessions as the STRING "None".

        _build_curriculum_matched_task maps that to its `norm in ("", "none")` branch and SILENTLY
        rolls the session out as a default UncoupledBlockTask(reward_baiting=True) -- even when the
        animal actually ran Coupled Baiting. That is the quiet version of the crash: in one D=10
        cohort, 42/249 sessions (17%) were simulated with the wrong task family, unwarned.
        """
        df = pd.DataFrame({
            "subject_id": ["B"] * 3,
            "curriculum_name": ["None", "None", "None"],   # literal string, NOT null
            "task": ["Coupled Baiting", "Coupled Baiting", "Coupled Without Baiting"],
        })
        out = _fill_offcurriculum_curriculum_name(df)
        # each session keeps its OWN task, so the coupled sessions are no longer
        # silently simulated as uncoupled
        self.assertEqual(list(out["curriculum_name"]),
                         ["Coupled Baiting", "Coupled Baiting", "Coupled Without Baiting"])

    def test_nan_null_is_handled_too(self):
        """`curriculum_name is not None` misses float NaN -- that is what raised in production."""
        import numpy as np
        df = pd.DataFrame({
            "subject_id": ["B"] * 2,
            "curriculum_name": [np.nan, np.nan],
            "task": ["Uncoupled Baiting", "Uncoupled Baiting"],
        })
        out = _fill_offcurriculum_curriculum_name(df)
        self.assertEqual(list(out["curriculum_name"].unique()), ["Uncoupled Baiting"])

    def test_falls_back_to_modal_task_when_the_session_has_none(self):
        df = pd.DataFrame({
            "subject_id": ["B"] * 3,
            "curriculum_name": [None, None, None],
            "task": ["Uncoupled Baiting", "Uncoupled Baiting", None],  # 3rd session has no task
        })
        out = _fill_offcurriculum_curriculum_name(df)
        self.assertEqual(list(out["curriculum_name"].unique()), ["Uncoupled Baiting"])

    def test_no_curriculum_and_no_task_raises_rather_than_guessing(self):
        """If neither field is available we must NOT invent a task family."""
        df = pd.DataFrame({"subject_id": ["C"], "curriculum_name": [None], "task": [None]})
        with self.assertRaises(ValueError):
            _fill_offcurriculum_curriculum_name(df)

    def test_noop_when_nothing_is_missing(self):
        df = pd.DataFrame({
            "subject_id": ["A"], "curriculum_name": ["Coupled Baiting"], "task": ["Coupled Baiting"],
        })
        pd.testing.assert_frame_equal(_fill_offcurriculum_curriculum_name(df), df)


if __name__ == "__main__":
    unittest.main()


class TestCurriculumMatchedTaskFamilies(unittest.TestCase):
    """The rollout must build the family the animal actually ran — including Random Walk."""

    def _build(self, name):
        from post_training_analysis.generative_analysis import _build_curriculum_matched_task
        return _build_curriculum_matched_task(curriculum_name=name, n_trials=50, seed=0)

    def test_random_walk_builds_the_gym_random_walk_task(self):
        """Random Walk is rare but REAL in training (9 sessions / 8,284 trials at D=614).

        It used to fall through to the `norm in ("", "none")` branch and be silently simulated as
        a default UNCOUPLED BAITING task — a completely different reward structure.
        """
        from aind_behavior_gym.dynamic_foraging.task import RandomWalkTask
        for name in ("Random Walk", "random walk", "RandomWalk"):
            self.assertIsInstance(self._build(name), RandomWalkTask, msg=name)

    def test_block_families_still_resolve(self):
        from aind_behavior_gym.dynamic_foraging.task import CoupledBlockTask, UncoupledBlockTask
        self.assertIsInstance(self._build("Coupled Baiting"), CoupledBlockTask)
        self.assertIsInstance(self._build("Uncoupled Baiting"), UncoupledBlockTask)
        self.assertIsInstance(self._build("Uncoupled Without Baiting"), UncoupledBlockTask)
        self.assertIsInstance(self._build("Coupled Without Baiting"), CoupledBlockTask)
        # baiting flag is read from the name
        self.assertFalse(self._build("Coupled Without Baiting").reward_baiting)
        self.assertTrue(self._build("Coupled Baiting").reward_baiting)

    def test_unknown_family_still_raises(self):
        with self.assertRaises(ValueError):
            self._build("Some Brand New Task")
