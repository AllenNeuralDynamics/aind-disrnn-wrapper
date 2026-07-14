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
    def test_offcurriculum_subject_gets_its_modal_task(self):
        """A subject with no curriculum_name is labelled with its most-common task."""
        df = pd.DataFrame({
            "subject_id": ["B"] * 5,
            "curriculum_name": [None] * 5,
            # modal task = Uncoupled Baiting (3 rows vs 2)
            "task": ["Uncoupled Baiting"] * 3 + ["Coupled Baiting"] * 2,
        })
        out = _fill_offcurriculum_curriculum_name(df)
        self.assertEqual(list(out["curriculum_name"].unique()), ["Uncoupled Baiting"])

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
        self.assertEqual(set(out.loc[out.subject_id == "A", "curriculum_name"]), {"Coupled Baiting"})
        self.assertEqual(set(out.loc[out.subject_id == "B", "curriculum_name"]), {"Uncoupled Baiting"})

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
