"""Tests for the BFM_*/DISRNN_* env-var resolver (expand step of the rename).

See AllenNeuralDynamics/aind-disrnn-dispatcher#80 and ADR-0007.
"""

import os
import sys
import unittest
import warnings
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.env_config import (  # noqa: E402
    LEGACY_PREFIX,
    PREFIX,
    _reset_deprecation_warnings,
    get_env,
    legacy_name,
)

NEW = "BFM_TEST_VARIABLE"
LEGACY = "DISRNN_TEST_VARIABLE"


class TestLegacyName(unittest.TestCase):
    def test_maps_prefix_and_keeps_suffix(self):
        self.assertEqual(legacy_name(NEW), LEGACY)
        self.assertEqual(legacy_name(PREFIX + "A_B"), LEGACY_PREFIX + "A_B")

    def test_rejects_a_name_without_the_new_prefix(self):
        with self.assertRaises(ValueError):
            legacy_name(LEGACY)


class TestGetEnv(unittest.TestCase):
    def setUp(self):
        _reset_deprecation_warnings()

    @mock.patch.dict(os.environ, {NEW: "new"}, clear=True)
    def test_new_only(self):
        self.assertEqual(get_env(NEW), "new")

    @mock.patch.dict(os.environ, {LEGACY: "legacy"}, clear=True)
    def test_legacy_only_is_honoured(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(get_env(NEW), "legacy")
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, DeprecationWarning)
        # The warning has to name the variable, or it is not actionable.
        self.assertIn(LEGACY, str(caught[0].message))
        self.assertIn(NEW, str(caught[0].message))

    @mock.patch.dict(os.environ, {NEW: "new", LEGACY: "legacy"}, clear=True)
    def test_new_wins_over_legacy_and_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(get_env(NEW), "new")
        self.assertEqual(caught, [])

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_neither_set_returns_the_default(self):
        self.assertIsNone(get_env(NEW))
        self.assertEqual(get_env(NEW, "fallback"), "fallback")

    @mock.patch.dict(os.environ, {LEGACY: "legacy"}, clear=True)
    def test_legacy_warns_once_per_variable(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_env(NEW)
            get_env(NEW)
            get_env(NEW)
        self.assertEqual(len(caught), 1)

    @mock.patch.dict(os.environ, {LEGACY: ""}, clear=True)
    def test_empty_legacy_value_is_set_not_absent(self):
        # "" is a value: falling through to the default here would silently
        # change behaviour for anyone clearing a variable by emptying it.
        self.assertEqual(get_env(NEW, "fallback"), "")


class TestNoStrayLegacyReads(unittest.TestCase):
    """Every DISRNN_* read must go through the resolver, not os.environ."""

    def test_no_direct_os_environ_reads_of_legacy_names(self):
        import re

        code_root = Path(__file__).resolve().parents[1]
        pattern = re.compile(
            r"os\.environ(?:\.get\(|\[)\s*[\"']DISRNN_[A-Z0-9_]+", re.MULTILINE
        )
        offenders = []
        for path in sorted(code_root.rglob("*.py")):
            if path.name == "env_config.py" or "tests" in path.parts:
                continue
            if pattern.search(path.read_text(encoding="utf-8", errors="replace")):
                offenders.append(str(path.relative_to(code_root)))
        self.assertEqual(
            offenders,
            [],
            f"legacy env reads bypassing utils.env_config.get_env: {offenders}",
        )


if __name__ == "__main__":
    unittest.main()
