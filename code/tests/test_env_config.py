"""Tests for the BFM_* env-var resolver (contract step of the rename).

The legacy DISRNN_* fallback was removed in
AllenNeuralDynamics/aind-dynamic-foraging-bfm-dispatcher#82; the tests that
covered it went with it. The guard test below stays, because it is what stops a
scattered os.environ lookup reintroducing the legacy prefix. See ADR-0007.
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.env_config import PREFIX, get_env  # noqa: E402

NEW = "BFM_TEST_VARIABLE"
LEGACY = "DISRNN_TEST_VARIABLE"


class TestGetEnv(unittest.TestCase):
    @mock.patch.dict(os.environ, {NEW: "new"}, clear=True)
    def test_reads_the_new_prefix(self):
        self.assertEqual(get_env(NEW), "new")

    @mock.patch.dict(os.environ, {}, clear=True)
    def test_absent_returns_the_default(self):
        self.assertIsNone(get_env(NEW))
        self.assertEqual(get_env(NEW, "fallback"), "fallback")

    @mock.patch.dict(os.environ, {NEW: ""}, clear=True)
    def test_empty_value_is_set_not_absent(self):
        self.assertEqual(get_env(NEW, "fallback"), "")

    @mock.patch.dict(os.environ, {LEGACY: "legacy"}, clear=True)
    def test_legacy_prefix_is_no_longer_honoured(self):
        # The contract step: a job launched from a pre-migration dispatcher now
        # gets the default, not its legacy value.
        self.assertIsNone(get_env(NEW))
        self.assertEqual(get_env(NEW, "fallback"), "fallback")

    def test_rejects_a_name_without_the_new_prefix(self):
        with self.assertRaises(ValueError):
            get_env(LEGACY)
        self.assertTrue(NEW.startswith(PREFIX))


class TestNoStrayLegacyReads(unittest.TestCase):
    """No os.environ lookup may reintroduce the legacy prefix."""

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
