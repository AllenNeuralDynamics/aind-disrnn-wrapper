"""Unit tests for run-name helper utilities."""

from __future__ import annotations

import unittest

try:
    from omegaconf import OmegaConf

    from utils.run_helpers import apply_dynamic_run_name_components

    RUN_HELPERS_DEPS_AVAILABLE = True
    RUN_HELPERS_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    RUN_HELPERS_DEPS_AVAILABLE = False
    RUN_HELPERS_IMPORT_ERROR = exc


@unittest.skipUnless(
    RUN_HELPERS_DEPS_AVAILABLE,
    f"run helper dependencies unavailable: {RUN_HELPERS_IMPORT_ERROR}",
)
class TestRunHelpers(unittest.TestCase):
    def test_apply_dynamic_run_name_components_appends_multisubject_once(self):
        cfg = OmegaConf.create(
            {
                "data": {
                    "multisubject": True,
                    "run_name_component": "mice_snapshot",
                },
                "model": {
                    "run_name_component": "disrnn_beta0.001_lr0.001",
                    "architecture": {"multisubject": True},
                },
            }
        )

        apply_dynamic_run_name_components(cfg)

        self.assertEqual(cfg.data.run_name_component, "mice_snapshot_multisubject")
        self.assertEqual(
            cfg.model.run_name_component,
            "disrnn_beta0.001_lr0.001_multisubject",
        )

        apply_dynamic_run_name_components(cfg)

        self.assertEqual(cfg.data.run_name_component, "mice_snapshot_multisubject")
        self.assertEqual(
            cfg.model.run_name_component,
            "disrnn_beta0.001_lr0.001_multisubject",
        )


if __name__ == "__main__":
    unittest.main()
