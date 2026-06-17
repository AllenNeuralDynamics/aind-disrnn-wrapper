"""Unit tests for run-name helper utilities."""

from __future__ import annotations

import unittest

try:
    from omegaconf import OmegaConf

    from utils.run_helpers import (
        apply_dynamic_run_name_components,
        apply_model_penalty_multipliers,
        resolve_disrnn_penalties,
        resolve_heldout_test_likelihood,
    )

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
    def test_resolve_disrnn_penalties_uses_beta_as_default_base(self):
        resolved = resolve_disrnn_penalties(
            {
                "beta": 1e-3,
                "latent_penalty": 1e-3,
                "update_net_latent_penalty_multiplier": 10,
            }
        )

        self.assertEqual(resolved["update_net_latent_penalty"], 1e-2)
        self.assertEqual(resolved["latent_penalty"], 1e-3)
        self.assertNotIn("update_net_latent_penalty_multiplier", resolved)

    def test_apply_model_penalty_multipliers_updates_disrnn_config(self):
        cfg = OmegaConf.create(
            {
                "model": {
                    "type": "disrnn",
                    "penalties": {
                        "beta": 1e-3,
                        "update_net_latent_penalty": "${.beta}",
                        "update_net_latent_penalty_multiplier": 10,
                    },
                }
            }
        )

        apply_model_penalty_multipliers(cfg)

        self.assertEqual(cfg.model.penalties.update_net_latent_penalty, 1e-2)
        self.assertFalse("update_net_latent_penalty_multiplier" in cfg.model.penalties)

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


    # --- Phase A regression guards (bugs #3/#4) -------------------------------

    def test_resolve_heldout_test_likelihood_reads_test_likelihood(self):
        # Fresh-eval summaries use "test_likelihood".
        self.assertEqual(
            resolve_heldout_test_likelihood({"test_likelihood": 0.5}), 0.5
        )

    def test_resolve_heldout_test_likelihood_reads_dedup_key(self):
        # The dedup-hit path reuses a checkpoint summary keyed differently.
        self.assertEqual(
            resolve_heldout_test_likelihood({"heldout_test_likelihood": 0.7}), 0.7
        )

    def test_resolve_heldout_test_likelihood_handles_missing_and_non_dict(self):
        # Bug #3/#4: failure-fallback summaries and non-dicts must not blow up.
        self.assertIsNone(
            resolve_heldout_test_likelihood({"enabled": True, "evaluation_failed": True})
        )
        self.assertIsNone(resolve_heldout_test_likelihood(None))
        self.assertIsNone(resolve_heldout_test_likelihood("not a dict"))

    def test_resolve_heldout_test_likelihood_zero_is_preserved(self):
        # 0.0 is a valid likelihood and must not be treated as "missing".
        self.assertEqual(
            resolve_heldout_test_likelihood({"test_likelihood": 0.0}), 0.0
        )


if __name__ == "__main__":
    unittest.main()
