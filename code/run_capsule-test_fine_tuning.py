"""DEPRECATED scratch script — superseded by the unified run_eval.py CLI.

This previously passed an inline config dict to run_heldout_subject_finetuning_from_config.
Move that dict into a YAML (see configs/config_heldout_subject_finetuning.yaml) and run:

    python run_eval.py finetune --config configs/config_heldout_subject_finetuning.yaml \\
        [--output-root /results/heldout_subject_finetuning]

(The standalone run_heldout_subject_finetuning.py argparse wrapper is also still available.)
"""

import sys

_EQUIVALENT = (
    "python run_eval.py finetune --config configs/config_heldout_subject_finetuning.yaml "
    "[--output-root /results/heldout_subject_finetuning]"
)


def main() -> None:
    sys.stderr.write(
        "DEPRECATED: superseded by the unified run_eval.py CLI.\n"
        f"Equivalent:\n    {_EQUIVALENT}\n"
        "See `python run_eval.py finetune --help`.\n"
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
