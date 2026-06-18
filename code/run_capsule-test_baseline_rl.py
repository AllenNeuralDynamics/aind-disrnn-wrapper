"""DEPRECATED scratch script — superseded by the unified run_eval.py CLI.

Equivalent:
    python run_eval.py baseline-rl \\
        --resolved-run /data/resolved_run-....json \\
        --fitting-df /data/df_baseline_rl_fitting_....pkl \\
        --model-aliases QLearning_L1F1_CK1_softmax QLearning_L2F1_softmax ForagingCompareThreshold \\
        --output-dir /results --n-rollouts-per-session 5
"""

import sys

_EQUIVALENT = (
    "python run_eval.py baseline-rl --resolved-run <JSON> --fitting-df <PKL> "
    "--model-aliases <ALIAS> ... --output-dir /results"
)


def main() -> None:
    sys.stderr.write(
        "DEPRECATED: superseded by the unified run_eval.py CLI.\n"
        f"Equivalent:\n    {_EQUIVALENT}\n"
        "See `python run_eval.py baseline-rl --help`.\n"
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
