"""DEPRECATED scratch script — superseded by the unified run_analysis.py CLI.

Equivalent:
    python run_analysis.py likelihood-comparison \\
        --model-dirs <RUN_A> <RUN_B> ... \\
        --checkpoint-policy best_eval --output-dir /results \\
        --model-labels "label A" "label B" ... \\
        --precomputed-session-metrics /code/session_metrics.pkl \\
        --no-include-heldout
"""

import sys

_EQUIVALENT = (
    "python run_analysis.py likelihood-comparison --model-dirs <RUN_A> <RUN_B> ... "
    "--checkpoint-policy best_eval --output-dir /results [--no-include-heldout]"
)


def main() -> None:
    sys.stderr.write(
        "DEPRECATED: superseded by the unified run_analysis.py CLI.\n"
        f"Equivalent:\n    {_EQUIVALENT}\n"
        "See `python run_analysis.py likelihood-comparison --help`.\n"
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
