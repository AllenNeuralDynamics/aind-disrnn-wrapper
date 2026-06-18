"""DEPRECATED scratch script — superseded by the unified run_analysis.py CLI.

Previously hard-coded a `run_post_training_analysis(...)` call with fixed /data and
/results paths. Use the standalone CLI, which takes the trained-run directory + options
as arguments and loads everything from the saved run.

Equivalent:
    python run_analysis.py generative --model-dir <RUN_DIR> \\
        --split train --checkpoint-policy best_eval --rollout-mode curriculum_matched \\
        --n-rollouts-per-session 5 --window-size 10 --output-dir /results \\
        --session-partitions train eval combined
"""

import sys

_EQUIVALENT = (
    "python run_analysis.py generative --model-dir <RUN_DIR> "
    "--split train --checkpoint-policy best_eval --output-dir /results"
)


def main() -> None:
    sys.stderr.write(
        "DEPRECATED: superseded by the unified run_analysis.py CLI.\n"
        f"Equivalent:\n    {_EQUIVALENT}\n"
        "See `python run_analysis.py generative --help`.\n"
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
