"""DEPRECATED scratch script — superseded by the unified run_eval.py CLI.

(The standalone run_embedding_space_analysis.py argparse wrapper is also still available;
both now route to the same run_embedding_space_analysis function.)

Equivalent:
    python run_eval.py embedding --model-dir <RUN_DIR> \\
        --checkpoint-policy best_eval --output-dir /results --task-column task
"""

import sys

_EQUIVALENT = (
    "python run_eval.py embedding --model-dir <RUN_DIR> "
    "--checkpoint-policy best_eval --output-dir /results"
)


def main() -> None:
    sys.stderr.write(
        "DEPRECATED: superseded by the unified run_eval.py CLI.\n"
        f"Equivalent:\n    {_EQUIVALENT}\n"
        "See `python run_eval.py embedding --help`.\n"
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
