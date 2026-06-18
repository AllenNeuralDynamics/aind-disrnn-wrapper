"""DEPRECATED scratch script — superseded by the unified run_eval.py CLI.

Equivalent:
    python run_eval.py from-histories \\
        --simulated-history <DIR>/simulated_session_history.pkl \\
        --resolved-run <DIR>/resolved_run.json \\
        --output-dir /results --window-size 10 --session-partitions train eval combined
"""

import sys

_EQUIVALENT = (
    "python run_eval.py from-histories --simulated-history <PKL> "
    "--resolved-run <JSON> --output-dir /results [--session-partitions train eval combined]"
)


def main() -> None:
    sys.stderr.write(
        "DEPRECATED: superseded by the unified run_eval.py CLI.\n"
        f"Equivalent:\n    {_EQUIVALENT}\n"
        "See `python run_eval.py from-histories --help`.\n"
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
