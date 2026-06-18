"""DEPRECATED scratch script — superseded by the unified run_eval.py CLI.

Equivalent (top-level likelihood-advantage analysis):
    python run_eval.py likelihood-advantage \\
        --model1-dir <RUN_A> --model2-dir <RUN_B> \\
        --split combined --checkpoint-policy best_eval --output-dir /results

The finer-grained state-space sub-analyses this script also exercised
(run_rnn_state_space_subject_analysis, run_rnn_state_space_condition_analysis,
run_baseline_q_space_subject_analysis, run_subject_embedding_baseline_parameter_analysis)
are not yet exposed as CLI sub-commands; call them via the Python API:
    from post_training_analysis import run_rnn_state_space_subject_analysis  # etc.
"""

import sys

_EQUIVALENT = (
    "python run_eval.py likelihood-advantage --model1-dir <RUN_A> --model2-dir <RUN_B> "
    "--split combined --checkpoint-policy best_eval --output-dir /results"
)


def main() -> None:
    sys.stderr.write(
        "DEPRECATED: superseded by the unified run_eval.py CLI.\n"
        f"Equivalent:\n    {_EQUIVALENT}\n"
        "See `python run_eval.py likelihood-advantage --help`.\n"
        "State-space sub-analyses remain available via the post_training_analysis Python API.\n"
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
