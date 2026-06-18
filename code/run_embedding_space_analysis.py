from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from post_training_analysis import run_embedding_space_analysis
from utils.run_helpers import configure_sys_logger

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone embedding-space analysis for multisubject GRU/disRNN runs.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the trained model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    parser.add_argument(
        "--checkpoint-policy",
        default="best_eval",
        help="Checkpoint resolution policy (default: best_eval).",
    )
    parser.add_argument(
        "--task-column",
        default="curriculum_name",
        help="df_han column to use as the task label.",
    )
    parser.add_argument(
        "--room-column",
        default="room",
        help="df_han column to use for room labels.",
    )
    parser.add_argument(
        "--weekday-column",
        default="weekday",
        help="df_han column to use for weekday labels.",
    )
    parser.add_argument(
        "--foraging-eff-column",
        default="foraging_eff_random_seed",
        help="df_han column to use for foraging efficiency values.",
    )
    parser.add_argument(
        "--bias-naive-column",
        default="bias_naive",
        help="df_han column to use for bias_naive values.",
    )
    parser.add_argument(
        "--reaction-time-column",
        default="reaction_time_median",
        help="df_han column to use for reaction_time_median values.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    configure_sys_logger()

    output_dir = None if args.output_dir in (None, "") else Path(args.output_dir).expanduser()
    logger.info(
        "Launching embedding-space analysis with model_dir=%s output_dir=%s checkpoint_policy=%s",
        args.model_dir,
        output_dir,
        args.checkpoint_policy,
    )
    result = run_embedding_space_analysis(
        args.model_dir,
        output_dir=output_dir,
        checkpoint_policy=args.checkpoint_policy,
        task_column=args.task_column,
        room_column=args.room_column,
        weekday_column=args.weekday_column,
        foraging_eff_column=args.foraging_eff_column,
        bias_naive_column=args.bias_naive_column,
        reaction_time_column=args.reaction_time_column,
    )
    logger.info(
        "Embedding-space analysis finished. output_dir=%s summary=%s",
        result.get("output_dir"),
        result.get("summary_path"),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
