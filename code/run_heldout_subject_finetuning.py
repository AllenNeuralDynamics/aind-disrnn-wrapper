from __future__ import annotations

import argparse
import json
from pathlib import Path

from post_training_analysis import run_heldout_subject_finetuning_from_config
from utils.run_helpers import configure_sys_logger


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune held-out subject embeddings for multisubject GRU/disRNN runs.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the standalone held-out fine-tuning YAML config.",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Optional override for output.output_root in the config.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    configure_sys_logger()
    result = run_heldout_subject_finetuning_from_config(
        Path(args.config).expanduser(),
        output_root=(
            None
            if args.output_root in (None, "")
            else Path(args.output_root).expanduser()
        ),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
