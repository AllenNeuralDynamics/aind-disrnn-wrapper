from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from post_training_analysis import run_heldout_subject_finetuning_from_config
from utils.run_helpers import configure_sys_logger

logger = logging.getLogger(__name__)


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
    config_path = Path(args.config).expanduser()
    output_root = (
        None
        if args.output_root in (None, "")
        else Path(args.output_root).expanduser()
    )
    logger.info(
        "Launching held-out subject fine-tuning with config=%s output_root_override=%s",
        config_path,
        output_root,
    )
    result = run_heldout_subject_finetuning_from_config(
        config_path,
        output_root=output_root,
    )
    logger.info(
        "Held-out subject fine-tuning finished. output_dir=%s summary=%s checkpoint_metrics=%s",
        result.get("output_dir"),
        result.get("summary_path"),
        result.get("checkpoint_metrics_path"),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
