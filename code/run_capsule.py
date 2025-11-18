"""Entry point for disRNN wrapper experiments."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from capsule_core.interfaces import DatasetLoader, ModelTrainer
from capsule_core.types import TrainerResult

logger = logging.getLogger(__name__)


def find_hydra_config() -> Path | None:
    """Locate the first config.yaml under /data/jobs."""

    candidates = list(Path("/data/jobs").rglob("config.yaml"))
    if not candidates:
        logger.warning("No config.yaml found under /data/jobs/")
        return None
    if len(candidates) > 1:
        logger.warning("Multiple config.yaml files found: %s. Using the first one.", candidates)
    return candidates[0]


def copy_input_folder(config_path: Path) -> None:
    source_dir = config_path.resolve().parents[1]
    destination_root = Path("/results/input")
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_dir = destination_root / source_dir.name
    logger.info("Copying Hydra inputs from %s to %s", source_dir, destination_dir)
    import shutil

    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)


def save_resolved_config(config: DictConfig, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=config, f=str(destination), resolve=True)
    logger.info("Saved resolved config to %s", destination)


def persist_output(output_obj: Any, destination: Path) -> None:
    if output_obj is None:
        logger.info("Trainer returned no output payload; skipping persistence.")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(output_obj, "model_dump_json"):
        payload = output_obj.model_dump_json(indent=2)
    elif hasattr(output_obj, "model_dump"):
        payload = json.dumps(output_obj.model_dump(), indent=2)
    else:
        payload = json.dumps(output_obj, indent=2, default=_json_default)
    destination.write_text(payload)
    logger.info("Wrote trainer output to %s", destination)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def configure_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(levelname)s:%(asctime)s:%(filename)s:%(lineno)d:    %(message)s",
        datefmt="%Y-%m-%d %H-%M-%S",
    )


def main() -> None:
    configure_logger()

    config_path = find_hydra_config()
    if config_path is None:
        logger.error("No config.yaml found. Exiting.")
        sys.exit(1)

    copy_input_folder(config_path)

    logger.info("Loading Hydra config from %s", config_path)
    hydra_config = OmegaConf.load(config_path)

    seed = hydra_config.get("seed") if isinstance(hydra_config, DictConfig) else None
    if seed is None:
        seed = int(time.time())
        logger.warning("No seed provided in config; using fallback seed %s", seed)
    hydra_config.seed = seed

    if "data" not in hydra_config or "model" not in hydra_config:
        raise KeyError("Hydra config must contain both 'data' and 'model' sections.")

    hydra_config.data.seed = hydra_config.data.get("seed", seed)
    hydra_config.model.seed = hydra_config.model.get("seed", seed)
    hydra_config.model.output_dir = hydra_config.model.get("output_dir", "/results")

    save_resolved_config(hydra_config, Path("/results/inputs.yaml"))

    dataset_loader: DatasetLoader = instantiate(hydra_config.data)
    dataset_bundle = dataset_loader.load()
    logger.info("Loaded dataset bundle with metadata: %s", dataset_bundle.metadata)

    model_trainer: ModelTrainer = instantiate(hydra_config.model)
    trainer_result: TrainerResult = model_trainer.fit(dataset_bundle)

    persist_output(trainer_result.output, Path("/results/outputs.json"))
    logger.info("All done, goodbye")


if __name__ == "__main__":
    main()
