"""Helpers for converting between DictConfig and JSON-serializable dicts."""

from omegaconf import DictConfig, OmegaConf


def dictconfig_to_json(cfg: DictConfig) -> dict:
    """Convert a Hydra DictConfig to a JSON-serializable mapping."""

    return OmegaConf.to_container(cfg, resolve=True)


def json_to_dictconfig(json_data: dict) -> DictConfig:
    """Build a DictConfig from a JSON-compatible dict."""

    return OmegaConf.create(json_data)
