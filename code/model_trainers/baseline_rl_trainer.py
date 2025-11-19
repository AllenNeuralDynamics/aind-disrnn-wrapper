from __future__ import annotations

from base.interfaces import ModelTrainer
from base.types import DatasetBundle


class BaselineRLTrainer(ModelTrainer):
    """Placeholder trainer for baseline RL comparisons."""

    def __init__(self, seed: int | None = None, **settings: object) -> None:
        super().__init__(seed=seed)
        self.settings = settings

    def fit(self, bundle: DatasetBundle, loggers: dict[str, object] | None = None):
        raise NotImplementedError(
            "Baseline RL trainer is not implemented yet; config received:"
            f" {self.settings}"
        )
