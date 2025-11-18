from __future__ import annotations

from capsule_core.interfaces import ModelTrainer
from capsule_core.types import DatasetBundle, TrainerResult


class BaselineRLTrainer(ModelTrainer):
    """Placeholder trainer for baseline RL comparisons."""

    def __init__(self, seed: int | None = None, **settings: object) -> None:
        super().__init__(seed=seed)
        self.settings = settings

    def fit(self, bundle: DatasetBundle, loggers: dict[str, object] | None = None) -> TrainerResult:
        raise NotImplementedError(
            "Baseline RL trainer is not implemented yet; config received:"
            f" {self.settings}"
        )
