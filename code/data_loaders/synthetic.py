from __future__ import annotations

from capsule_core.interfaces import DatasetLoader
from capsule_core.types import DatasetBundle


class SyntheticDatasetLoader(DatasetLoader):
    """Placeholder loader for synthetic experiments."""

    def __init__(self, seed: int | None = None, **settings: object) -> None:
        super().__init__(seed=seed)
        self.settings = settings

    def load(self) -> DatasetBundle:
        raise NotImplementedError(
            "Synthetic dataset loading is not implemented yet; config received:"
            f" {self.settings}"
        )
