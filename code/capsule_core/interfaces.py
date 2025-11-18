from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .types import DatasetBundle, TrainerResult


class DatasetLoader(ABC):
    """Protocol for building datasets from configuration."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    @abstractmethod
    def load(self) -> DatasetBundle:
        """Materialize the dataset bundle ready for training."""


class ModelTrainer(ABC):
    """Protocol for fitting a model given a dataset bundle."""

    def __init__(self, seed: Optional[int] = None) -> None:
        self.seed = seed

    @abstractmethod
    def fit(self, bundle: DatasetBundle, loggers: Optional[Dict[str, Any]] = None) -> TrainerResult:
        """Run training and return a structured result payload."""
