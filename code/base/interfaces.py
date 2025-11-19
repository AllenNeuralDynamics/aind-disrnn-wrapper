from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .types import DatasetBundle
import time
import logging


class DatasetLoader(ABC):
    """Protocol for building datasets from configuration."""

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is None:
            self.seed = int(time.time())
            logging.info(f"No seed set for DatasetLoader, using {self.seed}")
        else:
            self.seed = seed
            logging.info(f"Using seed {self.seed} for DatasetLoader")

    @abstractmethod
    def load(self) -> DatasetBundle:
        """Materialize the dataset bundle ready for training."""


class ModelTrainer(ABC):
    """Protocol for fitting a model given a dataset bundle."""

    def __init__(self, seed: Optional[int] = None) -> None:
        if seed is None:
            self.seed = int(time.time())
            logging.info(f"No seed set for ModelTrainer, using {self.seed}")
        else:
            self.seed = seed
            logging.info(f"Using seed {self.seed} for ModelTrainer")
            
    @abstractmethod
    def fit(self, bundle: DatasetBundle, loggers: Optional[Dict[str, Any]] = None) -> Any:
        """Run training and return a structured result payload."""
