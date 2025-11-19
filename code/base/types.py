from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class DatasetBundle:
    """Container for dataset material passed from loaders to trainers."""

    raw: Any
    train: Any
    eval: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerResult:
    """Standard payload returned by model trainers."""

    output: Any = None
    metrics: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)
