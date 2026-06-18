"""Model-agnostic held-out evaluation configuration.

``HeldoutEvalConfig`` bridges a (Hydra or plain-dict) run config to the held-out
evaluation logic. It is shared by disRNN, GRU, and baseline-RL evaluation as well as by
``run_capsule`` and the trainers, so it lives in a neutral module rather than inside a
model-specific one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from utils.load_mice_database import MICE_DATABASE_DATA_TYPES


def _cfg_get(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


@dataclass(frozen=True)
class HeldoutEvalConfig:
    test_subject_ids: list[Any] | None = None
    min_sessions: int = 10
    heldout_every_n: int = 5
    heldout_eval: bool | None = None  # None -> auto (enabled for mice-database runs)
    data_type: str | None = None
    mature_only: bool = True
    curricula: list[str] | None = None
    cols_to_retain: list[str] | None = None
    ignore_policy: str = "exclude"
    features: Mapping[str, Any] | None = None
    batch_size: int | None = None
    batch_mode: str = "random"
    heldout_example_sessions_per_subject: int = 1
    example_max_subjects: int = 6

    @classmethod
    def from_data_cfg(
        cls,
        data_cfg: Any,
        *,
        default_example_max_subjects: int = 6,
    ) -> "HeldoutEvalConfig":
        test_subject_ids = _cfg_get(data_cfg, "test_subject_ids", None)
        if test_subject_ids is not None and not isinstance(test_subject_ids, list):
            if isinstance(test_subject_ids, Sequence) and not isinstance(test_subject_ids, str):
                test_subject_ids = list(test_subject_ids)
            else:
                test_subject_ids = [test_subject_ids]

        curricula = _cfg_get(data_cfg, "curricula", None)
        if curricula is not None and not isinstance(curricula, list):
            curricula = list(curricula)

        cols_to_retain = _cfg_get(data_cfg, "cols_to_retain", None)
        if cols_to_retain is not None and not isinstance(cols_to_retain, list):
            cols_to_retain = list(cols_to_retain)

        features = _cfg_get(data_cfg, "features", None)
        if features is not None and not isinstance(features, Mapping):
            features = None

        return cls(
            test_subject_ids=test_subject_ids,
            min_sessions=int(_cfg_get(data_cfg, "min_sessions", 10)),
            heldout_every_n=int(_cfg_get(data_cfg, "heldout_every_n", 5)),
            heldout_eval=(
                None
                if _cfg_get(data_cfg, "heldout_eval", None) is None
                else bool(_cfg_get(data_cfg, "heldout_eval", None))
            ),
            # Read ``type`` from a raw config; fall back to the field name when the
            # config was round-tripped through ``asdict(HeldoutEvalConfig)``.
            data_type=_cfg_get(data_cfg, "type", _cfg_get(data_cfg, "data_type", None)),
            mature_only=bool(_cfg_get(data_cfg, "mature_only", True)),
            curricula=curricula,
            cols_to_retain=cols_to_retain,
            ignore_policy=str(_cfg_get(data_cfg, "ignore_policy", "exclude")),
            features=features,
            batch_size=_cfg_get(data_cfg, "batch_size", None),
            batch_mode=str(_cfg_get(data_cfg, "batch_mode", "random")),
            heldout_example_sessions_per_subject=int(
                _cfg_get(data_cfg, "heldout_example_sessions_per_subject", 1)
            ),
            example_max_subjects=int(
                _cfg_get(data_cfg, "example_max_subjects", default_example_max_subjects)
            ),
        )

    @property
    def enabled(self) -> bool:
        # Explicit data.heldout_eval wins; otherwise auto-enable for database runs.
        if self.heldout_eval is not None:
            return bool(self.heldout_eval)
        return self.data_type in MICE_DATABASE_DATA_TYPES

    def validate(self) -> None:
        if self.heldout_example_sessions_per_subject < 0:
            raise ValueError(
                "data.heldout_example_sessions_per_subject must be >= 0 for held-out plotting."
            )
        if self.example_max_subjects < 0:
            raise ValueError("example_max_subjects must be >= 0.")


def _resolve_heldout_eval_config(config_source: Any) -> HeldoutEvalConfig:
    if isinstance(config_source, HeldoutEvalConfig):
        return config_source

    if hasattr(config_source, "data"):
        data_cfg = config_source.data
        default_max_subjects = int(getattr(config_source, "example_max_subjects", 6))
    else:
        data_cfg = config_source
        default_max_subjects = 6

    return HeldoutEvalConfig.from_data_cfg(
        data_cfg,
        default_example_max_subjects=default_max_subjects,
    )


def should_run_heldout_eval(data_cfg: Any) -> bool:
    """Return True when held-out test subject selectors are configured."""
    return _resolve_heldout_eval_config(data_cfg).enabled
