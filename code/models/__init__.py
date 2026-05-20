"""Model definition helpers."""

from __future__ import annotations

__all__ = ["MultisubjectDisRnn", "MultisubjectDisRnnConfig"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from .multisubject_disrnn import MultisubjectDisRnn, MultisubjectDisRnnConfig

    exports = {
        "MultisubjectDisRnn": MultisubjectDisRnn,
        "MultisubjectDisRnnConfig": MultisubjectDisRnnConfig,
    }
    return exports[name]
