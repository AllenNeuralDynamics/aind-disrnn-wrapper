"""Shared evaluation primitives used by both training-time held-out evaluation and
standalone post-training analysis.

Modules here construct/score trained models from saved artifacts; they do not import
``model_trainers`` and contain no training loops. The legacy ``utils.*_evaluation`` and
``utils.disrnn_plotting`` paths remain as thin re-export shims for backward compatibility.
"""
