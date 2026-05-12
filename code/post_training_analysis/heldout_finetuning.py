"""Standalone held-out subject embedding fine-tuning for multisubject runs."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import aind_disrnn_utils.data_loader as dl
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from base.types import DatasetBundle
from data_loaders import mice as mice_loader
from disentangled_rnns.library import rnn_utils
from model_trainers.disrnn_trainer import DisrnnTrainer
from model_trainers.gru_trainer import GruTrainer
from models.gru_network import make_gru_network
from models.session_conditioning import resolve_session_conditioning_from_architecture
from post_training_analysis.generative_analysis import resolve_model_run
from utils.gru_evaluation import add_gru_model_results
from utils.multisubject import (
    append_subjects_to_index_maps,
    build_subject_embedding_update_mask,
    expand_local_multisubject_params,
    extend_session_context,
    extract_subject_embeddings_from_params,
    load_session_context_map,
    load_subject_index_map,
)
from utils.session_regularized_training import train_network_with_session_regularization

logger = logging.getLogger(__name__)

_DEFAULT_OUTPUT_ROOT = "/results/heldout_subject_finetuning"
_GRU_CONFIG_NAME = "gru_config.json"
_DISRNN_CONFIG_NAME = "disrnn_config.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _to_plain_dict(config_source: Any) -> dict[str, Any]:
    if isinstance(config_source, DictConfig):
        resolved = OmegaConf.to_container(config_source, resolve=True)
        if not isinstance(resolved, dict):
            raise ValueError("Expected mapping-style OmegaConf config.")
        return resolved
    if isinstance(config_source, Mapping):
        return dict(config_source)
    raise TypeError(
        "Held-out fine-tuning config must be a mapping, OmegaConf DictConfig, or path."
    )


def _load_config(config_source: str | Path | Mapping[str, Any] | DictConfig) -> dict[str, Any]:
    if isinstance(config_source, (str, Path)):
        cfg = OmegaConf.load(Path(config_source).expanduser())
        return _to_plain_dict(cfg)
    return _to_plain_dict(config_source)


def _normalize_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if value is None:
        return {}
    raise ValueError(f"Expected a mapping, got {type(value).__name__}.")


def _safe_slug(value: Any) -> str:
    text = str(value)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text)
    safe = safe.strip("_")
    return safe or "value"


def _next_available_dir(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = 1
    while True:
        candidate = Path(f"{path}__rerun_{suffix}")
        if not candidate.exists():
            return candidate
        suffix += 1


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _normalize_optional_list(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return list(value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return [value]


def _selector_fields_present(selector_cfg: Mapping[str, Any]) -> bool:
    return any(
        selector_cfg.get(key) is not None
        for key in ("test_subject_ids", "test_subject_start", "test_subject_end")
    )


def _heldout_selector_from_data_config(data_cfg: Mapping[str, Any]) -> dict[str, Any]:
    selector = {
        "test_subject_ids": _normalize_optional_list(data_cfg.get("test_subject_ids")),
        "test_subject_start": _coerce_optional_int(data_cfg.get("test_subject_start")),
        "test_subject_end": _coerce_optional_int(data_cfg.get("test_subject_end")),
        "mature_only": bool(data_cfg.get("mature_only", True)),
        "curricula": list(data_cfg.get("curricula") or []),
        "cols_to_retain": data_cfg.get("cols_to_retain"),
    }
    if (
        selector["test_subject_ids"] is None
        and selector["test_subject_start"] is None
        and selector["test_subject_end"] is None
    ):
        raise ValueError(
            "Source run does not define held-out subject selectors. Expected one of "
            "data.test_subject_ids or data.test_subject_start/end in the source run config."
        )
    return selector


def _resolve_heldout_selector(
    *,
    config: Mapping[str, Any],
    source_data_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    override_cfg = _normalize_mapping(config.get("heldout_subjects"))
    if not override_cfg or not _selector_fields_present(override_cfg):
        selector = _heldout_selector_from_data_config(source_data_cfg)
        logger.info(
            "Using held-out subject selectors from the source training run: ids=%s start=%s end=%s",
            selector["test_subject_ids"],
            selector["test_subject_start"],
            selector["test_subject_end"],
        )
        return selector

    test_subject_ids = _normalize_optional_list(override_cfg.get("test_subject_ids"))
    test_subject_start = _coerce_optional_int(override_cfg.get("test_subject_start"))
    test_subject_end = _coerce_optional_int(override_cfg.get("test_subject_end"))
    if test_subject_ids is not None and (
        test_subject_start is not None or test_subject_end is not None
    ):
        raise ValueError(
            "Specify either heldout_subjects.test_subject_ids or "
            "heldout_subjects.test_subject_start/end, not both."
        )

    selector = {
        "test_subject_ids": test_subject_ids,
        "test_subject_start": test_subject_start,
        "test_subject_end": test_subject_end,
        "mature_only": bool(
            override_cfg["mature_only"]
            if "mature_only" in override_cfg and override_cfg["mature_only"] is not None
            else source_data_cfg.get("mature_only", True)
        ),
        "curricula": list(
            override_cfg["curricula"]
            if "curricula" in override_cfg and override_cfg["curricula"] is not None
            else source_data_cfg.get("curricula") or []
        ),
        "cols_to_retain": (
            override_cfg["cols_to_retain"]
            if "cols_to_retain" in override_cfg
            and override_cfg["cols_to_retain"] is not None
            else source_data_cfg.get("cols_to_retain")
        ),
    }
    logger.info(
        "Using held-out subject selectors from the fine-tuning config: ids=%s start=%s end=%s",
        selector["test_subject_ids"],
        selector["test_subject_start"],
        selector["test_subject_end"],
    )
    return selector


def _heldout_selector_slug(selector: Mapping[str, Any]) -> str:
    if selector.get("test_subject_ids") is not None:
        subject_ids = selector["test_subject_ids"]
        if not isinstance(subject_ids, list):
            subject_ids = list(subject_ids)
        joined = "_".join(_safe_slug(subject_id) for subject_id in subject_ids[:8])
        if len(subject_ids) > 8:
            joined = f"{joined}_plus_{len(subject_ids) - 8}"
        return f"ids_{joined}"
    start = selector.get("test_subject_start")
    end = selector.get("test_subject_end")
    return f"rank_{start}_{end}"


def _resolved_output_root(
    config: Mapping[str, Any],
    *,
    output_root_override: str | Path | None,
) -> Path:
    output_cfg = _normalize_mapping(config.get("output"))
    output_root = output_root_override or output_cfg.get("output_root") or _DEFAULT_OUTPUT_ROOT
    return Path(str(output_root)).expanduser().resolve()


def _maybe_start_wandb_run(
    *,
    resolved_config: Mapping[str, Any],
    run_dir: Path,
    source_run: Any,
) -> Any | None:
    wandb_cfg = _normalize_mapping(resolved_config.get("wandb"))
    if not wandb_cfg:
        return None
    try:
        import wandb
    except ModuleNotFoundError:
        logger.warning(
            "Skipping W&B logging because wandb is not installed in the active runtime."
        )
        return None

    init_kwargs = dict(wandb_cfg)
    tags = list(init_kwargs.pop("tags", []) or [])
    tags.extend(["heldout_subject_finetuning", str(source_run.model_type)])
    init_kwargs.setdefault("dir", str(run_dir))
    init_kwargs.setdefault("name", run_dir.name)
    return wandb.init(
        **init_kwargs,
        config=resolved_config,
        tags=tags,
    )


def _resolve_runtime_config(
    config: Mapping[str, Any],
    *,
    source_run: Any,
    output_root_override: str | Path | None,
) -> dict[str, Any]:
    source_cfg = _normalize_mapping(config.get("source_run"))
    if "model_dir" not in source_cfg:
        raise ValueError("Config must set source_run.model_dir.")

    fine_tune_cfg = _normalize_mapping(config.get("heldout_finetuning"))
    source_data_cfg = _normalize_mapping(_normalize_mapping(source_run.run_config).get("data"))
    source_model_cfg = _normalize_mapping(_normalize_mapping(source_run.run_config).get("model"))
    heldout_selector = _resolve_heldout_selector(
        config=config,
        source_data_cfg=source_data_cfg,
    )

    if "n_steps" not in fine_tune_cfg:
        raise ValueError("Config must set heldout_finetuning.n_steps.")
    if "lr" not in fine_tune_cfg:
        raise ValueError("Config must set heldout_finetuning.lr.")

    checkpoint_every_n_steps = int(
        fine_tune_cfg.get("checkpoint_every_n_steps", fine_tune_cfg["n_steps"])
    )
    if checkpoint_every_n_steps < 0:
        raise ValueError("heldout_finetuning.checkpoint_every_n_steps must be >= 0.")

    seed = config.get("seed", source_run.seed)
    resolved = copy.deepcopy(config)
    resolved["seed"] = int(seed if seed is not None else 0)
    resolved.setdefault("source_run", {})
    resolved["source_run"]["checkpoint_policy"] = source_cfg.get(
        "checkpoint_policy",
        "best_eval",
    )
    resolved.setdefault("output", {})
    resolved["output"]["output_root"] = str(
        _resolved_output_root(config, output_root_override=output_root_override)
    )
    resolved["heldout_subjects"] = heldout_selector
    resolved["heldout_finetuning"] = {
        "n_steps": int(fine_tune_cfg["n_steps"]),
        "lr": float(fine_tune_cfg["lr"]),
        "checkpoint_every_n_steps": checkpoint_every_n_steps,
        "batch_size": fine_tune_cfg.get("batch_size", source_data_cfg.get("batch_size")),
        "batch_mode": str(fine_tune_cfg.get("batch_mode", "single")),
        "checkpoint_plot_split_examples_every_n": int(
            fine_tune_cfg.get("checkpoint_plot_split_examples_every_n", 0)
        ),
        "checkpoint_save_output_df_every_n": int(
            fine_tune_cfg.get("checkpoint_save_output_df_every_n", 0)
        ),
        "train_example_sessions_per_subject": int(
            fine_tune_cfg.get(
                "train_example_sessions_per_subject",
                source_data_cfg.get("train_example_sessions_per_subject", 1),
            )
        ),
        "eval_example_sessions_per_subject": int(
            fine_tune_cfg.get(
                "eval_example_sessions_per_subject",
                source_data_cfg.get("eval_example_sessions_per_subject", 1),
            )
        ),
        "example_max_subjects": int(
            fine_tune_cfg.get(
                "example_max_subjects",
                source_data_cfg.get("example_max_subjects", 6),
            )
        ),
        "keep_media_files": bool(fine_tune_cfg.get("keep_media_files", True)),
        "max_grad_norm": float(
            fine_tune_cfg.get(
                "max_grad_norm",
                _normalize_mapping(source_model_cfg.get("training")).get("max_grad_norm", 1.0),
            )
        ),
    }
    return resolved


def _resolve_output_run_dir(
    *,
    resolved_config: Mapping[str, Any],
    source_run: Any,
    heldout_selector: Mapping[str, Any],
) -> Path:
    output_root = Path(resolved_config["output"]["output_root"]).expanduser().resolve()
    source_run_slug = _safe_slug(Path(source_run.model_dir).name)
    checkpoint_component = (
        f"{_safe_slug(source_run.checkpoint_policy)}_{_safe_slug(source_run.checkpoint_label)}"
    )
    selector_component = _heldout_selector_slug(heldout_selector)
    run_name_suffix = resolved_config.get("output", {}).get("run_name_suffix")
    if run_name_suffix:
        selector_component = f"{selector_component}_{_safe_slug(run_name_suffix)}"
    return _next_available_dir(
        output_root / source_run_slug / checkpoint_component / selector_component
    )


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _save_params(path: Path, params: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))


def _load_params(params_path: str | Path) -> Any:
    with Path(params_path).open("r") as f:
        return rnn_utils.to_np(json.load(f))


def _load_heldout_snapshot_selection(
    *,
    heldout_selector: Mapping[str, Any],
) -> tuple[pd.DataFrame, list[Any]]:
    from utils.load_mice_snapshot import load_mice_snapshot

    df, subject_ids = load_mice_snapshot(
        subject_ids=heldout_selector.get("test_subject_ids"),
        subject_start=heldout_selector.get("test_subject_start"),
        subject_end=heldout_selector.get("test_subject_end"),
        mature_only=bool(heldout_selector.get("mature_only", True)),
        curricula=heldout_selector.get("curricula"),
        cols_to_retain=heldout_selector.get("cols_to_retain"),
    )
    if len(df) == 0:
        raise ValueError("Held-out subject selection resolved to an empty snapshot.")
    return df, list(subject_ids)


def _build_local_heldout_bundle(
    *,
    heldout_df: pd.DataFrame,
    heldout_subject_ids: Sequence[Any],
    source_data_cfg: Mapping[str, Any],
    fine_tune_cfg: Mapping[str, Any],
) -> DatasetBundle:
    skip_subjects_with_insufficient_sessions = bool(
        fine_tune_cfg.get("skip_subjects_with_insufficient_sessions", True)
    )
    metadata = {
        "subject_ids": list(heldout_subject_ids),
        "mature_only": bool(source_data_cfg.get("mature_only", True)),
        "curricula": list(source_data_cfg.get("curricula") or []),
        "ignore_policy": str(source_data_cfg.get("ignore_policy", "exclude")),
        "features": source_data_cfg.get("features"),
        "eval_every_n": int(source_data_cfg.get("eval_every_n", 2)),
        "heldout_example_sessions_per_subject": int(
            max(
                fine_tune_cfg.get("train_example_sessions_per_subject", 1),
                fine_tune_cfg.get("eval_example_sessions_per_subject", 1),
            )
        ),
        "train_example_sessions_per_subject": int(
            fine_tune_cfg.get("train_example_sessions_per_subject", 1)
        ),
        "eval_example_sessions_per_subject": int(
            fine_tune_cfg.get("eval_example_sessions_per_subject", 1)
        ),
        "example_max_subjects": int(fine_tune_cfg.get("example_max_subjects", 6)),
    }
    return mice_loader._build_multisubject_bundle(
        df=heldout_df,
        resolved_subject_ids=heldout_subject_ids,
        ignore_policy=str(source_data_cfg.get("ignore_policy", "exclude")),
        features=source_data_cfg.get("features"),
        eval_every_n=int(source_data_cfg.get("eval_every_n", 2)),
        batch_size=fine_tune_cfg.get("batch_size"),
        batch_mode=str(fine_tune_cfg.get("batch_mode", "single")),
        metadata=metadata,
        skip_subjects_with_insufficient_sessions=skip_subjects_with_insufficient_sessions,
    )


def _remap_dataset_subject_indices(
    dataset: Any,
    *,
    local_to_global_subject_index: Mapping[int, int],
) -> Any:
    xs, ys = dataset.get_all()
    remapped_xs = np.asarray(xs).copy()
    subject_values = np.asarray(np.rint(remapped_xs[..., 0]), dtype=int)
    valid_mask = subject_values >= 0
    remapped_subject_values = subject_values.copy()
    for local_index, global_index in local_to_global_subject_index.items():
        remapped_subject_values[subject_values == int(local_index)] = int(global_index)
    remapped_xs[..., 0] = np.where(valid_mask, remapped_subject_values, subject_values).astype(
        remapped_xs.dtype,
        copy=False,
    )
    dataset_class = dataset.__class__
    return dataset_class(
        remapped_xs,
        ys,
        y_type=getattr(dataset, "y_type", "categorical"),
        n_classes=getattr(dataset, "n_classes", None),
        x_names=list(getattr(dataset, "x_names", [])),
        y_names=list(getattr(dataset, "y_names", [])),
        batch_size=getattr(dataset, "batch_size", None),
        batch_mode=getattr(dataset, "batch_mode", "random"),
    )


def _build_extended_session_metadata(
    *,
    source_run: Any,
    total_subject_count: int,
    local_bundle: DatasetBundle,
    local_to_global_subject_index: Mapping[int, int],
) -> tuple[dict[str, Any] | None, list[int]]:
    source_session_context = None
    session_max_index_by_subject_index = [0] * int(total_subject_count)
    if source_run.session_context_map_path:
        source_session_context = load_session_context_map(source_run.session_context_map_path)
        for row in source_session_context.get("per_subject", []):
            subject_index = int(row["subject_index"])
            session_max_index_by_subject_index[subject_index] = int(
                len(row.get("ordered_session_ids") or [])
            )

    local_session_context = _normalize_mapping(local_bundle.metadata.get("session_context"))
    appended_rows: list[dict[str, Any]] = []
    for row in local_session_context.get("per_subject", []):
        local_subject_index = int(row["subject_index"])
        global_subject_index = int(local_to_global_subject_index[local_subject_index])
        ordered_session_ids = [str(value) for value in row.get("ordered_session_ids") or []]
        appended_rows.append(
            {
                "subject_id": row.get("subject_id"),
                "subject_index": global_subject_index,
                "ordered_session_ids": ordered_session_ids,
                "ordered_source_session_ids": [
                    str(value)
                    for value in (
                        row.get("ordered_source_session_ids")
                        or row.get("ordered_session_ids")
                        or []
                    )
                ],
            }
        )
        session_max_index_by_subject_index[global_subject_index] = int(len(ordered_session_ids))

    if source_session_context is None:
        if appended_rows:
            source_session_context = {
                "indexing": "1_based",
                "per_subject": [],
            }
        else:
            return None, session_max_index_by_subject_index
    return extend_session_context(
        source_session_context,
        appended_rows=appended_rows,
    ), session_max_index_by_subject_index


def _maybe_prepend_session_indices(
    *,
    dataset_bundle: DatasetBundle,
    architecture: Mapping[str, Any],
) -> DatasetBundle:
    from utils.multisubject import prepend_session_index_to_multisubject_split_datasets

    metadata = dict(dataset_bundle.metadata)
    max_n_subjects = int(metadata.get("num_subjects", 0))
    subject_embedding_size = architecture.get("subject_embedding_size")
    session_cfg = resolve_session_conditioning_from_architecture(
        architecture=architecture,
        metadata=metadata,
        multisubject=True,
        max_n_subjects=max_n_subjects,
        subject_embedding_size=(
            None if subject_embedding_size is None else int(subject_embedding_size)
        ),
        context="Held-out fine-tuning",
    )
    if not bool(session_cfg["enabled"]):
        return dataset_bundle

    dataset_full, dataset_train, dataset_eval = prepend_session_index_to_multisubject_split_datasets(
        dataset=dataset_bundle.extras["dataset"],
        dataset_train=dataset_bundle.train_set,
        dataset_eval=dataset_bundle.eval_set,
        metadata=metadata,
    )
    return DatasetBundle(
        raw=dataset_bundle.raw,
        train_set=dataset_train,
        eval_set=dataset_eval,
        metadata=metadata,
        extras={"dataset": dataset_full},
    )


def _build_global_heldout_bundle(
    *,
    source_run: Any,
    heldout_selector: Mapping[str, Any],
    fine_tune_cfg: Mapping[str, Any],
    architecture: Mapping[str, Any],
) -> tuple[DatasetBundle, list[Any], list[int], dict[Any, int], dict[int, Any]]:
    if source_run.subject_index_map_path is None:
        raise FileNotFoundError(
            "Held-out fine-tuning requires outputs/subject_index_map.json in the source run."
        )
    source_data_cfg = _normalize_mapping(_normalize_mapping(source_run.run_config).get("data"))
    heldout_df, heldout_subject_ids = _load_heldout_snapshot_selection(
        heldout_selector=heldout_selector
    )
    local_bundle = _build_local_heldout_bundle(
        heldout_df=heldout_df,
        heldout_subject_ids=heldout_subject_ids,
        source_data_cfg=source_data_cfg,
        fine_tune_cfg=fine_tune_cfg,
    )
    retained_heldout_subject_ids = list(local_bundle.metadata.get("subject_ids") or [])
    skipped_subject_rows = list(
        local_bundle.metadata.get("skipped_subjects_with_insufficient_sessions") or []
    )
    if skipped_subject_rows:
        logger.warning(
            "Skipped %d held-out subject(s) during fine-tuning bundle construction due to "
            "insufficient sessions for the inherited eval_every_n: %s",
            len(skipped_subject_rows),
            [row.get("subject_id") for row in skipped_subject_rows],
        )
    if not retained_heldout_subject_ids:
        raise ValueError(
            "No held-out subjects remain after filtering and per-subject split validation."
        )
    source_subject_id_to_index, source_index_to_subject_id = load_subject_index_map(
        source_run.subject_index_map_path
    )
    updated_subject_id_to_index, updated_index_to_subject_id = append_subjects_to_index_maps(
        source_subject_id_to_index,
        source_index_to_subject_id,
        retained_heldout_subject_ids,
    )
    added_subject_ids = [
        subject_id
        for subject_id in retained_heldout_subject_ids
        if subject_id not in source_subject_id_to_index
    ]
    if not added_subject_ids:
        raise ValueError(
            "Held-out subject selection did not add any new subjects beyond the source run."
        )

    local_subject_id_to_index = dict(local_bundle.metadata["subject_id_to_index"])
    local_to_global_subject_index = {
        int(local_index): int(updated_subject_id_to_index[subject_id])
        for subject_id, local_index in local_subject_id_to_index.items()
    }
    appended_subject_indices = [
        int(updated_subject_id_to_index[subject_id]) for subject_id in added_subject_ids
    ]
    remapped_full = _remap_dataset_subject_indices(
        local_bundle.extras["dataset"],
        local_to_global_subject_index=local_to_global_subject_index,
    )
    remapped_train = _remap_dataset_subject_indices(
        local_bundle.train_set,
        local_to_global_subject_index=local_to_global_subject_index,
    )
    remapped_eval = _remap_dataset_subject_indices(
        local_bundle.eval_set,
        local_to_global_subject_index=local_to_global_subject_index,
    )
    raw_df = local_bundle.raw.copy()
    raw_df["subject_index"] = raw_df["subject_id"].map(updated_subject_id_to_index)

    session_context, session_max_index_by_subject_index = _build_extended_session_metadata(
        source_run=source_run,
        total_subject_count=len(updated_index_to_subject_id),
        local_bundle=local_bundle,
        local_to_global_subject_index=local_to_global_subject_index,
    )
    metadata = dict(local_bundle.metadata)
    ordered_subject_ids = [
        updated_index_to_subject_id[index] for index in sorted(updated_index_to_subject_id)
    ]
    metadata.update(
        {
            "subject_ids": ordered_subject_ids,
            "subject_id_to_index": updated_subject_id_to_index,
            "index_to_subject_id": updated_index_to_subject_id,
            "num_subjects": int(len(updated_index_to_subject_id)),
            "session_max_index_by_subject_index": session_max_index_by_subject_index,
            "train_session_ids": list(local_bundle.metadata.get("train_session_ids") or []),
            "eval_session_ids": list(local_bundle.metadata.get("eval_session_ids") or []),
            "heldout_subject_ids": list(added_subject_ids),
            "multisubject": True,
        }
    )
    if session_context is not None:
        metadata["session_context"] = session_context

    global_bundle = DatasetBundle(
        raw=raw_df,
        train_set=remapped_train,
        eval_set=remapped_eval,
        metadata=metadata,
        extras={"dataset": remapped_full},
    )
    global_bundle = _maybe_prepend_session_indices(
        dataset_bundle=global_bundle,
        architecture=architecture,
    )
    return (
        global_bundle,
        added_subject_ids,
        appended_subject_indices,
        updated_subject_id_to_index,
        updated_index_to_subject_id,
    )


def _infer_n_action_logits(
    dataset: Any,
    yhat: np.ndarray | None = None,
    *,
    ignore_policy: str,
) -> int:
    n_action_logits = int(getattr(dataset, "n_classes", 0))
    if n_action_logits > 0:
        return n_action_logits
    if yhat is not None:
        inferred = int(np.asarray(yhat).shape[2])
        if inferred > 2 and str(ignore_policy) == "exclude":
            return inferred - 1
        return inferred
    return 2 if str(ignore_policy) == "exclude" else 3


def _compute_supervised_loss_from_outputs(
    yhat: np.ndarray,
    ys: np.ndarray,
    *,
    loss_name: str,
    loss_param: Any,
    n_action_logits: int,
) -> float:
    logits = np.asarray(yhat)[:, :, : int(n_action_logits)]
    targets = np.asarray(ys)
    if targets.ndim != 3 or targets.shape[-1] != 1:
        raise ValueError(
            "Held-out fine-tuning expects targets shaped [timesteps, sessions, 1]."
        )
    targets = np.squeeze(targets, axis=-1)
    valid_mask = np.isfinite(targets) & (targets != -1)
    if not np.any(valid_mask):
        return float("nan")

    safe_targets = np.where(valid_mask, targets, 0).astype(int)
    logits_stable = logits - np.max(logits, axis=-1, keepdims=True)
    log_probs = logits_stable - np.log(np.sum(np.exp(logits_stable), axis=-1, keepdims=True))
    selected_log_probs = log_probs[valid_mask]
    selected_targets = safe_targets[valid_mask]
    supervised_loss = -np.mean(
        selected_log_probs[np.arange(selected_log_probs.shape[0]), selected_targets]
    )

    resolved_loss_name = str(loss_name).strip().lower()
    if resolved_loss_name != "penalized_categorical":
        return float(supervised_loss)

    if isinstance(loss_param, Mapping):
        if "penalty_scale" in loss_param:
            penalty_scale = float(loss_param["penalty_scale"])
        elif "value" in loss_param:
            penalty_scale = float(loss_param["value"])
        else:
            penalty_scale = 1.0
    else:
        penalty_scale = float(loss_param)
    penalty_values = np.asarray(yhat)[..., -1]
    penalty = float(np.mean(penalty_values[valid_mask]))
    return float(supervised_loss + penalty_scale * penalty)


def _make_gru_runtime(
    *,
    source_run: Any,
    output_dir: Path,
) -> tuple[GruTrainer, Any, Any, dict[str, Any], Mapping[str, Any]]:
    source_model_cfg = _normalize_mapping(_normalize_mapping(source_run.run_config).get("model"))
    architecture = _normalize_mapping(source_model_cfg.get("architecture"))
    training_cfg = _normalize_mapping(source_model_cfg.get("training"))
    trainer = GruTrainer(
        architecture=architecture,
        training=training_cfg,
        output_dir=str(output_dir),
        seed=source_run.seed,
    )
    return trainer, architecture, training_cfg, copy.deepcopy(source_run.model_config), source_model_cfg


def _make_disrnn_runtime(
    *,
    source_run: Any,
    output_dir: Path,
) -> tuple[DisrnnTrainer, Mapping[str, Any], Mapping[str, Any], dict[str, Any], Mapping[str, Any]]:
    source_model_cfg = _normalize_mapping(_normalize_mapping(source_run.run_config).get("model"))
    architecture = _normalize_mapping(source_model_cfg.get("architecture"))
    penalties = _normalize_mapping(source_model_cfg.get("penalties"))
    training_cfg = _normalize_mapping(source_model_cfg.get("training"))
    distillation_cfg = source_model_cfg.get("distillation")
    trainer = DisrnnTrainer(
        architecture=architecture,
        penalties=penalties,
        training=training_cfg,
        distillation=distillation_cfg,
        output_dir=str(output_dir),
        seed=source_run.seed,
    )
    return trainer, architecture, training_cfg, copy.deepcopy(source_run.model_config), source_model_cfg


def _save_state_space_figures(
    *,
    trainer: Any,
    params: Any,
    bundle: DatasetBundle,
    checkpoint_dir: Path,
) -> dict[str, str | None]:
    plot_paths = {
        "subject_embedding_state_space": None,
        "subject_session_context_state_space": None,
    }
    subject_embedding_fig = trainer._plot_subject_embedding_state_space(
        params=params,
        raw_df=bundle.raw,
        metadata=bundle.metadata,
    )
    if subject_embedding_fig is not None:
        path = checkpoint_dir / "subject_embedding_state_space.png"
        subject_embedding_fig.savefig(path)
        plt.close(subject_embedding_fig)
        plot_paths["subject_embedding_state_space"] = str(path)

    subject_session_context_fig = trainer._plot_subject_session_context_state_space(
        params=params,
        raw_df=bundle.raw,
        metadata=bundle.metadata,
        session_curriculum_lambda=1.0,
    )
    if subject_session_context_fig is not None:
        path = checkpoint_dir / "subject_session_context_state_space.png"
        subject_session_context_fig.savefig(path)
        plt.close(subject_session_context_fig)
        plot_paths["subject_session_context_state_space"] = str(path)
    return plot_paths


def _evaluate_checkpoint(
    *,
    model_type: str,
    trainer: Any,
    params: Any,
    make_train_network: Any,
    make_eval_network: Any,
    bundle: DatasetBundle,
    source_run: Any,
    fine_tune_cfg: Mapping[str, Any],
    checkpoint_dir: Path,
    step: int,
    loss_name: str,
    loss_param: Any,
) -> dict[str, Any]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _save_params(checkpoint_dir / "params.json", params)

    dataset_full = bundle.extras["dataset"]
    dataset_train = bundle.train_set
    dataset_eval = bundle.eval_set
    ignore_policy = str(bundle.metadata.get("ignore_policy", "exclude"))

    xs_train, ys_train = dataset_train.get_all()
    xs_eval, ys_eval = dataset_eval.get_all()
    xs_full, _ = dataset_full.get_all()

    yhat_train_for_loss, _ = rnn_utils.eval_network(make_train_network, params, xs_train)
    yhat_eval_for_loss, _ = rnn_utils.eval_network(make_train_network, params, xs_eval)
    train_loss = _compute_supervised_loss_from_outputs(
        np.asarray(yhat_train_for_loss),
        ys_train,
        loss_name=loss_name,
        loss_param=loss_param,
        n_action_logits=_infer_n_action_logits(
            dataset_train,
            np.asarray(yhat_train_for_loss),
            ignore_policy=ignore_policy,
        ),
    )
    eval_loss = _compute_supervised_loss_from_outputs(
        np.asarray(yhat_eval_for_loss),
        ys_eval,
        loss_name=loss_name,
        loss_param=loss_param,
        n_action_logits=_infer_n_action_logits(
            dataset_eval,
            np.asarray(yhat_eval_for_loss),
            ignore_policy=ignore_policy,
        ),
    )

    yhat_train_eval, _ = rnn_utils.eval_network(make_eval_network, params, xs_train)
    yhat_eval_eval, _ = rnn_utils.eval_network(make_eval_network, params, xs_eval)
    train_n_action_logits = _infer_n_action_logits(
        dataset_train,
        np.asarray(yhat_train_eval),
        ignore_policy=ignore_policy,
    )
    eval_n_action_logits = _infer_n_action_logits(
        dataset_eval,
        np.asarray(yhat_eval_eval),
        ignore_policy=ignore_policy,
    )
    train_likelihood = float(
        rnn_utils.normalized_likelihood(
            ys_train,
            np.asarray(yhat_train_eval)[:, :, :train_n_action_logits],
        )
    )
    eval_likelihood = float(
        rnn_utils.normalized_likelihood(
            ys_eval,
            np.asarray(yhat_eval_eval)[:, :, :eval_n_action_logits],
        )
    )

    yhat_full_eval, network_states_full = rnn_utils.eval_network(
        make_eval_network,
        params,
        xs_full,
    )
    if model_type == "gru":
        output_df = add_gru_model_results(
            bundle.raw.copy(),
            np.asarray(network_states_full),
            np.asarray(yhat_full_eval),
            ignore_policy=ignore_policy,
        )
    else:
        output_df = dl.add_model_results(
            bundle.raw.copy(),
            np.asarray(network_states_full),
            yhat_full_eval,
            ignore_policy=ignore_policy,
        )

    record: dict[str, Any] = {
        "step": int(step),
        "params_path": str(checkpoint_dir / "params.json"),
        "train_loss": float(train_loss),
        "eval_loss": float(eval_loss),
        "train_likelihood": float(train_likelihood),
        "eval_likelihood": float(eval_likelihood),
    }
    record["plot_paths"] = _save_state_space_figures(
        trainer=trainer,
        params=params,
        bundle=bundle,
        checkpoint_dir=checkpoint_dir,
    )

    plot_examples_every_n = int(fine_tune_cfg.get("checkpoint_plot_split_examples_every_n", 0))
    save_output_df_every_n = int(fine_tune_cfg.get("checkpoint_save_output_df_every_n", 0))
    should_plot_examples = (
        plot_examples_every_n > 0
        and (step == 0 or step % plot_examples_every_n == 0)
    )
    should_save_output_df = (
        save_output_df_every_n > 0
        and (step == 0 or step % save_output_df_every_n == 0)
    )
    if should_save_output_df:
        output_df_path = checkpoint_dir / "output_df.csv"
        output_df.to_csv(output_df_path, index=False)
        record["output_df_path"] = str(output_df_path)
    if should_plot_examples:
        n_action_logits_full = _infer_n_action_logits(
            dataset_full,
            np.asarray(yhat_full_eval),
            ignore_policy=ignore_policy,
        )
        split_summaries = trainer._generate_split_examples(
            output_dir=checkpoint_dir,
            output_df=output_df,
            network_states_full=np.asarray(network_states_full),
            yhat_full=np.asarray(yhat_full_eval),
            params=params,
            metadata=dict(bundle.metadata),
            n_action_logits=n_action_logits_full,
            log_scope=f"Checkpoint step {step}",
        )
        if not bool(fine_tune_cfg.get("keep_media_files", True)):
            trainer._cleanup_split_example_media(split_summaries)
        record["split_examples"] = split_summaries
    return record


def _plot_checkpoint_metric_curve(
    *,
    checkpoint_records: Sequence[Mapping[str, Any]],
    metric_key_train: str,
    metric_key_eval: str,
    title: str,
    ylabel: str,
    output_path: Path,
    log_scale: bool = False,
) -> Path:
    if not checkpoint_records:
        raise ValueError("checkpoint_records must not be empty.")
    steps = np.asarray([int(record["step"]) for record in checkpoint_records], dtype=int)
    train_values = np.asarray(
        [float(record[metric_key_train]) for record in checkpoint_records],
        dtype=float,
    )
    eval_values = np.asarray(
        [float(record[metric_key_eval]) for record in checkpoint_records],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    if log_scale:
        ax.semilogy(steps, train_values, marker="o", color="black", label="Train")
        ax.semilogy(steps, eval_values, marker="o", color="tab:red", label="Eval")
    else:
        ax.plot(steps, train_values, marker="o", color="black", label="Train")
        ax.plot(steps, eval_values, marker="o", color="tab:red", label="Eval")
    ax.set_xlabel("Fine-tuning Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def _save_model_config(
    *,
    model_type: str,
    model_config: dict[str, Any],
    output_dir: Path,
    metadata: Mapping[str, Any],
) -> Path:
    saved_config = copy.deepcopy(model_config)
    if model_type == "disrnn":
        saved_config["max_n_subjects"] = int(metadata["num_subjects"])
        if metadata.get("session_max_index_by_subject_index") is not None:
            saved_config["session_max_index_by_subject_index"] = list(
                metadata["session_max_index_by_subject_index"]
            )
        path = output_dir / _DISRNN_CONFIG_NAME
    else:
        path = output_dir / _GRU_CONFIG_NAME
    _save_json(path, saved_config)
    return path


def _final_metrics_summary(checkpoint_records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not checkpoint_records:
        return {}
    final_record = checkpoint_records[-1]
    return {
        "train_loss": float(final_record["train_loss"]),
        "eval_loss": float(final_record["eval_loss"]),
        "train_likelihood": float(final_record["train_likelihood"]),
        "eval_likelihood": float(final_record["eval_likelihood"]),
    }


def run_heldout_subject_finetuning_from_config(
    config_source: str | Path | Mapping[str, Any] | DictConfig,
    *,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Fine-tune held-out subject embeddings for a trained multisubject GRU/disRNN."""
    config = _load_config(config_source)
    source_cfg = _normalize_mapping(config.get("source_run"))
    source_model_dir = source_cfg.get("model_dir")
    if source_model_dir is None:
        raise ValueError("Config must define source_run.model_dir.")

    resolved_source_run = resolve_model_run(
        source_model_dir,
        split="train",
        checkpoint_policy=str(source_cfg.get("checkpoint_policy", "best_eval")),
    )
    if not bool(resolved_source_run.multisubject):
        raise ValueError("Held-out fine-tuning currently supports multisubject runs only.")
    if str(resolved_source_run.model_type).lower() not in {"gru", "disrnn"}:
        raise ValueError(
            "Held-out fine-tuning currently supports only GRU and disRNN source runs."
        )

    resolved_config = _resolve_runtime_config(
        config,
        source_run=resolved_source_run,
        output_root_override=output_root,
    )
    heldout_selector = _normalize_mapping(resolved_config.get("heldout_subjects"))
    run_dir = _resolve_output_run_dir(
        resolved_config=resolved_config,
        source_run=resolved_source_run,
        heldout_selector=heldout_selector,
    )
    outputs_dir = run_dir / "outputs"
    checkpoints_root = outputs_dir / "checkpoints"
    figures_dir = outputs_dir / "figures"
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(
        config=OmegaConf.create(resolved_config),
        f=str(run_dir / "resolved_config.yaml"),
        resolve=True,
    )
    _save_json(run_dir / "resolved_source_run.json", resolved_source_run.to_dict())
    wandb_run = _maybe_start_wandb_run(
        resolved_config=resolved_config,
        run_dir=run_dir,
        source_run=resolved_source_run,
    )
    logger.info(
        "Starting held-out subject fine-tuning: source_model_dir=%s model_type=%s checkpoint=%s "
        "heldout_ids=%s heldout_start=%s heldout_end=%s output_dir=%s",
        resolved_source_run.model_dir,
        resolved_source_run.model_type,
        resolved_source_run.checkpoint_label,
        heldout_selector.get("test_subject_ids"),
        heldout_selector.get("test_subject_start"),
        heldout_selector.get("test_subject_end"),
        run_dir,
    )

    try:
        if resolved_source_run.model_type == "gru":
            trainer, architecture, source_training_cfg, model_config, source_model_cfg = (
                _make_gru_runtime(
                    source_run=resolved_source_run,
                    output_dir=outputs_dir,
                )
            )
        else:
            trainer, architecture, source_training_cfg, model_config, source_model_cfg = (
                _make_disrnn_runtime(
                    source_run=resolved_source_run,
                    output_dir=outputs_dir,
                )
            )

        bundle, heldout_subject_ids, appended_subject_indices, _, _ = _build_global_heldout_bundle(
            source_run=resolved_source_run,
            heldout_selector=heldout_selector,
            fine_tune_cfg=resolved_config["heldout_finetuning"],
            architecture=architecture,
        )
        source_params = _load_params(resolved_source_run.params_path)
        params = expand_local_multisubject_params(
            source_params,
            n_new_subjects=len(appended_subject_indices),
            init="mean",
        )
        parameter_update_mask = build_subject_embedding_update_mask(
            params,
            trainable_subject_indices=appended_subject_indices,
        )

        ignore_policy = str(bundle.metadata.get("ignore_policy", "exclude"))
        if resolved_source_run.model_type == "gru":
            output_size = int(
                _normalize_mapping(model_config).get(
                    "output_size",
                    2 if ignore_policy == "exclude" else 3,
                )
            )
            session_cfg = resolve_session_conditioning_from_architecture(
                architecture=architecture,
                metadata=bundle.metadata,
                multisubject=True,
                max_n_subjects=int(bundle.metadata["num_subjects"]),
                subject_embedding_size=int(architecture["subject_embedding_size"]),
                context="Held-out fine-tuning GRU",
            )
            gru_make_network = make_gru_network(
                hidden_size=int(architecture["hidden_size"]),
                output_size=output_size,
                multisubject=True,
                max_n_subjects=int(bundle.metadata["num_subjects"]),
                subject_embedding_size=int(architecture["subject_embedding_size"]),
                subject_embedding_init=str(architecture.get("subject_embedding_init", "zeros")),
                session_encoding_type=str(session_cfg["session_encoding_type"]),
                session_integration_type=str(session_cfg["session_integration_type"]),
                session_fourier_k=int(session_cfg["session_fourier_k"]),
                session_delta_n_layers=int(session_cfg["session_delta_n_layers"]),
                session_delta_hidden_size=int(session_cfg["session_delta_hidden_size"]),
                session_max_index_by_subject_index=list(
                    bundle.metadata.get("session_max_index_by_subject_index") or []
                ),
            )
            make_train_network = gru_make_network
            make_eval_network = gru_make_network
            runtime_model_config = model_config
        else:
            disrnn_config, noiseless_config = trainer._build_network_configs(
                dataset=bundle.extras["dataset"],
                ignore_policy=ignore_policy,
                metadata=bundle.metadata,
            )
            make_train_network = trainer._make_network_factory(
                disrnn_config, multisubject=True
            )
            make_eval_network = trainer._make_network_factory(
                noiseless_config, multisubject=True
            )
            runtime_model_config = asdict(disrnn_config)

        loss_name = str(source_training_cfg.get("loss", "categorical"))
        loss_param = source_training_cfg.get("loss_param", 1)
        n_action_logits = _infer_n_action_logits(
            bundle.train_set,
            ignore_policy=ignore_policy,
        )
        optimizer = optax.adam(float(resolved_config["heldout_finetuning"]["lr"]))
        opt_state = None
        rng = jax.random.PRNGKey(int(resolved_config["seed"]))
        checkpoint_every_n_steps = int(
            resolved_config["heldout_finetuning"]["checkpoint_every_n_steps"]
        )
        total_steps = int(resolved_config["heldout_finetuning"]["n_steps"])
        evaluation_steps = [0]
        if checkpoint_every_n_steps <= 0:
            if total_steps > 0:
                evaluation_steps.append(total_steps)
        else:
            next_step = checkpoint_every_n_steps
            while next_step < total_steps:
                evaluation_steps.append(next_step)
                next_step += checkpoint_every_n_steps
            if total_steps not in evaluation_steps:
                evaluation_steps.append(total_steps)

        checkpoint_records: list[dict[str, Any]] = []
        optimization_history = {
            "training_loss": [],
            "validation_loss": [],
        }
        previous_step = 0
        for step in evaluation_steps:
            if step > previous_step:
                chunk_key, rng = jax.random.split(rng)
                params, opt_state, chunk_losses = train_network_with_session_regularization(
                    make_train_network,
                    bundle.train_set,
                    bundle.eval_set,
                    loss=loss_name,
                    loss_param=loss_param,
                    n_action_logits=int(n_action_logits),
                    session_regularization_apply=None,
                    session_regularization_scale=0.0,
                    opt=optimizer,
                    parameter_update_mask=parameter_update_mask,
                    params=params,
                    opt_state=opt_state,
                    n_steps=step - previous_step,
                    max_grad_norm=float(resolved_config["heldout_finetuning"]["max_grad_norm"]),
                    random_key=chunk_key,
                    report_progress_by="log",
                    log_losses_every=1,
                )
                optimization_history["training_loss"].extend(
                    np.asarray(chunk_losses["training_loss"], dtype=float).tolist()
                )
                optimization_history["validation_loss"].extend(
                    np.asarray(chunk_losses["validation_loss"], dtype=float).tolist()
                )
            checkpoint_record = _evaluate_checkpoint(
                model_type=resolved_source_run.model_type,
                trainer=trainer,
                params=params,
                make_train_network=make_train_network,
                make_eval_network=make_eval_network,
                bundle=bundle,
                source_run=resolved_source_run,
                fine_tune_cfg=resolved_config["heldout_finetuning"],
                checkpoint_dir=checkpoints_root / f"step_{step}",
                step=step,
                loss_name=loss_name,
                loss_param=loss_param,
            )
            checkpoint_records.append(checkpoint_record)
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "checkpoint/train_loss": float(checkpoint_record["train_loss"]),
                        "checkpoint/eval_loss": float(checkpoint_record["eval_loss"]),
                        "checkpoint/train_likelihood": float(
                            checkpoint_record["train_likelihood"]
                        ),
                        "checkpoint/eval_likelihood": float(
                            checkpoint_record["eval_likelihood"]
                        ),
                    },
                    step=int(step),
                )
            previous_step = step

        _save_params(outputs_dir / "params.json", params)
        _save_json(outputs_dir / "checkpoint_metrics.json", checkpoint_records)
        _save_json(
            checkpoints_root / "index.json",
            {
                "n_steps": int(total_steps),
                "checkpoint_every_n_steps": int(checkpoint_every_n_steps),
                "count": int(len(checkpoint_records)),
                "checkpoints": checkpoint_records,
            },
        )
        _save_json(outputs_dir / "optimization_trace.json", optimization_history)

        model_config_path = _save_model_config(
            model_type=resolved_source_run.model_type,
            model_config=runtime_model_config,
            output_dir=outputs_dir,
            metadata=bundle.metadata,
        )
        subject_artifacts = trainer._save_multisubject_artifacts(
            params=params,
            metadata=bundle.metadata,
            raw_df=bundle.raw,
        )

        loss_curve_path = _plot_checkpoint_metric_curve(
            checkpoint_records=checkpoint_records,
            metric_key_train="train_loss",
            metric_key_eval="eval_loss",
            title="Held-out Subject Fine-tuning Loss",
            ylabel="Loss",
            output_path=figures_dir / "loss_across_checkpoints.png",
            log_scale=True,
        )
        likelihood_curve_path = _plot_checkpoint_metric_curve(
            checkpoint_records=checkpoint_records,
            metric_key_train="train_likelihood",
            metric_key_eval="eval_likelihood",
            title="Held-out Subject Fine-tuning Likelihood",
            ylabel="Normalized Likelihood",
            output_path=figures_dir / "likelihood_across_checkpoints.png",
            log_scale=False,
        )

        summary = {
            "output_dir": str(run_dir),
            "outputs_dir": str(outputs_dir),
            "source_model_dir": str(resolved_source_run.model_dir),
            "source_checkpoint_policy": str(resolved_source_run.checkpoint_policy),
            "source_checkpoint_step": resolved_source_run.checkpoint_step,
            "source_checkpoint_label": str(resolved_source_run.checkpoint_label),
            "source_params_path": str(resolved_source_run.params_path),
            "model_type": str(resolved_source_run.model_type),
            "multisubject": True,
            "seed": int(resolved_config["seed"]),
            "heldout_subject_ids": list(heldout_subject_ids),
            "trainable_subject_indices": [int(index) for index in appended_subject_indices],
            "n_steps": int(total_steps),
            "lr": float(resolved_config["heldout_finetuning"]["lr"]),
            "checkpoint_every_n_steps": int(checkpoint_every_n_steps),
            "metrics": _final_metrics_summary(checkpoint_records),
            "artifacts": {
                "resolved_config": str(run_dir / "resolved_config.yaml"),
                "resolved_source_run": str(run_dir / "resolved_source_run.json"),
                "params": str(outputs_dir / "params.json"),
                "checkpoint_metrics": str(outputs_dir / "checkpoint_metrics.json"),
                "optimization_trace": str(outputs_dir / "optimization_trace.json"),
                "model_config": str(model_config_path),
                "subject_index_map": subject_artifacts["subject_index_map"],
                "subject_embeddings": subject_artifacts["subject_embeddings"],
                "loss_curve": str(loss_curve_path),
                "likelihood_curve": str(likelihood_curve_path),
            },
        }
        if "session_context_map" in subject_artifacts:
            summary["artifacts"]["session_context_map"] = subject_artifacts[
                "session_context_map"
            ]
        _save_json(outputs_dir / "output_summary.json", summary)
        if wandb_run is not None:
            for key, value in summary["metrics"].items():
                wandb_run.summary[f"final/{key}"] = float(value)
            wandb_run.summary["output_dir"] = str(run_dir)
        logger.info(
            "Completed held-out subject fine-tuning: output_dir=%s summary=%s "
            "checkpoint_metrics=%s final_train_likelihood=%s final_eval_likelihood=%s",
            run_dir,
            outputs_dir / "output_summary.json",
            outputs_dir / "checkpoint_metrics.json",
            summary["metrics"].get("train_likelihood"),
            summary["metrics"].get("eval_likelihood"),
        )

        return {
            "output_dir": str(run_dir),
            "outputs_dir": str(outputs_dir),
            "summary_path": str(outputs_dir / "output_summary.json"),
            "checkpoint_metrics_path": str(outputs_dir / "checkpoint_metrics.json"),
            "params_path": str(outputs_dir / "params.json"),
            "subject_index_map_path": subject_artifacts["subject_index_map"],
            "subject_embeddings_path": subject_artifacts["subject_embeddings"],
            "model_config_path": str(model_config_path),
            "loss_curve_path": str(loss_curve_path),
            "likelihood_curve_path": str(likelihood_curve_path),
        }
    finally:
        if wandb_run is not None:
            wandb_run.finish()
