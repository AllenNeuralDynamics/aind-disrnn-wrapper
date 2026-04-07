"""Standalone post-training generative analysis helpers.

This module is intentionally light on import-time dependencies so artifact
resolution and switch-statistics helpers remain usable in minimal Python
environments. Runtime-heavy dependencies such as pandas, NumPy, JAX, Haiku,
and the AIND packages are imported lazily inside the functions that need them.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import logging
import math
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

logger = logging.getLogger(__name__)

_TRAIN_SPLIT = "train"
_HELDOUT_SPLIT = "heldout"
_CHECKPOINT_POLICY_BEST_EVAL = "best_eval"
_CHECKPOINT_POLICY_BEST_HELDOUT = "best_heldout"
_CHECKPOINT_POLICY_FINAL = "final"
_ROLLOUT_MODE_CURRICULUM_MATCHED = "curriculum_matched"
_DEFAULT_WINDOW_SIZE = 10
_SUBJECT_LEVEL_MIN_ANIMAL_N = 5
_REWARD_CONDITIONS = ("rewarded", "unrewarded")
_RUN_LENGTH_CONDITIONS = ("run_length_1", "run_length_gt1")


@dataclass
class ResolvedModelRun:
    """Resolved metadata and artifact paths for a trained model run."""

    model_dir: str
    inputs_path: str
    outputs_dir: str
    model_type: str
    split: str
    checkpoint_policy: str
    checkpoint_step: int | None
    checkpoint_label: str
    params_path: str
    config_path: str
    seed: int | None
    multisubject: bool
    mature_only: bool
    ignore_policy: str
    curricula: list[str]
    features: dict[str, Any] | None
    selection: dict[str, Any]
    output_summary_path: str | None = None
    checkpoint_index_path: str | None = None
    checkpoint_selection_reason: str | None = None
    fallback_reason: str | None = None
    model_config: dict[str, Any] = field(default_factory=dict)
    run_config: dict[str, Any] = field(default_factory=dict)
    resolved_subject_ids: list[Any] | None = None
    resolved_session_ids: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return _to_serializable(asdict(self))


def resolve_model_run(
    model_dir: str | Path,
    split: str = _TRAIN_SPLIT,
    checkpoint_policy: str = _CHECKPOINT_POLICY_BEST_EVAL,
) -> ResolvedModelRun:
    """Resolve the trained-model artifacts needed for post-training analysis."""

    started_at = time.perf_counter()
    normalized_split = _normalize_split(split)
    normalized_policy = _normalize_checkpoint_policy(checkpoint_policy)
    model_dir_path = Path(model_dir).expanduser().resolve()
    inputs_path = model_dir_path / "inputs.yaml"
    outputs_dir = model_dir_path / "outputs"

    if not inputs_path.exists():
        raise FileNotFoundError(f"Could not find run inputs at {inputs_path}")
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Could not find run outputs at {outputs_dir}")

    run_config = _load_structured_file(inputs_path)
    if not isinstance(run_config, dict):
        raise ValueError(f"Expected mapping-style config in {inputs_path}")

    data_cfg = _as_dict(run_config.get("data", {}))
    model_cfg = _as_dict(run_config.get("model", {}))
    architecture_cfg = _as_dict(model_cfg.get("architecture", {}))
    model_type = str(model_cfg.get("type", "")).strip().lower()
    if model_type not in {"disrnn", "gru"}:
        raise ValueError(
            f"Unsupported model.type='{model_type}' in {inputs_path}. "
            "Only GRU and disRNN runs are supported."
        )

    selection = _resolve_split_selection(data_cfg, normalized_split)
    config_name = "disrnn_config.json" if model_type == "disrnn" else "gru_config.json"
    config_path = outputs_dir / config_name
    if not config_path.exists():
        raise FileNotFoundError(f"Could not find saved model config at {config_path}")
    model_config = _load_structured_file(config_path)
    if not isinstance(model_config, dict):
        raise ValueError(f"Expected mapping-style model config in {config_path}")

    output_summary_path = outputs_dir / "output_summary.json"
    checkpoint_index_path = outputs_dir / "checkpoints" / "index.json"
    output_summary = (
        _load_structured_file(output_summary_path)
        if output_summary_path.exists()
        else {}
    )
    checkpoint_index = (
        _load_structured_file(checkpoint_index_path)
        if checkpoint_index_path.exists()
        else {}
    )
    if not isinstance(output_summary, dict):
        output_summary = {}
    if not isinstance(checkpoint_index, dict):
        checkpoint_index = {}

    params_path, checkpoint_step, checkpoint_label, reason, fallback_reason = (
        _resolve_checkpoint_artifact(
            model_dir=model_dir_path,
            outputs_dir=outputs_dir,
            checkpoint_policy=normalized_policy,
            checkpoint_index=checkpoint_index,
            output_summary=output_summary,
        )
    )

    resolved = ResolvedModelRun(
        model_dir=str(model_dir_path),
        inputs_path=str(inputs_path),
        outputs_dir=str(outputs_dir),
        model_type=model_type,
        split=normalized_split,
        checkpoint_policy=normalized_policy,
        checkpoint_step=checkpoint_step,
        checkpoint_label=checkpoint_label,
        params_path=str(params_path),
        config_path=str(config_path),
        seed=_coerce_optional_int(run_config.get("seed", model_cfg.get("seed"))),
        multisubject=bool(
            architecture_cfg.get("multisubject", False)
            or data_cfg.get("multisubject", False)
        ),
        mature_only=bool(data_cfg.get("mature_only", True)),
        ignore_policy=str(data_cfg.get("ignore_policy", "exclude")),
        curricula=[str(v) for v in data_cfg.get("curricula") or []],
        features=_coerce_optional_dict(data_cfg.get("features")),
        selection=selection,
        output_summary_path=str(output_summary_path) if output_summary_path.exists() else None,
        checkpoint_index_path=str(checkpoint_index_path)
        if checkpoint_index_path.exists()
        else None,
        checkpoint_selection_reason=reason,
        fallback_reason=fallback_reason,
        model_config=_to_serializable(model_config),
        run_config=_to_serializable(run_config),
    )
    logger.info(
        "Resolved model run in %.2fs: model_type=%s split=%s checkpoint=%s step=%s",
        time.perf_counter() - started_at,
        resolved.model_type,
        resolved.split,
        resolved.checkpoint_label,
        resolved.checkpoint_step,
    )
    return resolved


def load_animal_session_history(
    model_dir: str | Path | ResolvedModelRun,
    split: str = _TRAIN_SPLIT,
):
    """Load the animal sessions that correspond to a model run split.

    V1 now reconstructs animal trial histories directly from the behavioral
    snapshot. That keeps post-training analysis independent from the raw NWB
    loading path and avoids re-querying/rehydrating the original session
    assets when the snapshot already contains the trial-level choice and reward
    histories we need for switch statistics.
    """

    started_at = time.perf_counter()
    resolved_run = _coerce_resolved_run(model_dir, split=split)
    if resolved_run.multisubject:
        raise NotImplementedError(
            "V1 post-training generative analysis currently supports "
            "single-subject GRU/disRNN runs only."
        )

    load_mice_snapshot_mod = importlib.import_module("utils.load_mice_snapshot")

    snapshot_cols = [
        "trial",
        "subject_id",
        "ses_idx",
        "animal_response",
        "earned_reward",
        "curriculum_name",
        "current_stage_actual",
    ]
    logger.info(
        "Loading animal session history from snapshot for model_dir=%s split=%s",
        resolved_run.model_dir,
        resolved_run.split,
    )
    snapshot_started_at = time.perf_counter()
    snapshot_df, selected_subject_ids = load_mice_snapshot_mod.load_mice_snapshot(
        subject_ids=resolved_run.selection.get("subject_ids"),
        subject_start=resolved_run.selection.get("subject_start"),
        subject_end=resolved_run.selection.get("subject_end"),
        mature_only=resolved_run.mature_only,
        curricula=resolved_run.curricula or None,
        cols_to_retain=snapshot_cols,
    )
    logger.info(
        "Loaded snapshot selection in %.2fs: %d trial rows across %d selected subject%s",
        time.perf_counter() - snapshot_started_at,
        len(snapshot_df),
        len(selected_subject_ids),
        "" if len(selected_subject_ids) == 1 else "s",
    )

    if len(snapshot_df) == 0:
        raise ValueError(
            "Snapshot selection resolved to an empty dataset for "
            f"{resolved_run.model_dir} split={resolved_run.split}."
        )

    snapshot_df = snapshot_df.copy()
    snapshot_df["subject_id"] = snapshot_df["subject_id"].map(_normalize_identifier)
    snapshot_df["ses_idx"] = snapshot_df["ses_idx"].map(str)

    # The snapshot already contains trial-level choice/reward history and the
    # session annotations added by load_mice_snapshot(). Building the canonical
    # session dataframe from this result avoids the heavier raw-data/NWB path.
    build_started_at = time.perf_counter()
    session_history = _build_session_history_dataframe(snapshot_df)
    logger.info(
        "Built session history dataframe in %.2fs: %d session%s",
        time.perf_counter() - build_started_at,
        len(session_history),
        "" if len(session_history) == 1 else "s",
    )
    selected_subject_ids = _unique_preserve_order(selected_subject_ids)
    resolved_run.resolved_subject_ids = [
        _normalize_identifier(value) for value in selected_subject_ids
    ]
    resolved_run.resolved_session_ids = sorted(session_history["ses_idx"].tolist())
    logger.info(
        "Finished loading animal session history in %.2fs",
        time.perf_counter() - started_at,
    )
    return session_history


def simulate_model_sessions(
    resolved_run: ResolvedModelRun | Mapping[str, Any],
    animal_sessions,
    rollout_mode: str = _ROLLOUT_MODE_CURRICULUM_MATCHED,
    n_rollouts_per_session: int = 1,
):
    """Simulate one or more model rollouts per source animal session."""

    normalized_rollout_mode = str(rollout_mode).strip().lower()
    if normalized_rollout_mode != _ROLLOUT_MODE_CURRICULUM_MATCHED:
        raise NotImplementedError(
            "V1 only supports rollout_mode='curriculum_matched'. "
            f"Received {rollout_mode!r}."
        )
    if int(n_rollouts_per_session) <= 0:
        raise ValueError("n_rollouts_per_session must be >= 1.")

    run = _coerce_resolved_run(resolved_run)
    if run.multisubject:
        raise NotImplementedError(
            "V1 post-training simulation currently supports only single-subject runs."
        )
    if run.ignore_policy != "exclude":
        raise NotImplementedError(
            "V1 post-training simulation expects ignore_policy='exclude'. "
            f"Received {run.ignore_policy!r}."
        )

    np = _import_dependency("numpy")
    pd = _import_dependency("pandas")
    runner = _restore_model_runner(run)
    animal_rows = list(_iter_session_records(animal_sessions))
    total_source_sessions = len(animal_rows)
    total_requested_rollouts = total_source_sessions * int(n_rollouts_per_session)

    logger.info(
        "Starting model simulation for %d source sessions (%d rollout%s) from %s "
        "[checkpoint=%s, mode=%s]",
        total_source_sessions,
        total_requested_rollouts,
        "" if total_requested_rollouts == 1 else "s",
        run.model_dir,
        run.checkpoint_label,
        normalized_rollout_mode,
    )

    records = []
    completed_rollouts = 0
    for session_index, animal_row in enumerate(animal_rows, start=1):
        source_ses_idx = str(animal_row.get("ses_idx"))
        subject_id = _normalize_identifier(animal_row.get("subject_id"))
        session_date = str(animal_row.get("session_date"))
        curriculum_name = animal_row.get("curriculum_name")
        current_stage_actual = animal_row.get("current_stage_actual")
        nwb_suffix = animal_row.get("nwb_suffix")
        nwb_name = animal_row.get("nwb_name")
        n_trials = int(animal_row.get("n_trials", 0))
        if n_trials <= 0:
            logger.info(
                "Skipping session %d/%d: subject=%s ses_idx=%s has n_trials=%s",
                session_index,
                total_source_sessions,
                subject_id,
                source_ses_idx,
                animal_row.get("n_trials"),
            )
            continue

        logger.info(
            "Simulating session %d/%d: subject=%s ses_idx=%s curriculum=%s n_trials=%d "
            "(%d rollout%s)",
            session_index,
            total_source_sessions,
            subject_id,
            source_ses_idx,
            curriculum_name,
            n_trials,
            int(n_rollouts_per_session),
            "" if int(n_rollouts_per_session) == 1 else "s",
        )
        for rollout_index in range(int(n_rollouts_per_session)):
            random_seed = _derive_session_seed(
                run.seed,
                source_ses_idx,
                rollout_index=rollout_index,
            )
            task = _build_curriculum_matched_task(
                curriculum_name=curriculum_name,
                n_trials=n_trials,
                seed=random_seed,
            )
            if hasattr(task, "reset"):
                task.reset()

            state = copy.deepcopy(runner.initial_state)
            rng = np.random.default_rng(random_seed)
            prev_choice = -1.0
            prev_reward = -1.0
            choice_history: list[float] = []
            reward_history: list[float] = []

            for _ in range(n_trials):
                logits, state = runner.step([prev_choice, prev_reward], state)
                probs = _softmax(logits)
                choice = int(rng.choice(runner.n_actions, p=probs))
                reward = _step_task_reward(task, choice)

                choice_history.append(float(choice))
                reward_history.append(float(reward))
                prev_choice = float(choice)
                prev_reward = float(reward)

            if int(n_rollouts_per_session) == 1:
                simulated_ses_idx = source_ses_idx
            else:
                simulated_ses_idx = f"{source_ses_idx}__rollout_{rollout_index}"

            records.append(
                {
                    "subject_id": subject_id,
                    "ses_idx": simulated_ses_idx,
                    "source_ses_idx": source_ses_idx,
                    "session_date": session_date,
                    "curriculum_name": curriculum_name,
                    "current_stage_actual": current_stage_actual,
                    "n_trials": n_trials,
                    "choice_history": choice_history,
                    "reward_history": reward_history,
                    "nwb_suffix": nwb_suffix,
                    "nwb_name": nwb_name,
                    "random_seed": int(random_seed),
                    "model_dir": run.model_dir,
                    "checkpoint_step": run.checkpoint_step,
                    "rollout_mode": normalized_rollout_mode,
                }
            )
            completed_rollouts += 1
            logger.info(
                "Completed rollout %d/%d for ses_idx=%s seed=%d",
                completed_rollouts,
                total_requested_rollouts,
                simulated_ses_idx,
                int(random_seed),
            )

    logger.info(
        "Finished model simulation: generated %d simulated session%s from %d source session%s.",
        len(records),
        "" if len(records) == 1 else "s",
        total_source_sessions,
        "" if total_source_sessions == 1 else "s",
    )
    return pd.DataFrame.from_records(records)


def compute_switch_stats(
    animal_sessions,
    simulated_sessions,
    window_size: int = _DEFAULT_WINDOW_SIZE,
) -> dict[str, Any]:
    """Compute switch-based statistics for real and simulated sessions."""

    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    animal_with_switches = insert_switch_info(animal_sessions)
    simulated_with_switches = insert_switch_info(simulated_sessions)

    animal_summary = _compute_switch_summary(animal_with_switches, window_size)
    simulated_summary = _compute_switch_summary(simulated_with_switches, window_size)

    return {
        "window_size": window_size,
        "animal": animal_summary,
        "simulated": simulated_summary,
        "comparison": _compare_switch_summaries(animal_summary, simulated_summary),
        "subject_level": _compute_subject_level_comparison(
            animal_with_switches,
            simulated_with_switches,
        ),
    }


def run_post_training_analysis(
    model_dir: str | Path,
    *,
    split: str = _TRAIN_SPLIT,
    checkpoint_policy: str = _CHECKPOINT_POLICY_BEST_EVAL,
    rollout_mode: str = _ROLLOUT_MODE_CURRICULUM_MATCHED,
    n_rollouts_per_session: int = 1,
    window_size: int = _DEFAULT_WINDOW_SIZE,
    output_dir: str | Path | None = None,
    save_animal_session_history: bool = False,
) -> dict[str, Any]:
    """Run the end-to-end standalone post-training generative analysis.

    Parameters
    ----------
    output_dir:
        Optional override for where analysis outputs are written. If omitted,
        outputs are saved under ``<model_dir>/outputs/post_training_analysis/``
        with a run-specific subdirectory.
    save_animal_session_history:
        If ``True``, also save ``animal_session_history.pkl``. This defaults to
        ``False`` because the animal history is reproducible from the resolved
        run configuration and snapshot data.
    """

    started_at = time.perf_counter()
    logger.info(
        "Starting post-training analysis: model_dir=%s split=%s checkpoint_policy=%s "
        "rollout_mode=%s n_rollouts_per_session=%d window_size=%d "
        "save_animal_session_history=%s output_dir=%s",
        model_dir,
        split,
        checkpoint_policy,
        rollout_mode,
        int(n_rollouts_per_session),
        int(window_size),
        save_animal_session_history,
        output_dir,
    )

    stage_started_at = time.perf_counter()
    resolved_run = resolve_model_run(
        model_dir=model_dir,
        split=split,
        checkpoint_policy=checkpoint_policy,
    )
    logger.info(
        "Completed resolve_model_run in %.2fs",
        time.perf_counter() - stage_started_at,
    )

    stage_started_at = time.perf_counter()
    animal_sessions = load_animal_session_history(resolved_run, split=split)
    logger.info(
        "Completed load_animal_session_history in %.2fs",
        time.perf_counter() - stage_started_at,
    )

    stage_started_at = time.perf_counter()
    simulated_sessions = simulate_model_sessions(
        resolved_run=resolved_run,
        animal_sessions=animal_sessions,
        rollout_mode=rollout_mode,
        n_rollouts_per_session=n_rollouts_per_session,
    )
    logger.info(
        "Completed simulate_model_sessions in %.2fs",
        time.perf_counter() - stage_started_at,
    )

    stage_started_at = time.perf_counter()
    switch_stats = compute_switch_stats(
        animal_sessions=animal_sessions,
        simulated_sessions=simulated_sessions,
        window_size=window_size,
    )
    logger.info(
        "Completed compute_switch_stats in %.2fs",
        time.perf_counter() - stage_started_at,
    )

    stage_started_at = time.perf_counter()
    analysis_output_dir = _resolve_analysis_output_dir(
        resolved_run=resolved_run,
        output_dir=output_dir,
    )
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_run_path = analysis_output_dir / "resolved_run.json"
    animal_history_path = analysis_output_dir / "animal_session_history.pkl"
    simulated_history_path = analysis_output_dir / "simulated_session_history.pkl"
    switch_stats_path = analysis_output_dir / "switch_stats.json"

    resolved_run_path.write_text(json.dumps(resolved_run.to_dict(), indent=2))
    if save_animal_session_history:
        with animal_history_path.open("wb") as f:
            pickle.dump(animal_sessions, f)
    with simulated_history_path.open("wb") as f:
        pickle.dump(simulated_sessions, f)

    figure_paths = _save_switch_figures(
        switch_stats=switch_stats,
        output_dir=analysis_output_dir / "figures",
    )
    switch_stats_with_figures = dict(switch_stats)
    switch_stats_with_figures["figure_paths"] = {
        name: str(path) for name, path in figure_paths.items()
    }
    switch_stats_path.write_text(json.dumps(_to_serializable(switch_stats_with_figures), indent=2))
    logger.info(
        "Saved analysis outputs in %.2fs to %s",
        time.perf_counter() - stage_started_at,
        analysis_output_dir,
    )
    logger.info(
        "Finished post-training analysis in %.2fs",
        time.perf_counter() - started_at,
    )

    result = {
        "resolved_run": str(resolved_run_path),
        "simulated_session_history": str(simulated_history_path),
        "switch_stats": str(switch_stats_path),
        "figure_paths": {name: str(path) for name, path in figure_paths.items()},
        "output_dir": str(analysis_output_dir),
    }
    if save_animal_session_history:
        result["animal_session_history"] = str(animal_history_path)
    return result


def extract_switches_and_run_lengths(choices: Sequence[Any]) -> tuple[list[int], list[int]]:
    """Return switch indices and their preceding run lengths."""

    normalized_choices = _clean_choice_history(choices)
    if len(normalized_choices) < 2:
        return [], []

    switch_indices: list[int] = []
    for idx in range(1, len(normalized_choices)):
        if normalized_choices[idx] != normalized_choices[idx - 1]:
            switch_indices.append(idx)

    if not switch_indices:
        return [], []

    previous_switch = 0
    run_lengths = []
    for switch_idx in switch_indices:
        run_lengths.append(switch_idx - previous_switch)
        previous_switch = switch_idx
    return switch_indices, run_lengths


def insert_switch_info(session_rows):
    """Insert switch indices and run lengths into session-history rows."""

    updated_records = []
    for row in _iter_session_records(session_rows):
        updated = dict(row)
        switch_indices, run_lengths = extract_switches_and_run_lengths(
            row.get("choice_history", [])
        )
        updated["switch_indices"] = switch_indices
        updated["run_lengths"] = run_lengths
        updated_records.append(updated)
    return _restore_input_container(session_rows, updated_records)


def compute_switch_probabilities_around_switches(
    session_rows,
    window_size: int,
) -> list[dict[str, Any]]:
    """Compute switch-event windows around every switch."""

    window_size = int(window_size)
    events: list[dict[str, Any]] = []
    for row in _iter_session_records(session_rows):
        choices = _clean_choice_history(row.get("choice_history", []))
        switch_indices = [int(v) for v in row.get("switch_indices", [])]
        if not choices or not switch_indices:
            continue

        for switch_idx in switch_indices:
            if switch_idx <= 0:
                continue

            switch_events = [math.nan] * (2 * window_size + 1)
            start_idx = max(0, switch_idx - window_size)
            end_idx = min(len(choices), switch_idx + window_size + 1)
            for trial_idx in range(start_idx, end_idx):
                if trial_idx == 0:
                    continue
                rel_pos = trial_idx - switch_idx
                arr_idx = rel_pos + window_size
                if 0 <= arr_idx < len(switch_events):
                    switch_events[arr_idx] = float(choices[trial_idx] != choices[trial_idx - 1])

            events.append(
                {
                    "session_idx": row.get("ses_idx"),
                    "subject_id": row.get("subject_id"),
                    "session_date": row.get("session_date"),
                    "switch_trial": switch_idx,
                    "switch_events": switch_events,
                    "relative_positions": list(range(-window_size, window_size + 1)),
                    "window_size": window_size,
                }
            )

    return events


def create_pooled_switch_analysis(
    switch_analysis_data: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Pool switch-event windows into switch probabilities per relative position."""

    if not switch_analysis_data:
        return []

    relative_positions = list(switch_analysis_data[0].get("relative_positions", []))
    pooled: list[dict[str, Any]] = []
    for idx, rel_pos in enumerate(relative_positions):
        values = []
        for event in switch_analysis_data:
            switch_events = list(event.get("switch_events", []))
            if idx >= len(switch_events):
                continue
            value = switch_events[idx]
            if _is_nan(value):
                continue
            values.append(float(value))

        if not values:
            continue

        pooled.append(
            {
                "relative_position": int(rel_pos),
                "switch_probability": _mean(values),
                "switch_probability_sem": _sem(values),
                "n_events": len(values),
                "switch_probability_std": _std(values),
            }
        )

    return pooled


def compute_conditional_switch_probabilities_by_reward(
    session_rows,
    window_size: int,
) -> dict[str, list[dict[str, Any]]]:
    """Compute switch-event windows conditioned on reward outcome."""

    conditioned = {"rewarded": [], "unrewarded": []}
    for row in _iter_session_records(session_rows):
        choices, rewards = _aligned_choice_reward_histories(
            row.get("choice_history", []),
            row.get("reward_history", []),
        )
        switch_indices = [int(v) for v in row.get("switch_indices", [])]
        if not choices or not rewards or not switch_indices:
            continue

        for switch_idx in switch_indices:
            if switch_idx <= 0 or switch_idx >= len(rewards):
                continue
            condition = "rewarded" if rewards[switch_idx] > 0 else "unrewarded"
            switch_events = _build_switch_window(choices, switch_idx, int(window_size))
            conditioned[condition].append(
                {
                    "session_idx": row.get("ses_idx"),
                    "subject_id": row.get("subject_id"),
                    "session_date": row.get("session_date"),
                    "switch_trial": switch_idx,
                    "switch_rewarded": rewards[switch_idx] > 0,
                    "switch_events": switch_events,
                    "relative_positions": list(range(-int(window_size), int(window_size) + 1)),
                    "window_size": int(window_size),
                }
            )

    return conditioned


def analyze_post_switch_probability_by_reward(
    conditional_analysis_data: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Compute p_switch(t+1) for rewarded and unrewarded switch trials."""

    return {
        condition: _summarize_post_switch_probability(events)
        for condition, events in conditional_analysis_data.items()
    }


def compute_conditional_switch_probabilities_by_reward_and_run_length(
    session_rows,
    window_size: int,
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Compute switch-event windows conditioned on reward and run length."""

    conditioned = {
        "rewarded": {"run_length_1": [], "run_length_gt1": []},
        "unrewarded": {"run_length_1": [], "run_length_gt1": []},
    }

    for row in _iter_session_records(session_rows):
        choices, rewards = _aligned_choice_reward_histories(
            row.get("choice_history", []),
            row.get("reward_history", []),
        )
        switch_indices = [int(v) for v in row.get("switch_indices", [])]
        run_lengths = [int(v) for v in row.get("run_lengths", [])]
        if not choices or not rewards or not switch_indices or not run_lengths:
            continue

        for pos, switch_idx in enumerate(switch_indices):
            if switch_idx <= 0 or switch_idx >= len(rewards) or pos >= len(run_lengths):
                continue

            reward_condition = "rewarded" if rewards[switch_idx] > 0 else "unrewarded"
            run_condition = "run_length_1" if run_lengths[pos] == 1 else "run_length_gt1"
            switch_events = _build_switch_window(choices, switch_idx, int(window_size))
            conditioned[reward_condition][run_condition].append(
                {
                    "session_idx": row.get("ses_idx"),
                    "subject_id": row.get("subject_id"),
                    "session_date": row.get("session_date"),
                    "switch_trial": switch_idx,
                    "switch_rewarded": rewards[switch_idx] > 0,
                    "preceding_run_length": run_lengths[pos],
                    "switch_events": switch_events,
                    "relative_positions": list(range(-int(window_size), int(window_size) + 1)),
                    "window_size": int(window_size),
                }
            )

    return conditioned


def analyze_post_switch_probability_by_reward_and_run_length(
    conditional_analysis_data: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Compute p_switch(t+1) for reward x run-length conditions."""

    results: dict[str, dict[str, dict[str, Any]]] = {}
    for reward_condition, run_groups in conditional_analysis_data.items():
        results[reward_condition] = {}
        for run_condition, events in run_groups.items():
            results[reward_condition][run_condition] = _summarize_post_switch_probability(
                events
            )
    return results


def _compute_subject_level_comparison(
    animal_sessions,
    simulated_sessions,
) -> dict[str, Any]:
    animal_subject_stats = _build_subject_level_metric_summary(
        animal_sessions,
        average_rollouts_by_source=False,
    )
    simulated_subject_stats = _build_subject_level_metric_summary(
        simulated_sessions,
        average_rollouts_by_source=True,
    )

    comparison = {
        "post_switch_by_reward": {},
        "post_switch_by_reward_and_run_length": {},
    }

    for reward_condition in _REWARD_CONDITIONS:
        points = _build_subject_comparison_points(
            animal_subject_stats,
            simulated_subject_stats,
            reward_condition=reward_condition,
        )
        comparison["post_switch_by_reward"][reward_condition] = {
            "points": points,
            "summary": _summarize_subject_comparison_points(points),
        }

    for reward_condition in _REWARD_CONDITIONS:
        comparison["post_switch_by_reward_and_run_length"][reward_condition] = {}
        for run_condition in _RUN_LENGTH_CONDITIONS:
            points = _build_subject_comparison_points(
                animal_subject_stats,
                simulated_subject_stats,
                reward_condition=reward_condition,
                run_condition=run_condition,
            )
            comparison["post_switch_by_reward_and_run_length"][reward_condition][
                run_condition
            ] = {
                "points": points,
                "summary": _summarize_subject_comparison_points(points),
            }

    return comparison


def _build_subject_level_metric_summary(
    session_rows,
    *,
    average_rollouts_by_source: bool,
) -> dict[Any, dict[str, Any]]:
    session_counts = [
        _extract_subject_metric_counts(row)
        for row in _iter_session_records(session_rows)
    ]
    if not session_counts:
        return {}

    subject_order = _unique_preserve_order(
        [record["subject_id"] for record in session_counts]
    )
    if average_rollouts_by_source and all(
        record["source_ses_idx"] not in (None, "") for record in session_counts
    ):
        per_subject_records, subject_session_counts = _average_rollout_metric_counts(
            session_counts
        )
    else:
        per_subject_records, subject_session_counts = _collect_direct_metric_counts(
            session_counts
        )

    summaries: dict[Any, dict[str, Any]] = {}
    for subject_id in subject_order:
        subject_records = per_subject_records.get(subject_id, [])
        if not subject_records:
            continue

        reward_totals = _new_reward_count_tree()
        reward_run_totals = _new_reward_run_count_tree()
        for record in subject_records:
            _accumulate_reward_counts(reward_totals, record["reward"])
            _accumulate_reward_run_counts(
                reward_run_totals,
                record["reward_and_run_length"],
            )

        summaries[subject_id] = {
            "post_switch_by_reward": {
                reward_condition: {
                    "probability": _probability_from_counts(
                        reward_totals[reward_condition]
                    ),
                    "n": _normalize_count_output(
                        reward_totals[reward_condition]["n"]
                    ),
                }
                for reward_condition in _REWARD_CONDITIONS
            },
            "post_switch_by_reward_and_run_length": {
                reward_condition: {
                    run_condition: {
                        "probability": _probability_from_counts(
                            reward_run_totals[reward_condition][run_condition]
                        ),
                        "n": _normalize_count_output(
                            reward_run_totals[reward_condition][run_condition]["n"]
                        ),
                    }
                    for run_condition in _RUN_LENGTH_CONDITIONS
                }
                for reward_condition in _REWARD_CONDITIONS
            },
            "n_source_sessions": int(subject_session_counts.get(subject_id, 0)),
        }

    return summaries


def _average_rollout_metric_counts(
    session_counts: Sequence[Mapping[str, Any]],
) -> tuple[dict[Any, list[dict[str, Any]]], dict[Any, int]]:
    grouped: dict[tuple[Any, str], list[Mapping[str, Any]]] = {}
    for record in session_counts:
        source_ses_idx = str(record["source_ses_idx"])
        grouped.setdefault((record["subject_id"], source_ses_idx), []).append(record)

    per_subject_records: dict[Any, list[dict[str, Any]]] = {}
    subject_session_counts: dict[Any, int] = {}
    for (subject_id, _source_ses_idx), records in grouped.items():
        averaged_reward = _new_reward_count_tree()
        averaged_reward_run = _new_reward_run_count_tree()
        n_records = float(len(records))

        for reward_condition in _REWARD_CONDITIONS:
            averaged_reward[reward_condition]["successes"] = (
                sum(
                    float(record["reward"][reward_condition]["successes"])
                    for record in records
                )
                / n_records
            )
            averaged_reward[reward_condition]["n"] = (
                sum(float(record["reward"][reward_condition]["n"]) for record in records)
                / n_records
            )
            for run_condition in _RUN_LENGTH_CONDITIONS:
                averaged_reward_run[reward_condition][run_condition]["successes"] = (
                    sum(
                        float(
                            record["reward_and_run_length"][reward_condition][
                                run_condition
                            ]["successes"]
                        )
                        for record in records
                    )
                    / n_records
                )
                averaged_reward_run[reward_condition][run_condition]["n"] = (
                    sum(
                        float(
                            record["reward_and_run_length"][reward_condition][
                                run_condition
                            ]["n"]
                        )
                        for record in records
                    )
                    / n_records
                )

        per_subject_records.setdefault(subject_id, []).append(
            {
                "reward": averaged_reward,
                "reward_and_run_length": averaged_reward_run,
            }
        )
        subject_session_counts[subject_id] = subject_session_counts.get(subject_id, 0) + 1

    return per_subject_records, subject_session_counts


def _collect_direct_metric_counts(
    session_counts: Sequence[Mapping[str, Any]],
) -> tuple[dict[Any, list[dict[str, Any]]], dict[Any, int]]:
    per_subject_records: dict[Any, list[dict[str, Any]]] = {}
    session_keys_by_subject: dict[Any, list[str]] = {}

    for record in session_counts:
        subject_id = record["subject_id"]
        per_subject_records.setdefault(subject_id, []).append(
            {
                "reward": _copy_reward_count_tree(record["reward"]),
                "reward_and_run_length": _copy_reward_run_count_tree(
                    record["reward_and_run_length"]
                ),
            }
        )
        session_key = (
            record["source_ses_idx"]
            if record["source_ses_idx"] not in (None, "")
            else record["ses_idx"]
        )
        session_keys_by_subject.setdefault(subject_id, []).append(str(session_key))

    subject_session_counts = {
        subject_id: len(_unique_preserve_order(session_keys))
        for subject_id, session_keys in session_keys_by_subject.items()
    }
    return per_subject_records, subject_session_counts


def _extract_subject_metric_counts(row: Mapping[str, Any]) -> dict[str, Any]:
    choices, rewards = _aligned_choice_reward_histories(
        row.get("choice_history", []),
        row.get("reward_history", []),
    )
    switch_indices = [int(v) for v in row.get("switch_indices", [])]
    run_lengths = [int(v) for v in row.get("run_lengths", [])]
    reward_counts = _new_reward_count_tree()
    reward_run_counts = _new_reward_run_count_tree()

    for position, switch_idx in enumerate(switch_indices):
        if (
            switch_idx <= 0
            or switch_idx >= len(rewards)
            or switch_idx + 1 >= len(choices)
            or position >= len(run_lengths)
        ):
            continue

        reward_condition = "rewarded" if rewards[switch_idx] > 0 else "unrewarded"
        run_condition = "run_length_1" if run_lengths[position] == 1 else "run_length_gt1"
        post_switch = float(choices[switch_idx + 1] != choices[switch_idx])
        reward_counts[reward_condition]["successes"] += post_switch
        reward_counts[reward_condition]["n"] += 1.0
        reward_run_counts[reward_condition][run_condition]["successes"] += post_switch
        reward_run_counts[reward_condition][run_condition]["n"] += 1.0

    return {
        "subject_id": _normalize_identifier(row.get("subject_id")),
        "ses_idx": str(row.get("ses_idx")),
        "source_ses_idx": row.get("source_ses_idx"),
        "reward": reward_counts,
        "reward_and_run_length": reward_run_counts,
    }


def _build_subject_comparison_points(
    animal_subject_stats: Mapping[Any, Mapping[str, Any]],
    simulated_subject_stats: Mapping[Any, Mapping[str, Any]],
    *,
    reward_condition: str,
    run_condition: str | None = None,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    animal_subject_ids = _unique_preserve_order(list(animal_subject_stats.keys()))

    for subject_id in animal_subject_ids:
        if subject_id not in simulated_subject_stats:
            continue
        animal_metrics = animal_subject_stats[subject_id]
        simulated_metrics = simulated_subject_stats[subject_id]
        if run_condition is None:
            animal_leaf = _as_dict(
                _as_dict(animal_metrics.get("post_switch_by_reward", {})).get(
                    reward_condition,
                    {},
                )
            )
            simulated_leaf = _as_dict(
                _as_dict(simulated_metrics.get("post_switch_by_reward", {})).get(
                    reward_condition,
                    {},
                )
            )
        else:
            animal_leaf = _as_dict(
                _as_dict(
                    _as_dict(
                        animal_metrics.get("post_switch_by_reward_and_run_length", {})
                    ).get(reward_condition, {})
                ).get(run_condition, {})
            )
            simulated_leaf = _as_dict(
                _as_dict(
                    _as_dict(
                        simulated_metrics.get("post_switch_by_reward_and_run_length", {})
                    ).get(reward_condition, {})
                ).get(run_condition, {})
            )

        points.append(
            {
                "subject_id": subject_id,
                "animal_probability": _coerce_probability(animal_leaf.get("probability")),
                "animal_n": _normalize_count_output(animal_leaf.get("n", 0)),
                "animal_n_sessions": int(animal_metrics.get("n_source_sessions", 0)),
                "simulated_probability": _coerce_probability(
                    simulated_leaf.get("probability")
                ),
                "simulated_effective_n": _normalize_count_output(
                    simulated_leaf.get("n", 0)
                ),
                "simulated_n_source_sessions": int(
                    simulated_metrics.get("n_source_sessions", 0)
                ),
            }
        )

    return points


def _summarize_subject_comparison_points(
    points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid_points = _select_valid_subject_points(points)
    xs = [float(point["animal_probability"]) for point in valid_points]
    ys = [float(point["simulated_probability"]) for point in valid_points]

    return {
        "n_subjects": len(valid_points),
        "correlation": _pearson_correlation(xs, ys),
        "rmse": _rmse(xs, ys),
        "bias": _mean_difference(xs, ys),
    }


def _select_valid_subject_points(
    points: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    valid_points = []
    for point in points:
        animal_probability = _coerce_probability(point.get("animal_probability"))
        simulated_probability = _coerce_probability(point.get("simulated_probability"))
        animal_n = point.get("animal_n")
        if animal_probability is None or simulated_probability is None:
            continue
        if animal_n is None or float(animal_n) < float(_SUBJECT_LEVEL_MIN_ANIMAL_N):
            continue
        valid_points.append(dict(point))
    return valid_points


def _new_reward_count_tree() -> dict[str, dict[str, float]]:
    return {
        reward_condition: {"successes": 0.0, "n": 0.0}
        for reward_condition in _REWARD_CONDITIONS
    }


def _new_reward_run_count_tree() -> dict[str, dict[str, dict[str, float]]]:
    return {
        reward_condition: {
            run_condition: {"successes": 0.0, "n": 0.0}
            for run_condition in _RUN_LENGTH_CONDITIONS
        }
        for reward_condition in _REWARD_CONDITIONS
    }


def _copy_reward_count_tree(
    counts: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    copied = _new_reward_count_tree()
    _accumulate_reward_counts(copied, counts)
    return copied


def _copy_reward_run_count_tree(
    counts: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, dict[str, dict[str, float]]]:
    copied = _new_reward_run_count_tree()
    _accumulate_reward_run_counts(copied, counts)
    return copied


def _accumulate_reward_counts(
    target: dict[str, dict[str, float]],
    source: Mapping[str, Mapping[str, Any]],
) -> None:
    for reward_condition in _REWARD_CONDITIONS:
        source_leaf = _as_dict(source.get(reward_condition, {}))
        target[reward_condition]["successes"] += float(source_leaf.get("successes", 0.0))
        target[reward_condition]["n"] += float(source_leaf.get("n", 0.0))


def _accumulate_reward_run_counts(
    target: dict[str, dict[str, dict[str, float]]],
    source: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    for reward_condition in _REWARD_CONDITIONS:
        source_reward_group = _as_dict(source.get(reward_condition, {}))
        for run_condition in _RUN_LENGTH_CONDITIONS:
            source_leaf = _as_dict(source_reward_group.get(run_condition, {}))
            target[reward_condition][run_condition]["successes"] += float(
                source_leaf.get("successes", 0.0)
            )
            target[reward_condition][run_condition]["n"] += float(
                source_leaf.get("n", 0.0)
            )


def _probability_from_counts(counts: Mapping[str, Any]) -> float:
    total = float(counts.get("n", 0.0))
    if total <= 0:
        return math.nan
    return float(float(counts.get("successes", 0.0)) / total)


def _normalize_count_output(value: Any) -> int | float:
    numeric = float(value)
    if abs(numeric - round(numeric)) < 1e-9:
        return int(round(numeric))
    return numeric


def _compute_switch_summary(session_rows, window_size: int) -> dict[str, Any]:
    pooled_events = compute_switch_probabilities_around_switches(session_rows, window_size)
    reward_conditioned = compute_conditional_switch_probabilities_by_reward(
        session_rows,
        window_size,
    )
    reward_and_run_length = compute_conditional_switch_probabilities_by_reward_and_run_length(
        session_rows,
        window_size,
    )
    return {
        "num_sessions": len(list(_iter_session_records(session_rows))),
        "pooled_switch_probability": create_pooled_switch_analysis(pooled_events),
        "post_switch_by_reward": analyze_post_switch_probability_by_reward(
            reward_conditioned
        ),
        "post_switch_by_reward_and_run_length": (
            analyze_post_switch_probability_by_reward_and_run_length(
                reward_and_run_length
            )
        ),
    }


def _compare_switch_summaries(
    animal_summary: Mapping[str, Any],
    simulated_summary: Mapping[str, Any],
) -> dict[str, Any]:
    comparison = {
        "post_switch_by_reward": {},
        "post_switch_by_reward_and_run_length": {},
    }

    animal_reward = _as_dict(animal_summary.get("post_switch_by_reward", {}))
    simulated_reward = _as_dict(simulated_summary.get("post_switch_by_reward", {}))
    for condition in sorted(set(animal_reward) | set(simulated_reward)):
        animal_prob = _coerce_probability(animal_reward.get(condition, {}).get("probability"))
        simulated_prob = _coerce_probability(
            simulated_reward.get(condition, {}).get("probability")
        )
        comparison["post_switch_by_reward"][condition] = {
            "animal_probability": animal_prob,
            "simulated_probability": simulated_prob,
            "delta_probability": _finite_difference(simulated_prob, animal_prob),
        }

    animal_nested = _as_dict(animal_summary.get("post_switch_by_reward_and_run_length", {}))
    simulated_nested = _as_dict(
        simulated_summary.get("post_switch_by_reward_and_run_length", {})
    )
    reward_conditions = sorted(set(animal_nested) | set(simulated_nested))
    for reward_condition in reward_conditions:
        comparison["post_switch_by_reward_and_run_length"][reward_condition] = {}
        animal_groups = _as_dict(animal_nested.get(reward_condition, {}))
        simulated_groups = _as_dict(simulated_nested.get(reward_condition, {}))
        for run_condition in sorted(set(animal_groups) | set(simulated_groups)):
            animal_prob = _coerce_probability(
                animal_groups.get(run_condition, {}).get("probability")
            )
            simulated_prob = _coerce_probability(
                simulated_groups.get(run_condition, {}).get("probability")
            )
            comparison["post_switch_by_reward_and_run_length"][reward_condition][
                run_condition
            ] = {
                "animal_probability": animal_prob,
                "simulated_probability": simulated_prob,
                "delta_probability": _finite_difference(simulated_prob, animal_prob),
            }

    return comparison


def _normalize_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in {_TRAIN_SPLIT, _HELDOUT_SPLIT}:
        raise ValueError(f"Unsupported split={split!r}. Use 'train' or 'heldout'.")
    return normalized


def _normalize_checkpoint_policy(checkpoint_policy: str) -> str:
    normalized = str(checkpoint_policy).strip().lower()
    if normalized not in {
        _CHECKPOINT_POLICY_BEST_EVAL,
        _CHECKPOINT_POLICY_BEST_HELDOUT,
        _CHECKPOINT_POLICY_FINAL,
    }:
        raise ValueError(
            f"Unsupported checkpoint_policy={checkpoint_policy!r}. "
            "Use 'best_eval', 'best_heldout', or 'final'."
        )
    return normalized


def _resolve_split_selection(
    data_cfg: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    if split == _TRAIN_SPLIT:
        return {
            "subject_ids": data_cfg.get("subject_ids"),
            "subject_start": data_cfg.get("subject_start"),
            "subject_end": data_cfg.get("subject_end"),
        }
    return {
        "subject_ids": data_cfg.get("test_subject_ids"),
        "subject_start": data_cfg.get("test_subject_start"),
        "subject_end": data_cfg.get("test_subject_end"),
    }


def _resolve_checkpoint_artifact(
    *,
    model_dir: Path,
    outputs_dir: Path,
    checkpoint_policy: str,
    checkpoint_index: Mapping[str, Any],
    output_summary: Mapping[str, Any],
) -> tuple[Path, int | None, str, str | None, str | None]:
    reason = None
    fallback_reason = None
    final_params_path = outputs_dir / "params.json"
    if checkpoint_policy == _CHECKPOINT_POLICY_FINAL:
        final_step = _infer_final_step(checkpoint_index, output_summary)
        return (
            final_params_path,
            final_step,
            "final",
            "Selected top-level final params.json.",
            None,
        )

    if checkpoint_policy == _CHECKPOINT_POLICY_BEST_EVAL:
        checkpoints = list(_as_dict(checkpoint_index).get("checkpoints", []) or [])
        if checkpoints:
            best_record = max(
                checkpoints,
                key=lambda item: float(item.get("eval_likelihood", float("-inf"))),
            )
            candidate = _resolve_artifact_path(
                model_dir=model_dir,
                outputs_dir=outputs_dir,
                reported_path=best_record.get("params_path"),
            )
            if candidate.exists():
                return (
                    candidate,
                    _coerce_optional_int(best_record.get("step")),
                    f"step_{best_record.get('step')}",
                    "Selected checkpoint with maximum eval_likelihood from "
                    "outputs/checkpoints/index.json.",
                    None,
                )
            fallback_reason = (
                "best_eval checkpoint params were referenced in "
                "outputs/checkpoints/index.json but the params file was missing; "
                "falling back to final params.json."
            )
        else:
            fallback_reason = (
                "outputs/checkpoints/index.json was missing or empty; "
                "falling back to final params.json."
            )

    if checkpoint_policy == _CHECKPOINT_POLICY_BEST_HELDOUT:
        heldout_records = list(
            _as_dict(output_summary).get("heldout_test_checkpoints", []) or []
        )
        if heldout_records:
            best_record = max(
                heldout_records,
                key=lambda item: float(
                    item.get("heldout_test_likelihood", float("-inf"))
                ),
            )
            candidate = _resolve_artifact_path(
                model_dir=model_dir,
                outputs_dir=outputs_dir,
                reported_path=best_record.get("params_path"),
            )
            if candidate.exists():
                return (
                    candidate,
                    _coerce_optional_int(best_record.get("step")),
                    f"step_{best_record.get('step')}",
                    "Selected checkpoint with maximum heldout_test_likelihood from "
                    "outputs/output_summary.json.",
                    None,
                )
            fallback_reason = (
                "best_heldout checkpoint params were referenced in "
                "outputs/output_summary.json but the params file was missing; "
                "falling back to final params.json."
            )
        else:
            fallback_reason = (
                "outputs/output_summary.json did not include heldout_test_checkpoints; "
                "falling back to final params.json."
            )

    final_step = _infer_final_step(checkpoint_index, output_summary)
    return (
        final_params_path,
        final_step,
        "final",
        reason,
        fallback_reason,
    )


def _infer_final_step(
    checkpoint_index: Mapping[str, Any],
    output_summary: Mapping[str, Any],
) -> int | None:
    if "n_steps" in checkpoint_index:
        return _coerce_optional_int(checkpoint_index.get("n_steps"))
    checkpoints = list(_as_dict(output_summary).get("checkpoints", []) or [])
    if checkpoints:
        last = max(checkpoints, key=lambda item: int(item.get("step", -1)))
        return _coerce_optional_int(last.get("step"))
    return None


def _resolve_artifact_path(
    *,
    model_dir: Path,
    outputs_dir: Path,
    reported_path: Any,
) -> Path:
    if reported_path is None:
        return outputs_dir / "params.json"

    candidate = Path(str(reported_path)).expanduser()
    if candidate.exists():
        return candidate.resolve()

    reported_str = str(reported_path).replace("\\", "/")
    if reported_str.startswith("outputs/"):
        return (model_dir / reported_str).resolve()

    marker = "/outputs/"
    if marker in reported_str:
        suffix = reported_str.split(marker, 1)[1]
        return (outputs_dir / suffix).resolve()

    marker = "outputs/"
    if marker in reported_str:
        suffix = reported_str.split(marker, 1)[1]
        return (outputs_dir / suffix).resolve()

    return (outputs_dir / Path(reported_str).name).resolve()


def _load_structured_file(path: Path) -> Any:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text())
    return _parse_simple_yaml(path.read_text())


def _parse_simple_yaml(text: str) -> Any:
    """Parse a simple YAML subset used by the saved Hydra inputs.

    The fallback parser supports nested mappings, scalar lists, nulls, booleans,
    integers, and floats. It intentionally does not aim to be a complete YAML
    implementation.
    """

    lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        line = _strip_yaml_comment(raw_line.rstrip("\n"))
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        lines.append((indent, line.lstrip(" ")))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        if index >= len(lines):
            return {}, index
        if lines[index][1].startswith("- "):
            return parse_list(index, indent)
        return parse_mapping(index, indent)

    def parse_mapping(index: int, indent: int) -> tuple[dict[str, Any], int]:
        mapping: dict[str, Any] = {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(
                    f"Unexpected indentation while parsing YAML near: {content!r}"
                )
            if content.startswith("- "):
                break

            key, separator, remainder = content.partition(":")
            if separator != ":":
                raise ValueError(f"Invalid YAML line: {content!r}")
            key = key.strip()
            remainder = remainder.strip()
            index += 1
            if not remainder:
                if index < len(lines) and (
                    lines[index][0] > current_indent
                    or (
                        lines[index][0] >= current_indent
                        and lines[index][1].startswith("- ")
                    )
                ):
                    value, index = parse_block(index, lines[index][0])
                else:
                    value = {}
            else:
                value = _parse_scalar(remainder)
            mapping[key] = value
        return mapping, index

    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        items: list[Any] = []
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent or not content.startswith("- "):
                break
            if current_indent > indent:
                raise ValueError(
                    f"Unexpected indentation while parsing YAML list near: {content!r}"
                )

            item = content[2:].strip()
            index += 1
            if not item:
                if index < len(lines) and lines[index][0] > current_indent:
                    value, index = parse_block(index, lines[index][0])
                else:
                    value = None
            else:
                value = _parse_scalar(item)
            items.append(value)
        return items, index

    parsed, final_index = parse_block(0, lines[0][0] if lines else 0)
    if final_index != len(lines):
        raise ValueError("Could not fully parse YAML document.")
    return parsed


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    result_chars = []
    for char in line:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            break
        result_chars.append(char)
    return "".join(result_chars).rstrip()


def _parse_scalar(value: str) -> Any:
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in {"null", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if stripped.startswith(("'", '"')) and stripped.endswith(("'", '"')) and len(stripped) >= 2:
        return stripped[1:-1]
    try:
        if any(token in stripped for token in (".", "e", "E")):
            float_value = float(stripped)
            if math.isfinite(float_value):
                return float_value
        return int(stripped)
    except ValueError:
        return stripped


def _build_session_history_dataframe(raw_df):
    pd = _import_dependency("pandas")
    df = raw_df.copy()
    if "subject_id" not in df.columns or "ses_idx" not in df.columns:
        raise ValueError("Raw trial dataframe must include 'subject_id' and 'ses_idx'.")

    choice_col = _first_existing_column(df.columns, ("animal_response", "action", "choice"))
    reward_col = _first_existing_column(df.columns, ("earned_reward", "rewarded", "reward"))
    if choice_col is None or reward_col is None:
        raise ValueError(
            "Raw trial dataframe must include a choice column and a reward column. "
            f"Observed columns: {list(df.columns)}"
        )

    trial_col = _first_existing_column(
        df.columns,
        ("trial", "trial_index", "trial_number"),
    )
    if trial_col is None:
        df["trial"] = df.groupby("ses_idx").cumcount()
        trial_col = "trial"

    if "session_date" not in df.columns:
        df["session_date"] = df["ses_idx"].map(_extract_session_date)

    records = []
    grouped = df.sort_values(["subject_id", "ses_idx", trial_col]).groupby(
        ["subject_id", "ses_idx"],
        sort=False,
    )
    for (subject_id, ses_idx), session_df in grouped:
        session_df = session_df.sort_values(trial_col).reset_index(drop=True)
        raw_choices = session_df[choice_col].tolist()
        choice_history = [_normalize_choice_value(value) for value in raw_choices]
        reward_history = [
            _normalize_reward_value(value) for value in session_df[reward_col].tolist()
        ]
        session_date = (
            session_df["session_date"].iloc[0]
            if "session_date" in session_df.columns
            else _extract_session_date(ses_idx)
        )
        nwb_suffix = session_df["nwb_suffix"].iloc[0] if "nwb_suffix" in session_df.columns else None
        nwb_name = _build_nwb_name(
            subject_id=subject_id,
            session_date=session_date,
            ses_idx=ses_idx,
            nwb_suffix=nwb_suffix,
            existing_value=(
                session_df["nwb_name"].iloc[0] if "nwb_name" in session_df.columns else None
            ),
        )

        records.append(
            {
                "subject_id": _normalize_identifier(subject_id),
                "ses_idx": str(ses_idx),
                "session_date": str(session_date),
                "curriculum_name": _first_non_null(session_df.get("curriculum_name")),
                "current_stage_actual": _first_non_null(
                    session_df.get("current_stage_actual")
                ),
                "n_trials": int(len(session_df)),
                "choice_history": choice_history,
                "reward_history": reward_history,
                "nwb_suffix": nwb_suffix,
                "nwb_name": nwb_name,
            }
        )

    return pd.DataFrame.from_records(records)


def _build_nwb_name(
    *,
    subject_id: Any,
    session_date: Any,
    ses_idx: Any,
    nwb_suffix: Any,
    existing_value: Any,
) -> str | None:
    if existing_value not in (None, "") and not _is_nan(existing_value):
        return str(existing_value)
    if nwb_suffix not in (None, "") and not _is_nan(nwb_suffix):
        return f"{subject_id}_{session_date}_{nwb_suffix}.nwb"
    if ses_idx not in (None, ""):
        return f"{ses_idx}.nwb"
    return None


def _restore_model_runner(resolved_run: ResolvedModelRun):
    np = _import_dependency("numpy")
    jax = _import_dependency("jax")
    jnp = importlib.import_module("jax.numpy")
    hk = _import_dependency("haiku")

    if resolved_run.multisubject:
        raise NotImplementedError(
            "V1 post-training generative analysis does not yet restore "
            "multisubject models for simulation."
        )

    params = _json_tree_to_arrays(json.loads(Path(resolved_run.params_path).read_text()))
    model_type = resolved_run.model_type

    if model_type == "disrnn":
        disrnn_mod = importlib.import_module("disentangled_rnns.library.disrnn")
        config_dict = copy.deepcopy(_as_dict(resolved_run.model_config))
        config_dict["noiseless_mode"] = True
        config_dict["latent_penalty"] = 0.0
        config_dict["choice_net_latent_penalty"] = 0.0
        config_dict["update_net_obs_penalty"] = 0.0
        config_dict["update_net_latent_penalty"] = 0.0
        config_dict["l2_scale"] = 0.0
        config = disrnn_mod.DisRnnConfig(**config_dict)

        def make_network():
            return disrnn_mod.HkDisentangledRNN(config)

        n_actions = int(config.output_size)
    elif model_type == "gru":
        if resolved_run.multisubject:
            raise NotImplementedError("V1 GRU simulation does not support multisubject runs.")
        make_gru_network = importlib.import_module("models.gru_network").make_gru_network
        config_dict = copy.deepcopy(_as_dict(resolved_run.model_config))
        architecture = _as_dict(config_dict.get("architecture", {}))
        output_size = int(config_dict.get("output_size", architecture.get("output_size", 0)))
        make_network = make_gru_network(
            hidden_size=int(architecture["hidden_size"]),
            output_size=output_size,
            multisubject=False,
        )
        n_actions = int(output_size)
    else:
        raise ValueError(f"Unsupported model_type={model_type!r}")

    if n_actions != 2:
        raise NotImplementedError(
            "V1 generative simulation supports two-action models only. "
            f"Resolved output size: {n_actions}."
        )

    def initial_state_fn():
        core = make_network()
        return core.initial_state(batch_size=1)

    def step_fn(inputs, prev_state):
        core = make_network()
        return core(inputs, prev_state)

    initial_state_transform = hk.without_apply_rng(hk.transform(initial_state_fn))
    step_transform = hk.without_apply_rng(hk.transform(step_fn))

    initial_state_params = initial_state_transform.init(jax.random.PRNGKey(0))
    initial_state = initial_state_transform.apply(initial_state_params)

    class _ModelRunner:
        def __init__(self, initial_state, n_actions: int) -> None:
            self.initial_state = initial_state
            self.n_actions = int(n_actions)

        def step(self, inputs: Sequence[float], prev_state):
            input_array = jnp.asarray([list(inputs)], dtype=jnp.float32)
            logits, next_state = step_transform.apply(params, input_array, prev_state)
            action_logits = np.asarray(logits)[0, : self.n_actions]
            if int(action_logits.shape[0]) != self.n_actions:
                raise ValueError(
                    "Model output logits width does not match n_actions: "
                    f"{action_logits.shape[0]} vs {self.n_actions}."
                )
            return action_logits, next_state

    return _ModelRunner(initial_state=initial_state, n_actions=n_actions)


def _build_curriculum_matched_task(
    *,
    curriculum_name: Any,
    n_trials: int,
    seed: int,
):
    task_mod = importlib.import_module("aind_behavior_gym.dynamic_foraging.task")
    curriculum = str(curriculum_name) if curriculum_name is not None else "None"
    if curriculum == "Uncoupled Baiting":
        return task_mod.UncoupledBlockTask(
            reward_baiting=True,
            num_trials=int(n_trials),
            seed=int(seed),
        )
    if curriculum == "Uncoupled Without Baiting":
        return task_mod.UncoupledBlockTask(
            reward_baiting=False,
            num_trials=int(n_trials),
            seed=int(seed),
        )
    if curriculum == "Coupled Baiting":
        return task_mod.CoupledBlockTask(
            reward_baiting=True,
            num_trials=int(n_trials),
            seed=int(seed),
        )
    if curriculum == "None":
        return task_mod.UncoupledBlockTask(
            reward_baiting=True,
            num_trials=int(n_trials),
            seed=int(seed),
        )
    raise ValueError(f"Unsupported curriculum_name={curriculum_name!r}")


def _step_task_reward(task: Any, action: int) -> float:
    if hasattr(task, "step"):
        result = task.step(int(action))
    elif hasattr(task, "trial"):
        result = task.trial(int(action))
    elif hasattr(task, "update"):
        result = task.update(int(action))
    else:
        raise AttributeError(
            "Could not find a supported task stepping method. "
            "Expected one of: step, trial, update."
        )
    return float(_extract_reward_from_step_result(result))


def _extract_reward_from_step_result(result: Any) -> float:
    if isinstance(result, Mapping):
        for key in ("reward", "earned_reward"):
            if key in result:
                return _normalize_reward_value(result[key])
    if isinstance(result, tuple):
        if len(result) >= 2 and _is_numeric_like(result[1]):
            return _normalize_reward_value(result[1])
        if len(result) >= 1 and _is_numeric_like(result[0]):
            return _normalize_reward_value(result[0])
    if _is_numeric_like(result):
        return _normalize_reward_value(result)
    raise ValueError(f"Unable to extract reward from task step result: {result!r}")


def _resolve_analysis_output_dir(
    *,
    resolved_run: ResolvedModelRun,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()

    step_label = (
        f"step_{resolved_run.checkpoint_step}"
        if resolved_run.checkpoint_step is not None
        else resolved_run.checkpoint_label
    )
    return (
        Path(resolved_run.outputs_dir)
        / "post_training_analysis"
        / f"{resolved_run.split}_{resolved_run.checkpoint_policy}_{step_label}"
    )


def _save_switch_figures(
    *,
    switch_stats: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Path]:
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except ModuleNotFoundError:
        logger.warning(
            "matplotlib is unavailable; skipping switch-analysis figure generation."
        )
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_paths: dict[str, Path] = {}

    pooled_path = output_dir / "pooled_switch_probability.png"
    _plot_pooled_switch_probability(plt, switch_stats, pooled_path)
    figure_paths["pooled_switch_probability"] = pooled_path

    reward_path = output_dir / "post_switch_by_reward.png"
    _plot_post_switch_by_reward(plt, switch_stats, reward_path)
    figure_paths["post_switch_by_reward"] = reward_path

    reward_run_path = output_dir / "post_switch_by_reward_and_run_length.png"
    _plot_post_switch_by_reward_and_run_length(plt, switch_stats, reward_run_path)
    figure_paths["post_switch_by_reward_and_run_length"] = reward_run_path

    reward_subject_scatter_path = output_dir / "post_switch_by_reward_subject_scatter.png"
    _plot_post_switch_by_reward_subject_scatter(
        plt,
        switch_stats,
        reward_subject_scatter_path,
    )
    figure_paths["post_switch_by_reward_subject_scatter"] = reward_subject_scatter_path

    reward_run_subject_scatter_path = (
        output_dir / "post_switch_by_reward_and_run_length_subject_scatter.png"
    )
    _plot_post_switch_by_reward_and_run_length_subject_scatter(
        plt,
        switch_stats,
        reward_run_subject_scatter_path,
    )
    figure_paths[
        "post_switch_by_reward_and_run_length_subject_scatter"
    ] = reward_run_subject_scatter_path

    return figure_paths


def _plot_pooled_switch_probability(plt, switch_stats: Mapping[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color in (("animal", "black"), ("simulated", "tab:blue")):
        pooled = list(_as_dict(switch_stats.get(label, {})).get("pooled_switch_probability", []))
        if not pooled:
            continue
        xs = [row["relative_position"] for row in pooled]
        ys = [row["switch_probability"] for row in pooled]
        yerr = [row["switch_probability_sem"] for row in pooled]
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=3, label=label, color=color)
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.6)
    ax.set_xlabel("Relative trial position")
    ax.set_ylabel("Switch probability")
    ax.set_title("Switch Probability Around Switches")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_by_reward(plt, switch_stats: Mapping[str, Any], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    conditions = ["rewarded", "unrewarded"]
    labels = ["Animal", "Simulated"]
    offsets = [-0.18, 0.18]
    width = 0.32
    x_positions = [0, 1]

    for idx, source in enumerate(("animal", "simulated")):
        condition_stats = _as_dict(
            _as_dict(switch_stats.get(source, {})).get("post_switch_by_reward", {})
        )
        probs = [
            _coerce_probability(condition_stats.get(condition, {}).get("probability"))
            for condition in conditions
        ]
        sems = [
            _coerce_probability(condition_stats.get(condition, {}).get("sem"))
            for condition in conditions
        ]
        xs = [x + offsets[idx] for x in x_positions]
        ax.bar(xs, probs, width=width, yerr=sems, capsize=4, label=labels[idx], alpha=0.8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Rewarded", "Unrewarded"])
    ax.set_ylabel("p_switch(t+1)")
    ax.set_title("Post-switch Probability By Reward")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_by_reward_and_run_length(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = [
        ("rewarded", "run_length_1", "Rewarded\nRun=1"),
        ("rewarded", "run_length_gt1", "Rewarded\nRun>1"),
        ("unrewarded", "run_length_1", "Unrewarded\nRun=1"),
        ("unrewarded", "run_length_gt1", "Unrewarded\nRun>1"),
    ]
    labels = ["Animal", "Simulated"]
    offsets = [-0.18, 0.18]
    width = 0.32
    x_positions = [0, 1, 2, 3]

    for idx, source in enumerate(("animal", "simulated")):
        nested = _as_dict(
            _as_dict(switch_stats.get(source, {})).get(
                "post_switch_by_reward_and_run_length",
                {},
            )
        )
        probs = []
        sems = []
        for reward_condition, run_condition, _ in conditions:
            stats = _as_dict(_as_dict(nested.get(reward_condition, {})).get(run_condition, {}))
            probs.append(_coerce_probability(stats.get("probability")))
            sems.append(_coerce_probability(stats.get("sem")))
        xs = [x + offsets[idx] for x in x_positions]
        ax.bar(xs, probs, width=width, yerr=sems, capsize=4, label=labels[idx], alpha=0.8)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, _, label in conditions])
    ax.set_ylabel("p_switch(t+1)")
    ax.set_title("Post-switch Probability By Reward And Run Length")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_by_reward_subject_scatter(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    subject_level = _as_dict(switch_stats.get("subject_level", {}))
    reward_stats = _as_dict(subject_level.get("post_switch_by_reward", {}))

    for axis, reward_condition, label in zip(
        axes,
        _REWARD_CONDITIONS,
        ("Rewarded", "Unrewarded"),
        strict=False,
    ):
        panel_stats = _as_dict(reward_stats.get(reward_condition, {}))
        _plot_subject_level_scatter_panel(
            ax=axis,
            points=list(panel_stats.get("points", [])),
            summary=_as_dict(panel_stats.get("summary", {})),
            title=label,
        )

    fig.suptitle("Subject-Level Post-switch Probability By Reward", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_post_switch_by_reward_and_run_length_subject_scatter(
    plt,
    switch_stats: Mapping[str, Any],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    subject_level = _as_dict(switch_stats.get("subject_level", {}))
    reward_run_stats = _as_dict(
        subject_level.get("post_switch_by_reward_and_run_length", {})
    )
    panel_definitions = [
        ("rewarded", "run_length_1", "Rewarded / Run=1"),
        ("rewarded", "run_length_gt1", "Rewarded / Run>1"),
        ("unrewarded", "run_length_1", "Unrewarded / Run=1"),
        ("unrewarded", "run_length_gt1", "Unrewarded / Run>1"),
    ]

    for axis, (reward_condition, run_condition, label) in zip(
        axes.flatten(),
        panel_definitions,
        strict=False,
    ):
        panel_stats = _as_dict(
            _as_dict(reward_run_stats.get(reward_condition, {})).get(
                run_condition,
                {},
            )
        )
        _plot_subject_level_scatter_panel(
            ax=axis,
            points=list(panel_stats.get("points", [])),
            summary=_as_dict(panel_stats.get("summary", {})),
            title=label,
        )

    fig.suptitle(
        "Subject-Level Post-switch Probability By Reward And Run Length",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_subject_level_scatter_panel(
    *,
    ax,
    points: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    title: str,
) -> None:
    valid_points = _select_valid_subject_points(points)
    xs = [float(point["animal_probability"]) for point in valid_points]
    ys = [float(point["simulated_probability"]) for point in valid_points]

    if valid_points:
        ax.scatter(
            xs,
            ys,
            alpha=0.7,
            s=45,
            color="tab:blue",
            edgecolors="black",
            linewidths=0.7,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No valid subjects",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )

    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", alpha=0.5, linewidth=1.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Animal p_switch(t+1)")
    ax.set_ylabel("Simulation p_switch(t+1)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if int(summary.get("n_subjects", 0)) >= 2:
        annotation_lines = [
            f"r={float(summary['correlation']):.3f}"
            if summary.get("correlation") is not None
            else "r=None",
            f"RMSE={float(summary['rmse']):.3f}"
            if summary.get("rmse") is not None
            else "RMSE=None",
            f"Bias={float(summary['bias']):.3f}"
            if summary.get("bias") is not None
            else "Bias=None",
            f"n={int(summary['n_subjects'])}",
        ]
        ax.text(
            0.05,
            0.95,
            "\n".join(annotation_lines),
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )


def _derive_session_seed(seed: int | None, session_id: str, *, rollout_index: int) -> int:
    base = "0" if seed is None else str(seed)
    payload = f"{base}:{session_id}:{rollout_index}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return int(digest[:8], 16)


def _softmax(logits: Sequence[float]) -> list[float]:
    values = [float(v) for v in logits]
    max_value = max(values)
    exp_values = [math.exp(v - max_value) for v in values]
    denom = sum(exp_values)
    if denom <= 0:
        raise ValueError(f"Invalid logits for softmax: {logits!r}")
    return [value / denom for value in exp_values]


def _clean_choice_history(choices: Sequence[Any]) -> list[int]:
    normalized = []
    for value in choices:
        coerced = _normalize_choice_value(value)
        if _is_nan(coerced):
            continue
        normalized.append(int(coerced))
    return normalized


def _aligned_choice_reward_histories(
    choices: Sequence[Any],
    rewards: Sequence[Any],
) -> tuple[list[int], list[float]]:
    cleaned_choices: list[int] = []
    cleaned_rewards: list[float] = []
    max_len = min(len(choices), len(rewards))
    for idx in range(max_len):
        choice = _normalize_choice_value(choices[idx])
        reward = _normalize_reward_value(rewards[idx])
        if _is_nan(choice):
            continue
        cleaned_choices.append(int(choice))
        cleaned_rewards.append(float(reward))
    return cleaned_choices, cleaned_rewards


def _build_switch_window(choices: Sequence[int], switch_idx: int, window_size: int) -> list[float]:
    switch_events = [math.nan] * (2 * window_size + 1)
    start_idx = max(0, switch_idx - window_size)
    end_idx = min(len(choices), switch_idx + window_size + 1)
    for trial_idx in range(start_idx, end_idx):
        if trial_idx == 0:
            continue
        rel_pos = trial_idx - switch_idx
        arr_idx = rel_pos + window_size
        if 0 <= arr_idx < len(switch_events):
            switch_events[arr_idx] = float(choices[trial_idx] != choices[trial_idx - 1])
    return switch_events


def _summarize_post_switch_probability(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not events:
        return {"probability": math.nan, "sem": math.nan, "n": 0}

    window_size = int(events[0].get("window_size", 0))
    post_switch_idx = window_size + 1
    values = []
    for event in events:
        switch_events = list(event.get("switch_events", []))
        if post_switch_idx >= len(switch_events):
            continue
        value = switch_events[post_switch_idx]
        if _is_nan(value):
            continue
        values.append(float(value))

    if not values:
        return {"probability": math.nan, "sem": math.nan, "n": 0}
    return {
        "probability": _mean(values),
        "sem": _sem(values),
        "n": len(values),
    }


def _iter_session_records(rows) -> Iterator[dict[str, Any]]:
    if rows is None:
        return iter(())

    if hasattr(rows, "to_dict") and hasattr(rows, "columns"):
        for record in rows.to_dict(orient="records"):
            yield dict(record)
        return

    for record in rows:
        if isinstance(record, Mapping):
            yield dict(record)
        else:
            raise TypeError(
                "Session rows must be a pandas DataFrame or an iterable of mappings."
            )


def _restore_input_container(original, records: list[dict[str, Any]]):
    if hasattr(original, "to_dict") and hasattr(original, "columns"):
        pd = _import_dependency("pandas")
        return pd.DataFrame.from_records(records)
    return records


def _coerce_resolved_run(
    value: str | Path | ResolvedModelRun | Mapping[str, Any],
    *,
    split: str | None = None,
) -> ResolvedModelRun:
    if isinstance(value, ResolvedModelRun):
        return value
    if isinstance(value, Mapping):
        return ResolvedModelRun(**dict(value))
    return resolve_model_run(value, split=split or _TRAIN_SPLIT)


def _json_tree_to_arrays(value: Any) -> Any:
    np = _import_dependency("numpy")
    if isinstance(value, list):
        converted = [_json_tree_to_arrays(item) for item in value]
        try:
            array_value = np.asarray(converted)
            if getattr(array_value, "dtype", None) != object:
                return array_value
        except Exception:
            pass
        return converted
    if isinstance(value, dict):
        return {key: _json_tree_to_arrays(item) for key, item in value.items()}
    return value


def _import_dependency(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Required dependency '{module_name}' is not available in the current "
            "Python environment. This module is expected to run inside the "
            "Code Ocean runtime described by environment/Dockerfile."
        ) from exc


def _first_existing_column(columns: Iterable[Any], candidates: Sequence[str]) -> str | None:
    normalized = {str(column).lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in normalized:
            return str(normalized[candidate.lower()])
    return None


def _extract_session_date(ses_idx: Any) -> str | None:
    if ses_idx in (None, ""):
        return None
    parts = str(ses_idx).split("_")
    if len(parts) >= 2:
        return parts[1]
    return None


def _normalize_choice_value(value: Any) -> float:
    if value is None or _is_nan(value):
        return math.nan
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"left", "l"}:
            return 0.0
        if stripped in {"right", "r"}:
            return 1.0
        if stripped in {"none", "nan", ""}:
            return math.nan
        try:
            value = float(stripped)
        except ValueError:
            return math.nan
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return math.nan
    if value_f == 2.0:
        return math.nan
    return value_f


def _normalize_reward_value(value: Any) -> float:
    if value is None or _is_nan(value):
        return 0.0
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"true", "rewarded"}:
            return 1.0
        if stripped in {"false", "unrewarded", ""}:
            return 0.0
        try:
            return float(stripped)
        except ValueError:
            return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _first_non_null(series_like: Any) -> Any:
    if series_like is None:
        return None
    for value in list(series_like):
        if value is None or _is_nan(value):
            continue
        return value
    return None


def _unique_preserve_order(values: Sequence[Any]) -> list[Any]:
    seen = set()
    unique_values = []
    for value in values:
        normalized = _normalize_identifier(value)
        key = json.dumps(normalized, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(normalized)
    return unique_values


def _normalize_identifier(value: Any) -> Any:
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(value, np.generic):
            return value.item()
    except ModuleNotFoundError:
        pass
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _coerce_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_optional_dict(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _is_nan(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        return False


def _is_numeric_like(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _std(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    mean_value = _mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return float(math.sqrt(variance))


def _sem(values: Sequence[float]) -> float:
    if not values:
        return math.nan
    return float(_std(values) / math.sqrt(len(values)))


def _rmse(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if not xs or not ys or len(xs) != len(ys):
        return None
    return float(math.sqrt(sum((y - x) ** 2 for x, y in zip(xs, ys)) / len(xs)))


def _mean_difference(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if not xs or not ys or len(xs) != len(ys):
        return None
    return float(sum((y - x) for x, y in zip(xs, ys)) / len(xs))


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x_mean = _mean(xs)
    y_mean = _mean(ys)
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    denominator = math.sqrt(
        sum(value * value for value in x_centered)
        * sum(value * value for value in y_centered)
    )
    if denominator <= 0:
        return None
    numerator = sum(x_value * y_value for x_value, y_value in zip(x_centered, y_centered))
    return float(numerator / denominator)


def _coerce_probability(value: Any) -> float | None:
    if value is None or _is_nan(value):
        return None
    return float(value)


def _finite_difference(lhs: float | None, rhs: float | None) -> float | None:
    if lhs is None or rhs is None:
        return None
    return float(lhs - rhs)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, ResolvedModelRun):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        import numpy as np  # type: ignore[import-not-found]

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return _to_serializable(value.item())
    except ModuleNotFoundError:
        pass
    return value
