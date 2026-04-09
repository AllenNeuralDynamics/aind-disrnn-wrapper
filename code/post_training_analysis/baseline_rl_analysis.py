"""Baseline RL post-training analysis from resolved session selections.

This module mirrors the standalone post-training analysis flow used for GRU
and disRNN models, but it sources per-session fitted baseline RL parameters
from a saved dataframe instead of a run directory of learned network weights.
Heavy dependencies are imported lazily so the module remains importable in
lightweight test environments.
"""

from __future__ import annotations

import ast
import json
import logging
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from post_training_analysis.generative_analysis import (
    ResolvedModelRun,
    build_curriculum_matched_task,
    compute_history_dependent_switch_stats,
    compute_switch_stats,
    derive_session_seed,
    load_animal_session_history,
    save_history_dependent_switch_figures,
    save_switch_figures,
    to_serializable,
)

logger = logging.getLogger(__name__)

DEFAULT_BASELINE_RL_MODEL_ALIASES = (
    "QLearning_L1F1_CK1_softmax",
    "QLearning_L2F1_softmax",
    "ForagingCompareThreshold",
)
_SUPPORTED_SESSION_ID_POLICIES = {"auto", "nwb_name", "ses_idx"}
_SUPPORTED_FIT_GAP_POLICIES = {"per_model_skip"}


@dataclass(frozen=True)
class _BaselineRlAliasSpec:
    preset_name: str
    fallback_agent_class: str
    fallback_agent_kwargs: dict[str, Any]


_BASELINE_RL_ALIAS_SPECS: dict[str, _BaselineRlAliasSpec] = {
    "QLearning_L1F1_CK1_softmax": _BaselineRlAliasSpec(
        preset_name="Bari2019",
        fallback_agent_class="ForagerQLearning",
        fallback_agent_kwargs={
            "number_of_learning_rate": 1,
            "number_of_forget_rate": 1,
            "choice_kernel": "one_step",
            "action_selection": "softmax",
        },
    ),
    "QLearning_L2F1_softmax": _BaselineRlAliasSpec(
        preset_name="Hattori2019",
        fallback_agent_class="ForagerQLearning",
        fallback_agent_kwargs={
            "number_of_learning_rate": 2,
            "number_of_forget_rate": 1,
            "choice_kernel": "none",
            "action_selection": "softmax",
        },
    ),
    "ForagingCompareThreshold": _BaselineRlAliasSpec(
        preset_name="CompareToThreshold",
        fallback_agent_class="ForagerCompareThreshold",
        fallback_agent_kwargs={},
    ),
}


def run_baseline_rl_post_training_analysis(
    resolved_run_path: str | Path,
    fitting_df_path: str | Path,
    *,
    model_aliases: Sequence[str] = DEFAULT_BASELINE_RL_MODEL_ALIASES,
    output_dir: str | Path | None = None,
    n_rollouts_per_session: int = 1,
    session_id_policy: str = "auto",
    fit_gap_policy: str = "per_model_skip",
) -> dict[str, Any]:
    """Run post-training analysis for per-session baseline RL model fits."""

    resolved_run = _load_resolved_run_json(resolved_run_path)
    analysis_output_dir = _resolve_output_dir(resolved_run_path, output_dir=output_dir)
    analysis_output_dir.mkdir(parents=True, exist_ok=True)

    normalized_model_aliases = _normalize_model_aliases(model_aliases)
    normalized_session_id_policy = _normalize_session_id_policy(session_id_policy)
    normalized_fit_gap_policy = _normalize_fit_gap_policy(fit_gap_policy)

    animal_sessions = load_animal_session_history(resolved_run)
    selected_animal_sessions, session_resolution = _resolve_requested_animal_sessions(
        resolved_run=resolved_run,
        animal_sessions=animal_sessions,
        session_id_policy=normalized_session_id_policy,
    )
    requested_animal_history_path = analysis_output_dir / "animal_session_history.pkl"
    with requested_animal_history_path.open("wb") as f:
        pickle.dump(selected_animal_sessions, f)

    fitting_df = _load_fitting_dataframe(fitting_df_path)
    prepared_fitting_df = _prepare_fitting_dataframe(
        fitting_df,
        model_aliases=normalized_model_aliases,
    )

    per_model_payloads: dict[str, Any] = {}
    fit_coverage_summary: dict[str, Any] = {
        "resolved_run_path": str(Path(resolved_run_path).expanduser().resolve()),
        "fitting_df_path": str(Path(fitting_df_path).expanduser().resolve()),
        "requested_session_count": int(len(selected_animal_sessions)),
        "requested_session_ids": selected_animal_sessions["requested_session_id"].tolist(),
        "resolved_session_id_mode": str(session_resolution["mode"]),
        "requested_nwb_session_ids": selected_animal_sessions["session_key_nwb"].tolist(),
        "requested_ses_session_ids": selected_animal_sessions["session_key_ses"].tolist(),
        "models": {},
    }
    model_summary_rows: list[dict[str, Any]] = []

    for agent_alias in normalized_model_aliases:
        model_output_dir = analysis_output_dir / _safe_filename_component(agent_alias)
        model_output_dir.mkdir(parents=True, exist_ok=True)

        matched_fit_rows, fit_coverage = _select_fit_rows_for_alias(
            prepared_fitting_df,
            selected_animal_sessions,
            agent_alias=agent_alias,
            fit_gap_policy=normalized_fit_gap_policy,
        )
        fit_coverage_path = model_output_dir / "fit_coverage.json"

        model_result: dict[str, Any] = {
            "agent_alias": agent_alias,
            "status": "skipped" if fit_coverage["available_session_count"] == 0 else "completed",
            "available_session_count": int(fit_coverage["available_session_count"]),
            "missing_session_count": int(fit_coverage["missing_session_count"]),
            "available_session_ids": list(fit_coverage["available_session_ids"]),
            "missing_session_ids": list(fit_coverage["missing_session_ids"]),
            "fit_coverage_path": str(fit_coverage_path),
            "output_dir": str(model_output_dir),
        }
        fit_coverage["output_dir"] = str(model_output_dir)

        if fit_coverage["available_session_count"] == 0:
            fit_coverage["status"] = "skipped"
            fit_coverage["reason"] = (
                "No fitted sessions matched the requested resolved session set."
            )
            fit_coverage_path.write_text(json.dumps(to_serializable(fit_coverage), indent=2))
            model_result["reason"] = fit_coverage["reason"]
        else:
            fit_summary = _compute_fit_likelihood_summary(matched_fit_rows)
            simulated_sessions = _simulate_alias_sessions(
                resolved_run=resolved_run,
                agent_alias=agent_alias,
                fit_rows=matched_fit_rows,
                session_id_mode=str(session_resolution["mode"]),
                fit_gap_policy=normalized_fit_gap_policy,
                n_rollouts_per_session=n_rollouts_per_session,
            )
            simulated_history_path = model_output_dir / "simulated_session_history.pkl"
            with simulated_history_path.open("wb") as f:
                pickle.dump(simulated_sessions, f)

            alias_animal_sessions = selected_animal_sessions[
                selected_animal_sessions["requested_session_id"].isin(
                    fit_coverage["available_session_ids"]
                )
            ].reset_index(drop=True)

            switch_stats = compute_switch_stats(
                animal_sessions=alias_animal_sessions,
                simulated_sessions=simulated_sessions,
            )
            switch_figure_paths = save_switch_figures(
                switch_stats=switch_stats,
                output_dir=model_output_dir / "figures",
            )
            switch_stats_with_figures = dict(switch_stats)
            switch_stats_with_figures["figure_paths"] = {
                name: str(path) for name, path in switch_figure_paths.items()
            }
            switch_stats_path = model_output_dir / "switch_stats.json"
            switch_stats_path.write_text(
                json.dumps(to_serializable(switch_stats_with_figures), indent=2)
            )

            history_stats = compute_history_dependent_switch_stats(
                animal_sessions=alias_animal_sessions,
                simulated_sessions=simulated_sessions,
            )
            history_figure_paths = save_history_dependent_switch_figures(
                history_stats=history_stats,
                output_dir=model_output_dir / "figures",
            )
            history_stats_with_figures = dict(history_stats)
            history_stats_with_figures["figure_paths"] = {
                name: str(path) for name, path in history_figure_paths.items()
            }
            history_stats_path = model_output_dir / "history_dependent_switch_stats.json"
            history_stats_path.write_text(
                json.dumps(to_serializable(history_stats_with_figures), indent=2)
            )

            fit_coverage["status"] = "completed"
            fit_coverage["analysis_session_count"] = int(len(alias_animal_sessions))
            fit_coverage["pooled_total_log_likelihood"] = fit_summary[
                "pooled_total_log_likelihood"
            ]
            fit_coverage["pooled_total_trials"] = fit_summary["pooled_total_trials"]
            fit_coverage["pooled_trial_likelihood"] = fit_summary["pooled_trial_likelihood"]
            fit_coverage["weighted_average_lpt_secondary"] = fit_summary[
                "weighted_average_lpt_secondary"
            ]
            fit_coverage["weighted_average_lpt_secondary_trial_count"] = fit_summary[
                "weighted_average_lpt_secondary_trial_count"
            ]
            fit_coverage["simulated_session_history_path"] = str(simulated_history_path)
            fit_coverage["switch_stats_path"] = str(switch_stats_path)
            fit_coverage["history_dependent_switch_stats_path"] = str(history_stats_path)
            fit_coverage["figure_paths"] = {
                name: str(path)
                for name, path in {**switch_figure_paths, **history_figure_paths}.items()
            }
            fit_coverage_path.write_text(json.dumps(to_serializable(fit_coverage), indent=2))

            model_result.update(
                {
                    **fit_summary,
                    "analysis_session_count": int(len(alias_animal_sessions)),
                    "simulated_session_history_path": str(simulated_history_path),
                    "switch_stats_path": str(switch_stats_path),
                    "history_dependent_switch_stats_path": str(history_stats_path),
                    "figure_paths": fit_coverage["figure_paths"],
                }
            )

        per_model_payloads[agent_alias] = model_result
        fit_coverage_summary["models"][agent_alias] = {
            "status": fit_coverage.get("status"),
            "available_session_count": int(fit_coverage["available_session_count"]),
            "missing_session_count": int(fit_coverage["missing_session_count"]),
            "available_session_ids": list(fit_coverage["available_session_ids"]),
            "missing_session_ids": list(fit_coverage["missing_session_ids"]),
            "fit_coverage_path": str(fit_coverage_path),
            "output_dir": str(model_output_dir),
            "reason": fit_coverage.get("reason"),
            "simulated_session_history_path": model_result.get(
                "simulated_session_history_path"
            ),
            "switch_stats_path": model_result.get("switch_stats_path"),
            "history_dependent_switch_stats_path": model_result.get(
                "history_dependent_switch_stats_path"
            ),
        }

        row = {
            "agent_alias": agent_alias,
            "status": model_result["status"],
            "available_session_count": int(model_result["available_session_count"]),
            "missing_session_count": int(model_result["missing_session_count"]),
            "available_session_ids": json.dumps(model_result["available_session_ids"]),
            "missing_session_ids": json.dumps(model_result["missing_session_ids"]),
            "output_dir": str(model_output_dir),
            "fit_coverage_path": str(fit_coverage_path),
            "reason": model_result.get("reason"),
            "pooled_total_log_likelihood": model_result.get("pooled_total_log_likelihood"),
            "pooled_total_trials": model_result.get("pooled_total_trials"),
            "pooled_trial_likelihood": model_result.get("pooled_trial_likelihood"),
            "weighted_average_lpt_secondary": model_result.get(
                "weighted_average_lpt_secondary"
            ),
            "weighted_average_lpt_secondary_trial_count": model_result.get(
                "weighted_average_lpt_secondary_trial_count"
            ),
            "analysis_session_count": model_result.get("analysis_session_count"),
            "simulated_session_history_path": model_result.get(
                "simulated_session_history_path"
            ),
            "switch_stats_path": model_result.get("switch_stats_path"),
            "history_dependent_switch_stats_path": model_result.get(
                "history_dependent_switch_stats_path"
            ),
        }
        model_summary_rows.append(row)

    fit_coverage_summary_path = analysis_output_dir / "fit_coverage_summary.json"
    fit_coverage_summary_path.write_text(
        json.dumps(to_serializable(fit_coverage_summary), indent=2)
    )

    pd = _import_dependency("pandas")
    model_summary_df = pd.DataFrame.from_records(model_summary_rows)
    model_summary_csv_path = analysis_output_dir / "model_summary.csv"
    model_summary_json_path = analysis_output_dir / "model_summary.json"
    model_summary_df.to_csv(model_summary_csv_path, index=False)

    result = {
        "resolved_run_path": str(Path(resolved_run_path).expanduser().resolve()),
        "fitting_df_path": str(Path(fitting_df_path).expanduser().resolve()),
        "requested_session_count": int(len(selected_animal_sessions)),
        "requested_session_ids": selected_animal_sessions["requested_session_id"].tolist(),
        "resolved_session_id_mode": str(session_resolution["mode"]),
        "animal_session_history": str(requested_animal_history_path),
        "fit_coverage_summary": str(fit_coverage_summary_path),
        "model_summary_csv": str(model_summary_csv_path),
        "model_summary_json": str(model_summary_json_path),
        "output_dir": str(analysis_output_dir),
        "models": per_model_payloads,
    }
    model_summary_json_path.write_text(json.dumps(to_serializable(result), indent=2))
    return result


def _load_resolved_run_json(path: str | Path) -> ResolvedModelRun:
    path_obj = Path(path).expanduser().resolve()
    payload = json.loads(path_obj.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected mapping-style resolved_run payload in {path_obj}")

    allowed_fields = set(ResolvedModelRun.__dataclass_fields__.keys())
    filtered_payload = {
        key: value for key, value in payload.items() if key in allowed_fields
    }
    missing_required = [
        field_name
        for field_name in ("model_dir", "outputs_dir", "model_type", "split")
        if field_name not in filtered_payload
    ]
    if missing_required:
        raise ValueError(
            "resolved_run.json is missing required fields: "
            f"{sorted(missing_required)}"
        )
    return ResolvedModelRun(**filtered_payload)


def _resolve_output_dir(
    resolved_run_path: str | Path,
    *,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    resolved_path = Path(resolved_run_path).expanduser().resolve()
    return resolved_path.parent / "baseline_rl_post_training_analysis"


def _normalize_model_aliases(model_aliases: Sequence[str]) -> tuple[str, ...]:
    aliases = tuple(str(alias) for alias in model_aliases)
    if not aliases:
        raise ValueError("model_aliases must contain at least one alias.")

    unsupported = [alias for alias in aliases if alias not in _BASELINE_RL_ALIAS_SPECS]
    if unsupported:
        raise ValueError(
            "Unsupported baseline RL aliases: "
            f"{unsupported}. Supported aliases: {sorted(_BASELINE_RL_ALIAS_SPECS)}"
        )
    return aliases


def _normalize_session_id_policy(session_id_policy: str) -> str:
    normalized = str(session_id_policy).strip().lower()
    if normalized not in _SUPPORTED_SESSION_ID_POLICIES:
        raise ValueError(
            f"Unsupported session_id_policy={session_id_policy!r}. "
            f"Use one of {sorted(_SUPPORTED_SESSION_ID_POLICIES)}."
        )
    return normalized


def _normalize_fit_gap_policy(fit_gap_policy: str) -> str:
    normalized = str(fit_gap_policy).strip().lower()
    if normalized not in _SUPPORTED_FIT_GAP_POLICIES:
        raise ValueError(
            f"Unsupported fit_gap_policy={fit_gap_policy!r}. "
            f"Use one of {sorted(_SUPPORTED_FIT_GAP_POLICIES)}."
        )
    return normalized


def _resolve_requested_animal_sessions(
    *,
    resolved_run: ResolvedModelRun,
    animal_sessions: Any,
    session_id_policy: str,
):
    pd = _import_dependency("pandas")

    requested_ids_raw = [
        str(session_id) for session_id in (resolved_run.resolved_session_ids or [])
    ]
    if not requested_ids_raw:
        raise ValueError(
            "resolved_run.json does not contain any resolved_session_ids to analyze."
        )

    if len(set(requested_ids_raw)) != len(requested_ids_raw):
        raise ValueError("resolved_session_ids contains duplicates, which is unsupported.")

    records = []
    for record in _iter_session_records(animal_sessions):
        session_key_nwb = _normalize_nwb_name(record.get("nwb_name"))
        session_key_ses = _normalize_session_identifier(record.get("ses_idx"))
        augmented = dict(record)
        augmented["session_key_nwb"] = session_key_nwb
        augmented["session_key_ses"] = session_key_ses
        records.append(augmented)

    by_nwb = _build_unique_session_lookup(records, key_name="session_key_nwb")
    by_ses = _build_unique_session_lookup(records, key_name="session_key_ses")

    requested_nwb_ids = [_normalize_nwb_name(value) for value in requested_ids_raw]
    requested_ses_ids = [_normalize_session_identifier(value) for value in requested_ids_raw]
    nwb_matchable = all(value is not None and value in by_nwb for value in requested_nwb_ids)
    ses_matchable = all(value is not None and value in by_ses for value in requested_ses_ids)

    if session_id_policy == "auto":
        if nwb_matchable:
            resolved_mode = "nwb_name"
            lookup = by_nwb
            normalized_requested_ids = requested_nwb_ids
        elif ses_matchable:
            resolved_mode = "ses_idx"
            lookup = by_ses
            normalized_requested_ids = requested_ses_ids
        else:
            raise ValueError(
                "Could not resolve resolved_session_ids against animal session history. "
                f"Unmatched nwb_name ids: {_unmatched_ids(requested_nwb_ids, by_nwb)}; "
                f"unmatched ses_idx ids: {_unmatched_ids(requested_ses_ids, by_ses)}."
            )
    elif session_id_policy == "nwb_name":
        if not nwb_matchable:
            raise ValueError(
                "resolved_session_ids could not be matched as normalized nwb_name values. "
                f"Unmatched ids: {_unmatched_ids(requested_nwb_ids, by_nwb)}"
            )
        resolved_mode = "nwb_name"
        lookup = by_nwb
        normalized_requested_ids = requested_nwb_ids
    else:
        if not ses_matchable:
            raise ValueError(
                "resolved_session_ids could not be matched as ses_idx values. "
                f"Unmatched ids: {_unmatched_ids(requested_ses_ids, by_ses)}"
            )
        resolved_mode = "ses_idx"
        lookup = by_ses
        normalized_requested_ids = requested_ses_ids

    selected_records = []
    for raw_id, normalized_id in zip(requested_ids_raw, normalized_requested_ids):
        record = dict(lookup[normalized_id])
        record["requested_session_id"] = raw_id
        selected_records.append(record)

    return (
        pd.DataFrame.from_records(selected_records),
        {"mode": resolved_mode},
    )


def _build_unique_session_lookup(
    records: Sequence[Mapping[str, Any]],
    *,
    key_name: str,
) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    duplicates = set()
    for record in records:
        key = record.get(key_name)
        if key in (None, ""):
            continue
        key_text = str(key)
        if key_text in lookup:
            duplicates.add(key_text)
            continue
        lookup[key_text] = dict(record)

    if duplicates:
        raise ValueError(
            f"Animal session history contains duplicate {key_name} values: {sorted(duplicates)}"
        )
    return lookup


def _unmatched_ids(
    requested_ids: Sequence[str | None],
    lookup: Mapping[str, Any],
) -> list[str]:
    return [str(value) for value in requested_ids if value is None or value not in lookup]


def _load_fitting_dataframe(path: str | Path):
    path_obj = Path(path).expanduser().resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Could not find baseline RL fitting dataframe at {path_obj}")
    pd = _import_dependency("pandas")
    dataframe = pd.read_pickle(path_obj)
    if not hasattr(dataframe, "columns"):
        raise ValueError(f"Expected dataframe-like pickle at {path_obj}")
    return pd.DataFrame(dataframe).copy()


def _prepare_fitting_dataframe(
    fitting_df,
    *,
    model_aliases: Sequence[str],
):
    required_columns = {
        "nwb_name",
        "agent_alias",
        "log_likelihood",
        "LPT",
        "n_trials",
        "params",
    }
    missing_columns = sorted(required_columns - set(fitting_df.columns))
    if missing_columns:
        raise ValueError(
            "Baseline RL fitting dataframe is missing required columns: "
            f"{missing_columns}"
        )

    prepared = fitting_df.copy()
    prepared["agent_alias"] = prepared["agent_alias"].map(str)
    prepared = prepared[prepared["agent_alias"].isin(model_aliases)].copy()
    prepared["normalized_nwb_name"] = prepared["nwb_name"].map(_normalize_nwb_name)
    if prepared["normalized_nwb_name"].isna().any():
        missing_count = int(prepared["normalized_nwb_name"].isna().sum())
        raise ValueError(
            f"Baseline RL fitting dataframe contains {missing_count} rows with invalid nwb_name."
        )

    if "_id" not in prepared.columns:
        prepared["_id"] = prepared.index.map(str)
    return prepared.reset_index(drop=True)


def _select_fit_rows_for_alias(
    fitting_df,
    selected_animal_sessions,
    *,
    agent_alias: str,
    fit_gap_policy: str,
):
    if fit_gap_policy != "per_model_skip":
        raise ValueError(f"Unsupported fit_gap_policy={fit_gap_policy!r}")

    alias_fits = fitting_df[fitting_df["agent_alias"] == agent_alias].copy()
    requested_nwb_ids = selected_animal_sessions["session_key_nwb"].tolist()
    duplicate_nwb_ids = (
        alias_fits[alias_fits["normalized_nwb_name"].isin(requested_nwb_ids)]
        .groupby("normalized_nwb_name")
        .size()
    )
    duplicate_nwb_ids = duplicate_nwb_ids[duplicate_nwb_ids > 1]
    if not duplicate_nwb_ids.empty:
        raise ValueError(
            f"Ambiguous duplicate fit rows found for agent_alias={agent_alias!r} and "
            f"nwb_name keys={sorted(duplicate_nwb_ids.index.tolist())}"
        )

    merged = selected_animal_sessions.merge(
        alias_fits,
        how="left",
        left_on="session_key_nwb",
        right_on="normalized_nwb_name",
        suffixes=("", "_fit"),
        sort=False,
    )

    available_mask = merged["normalized_nwb_name"].notna()
    available_session_ids = merged.loc[available_mask, "requested_session_id"].tolist()
    missing_session_ids = merged.loc[~available_mask, "requested_session_id"].tolist()
    _log_fit_metadata_mismatches(
        merged.loc[available_mask].to_dict(orient="records"),
        agent_alias=agent_alias,
    )
    if missing_session_ids:
        logger.warning(
            "Baseline RL fit coverage for %s: missing %d/%d requested sessions: %s",
            agent_alias,
            len(missing_session_ids),
            len(merged),
            missing_session_ids,
        )
    else:
        logger.info(
            "Baseline RL fit coverage for %s: matched all %d requested sessions.",
            agent_alias,
            len(merged),
        )

    coverage = {
        "agent_alias": agent_alias,
        "fit_gap_policy": fit_gap_policy,
        "requested_session_count": int(len(merged)),
        "available_session_count": int(len(available_session_ids)),
        "missing_session_count": int(len(missing_session_ids)),
        "available_session_ids": list(available_session_ids),
        "missing_session_ids": list(missing_session_ids),
        "available_sessions": _coverage_session_records(
            merged.loc[available_mask].to_dict(orient="records")
        ),
        "missing_sessions": _coverage_session_records(
            merged.loc[~available_mask].to_dict(orient="records")
        ),
    }
    return merged.loc[available_mask].reset_index(drop=True), coverage


def _coverage_session_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    for record in records:
        normalized.append(
            {
                "requested_session_id": record.get("requested_session_id"),
                "nwb_name": record.get("nwb_name"),
                "ses_idx": record.get("ses_idx"),
                "subject_id": record.get("subject_id"),
                "session_date": record.get("session_date"),
                "curriculum_name": record.get("curriculum_name"),
                "fit_row_id": record.get("_id"),
            }
        )
    return normalized


def _log_fit_metadata_mismatches(
    records: Sequence[Mapping[str, Any]],
    *,
    agent_alias: str,
) -> None:
    for record in records:
        mismatches = []
        if _values_mismatch(record.get("subject_id_fit"), record.get("subject_id")):
            mismatches.append(
                f"subject_id fit={record.get('subject_id_fit')!r} "
                f"animal={record.get('subject_id')!r}"
            )
        if _values_mismatch(record.get("session_date_fit"), record.get("session_date")):
            mismatches.append(
                f"session_date fit={record.get('session_date_fit')!r} "
                f"animal={record.get('session_date')!r}"
            )
        if _values_mismatch(
            record.get("curriculum_name_fit"),
            record.get("curriculum_name"),
        ):
            mismatches.append(
                f"curriculum_name fit={record.get('curriculum_name_fit')!r} "
                f"animal={record.get('curriculum_name')!r}"
            )
        if _values_mismatch(
            record.get("current_stage_actual_fit"),
            record.get("current_stage_actual"),
        ):
            mismatches.append(
                f"current_stage_actual fit={record.get('current_stage_actual_fit')!r} "
                f"animal={record.get('current_stage_actual')!r}"
            )
        if _values_mismatch(record.get("n_trials_fit"), record.get("n_trials")):
            mismatches.append(
                f"n_trials fit={record.get('n_trials_fit')!r} "
                f"animal={record.get('n_trials')!r}"
            )
        if mismatches:
            logger.warning(
                "Fit metadata mismatch for %s requested_session_id=%s: %s",
                agent_alias,
                record.get("requested_session_id"),
                "; ".join(mismatches),
            )


def _values_mismatch(lhs: Any, rhs: Any) -> bool:
    if _is_missing_value(lhs):
        return False
    return lhs != rhs


def _compute_fit_likelihood_summary(fit_rows) -> dict[str, Any]:
    total_log_likelihood = 0.0
    total_trials = 0
    weighted_lpt_total = 0.0
    weighted_lpt_trials = 0

    for record in fit_rows.to_dict(orient="records"):
        n_trials = _coerce_positive_int(record.get("n_trials"))
        if n_trials is None:
            continue

        log_likelihood = _coerce_float(record.get("log_likelihood"))
        if log_likelihood is not None:
            total_log_likelihood += float(log_likelihood)
            total_trials += int(n_trials)

        lpt = _coerce_float(record.get("LPT"))
        if lpt is not None:
            weighted_lpt_total += float(lpt) * int(n_trials)
            weighted_lpt_trials += int(n_trials)

    pooled_trial_likelihood = (
        float(math.exp(total_log_likelihood / total_trials)) if total_trials > 0 else None
    )
    weighted_average_lpt_secondary = (
        float(weighted_lpt_total / weighted_lpt_trials)
        if weighted_lpt_trials > 0
        else None
    )

    return {
        "pooled_total_log_likelihood": (
            float(total_log_likelihood) if total_trials > 0 else None
        ),
        "pooled_total_trials": int(total_trials),
        "pooled_trial_likelihood": pooled_trial_likelihood,
        "weighted_average_lpt_secondary": weighted_average_lpt_secondary,
        "weighted_average_lpt_secondary_trial_count": int(weighted_lpt_trials),
    }


def _simulate_alias_sessions(
    *,
    resolved_run: ResolvedModelRun,
    agent_alias: str,
    fit_rows,
    session_id_mode: str,
    fit_gap_policy: str,
    n_rollouts_per_session: int,
):
    if int(n_rollouts_per_session) <= 0:
        raise ValueError("n_rollouts_per_session must be >= 1.")

    pd = _import_dependency("pandas")
    records = []
    fit_records = fit_rows.to_dict(orient="records")
    total_requested_rollouts = len(fit_records) * int(n_rollouts_per_session)
    completed_rollouts = 0

    for fit_record in fit_records:
        source_ses_idx = str(fit_record.get("ses_idx"))
        n_trials = int(fit_record.get("n_trials", 0))
        if n_trials <= 0:
            raise ValueError(
                f"Animal session {source_ses_idx!r} has invalid n_trials={n_trials}."
            )

        for rollout_index in range(int(n_rollouts_per_session)):
            rollout_seed = derive_session_seed(
                resolved_run.seed,
                source_ses_idx,
                rollout_index=rollout_index,
            )
            task = build_curriculum_matched_task(
                curriculum_name=fit_record.get("curriculum_name"),
                n_trials=n_trials,
                seed=rollout_seed,
            )
            agent = _instantiate_agent_for_alias(
                agent_alias,
                seed=rollout_seed,
            )
            params = _parse_fit_params(fit_record.get("params"))
            agent.set_params(**params)
            agent.perform(task)

            choice_history = _extract_history_from_agent(
                agent,
                getter_name="get_choice_history",
                expected_length=n_trials,
                label="choice history",
            )
            reward_history = _extract_history_from_agent(
                agent,
                getter_name="get_reward_history",
                expected_length=n_trials,
                label="reward history",
            )

            simulated_ses_idx = (
                source_ses_idx
                if int(n_rollouts_per_session) == 1
                else f"{source_ses_idx}__rollout_{rollout_index}"
            )
            records.append(
                {
                    "subject_id": fit_record.get("subject_id"),
                    "ses_idx": simulated_ses_idx,
                    "source_ses_idx": source_ses_idx,
                    "session_date": fit_record.get("session_date"),
                    "curriculum_name": fit_record.get("curriculum_name"),
                    "current_stage_actual": fit_record.get("current_stage_actual"),
                    "n_trials": int(n_trials),
                    "choice_history": choice_history,
                    "reward_history": reward_history,
                    "nwb_suffix": fit_record.get("nwb_suffix"),
                    "nwb_name": fit_record.get("nwb_name"),
                    "random_seed": int(rollout_seed),
                    "model_dir": resolved_run.model_dir,
                    "checkpoint_step": resolved_run.checkpoint_step,
                    "rollout_mode": "curriculum_matched",
                    "agent_alias": agent_alias,
                    "requested_session_id": fit_record.get("requested_session_id"),
                    "resolved_session_id_mode": session_id_mode,
                    "fit_gap_policy": fit_gap_policy,
                    "fit_row_id": fit_record.get("_id"),
                    "fit_log_likelihood": _coerce_float(fit_record.get("log_likelihood")),
                    "fit_lpt": _coerce_float(fit_record.get("LPT")),
                    "fit_aic": _coerce_float(fit_record.get("AIC")),
                    "fit_bic": _coerce_float(fit_record.get("BIC")),
                    "fit_n_trials": _coerce_positive_int(fit_record.get("n_trials_fit")),
                    "fit_params": params,
                }
            )
            completed_rollouts += 1
            logger.info(
                "Completed baseline RL rollout %d/%d for alias=%s ses_idx=%s seed=%d",
                completed_rollouts,
                total_requested_rollouts,
                agent_alias,
                simulated_ses_idx,
                rollout_seed,
            )

    return pd.DataFrame.from_records(records)


def _instantiate_agent_for_alias(
    agent_alias: str,
    *,
    seed: int,
):
    spec = _BASELINE_RL_ALIAS_SPECS[agent_alias]
    generative_model_mod = _import_dependency(
        "aind_dynamic_foraging_models.generative_model"
    )
    forager_collection_cls = getattr(generative_model_mod, "ForagerCollection", None)
    if forager_collection_cls is not None:
        try:
            return forager_collection_cls().get_preset_forager(
                spec.preset_name,
                seed=seed,
            )
        except Exception as exc:
            logger.warning(
                "Falling back to %s for alias=%s because preset %s could not be loaded: %s",
                spec.fallback_agent_class,
                agent_alias,
                spec.preset_name,
                exc,
            )

    agent_cls = getattr(generative_model_mod, spec.fallback_agent_class, None)
    if agent_cls is None:
        raise ValueError(
            f"Agent class {spec.fallback_agent_class!r} is not available for alias={agent_alias!r}."
        )
    return agent_cls(
        **dict(spec.fallback_agent_kwargs),
        seed=seed,
    )


def _parse_fit_params(value: Any) -> dict[str, Any]:
    parsed = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("Fit params string is empty.")
        parse_errors = []
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(stripped)
                break
            except Exception as exc:
                parse_errors.append(exc)
        else:
            raise ValueError(
                "Could not parse fit params string as JSON or Python literal."
            ) from parse_errors[-1]

    if not isinstance(parsed, Mapping):
        raise ValueError(
            f"Expected fit params to resolve to a mapping, got {type(parsed).__name__}."
        )
    return {
        str(key): _coerce_param_value(item)
        for key, item in dict(parsed).items()
    }


def _coerce_param_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _coerce_param_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_param_value(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _extract_history_from_agent(
    agent: Any,
    *,
    getter_name: str,
    expected_length: int,
    label: str,
) -> list[float]:
    getter = getattr(agent, getter_name, None)
    if getter is None:
        raise AttributeError(f"Simulated baseline RL agent is missing {getter_name}().")

    values = list(getter())
    if len(values) != int(expected_length):
        raise ValueError(
            f"Expected {label} length {expected_length}, got {len(values)}."
        )
    return [float(value) for value in values]


def _iter_session_records(rows) -> list[dict[str, Any]]:
    if hasattr(rows, "to_dict") and hasattr(rows, "columns"):
        return [dict(record) for record in rows.to_dict(orient="records")]
    return [dict(record) for record in rows]


def _normalize_nwb_name(value: Any) -> str | None:
    if _is_missing_value(value):
        return None
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return None
    if text.endswith(".nwb"):
        text = text[:-4]
    return text or None


def _normalize_session_identifier(value: Any) -> str | None:
    if _is_missing_value(value):
        return None
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return None
    return text or None


def _is_missing_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(math.isnan(value))
    except (TypeError, ValueError):
        pass
    return False


def _coerce_positive_int(value: Any) -> int | None:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def _coerce_float(value: Any) -> float | None:
    if _is_missing_value(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_filename_component(value: Any) -> str:
    normalized = str(value)
    safe = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_"
        for ch in normalized
    )
    return safe.strip("_") or "unknown"


def _import_dependency(module_name: str):
    try:
        import importlib

        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Required dependency {module_name!r} is not available in the current "
            "Python environment."
        ) from exc
