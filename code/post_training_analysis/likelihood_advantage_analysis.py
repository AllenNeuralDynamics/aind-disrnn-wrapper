"""Standalone log-likelihood advantage analysis helpers."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from post_training_analysis import likelihood_comparison as lc

logger = logging.getLogger(__name__)

_ANALYSIS_SPLITS = ("train", "eval", "combined", "heldout_test")
_SESSION_STAGE_ORDER = ("early", "mid", "late")
_SWITCH_X_PREV_OUTCOME_ORDER = (
    "rewarded-stay",
    "unrewarded-stay",
    "rewarded-switch",
    "unrewarded-switch",
)
_SESSION_STAGE_COLORS = {
    "early": "#4e79a7",
    "mid": "#59a14f",
    "late": "#e15759",
}
_EPSILON = 1e-6
_MANUAL_Q_PROBABILITY_VALIDATION_ATOL = 1e-4
_BASELINE_PROBABILITY_FRAME_COLUMNS = [
    "subject_id",
    "ses_idx",
    "trial_idx",
    "action",
    "p_rl",
    "p_rl_left",
    "p_rl_right",
    "q_rl_left",
    "q_rl_right",
]


@dataclass(frozen=True)
class VariableSpec:
    """Metadata for one analysis variable."""

    name: str
    label: str
    kind: str
    category_order: tuple[Any, ...] = ()
    n_bins: int = 5


_VARIABLE_SPECS = (
    VariableSpec("prev_outcome", "Previous outcome", "categorical", (0.0, 1.0)),
    VariableSpec("prev_action", "Previous action", "categorical", (0.0, 1.0)),
    VariableSpec("switch", "Switch", "categorical", (0.0, 1.0)),
    VariableSpec("history_pattern_1", "History pattern (1-back)", "categorical"),
    VariableSpec("history_pattern_2", "History pattern (2-back)", "categorical"),
    VariableSpec("history_pattern_3", "History pattern (3-back)", "categorical"),
    VariableSpec("recent_reward_rate_3", "Recent reward rate (3)", "continuous"),
    VariableSpec("recent_reward_rate_5", "Recent reward rate (5)", "continuous"),
    VariableSpec("recent_reward_rate_10", "Recent reward rate (10)", "continuous"),
    VariableSpec("trials_since_reward", "Trials since reward", "continuous"),
    VariableSpec("recent_switch_rate_5", "Recent switch rate (5)", "continuous"),
    VariableSpec("trial_position", "Trial position", "continuous"),
    VariableSpec(
        "switch_x_prev_outcome",
        "Switch x previous outcome",
        "categorical",
        _SWITCH_X_PREV_OUTCOME_ORDER,
    ),
)
_VARIABLE_SPEC_BY_NAME = {spec.name: spec for spec in _VARIABLE_SPECS}


def run_likelihood_advantage_analysis(
    model1_dir: str | Path,
    model2_dir: str | Path,
    *,
    split: str = "combined",
    checkpoint_policy: str = "best_eval",
    output_dir: str | Path | None = None,
    history_warmup: int = 10,
    top_k_variables: int = 3,
    jitter_seed: int = 0,
    include_rnn_state_space: bool = True,
    state_condition_columns: Sequence[str] | None = None,
    state_condition_values_by_column: Mapping[str, Sequence[Any]] | None = None,
    pca_seed: int = 0,
    pca_fit_fraction: float = 0.5,
    include_baseline_q_space: bool = True,
    baseline_q_condition_columns: Sequence[str] | None = None,
    baseline_q_condition_values_by_column: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    """Run a standalone trial-level likelihood advantage analysis."""

    pd = _import_pandas()
    np = _import_numpy()

    if history_warmup < 0:
        raise ValueError("history_warmup must be >= 0.")
    if top_k_variables <= 0:
        raise ValueError("top_k_variables must be > 0.")

    split_name = _normalize_analysis_split(split)
    run_model1 = lc._resolve_likelihood_run(
        model_dir=model1_dir,
        model_label="model1",
        model_index=0,
        checkpoint_policy=checkpoint_policy,
    )
    run_model2 = lc._resolve_likelihood_run(
        model_dir=model2_dir,
        model_label="baseline_rl",
        model_index=1,
        checkpoint_policy=checkpoint_policy,
    )
    _validate_model_pair(run_model1, run_model2)

    resolved_output_dir = _resolve_output_dir(
        run_model1,
        run_model2,
        split_name=split_name,
        output_dir=output_dir,
    )
    figures_dir = resolved_output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model1_evaluation = _load_model1_evaluation_for_split(
        run_model1,
        split_name=split_name,
    )
    base_trial_df = _build_canonical_trial_dataframe(
        model1_evaluation["raw_df"],
        metadata=model1_evaluation["metadata"],
    )
    if base_trial_df.empty:
        raise ValueError("No valid binary-choice trials are available for analysis.")

    model1_prob_df = _extract_model1_probability_frame(
        model1_evaluation["output_df"],
        n_action_logits=int(model1_evaluation["n_action_logits"]),
    )
    trial_df = _merge_probability_frame(
        base_trial_df,
        model1_prob_df,
        prob_column="p_model1",
        source_label=run_model1.model_label,
    )
    model1_state_df = _extract_model1_rnn_state_frame(
        model1_evaluation["output_df"],
    )
    trial_df = _merge_rnn_state_frame(
        trial_df,
        model1_state_df,
        source_label=run_model1.model_label,
    )

    baseline_prob_df = _build_baseline_probability_frame(
        run_model2,
        trial_df=base_trial_df,
        split_name=split_name,
        require_q_values=include_baseline_q_space,
    )
    baseline_q_alignment = dict(baseline_prob_df.attrs.get("q_alignment", {}) or {})
    trial_df = _merge_probability_frame(
        trial_df,
        baseline_prob_df,
        prob_column="p_rl",
        source_label=run_model2.model_label,
    )

    trial_df = pd.DataFrame(trial_df).copy()
    for probability_column in (
        "p_model1",
        "p_model1_left",
        "p_model1_right",
        "p_rl",
        "p_rl_left",
        "p_rl_right",
    ):
        if probability_column in trial_df.columns:
            trial_df[probability_column] = np.clip(
                trial_df[probability_column].to_numpy(dtype=float),
                _EPSILON,
                1.0 - _EPSILON,
            )
    trial_df["log_p_model1"] = np.log(trial_df["p_model1"].to_numpy(dtype=float))
    trial_df["log_p_rl"] = np.log(trial_df["p_rl"].to_numpy(dtype=float))
    trial_df["advantage"] = (
        trial_df["log_p_model1"].to_numpy(dtype=float)
        - trial_df["log_p_rl"].to_numpy(dtype=float)
    )
    trial_df = add_candidate_variables(trial_df, history_warmup=history_warmup)
    trial_df = _add_session_stage(trial_df)
    trial_df.attrs["baseline_q_alignment"] = baseline_q_alignment
    trial_df.attrs["q_source"] = _summarize_q_source(baseline_q_alignment)

    trial_advantage_pickle_path = resolved_output_dir / "trial_advantage.pkl"
    trial_df.to_pickle(trial_advantage_pickle_path)

    rnn_state_space_result: dict[str, Any] | None = None
    if include_rnn_state_space:
        default_condition_columns = [spec.name for spec in _VARIABLE_SPECS]
        rnn_state_space_result = run_rnn_state_space_condition_analysis(
            trial_advantage_pickle_path,
            condition_columns=(
                list(state_condition_columns)
                if state_condition_columns is not None
                else default_condition_columns
            ),
            condition_values_by_column=state_condition_values_by_column,
            output_dir=figures_dir / "rnn_state_space",
            pca_seed=pca_seed,
            pca_fit_fraction=pca_fit_fraction,
        )

    baseline_q_space_result: dict[str, Any] | None = None
    if include_baseline_q_space:
        default_q_condition_columns = [spec.name for spec in _VARIABLE_SPECS]
        baseline_q_space_result = run_baseline_q_space_condition_analysis(
            trial_advantage_pickle_path,
            condition_columns=(
                list(baseline_q_condition_columns)
                if baseline_q_condition_columns is not None
                else default_q_condition_columns
            ),
            condition_values_by_column=baseline_q_condition_values_by_column,
            output_dir=figures_dir / "baseline_q_space",
        )

    advantage_histogram_path = figures_dir / "advantage_histogram.png"
    _plot_advantage_histogram(
        trial_df,
        output_path=advantage_histogram_path,
        run_model1=run_model1,
        run_model2=run_model2,
        split_name=split_name,
    )

    subject_summary_df = _build_subject_mean_advantage_df(
        trial_df,
        jitter_seed=jitter_seed,
    )
    subject_mean_advantage_scatter_path = (
        figures_dir / "subject_mean_advantage_scatter.png"
    )
    _plot_subject_mean_advantage_scatter(
        subject_summary_df,
        output_path=subject_mean_advantage_scatter_path,
        split_name=split_name,
    )

    pooled_frames: list[Any] = []
    per_variable_plot_paths: dict[str, str] = {}
    for spec in _VARIABLE_SPECS:
        analysis_df = analyze_variable_bins(trial_df, spec.name)
        pooled_frames.append(analysis_df)
        plot_path = figures_dir / f"advantage_by_{_safe_filename_component(spec.name)}.png"
        _plot_variable_summary(
            analysis_df,
            variable_name=spec.name,
            output_path=plot_path,
        )
        per_variable_plot_paths[spec.name] = str(plot_path)

    pooled_summary_df = (
        pd.concat(pooled_frames, ignore_index=True)
        if pooled_frames
        else pd.DataFrame(
            columns=[
                "variable",
                "display_label",
                "bin",
                "bin_order",
                "n_trials",
                "mean_advantage",
                "sem_advantage",
            ]
        )
    )
    pooled_summary_df = _sort_summary_dataframe(
        pooled_summary_df,
        groupby_cols=(),
    )

    bin_summary_long_csv_path = resolved_output_dir / "bin_summary_long.csv"
    pooled_summary_df.to_csv(bin_summary_long_csv_path, index=False)

    bin_summary_wide_df = _build_wide_bin_summary(pooled_summary_df)
    bin_summary_wide_csv_path = resolved_output_dir / "bin_summary_wide.csv"
    bin_summary_wide_df.to_csv(bin_summary_wide_csv_path, index=False)

    summary_figure_path = figures_dir / "advantage_summary_figure.png"
    _plot_summary_figure(
        pooled_summary_df,
        output_path=summary_figure_path,
    )

    variable_ranking_df = _rank_variables(pooled_summary_df)
    variable_ranking_csv_path = resolved_output_dir / "variable_ranking.csv"
    variable_ranking_df.to_csv(variable_ranking_csv_path, index=False)

    top_variable_names = [
        str(value)
        for value in variable_ranking_df["variable"].dropna().head(top_k_variables).tolist()
    ]

    session_stage_frames = [
        analyze_variable_bins(
            trial_df,
            variable_name,
            groupby_cols=["session_stage"],
        )
        for variable_name in top_variable_names
    ]
    session_stage_summary_df = (
        pd.concat(session_stage_frames, ignore_index=True)
        if session_stage_frames
        else pd.DataFrame(
            columns=[
                "variable",
                "display_label",
                "session_stage",
                "bin",
                "bin_order",
                "n_trials",
                "mean_advantage",
                "sem_advantage",
            ]
        )
    )
    session_stage_summary_df = _sort_summary_dataframe(
        session_stage_summary_df,
        groupby_cols=("session_stage",),
    )
    session_stage_top3_figure_path = figures_dir / "session_stage_top_variables.png"
    _plot_session_stage_summary(
        session_stage_summary_df,
        variable_names=top_variable_names,
        output_path=session_stage_top3_figure_path,
    )

    subject_frames = [
        analyze_variable_bins(
            trial_df,
            spec.name,
            groupby_cols=["subject_id"],
        )
        for spec in _VARIABLE_SPECS
    ]
    subject_summary_bins_df = (
        pd.concat(subject_frames, ignore_index=True)
        if subject_frames
        else pd.DataFrame(
            columns=[
                "variable",
                "display_label",
                "subject_id",
                "bin",
                "bin_order",
                "n_trials",
                "mean_advantage",
                "sem_advantage",
            ]
        )
    )
    subject_summary_bins_df = _sort_summary_dataframe(
        subject_summary_bins_df,
        groupby_cols=("subject_id",),
    )
    subject_bin_summary_csv_path = resolved_output_dir / "subject_bin_summary.csv"
    subject_summary_bins_df.to_csv(subject_bin_summary_csv_path, index=False)

    summary_payload = {
        "analysis": {
            "split": split_name,
            "checkpoint_policy": checkpoint_policy,
            "history_warmup": int(history_warmup),
            "top_k_variables": int(top_k_variables),
            "jitter_seed": int(jitter_seed),
            "include_rnn_state_space": bool(include_rnn_state_space),
            "state_condition_columns": (
                list(state_condition_columns)
                if state_condition_columns is not None
                else [spec.name for spec in _VARIABLE_SPECS]
            ),
            "pca_seed": int(pca_seed),
            "pca_fit_fraction": float(pca_fit_fraction),
            "include_baseline_q_space": bool(include_baseline_q_space),
            "baseline_q_condition_columns": (
                list(baseline_q_condition_columns)
                if baseline_q_condition_columns is not None
                else [spec.name for spec in _VARIABLE_SPECS]
            ),
            "baseline_q_source": _summarize_q_source(baseline_q_alignment),
            "baseline_q_alignment": baseline_q_alignment,
        },
        "model1": {
            "resolved_run": run_model1.to_dict(),
            "artifact_selection_reason": run_model1.artifact_selection_reason,
        },
        "model2": {
            "resolved_run": run_model2.to_dict(),
            "artifact_selection_reason": run_model2.artifact_selection_reason,
        },
        "top_variables": top_variable_names,
        "artifacts": {
            "trial_advantage_pickle": str(trial_advantage_pickle_path),
            "advantage_histogram": str(advantage_histogram_path),
            "subject_mean_advantage_scatter": str(
                subject_mean_advantage_scatter_path
            ),
            "bin_summary_long_csv": str(bin_summary_long_csv_path),
            "bin_summary_wide_csv": str(bin_summary_wide_csv_path),
            "summary_figure": str(summary_figure_path),
            "variable_ranking_csv": str(variable_ranking_csv_path),
            "session_stage_top3_figure": str(session_stage_top3_figure_path),
            "subject_bin_summary_csv": str(subject_bin_summary_csv_path),
            "per_variable_plots": per_variable_plot_paths,
            "rnn_state_space": rnn_state_space_result,
            "baseline_q_space": baseline_q_space_result,
        },
    }
    summary_path = resolved_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=lc._json_default))

    return {
        "output_dir": str(resolved_output_dir),
        "summary": str(summary_path),
        "trial_advantage_pickle": str(trial_advantage_pickle_path),
        "advantage_histogram": str(advantage_histogram_path),
        "subject_mean_advantage_scatter": str(subject_mean_advantage_scatter_path),
        "bin_summary_long_csv": str(bin_summary_long_csv_path),
        "bin_summary_wide_csv": str(bin_summary_wide_csv_path),
        "summary_figure": str(summary_figure_path),
        "variable_ranking_csv": str(variable_ranking_csv_path),
        "session_stage_top3_figure": str(session_stage_top3_figure_path),
        "subject_bin_summary_csv": str(subject_bin_summary_csv_path),
        "rnn_state_pca_variance": (
            None
            if rnn_state_space_result is None
            else rnn_state_space_result["rnn_state_pca_variance"]
        ),
        "rnn_state_pca_variance_csv": (
            None
            if rnn_state_space_result is None
            else rnn_state_space_result["rnn_state_pca_variance_csv"]
        ),
        "rnn_state_condition_plots": (
            None
            if rnn_state_space_result is None
            else rnn_state_space_result["rnn_state_condition_plots"]
        ),
        "baseline_q_space_summary": (
            None
            if baseline_q_space_result is None
            else baseline_q_space_result["summary"]
        ),
        "baseline_q_condition_plots": (
            None
            if baseline_q_space_result is None
            else baseline_q_space_result["baseline_q_condition_plots"]
        ),
    }


def run_rnn_state_space_condition_analysis(
    trial_advantage_pickle: str | Path,
    *,
    condition_columns: Sequence[str] | None = None,
    condition_values_by_column: Mapping[str, Sequence[Any]] | None = None,
    output_dir: str | Path | None = None,
    pca_seed: int = 0,
    pca_fit_fraction: float = 0.5,
    n_variance_pcs: int = 10,
    n_plot_pcs: int = 4,
) -> dict[str, Any]:
    """Plot RNN state-space projections for condition-matched trials."""

    pd = _import_pandas()

    pickle_path = Path(trial_advantage_pickle).expanduser().resolve()
    trial_df = pd.read_pickle(pickle_path)
    resolved_output_dir = (
        pickle_path.parent / "figures" / "rnn_state_space"
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    state_columns = _rnn_state_columns(trial_df)
    if not state_columns:
        raise ValueError(
            "trial_advantage_pickle must contain rnn_state_* columns before "
            "state-space plotting can run."
        )
    if "advantage" not in pd.DataFrame(trial_df).columns:
        raise ValueError("trial_advantage_pickle must contain an advantage column.")

    pca_result = _fit_and_project_rnn_state_pca(
        trial_df,
        state_columns=state_columns,
        pca_seed=pca_seed,
        pca_fit_fraction=pca_fit_fraction,
    )
    variance_df = _build_pca_variance_dataframe(
        pca_result,
        n_variance_pcs=n_variance_pcs,
    )
    variance_csv_path = resolved_output_dir / "rnn_state_pca_variance.csv"
    variance_df.to_csv(variance_csv_path, index=False)

    variance_plot_path = resolved_output_dir / "rnn_state_pca_variance.png"
    _plot_rnn_state_pca_variance(
        variance_df,
        output_path=variance_plot_path,
    )

    selected_condition_columns = _resolve_state_condition_columns(
        trial_df,
        condition_columns=condition_columns,
    )
    condition_specs = _build_state_condition_specs(
        trial_df,
        condition_columns=selected_condition_columns,
        condition_values_by_column=condition_values_by_column or {},
    )
    condition_plot_paths: dict[str, dict[str, str]] = {}
    condition_root = resolved_output_dir / "conditions"
    for condition_column, condition_entries in condition_specs.items():
        column_dir = condition_root / _safe_filename_component(condition_column)
        condition_plot_paths[condition_column] = {}
        for condition_entry in condition_entries:
            plot_path = column_dir / f"{condition_entry['slug']}.png"
            _plot_rnn_state_condition_figure(
                pca_result,
                condition_column=condition_column,
                condition_label=str(condition_entry["label"]),
                condition_mask=condition_entry["mask"],
                output_path=plot_path,
                n_plot_pcs=n_plot_pcs,
            )
            condition_plot_paths[condition_column][str(condition_entry["label"])] = str(
                plot_path
            )

    return {
        "output_dir": str(resolved_output_dir),
        "trial_advantage_pickle": str(pickle_path),
        "rnn_state_pca_variance": str(variance_plot_path),
        "rnn_state_pca_variance_csv": str(variance_csv_path),
        "rnn_state_condition_plots": condition_plot_paths,
        "condition_columns": selected_condition_columns,
        "pca_fit_n_trials": int(pca_result["fit_n_trials"]),
        "n_trials_projected": int(pca_result["n_trials_projected"]),
    }


def run_rnn_state_space_subject_analysis(
    trial_advantage_pickle: str | Path,
    *,
    probability_column: str = "p_model1_left",
    subject_ids: Sequence[Any] | None = None,
    output_dir: str | Path | None = None,
    pca_seed: int = 0,
    pca_fit_fraction: float = 0.5,
    n_variance_pcs: int = 10,
    n_plot_pcs: int = 4,
) -> dict[str, Any]:
    """Plot shared-PCA RNN states per subject, colored by model probability."""

    pd = _import_pandas()

    pickle_path = Path(trial_advantage_pickle).expanduser().resolve()
    trial_df = pd.read_pickle(pickle_path)
    probability_column = str(probability_column)
    resolved_output_dir = (
        pickle_path.parent
        / "figures"
        / "rnn_state_space_subjects"
        / _safe_filename_component(probability_column)
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    _validate_subject_state_space_dataframe(
        trial_df,
        probability_column=probability_column,
    )
    state_columns = _rnn_state_columns(trial_df)
    pca_result = _fit_and_project_rnn_state_pca(
        trial_df,
        state_columns=state_columns,
        pca_seed=pca_seed,
        pca_fit_fraction=pca_fit_fraction,
        finite_value_columns=[],
    )
    variance_df = _build_pca_variance_dataframe(
        pca_result,
        n_variance_pcs=n_variance_pcs,
    )
    variance_csv_path = resolved_output_dir / "rnn_state_pca_variance.csv"
    variance_df.to_csv(variance_csv_path, index=False)

    variance_plot_path = resolved_output_dir / "rnn_state_pca_variance.png"
    _plot_rnn_state_pca_variance(
        variance_df,
        output_path=variance_plot_path,
    )

    selected_subject_ids = _resolve_subject_ids_for_state_space_plot(
        trial_df,
        subject_ids=subject_ids,
    )
    subject_plot_paths: dict[str, str] = {}
    subject_plot_dir = resolved_output_dir / "subjects"
    subject_series = pd.DataFrame(trial_df)["subject_id"].astype(str)
    for subject_id in selected_subject_ids:
        subject_mask = (subject_series == str(subject_id)).to_numpy(dtype=bool)
        plot_path = subject_plot_dir / f"{_safe_filename_component(subject_id)}.png"
        _plot_rnn_state_subject_probability_figure(
            pca_result,
            subject_id=str(subject_id),
            subject_mask=subject_mask,
            probability_column=probability_column,
            output_path=plot_path,
            n_plot_pcs=n_plot_pcs,
        )
        subject_plot_paths[str(subject_id)] = str(plot_path)

    summary_payload = {
        "trial_advantage_pickle": str(pickle_path),
        "output_dir": str(resolved_output_dir),
        "probability_column": probability_column,
        "subject_ids": selected_subject_ids,
        "pca_seed": int(pca_seed),
        "pca_fit_fraction": float(pca_fit_fraction),
        "pca_fit_n_trials": int(pca_result["fit_n_trials"]),
        "n_trials_projected": int(pca_result["n_trials_projected"]),
        "artifacts": {
            "rnn_state_pca_variance": str(variance_plot_path),
            "rnn_state_pca_variance_csv": str(variance_csv_path),
            "subject_probability_plots": subject_plot_paths,
        },
    }
    summary_path = resolved_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=lc._json_default))

    return {
        "output_dir": str(resolved_output_dir),
        "trial_advantage_pickle": str(pickle_path),
        "probability_column": probability_column,
        "rnn_state_pca_variance": str(variance_plot_path),
        "rnn_state_pca_variance_csv": str(variance_csv_path),
        "subject_probability_plots": subject_plot_paths,
        "subject_ids": selected_subject_ids,
        "pca_fit_n_trials": int(pca_result["fit_n_trials"]),
        "n_trials_projected": int(pca_result["n_trials_projected"]),
    }


def run_baseline_q_space_condition_analysis(
    trial_advantage_pickle: str | Path,
    *,
    condition_columns: Sequence[str] | None = None,
    condition_values_by_column: Mapping[str, Sequence[Any]] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Plot baseline RL Q-space for condition-matched trials."""

    pd = _import_pandas()

    pickle_path = Path(trial_advantage_pickle).expanduser().resolve()
    trial_df = pd.read_pickle(pickle_path)
    resolved_output_dir = (
        pickle_path.parent / "figures" / "baseline_q_space"
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    q_space_result = _build_baseline_q_space_result(
        trial_df,
        finite_value_columns=["advantage"],
    )
    selected_condition_columns = _resolve_state_condition_columns(
        trial_df,
        condition_columns=condition_columns,
    )
    condition_specs = _build_state_condition_specs(
        trial_df,
        condition_columns=selected_condition_columns,
        condition_values_by_column=condition_values_by_column or {},
    )

    condition_plot_paths: dict[str, dict[str, str]] = {}
    condition_root = resolved_output_dir / "conditions"
    for condition_column, condition_entries in condition_specs.items():
        column_dir = condition_root / _safe_filename_component(condition_column)
        condition_plot_paths[condition_column] = {}
        for condition_entry in condition_entries:
            plot_path = column_dir / f"{condition_entry['slug']}.png"
            _plot_baseline_q_space_condition_figure(
                q_space_result,
                condition_column=condition_column,
                condition_label=str(condition_entry["label"]),
                condition_mask=condition_entry["mask"],
                output_path=plot_path,
            )
            condition_plot_paths[condition_column][str(condition_entry["label"])] = str(
                plot_path
            )

    summary_payload = {
        "trial_advantage_pickle": str(pickle_path),
        "output_dir": str(resolved_output_dir),
        "condition_columns": selected_condition_columns,
        "n_trials_projected": int(q_space_result["n_trials_projected"]),
        "artifacts": {
            "baseline_q_condition_plots": condition_plot_paths,
        },
    }
    summary_path = resolved_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=lc._json_default))

    return {
        "output_dir": str(resolved_output_dir),
        "summary": str(summary_path),
        "trial_advantage_pickle": str(pickle_path),
        "condition_columns": selected_condition_columns,
        "baseline_q_condition_plots": condition_plot_paths,
        "n_trials_projected": int(q_space_result["n_trials_projected"]),
    }


def run_baseline_q_space_subject_analysis(
    trial_advantage_pickle: str | Path,
    *,
    probability_column: str = "p_rl_left",
    subject_ids: Sequence[Any] | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Plot baseline RL Q-space per subject, colored by baseline probability."""

    pd = _import_pandas()

    pickle_path = Path(trial_advantage_pickle).expanduser().resolve()
    trial_df = pd.read_pickle(pickle_path)
    probability_column = str(probability_column)
    resolved_output_dir = (
        pickle_path.parent
        / "figures"
        / "baseline_q_space_subjects"
        / _safe_filename_component(probability_column)
        if output_dir is None
        else Path(output_dir).expanduser().resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    _validate_baseline_q_space_dataframe(
        trial_df,
        required_value_columns=[probability_column],
    )
    q_space_result = _build_baseline_q_space_result(
        trial_df,
        finite_value_columns=[],
    )
    selected_subject_ids = _resolve_subject_ids_for_state_space_plot(
        trial_df,
        subject_ids=subject_ids,
    )
    subject_plot_paths: dict[str, str] = {}
    subject_plot_dir = resolved_output_dir / "subjects"
    subject_series = pd.DataFrame(trial_df)["subject_id"].astype(str)
    for subject_id in selected_subject_ids:
        subject_mask = (subject_series == str(subject_id)).to_numpy(dtype=bool)
        plot_path = subject_plot_dir / f"{_safe_filename_component(subject_id)}.png"
        _plot_baseline_q_space_subject_probability_figure(
            q_space_result,
            subject_id=str(subject_id),
            subject_mask=subject_mask,
            probability_column=probability_column,
            output_path=plot_path,
        )
        subject_plot_paths[str(subject_id)] = str(plot_path)

    summary_payload = {
        "trial_advantage_pickle": str(pickle_path),
        "output_dir": str(resolved_output_dir),
        "probability_column": probability_column,
        "subject_ids": selected_subject_ids,
        "n_trials_projected": int(q_space_result["n_trials_projected"]),
        "artifacts": {
            "subject_probability_plots": subject_plot_paths,
        },
    }
    summary_path = resolved_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2, default=lc._json_default))

    return {
        "output_dir": str(resolved_output_dir),
        "summary": str(summary_path),
        "trial_advantage_pickle": str(pickle_path),
        "probability_column": probability_column,
        "subject_probability_plots": subject_plot_paths,
        "subject_ids": selected_subject_ids,
        "n_trials_projected": int(q_space_result["n_trials_projected"]),
    }


def add_candidate_variables(
    trial_df: Any,
    *,
    history_warmup: int = 10,
) -> Any:
    """Add session-history candidate variables to a trial dataframe."""

    pd = _import_pandas()
    np = _import_numpy()

    required_columns = {"ses_idx", "trial_idx", "action", "reward"}
    missing_columns = [column for column in required_columns if column not in trial_df.columns]
    if missing_columns:
        raise ValueError(
            f"trial_df is missing required columns for candidate variables: {missing_columns}"
        )

    enriched_df = pd.DataFrame(trial_df).copy()
    enriched_df = _sort_trial_dataframe(enriched_df)

    categorical_columns = []
    for session_id, session_df in enriched_df.groupby("ses_idx", sort=False):
        session_df = session_df.sort_values("trial_idx").copy()
        index = session_df.index
        choices = session_df["action"].to_numpy(dtype=int)
        rewards = session_df["reward"].to_numpy(dtype=float)

        prev_outcome = np.full(len(session_df), np.nan, dtype=float)
        prev_action = np.full(len(session_df), np.nan, dtype=float)
        switch = np.full(len(session_df), np.nan, dtype=float)
        recent_reward_rate_3 = np.full(len(session_df), np.nan, dtype=float)
        recent_reward_rate_5 = np.full(len(session_df), np.nan, dtype=float)
        recent_reward_rate_10 = np.full(len(session_df), np.nan, dtype=float)
        trials_since_reward = np.full(len(session_df), np.nan, dtype=float)
        recent_switch_rate_5 = np.full(len(session_df), np.nan, dtype=float)
        history_pattern_1: list[str | float] = [math.nan] * len(session_df)
        history_pattern_2: list[str | float] = [math.nan] * len(session_df)
        history_pattern_3: list[str | float] = [math.nan] * len(session_df)
        switch_x_prev_outcome: list[str | float] = [math.nan] * len(session_df)

        encoded_trials = [
            _encode_history_trial(int(choice), float(reward))
            for choice, reward in zip(choices, rewards)
        ]

        last_reward_index: int | None = None
        switch_events = [math.nan] * len(session_df)
        for trial_offset in range(len(session_df)):
            if trial_offset > 0:
                prev_outcome[trial_offset] = rewards[trial_offset - 1]
                prev_action[trial_offset] = choices[trial_offset - 1]
                switch_events[trial_offset] = float(
                    choices[trial_offset] != choices[trial_offset - 1]
                )
                switch[trial_offset] = switch_events[trial_offset]

            if trial_offset >= 1:
                history_pattern_1[trial_offset] = encoded_trials[trial_offset - 1]
            if trial_offset >= 2:
                history_pattern_2[trial_offset] = "".join(
                    encoded_trials[trial_offset - 2 : trial_offset]
                )
            if trial_offset >= 3:
                history_pattern_3[trial_offset] = "".join(
                    encoded_trials[trial_offset - 3 : trial_offset]
                )

            if last_reward_index is not None:
                trials_since_reward[trial_offset] = float(
                    trial_offset - last_reward_index - 1
                )

            previous_rewards = rewards[:trial_offset]
            recent_reward_rate_3[trial_offset] = _mean_last_n(previous_rewards, n=3)
            recent_reward_rate_5[trial_offset] = _mean_last_n(previous_rewards, n=5)
            recent_reward_rate_10[trial_offset] = _mean_last_n(previous_rewards, n=10)
            recent_switch_rate_5[trial_offset] = _mean_last_n(
                switch_events[:trial_offset],
                n=5,
            )

            if trial_offset > 0 and not math.isnan(switch[trial_offset]):
                if prev_outcome[trial_offset] > 0:
                    switch_x_prev_outcome[trial_offset] = (
                        "rewarded-switch"
                        if switch[trial_offset] > 0
                        else "rewarded-stay"
                    )
                else:
                    switch_x_prev_outcome[trial_offset] = (
                        "unrewarded-switch"
                        if switch[trial_offset] > 0
                        else "unrewarded-stay"
                    )

            if rewards[trial_offset] > 0:
                last_reward_index = trial_offset

        session_length = int(len(session_df))
        denom = max(session_length - 1, 1)
        trial_position = (
            (session_df["trial_idx"].to_numpy(dtype=float) - 1.0) / float(denom)
        )

        warmup_mask = session_df["trial_idx"].to_numpy(dtype=int) <= int(history_warmup)
        for values in (
            prev_outcome,
            prev_action,
            switch,
            recent_reward_rate_3,
            recent_reward_rate_5,
            recent_reward_rate_10,
            trials_since_reward,
            recent_switch_rate_5,
        ):
            values[warmup_mask] = np.nan
        for values in (
            history_pattern_1,
            history_pattern_2,
            history_pattern_3,
            switch_x_prev_outcome,
        ):
            for position, masked in enumerate(warmup_mask):
                if masked:
                    values[position] = math.nan

        enriched_df.loc[index, "prev_outcome"] = prev_outcome
        enriched_df.loc[index, "prev_action"] = prev_action
        enriched_df.loc[index, "switch"] = switch
        enriched_df.loc[index, "recent_reward_rate_3"] = recent_reward_rate_3
        enriched_df.loc[index, "recent_reward_rate_5"] = recent_reward_rate_5
        enriched_df.loc[index, "recent_reward_rate_10"] = recent_reward_rate_10
        enriched_df.loc[index, "trials_since_reward"] = trials_since_reward
        enriched_df.loc[index, "recent_switch_rate_5"] = recent_switch_rate_5
        enriched_df.loc[index, "trial_position"] = trial_position
        enriched_df.loc[index, "history_pattern_1"] = history_pattern_1
        enriched_df.loc[index, "history_pattern_2"] = history_pattern_2
        enriched_df.loc[index, "history_pattern_3"] = history_pattern_3
        enriched_df.loc[index, "switch_x_prev_outcome"] = switch_x_prev_outcome
        categorical_columns.extend(
            [
                "history_pattern_1",
                "history_pattern_2",
                "history_pattern_3",
                "switch_x_prev_outcome",
            ]
        )

    enriched_df["switch_x_prev_outcome"] = pd.Categorical(
        enriched_df["switch_x_prev_outcome"],
        categories=list(_SWITCH_X_PREV_OUTCOME_ORDER),
        ordered=True,
    )
    return _sort_trial_dataframe(enriched_df)


def analyze_variable_bins(
    trial_df: Any,
    variable_name: str,
    *,
    groupby_cols: Sequence[str] | None = None,
) -> Any:
    """Aggregate mean advantage by quantile or categorical bins."""

    pd = _import_pandas()
    np = _import_numpy()

    if variable_name not in _VARIABLE_SPEC_BY_NAME:
        raise ValueError(f"Unsupported variable_name={variable_name!r}")

    spec = _VARIABLE_SPEC_BY_NAME[variable_name]
    grouping_columns = list(groupby_cols or [])
    required_columns = ["advantage", variable_name, *grouping_columns]
    missing_columns = [
        column
        for column in required_columns
        if column not in pd.DataFrame(trial_df).columns
    ]
    if missing_columns:
        raise ValueError(
            f"trial_df is missing required columns for bin analysis: {missing_columns}"
        )

    valid_df = pd.DataFrame(trial_df).copy()
    valid_df = valid_df.dropna(subset=["advantage", variable_name]).copy()
    if valid_df.empty:
        return pd.DataFrame(
            columns=[
                "variable",
                "display_label",
                *grouping_columns,
                "bin",
                "bin_order",
                "n_trials",
                "mean_advantage",
                "sem_advantage",
            ]
        )

    if spec.kind == "continuous":
        binned = _quantile_bin_series(
            valid_df[variable_name],
            q=int(spec.n_bins),
        )
        valid_df["_bin_key"] = binned
        categories = list(binned.cat.categories)
        valid_df["bin"] = valid_df["_bin_key"].astype(str)
        valid_df["bin_order"] = valid_df["_bin_key"].cat.codes.astype(int)
    else:
        categories = _resolve_categorical_categories(
            valid_df[variable_name],
            spec=spec,
        )
        valid_df["_bin_key"] = pd.Categorical(
            valid_df[variable_name],
            categories=categories,
            ordered=True,
        )
        valid_df = valid_df[valid_df["_bin_key"].notna()].copy()
        valid_df["bin"] = valid_df["_bin_key"].astype(str)
        valid_df["bin_order"] = valid_df["_bin_key"].cat.codes.astype(int)

    if valid_df.empty:
        return pd.DataFrame(
            columns=[
                "variable",
                "display_label",
                *grouping_columns,
                "bin",
                "bin_order",
                "n_trials",
                "mean_advantage",
                "sem_advantage",
            ]
        )

    grouped = (
        valid_df.groupby(
            [*grouping_columns, "bin", "bin_order"],
            dropna=False,
            observed=False,
        )
        ["advantage"]
        .agg(["size", "mean", "std"])
        .reset_index()
    )
    grouped["sem_advantage"] = np.where(
        grouped["size"].to_numpy(dtype=float) > 1,
        grouped["std"].to_numpy(dtype=float)
        / np.sqrt(grouped["size"].to_numpy(dtype=float)),
        0.0,
    )
    grouped = grouped.rename(
        columns={
            "size": "n_trials",
            "mean": "mean_advantage",
        }
    )
    grouped["variable"] = spec.name
    grouped["display_label"] = spec.label
    grouped = grouped.drop(columns=["std"])
    ordered_columns = [
        "variable",
        "display_label",
        *grouping_columns,
        "bin",
        "bin_order",
        "n_trials",
        "mean_advantage",
        "sem_advantage",
    ]
    return grouped.loc[:, ordered_columns]


def _normalize_analysis_split(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in _ANALYSIS_SPLITS:
        raise ValueError(
            f"Unsupported split={split!r}. Use one of {list(_ANALYSIS_SPLITS)}."
        )
    return normalized


def _validate_model_pair(run_model1: lc.ResolvedLikelihoodRun, run_model2: lc.ResolvedLikelihoodRun) -> None:
    if run_model1.model_type not in {"gru", "disrnn"}:
        raise ValueError(
            "model1_dir must resolve to a GRU or disRNN run, "
            f"received {run_model1.model_type!r}."
        )
    if run_model2.model_type != "baseline_rl":
        raise ValueError(
            "model2_dir must resolve to a baseline RL run, "
            f"received {run_model2.model_type!r}."
        )


def _resolve_output_dir(
    run_model1: lc.ResolvedLikelihoodRun,
    run_model2: lc.ResolvedLikelihoodRun,
    *,
    split_name: str,
    output_dir: str | Path | None,
) -> Path:
    if output_dir is not None:
        resolved = Path(output_dir).expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    pair_label = (
        f"{_safe_filename_component(Path(run_model1.model_dir).name)}"
        f"__vs__{_safe_filename_component(Path(run_model2.model_dir).name)}"
    )
    resolved = (
        Path(run_model1.model_dir).expanduser().resolve()
        / "outputs"
        / f"likelihood_advantage_analysis__{split_name}__{pair_label}"
    )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _load_model1_evaluation_for_split(
    run: lc.ResolvedLikelihoodRun,
    *,
    split_name: str,
) -> dict[str, Any]:
    if split_name in {"train", "eval", "combined"}:
        _, bundle = lc._load_training_bundle_for_run(run)
        split_payloads = lc._build_training_split_payloads(bundle)
        split_payload = split_payloads[split_name]
        raw_df = lc._normalize_raw_dataframe(split_payload["raw_df"])
        metadata = dict(getattr(bundle, "metadata", {}) or {})
        if run.model_type == "gru":
            output_df, n_action_logits = lc._evaluate_gru_dataset(
                run,
                dataset=split_payload["dataset"],
                raw_df=raw_df,
                metadata=metadata,
            )
        else:
            output_df, n_action_logits = lc._evaluate_disrnn_dataset(
                run,
                dataset=split_payload["dataset"],
                raw_df=raw_df,
                metadata=metadata,
            )
        return {
            "raw_df": raw_df,
            "output_df": output_df,
            "metadata": metadata,
            "n_action_logits": int(n_action_logits),
        }

    if split_name != "heldout_test":
        raise ValueError(f"Unsupported split_name={split_name!r}")
    if run.multisubject:
        raise NotImplementedError(
            "Held-out likelihood advantage analysis is not supported for multisubject runs."
        )

    from omegaconf import OmegaConf

    hydra_config = OmegaConf.load(run.inputs_path)
    if run.model_type == "gru":
        from utils.gru_evaluation import load_gru_heldout_subject_data

        heldout_data = load_gru_heldout_subject_data(hydra_config)
    else:
        from utils.disrnn_evaluation import load_disrnn_heldout_subject_data

        heldout_data = load_disrnn_heldout_subject_data(hydra_config)

    raw_df = lc._normalize_raw_dataframe(heldout_data["df_test"])
    metadata = lc._metadata_for_heldout_eval(run.run_config, raw_df=raw_df)
    if run.model_type == "gru":
        output_df, n_action_logits = lc._evaluate_gru_dataset(
            run,
            dataset=heldout_data["dataset_test"],
            raw_df=raw_df,
            metadata=metadata,
        )
    else:
        output_df, n_action_logits = lc._evaluate_disrnn_dataset(
            run,
            dataset=heldout_data["dataset_test"],
            raw_df=raw_df,
            metadata=metadata,
        )
    return {
        "raw_df": raw_df,
        "output_df": output_df,
        "metadata": metadata,
        "n_action_logits": int(n_action_logits),
    }


def _build_canonical_trial_dataframe(
    raw_df: Any,
    *,
    metadata: Mapping[str, Any],
) -> Any:
    pd = _import_pandas()

    normalized_df = lc._normalize_raw_dataframe(pd.DataFrame(raw_df).copy())
    if "ses_idx" not in normalized_df.columns or "animal_response" not in normalized_df.columns:
        raise ValueError("raw_df must include ses_idx and animal_response columns.")

    if "subject_id" not in normalized_df.columns:
        normalized_df["subject_id"] = "unknown"
    if "earned_reward" not in normalized_df.columns:
        normalized_df["earned_reward"] = math.nan
    if "source_ses_idx" not in normalized_df.columns:
        normalized_df["source_ses_idx"] = normalized_df["ses_idx"]

    normalized_df = _sort_raw_dataframe(normalized_df)
    valid_mask = normalized_df["animal_response"].isin([0, 1])
    trial_df = normalized_df.loc[valid_mask].copy()
    if trial_df.empty:
        return pd.DataFrame(
            columns=[
                "subject_id",
                "ses_idx",
                "session_id",
                "session_idx",
                "trial_idx",
                "action",
                "reward",
                "curriculum_name",
            ]
        )

    subject_curriculum_map = lc._build_subject_curriculum_map(
        normalized_df,
        metadata=metadata,
    )
    trial_df["session_id"] = trial_df["source_ses_idx"].where(
        trial_df["source_ses_idx"].notna(),
        trial_df["ses_idx"],
    )
    trial_df["action"] = trial_df["animal_response"].astype(int)
    trial_df["reward"] = trial_df["earned_reward"].astype(float)
    trial_df["trial_idx"] = trial_df.groupby("ses_idx", sort=False).cumcount() + 1

    session_rows = trial_df[["subject_id", "ses_idx"]].drop_duplicates().copy()
    session_rows["session_idx"] = (
        session_rows.groupby("subject_id", sort=False).cumcount() + 1
    )
    session_index_map = {
        (row["subject_id"], row["ses_idx"]): int(row["session_idx"])
        for row in session_rows.to_dict(orient="records")
    }
    trial_df["session_idx"] = [
        session_index_map[(subject_id, session_id)]
        for subject_id, session_id in zip(
            trial_df["subject_id"].tolist(),
            trial_df["ses_idx"].tolist(),
        )
    ]
    trial_df["curriculum_name"] = [
        str(subject_curriculum_map.get(subject_id, "Unknown"))
        for subject_id in trial_df["subject_id"].tolist()
    ]
    trial_df = trial_df.loc[
        :,
        [
            "subject_id",
            "ses_idx",
            "session_id",
            "session_idx",
            "trial_idx",
            "action",
            "reward",
            "curriculum_name",
        ],
    ].reset_index(drop=True)
    return _sort_trial_dataframe(trial_df)


def _extract_model1_probability_frame(
    output_df: Any,
    *,
    n_action_logits: int,
) -> Any:
    pd = _import_pandas()
    np = _import_numpy()

    if n_action_logits < 2:
        raise ValueError(
            "Expected n_action_logits >= 2 so left/right action probabilities can be "
            f"extracted, received {n_action_logits}"
        )

    normalized_output_df = lc._normalize_raw_dataframe(pd.DataFrame(output_df).copy())
    if "ses_idx" not in normalized_output_df.columns or "animal_response" not in normalized_output_df.columns:
        raise ValueError("output_df must include ses_idx and animal_response.")
    normalized_output_df = _sort_raw_dataframe(normalized_output_df)

    records: list[dict[str, Any]] = []
    for session_id in list(dict.fromkeys(normalized_output_df["ses_idx"].tolist())):
        session_df = normalized_output_df[normalized_output_df["ses_idx"] == session_id].copy()
        session_df = session_df.sort_values("trial") if "trial" in session_df.columns else session_df
        action_probabilities = _aligned_action_probabilities_from_output_df(
            session_df,
            n_action_logits=n_action_logits,
        )
        if action_probabilities.ndim != 2 or action_probabilities.shape[0] != len(session_df):
            raise ValueError(
                "Aligned action probabilities must be row-aligned to the session dataframe."
            )

        choices = session_df["animal_response"].to_numpy(dtype=int)
        binary_mask = (choices == 0) | (choices == 1)
        if not np.any(binary_mask):
            continue
        binary_choices = choices[binary_mask]
        binary_action_probabilities = action_probabilities[binary_mask]
        if binary_action_probabilities.shape[1] < 2:
            raise ValueError(
                "Aligned action probabilities must include left and right columns."
            )
        chosen_probabilities = binary_action_probabilities[
            np.arange(len(binary_choices)),
            binary_choices,
        ]
        left_probabilities = binary_action_probabilities[:, 0]
        right_probabilities = binary_action_probabilities[:, 1]
        subject_values = (
            session_df.loc[binary_mask, "subject_id"].tolist()
            if "subject_id" in session_df.columns
            else ["unknown"] * int(np.sum(binary_mask))
        )
        for trial_idx, (
            subject_id,
            action,
            probability,
            left_probability,
            right_probability,
        ) in enumerate(
            zip(
                subject_values,
                binary_choices.tolist(),
                chosen_probabilities.tolist(),
                left_probabilities.tolist(),
                right_probabilities.tolist(),
            ),
            start=1,
        ):
            records.append(
                {
                    "subject_id": lc._normalize_identifier(subject_id),
                    "ses_idx": str(session_id),
                    "trial_idx": int(trial_idx),
                    "action": int(action),
                    "p_model1": float(probability),
                    "p_model1_left": float(left_probability),
                    "p_model1_right": float(right_probability),
                }
            )

    return pd.DataFrame.from_records(
        records,
        columns=[
            "subject_id",
            "ses_idx",
            "trial_idx",
            "action",
            "p_model1",
            "p_model1_left",
            "p_model1_right",
        ],
    )


def _extract_model1_rnn_state_frame(output_df: Any) -> Any:
    pd = _import_pandas()
    np = _import_numpy()

    normalized_output_df = lc._normalize_raw_dataframe(pd.DataFrame(output_df).copy())
    if "ses_idx" not in normalized_output_df.columns or "animal_response" not in normalized_output_df.columns:
        raise ValueError("output_df must include ses_idx and animal_response.")
    normalized_output_df = _sort_raw_dataframe(normalized_output_df)

    latent_columns = _latent_state_columns(normalized_output_df)
    if not latent_columns:
        raise ValueError(
            "Model output dataframe does not contain latent_* columns for RNN states."
        )

    records: list[dict[str, Any]] = []
    for session_id in list(dict.fromkeys(normalized_output_df["ses_idx"].tolist())):
        session_df = normalized_output_df[normalized_output_df["ses_idx"] == session_id].copy()
        session_df = session_df.sort_values("trial") if "trial" in session_df.columns else session_df
        choices = session_df["animal_response"].to_numpy(dtype=int)
        binary_mask = (choices == 0) | (choices == 1)
        if not np.any(binary_mask):
            continue

        subject_values = (
            session_df.loc[binary_mask, "subject_id"].tolist()
            if "subject_id" in session_df.columns
            else ["unknown"] * int(np.sum(binary_mask))
        )
        state_values = session_df.loc[binary_mask, latent_columns].to_numpy(dtype=float)
        for trial_idx, (subject_id, action, state_row) in enumerate(
            zip(
                subject_values,
                choices[binary_mask].tolist(),
                state_values.tolist(),
            ),
            start=1,
        ):
            record = {
                "subject_id": lc._normalize_identifier(subject_id),
                "ses_idx": str(session_id),
                "trial_idx": int(trial_idx),
                "action": int(action),
            }
            for state_idx, state_value in enumerate(state_row):
                record[f"rnn_state_{state_idx}"] = float(state_value)
            records.append(record)

    state_columns = [f"rnn_state_{idx}" for idx in range(len(latent_columns))]
    return pd.DataFrame.from_records(
        records,
        columns=["subject_id", "ses_idx", "trial_idx", "action", *state_columns],
    )


def _latent_state_columns(df: Any) -> list[str]:
    columns = []
    for column in _import_pandas().DataFrame(df).columns:
        match = re.fullmatch(r"latent_(\d+)", str(column))
        if match is not None:
            columns.append((int(match.group(1)), str(column)))
    return [column for _, column in sorted(columns, key=lambda item: item[0])]


def _rnn_state_columns(df: Any) -> list[str]:
    columns = []
    for column in _import_pandas().DataFrame(df).columns:
        match = re.fullmatch(r"rnn_state_(\d+)", str(column))
        if match is not None:
            columns.append((int(match.group(1)), str(column)))
    return [column for _, column in sorted(columns, key=lambda item: item[0])]


def _build_baseline_probability_frame(
    run: lc.ResolvedLikelihoodRun,
    *,
    trial_df: Any,
    split_name: str,
    require_q_values: bool = True,
) -> Any:
    pd = _import_pandas()

    baseline_output = lc._load_baseline_output_for_run(run)
    if lc._baseline_output_is_multisubject_per_subject(baseline_output):
        if split_name == "heldout_test":
            raise NotImplementedError(
                "Held-out likelihood advantage analysis is not supported for multisubject "
                "baseline RL per-subject fits."
            )
        fitted_params_per_subject = baseline_output.get("fitted_params_per_subject")
        if not isinstance(fitted_params_per_subject, Mapping) or not fitted_params_per_subject:
            raise ValueError(
                "baseline_rl_output.json is missing fitted_params_per_subject."
            )

        all_frames = []
        subject_order = list(dict.fromkeys(pd.DataFrame(trial_df)["subject_id"].tolist()))
        for subject_id in subject_order:
            subject_df = pd.DataFrame(trial_df)[
                pd.DataFrame(trial_df)["subject_id"] == subject_id
            ].copy()
            subject_fit_summary = fitted_params_per_subject.get(str(subject_id))
            if subject_fit_summary is None:
                raise ValueError(
                    f"Missing fitted_params_per_subject entry for subject_id={subject_id!r}"
                )
            fitted_params = lc._extract_subject_fitted_params(subject_fit_summary)
            all_frames.append(
                _rollout_baseline_probabilities(
                    run,
                    trial_df=subject_df,
                    fitted_params=fitted_params,
                    require_q_values=require_q_values,
                )
            )
        combined_df = (
            pd.concat(all_frames, ignore_index=True)
            if all_frames
            else pd.DataFrame(columns=_BASELINE_PROBABILITY_FRAME_COLUMNS)
        )
        combined_df.attrs["q_alignment"] = {
            str(frame.attrs.get("subject_id", index)): frame.attrs.get("q_alignment", {})
            for index, frame in enumerate(all_frames)
        }
        combined_df.attrs["q_source"] = _summarize_q_source(combined_df.attrs["q_alignment"])
        return combined_df

    fitted_params = baseline_output.get("fitted_params")
    if not isinstance(fitted_params, Mapping) or not fitted_params:
        raise ValueError("baseline_rl_output.json is missing fitted_params.")
    return _rollout_baseline_probabilities(
        run,
        trial_df=pd.DataFrame(trial_df).copy(),
        fitted_params=dict(fitted_params),
        require_q_values=require_q_values,
    )


def _rollout_baseline_probabilities(
    run: lc.ResolvedLikelihoodRun,
    *,
    trial_df: Any,
    fitted_params: Mapping[str, Any],
    require_q_values: bool = True,
) -> Any:
    pd = _import_pandas()

    baseline_output = lc._load_baseline_output_for_run(run)
    agent_class_name, agent_kwargs = lc._resolve_baseline_agent_spec(run, baseline_output)
    session_payloads = _session_rollout_payloads_from_trial_df(trial_df)
    if not session_payloads:
        return pd.DataFrame(columns=_BASELINE_PROBABILITY_FRAME_COLUMNS)

    choice_prob_sessions, q_value_sessions, q_alignment = (
        _perform_baseline_agent_rollout_with_q_histories(
            agent_class_name=agent_class_name,
            agent_kwargs=agent_kwargs,
            fitted_params=fitted_params,
            choice_sessions=[payload["choices"] for payload in session_payloads],
            reward_sessions=[payload["rewards"] for payload in session_payloads],
            seed=run.seed,
            require_q_values=require_q_values,
        )
    )
    if len(choice_prob_sessions) != len(session_payloads):
        raise ValueError(
            "Baseline rollout returned a different number of sessions than requested: "
            f"expected={len(session_payloads)} received={len(choice_prob_sessions)}"
        )
    if q_value_sessions is not None and len(q_value_sessions) != len(session_payloads):
        raise ValueError(
            "Baseline Q-history recovery returned a different number of sessions than "
            f"requested: expected={len(session_payloads)} received={len(q_value_sessions)}"
        )
    result_df = _probability_frame_from_rollout_sessions(
        session_payloads,
        choice_prob_sessions=choice_prob_sessions,
        q_value_sessions=q_value_sessions,
    )
    result_df.attrs["q_alignment"] = q_alignment
    result_df.attrs["q_source"] = _summarize_q_source(q_alignment)
    if session_payloads:
        result_df.attrs["subject_id"] = str(session_payloads[0]["subject_id"])
    return result_df


def _perform_baseline_agent_rollout_with_q_histories(
    *,
    agent_class_name: str,
    agent_kwargs: Mapping[str, Any],
    fitted_params: Mapping[str, Any],
    choice_sessions: Sequence[Any],
    reward_sessions: Sequence[Any],
    seed: int | None,
    require_q_values: bool,
) -> tuple[list[Any], list[Any] | None, dict[str, Any]]:
    from aind_dynamic_foraging_models import generative_model

    agent_class_obj = getattr(generative_model, agent_class_name, None)
    if agent_class_obj is None:
        raise ValueError(
            f"Agent class {agent_class_name!r} was not found in generative_model."
        )

    agent = agent_class_obj(
        **dict(agent_kwargs),
        seed=seed,
    )
    agent.set_params(
        **{
            str(param_name): float(param_value)
            for param_name, param_value in dict(fitted_params).items()
        }
    )
    choice_prob_sessions = list(
        agent.perform_closed_loop_multi_session(
            list(choice_sessions),
            list(reward_sessions),
        )
    )
    if not require_q_values:
        return choice_prob_sessions, None, {
            "status": "not_requested",
            "q_source": "not_requested",
        }

    expected_trials_per_session = [len(session) for session in choice_sessions]
    try:
        q_value_sessions, q_alignment = _extract_policy_time_q_histories(
            agent,
            choice_prob_sessions=choice_prob_sessions,
            expected_trials_per_session=expected_trials_per_session,
        )
    except ValueError as recovery_error:
        if str(agent_class_name) != "ForagerQLearning":
            raise ValueError(
                f"Could not recover policy-time Q values for baseline agent "
                f"{agent_class_name!r}. Q-space plotting is supported when the fitted "
                "agent exposes two-action Q histories or for supported ForagerQLearning "
                "softmax configurations. Run with include_baseline_q_space=False to "
                "skip baseline Q-space plots."
            ) from recovery_error
        try:
            q_value_sessions, q_alignment = _manual_forager_q_learning_q_histories(
                agent_kwargs=agent_kwargs,
                fitted_params=fitted_params,
                choice_sessions=choice_sessions,
                reward_sessions=reward_sessions,
                choice_prob_sessions=choice_prob_sessions,
            )
        except ValueError as manual_error:
            raise ValueError(
                f"Could not recover policy-time Q values for baseline agent "
                f"{agent_class_name!r}. Manual ForagerQLearning Q-space fallback "
                f"failed: {manual_error}. Run with include_baseline_q_space=False "
                "to skip baseline Q-space plots."
            ) from manual_error
    else:
        q_alignment = dict(q_alignment)
        q_alignment.setdefault("q_source", "agent_exposed_history")
        q_alignment.setdefault("agent_class", str(agent_class_name))
    return choice_prob_sessions, q_value_sessions, q_alignment


def _manual_forager_q_learning_q_histories(
    *,
    agent_kwargs: Mapping[str, Any],
    fitted_params: Mapping[str, Any],
    choice_sessions: Sequence[Any],
    reward_sessions: Sequence[Any],
    choice_prob_sessions: Sequence[Any],
) -> tuple[list[Any], dict[str, Any]]:
    np = _import_numpy()

    normalized_agent_kwargs = dict(agent_kwargs)
    normalized_params = dict(fitted_params)
    if not (
        len(choice_sessions) == len(reward_sessions) == len(choice_prob_sessions)
    ):
        raise ValueError(
            "manual fallback requires the same number of choice, reward, and "
            "probability sessions"
        )
    action_selection = str(
        normalized_agent_kwargs.get("action_selection", "softmax")
    ).lower()
    choice_kernel = _normalize_manual_q_choice_kernel(
        normalized_agent_kwargs.get("choice_kernel", "none")
    )
    number_of_learning_rate = _safe_int(
        normalized_agent_kwargs.get("number_of_learning_rate", 1),
        default=1,
    )
    number_of_forget_rate = _safe_int(
        normalized_agent_kwargs.get("number_of_forget_rate", 0),
        default=0,
    )

    if action_selection != "softmax":
        raise ValueError(
            "manual fallback supports ForagerQLearning only with "
            f"action_selection='softmax', received {action_selection!r}"
        )
    if choice_kernel not in {"none", "one_step"}:
        raise ValueError(
            "manual fallback supports ForagerQLearning only with "
            f"choice_kernel in {{'none', 'one_step'}}, received {choice_kernel!r}"
        )
    if number_of_learning_rate not in {1, 2}:
        raise ValueError(
            "manual fallback supports number_of_learning_rate in {1, 2}, "
            f"received {number_of_learning_rate!r}"
        )
    if number_of_forget_rate not in {0, 1}:
        raise ValueError(
            "manual fallback supports number_of_forget_rate in {0, 1}, "
            f"received {number_of_forget_rate!r}"
        )

    initial_q, initial_q_source = _resolve_manual_q_initial_values(
        normalized_agent_kwargs,
        normalized_params,
    )
    forget_rate, forget_rate_source = _resolve_manual_q_forget_rate(
        normalized_params,
        number_of_forget_rate=number_of_forget_rate,
    )

    q_value_sessions: list[Any] = []
    for choices, rewards in zip(choice_sessions, reward_sessions):
        q_value_sessions.append(
            _manual_forager_q_learning_session_q_values(
                choices=choices,
                rewards=rewards,
                fitted_params=normalized_params,
                number_of_learning_rate=number_of_learning_rate,
                number_of_forget_rate=number_of_forget_rate,
                forget_rate=forget_rate,
                initial_q=initial_q,
            )
        )

    validation = "manual_validation_skipped_choice_kernel"
    validation_details: dict[str, Any] = {
        "reason": "choice_kernel alters action probabilities beyond raw Q values"
    }
    if choice_kernel == "none":
        validation_details = _validate_manual_forager_q_learning_probabilities(
            q_value_sessions,
            choice_prob_sessions=choice_prob_sessions,
            fitted_params=normalized_params,
        )
        validation = "manual_validation_passed"

    return q_value_sessions, {
        "status": "recovered",
        "q_source": "manual_forager_q_learning",
        "alignment": "manual_policy_time",
        "agent_class": "ForagerQLearning",
        "action_selection": action_selection,
        "choice_kernel": choice_kernel,
        "number_of_learning_rate": int(number_of_learning_rate),
        "number_of_forget_rate": int(number_of_forget_rate),
        "initial_q": np.asarray(initial_q, dtype=float).tolist(),
        "initial_q_source": initial_q_source,
        "forget_rate_unchosen": float(forget_rate),
        "forget_rate_source": forget_rate_source,
        "validation": validation,
        "validation_details": validation_details,
    }


def _manual_forager_q_learning_session_q_values(
    *,
    choices: Any,
    rewards: Any,
    fitted_params: Mapping[str, Any],
    number_of_learning_rate: int,
    number_of_forget_rate: int,
    forget_rate: float,
    initial_q: Any,
) -> Any:
    np = _import_numpy()

    choice_array = np.asarray(choices, dtype=int)
    reward_array = np.asarray(rewards, dtype=float)
    if choice_array.ndim != 1 or reward_array.ndim != 1:
        raise ValueError("manual fallback expects 1D choice and reward sessions")
    if choice_array.shape[0] != reward_array.shape[0]:
        raise ValueError(
            "manual fallback choice/reward length mismatch: "
            f"choices={choice_array.shape[0]} rewards={reward_array.shape[0]}"
        )
    if not np.isin(choice_array, [0, 1]).all():
        raise ValueError("manual fallback supports only binary actions 0/1")

    q_values = np.asarray(initial_q, dtype=float).reshape(2).copy()
    q_history = np.zeros((2, int(choice_array.shape[0])), dtype=float)
    for trial_index, (choice, reward) in enumerate(zip(choice_array, reward_array)):
        q_history[:, trial_index] = q_values
        chosen_action = int(choice)
        unchosen_action = 1 - chosen_action
        learn_rate = _resolve_manual_q_learning_rate(
            fitted_params,
            reward=float(reward),
            number_of_learning_rate=int(number_of_learning_rate),
        )
        q_values[chosen_action] = q_values[chosen_action] + learn_rate * (
            float(reward) - q_values[chosen_action]
        )
        if int(number_of_forget_rate) == 1:
            q_values[unchosen_action] = q_values[unchosen_action] + float(
                forget_rate
            ) * (float(initial_q[unchosen_action]) - q_values[unchosen_action])
    return q_history


def _validate_manual_forager_q_learning_probabilities(
    q_value_sessions: Sequence[Any],
    *,
    choice_prob_sessions: Sequence[Any],
    fitted_params: Mapping[str, Any],
) -> dict[str, Any]:
    np = _import_numpy()

    max_abs_error = 0.0
    for q_session, choice_prob_session in zip(q_value_sessions, choice_prob_sessions):
        q_values = _align_policy_time_q_session(
            q_session,
            n_trials=int(np.asarray(q_session).shape[-1]),
        )[0]
        agent_probabilities = _align_binary_choice_prob_session(
            choice_prob_session,
            n_trials=int(q_values.shape[1]),
        )
        manual_probabilities = _manual_forager_q_learning_softmax_probabilities(
            q_values,
            fitted_params=fitted_params,
        )
        session_max_abs_error = float(
            np.max(np.abs(agent_probabilities - manual_probabilities))
        )
        max_abs_error = max(max_abs_error, session_max_abs_error)
        if not np.allclose(
            agent_probabilities,
            manual_probabilities,
            atol=_MANUAL_Q_PROBABILITY_VALIDATION_ATOL,
            rtol=_MANUAL_Q_PROBABILITY_VALIDATION_ATOL,
        ):
            raise ValueError(
                "manual ForagerQLearning Q fallback did not reproduce live-agent "
                "softmax probabilities for choice_kernel='none' "
                f"(max_abs_error={session_max_abs_error:.6g})"
            )
    return {
        "max_abs_probability_error": max_abs_error,
        "atol": _MANUAL_Q_PROBABILITY_VALIDATION_ATOL,
        "rtol": _MANUAL_Q_PROBABILITY_VALIDATION_ATOL,
    }


def _manual_forager_q_learning_softmax_probabilities(
    q_values: Any,
    *,
    fitted_params: Mapping[str, Any],
) -> Any:
    np = _import_numpy()

    q_array = np.asarray(q_values, dtype=float)
    if q_array.ndim != 2 or q_array.shape[0] != 2:
        raise ValueError(
            "manual softmax validation expects Q values with shape (2, n_trials), "
            f"received {q_array.shape}"
        )
    inverse_temperature = _optional_float_param(
        fitted_params,
        (
            "softmax_inverse_temperature",
            "inverse_temperature",
            "beta",
            "temperature_inverse",
        ),
        default=1.0,
    )
    bias_left = _optional_float_param(
        fitted_params,
        ("biasL", "bias_left", "left_bias", "bias"),
        default=0.0,
    )
    logits = np.vstack(
        [
            float(inverse_temperature) * q_array[0, :] + float(bias_left),
            float(inverse_temperature) * q_array[1, :],
        ]
    )
    stabilized = logits - np.max(logits, axis=0, keepdims=True)
    exp_values = np.exp(stabilized)
    denom = np.sum(exp_values, axis=0, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return exp_values / denom


def _resolve_manual_q_initial_values(
    agent_kwargs: Mapping[str, Any],
    fitted_params: Mapping[str, Any],
) -> tuple[Any, str]:
    np = _import_numpy()

    for source_name, source_mapping in (
        ("fitted_param", fitted_params),
        ("agent_kwarg", agent_kwargs),
    ):
        for key in ("initial_q", "initial_Q", "q_initial", "q0", "Q0"):
            if key not in source_mapping:
                continue
            raw_value = source_mapping[key]
            value = np.asarray(raw_value, dtype=float)
            if value.size == 1:
                return np.full(2, float(value.reshape(-1)[0]), dtype=float), (
                    f"{source_name}:{key}"
                )
            if value.size == 2:
                return value.reshape(2).astype(float), f"{source_name}:{key}"
            raise ValueError(
                "manual fallback supports scalar or two-action initial Q values, "
                f"received {key} with shape={value.shape}"
            )
    return np.full(2, 0.5, dtype=float), "default_neutral"


def _resolve_manual_q_learning_rate(
    fitted_params: Mapping[str, Any],
    *,
    reward: float,
    number_of_learning_rate: int,
) -> float:
    if int(number_of_learning_rate) == 1:
        return _required_float_param(
            fitted_params,
            ("learn_rate", "learning_rate", "alpha", "learn_rate_rew"),
            label="single learning rate",
        )
    if float(reward) > 0.0:
        return _required_float_param(
            fitted_params,
            ("learn_rate_rew", "learn_rate", "learning_rate", "alpha"),
            label="reward learning rate",
        )
    return _required_float_param(
        fitted_params,
        ("learn_rate_unrew", "learn_rate", "learning_rate", "alpha"),
        label="unrewarded learning rate",
    )


def _resolve_manual_q_forget_rate(
    fitted_params: Mapping[str, Any],
    *,
    number_of_forget_rate: int,
) -> tuple[float, str]:
    if int(number_of_forget_rate) == 0:
        return 0.0, "not_used"
    value, source = _required_float_param_with_source(
        fitted_params,
        (
            "forget_rate_unchosen",
            "forget_rate",
            "forget_rate_unselected",
            "forget_rate_unchoosen",
        ),
        label="unchosen forget rate",
    )
    return value, source


def _required_float_param(
    params: Mapping[str, Any],
    names: Sequence[str],
    *,
    label: str,
) -> float:
    value, _ = _required_float_param_with_source(params, names, label=label)
    return value


def _required_float_param_with_source(
    params: Mapping[str, Any],
    names: Sequence[str],
    *,
    label: str,
) -> tuple[float, str]:
    for name in names:
        if name not in params or params[name] is None:
            continue
        return float(params[name]), str(name)
    raise ValueError(
        f"manual ForagerQLearning fallback requires {label}; tried {list(names)!r}"
    )


def _optional_float_param(
    params: Mapping[str, Any],
    names: Sequence[str],
    *,
    default: float,
) -> float:
    for name in names:
        if name in params and params[name] is not None:
            return float(params[name])
    return float(default)


def _normalize_manual_q_choice_kernel(value: Any) -> str:
    if value is None:
        return "none"
    normalized = str(value).strip().lower()
    if normalized in {"", "none", "null", "false"}:
        return "none"
    return normalized


def _safe_int(value: Any, *, default: int) -> int:
    if value is None:
        return int(default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(float(value))


def _extract_policy_time_q_histories(
    eval_agent: Any,
    *,
    choice_prob_sessions: Sequence[Any],
    expected_trials_per_session: Sequence[int],
) -> tuple[list[Any], dict[str, Any]]:
    try:
        from utils.baseline_rl_evaluation import _extract_q_histories
    except Exception:  # pragma: no cover - import should be available in repo runtime
        _extract_q_histories = None

    if _extract_q_histories is not None:
        exact_histories = _extract_q_histories(
            eval_agent,
            [
                _import_numpy().asarray(choice_prob_session, dtype=float)
                for choice_prob_session in choice_prob_sessions
            ],
        )
        if exact_histories is not None:
            return list(exact_histories), {
                "status": "recovered",
                "alignment": "trial_aligned",
                "source": "utils.baseline_rl_evaluation._extract_q_histories",
            }

    recovered = _coerce_policy_time_q_history_candidate(
        eval_agent,
        expected_trials_per_session=[int(value) for value in expected_trials_per_session],
    )
    if recovered is not None:
        q_histories, alignment_modes = recovered
        return q_histories, {
            "status": "recovered",
            "alignment": _summarize_alignment_modes(alignment_modes),
            "session_alignments": alignment_modes,
            "source": "policy_time_fallback",
        }

    raise ValueError(
        "Could not recover policy-time baseline RL Q-value histories from the rollout "
        "agent. Q-space plotting requires q_rl_left/q_rl_right; verify the baseline "
        "agent exposes Q histories aligned to trial time or pre/post histories with "
        "n_trials + 1 entries."
    )


def _coerce_policy_time_q_history_candidate(
    candidate: Any,
    *,
    expected_trials_per_session: Sequence[int],
    depth: int = 0,
    seen: set[int] | None = None,
) -> tuple[list[Any], list[str]] | None:
    np = _import_numpy()

    if candidate is None or depth > 6:
        return None

    if seen is None:
        seen = set()

    if not isinstance(candidate, (str, bytes, int, float, bool, np.generic)):
        candidate_id = id(candidate)
        if candidate_id in seen:
            return None
        seen.add(candidate_id)

    expected_lengths = [int(value) for value in expected_trials_per_session]
    expected_n_sessions = len(expected_lengths)

    if isinstance(candidate, np.ndarray) or hasattr(candidate, "__array__"):
        arr = np.asarray(candidate, dtype=float)
        if arr.ndim == 2 and expected_n_sessions == 1:
            try:
                aligned_session, alignment_mode = _align_policy_time_q_session(
                    arr,
                    n_trials=expected_lengths[0],
                )
            except ValueError:
                return None
            return [aligned_session], [alignment_mode]
        if arr.ndim == 3:
            for axis in range(arr.ndim):
                if arr.shape[axis] != expected_n_sessions:
                    continue
                aligned_sessions: list[Any] = []
                alignment_modes: list[str] = []
                valid = True
                for session_index, expected_trials in enumerate(expected_lengths):
                    session_arr = np.take(arr, session_index, axis=axis)
                    try:
                        aligned_session, alignment_mode = _align_policy_time_q_session(
                            session_arr,
                            n_trials=expected_trials,
                        )
                    except ValueError:
                        valid = False
                        break
                    aligned_sessions.append(aligned_session)
                    alignment_modes.append(alignment_mode)
                if valid:
                    return aligned_sessions, alignment_modes
        return None

    if isinstance(candidate, (list, tuple)):
        if len(candidate) == expected_n_sessions:
            aligned_sessions = []
            alignment_modes = []
            valid = True
            for item, expected_trials in zip(candidate, expected_lengths):
                recovered = _coerce_policy_time_q_history_candidate(
                    item,
                    expected_trials_per_session=[expected_trials],
                    depth=depth + 1,
                    seen=seen,
                )
                if recovered is None:
                    valid = False
                    break
                item_sessions, item_modes = recovered
                if len(item_sessions) != 1:
                    valid = False
                    break
                aligned_sessions.append(item_sessions[0])
                alignment_modes.append(item_modes[0])
            if valid:
                return aligned_sessions, alignment_modes

        for item in candidate:
            recovered = _coerce_policy_time_q_history_candidate(
                item,
                expected_trials_per_session=expected_lengths,
                depth=depth + 1,
                seen=seen,
            )
            if recovered is not None:
                return recovered
        return None

    if isinstance(candidate, Mapping):
        preferred_keys = [
            "q_values",
            "q_value_history",
            "q_values_history",
            "q_history",
            "Q",
            "Qs",
            "history",
            "values",
        ]
        for key in preferred_keys:
            if key in candidate:
                recovered = _coerce_policy_time_q_history_candidate(
                    candidate[key],
                    expected_trials_per_session=expected_lengths,
                    depth=depth + 1,
                    seen=seen,
                )
                if recovered is not None:
                    return recovered

        for key, value in candidate.items():
            if isinstance(key, str) and any(
                token in key.lower() for token in ("q", "value", "history")
            ):
                recovered = _coerce_policy_time_q_history_candidate(
                    value,
                    expected_trials_per_session=expected_lengths,
                    depth=depth + 1,
                    seen=seen,
                )
                if recovered is not None:
                    return recovered

        for value in candidate.values():
            recovered = _coerce_policy_time_q_history_candidate(
                value,
                expected_trials_per_session=expected_lengths,
                depth=depth + 1,
                seen=seen,
            )
            if recovered is not None:
                return recovered
        return None

    if hasattr(candidate, "__dict__"):
        attrs = vars(candidate)
        preferred_attr_names = [
            "q_value_history",
            "q_values_history",
            "Q_history",
            "Q_values_history",
            "Qs_history",
            "q_values",
            "q_history",
            "session_q_values",
            "q_values_all_sessions",
        ]
        for attr_name in preferred_attr_names:
            if attr_name in attrs:
                recovered = _coerce_policy_time_q_history_candidate(
                    attrs[attr_name],
                    expected_trials_per_session=expected_lengths,
                    depth=depth + 1,
                    seen=seen,
                )
                if recovered is not None:
                    return recovered

        for attr_name, value in attrs.items():
            if any(token in attr_name.lower() for token in ("q", "value", "history")):
                recovered = _coerce_policy_time_q_history_candidate(
                    value,
                    expected_trials_per_session=expected_lengths,
                    depth=depth + 1,
                    seen=seen,
                )
                if recovered is not None:
                    return recovered

        for value in attrs.values():
            recovered = _coerce_policy_time_q_history_candidate(
                value,
                expected_trials_per_session=expected_lengths,
                depth=depth + 1,
                seen=seen,
            )
            if recovered is not None:
                return recovered

    return None


def _align_policy_time_q_session(q_session: Any, *, n_trials: int) -> tuple[Any, str]:
    np = _import_numpy()

    arr = np.asarray(q_session, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"Q session must be 2D, received shape={arr.shape}")

    if arr.shape[0] == 2 and arr.shape[1] in {int(n_trials), int(n_trials) + 1}:
        alignment = (
            "trial_aligned"
            if arr.shape[1] == int(n_trials)
            else "prepost_first_n"
        )
        return arr[:2, : int(n_trials)], alignment
    if arr.shape[1] == 2 and arr.shape[0] in {int(n_trials), int(n_trials) + 1}:
        alignment = (
            "trial_aligned"
            if arr.shape[0] == int(n_trials)
            else "prepost_first_n"
        )
        return arr[: int(n_trials), :2].T, alignment
    if arr.shape[1] in {int(n_trials), int(n_trials) + 1} and arr.shape[0] >= 2:
        alignment = (
            "trial_aligned"
            if arr.shape[1] == int(n_trials)
            else "prepost_first_n"
        )
        return arr[:2, : int(n_trials)], alignment
    if arr.shape[0] in {int(n_trials), int(n_trials) + 1} and arr.shape[1] >= 2:
        alignment = (
            "trial_aligned"
            if arr.shape[0] == int(n_trials)
            else "prepost_first_n"
        )
        return arr[: int(n_trials), :2].T, alignment

    raise ValueError(
        "Unable to align baseline RL Q values with policy-time trials: "
        f"q_shape={arr.shape}, n_trials={n_trials}"
    )


def _summarize_alignment_modes(alignment_modes: Sequence[str]) -> str:
    unique_modes = list(dict.fromkeys(str(mode) for mode in alignment_modes))
    if not unique_modes:
        return "unknown"
    if len(unique_modes) == 1:
        return unique_modes[0]
    return "mixed"


def _summarize_q_source(q_alignment: Mapping[str, Any] | None) -> str:
    if not isinstance(q_alignment, Mapping) or not q_alignment:
        return "unknown"
    q_source = q_alignment.get("q_source")
    if q_source is not None:
        return str(q_source)
    nested_sources = [
        str(value.get("q_source"))
        for value in q_alignment.values()
        if isinstance(value, Mapping) and value.get("q_source") is not None
    ]
    unique_sources = list(dict.fromkeys(nested_sources))
    if not unique_sources:
        return "unknown"
    if len(unique_sources) == 1:
        return unique_sources[0]
    return "mixed"


def _session_rollout_payloads_from_trial_df(trial_df: Any) -> list[dict[str, Any]]:
    pd = _import_pandas()
    np = _import_numpy()

    sorted_df = _sort_trial_dataframe(pd.DataFrame(trial_df).copy())
    payloads: list[dict[str, Any]] = []
    ordered_session_ids = list(dict.fromkeys(sorted_df["ses_idx"].tolist()))
    for session_id in ordered_session_ids:
        session_df = sorted_df[sorted_df["ses_idx"] == session_id].copy()
        if session_df.empty:
            continue
        payloads.append(
            {
                "subject_id": lc._normalize_identifier(session_df["subject_id"].iloc[0]),
                "ses_idx": str(session_id),
                "choices": session_df["action"].to_numpy(dtype=int),
                "rewards": session_df["reward"].to_numpy(dtype=float),
                "trial_indices": session_df["trial_idx"].to_numpy(dtype=int),
            }
        )
    return payloads


def _probability_frame_from_rollout_sessions(
    session_payloads: Sequence[Mapping[str, Any]],
    *,
    choice_prob_sessions: Sequence[Any],
    q_value_sessions: Sequence[Any] | None = None,
) -> Any:
    pd = _import_pandas()

    records: list[dict[str, Any]] = []
    q_sessions = (
        [None] * len(session_payloads)
        if q_value_sessions is None
        else list(q_value_sessions)
    )
    for payload, choice_prob_session, q_value_session in zip(
        session_payloads,
        choice_prob_sessions,
        q_sessions,
    ):
        choices = payload["choices"]
        aligned_probabilities = _align_binary_choice_prob_session(
            choice_prob_session,
            n_trials=int(len(choices)),
        )
        if aligned_probabilities.shape[1] != len(choices):
            raise ValueError(
                "Baseline choice probabilities did not align exactly to the requested "
                f"trial count for session {payload['ses_idx']!r}: "
                f"expected={len(choices)} received={aligned_probabilities.shape[1]}"
            )
        aligned_q_values = None
        if q_value_session is not None:
            aligned_q_values, _ = _align_policy_time_q_session(
                q_value_session,
                n_trials=int(len(choices)),
            )
        for trial_idx, action in enumerate(choices.tolist(), start=1):
            record = {
                "subject_id": payload["subject_id"],
                "ses_idx": payload["ses_idx"],
                "trial_idx": int(trial_idx),
                "action": int(action),
                "p_rl": float(aligned_probabilities[int(action), trial_idx - 1]),
                "p_rl_left": float(aligned_probabilities[0, trial_idx - 1]),
                "p_rl_right": float(aligned_probabilities[1, trial_idx - 1]),
            }
            if aligned_q_values is not None:
                record["q_rl_left"] = float(aligned_q_values[0, trial_idx - 1])
                record["q_rl_right"] = float(aligned_q_values[1, trial_idx - 1])
            records.append(record)
    return pd.DataFrame.from_records(
        records,
        columns=(
            _BASELINE_PROBABILITY_FRAME_COLUMNS
            if q_value_sessions is not None
            else [
                "subject_id",
                "ses_idx",
                "trial_idx",
                "action",
                "p_rl",
                "p_rl_left",
                "p_rl_right",
            ]
        ),
    )


def _align_binary_choice_prob_session(choice_prob_session: Any, *, n_trials: int) -> Any:
    np = _import_numpy()

    probabilities = np.asarray(choice_prob_session, dtype=float)
    if probabilities.ndim != 2:
        raise ValueError(
            f"choice_prob_session must be 2D, received shape={probabilities.shape}"
        )
    if probabilities.shape[0] == 2:
        aligned = probabilities
    elif probabilities.shape[1] == 2:
        aligned = probabilities.T
    else:
        raise ValueError(
            "choice_prob_session must have one axis of length 2 for binary choices, "
            f"received shape={probabilities.shape}"
        )

    if aligned.shape[1] != int(n_trials):
        raise ValueError(
            "choice_prob_session length mismatch: "
            f"expected n_trials={n_trials}, received shape={aligned.shape}"
        )
    return aligned


def _aligned_action_probabilities_from_output_df(
    session_df: Any,
    *,
    n_action_logits: int,
) -> Any:
    np = _import_numpy()
    pd = _import_pandas()

    if n_action_logits <= 0:
        raise ValueError(f"Expected n_action_logits > 0, got {n_action_logits}")

    normalized_session_df = pd.DataFrame(session_df).copy()
    probability_columns = [f"choice_prob_{index}" for index in range(n_action_logits)]
    if all(column in normalized_session_df.columns for column in probability_columns):
        probabilities = normalized_session_df[probability_columns].to_numpy(dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != n_action_logits:
            raise ValueError(
                "Aligned probability columns have unexpected shape: "
                f"{probabilities.shape}, expected (_, {n_action_logits})"
            )
        return probabilities

    disrnn_logit_columns = ["logit(left)", "logit(right)", "logit(ignore)"][:n_action_logits]
    if all(column in normalized_session_df.columns for column in disrnn_logit_columns):
        return _probs_from_logits_2d(
            normalized_session_df[disrnn_logit_columns].to_numpy(dtype=float)
        )

    gru_logit_columns = [f"choice_logit_{index}" for index in range(n_action_logits)]
    if all(column in normalized_session_df.columns for column in gru_logit_columns):
        return _probs_from_logits_2d(
            normalized_session_df[gru_logit_columns].to_numpy(dtype=float)
        )

    available_columns = ", ".join(str(column) for column in normalized_session_df.columns)
    raise ValueError(
        "Could not find aligned action probability/logit columns. "
        f"Expected one of {probability_columns}, {disrnn_logit_columns}, or "
        f"{gru_logit_columns}. Available columns: {available_columns}"
    )


def _probs_from_logits_2d(logits: Any) -> Any:
    np = _import_numpy()

    logits_array = np.asarray(logits, dtype=float)
    if logits_array.ndim != 2:
        raise ValueError(f"logits must be 2D, received shape={logits_array.shape}")
    stabilized = logits_array - np.max(logits_array, axis=1, keepdims=True)
    exp_values = np.exp(stabilized)
    denom = np.sum(exp_values, axis=1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return exp_values / denom


def _merge_probability_frame(
    trial_df: Any,
    probability_df: Any,
    *,
    prob_column: str,
    source_label: str,
) -> Any:
    pd = _import_pandas()

    left_df = _sort_trial_dataframe(pd.DataFrame(trial_df).copy())
    right_df = pd.DataFrame(probability_df).copy()
    key_columns = ["ses_idx", "trial_idx"]
    expected_columns = {*key_columns, "action", prob_column}
    missing_columns = [column for column in expected_columns if column not in right_df.columns]
    if missing_columns:
        raise ValueError(
            f"Probability frame for {source_label!r} is missing columns: {missing_columns}"
        )

    duplicated = right_df.duplicated(subset=key_columns)
    if duplicated.any():
        raise ValueError(
            f"Probability frame for {source_label!r} contains duplicate ses_idx/trial_idx keys."
        )

    excluded_columns = {"subject_id", *key_columns, "action", prob_column}
    extra_probability_columns = [
        str(column)
        for column in right_df.columns
        if column not in excluded_columns and column not in left_df.columns
    ]
    merge_columns = [*key_columns, "action", prob_column, *extra_probability_columns]
    merged = left_df.merge(
        right_df.loc[:, merge_columns],
        on=key_columns,
        how="left",
        validate="one_to_one",
        suffixes=("", "_prob"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        missing_rows = merged.loc[merged["_merge"] != "both", ["ses_idx", "trial_idx"]]
        raise ValueError(
            f"Probability frame for {source_label!r} does not match the canonical "
            f"trial set. Missing keys: {missing_rows.head(10).to_dict(orient='records')}"
        )
    if "action_prob" in merged.columns:
        mismatch_mask = merged["action"].astype(int) != merged["action_prob"].astype(int)
        if mismatch_mask.any():
            mismatch_rows = merged.loc[
                mismatch_mask,
                ["ses_idx", "trial_idx", "action", "action_prob"],
            ]
            raise ValueError(
                f"Action mismatch while merging probabilities for {source_label!r}: "
                f"{mismatch_rows.head(10).to_dict(orient='records')}"
            )
        merged = merged.drop(columns=["action_prob"])

    merged = merged.drop(columns=["_merge"])
    value_columns = [prob_column, *extra_probability_columns]
    missing_value_mask = merged.loc[:, value_columns].isna().any(axis=1)
    if missing_value_mask.any():
        missing_rows = merged.loc[missing_value_mask, ["ses_idx", "trial_idx"]]
        raise ValueError(
            f"Merged probability columns {value_columns!r} contain NaNs for "
            f"{source_label!r}: "
            f"{missing_rows.head(10).to_dict(orient='records')}"
        )
    return _sort_trial_dataframe(merged)


def _merge_rnn_state_frame(
    trial_df: Any,
    state_df: Any,
    *,
    source_label: str,
) -> Any:
    pd = _import_pandas()

    left_df = _sort_trial_dataframe(pd.DataFrame(trial_df).copy())
    right_df = pd.DataFrame(state_df).copy()
    state_columns = _rnn_state_columns(right_df)
    expected_columns = {"ses_idx", "trial_idx", "action", *state_columns}
    missing_columns = [column for column in expected_columns if column not in right_df.columns]
    if not state_columns:
        raise ValueError(f"RNN state frame for {source_label!r} has no rnn_state_* columns.")
    if missing_columns:
        raise ValueError(
            f"RNN state frame for {source_label!r} is missing columns: {missing_columns}"
        )

    duplicated = right_df.duplicated(subset=["ses_idx", "trial_idx"])
    if duplicated.any():
        raise ValueError(
            f"RNN state frame for {source_label!r} contains duplicate ses_idx/trial_idx keys."
        )

    merged = left_df.merge(
        right_df.loc[:, ["ses_idx", "trial_idx", "action", *state_columns]],
        on=["ses_idx", "trial_idx"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_state"),
        indicator=True,
    )
    if not (merged["_merge"] == "both").all():
        missing_rows = merged.loc[merged["_merge"] != "both", ["ses_idx", "trial_idx"]]
        raise ValueError(
            f"RNN state frame for {source_label!r} does not match the canonical "
            f"trial set. Missing keys: {missing_rows.head(10).to_dict(orient='records')}"
        )
    if "action_state" in merged.columns:
        mismatch_mask = merged["action"].astype(int) != merged["action_state"].astype(int)
        if mismatch_mask.any():
            mismatch_rows = merged.loc[
                mismatch_mask,
                ["ses_idx", "trial_idx", "action", "action_state"],
            ]
            raise ValueError(
                f"Action mismatch while merging RNN states for {source_label!r}: "
                f"{mismatch_rows.head(10).to_dict(orient='records')}"
            )
        merged = merged.drop(columns=["action_state"])

    merged = merged.drop(columns=["_merge"])
    missing_state_mask = merged[state_columns].isna().any(axis=1)
    if missing_state_mask.any():
        missing_rows = merged.loc[missing_state_mask, ["ses_idx", "trial_idx"]]
        raise ValueError(
            f"Merged RNN states contain NaNs for {source_label!r}: "
            f"{missing_rows.head(10).to_dict(orient='records')}"
        )
    return _sort_trial_dataframe(merged)


def _add_session_stage(trial_df: Any) -> Any:
    pd = _import_pandas()

    enriched_df = _sort_trial_dataframe(pd.DataFrame(trial_df).copy())
    session_rows = enriched_df[["subject_id", "ses_idx", "session_idx"]].drop_duplicates()
    session_rows["n_sessions"] = session_rows.groupby("subject_id", sort=False)[
        "ses_idx"
    ].transform("size")
    session_rows["session_stage"] = [
        _session_stage_from_progress(
            (float(session_idx) - 1.0) / float(max(int(n_sessions) - 1, 1))
        )
        for session_idx, n_sessions in zip(
            session_rows["session_idx"].tolist(),
            session_rows["n_sessions"].tolist(),
        )
    ]
    session_rows["session_stage"] = pd.Categorical(
        session_rows["session_stage"],
        categories=list(_SESSION_STAGE_ORDER),
        ordered=True,
    )
    stage_map = {
        str(row["ses_idx"]): row["session_stage"]
        for row in session_rows.to_dict(orient="records")
    }
    enriched_df["session_stage"] = pd.Categorical(
        [stage_map[str(session_id)] for session_id in enriched_df["ses_idx"].tolist()],
        categories=list(_SESSION_STAGE_ORDER),
        ordered=True,
    )
    return enriched_df


def _session_stage_from_progress(progress: float) -> str:
    if progress < (1.0 / 3.0):
        return "early"
    if progress < (2.0 / 3.0):
        return "mid"
    return "late"


def _build_subject_mean_advantage_df(
    trial_df: Any,
    *,
    jitter_seed: int,
) -> Any:
    pd = _import_pandas()
    np = _import_numpy()

    summary_df = (
        pd.DataFrame(trial_df)
        .groupby("subject_id", sort=False)
        .agg(
            mean_advantage=("advantage", "mean"),
            curriculum_name=("curriculum_name", _resolve_subject_curriculum),
        )
        .reset_index()
    )
    rng = np.random.default_rng(int(jitter_seed))
    summary_df["x_jitter"] = rng.uniform(-0.18, 0.18, size=len(summary_df))
    palette = lc._build_curriculum_palette(summary_df)
    summary_df["color"] = [
        palette.get(str(curriculum_name), lc._DEFAULT_CURRICULUM_COLORS["Unknown"])
        for curriculum_name in summary_df["curriculum_name"].tolist()
    ]
    return summary_df


def _resolve_subject_curriculum(values: Any) -> str:
    unique_values = [
        str(value)
        for value in values.tolist()
        if value is not None and str(value).strip()
    ]
    if not unique_values:
        return "Unknown"
    return lc._resolve_comparison_curriculum_name(*unique_values)


def _fit_and_project_rnn_state_pca(
    trial_df: Any,
    *,
    state_columns: Sequence[str],
    pca_seed: int,
    pca_fit_fraction: float,
    finite_value_columns: Sequence[str] = ("advantage",),
) -> dict[str, Any]:
    np = _import_numpy()
    pd = _import_pandas()

    if not 0.0 < float(pca_fit_fraction) <= 1.0:
        raise ValueError("pca_fit_fraction must be in the interval (0, 1].")

    df = pd.DataFrame(trial_df).copy().reset_index(drop=True)
    state_matrix = df.loc[:, list(state_columns)].to_numpy(dtype=float)
    finite_value_column_names = [str(column) for column in finite_value_columns]
    missing_value_columns = [
        column for column in finite_value_column_names if column not in df.columns
    ]
    if missing_value_columns:
        raise ValueError(
            "trial_df is missing finite value columns for PCA: "
            f"{missing_value_columns}"
        )
    advantages = (
        df["advantage"].to_numpy(dtype=float)
        if "advantage" in df.columns
        else np.full(len(df), np.nan, dtype=float)
    )
    finite_mask = np.isfinite(state_matrix).all(axis=1)
    for column in finite_value_column_names:
        finite_mask = finite_mask & np.isfinite(df[column].to_numpy(dtype=float))
    n_valid = int(np.sum(finite_mask))
    if n_valid < 2:
        finite_label = (
            ", ".join(finite_value_column_names)
            if finite_value_column_names
            else "state values"
        )
        raise ValueError(
            f"At least two finite RNN-state trials with finite {finite_label} are required "
            "to fit PCA."
        )

    valid_states = state_matrix[finite_mask]
    fit_n = int(math.floor(float(pca_fit_fraction) * n_valid))
    fit_n = min(n_valid, max(2, fit_n))
    rng = np.random.default_rng(int(pca_seed))
    fit_indices = np.sort(rng.choice(np.arange(n_valid), size=fit_n, replace=False))
    fit_states = valid_states[fit_indices]
    state_mean = np.mean(fit_states, axis=0)
    centered_fit = fit_states - state_mean
    _, singular_values, vt = np.linalg.svd(centered_fit, full_matrices=False)
    components = vt
    explained_variance = (singular_values ** 2) / float(max(fit_n - 1, 1))
    total_variance = float(np.sum(np.var(fit_states, axis=0, ddof=1)))
    if total_variance > 0:
        explained_variance_ratio = explained_variance / total_variance
    else:
        explained_variance_ratio = np.zeros_like(explained_variance)

    scores = np.full((len(df), components.shape[0]), np.nan, dtype=float)
    scores[finite_mask, :] = (valid_states - state_mean) @ components.T
    return {
        "trial_df": df,
        "state_columns": list(state_columns),
        "finite_value_columns": finite_value_column_names,
        "finite_mask": finite_mask,
        "scores": scores,
        "advantages": advantages,
        "components": components,
        "state_mean": state_mean,
        "explained_variance": explained_variance,
        "explained_variance_ratio": explained_variance_ratio,
        "cumulative_explained_variance_ratio": np.cumsum(explained_variance_ratio),
        "fit_n_trials": fit_n,
        "n_trials_projected": n_valid,
        "fit_indices": fit_indices,
    }


def _build_pca_variance_dataframe(
    pca_result: Mapping[str, Any],
    *,
    n_variance_pcs: int,
) -> Any:
    pd = _import_pandas()

    if int(n_variance_pcs) <= 0:
        raise ValueError("n_variance_pcs must be > 0.")
    explained_variance = pca_result["explained_variance"]
    explained_variance_ratio = pca_result["explained_variance_ratio"]
    cumulative = pca_result["cumulative_explained_variance_ratio"]
    n_rows = min(int(n_variance_pcs), int(len(explained_variance_ratio)))
    return pd.DataFrame.from_records(
        [
            {
                "pc": int(pc_idx + 1),
                "explained_variance": float(explained_variance[pc_idx]),
                "explained_variance_ratio": float(explained_variance_ratio[pc_idx]),
                "cumulative_explained_variance_ratio": float(cumulative[pc_idx]),
            }
            for pc_idx in range(n_rows)
        ]
    )


def _resolve_state_condition_columns(
    trial_df: Any,
    *,
    condition_columns: Sequence[str] | None,
) -> list[str]:
    pd = _import_pandas()

    df = pd.DataFrame(trial_df)
    selected_columns = (
        [spec.name for spec in _VARIABLE_SPECS if spec.name in df.columns]
        if condition_columns is None
        else [str(column) for column in condition_columns]
    )
    missing_columns = [column for column in selected_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"trial_advantage dataframe is missing condition columns: {missing_columns}"
        )
    return selected_columns


def _build_state_condition_specs(
    trial_df: Any,
    *,
    condition_columns: Sequence[str],
    condition_values_by_column: Mapping[str, Sequence[Any]],
) -> dict[str, list[dict[str, Any]]]:
    pd = _import_pandas()

    df = pd.DataFrame(trial_df).copy().reset_index(drop=True)
    condition_specs: dict[str, list[dict[str, Any]]] = {}
    for condition_column in condition_columns:
        series = df[condition_column]
        explicit_values = condition_values_by_column.get(condition_column)
        entries: list[dict[str, Any]] = []
        if explicit_values is not None:
            for value in explicit_values:
                mask = (series == value).fillna(False).to_numpy(dtype=bool)
                label = _format_condition_label(value)
                entries.append(
                    {
                        "label": label,
                        "slug": _safe_filename_component(label),
                        "mask": mask,
                    }
                )
        elif _condition_column_is_continuous(series, condition_column=condition_column):
            binned = _quantile_bin_series(
                series,
                q=int(_VARIABLE_SPEC_BY_NAME.get(
                    condition_column,
                    VariableSpec(condition_column, condition_column, "continuous"),
                ).n_bins),
            )
            for category in list(binned.cat.categories):
                label = str(category)
                mask = (binned == category).fillna(False).to_numpy(dtype=bool)
                entries.append(
                    {
                        "label": label,
                        "slug": _safe_filename_component(label),
                        "mask": mask,
                    }
                )
        else:
            spec = _VARIABLE_SPEC_BY_NAME.get(
                condition_column,
                VariableSpec(condition_column, condition_column, "categorical"),
            )
            categories = _resolve_categorical_categories(series, spec=spec)
            for category in categories:
                label = _format_condition_label(category)
                mask = (series == category).fillna(False).to_numpy(dtype=bool)
                entries.append(
                    {
                        "label": label,
                        "slug": _safe_filename_component(label),
                        "mask": mask,
                    }
                )
        condition_specs[condition_column] = entries
    return condition_specs


def _condition_column_is_continuous(series: Any, *, condition_column: str) -> bool:
    pd = _import_pandas()

    spec = _VARIABLE_SPEC_BY_NAME.get(condition_column)
    if spec is not None:
        return spec.kind == "continuous"
    values = pd.Series(series).dropna()
    if values.empty:
        return False
    return bool(pd.api.types.is_numeric_dtype(values) and values.nunique() > 10)


def _format_condition_label(value: Any) -> str:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(numeric_value) and numeric_value.is_integer():
        return str(int(numeric_value))
    return str(value)


def _plot_advantage_histogram(
    trial_df: Any,
    *,
    output_path: Path,
    run_model1: lc.ResolvedLikelihoodRun,
    run_model2: lc.ResolvedLikelihoodRun,
    split_name: str,
) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)
    ax.hist(
        _import_numpy().asarray(trial_df["advantage"], dtype=float),
        bins=40,
        color="#4e79a7",
        alpha=0.85,
        edgecolor="white",
    )
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
    ax.set_xlabel("Log-likelihood advantage")
    ax.set_ylabel("Trial count")
    ax.set_title(
        f"{run_model1.model_type.upper()} vs baseline RL advantage\n"
        f"Split={split_name} | {run_model1.checkpoint_label} vs {run_model2.checkpoint_label}"
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_subject_mean_advantage_scatter(
    subject_summary_df: Any,
    *,
    output_path: Path,
    split_name: str,
) -> None:
    plt = _import_pyplot()

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=120)
    if subject_summary_df.empty:
        ax.text(0.5, 0.5, "No subject data", ha="center", va="center", transform=ax.transAxes)
    else:
        for curriculum_name, curriculum_df in subject_summary_df.groupby(
            "curriculum_name", sort=False
        ):
            ax.scatter(
                curriculum_df["x_jitter"].to_numpy(dtype=float),
                curriculum_df["mean_advantage"].to_numpy(dtype=float),
                color=curriculum_df["color"].iloc[0],
                label=str(curriculum_name),
                s=55,
                alpha=0.9,
                edgecolors="black",
                linewidths=0.4,
            )
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1.3)
    ax.set_xlim(-0.25, 0.25)
    ax.set_xticks([])
    ax.set_ylabel("Mean advantage by subject")
    ax.set_title(f"Subject mean advantage by curriculum\nSplit={split_name}")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Curriculum", loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_rnn_state_pca_variance(
    variance_df: Any,
    *,
    output_path: Path,
) -> None:
    plt = _import_pyplot()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    if variance_df.empty:
        ax.text(0.5, 0.5, "No PCA variance data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        pc_labels = [f"PC{int(pc)}" for pc in variance_df["pc"].tolist()]
        x_positions = list(range(len(pc_labels)))
        ax.bar(
            x_positions,
            variance_df["explained_variance_ratio"].to_numpy(dtype=float),
            color="#4e79a7",
            alpha=0.9,
            label="Explained",
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(pc_labels)
        ax.set_ylabel("Explained variance ratio")
        ax.set_ylim(
            0.0,
            max(
                1.0,
                float(variance_df["cumulative_explained_variance_ratio"].max()) * 1.05,
            ),
        )

        ax_cumulative = ax.twinx()
        ax_cumulative.plot(
            x_positions,
            variance_df["cumulative_explained_variance_ratio"].to_numpy(dtype=float),
            color="#e15759",
            marker="o",
            linewidth=2.0,
            label="Cumulative",
        )
        ax_cumulative.set_ylabel("Cumulative explained variance ratio")
        ax_cumulative.set_ylim(ax.get_ylim())

        handles, labels = ax.get_legend_handles_labels()
        cumulative_handles, cumulative_labels = ax_cumulative.get_legend_handles_labels()
        ax.legend(
            handles + cumulative_handles,
            labels + cumulative_labels,
            loc="best",
            frameon=False,
        )
        ax.set_title("RNN State PCA Variance")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_rnn_state_condition_figure(
    pca_result: Mapping[str, Any],
    *,
    condition_column: str,
    condition_label: str,
    condition_mask: Any,
    output_path: Path,
    n_plot_pcs: int,
) -> None:
    plt = _import_pyplot()
    np = _import_numpy()

    if int(n_plot_pcs) < 2:
        raise ValueError("n_plot_pcs must be >= 2.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = np.asarray(pca_result["scores"], dtype=float)
    finite_mask = np.asarray(pca_result["finite_mask"], dtype=bool)
    advantages = np.asarray(pca_result["advantages"], dtype=float)
    condition_mask = np.asarray(condition_mask, dtype=bool)
    highlight_mask = finite_mask & condition_mask & np.isfinite(advantages)
    highlight_count = int(np.sum(highlight_mask))
    background_scores = scores[finite_mask]
    highlight_scores = scores[highlight_mask]
    highlight_advantages = advantages[highlight_mask]
    highlight_mean_advantage = (
        float(np.mean(highlight_advantages))
        if highlight_advantages.size
        else math.nan
    )
    highlight_sem_advantage = (
        float(np.std(highlight_advantages, ddof=1) / math.sqrt(highlight_advantages.size))
        if highlight_advantages.size > 1
        else 0.0 if highlight_advantages.size == 1 else math.nan
    )
    finite_advantages = advantages[finite_mask]
    max_abs_advantage = (
        float(np.nanmax(np.abs(finite_advantages)))
        if finite_advantages.size
        else 1.0
    )
    if not np.isfinite(max_abs_advantage) or max_abs_advantage <= 0.0:
        max_abs_advantage = 1.0

    pc_pairs = list(combinations(range(int(n_plot_pcs)), 2))
    n_panels = max(1, len(pc_pairs))
    n_cols = min(3, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.6 * n_rows),
        dpi=120,
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    last_scatter = None
    for axis, (pc_x, pc_y) in zip(axes_flat, pc_pairs, strict=False):
        if scores.shape[1] <= max(pc_x, pc_y):
            axis.text(
                0.5,
                0.5,
                "Not enough PCs",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            continue

        axis.scatter(
            background_scores[:, pc_x],
            background_scores[:, pc_y],
            color="#8c8c8c",
            alpha=0.06,
            s=8,
            linewidths=0,
        )
        if highlight_count > 0:
            last_scatter = axis.scatter(
                highlight_scores[:, pc_x],
                highlight_scores[:, pc_y],
                c=highlight_advantages,
                cmap="coolwarm",
                vmin=-max_abs_advantage,
                vmax=max_abs_advantage,
                s=8,
                alpha=0.9,
                linewidths=0,
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No matching trials",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.axhline(0.0, color="#d0d0d0", linewidth=0.7, zorder=0)
        axis.axvline(0.0, color="#d0d0d0", linewidth=0.7, zorder=0)
        axis.set_xlabel(f"PC{pc_x + 1}")
        axis.set_ylabel(f"PC{pc_y + 1}")
        axis.set_title(f"PC{pc_x + 1} vs PC{pc_y + 1}")

    for axis in axes_flat[len(pc_pairs) :]:
        axis.axis("off")

    if math.isfinite(highlight_mean_advantage) and math.isfinite(highlight_sem_advantage):
        advantage_summary = (
            f"advantage={highlight_mean_advantage:.4g} +/- "
            f"{highlight_sem_advantage:.4g}"
        )
    else:
        advantage_summary = "advantage=nan +/- nan"
    fig.suptitle(
        f"{condition_column} = {condition_label} "
        f"(n={highlight_count}, {advantage_summary})"
    )
    fig.tight_layout(rect=(0, 0, 0.88, 0.94))
    if last_scatter is not None:
        colorbar_axis = fig.add_axes((0.91, 0.14, 0.018, 0.74))
        colorbar = fig.colorbar(last_scatter, cax=colorbar_axis)
        colorbar.set_label("Log-likelihood advantage")
    fig.savefig(output_path)
    plt.close(fig)


def _validate_subject_state_space_dataframe(
    trial_df: Any,
    *,
    probability_column: str,
) -> None:
    pd = _import_pandas()

    df = pd.DataFrame(trial_df)
    required_columns = ["subject_id", str(probability_column)]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "trial_advantage_pickle is missing required columns for subject "
            f"state-space analysis: {missing_columns}"
        )
    if not _rnn_state_columns(df):
        raise ValueError(
            "trial_advantage_pickle must contain rnn_state_* columns before "
            "subject state-space plotting can run."
        )


def _resolve_subject_ids_for_state_space_plot(
    trial_df: Any,
    *,
    subject_ids: Sequence[Any] | None,
) -> list[str]:
    pd = _import_pandas()

    df = pd.DataFrame(trial_df)
    if "subject_id" not in df.columns:
        raise ValueError("trial_advantage_pickle must contain a subject_id column.")

    subject_series = df["subject_id"]
    available_subject_ids = list(
        dict.fromkeys(subject_series[subject_series.notna()].map(str).tolist())
    )
    if subject_ids is None:
        return available_subject_ids

    requested_subject_ids = (
        [str(subject_ids)]
        if isinstance(subject_ids, (str, bytes))
        else [str(subject_id) for subject_id in subject_ids]
    )
    missing_subject_ids = [
        subject_id
        for subject_id in requested_subject_ids
        if subject_id not in set(available_subject_ids)
    ]
    if missing_subject_ids:
        raise ValueError(
            "Requested subject_ids are not present in trial_advantage_pickle: "
            f"{missing_subject_ids}"
        )
    return requested_subject_ids


def _plot_rnn_state_subject_probability_figure(
    pca_result: Mapping[str, Any],
    *,
    subject_id: str,
    subject_mask: Any,
    probability_column: str,
    output_path: Path,
    n_plot_pcs: int,
) -> None:
    plt = _import_pyplot()
    np = _import_numpy()

    if int(n_plot_pcs) < 2:
        raise ValueError("n_plot_pcs must be >= 2.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pca_result["trial_df"]
    if probability_column not in df.columns:
        raise ValueError(
            f"trial_advantage_pickle must contain probability_column={probability_column!r}."
        )

    scores = np.asarray(pca_result["scores"], dtype=float)
    finite_mask = np.asarray(pca_result["finite_mask"], dtype=bool)
    probability_values = df[probability_column].to_numpy(dtype=float)
    subject_mask = np.asarray(subject_mask, dtype=bool)
    if subject_mask.shape[0] != finite_mask.shape[0]:
        raise ValueError("subject_mask must be row-aligned to the PCA trial dataframe.")

    highlight_mask = finite_mask & subject_mask & np.isfinite(probability_values)
    highlight_count = int(np.sum(highlight_mask))
    background_scores = scores[finite_mask]
    highlight_scores = scores[highlight_mask]
    highlight_probabilities = probability_values[highlight_mask]
    highlight_mean_probability = (
        float(np.mean(highlight_probabilities))
        if highlight_probabilities.size
        else math.nan
    )
    highlight_sem_probability = (
        float(
            np.std(highlight_probabilities, ddof=1)
            / math.sqrt(highlight_probabilities.size)
        )
        if highlight_probabilities.size > 1
        else 0.0 if highlight_probabilities.size == 1 else math.nan
    )

    pc_pairs = list(combinations(range(int(n_plot_pcs)), 2))
    n_panels = max(1, len(pc_pairs))
    n_cols = min(3, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.2 * n_cols, 4.6 * n_rows),
        dpi=120,
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    last_scatter = None
    for axis, (pc_x, pc_y) in zip(axes_flat, pc_pairs, strict=False):
        if scores.shape[1] <= max(pc_x, pc_y):
            axis.text(
                0.5,
                0.5,
                "Not enough PCs",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            axis.set_axis_off()
            continue

        axis.scatter(
            background_scores[:, pc_x],
            background_scores[:, pc_y],
            color="#8c8c8c",
            alpha=0.06,
            s=8,
            linewidths=0,
        )
        if highlight_count > 0:
            last_scatter = axis.scatter(
                highlight_scores[:, pc_x],
                highlight_scores[:, pc_y],
                c=highlight_probabilities,
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
                s=8,
                alpha=0.9,
                linewidths=0,
            )
        else:
            axis.text(
                0.5,
                0.5,
                "No matching trials",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        axis.axhline(0.0, color="#d0d0d0", linewidth=0.7, zorder=0)
        axis.axvline(0.0, color="#d0d0d0", linewidth=0.7, zorder=0)
        axis.set_xlabel(f"PC{pc_x + 1}")
        axis.set_ylabel(f"PC{pc_y + 1}")
        axis.set_title(f"PC{pc_x + 1} vs PC{pc_y + 1}")

    for axis in axes_flat[len(pc_pairs) :]:
        axis.axis("off")

    if math.isfinite(highlight_mean_probability) and math.isfinite(highlight_sem_probability):
        probability_summary = (
            f"mean={highlight_mean_probability:.4g} +/- "
            f"{highlight_sem_probability:.4g}"
        )
    else:
        probability_summary = "mean=nan +/- nan"
    fig.suptitle(
        f"subject_id={subject_id} | {probability_column} "
        f"(n={highlight_count}, {probability_summary})"
    )
    fig.tight_layout(rect=(0, 0, 0.88, 0.94))
    if last_scatter is not None:
        colorbar_axis = fig.add_axes((0.91, 0.14, 0.018, 0.74))
        colorbar = fig.colorbar(last_scatter, cax=colorbar_axis)
        colorbar.set_label(probability_column)
    fig.savefig(output_path)
    plt.close(fig)


def _validate_baseline_q_space_dataframe(
    trial_df: Any,
    *,
    required_value_columns: Sequence[str] = (),
) -> None:
    pd = _import_pandas()

    df = pd.DataFrame(trial_df)
    required_columns = ["q_rl_left", "q_rl_right", *[str(column) for column in required_value_columns]]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "trial_advantage_pickle is missing required columns for baseline Q-space "
            f"analysis: {missing_columns}"
        )


def _build_baseline_q_space_result(
    trial_df: Any,
    *,
    finite_value_columns: Sequence[str],
) -> dict[str, Any]:
    pd = _import_pandas()
    np = _import_numpy()

    _validate_baseline_q_space_dataframe(
        trial_df,
        required_value_columns=finite_value_columns,
    )
    df = pd.DataFrame(trial_df).copy().reset_index(drop=True)
    q_left = df["q_rl_left"].to_numpy(dtype=float)
    q_right = df["q_rl_right"].to_numpy(dtype=float)
    finite_mask = np.isfinite(q_left) & np.isfinite(q_right)
    for column in finite_value_columns:
        finite_mask = finite_mask & np.isfinite(df[str(column)].to_numpy(dtype=float))
    n_valid = int(np.sum(finite_mask))
    if n_valid < 1:
        raise ValueError(
            "At least one finite baseline RL Q-space trial is required for plotting."
        )
    x_limits, y_limits = _baseline_q_space_axis_limits(q_left[finite_mask], q_right[finite_mask])
    return {
        "trial_df": df,
        "finite_mask": finite_mask,
        "q_left": q_left,
        "q_right": q_right,
        "x_limits": x_limits,
        "y_limits": y_limits,
        "n_trials_projected": n_valid,
    }


def _baseline_q_space_axis_limits(q_left: Any, q_right: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    np = _import_numpy()

    q_left = np.asarray(q_left, dtype=float)
    q_right = np.asarray(q_right, dtype=float)

    def _limits(values: Any) -> tuple[float, float]:
        finite_values = np.asarray(values, dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return (-1.0, 1.0)
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        if math.isclose(min_value, max_value):
            padding = max(0.05, abs(min_value) * 0.05)
        else:
            padding = max(0.05, (max_value - min_value) * 0.05)
        return (min_value - padding, max_value + padding)

    return _limits(q_left), _limits(q_right)


def _plot_baseline_q_space_condition_figure(
    q_space_result: Mapping[str, Any],
    *,
    condition_column: str,
    condition_label: str,
    condition_mask: Any,
    output_path: Path,
) -> None:
    plt = _import_pyplot()
    np = _import_numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = q_space_result["trial_df"]
    q_left = np.asarray(q_space_result["q_left"], dtype=float)
    q_right = np.asarray(q_space_result["q_right"], dtype=float)
    finite_mask = np.asarray(q_space_result["finite_mask"], dtype=bool)
    advantages = df["advantage"].to_numpy(dtype=float)
    condition_mask = np.asarray(condition_mask, dtype=bool)
    highlight_mask = finite_mask & condition_mask & np.isfinite(advantages)
    highlight_advantages = advantages[highlight_mask]
    highlight_count = int(np.sum(highlight_mask))
    highlight_mean_advantage = (
        float(np.mean(highlight_advantages))
        if highlight_advantages.size
        else math.nan
    )
    highlight_sem_advantage = (
        float(np.std(highlight_advantages, ddof=1) / math.sqrt(highlight_advantages.size))
        if highlight_advantages.size > 1
        else 0.0 if highlight_advantages.size == 1 else math.nan
    )
    finite_advantages = advantages[finite_mask]
    max_abs_advantage = (
        float(np.nanmax(np.abs(finite_advantages)))
        if finite_advantages.size
        else 1.0
    )
    if not np.isfinite(max_abs_advantage) or max_abs_advantage <= 0.0:
        max_abs_advantage = 1.0

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=120)
    ax.scatter(
        q_left[finite_mask],
        q_right[finite_mask],
        color="#8c8c8c",
        alpha=0.06,
        s=8,
        linewidths=0,
    )
    last_scatter = None
    if highlight_count > 0:
        last_scatter = ax.scatter(
            q_left[highlight_mask],
            q_right[highlight_mask],
            c=highlight_advantages,
            cmap="coolwarm",
            vmin=-max_abs_advantage,
            vmax=max_abs_advantage,
            s=8,
            alpha=0.9,
            linewidths=0,
        )
    else:
        ax.text(0.5, 0.5, "No matching trials", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0.0, color="#d0d0d0", linewidth=0.7, zorder=0)
    ax.axvline(0.0, color="#d0d0d0", linewidth=0.7, zorder=0)
    ax.set_xlim(*q_space_result["x_limits"])
    ax.set_ylim(*q_space_result["y_limits"])
    ax.set_xlabel("Q(left)")
    ax.set_ylabel("Q(right)")
    if math.isfinite(highlight_mean_advantage) and math.isfinite(highlight_sem_advantage):
        value_summary = (
            f"advantage={highlight_mean_advantage:.4g} +/- "
            f"{highlight_sem_advantage:.4g}"
        )
    else:
        value_summary = "advantage=nan +/- nan"
    fig.suptitle(
        f"Baseline Q-space | {condition_column} = {condition_label} "
        f"(n={highlight_count}, {value_summary})"
    )
    fig.tight_layout(rect=(0, 0, 0.86, 0.94))
    if last_scatter is not None:
        colorbar_axis = fig.add_axes((0.89, 0.16, 0.025, 0.72))
        colorbar = fig.colorbar(last_scatter, cax=colorbar_axis)
        colorbar.set_label("Log-likelihood advantage")
    fig.savefig(output_path)
    plt.close(fig)


def _plot_baseline_q_space_subject_probability_figure(
    q_space_result: Mapping[str, Any],
    *,
    subject_id: str,
    subject_mask: Any,
    probability_column: str,
    output_path: Path,
) -> None:
    plt = _import_pyplot()
    np = _import_numpy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = q_space_result["trial_df"]
    if probability_column not in df.columns:
        raise ValueError(
            f"trial_advantage_pickle must contain probability_column={probability_column!r}."
        )

    q_left = np.asarray(q_space_result["q_left"], dtype=float)
    q_right = np.asarray(q_space_result["q_right"], dtype=float)
    finite_mask = np.asarray(q_space_result["finite_mask"], dtype=bool)
    probability_values = df[probability_column].to_numpy(dtype=float)
    subject_mask = np.asarray(subject_mask, dtype=bool)
    if subject_mask.shape[0] != finite_mask.shape[0]:
        raise ValueError("subject_mask must be row-aligned to the Q-space dataframe.")

    highlight_mask = finite_mask & subject_mask & np.isfinite(probability_values)
    highlight_probabilities = probability_values[highlight_mask]
    highlight_count = int(np.sum(highlight_mask))
    highlight_mean_probability = (
        float(np.mean(highlight_probabilities))
        if highlight_probabilities.size
        else math.nan
    )
    highlight_sem_probability = (
        float(
            np.std(highlight_probabilities, ddof=1)
            / math.sqrt(highlight_probabilities.size)
        )
        if highlight_probabilities.size > 1
        else 0.0 if highlight_probabilities.size == 1 else math.nan
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=120)
    ax.scatter(
        q_left[finite_mask],
        q_right[finite_mask],
        color="#8c8c8c",
        alpha=0.06,
        s=8,
        linewidths=0,
    )
    last_scatter = None
    if highlight_count > 0:
        last_scatter = ax.scatter(
            q_left[highlight_mask],
            q_right[highlight_mask],
            c=highlight_probabilities,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=8,
            alpha=0.9,
            linewidths=0,
        )
    else:
        ax.text(0.5, 0.5, "No matching trials", ha="center", va="center", transform=ax.transAxes)
    ax.axhline(0.0, color="#d0d0d0", linewidth=0.7, zorder=0)
    ax.axvline(0.0, color="#d0d0d0", linewidth=0.7, zorder=0)
    ax.set_xlim(*q_space_result["x_limits"])
    ax.set_ylim(*q_space_result["y_limits"])
    ax.set_xlabel("Q(left)")
    ax.set_ylabel("Q(right)")
    if math.isfinite(highlight_mean_probability) and math.isfinite(highlight_sem_probability):
        value_summary = (
            f"mean={highlight_mean_probability:.4g} +/- "
            f"{highlight_sem_probability:.4g}"
        )
    else:
        value_summary = "mean=nan +/- nan"
    fig.suptitle(
        f"Baseline Q-space | subject_id={subject_id} | {probability_column} "
        f"(n={highlight_count}, {value_summary})"
    )
    fig.tight_layout(rect=(0, 0, 0.86, 0.94))
    if last_scatter is not None:
        colorbar_axis = fig.add_axes((0.89, 0.16, 0.025, 0.72))
        colorbar = fig.colorbar(last_scatter, cax=colorbar_axis)
        colorbar.set_label(probability_column)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_variable_summary(
    summary_df: Any,
    *,
    variable_name: str,
    output_path: Path,
) -> None:
    plt = _import_pyplot()
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    _plot_pooled_summary_on_axis(ax, summary_df, variable_name=variable_name)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_summary_figure(
    pooled_summary_df: Any,
    *,
    output_path: Path,
) -> None:
    plt = _import_pyplot()

    n_variables = len(_VARIABLE_SPECS)
    n_cols = 3
    n_rows = int(math.ceil(n_variables / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.75 * n_rows), dpi=120)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for axis, spec in zip(axes_flat, _VARIABLE_SPECS, strict=False):
        spec_df = pooled_summary_df[pooled_summary_df["variable"] == spec.name].copy()
        _plot_pooled_summary_on_axis(axis, spec_df, variable_name=spec.name)

    for axis in axes_flat[len(_VARIABLE_SPECS) :]:
        axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_pooled_summary_on_axis(axis: Any, summary_df: Any, *, variable_name: str) -> None:
    spec = _VARIABLE_SPEC_BY_NAME[variable_name]
    ordered_df = _summary_df_for_variable(summary_df, variable_name=variable_name)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1.1)

    if ordered_df.empty:
        axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(spec.label)
        return

    x_positions = list(range(len(ordered_df)))
    axis.bar(
        x_positions,
        ordered_df["mean_advantage"].to_numpy(dtype=float),
        yerr=ordered_df["sem_advantage"].to_numpy(dtype=float),
        color="#4e79a7",
        alpha=0.9,
        width=0.8,
        capsize=3,
    )
    axis.set_xticks(x_positions)
    axis.set_xticklabels(ordered_df["bin"].tolist(), rotation=35, ha="right")
    axis.set_ylabel("Mean advantage")
    axis.set_title(spec.label)


def _plot_session_stage_summary(
    session_stage_summary_df: Any,
    *,
    variable_names: Sequence[str],
    output_path: Path,
) -> None:
    plt = _import_pyplot()
    np = _import_numpy()

    n_panels = max(1, len(variable_names))
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5.2), dpi=120)
    if n_panels == 1:
        axes = [axes]

    for axis, variable_name in zip(axes, variable_names, strict=False):
        spec = _VARIABLE_SPEC_BY_NAME[variable_name]
        variable_df = _summary_df_for_variable(
            session_stage_summary_df,
            variable_name=variable_name,
        )
        axis.axhline(0.0, color="black", linestyle="--", linewidth=1.1)
        if variable_df.empty:
            axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
            axis.set_title(spec.label)
            continue

        bin_rows = (
            variable_df.loc[:, ["bin", "bin_order"]]
            .drop_duplicates()
            .sort_values("bin_order")
            .reset_index(drop=True)
        )
        x_positions = np.arange(len(bin_rows))
        bar_width = 0.24
        offsets = {
            stage: (stage_index - (len(_SESSION_STAGE_ORDER) - 1) / 2.0) * bar_width
            for stage_index, stage in enumerate(_SESSION_STAGE_ORDER)
        }

        for stage_name in _SESSION_STAGE_ORDER:
            stage_df = variable_df[variable_df["session_stage"] == stage_name].copy()
            merged_df = bin_rows.merge(
                stage_df.loc[:, ["bin", "bin_order", "mean_advantage", "sem_advantage"]],
                on=["bin", "bin_order"],
                how="left",
            )
            axis.bar(
                x_positions + offsets[stage_name],
                merged_df["mean_advantage"].fillna(0.0).to_numpy(dtype=float),
                yerr=merged_df["sem_advantage"].fillna(0.0).to_numpy(dtype=float),
                width=bar_width,
                color=_SESSION_STAGE_COLORS[stage_name],
                alpha=0.9,
                capsize=3,
                label=stage_name,
            )

        axis.set_xticks(x_positions)
        axis.set_xticklabels(bin_rows["bin"].tolist(), rotation=35, ha="right")
        axis.set_ylabel("Mean advantage")
        axis.set_title(spec.label)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(handles, labels, title="Session stage", loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _summary_df_for_variable(summary_df: Any, *, variable_name: str) -> Any:
    pd = _import_pandas()

    filtered_df = pd.DataFrame(summary_df)[pd.DataFrame(summary_df)["variable"] == variable_name].copy()
    if filtered_df.empty:
        return filtered_df
    sort_columns = []
    if "session_stage" in filtered_df.columns:
        filtered_df["session_stage"] = pd.Categorical(
            filtered_df["session_stage"],
            categories=list(_SESSION_STAGE_ORDER),
            ordered=True,
        )
        sort_columns.append("session_stage")
    sort_columns.append("bin_order")
    return filtered_df.sort_values(sort_columns).reset_index(drop=True)


def _build_wide_bin_summary(pooled_summary_df: Any) -> Any:
    pd = _import_pandas()

    if pd.DataFrame(pooled_summary_df).empty:
        return pd.DataFrame(columns=["variable", "display_label"])

    sorted_df = _sort_summary_dataframe(pd.DataFrame(pooled_summary_df).copy(), groupby_cols=())
    wide_df = sorted_df.pivot_table(
        index=["variable", "display_label"],
        columns="bin",
        values="mean_advantage",
        aggfunc="first",
    ).reset_index()
    wide_df.columns = [
        str(column) if not isinstance(column, tuple) else "__".join(str(item) for item in column if item)
        for column in wide_df.columns
    ]
    return wide_df


def _rank_variables(pooled_summary_df: Any) -> Any:
    pd = _import_pandas()

    ranking_rows = []
    summary_df = pd.DataFrame(pooled_summary_df).copy()
    for spec in _VARIABLE_SPECS:
        variable_df = summary_df[summary_df["variable"] == spec.name].copy()
        if variable_df.empty:
            ranking_rows.append(
                {
                    "variable": spec.name,
                    "display_label": spec.label,
                    "score": math.nan,
                    "n_bins": 0,
                }
            )
            continue
        mean_values = variable_df["mean_advantage"].dropna().astype(float).tolist()
        score = (
            float(max(mean_values) - min(mean_values))
            if mean_values
            else math.nan
        )
        ranking_rows.append(
            {
                "variable": spec.name,
                "display_label": spec.label,
                "score": score,
                "n_bins": int(variable_df["bin"].nunique()),
            }
        )
    ranking_df = pd.DataFrame.from_records(ranking_rows)
    ranking_df["sort_score"] = ranking_df["score"].fillna(float("-inf"))
    ranking_df = ranking_df.sort_values(
        ["sort_score", "display_label"],
        ascending=[False, True],
    ).drop(columns=["sort_score"]).reset_index(drop=True)
    return ranking_df


def _sort_summary_dataframe(summary_df: Any, *, groupby_cols: Sequence[str]) -> Any:
    pd = _import_pandas()

    sorted_df = pd.DataFrame(summary_df).copy()
    if sorted_df.empty:
        return sorted_df
    if "session_stage" in groupby_cols and "session_stage" in sorted_df.columns:
        sorted_df["session_stage"] = pd.Categorical(
            sorted_df["session_stage"],
            categories=list(_SESSION_STAGE_ORDER),
            ordered=True,
        )
    sort_columns = ["variable", *groupby_cols, "bin_order"]
    return sorted_df.sort_values(sort_columns).reset_index(drop=True)


def _sort_raw_dataframe(df: Any) -> Any:
    pd = _import_pandas()

    sorted_df = pd.DataFrame(df).copy()
    if "ses_idx" in sorted_df.columns:
        ordered_session_ids = list(dict.fromkeys(sorted_df["ses_idx"].tolist()))
        sorted_df["ses_idx"] = pd.Categorical(
            sorted_df["ses_idx"],
            categories=ordered_session_ids,
            ordered=True,
        )
    sort_columns = ["ses_idx"]
    if "trial" in sorted_df.columns:
        sort_columns.append("trial")
    sorted_df = sorted_df.sort_values(sort_columns).reset_index(drop=True)
    if "ses_idx" in sorted_df.columns:
        sorted_df["ses_idx"] = sorted_df["ses_idx"].astype(str)
    return sorted_df


def _sort_trial_dataframe(df: Any) -> Any:
    pd = _import_pandas()

    sorted_df = pd.DataFrame(df).copy()
    sort_columns = []
    if "subject_id" in sorted_df.columns:
        sort_columns.append("subject_id")
    if "session_idx" in sorted_df.columns:
        sort_columns.append("session_idx")
    elif "ses_idx" in sorted_df.columns:
        sort_columns.append("ses_idx")
    if "trial_idx" in sorted_df.columns:
        sort_columns.append("trial_idx")
    if sort_columns:
        sorted_df = sorted_df.sort_values(sort_columns).reset_index(drop=True)
    return sorted_df


def _encode_history_trial(choice: int, reward: float) -> str:
    if int(choice) == 0:
        return "L" if float(reward) > 0 else "l"
    return "R" if float(reward) > 0 else "r"


def _mean_last_n(values: Sequence[Any], *, n: int) -> float:
    np = _import_numpy()

    if n <= 0:
        raise ValueError("n must be > 0.")
    finite_values = np.asarray(list(values), dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return math.nan
    window = finite_values[-int(n) :]
    if window.size == 0:
        return math.nan
    return float(np.mean(window))


def _quantile_bin_series(values: Any, *, q: int) -> Any:
    pd = _import_pandas()

    series = pd.Series(values, copy=True)
    if series.nunique(dropna=True) <= 1:
        return pd.Series(
            pd.Categorical(
                ["all"] * len(series),
                categories=["all"],
                ordered=True,
            ),
            index=series.index,
        )
    try:
        return pd.qcut(series.astype(float), q=int(q), duplicates="drop")
    except ValueError:
        return pd.Series(
            pd.Categorical(
                ["all"] * len(series),
                categories=["all"],
                ordered=True,
            ),
            index=series.index,
        )


def _resolve_categorical_categories(values: Any, *, spec: VariableSpec) -> list[Any]:
    pd = _import_pandas()

    series = pd.Series(values, copy=True)
    if spec.category_order:
        observed = set(series.dropna().tolist())
        return [category for category in spec.category_order if category in observed]

    unique_values = list(dict.fromkeys(series.dropna().tolist()))
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in unique_values):
        return sorted(unique_values)
    return sorted(unique_values, key=lambda value: str(value))


def _safe_filename_component(value: Any) -> str:
    raw = str(value).strip()
    if not raw:
        return "value"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._")
    return safe or "value"


def _import_numpy():
    import numpy as np

    return np


def _import_pandas():
    return lc._import_pandas()


def _import_pyplot():
    import matplotlib.pyplot as plt

    return plt
