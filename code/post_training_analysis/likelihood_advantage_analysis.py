"""Standalone log-likelihood advantage analysis helpers."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
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

    baseline_prob_df = _build_baseline_probability_frame(
        run_model2,
        trial_df=base_trial_df,
        split_name=split_name,
    )
    trial_df = _merge_probability_frame(
        trial_df,
        baseline_prob_df,
        prob_column="p_rl",
        source_label=run_model2.model_label,
    )

    trial_df = pd.DataFrame(trial_df).copy()
    trial_df["p_model1"] = np.clip(
        trial_df["p_model1"].to_numpy(dtype=float),
        _EPSILON,
        1.0 - _EPSILON,
    )
    trial_df["p_rl"] = np.clip(
        trial_df["p_rl"].to_numpy(dtype=float),
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

    trial_advantage_pickle_path = resolved_output_dir / "trial_advantage.pkl"
    trial_df.to_pickle(trial_advantage_pickle_path)

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

    if n_action_logits <= 0:
        raise ValueError(f"Expected n_action_logits > 0, received {n_action_logits}")

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
        chosen_probabilities = action_probabilities[binary_mask, binary_choices]
        subject_values = (
            session_df.loc[binary_mask, "subject_id"].tolist()
            if "subject_id" in session_df.columns
            else ["unknown"] * int(np.sum(binary_mask))
        )
        for trial_idx, (subject_id, action, probability) in enumerate(
            zip(
                subject_values,
                binary_choices.tolist(),
                chosen_probabilities.tolist(),
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
                }
            )

    return pd.DataFrame.from_records(
        records,
        columns=["subject_id", "ses_idx", "trial_idx", "action", "p_model1"],
    )


def _build_baseline_probability_frame(
    run: lc.ResolvedLikelihoodRun,
    *,
    trial_df: Any,
    split_name: str,
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
                )
            )
        return (
            pd.concat(all_frames, ignore_index=True)
            if all_frames
            else pd.DataFrame(columns=["subject_id", "ses_idx", "trial_idx", "action", "p_rl"])
        )

    fitted_params = baseline_output.get("fitted_params")
    if not isinstance(fitted_params, Mapping) or not fitted_params:
        raise ValueError("baseline_rl_output.json is missing fitted_params.")
    return _rollout_baseline_probabilities(
        run,
        trial_df=pd.DataFrame(trial_df).copy(),
        fitted_params=dict(fitted_params),
    )


def _rollout_baseline_probabilities(
    run: lc.ResolvedLikelihoodRun,
    *,
    trial_df: Any,
    fitted_params: Mapping[str, Any],
) -> Any:
    pd = _import_pandas()

    baseline_output = lc._load_baseline_output_for_run(run)
    agent_class_name, agent_kwargs = lc._resolve_baseline_agent_spec(run, baseline_output)
    session_payloads = _session_rollout_payloads_from_trial_df(trial_df)
    if not session_payloads:
        return pd.DataFrame(
            columns=["subject_id", "ses_idx", "trial_idx", "action", "p_rl"]
        )

    choice_prob_sessions = lc._perform_baseline_agent_rollout(
        agent_class_name=agent_class_name,
        agent_kwargs=agent_kwargs,
        fitted_params=fitted_params,
        choice_sessions=[payload["choices"] for payload in session_payloads],
        reward_sessions=[payload["rewards"] for payload in session_payloads],
        seed=run.seed,
    )
    if len(choice_prob_sessions) != len(session_payloads):
        raise ValueError(
            "Baseline rollout returned a different number of sessions than requested: "
            f"expected={len(session_payloads)} received={len(choice_prob_sessions)}"
        )
    return _probability_frame_from_rollout_sessions(
        session_payloads,
        choice_prob_sessions=choice_prob_sessions,
    )


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
) -> Any:
    pd = _import_pandas()

    records: list[dict[str, Any]] = []
    for payload, choice_prob_session in zip(session_payloads, choice_prob_sessions):
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
        for trial_idx, action in enumerate(choices.tolist(), start=1):
            records.append(
                {
                    "subject_id": payload["subject_id"],
                    "ses_idx": payload["ses_idx"],
                    "trial_idx": int(trial_idx),
                    "action": int(action),
                    "p_rl": float(aligned_probabilities[int(action), trial_idx - 1]),
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["subject_id", "ses_idx", "trial_idx", "action", "p_rl"],
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
    expected_columns = {"ses_idx", "trial_idx", "action", prob_column}
    missing_columns = [column for column in expected_columns if column not in right_df.columns]
    if missing_columns:
        raise ValueError(
            f"Probability frame for {source_label!r} is missing columns: {missing_columns}"
        )

    duplicated = right_df.duplicated(subset=["ses_idx", "trial_idx"])
    if duplicated.any():
        raise ValueError(
            f"Probability frame for {source_label!r} contains duplicate ses_idx/trial_idx keys."
        )

    merged = left_df.merge(
        right_df.loc[:, ["ses_idx", "trial_idx", "action", prob_column]],
        on=["ses_idx", "trial_idx"],
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
    if merged[prob_column].isna().any():
        missing_rows = merged.loc[merged[prob_column].isna(), ["ses_idx", "trial_idx"]]
        raise ValueError(
            f"Merged probability column {prob_column!r} contains NaNs for {source_label!r}: "
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
