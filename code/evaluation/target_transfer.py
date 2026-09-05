"""Shared trial-level contract for external target-transfer evaluation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

TRIAL_KEY_COLUMNS = ("subject_id", "ses_idx", "trial")


def target_test_mask(
    trial_df: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> np.ndarray:
    """Return the immutable target-test membership encoded by the bundle metadata."""
    if str(metadata.get("split_strategy")) == "within_session_prefix_suffix":
        partition_column = str(
            metadata.get("trial_partition_column", "external_split_partition")
        )
        if partition_column not in trial_df.columns:
            raise ValueError(
                "Prefix/suffix target evaluation requires raw column "
                f"{partition_column!r}."
            )
        test_partition = str(metadata.get("test_trial_partition", "test"))
        mask = trial_df[partition_column].astype(str).eq(test_partition).to_numpy()
    else:
        eval_session_ids = metadata.get("eval_session_ids")
        if not isinstance(eval_session_ids, list) or not eval_session_ids:
            raise ValueError(
                "Target evaluation requires non-empty metadata['eval_session_ids']."
            )
        eval_ids = {str(value) for value in eval_session_ids}
        mask = trial_df["ses_idx"].astype(str).isin(eval_ids).to_numpy()

    if not bool(np.any(mask)):
        raise ValueError("Target-test membership resolved to zero trials.")
    return np.asarray(mask, dtype=bool)


def build_binary_trial_predictions(
    trial_df: pd.DataFrame,
    metadata: Mapping[str, Any],
    *,
    probability_choice_1: Sequence[float] | np.ndarray,
    model: str,
) -> pd.DataFrame:
    """Build the canonical scored test-trial table for a binary-choice model.

    ``probability_choice_1`` must align one-to-one with ``trial_df`` before the
    immutable target-test mask is applied.  Keeping the test selection here gives
    GRU and Q-learning one parity-checkable output contract.
    """
    missing = [column for column in TRIAL_KEY_COLUMNS if column not in trial_df.columns]
    if missing:
        raise ValueError(f"Target trial dataframe is missing key columns: {missing}.")
    if "animal_response" not in trial_df.columns:
        raise ValueError("Target trial dataframe is missing 'animal_response'.")

    probabilities = np.asarray(probability_choice_1, dtype=float).reshape(-1)
    if len(probabilities) != len(trial_df):
        raise ValueError(
            "Target probability/table length mismatch: "
            f"probabilities={len(probabilities)}, rows={len(trial_df)}."
        )
    mask = target_test_mask(trial_df, metadata)
    scored = trial_df.loc[mask].copy()
    probability_1 = probabilities[mask]

    if str(metadata.get("ignore_policy", "exclude")) == "exclude":
        valid_choice = scored["animal_response"].isin([0, 1]).to_numpy()
        scored = scored.loc[valid_choice].copy()
        probability_1 = probability_1[valid_choice]
    if scored.empty:
        raise ValueError("Target-test membership resolved to zero scorable trials.")

    if not np.all(np.isfinite(probability_1)):
        raise ValueError("Scored target-test probabilities must all be finite.")
    if np.any((probability_1 < 0.0) | (probability_1 > 1.0)):
        raise ValueError("Scored target-test probabilities must lie in [0, 1].")
    choices = scored["animal_response"].to_numpy(dtype=int)
    if np.any((choices != 0) & (choices != 1)):
        raise ValueError("Target-test choices must be binary 0/1.")

    probability_chosen = np.where(choices == 1, probability_1, 1.0 - probability_1)
    probability_chosen = np.clip(probability_chosen, 1e-10, 1.0 - 1e-10)
    predicted_choice = (probability_1 >= 0.5).astype(int)

    keep_columns = list(TRIAL_KEY_COLUMNS)
    for optional_column in ("source_ses_idx", "dataset_id", "species"):
        if optional_column in scored.columns:
            keep_columns.append(optional_column)
    result = scored[keep_columns].reset_index(drop=True)
    result["split"] = "test"
    result["model"] = str(model)
    result["choice"] = choices
    result["probability_choice_0"] = 1.0 - probability_1
    result["probability_choice_1"] = probability_1
    result["probability_chosen"] = probability_chosen
    result["predicted_choice"] = predicted_choice
    result["log_likelihood_nats"] = np.log(probability_chosen)
    result["log_likelihood_bits"] = np.log2(probability_chosen)
    result["brier_score"] = np.square(probability_1 - choices)
    result["correct"] = (predicted_choice == choices).astype(int)
    return result


def summarize_binary_trial_predictions(predictions: pd.DataFrame) -> dict[str, Any]:
    """Return pooled and per-subject metrics without changing trial membership."""
    required = {
        "subject_id",
        "log_likelihood_nats",
        "log_likelihood_bits",
        "brier_score",
        "correct",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing metric columns: {missing}.")
    if predictions.empty:
        raise ValueError("Cannot summarize an empty target prediction table.")

    def _summary(frame: pd.DataFrame) -> dict[str, Any]:
        mean_log_likelihood_nats = float(frame["log_likelihood_nats"].mean())
        return {
            "n_trials": int(len(frame)),
            "total_log_likelihood_nats": float(frame["log_likelihood_nats"].sum()),
            "total_log_likelihood_bits": float(frame["log_likelihood_bits"].sum()),
            "mean_log_likelihood_nats": mean_log_likelihood_nats,
            "mean_log_likelihood_bits": float(frame["log_likelihood_bits"].mean()),
            "normalized_likelihood": float(np.exp(mean_log_likelihood_nats)),
            "brier_score": float(frame["brier_score"].mean()),
            "accuracy": float(frame["correct"].mean()),
        }

    per_subject = []
    for subject_id, subject_predictions in predictions.groupby(
        "subject_id", sort=False
    ):
        per_subject.append(
            {
                "subject_id": subject_id,
                **_summary(subject_predictions),
            }
        )
    return {
        **_summary(predictions),
        "per_subject": per_subject,
        "calibration_columns": ["choice", "probability_choice_1"],
    }


def assert_aligned_test_trial_keys(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> None:
    """Raise when two model outputs do not score exactly the same ordered trials."""
    for label, frame in (("left", left), ("right", right)):
        missing = [
            column for column in TRIAL_KEY_COLUMNS if column not in frame.columns
        ]
        if missing:
            raise ValueError(f"{label} prediction table is missing keys: {missing}.")
        if frame.duplicated(list(TRIAL_KEY_COLUMNS)).any():
            raise ValueError(f"{label} prediction table has duplicate trial keys.")

    left_keys = list(left.loc[:, TRIAL_KEY_COLUMNS].itertuples(index=False, name=None))
    right_keys = list(
        right.loc[:, TRIAL_KEY_COLUMNS].itertuples(index=False, name=None)
    )
    if left_keys != right_keys:
        left_set = set(left_keys)
        right_set = set(right_keys)
        raise ValueError(
            "Target-test trial keys are not identical and ordered across models; "
            f"left_only={list(left_set - right_set)[:5]}, "
            f"right_only={list(right_set - left_set)[:5]}."
        )
