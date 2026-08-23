"""Driver: 3-way generative rollout + model-vs-real ignore-statistics report.

Runs on the compute node (needs jax/haiku + the snapshot DB, so NOT importable
in the Mac sandbox). Given a saved 3-way (ignore_policy="include") run dir:

  1. roll out the model over its split's sessions (curriculum-matched),
  2. compute ignore statistics on the MODEL rollouts (raw choice_history, keeps 2),
  3. compute ignore statistics on the REAL sessions from the raw snapshot
     ``animal_response`` column (NOT the NaN-collapsed choice_history),
  4. compare the two, write JSON + CSV + a 4-panel figure.

CLI:
    python -m post_training_analysis.ignore_generative_report \
        --model-dir <RUN_DIR> --split train \
        --checkpoint-policy best_eval \
        --n-rollouts-per-session 1 \
        --output-dir <OUT>

Deliberately standalone from run_post_training_analysis so the exclude/2-way
pipeline and the other analyses are untouched.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_real_choice_sequences(resolved_run: Any) -> list[list[int]]:
    """Real per-session choice sequences over raw {0,1,2}, straight from the
    snapshot's animal_response (preserving ignore=2). Mirrors the selection in
    load_animal_session_history but keeps the raw codes."""
    load_db = importlib.import_module("utils.load_mice_database")
    snapshot_cols = [
        "trial", "subject_id", "ses_idx", "animal_response",
        "earned_reward", "curriculum_name", "current_stage_actual",
    ]
    selection_subject_ids = (
        resolved_run.trained_subject_ids
        if resolved_run.multisubject
        else resolved_run.selection.get("subject_ids")
    )
    snapshot_df, _ = load_db.load_mice_from_database(
        split=resolved_run.split,
        subject_ids=selection_subject_ids,
        curricula=resolved_run.curricula or None,
        subject_ratio=resolved_run.selection.get("subject_ratio"),
        min_sessions=int(resolved_run.selection.get("min_sessions", 10)),
        heldout_every_n=int(resolved_run.selection.get("heldout_every_n", 5)),
        seed=resolved_run.selection.get("subject_sample_seed"),
        mature_only=resolved_run.mature_only,
        cols_to_retain=snapshot_cols,
        snapshot=resolved_run.selection.get("snapshot"),
    )
    if len(snapshot_df) == 0:
        raise ValueError("Empty snapshot selection for the real-data side.")
    trial_col = "trial" if "trial" in snapshot_df.columns else None
    seqs: list[list[int]] = []
    grp = snapshot_df.groupby(["subject_id", "ses_idx"], sort=False)
    for _, sdf in grp:
        if trial_col:
            sdf = sdf.sort_values(trial_col)
        seqs.append([int(v) for v in sdf["animal_response"].tolist()])
    return seqs


def run_ignore_generative_report(
    model_dir: str,
    *,
    split: str = "train",
    checkpoint_policy: str = "best_eval",
    n_rollouts_per_session: int = 1,
    output_dir: str | None = None,
) -> dict[str, Any]:
    ga = importlib.import_module("post_training_analysis.generative_analysis")
    ig = importlib.import_module("post_training_analysis.ignore_statistics")
    resolve = importlib.import_module("post_training_analysis.generative_analysis").resolve_model_run

    resolved = resolve(model_dir, split=split, checkpoint_policy=checkpoint_policy)
    if resolved.ignore_policy != "include":
        raise ValueError(
            "This report is for 3-way (ignore_policy='include') runs; "
            f"got ignore_policy={resolved.ignore_policy!r}."
        )

    # --- model side: roll out, take raw choice_history (keeps ignore=2) ---
    animal_sessions = ga.load_animal_session_history(resolved, split=split)
    sim_df = ga.simulate_model_sessions(
        resolved_run=resolved,
        animal_sessions=animal_sessions,
        rollout_mode="curriculum_matched",
        n_rollouts_per_session=int(n_rollouts_per_session),
    )
    model_stats = ig.compute_ignore_statistics(sim_df)  # reads choice_history

    # --- real side: raw animal_response straight from snapshot (keeps 2) ---
    real_seqs = _load_real_choice_sequences(resolved)
    real_stats = ig.compute_ignore_statistics(real_seqs)

    comparison = ig.compare_sides(model_stats, real_stats)

    out = Path(output_dir) if output_dir else Path(model_dir) / "ignore_generative_report"
    out.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model_dir": str(resolved.model_dir),
        "checkpoint_step": getattr(resolved, "checkpoint_step", None),
        "split": split,
        "n_rollouts_per_session": int(n_rollouts_per_session),
        "model": model_stats,
        "real": real_stats,
        "comparison": comparison,
    }
    (out / "ignore_stats.json").write_text(json.dumps(bundle, indent=2, default=str))
    _write_csv(out / "ignore_stats_summary.csv", model_stats, real_stats, comparison)
    _plot(out / "ignore_generative_comparison.png", model_stats, real_stats, comparison, resolved)
    logger.info("Wrote ignore-generative report to %s", out)
    return bundle


def _write_csv(path: Path, model_stats, real_stats, comparison) -> None:
    import csv
    rows = [
        ("ignore_rate", model_stats["overall"]["ignore_rate"], real_stats["overall"]["ignore_rate"]),
        ("mean_run_length", model_stats["run_lengths"]["mean_run_length"], real_stats["run_lengths"]["mean_run_length"]),
        ("max_run_length", model_stats["run_lengths"]["max_run_length"], real_stats["run_lengths"]["max_run_length"]),
        ("p_ignore_given_prev_ignore",
         model_stats["autocorrelation"]["p_ignore_given_prev_ignore"],
         real_stats["autocorrelation"]["p_ignore_given_prev_ignore"]),
        ("p_ignore_given_prev_not_ignore",
         model_stats["autocorrelation"]["p_ignore_given_prev_not_ignore"],
         real_stats["autocorrelation"]["p_ignore_given_prev_not_ignore"]),
        ("n_trials", model_stats["overall"]["n_trials"], real_stats["overall"]["n_trials"]),
        ("n_sessions", model_stats["overall"]["n_sessions"], real_stats["overall"]["n_sessions"]),
    ]
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["statistic", "model", "real"])
        for name, m, r in rows:
            w.writerow([name, m, r])
        w.writerow([])
        w.writerow(["comparison_metric", "value", ""])
        for k, v in comparison.items():
            w.writerow([k, v, ""])


def _plot(path: Path, model_stats, real_stats, comparison, resolved) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    MC, RC = "#C44E52", "#4C72B0"  # model=red, real=blue

    # (a) overall ignore rate
    ax = axes[0, 0]
    ax.bar([0, 1], [model_stats["overall"]["ignore_rate"], real_stats["overall"]["ignore_rate"]],
           color=[MC, RC], width=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["model", "real"])
    ax.set_ylabel("overall ignore rate")
    ax.set_title(f"(a) Ignore rate  (Δ={comparison['ignore_rate_diff']:+.4f})")

    # (b) ignore rate vs within-session position
    ax = axes[0, 1]
    mc, rc = model_stats["by_position"], real_stats["by_position"]
    ax.plot(mc["bin_center"], mc["ignore_rate"], "-o", color=MC, label="model")
    ax.plot(rc["bin_center"], rc["ignore_rate"], "-o", color=RC, label="real")
    ax.set_xlabel("fractional position in session"); ax.set_ylabel("ignore rate")
    ax.set_title(f"(b) Ignore rate vs position  (L1={comparison['position_curve_L1']:.4f})")
    ax.legend()

    # (c) run-length distribution (normalized)
    ax = axes[1, 0]
    for stats, color, lab in ((model_stats, MC, "model"), (real_stats, RC, "real")):
        rl = stats["run_lengths"]
        if rl["run_length"]:
            tot = sum(rl["count"])
            ax.plot(rl["run_length"], [c / tot for c in rl["count"]], "-o", color=color, label=lab)
    ax.set_xlabel("consecutive-ignore run length"); ax.set_ylabel("fraction of runs")
    ax.set_yscale("log"); ax.set_title("(c) Ignore run-length distribution"); ax.legend()

    # (d) transition matrix difference (model - real)
    ax = axes[1, 1]
    mt = np.array(model_stats["transition_matrix"]["probs"], dtype=float)
    rt = np.array(real_stats["transition_matrix"]["probs"], dtype=float)
    diff = mt - rt
    im = ax.imshow(diff, cmap="RdBu_r", vmin=-np.nanmax(np.abs(diff)) if np.isfinite(np.nanmax(np.abs(diff))) else -1,
                   vmax=np.nanmax(np.abs(diff)) if np.isfinite(np.nanmax(np.abs(diff))) else 1)
    labels = model_stats["transition_matrix"]["labels"]
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("to"); ax.set_ylabel("from")
    ax.set_title(f"(d) P(transition) model−real  (‖·‖F={comparison['transition_matrix_frobenius']:.3f})")
    for i in range(3):
        for j in range(3):
            if np.isfinite(diff[i, j]):
                ax.text(j, i, f"{diff[i, j]:+.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)

    step = getattr(resolved, "checkpoint_step", "?")
    fig.suptitle(
        f"3-way model vs real: ignore-trial statistics\n{Path(str(resolved.model_dir)).name}  (step {step})",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="3-way generative ignore-statistics report.")
    p.add_argument("--model-dir", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--checkpoint-policy", default="best_eval")
    p.add_argument("--n-rollouts-per-session", type=int, default=1)
    p.add_argument("--output-dir", default=None)
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_arg_parser().parse_args(argv)
    bundle = run_ignore_generative_report(
        args.model_dir,
        split=args.split,
        checkpoint_policy=args.checkpoint_policy,
        n_rollouts_per_session=args.n_rollouts_per_session,
        output_dir=args.output_dir,
    )
    print(json.dumps(bundle["comparison"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
