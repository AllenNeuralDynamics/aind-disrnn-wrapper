"""Beaker-side held-out re-scoring for a finished multisubject GRU/disRNN run.

Backfills held-out metrics (e.g. the 3-way ignore-class precision/recall/F1/
PR-AUC added after some runs were trained) onto an ALREADY-FINISHED run WITHOUT
re-training and WITHOUT changing the model — the exact-reproduction path.

Difference vs the `DISRNN_RESTORE_FROM_RUN_ID` restore path (`run_hpc`):
  * restore  -> resumes the TRAINING entrypoint, then runs end-of-train held-out.
    Even with n_steps capped it evaluates a fresh held-out draw off the restored
    checkpoint set, so it does NOT reproduce the original held-out numbers.
  * this tool -> runs the held-out fine-tune ONLY, pointed at the downloaded
    checkpoint tree, reading every knob (seed, checkpoint_policy, heldout set,
    finetune n_steps/lr) from the SOURCE run's own config, and re-injects the
    result into the ORIGINAL W&B run under the `heldout/` namespace. Reproduces
    the source LR to <1e-7 (proto-validated); the new ignore metrics attach to
    the identical held-out result the original produced.

This is the Beaker port of the HPC `resume_heldout.py`. The only Beaker-specific
work is reconstructing `<model_dir>/inputs.yaml` from the W&B run config, because
the `gru-output-<rid>` artifact ships the `outputs/` tree but NOT inputs.yaml.

Usage (inside a Beaker container that can reach GCS + W&B):
    python resume_heldout_beaker.py --run-id <wandb_run_id> \
        [--entity AIND-disRNN] [--project mice_ignore_scaling] \
        [--work-dir /results/rescore]

Reads WANDB_API_KEY from env (or ~/.netrc).
"""
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

import wandb
import yaml

logger = logging.getLogger("resume_heldout_beaker")


def _fetch_run_config(entity: str, project: str, run_id: str) -> dict:
    """Fetch a run's config via the W&B GraphQL endpoint directly.

    Avoids ``wandb.Api().run()`` — that auto-loads the run's Sweep, and the
    Sweep.get() query is broken in the container's wandb build (raises
    'Object of type Api is not JSON serializable'). A plain GraphQL POST for the
    config field sidesteps the whole run/sweep object graph.
    """
    key = os.environ["WANDB_API_KEY"]
    q = (
        'query{project(name:"%s",entityName:"%s"){run(name:"%s"){config}}}'
        % (project, entity, run_id)
    )
    body = json.dumps({"query": q}).encode()
    auth = base64.b64encode(f"api:{key}".encode()).decode()
    req = urllib.request.Request(
        "https://api.wandb.ai/graphql",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Basic " + auth},
    )
    resp = json.load(urllib.request.urlopen(req))
    raw = resp["data"]["project"]["run"]["config"]
    return json.loads(raw) if isinstance(raw, str) else raw


def _unwrap(d):
    """Strip W&B config's ``{"value":..., "desc":...}`` wrappers, recursively."""
    if isinstance(d, dict):
        if set(d.keys()) <= {"value", "desc"} and "value" in d:
            return _unwrap(d["value"])
        return {k: _unwrap(v) for k, v in d.items()}
    if isinstance(d, list):
        return [_unwrap(x) for x in d]
    return d


def _reconstruct_inputs_yaml(cfg: dict, model_dir: Path) -> None:
    """Rebuild ``<model_dir>/inputs.yaml`` from the source run's config.

    ``resolve_model_run`` requires inputs.yaml (data + model blocks) to rebuild
    the network and the reserved held-out subject set. The training-output
    artifact does not include it, but the W&B run config carries the full nested
    ``data`` and ``model`` blocks. Write exactly those (+ seed) so the resolver
    sees the same config the original training run used.
    """
    cfg = _unwrap(cfg)
    for key in ("data", "model"):
        if key not in cfg or not isinstance(cfg[key], dict):
            raise ValueError(
                f"Source run config is missing a nested '{key}' block; cannot "
                f"reconstruct inputs.yaml (keys present: {sorted(cfg)[:20]})."
            )
    inputs = {"data": cfg["data"], "model": cfg["model"], "seed": cfg.get("seed")}
    (model_dir / "inputs.yaml").write_text(yaml.safe_dump(inputs, sort_keys=False))
    logger.info(
        "Reconstructed inputs.yaml (model.type=%s, ignore_policy=%s, seed=%s)",
        cfg["model"].get("type"),
        cfg["data"].get("ignore_policy"),
        cfg.get("seed"),
    )


def _build_finetune_config(cfg: dict, model_dir: Path) -> dict:
    """Mirror training_runner._run_auto_heldout_finetune's config, sourced from
    the run's own auto_heldout_finetune block so the held-out draw is identical.
    """
    cfg = _unwrap(cfg)
    auto = cfg["model"].get("training", {}).get("auto_heldout_finetune", {}) or {}
    return {
        "source_run": {
            "model_dir": str(model_dir),
            # The original auto-held-out used this policy; match it exactly.
            "checkpoint_policy": str(auto.get("checkpoint_policy", "best_eval")),
        },
        # All None -> inherit the reserved held-out set from inputs.yaml.
        "heldout_subjects": {
            k: None
            for k in (
                "test_subject_ids", "curricula", "min_sessions",
                "heldout_every_n", "mature_only", "cols_to_retain",
            )
        },
        "heldout_finetuning": {
            "n_steps": int(auto.get("n_steps", 500)),
            "lr": float(auto.get("lr", 1e-3)),
            "checkpoint_every_n_steps": int(auto.get("checkpoint_every_n_steps", 100)),
            "batch_size": auto.get("batch_size", None),
            "batch_mode": str(auto.get("batch_mode", "single")),
            "keep_media_files": bool(auto.get("keep_media_files", True)),
            "checkpoint_plot_split_examples_every_n": int(
                auto.get("checkpoint_plot_split_examples_every_n", 100)
            ),
            "checkpoint_save_output_df_every_n": int(
                auto.get("checkpoint_save_output_df_every_n", 0)
            ),
            "train_example_sessions_per_subject": int(
                auto.get("train_example_sessions_per_subject", 1)
            ),
            "eval_example_sessions_per_subject": int(
                auto.get("eval_example_sessions_per_subject", 1)
            ),
            "example_max_subjects": int(auto.get("example_max_subjects", 1)),
        },
        "output": {
            "output_root": str(model_dir.parent / "heldout_rescore"),
            "run_name_suffix": None,
        },
        "seed": cfg.get("seed"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True, help="Source W&B run id to re-score.")
    p.add_argument("--entity", default="AIND-disRNN")
    p.add_argument("--project", default="mice_ignore_scaling")
    p.add_argument("--work-dir", default="/results/rescore")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
    )
    # The wrapper package lives under code/; make its modules importable.
    repo_code = Path(__file__).resolve().parent / "code"
    sys.path.insert(0, str(repo_code))
    from post_training_analysis import run_heldout_subject_finetuning_from_config

    work = Path(args.work_dir).expanduser()
    model_dir = work / args.run_id
    outputs_dir = model_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()

    # 1) Download the training-output artifact into <model_dir>/outputs so the
    #    checkpoints/step_<N>/ tree lands where resolve_model_run looks.
    mtype = "gru"  # ignore-scaling grid is all GRU; disRNN would be "disrnn"
    artifact_ref = f"{args.entity}/{args.project}/{mtype}-output-{args.run_id}:latest"
    logger.info("Downloading %s -> %s", artifact_ref, outputs_dir)
    api.artifact(artifact_ref, type="training-output").download(root=str(outputs_dir))
    steps = sorted((outputs_dir / "checkpoints").glob("step_*"))
    if not steps:
        raise RuntimeError(f"No checkpoints/step_* under {outputs_dir} after download.")
    logger.info("Downloaded %d checkpoint(s); latest=%s", len(steps), steps[-1].name)

    # 2) Reconstruct inputs.yaml from the run config (via GraphQL, not api.run()).
    src_cfg = _fetch_run_config(args.entity, args.project, args.run_id)
    _reconstruct_inputs_yaml(src_cfg, model_dir)

    # 3) Resume the ORIGINAL W&B run and inject held-out-only metrics into it.
    finetune_config = _build_finetune_config(src_cfg, model_dir)
    wandb_run = wandb.init(
        entity=args.entity, project=args.project, id=args.run_id, resume="must"
    )
    step_offset = int(getattr(wandb_run, "step", 0) or 0) + 1
    logger.info(
        "Re-scoring held-out for run %s (checkpoint_policy=%s n_steps=%s)",
        args.run_id,
        finetune_config["source_run"]["checkpoint_policy"],
        finetune_config["heldout_finetuning"]["n_steps"],
    )
    result = run_heldout_subject_finetuning_from_config(
        finetune_config,
        wandb_run=wandb_run,
        wandb_key_prefix="heldout",
        wandb_step_offset=step_offset,
    )
    wandb_run.finish()
    logger.info("Held-out re-score finished. output_dir=%s", result.get("output_dir"))
    print(json.dumps({"run_id": args.run_id, "output_dir": result.get("output_dir")}))


if __name__ == "__main__":
    main()
