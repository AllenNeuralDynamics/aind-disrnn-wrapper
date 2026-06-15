# AI Hub / Beaker migration — handoff & decision log

Distilled context from the design session, so work can continue inside Code
Ocean. Full technical plan: [`../ai2_migrate_plan.md`](../ai2_migrate_plan.md).
MVP runbook: [`README.md`](README.md).

## Goal

Run the disRNN training stack on the Allen **AI Hub / Beaker** platform,
alongside the existing Code Ocean (CO) and HPC/SLURM paths.

## Architecture decisions

- **Control plane / compute plane split.** The **dispatcher** (running in CO)
  is the control plane that submits jobs; **Beaker** is the GPU compute plane.
  CO stays cheap (CPU), GPUs burst on Beaker.
- **W&B sweep is the orchestration mechanism** (chosen over shipping resolved
  config JSON). The sweep controller lives on W&B cloud; Beaker tasks run
  `wandb agent <SWEEP_ID>`. So **dispatcher → Beaker passes only SWEEP_ID +
  run/agent count** — not a per-job config.
- **Consequence:** sweep agents invoke `python -m run_hpc <overrides>`, so the
  wrapper re-composes Hydra config on Beaker. The **dispatcher's Hydra configs
  must be baked into the wrapper image** (sibling layout) — handled in
  `beaker/Dockerfile`. This is the accepted trade for going sweep-first.
- **Sweeps defined in the dispatcher** (consistent with the CO workflow), unlike
  the HPC path where sweeps live in the wrapper (`sweeps/` + `launch_wandb_sweep.py`).

## GPU facts (from internal `allenOCTO/aihub` docs)

- Clusters are **H200** (training) and **L40s** (debug) — **no small GPUs**.
  L40s is the target for our tiny disRNN runs; H200 is overkill for one run.
- **No fractional GPU scheduling** (`gpuCount` is an integer, whole-GPU exclusive;
  no MPS/MIG documented). Pack many runs per GPU instead:
  - **Time-slicing** (recommended first; no code change): run N `wandb agent`
    processes in one task with `XLA_PYTHON_CLIENT_MEM_FRACTION≈1/N`.
  - **`jax.vmap`** across hyperparameters (max efficiency; needs a pure
    `lax.scan` refactor of the trainer).
- Open question to AI Hub team: whether fractional GPU / MPS / MIG is available
  (a Slack message draft exists; not yet sent).
- Use lowest GPU class, lowest `gpuCount`, `priority: low`, `preemptible: true`,
  `autoResume: true`; checkpoint ≤30 min (SIGTERM → ~30s → requeue).

## MVP scope (compute-plane-first)

Simplest pipeline: **simulate RL agent → fit disRNN with default settings**,
run as a 1-point W&B sweep on one L40s GPU. Artifacts in `beaker/`:
`Dockerfile`, `sweep_mvp.yaml`, `experiment_mvp.yaml`, `README.md` (runbook).

## State (as of this handoff)

- ✅ Migration plan + MVP scaffold written and committed on branch `ai_hub`.
- ✅ **MVP code path validated locally** on an HPC compute node (CPU,
  `disrnn-cpu`): synthetic-RL → disRNN trains and logs (W&B offline). This is the
  exact command a Beaker sweep agent runs.
- ✅ **Fixed a default-config bug** found during validation: `data.batch_size`
  was `null` but `batch_mode: random` requires it → defaulted to `512` in the
  dispatcher's `base.yaml` (committed). `data=synthetic model=disrnn` now runs
  out of the box.
- 🧭 **Build decision: Code Ocean.** HPC compute nodes have no Docker (only
  Apptainer; userns + fakeroot + egress all work, but no OCI builder), so we
  build in CO (Docker + creds) and push to **Beaker's own registry** via
  `beaker image create`. ghcr.io `docker:` ref is the documented fallback.
- ⏳ Image not yet built. We **have** an AI Hub workspace + Beaker token.

## Placeholders to fill before submitting

- `ai1/<workspace>` — our AI Hub workspace name.
- L40s **cluster name** — confirm the live name (`clusters.md` lists
  `ai-hub-aws-uswest-l40s`).
- `<username>/disrnn-wrapper` — image ref from `beaker image get`.
- `<prefix>-wandb-api-key` — Beaker secret name for the W&B key.

## Next steps

1. Build + push the wrapper image; create the W&B secret (runbook steps 1–3).
2. `wandb sweep beaker/sweep_mvp.yaml` → fill SWEEP_ID → submit; verify it
   trains and logs to W&B.
3. Scale: `replicas` (agents across GPUs) and/or in-GPU packing.
4. **Control plane (Layer 3):** add a "submit to Beaker" mode to the dispatcher
   capsule so CO creates the sweep + submits the experiment. **Verify CO→Beaker
   network egress + `BEAKER_TOKEN` secret in CO first** — this is the main
   unproven assumption; if CO lacks egress, fall back to a local launcher that
   reads the same dispatcher configs.
