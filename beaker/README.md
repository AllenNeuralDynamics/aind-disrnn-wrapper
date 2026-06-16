# Beaker (AI Hub) MVP

Run the disRNN wrapper on the Allen AI Hub / Beaker platform. This MVP proves
the **compute plane**: a single default *synthetic-RL → disRNN* run, executed as
a W&B sweep agent on one GPU. See [`../ai2_migrate_plan.md`](../ai2_migrate_plan.md)
for the full migration design and [`HANDOFF.md`](HANDOFF.md) for the decision log.

## Files

| File | What it is |
|---|---|
| `Dockerfile` | GPU image: wrapper + dispatcher configs in sibling layout (so sweep agents re-run Hydra). |
| `sweep_mvp.yaml` | Trivial W&B sweep: `data=synthetic model=disrnn` (default settings). |
| `experiment_mvp.yaml` | Beaker spec: one `wandb agent` on one L40s GPU. |

## Mechanism

W&B sweep is the orchestrator. The sweep controller (W&B cloud) holds the
hyperparameter space; each Beaker task runs `wandb agent <SWEEP_ID>` and pulls
combos. So **Beaker tasks only need SWEEP_ID + a run count** — no config JSON is
shipped from the dispatcher. Each agent invokes `python -m run_hpc <overrides>`,
which re-composes the Hydra config from the dispatcher configs baked into the image.

## Image source

Beaker accepts two image sources:
- **Beaker's own registry (used here):** `beaker image create` uploads from a
  local Docker daemon; referenced as `image: { beaker: <username>/disrnn-wrapper }`.
  Simplest — no external registry or pull credentials.
- **External registry** (ghcr.io / Docker Hub): `image: { docker: ghcr.io/... }`,
  Beaker pulls it (private needs pull creds in the workspace). Documented in
  `experiment_mvp.yaml` as a fallback for builds done without the beaker CLI.

## Runbook — build in Code Ocean (has Docker + creds; the HPC box has neither)

> The Dockerfile **git-clones** both repos (`aind-disrnn-dispatcher` +
> `aind-disrnn-wrapper`, default branch `ai_hub`) into a sibling layout inside
> the image. All repos/deps are public, so there's no token and no build context
> to arrange — you can build from anywhere.

```bash
# 0. Prereqs: beaker CLI configured (`beaker config` — check `beaker account whoami`)
#    and your W&B key. No GitHub token needed (public repos).

# 1. Build + push in one step (builds x86, `beaker image create`).
WS=ai1/aind-dynamic-foraging-foundation-model   # our AI Hub workspace (build_and_push.sh defaults to it)
bash beaker/build_and_push.sh
beaker image get disrnn-wrapper      # note the <username>/disrnn-wrapper ref it prints

# 2. Store the W&B key as a Beaker secret.
beaker secret write <prefix>-wandb-api-key -w "$WS" "$WANDB_API_KEY"

# 3. Create the W&B sweep -> SWEEP_ID.
wandb sweep beaker/sweep_mvp.yaml

# 4. Fill <username>/<SWEEP_ID>/<prefix> in experiment_mvp.yaml, then submit.
beaker experiment create -w "$WS" beaker/experiment_mvp.yaml
```

## Placeholders to fill

- ~~`ai1/<workspace>`~~ — resolved: **`ai1/aind-dynamic-foraging-foundation-model`**.
- ~~`ai1/ai-hub-aws-uswest-l40s`~~ — resolved: **`ai1/octo-hub-aws-l40s`** (L40s).
- `<username>/disrnn-wrapper` — image ref from `beaker image get`.
- `<prefix>-wandb-api-key` — Beaker secret name.
- `<SWEEP_ID>` — from `wandb sweep`.

## Next (after the MVP runs)

- **Scale**: bump `replicas` (agents across GPUs) and/or pack agents per GPU
  (time-slicing) — see the parallelism section of `../ai2_migrate_plan.md`.
- **Control plane**: add a "submit to Beaker" mode to the dispatcher capsule so
  CO creates the sweep and submits the experiment (verify CO→Beaker egress first).
