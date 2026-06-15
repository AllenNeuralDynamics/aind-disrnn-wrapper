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

## Runbook (needs Docker + Beaker CLI + tokens — not available on the HPC box)

```bash
# 0. Prereqs: beaker CLI configured (`beaker config`), GITHUB_TOKEN for private
#    git deps, WANDB key. Replace <workspace>, <username>, <prefix>, <SWEEP_ID>.

# 1. Build the image from the PARENT dir holding both repos as siblings.
cd /path/to/code        # contains aind-disrnn-wrapper/ and aind-disrnn-dispatcher/
DOCKER_BUILDKIT=1 docker build \
  --platform linux/amd64 \
  --secret id=gh_token,env=GITHUB_TOKEN \
  -f aind-disrnn-wrapper/beaker/Dockerfile \
  -t disrnn-wrapper .

# 2. Push to Beaker.
beaker image create --name disrnn-wrapper -w ai1/<workspace> disrnn-wrapper
beaker image get <image-id>      # note the <username>/disrnn-wrapper ref

# 3. Store the W&B key as a Beaker secret.
beaker secret write <prefix>-wandb-api-key -w ai1/<workspace> "$WANDB_API_KEY"

# 4. Create the W&B sweep -> SWEEP_ID.
wandb sweep beaker/sweep_mvp.yaml

# 5. Fill <username>/<SWEEP_ID>/<prefix> in experiment_mvp.yaml, then submit.
beaker experiment create -w ai1/<workspace> beaker/experiment_mvp.yaml
```

## Placeholders to fill

- `ai1/<workspace>` — our AI Hub workspace.
- `ai1/ai-hub-aws-uswest-l40s` — confirm the live L40s cluster name.
- `<username>/disrnn-wrapper` — image ref from `beaker image get`.
- `<prefix>-wandb-api-key` — Beaker secret name.
- `<SWEEP_ID>` — from `wandb sweep`.

## Next (after the MVP runs)

- **Scale**: bump `replicas` (agents across GPUs) and/or pack agents per GPU
  (time-slicing) — see the parallelism section of `../ai2_migrate_plan.md`.
- **Control plane**: add a "submit to Beaker" mode to the dispatcher capsule so
  CO creates the sweep and submits the experiment (verify CO→Beaker egress first).
