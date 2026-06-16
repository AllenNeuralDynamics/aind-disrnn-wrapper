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

## Runbook

The image is **built on a machine with Docker** (a MacBook, in our setup) and
pushed to Beaker's registry. Everything after that — secret, sweep, submit —
runs from the **Code Ocean control plane**, or any box with the `beaker` CLI.

> **Why not build in Code Ocean?** CO capsules run as unprivileged Docker
> containers whose seccomp profile blocks creating user/mount namespaces, so no
> container builder (Docker, rootless Docker, Podman, buildah) can run there. CO
> drives Beaker fine — it just can't build images.
>
> The Dockerfile **git-clones** both repos (`aind-disrnn-dispatcher` +
> `aind-disrnn-wrapper`, default branch `ai_hub`) into a sibling layout inside
> the image. All repos/deps are public, so there's no token and no build context
> to arrange — you can build from anywhere with Docker.

### 1. Build + push the image (on a Mac / any Docker box)

Needs only Docker + the Beaker CLI — **no W&B** (that's runtime only) and **no
GitHub token** (public repos).

```bash
# Docker Desktop running (whale icon steady):
docker --version

# Beaker CLI + login (once):
curl -fsSL https://beaker.org/install | sh
beaker account login             # browser login with AI1 creds
beaker account whoami            # expect: han-hou

# Get the Dockerfile + build script, then build (linux/amd64) + push:
git clone https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper.git
cd aind-disrnn-wrapper && git checkout ai_hub
bash beaker/build_and_push.sh    # builds linux/amd64, then `beaker image create`
beaker image get disrnn-wrapper  # note the ref: han-hou/disrnn-wrapper
```

> **Apple Silicon (M-series):** `build_and_push.sh` already passes
> `--platform linux/amd64` (Beaker nodes are x86); the build runs under emulation,
> so it's slower but correct. Plain equivalent without the script:
> ```bash
> docker build --platform linux/amd64 -f beaker/Dockerfile -t disrnn-wrapper beaker/
> beaker image create --name disrnn-wrapper \
>   --workspace ai1/aind-dynamic-foraging-foundation-model disrnn-wrapper
> ```
> Rebuild only when code/deps change — otherwise reuse the same image ref.

### 2. Submit a run (from Code Ocean, or any box with the beaker CLI)

```bash
WS=ai1/aind-dynamic-foraging-foundation-model

# W&B key as a Beaker secret (once per workspace):
beaker secret write han-wandb-api-key -w "$WS" "$WANDB_API_KEY"

# Create the W&B sweep -> SWEEP_ID:
wandb sweep beaker/sweep_mvp.yaml

# Fill the image ref + SWEEP_ID + secret name into experiment_mvp.yaml, then submit:
beaker experiment create -w "$WS" beaker/experiment_mvp.yaml
```

## Active development — no rebuild on code changes

The image bakes only the **environment** (Python + JAX + pinned deps). The actual
code is **pulled fresh at container startup** by `beaker/entrypoint.sh`, which the
experiment spec invokes before `wandb agent`:

```
command: [bash, /workspace/aind-disrnn-wrapper/beaker/entrypoint.sh, wandb, agent, ...]
```

So the loop while iterating is just:

> **edit → push to `ai_hub` → trigger a new run.** No Mac, no rebuild.

`entrypoint.sh` `git fetch`es both repos to `WRAPPER_REF` / `DISPATCHER_REF`
(default `ai_hub`) and logs the resolved commit SHAs at the top of the job log.

**Rebuild the image only when dependencies change** (`pyproject.toml` or the
pinned git deps). Symptom of a needed rebuild: a run fails with
`ImportError` / `ModuleNotFoundError` after a pull. To rebuild over the existing
image, re-run the build with `--force-rebuild` (it replaces the old image only
after the new build succeeds); otherwise the script stops rather than touch it.

**Pin a run for reproducibility** by passing a commit SHA instead of the branch:
set `WRAPPER_REF` / `DISPATCHER_REF` to the SHA in `experiment_mvp.yaml` (same
image, exact code).

> One-time: this only works once an image containing `beaker/entrypoint.sh` exists,
> so build once after this change lands on `ai_hub`. After that, code edits never
> need a rebuild.

## Placeholders to fill

- ~~`ai1/<workspace>`~~ — resolved: **`ai1/aind-dynamic-foraging-foundation-model`**.
- ~~`ai1/ai-hub-aws-uswest-l40s`~~ — resolved: **`ai1/octo-hub-aws-l40s`** (L40s).
- `<username>/disrnn-wrapper` — image ref from `beaker image get`.
- `<prefix>-wandb-api-key` — Beaker secret name.
- `<SWEEP_ID>` — from `wandb sweep`.

## Next (after the MVP runs)

- **Scale**: bump `replicas` (agents across GPUs) and/or pack agents per GPU
  (time-slicing) — see the parallelism section of `../ai2_migrate_plan.md`.
- **Control plane**: add a "submit to Beaker" launcher to the dispatcher capsule
  so CO creates the sweep and submits the experiment. CO→Beaker egress + the
  `beaker` CLI/token are **confirmed working in CO** (the launcher is the
  remaining piece).
