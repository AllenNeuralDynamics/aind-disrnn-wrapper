# Running disRNN on AI Hub / Beaker

Run the disRNN training stack on the Allen **AI Hub / Beaker** GPU platform.

**Architecture (two planes):**
- **Build plane — a Mac (or any box with Docker).** Builds the GPU image and
  pushes it to Beaker's registry. Needed rarely (only when dependencies change).
  *Code Ocean can't build* — its capsules are unprivileged Docker containers whose
  seccomp profile blocks the namespace creation every image builder needs.
- **Control plane — Code Ocean (or any box with the `beaker` CLI).** Creates the
  W&B sweep and submits the Beaker experiment. No GPU, no Docker.

**Key idea — code is *not* frozen in the image.** The image bakes only the
environment (Python + JAX + pinned deps). The actual code is pulled fresh from
GitHub at job startup by `entrypoint.sh`, so day-to-day you just **edit → push →
re-run**, never rebuild. See [Controlling the code version](#3-controlling-the-code-version).

## Files

| File | What it is | Rebuild image to change? |
|---|---|---|
| `Dockerfile` | GPU image: clones both repos (sibling layout) + installs deps | **Yes** |
| `entrypoint.sh` | Runtime bootstrap: pulls latest code, then `exec`s the job | **Yes** (baked, runs before the pull) |
| `build_and_push.sh` | Builds (`linux/amd64`) and pushes via `beaker image create` | n/a |
| `smoke.yaml` | Beaker spec: image sanity check (GPU + config, no W&B) | No |
| `sweep_mvp.yaml` | W&B sweep: `data=synthetic model=disrnn` + the `run_hpc` command | No |
| `experiment_mvp.yaml` | Beaker spec: one `wandb agent` on one L40s GPU | No |

## Resolved settings

| | |
|---|---|
| Workspace | `ai1/aind-dynamic-foraging-foundation-model` |
| Cluster (L40s) | `ai1/octo-hub-aws-l40s` |
| Image ref | `han-hou/disrnn-wrapper` (from `beaker image get disrnn-wrapper`) |
| W&B secret | `han-wandb-api-key` (a Beaker secret holding `WANDB_API_KEY`) |

---

## 1. Build & push the image (on a Mac / any Docker box)

Needs only Docker + the Beaker CLI — **no W&B** (runtime only) and **no GitHub
token** (all repos/deps are public; the Dockerfile clones them itself, so there's
no build context to arrange).

```bash
# Docker Desktop running (whale icon steady):  docker --version
# Beaker CLI (once):
curl -fsSL https://beaker.org/install | sh
beaker account login              # browser login with AI1 creds
beaker account whoami             # expect: han-hou

git clone https://github.com/AllenNeuralDynamics/aind-disrnn-wrapper.git
cd aind-disrnn-wrapper && git checkout ai_hub
bash beaker/build_and_push.sh
beaker image get disrnn-wrapper   # note the ref: han-hou/disrnn-wrapper
```

`build_and_push.sh` options (config is via flags, not env vars):

| Flag | Meaning | Default |
|---|---|---|
| `--name NAME` | Beaker image name | `disrnn-wrapper` |
| `--workspace WS` | Beaker workspace | `ai1/aind-dynamic-foraging-foundation-model` |
| `--ref REF` | Branch/tag/SHA baked into **both** repos | `ai_hub` |
| `--wrapper-ref` / `--dispatcher-ref` | Override one repo's baked ref | `ai_hub` |
| `--force-rebuild` | Bust Docker's cache → fresh clone + reinstall | off |
| `--force-override-beaker` | Replace an existing Beaker image (delete only after a successful build) | off |

By default the script **won't touch an existing image** — it warns and stops. A
real dependency-change rebuild is therefore:

```bash
bash beaker/build_and_push.sh --force-rebuild --force-override-beaker
```

> Apple Silicon builds `linux/amd64` under emulation (Beaker nodes are x86) — slower
> but correct; the flag is already set. The image name/ref stays stable across
> rebuilds, so nothing downstream needs editing.

## 2. Run a job (from Code Ocean)

**First, a smoke test** — proves the image works on a node with no W&B/sweep/training
(image pull, runtime code pull, GPU, Hydra config). Fills in `han-hou/disrnn-wrapper`,
then:

```bash
WS=ai1/aind-dynamic-foraging-foundation-model
beaker experiment create -w "$WS" beaker/smoke.yaml
# watch https://beaker.org/ex/<id> for "JAX devices: [Cuda…]" and "SMOKE OK"
```

**Then the real MVP** — a W&B sweep, where each Beaker task runs `wandb agent` and
pulls hyperparameter combos from the W&B controller:

```bash
# W&B key as a Beaker secret (once per workspace):
beaker secret write han-wandb-api-key -w "$WS" "$WANDB_API_KEY"

# Create the sweep -> prints SWEEP_ID (e.g. AIND-disRNN/beaker_mvp/abc123):
wandb sweep beaker/sweep_mvp.yaml

# In experiment_mvp.yaml set the image ref + <SWEEP_ID> + secret name, then submit:
beaker experiment create -w "$WS" beaker/experiment_mvp.yaml
```

Monitor: `beaker experiment get <id>`, the UI at `https://beaker.org/ex/<id>`, or
`beaker experiment tasks <id>` → `beaker job logs <job-id>`. Runs also show in W&B
under `AIND-disRNN/beaker_mvp`.

## 3. Controlling the code version

At container startup `entrypoint.sh` `git fetch`es **both** repos to
`WRAPPER_REF` / `DISPATCHER_REF` (default `ai_hub`), logs the resolved commit
SHAs, then `exec`s the job. The Beaker spec invokes it as:

```yaml
command: [bash, /workspace/aind-disrnn-wrapper/beaker/entrypoint.sh, wandb, agent, ...]
```

Consequences:

- **Iterate without rebuilding:** edit → push to `ai_hub` → submit a new run.
- **Run a different branch / commit:** set the refs in the spec you submit — this
  is the *only* knob that affects what runs:
  ```yaml
  envVars:
    - { name: WRAPPER_REF,    value: my-feature-branch }   # branch, tag, or SHA
    - { name: DISPATCHER_REF, value: my-feature-branch }
  ```
  Omit them and it defaults to `ai_hub`. Use a **SHA** to pin a run for
  reproducibility.
- **Build-time refs don't matter for what runs.** The Dockerfile/`build_and_push.sh`
  refs only seed the baked clone; the runtime pull overwrites it.
- **Rebuild only on dependency changes** (`pyproject.toml` / pinned git deps).
  Symptom: a run fails with `ImportError` / `ModuleNotFoundError` after a pull.

## 4. Where to change what

| Want to change… | Edit | Rebuild image? |
|---|---|---|
| **Dependencies / base environment** | `Dockerfile` (then `pyproject.toml` for the actual deps) | **Yes** — `--force-rebuild --force-override-beaker` |
| **The command, cluster, GPU count, replicas, secret, refs** | the spec YAML (`experiment_mvp.yaml` / `smoke.yaml`) | No |
| **The hyperparameter grid / `run_hpc` overrides** | `sweep_mvp.yaml` (read by `wandb sweep` at submit) | No |
| **The startup/bootstrap logic** (how code is pulled) | `entrypoint.sh` | **Yes** (it's baked and runs before the pull) |
| **Application code / Hydra configs** | the wrapper / dispatcher repos directly | No — push to the ref the job uses |

> Rule of thumb: anything **inside the image** (Dockerfile, entrypoint.sh) needs a
> rebuild; anything in a **spec, sweep, or the app repos** does not.

---

For the broader migration design (HPC/SLURM path, GPU packing, `jax.vmap`
scaling), see [`../ai2_migrate_plan.md`](../ai2_migrate_plan.md).
