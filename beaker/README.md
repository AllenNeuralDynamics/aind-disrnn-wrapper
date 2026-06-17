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

## Migration status

Where the CO → Beaker migration stands (this section is the running log).

**Done & validated on Beaker (cluster `ai1/octo-hub-aws-l40s`, workspace
`ai1/aind-dynamic-foraging-foundation-model`):**

1. **Beaker access** — CLI installed (via `environment/postInstall` in the dispatcher),
   `BEAKER_TOKEN` auth confirmed; workspace + cluster reachable from Code Ocean.
2. **Image build** — `Dockerfile` git-clones both repos + installs deps; built on a
   Mac (`build_and_push.sh`, `--platform linux/amd64`) and pushed to Beaker's registry
   as `han-hou/disrnn-wrapper`. CO can't build (seccomp), so the Mac is the build box.
3. **Runtime code-pull** — `entrypoint.sh` git-fetches both repos to
   `WRAPPER_REF`/`DISPATCHER_REF` at job start → **code edits need no rebuild**.
4. **Smoke test** — `smoke.yaml` confirmed image pull + runtime pull + GPU visible +
   Hydra config composition, no W&B.
5. **Control plane** — the dispatcher's `launch_beaker.py` creates a W&B sweep, saves a
   reproducibility record to `/results`, and submits the Beaker experiment (the
   dispatcher → wrapper hand-off, mirroring CO). A 1-point MVP run succeeded.
6. **Reproducibility** — each run stamps `wrapper_commit` / `dispatcher_commit` /
   `CO_COMPUTATION_ID` into its W&B config; the dispatcher saves the sweep YAML + IDs +
   commit to `/results`.
7. **Scale-out (array of jobs)** — `replicas: 4` validated: 4 `wandb agent`s across 4
   GPUs sharing one sweep (`experiment_scaling.yaml`).

**Observed:** a single run shows **100% GPU util but only ~30% power** on the L40s
(vs ~55% on a T4 in CO). Per the W&B system metrics: util 96%, power ~30% (106/350 W),
memory-bandwidth util ~1%, host CPU ~4%. So the workload is **low-occupancy /
host-eval-bound, not compute- or CPU-bound** (≈0.46 s/step for a tiny `latent_size=5`
model with `eval_every_n=2`).

8. **GPU packing (time-slicing) — tested, no gain.** Packed M `wandb agent`s on one
   L40s (`pack_gpu.sh`, `XLA_PYTHON_CLIENT_MEM_FRACTION≈0.9/M`) and measured throughput:

   | M (agents/GPU) | per-run elapsed | aggregate throughput vs M=1 |
   |---|---|---|
   | 1 | 95 s | 1.00× |
   | 4 | 335 s | 1.14× |
   | 8 | 661 s | 1.15× |

   Per-run latency scales ~linearly with M, so throughput **plateaus at ~1.15×**
   (M-independent). Cause: **100% util means low-occupancy kernels are resident with no
   idle gaps**, so without **MPS** the packed kernels just **serialize** on the one GPU
   context (board power stays ~30%). Not CPU- or memory-bound. Conclusion: **no-MPS
   packing is a dead end for this workload.**

9. **L40s vs CO-T4 (apples-to-apples, full config, seed 0).** L40s is **~1.9× faster
   per run** (train **0.243 vs 0.460 s/step**; 1352 vs 2549 s total). But it's only modest
   for a ~4–5× bigger card — host/eval-bound on both (util ~pinned, power 30% L40s /
   57% T4, mem-BW ~1%) — and the L40s draws **~1.4× more energy/run** (105 W vs 40 W).
   So L40s wins on wall-clock, T4 on energy; neither is compute-bound.

**Next (per-GPU efficiency lever, not packing):** the headroom is *intra-kernel*
(occupancy), reclaimable only by **`jax.vmap`** (batch runs into one fatter kernel) or
**MPS** (concurrent kernels) — plus cutting **`eval_every_n=2`**, a likely free win.
For *scale*, use **`replicas`** across GPUs (validated, linear).

## Files

| File | What it is | Rebuild image to change? |
|---|---|---|
| `Dockerfile` | GPU image: clones both repos (sibling layout) + installs deps | **Yes** |
| `entrypoint.sh` | Runtime bootstrap: pulls latest code, then `exec`s the job | **Yes** (baked, runs before the pull) |
| `build_and_push.sh` | Builds (`linux/amd64`) and pushes via `beaker image create` | n/a |
| `smoke.yaml` | Beaker spec: image sanity check (GPU + config, no W&B) | No |
| `pack_gpu.sh` | Time-slicing: pack M `wandb agent`s onto one GPU | No (pulled at runtime, like the app code) |

> The sweep definition and the production Beaker job spec live in the **dispatcher**
> (control plane), not here — `aind-disrnn-dispatcher/code/beaker/`
> (`sweep_mvp.yaml`, `experiment_mvp.yaml`). This repo only builds the image and
> ships `smoke.yaml` to test it.

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

## 2. Test the image (smoke test)

After building, prove the image works on a node — no W&B/sweep/training, just
image pull, runtime code pull, GPU, and Hydra config composition:

```bash
WS=ai1/aind-dynamic-foraging-foundation-model
beaker experiment create -w "$WS" beaker/smoke.yaml
# watch https://beaker.org/ex/<id> for "JAX devices: [Cuda…]" and "SMOKE OK"
```

**To actually run training** (the W&B-sweep MVP), use the **dispatcher** control
plane — `wandb sweep` → submit `experiment_mvp.yaml` — documented in
`aind-disrnn-dispatcher/code/beaker/README.md`.

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
| **The command, cluster, GPU count, replicas, secret, refs** | the spec YAML — `smoke.yaml` (here) or `experiment_mvp.yaml` (dispatcher) | No |
| **The hyperparameter grid / `run_hpc` overrides** | `sweep_mvp.yaml` in the dispatcher (read by `wandb sweep` at submit) | No |
| **The startup/bootstrap logic** (how code is pulled) | `entrypoint.sh` | **Yes** (it's baked and runs before the pull) |
| **Application code / Hydra configs** | the wrapper / dispatcher repos directly | No — push to the ref the job uses |

> Rule of thumb: anything **inside the image** (Dockerfile, entrypoint.sh) needs a
> rebuild; anything in a **spec, sweep, or the app repos** does not.

---

For the broader migration design (HPC/SLURM path, GPU packing, `jax.vmap`
scaling), see [`../ai2_migrate_plan.md`](../ai2_migrate_plan.md).
