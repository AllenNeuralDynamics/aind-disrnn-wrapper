# Migrating the disRNN wrapper to Allen AI Hub (Beaker)

Plan for running the disRNN training stack on the Allen Institute's **AI Hub /
Beaker** platform, migrating off the current HPC/SLURM path. All facts below
come from the internal docs in `allenOCTO/aihub` (`docs/`):

- Hello world (Docker + spec workflow): `getting-started/your_helloworld.md`
- Clusters & GPU types: `getting-started/clusters.md`
- GPU best practices / priority / preemption: `data-science/gpu-usage.md`
- S3 access: `tutorials/tutorial_s3.md`
- FSx/Lustre workspace storage: `tutorials/howto_workspace_fsx.md`

## TL;DR

- Submission path: **build a Docker image → `beaker image create` → write an
  experiment-spec YAML → `beaker experiment create -w <workspace>`.** We already
  maintain `environment/Dockerfile`, so this fits.
- Every submit needs a **workspace** (`-w ai1/<workspace>`) — drives budget
  attribution; dormant workspaces get paused.
- GPUs are **large (H200; L40s for debugging)** — our tiny disRNN runs under-fill
  them, so **packing many runs onto one GPU is essential** (below).
- Make runs **low-priority, preemptible, and checkpoint-resumable**.
- Fix the **sibling-repo config dependency** first.

## Clusters / GPU types (`clusters.md`)

| Cluster | Hardware | Best for |
|---|---|---|
| `octo.hub-aws-p5en` | NVIDIA H200 | Large training |
| `ai-hub-gcp-uswest-h200`, `cit-gpu-onprem-h200` | H200 | (rolling out 2026) |
| `ai-hub-aws-uswest-l40s` | NVIDIA L40s | Interactive debugging |

> Cluster names change as nodes come up — re-check if you get an error.

Per `gpu-usage.md`, request the **smallest GPU class that works** (prefer L40s
for our small models), the **lowest `gpuCount`**, and the **lowest priority**
(Low/Normal; High is audited). Prefer **`aws`-named clusters** to colocate with
our S3 data and avoid egress.

## GPU partitioning & parallelism

**No fractional GPUs.** `resources.gpuCount` is an integer and Beaker allocates
**whole GPUs exclusively** — it won't co-schedule two jobs on one GPU
(`gpu-usage.md` documents whole-GPU allocation; no MPS/MIG/fraction support is
mentioned).

So there are two ways to parallelize:

**Across many GPUs** — `replicas: N` in the task spec runs N copies on N GPUs at
once. The W&B sweep shards naturally: each replica runs its own `wandb agent`
pulling from the same sweep queue (subject to the per-experiment GPU cap AI Hub
sets for us).

**Many runs on one GPU** — done **inside a single task**, since the GPUs are far
bigger than one disRNN run needs. Two options, with very different effort:

| | **(a) Time-slicing — multi-process** | **(b) `jax.vmap`** |
|---|---|---|
| Code change | **None** — training code untouched | Refactor trainer to a pure `lax.scan` |
| How | Launch N `wandb agent` procs on one GPU | Batch N runs into one compiled call |
| Efficiency | Good (fills idle gaps); context-switch overhead, no true overlap w/o MPS | Best — true single-process parallelism |
| Covers arch sweeps? | Yes (each proc independent) | No — only continuous knobs; arch needs an outer loop |
| Portable to HPC | Yes | Yes |

**Recommended first move: (a) time-slicing** — near-zero engineering, fully
portable. Reach for `vmap` later only if you need to truly saturate an H200.

Both techniques are plain JAX/CUDA, so they work identically on our HPC SLURM
cluster too — implement once, use on both.

### (a) Time-slicing: multiple processes, one GPU (no code change)

Zero change to `run_hpc.py` / `DisrnnTrainer` / sweep configs. You only (1) cap
JAX memory per process and (2) launch N `wandb agent` processes in the
background — each an independent agent pulling its own combos from the same
sweep, exactly as today, just N at once on one GPU.

JAX preallocates ~75% of the GPU by default, so the 2nd process OOMs. Cap each
to a slice (`MEM_FRACTION ≈ 1/N`):

```bash
# job/pack_gpu.sh — run N agents packed onto one GPU (time-sliced)
# Usage: pack_gpu.sh <SWEEP_ID> <N_PROCS> [RUNS_PER_PROC]
set -euo pipefail
SWEEP_ID="$1"; N_PROCS="${2:-10}"; RUNS_PER_PROC="${3:-5}"

# Each process reserves ~1/N of GPU memory (0.95 leaves CUDA-context headroom).
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(python -c "print(round(0.95/$N_PROCS, 3))")

pids=()
for i in $(seq 1 "$N_PROCS"); do
    wandb agent --count "$RUNS_PER_PROC" "$SWEEP_ID" &
    pids+=($!); sleep 2          # stagger init so they don't all compile at once
done
fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
exit $fail
```

On Beaker, point the task command at it (no other spec change):
```yaml
command: [bash, job/pack_gpu.sh, "<SWEEP_ID>", "10", "5"]
envVars:
  - name: WANDB_API_KEY
    secret: <prefix>-wandb-api-key
resources:
  gpuCount: 1
```

On our current SLURM, swap the last line of `job/wandb_sweep_gpu.slurm`:
```bash
# was: wandb agent --count "$AGENT_COUNT" "$SWEEP_ID"
bash "$WRAPPER_ROOT/job/pack_gpu.sh" "$SWEEP_ID" 8 5
```

**Tuning N**: start at ~8–10, watch `nvidia-smi`. Util→100% & memory fits → push
higher; OOM → lower N/`MEM_FRACTION`; util stays low → CPU/data-loading bound,
add `cpuCount` instead. Tiny disRNN kernels leave the GPU idle between launches,
so time-slicing fills the gaps and N can go high before memory is the limit.

### (b) `jax.vmap` across hyperparameters (max efficiency, needs refactor)

`vmap` vectorizes a **pure function over an array axis** — all runs must share
identical shapes and have no per-run Python branching. So:

- ✅ **Vmappable** (scalars flowing into the loss/optimizer): `lr`,
  penalties/`beta` and the four `*_penalty` terms, `loss_param`, and the **seed**
  (vmap over PRNG keys).
- ❌ **Not vmappable**: architecture knobs that change tensor shapes
  (`*_n_units_per_layer`, `*_n_layers`) — loop over those (or separate runs) and
  vmap the continuous knobs within each.

Two stack-specific gotchas:
1. `rnn_utils.train_network` isn't vmap-friendly (Python loops, logging,
   plotting) — reimplement the core step as a pure `lax.scan` over steps.
2. `optax.adam(lr)` bakes `lr` in at construction; use
   `optax.inject_hyperparams(optax.adam)(learning_rate=lr)` so `lr` is traceable.

Pattern:
```python
import jax, jax.numpy as jnp, optax, haiku as hk, itertools

def make_train_one(xs, ys, n_steps, model_fn):
    """train_one(key, hparams) -> (final_loss, params). xs/ys closed over (const)."""
    net = hk.transform(model_fn)

    def loss_fn(params, key, beta):
        logits = net.apply(params, key, xs)
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, ys).mean()
        l2 = sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))
        return nll + beta * l2                      # penalty strength is vmapped

    def train_one(key, hparams):
        lr, beta = hparams["lr"], hparams["beta"]
        init_key, train_key = jax.random.split(key)
        params = net.init(init_key, xs)
        opt = optax.inject_hyperparams(optax.adam)(learning_rate=lr)  # lr vmappable
        opt_state = opt.init(params)

        def step(carry, step_key):
            params, opt_state = carry
            loss, grads = jax.value_and_grad(loss_fn)(params, step_key, beta)
            updates, opt_state = opt.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), loss

        keys = jax.random.split(train_key, n_steps)
        (params, _), losses = jax.lax.scan(step, (params, opt_state), keys)
        return losses[-1], params
    return train_one

# Build the grid as batched arrays, then vmap the whole thing in one compiled call
combos = list(itertools.product([1e-3, 3e-3, 1e-2], [1e-3, 1e-2], range(5)))  # lr × beta × seed
hparams = {"lr": jnp.array([c[0] for c in combos]),
           "beta": jnp.array([c[1] for c in combos])}
keys = jax.vmap(jax.random.PRNGKey)(jnp.array([c[2] for c in combos]))

train_one = make_train_one(train_xs, train_ys, n_steps=5000, model_fn=my_model)
final_losses, params = jax.vmap(train_one, in_axes=(0, {"lr": 0, "beta": 0}))(keys, hparams)
# final_losses.shape == (len(combos),) — every combo trained in parallel on one GPU
```

Mapping to `DisrnnTrainer`: split into an **outer Python loop over architecture
combos** (each builds its own `hk.transform`) and an **inner `vmap`** over
`lr` / penalties / `loss_param` / seed. The warmup phase (penalties zeroed)
becomes a first `lax.scan` with `beta=0` inside `train_one`. Drop per-step W&B
logging inside the scan (it's pure) — log the returned arrays after the call.
Memory scales with the number of combos, so chunk huge grids with `jax.lax.map`
or by slicing the combo arrays to fit the GPU.

## Submission workflow (`your_helloworld.md`, `tutorial_s3.md`)

```bash
# 1. Build & push the image (GPU variant of environment/Dockerfile)
docker build --platform linux/amd64 -t disrnn-wrapper environment/
beaker image create --name disrnn-wrapper -w ai1/<workspace> disrnn-wrapper

# 2. Store secrets once (literal AWS creds are blocked — must be secrets)
beaker secret write <prefix>-wandb-api-key -w ai1/<workspace> "$WANDB_API_KEY"

# 3. Launch an experiment spec
beaker experiment create -w ai1/<workspace> experiment.yaml
```

Minimal `experiment.yaml` shape (from the hello-world + GPU best-practice docs):
```yaml
version: v2
description: disRNN synthetic smoke test
tasks:
  - name: train
    image:
      beaker: <username>/disrnn-wrapper      # not the ai1/<workspace> path
    command: [python, -u, -m, run_hpc, -m, wandb.project=beaker_test, data=synthetic]
    context:
      priority: low
      preemptible: true
      autoResume: true
    constraints:
      cluster:
        - ai1/ai-hub-aws-uswest-l40s         # smallest GPU class
    resources:
      gpuCount: 1
    envVars:
      - name: WANDB_API_KEY
        secret: <prefix>-wandb-api-key
    result:
      path: /output
```

## Storage (`tutorial_s3.md`, `howto_workspace_fsx.md`)

- **Results dataset**: write to `result.path` (e.g. `/output`); Beaker saves it
  as a downloadable dataset (`beaker experiment results <id>`).
- **S3 via boto3**: store AWS creds as Beaker secrets (literals are blocked),
  inject as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`; `boto3.client("s3")`
  picks them up. Natural fit for our AIND data loaders. Use an `aws` cluster.
- **FSx/Lustre** (preview): fast POSIX scratch + persistent project folders
  backed by S3, on `aws` clusters, **shared/project workspaces only**. Temporary
  (may be purged) and currently institute-wide readable — copy keepers elsewhere.

## Preemption & checkpointing (`gpu-usage.md`)

Default for background work (sweeps): `priority: low`, `preemptible: true`,
`autoResume: true`. On preemption a job gets **SIGTERM, ~30s to save, then
requeue + restart**. So our code must handle SIGTERM, checkpoint at least every
30 min, and resume from a checkpoint. Our runs are short (~5k steps) but sweeps
should still be resumable so an interrupted agent picks up cleanly.

## ⚠️ Gotcha: sibling-repo config dependency

`code/run_hpc.py` hard-codes
`config_path="../../aind-disrnn-dispatcher/code/config"` and reads the
dispatcher's Hydra configs as a filesystem sibling. In a Docker image we control
the layout, so make the configs available in the image. Options, best first:

1. **Package the dispatcher configs as a pip dependency** and resolve via
   `importlib.resources` instead of a relative path — decouples us from the
   sibling layout; works the same on CO / HPC / Beaker. **Do this first.**
2. **Install/copy the dispatcher into the image** — our `Dockerfile` already
   `pip install`s `git+https://…` deps; add the dispatcher the same way (or
   `COPY` its `code/config` to the expected path). Keeps the brittle coupling.
3. **Vendor a copy** into this repo — loses single-source-of-truth; avoid.

## Migration steps

1. **Decouple configs** (gotcha option 1).
2. **Beaker access**: install CLI, `beaker config`, confirm our workspace
   (`beaker workspace list ai1`) and the cluster(s)/budget AI Hub assigned.
3. **Image**: GPU variant of `environment/Dockerfile` → `beaker image create`.
4. **Secrets**: `WANDB_API_KEY`, AWS creds for data loaders (+ git token if the
   image installs private repos).
5. **Smoke-test** one synthetic run via an experiment spec (above).
6. **GPU packing**: implement `vmap` and/or multi-process so one GPU runs many
   trainings (portable to HPC).
7. **Port the sweep launcher**: adapt `code/launch_wandb_sweep.py` to emit
   `beaker experiment create` (templated spec, `replicas` = old array size)
   instead of `sbatch`; keep the lineage injection + auto agent-count logic.
8. **Storage decision**: S3 (boto3 + secrets) for data + outputs; `/output`
   and/or W&B for artifacts.

## Open questions for AI Hub

- Which **workspace, cluster, budget, and per-experiment GPU cap** are ours.
- Whether **NVIDIA MPS** is enabled (affects single-GPU packing efficiency, not
  feasibility).
- Whether an **L40s cluster** is live now (June 2026) for our small-model runs.
