# Training Codebase Guide

> **Living document.** This is the single reference for what the training codebase
> includes and how to use it. **When you add or change a feature, update this file**
> (and add a dated entry to the [Changelog](#changelog) at the bottom). Keep the
> inline comments in `configs/*.yaml` as the authoritative per-key reference; this
> guide explains the concepts, workflows, and how the pieces fit together.

Scope: everything under `code/` that trains GRU / disRNN / baseline-RL models on
mouse foraging behavior (and synthetic agents), evaluates them, and runs
post-training analysis. For the post-training *analysis* entrypoints specifically,
see also the repo-root [README.md](../README.md).

---

## 1. Overview

The pipeline is a small, composable stack wired together by Hydra config:

```
config.yaml ──► run_capsule.main()
                  │
                  ├─ instantiate(data)   ──► DatasetLoader.load() ──► DatasetBundle
                  │                                                    (raw df, train_set, eval_set, metadata)
                  ├─ instantiate(model)  ──► ModelTrainer.fit(bundle) ──► trained params + metrics
                  │                                                       (+ checkpoints, plots, W&B)
                  └─ held-out evaluation (single-subject) / auto fine-tuning (multisubject)
```

- **Data loaders** (`data_loaders/`) turn a data source (foraging database, docDB,
  or synthetic agents) into a `DatasetBundle` with train/eval splits.
- **Model trainers** (`model_trainers/`) fit a model and own the training loop,
  checkpointing, evaluation, plotting, and W&B logging.
- **Models** (`models/`) are the JAX/Haiku network definitions (GRU, multisubject
  disRNN, session-conditioning, subject embeddings).
- **`run_capsule.py`** is the orchestrator: load config → load data → train →
  evaluate held-out → log to Weights & Biases.
- **`post_training_analysis/`** holds analysis run *after* training
  (generative rollouts, embedding space, likelihood comparisons, held-out
  fine-tuning).

Everything is **JAX/Haiku** built on the upstream `disentangled_rnns` library and
the `aind_disrnn_utils` data-loader package.

---

## 1.5 Run lifecycle & key switches (read first)

> Two independent sessions once misread the log line *"Skipping held-out evaluation
> for multisubject …"* as *"no held-out numbers at all."* It only means the
> **per-checkpoint** held-out eval is off — the **end-of-training held-out
> fine-tune still runs**. This section exists so that never happens again. If a
> log message and this section disagree, trust the code path named here.

### The four phases of a run (in order)
1. **Warmup** — `n_warmup_steps` of penalty ramp-up (disRNN) / session-curriculum
   warmup. Loss is high and bottlenecks are held open here; this is *not* the model
   failing. **The W&B `_step` axis includes the warmup offset**, so main-phase
   progress is `(_step − n_warmup_steps) / n_steps`, not `_step / n_steps`.
2. **Main training** — the `while steps_completed < n_steps` loop. Checkpoints are
   written every `checkpoint_every_n_steps` (chunked path only — see resumability
   below).
3. **Artifact upload** — after the main loop, the *entire* `output_dir` (params +
   `train_state.pkl` + plots) is uploaded to W&B as `<mtype>-output-<run_id>`
   (type `training-output`). **This happens once, at the end — not per checkpoint.**
   A run that hasn't finished has no restorable artifact yet.
4. **Held-out fine-tune** (multisubject GRU/disRNN, on by default) — fine-tune a
   fresh subject embedding on each reserved held-out mouse, predict its other
   sessions, log `heldout/*`. This keeps the job in `running`/`idle` briefly after
   training + upload are already done.

### The two DIFFERENT held-out switches (the source of the confusion)
| Switch | What it controls | Default | Skipping it means |
|---|---|---|---|
| `model.training.checkpoint_run_heldout_eval` | **per-checkpoint** held-out eval *during* training (expensive) | often `false` | only the mid-training held-out curve is off |
| `model.training.auto_heldout_finetune.enabled` | the **end-of-training** held-out fine-tune+test (the real figure of merit) | **`true`** | **NO `heldout/*` metrics at all** |

These are **not the same knob.** The *"Skipping held-out evaluation … seen-subject
personalization only"* log line refers to the **first** (per-checkpoint eval); the
second still runs and produces `heldout/final/eval_likelihood`. Full detail in §8.

- **Held-out-mouse likelihood is the correct figure of merit** for generalization
  questions; within-subject `checkpoint/eval_likelihood` **saturates** (~0.72–0.75
  across model sizes) and cannot discriminate. Read `heldout/final/eval_likelihood`.

### Checkpoints, resumability, and extendability (three distinct things)
- **Checkpointing** — only the *chunked* training path writes resumable state, so
  it requires `checkpoint_every_n_steps > 0`. Each checkpoint writes params +
  `train_state.pkl` (optimizer state, `steps_completed`, rng) under
  `output_dir/checkpoints/step_*`.
- **Resumability = WITHIN one experiment (preemption recovery).** A preempted task
  restarts as the *same* task with the *same* `/results` dataset (anchored by
  `DISRNN_RESUMABLE_OUTPUT_DIR=/results/run`). On start, `find_latest_resumable_state`
  loads the newest checkpoint and the loop continues from `steps_completed`,
  **skipping warmup** (warmup is already folded into the checkpointed params).
  Gated by `auto_resume` (default `true`) + `checkpoint_every_n_steps > 0`; W&B
  continuity via a deterministic `WANDB_RUN_ID` + `WANDB_RESUME=allow`. Automatic,
  no flags needed.
- **Extendability = ACROSS experiments (staged horizons).** A *new* experiment gets
  a *fresh, empty* `/results`, so it does **not** see the old run's checkpoints and
  would restart from scratch — **unless** you set
  `model.training.restore_from_run_id=<source W&B run name>` (or env
  `DISRNN_RESTORE_FROM_RUN_ID`). That downloads the source run's
  `<mtype>-output-<run_id>:latest` artifact into the new run's `outputs/` so resume
  finds the checkpoint and continues (skipping warmup). Set the new `n_steps`
  **larger** than the source's. Prerequisite: the source run must have **finished**
  (phase 3 above) so its artifact is `COMMITTED`. Trainer-agnostic (GRU + disRNN
  upload the same artifact shape). Helper:
  `run_helpers.maybe_restore_checkpoint_from_wandb`, called from `run_hpc.py`.

### Other switches that surprise people
- **`length_bucketing`** (+ `length_bucket_grid`, default 128): trims each
  `random`-mode batch's RNN unroll to the batch's own session length instead of the
  global `T_max`. Big speedup (~1.86× measured on 100-mice disRNN). Requires
  `batch_mode=random`; no-op under `single`/full-batch. Runs in the **training**
  loop (both trainers), *not* rollout/generation. The disRNN/GRU configs must
  **declare** the key (Hydra struct mode rejects overriding an absent key).
- **disRNN has NO early stopping.** Only `gru_trainer` has opt-in loss-based early
  stopping. A disRNN sweep must **omit** `early_stopping` keys or Hydra errors.
- **Bottleneck sigma convention:** small σ = **OPEN** (info flows), σ→1 = **CLOSED**.
  So openness *decreases* as σ→1 (sparser). At init all bottlenecks are open.
- **Reading the sparsity metrics — mind the threshold.** A single hard-threshold
  `frac_open` is **misleading**: `bottlenecks/<fam>_frac_open` (σ<0.1) *saturates* to
  0 while channels sit at σ≈0.85 (mostly-but-not-fully closed), whereas the dashboard
  `plot_bottlenecks` figure calls a latent "open" at the far more permissive σ<0.97
  (it logs `open latents by bottleneck > 0.03`, i.e. 1−σ>0.03) — so the same run can
  read `frac_open=0.0` in the scalar and "almost all open" in the figure. They are not
  contradictory; they are different cutoffs on the same σ distribution. **Prefer the
  threshold-free readouts** logged alongside: `bottlenecks/<fam>_n_eff_open_frac` (the
  normalized participation ratio of channel openness `1−σ`, in [1/n,1], **comparable
  across families of different size** — 5-element latent vs the matrix-shaped
  update-net bottleneck), plus `sigma_median` / `sigma_p10` / `sigma_p90` (catch a
  bimodal open/closed split) and `total_openness` (=Σ(1−σ), raw capacity). `frac_open`
  is kept as a multi-threshold curve (`_frac_open_s0p03/0p1/0p5/0p9/0p97`) so both the
  strict and figure conventions are visible.

---

## 2. Repository layout

| Path | What it is |
|------|------------|
| `run_capsule.py` | **Main entry point** — train + evaluate one run from a config |
| `run` | Code Ocean launcher (`python -u run_capsule.py "$@"`) |
| `configs/` | Hydra config templates (`config_gru.yaml`, `config_disrnn.yaml`, `config_baseline_rl.yaml`, `config_heldout_subject_finetuning.yaml`) — every key is documented inline |
| `base/` | `interfaces.py` (`DatasetLoader`, `ModelTrainer` ABCs), `types.py` (`DatasetBundle`) |
| `data_loaders/` | `mice.py` (database + docDB loaders), `synthetic.py` (synthetic agents / task-trained RNN) |
| `models/` | `gru_network.py`, `multisubject_disrnn.py`, `session_conditioning.py`, `subject_embedding_initialization.py` |
| `model_trainers/` | `base_multisubject_trainer.py` (shared base), `gru_trainer.py`, `disrnn_trainer.py`, `baseline_rl_trainer.py` |
| `utils/` | `run_helpers.py`, `multisubject.py`, `session_regularized_training.py`, `load_mice_database.py` (the `*_evaluation.py` / `disrnn_plotting.py` here are now deprecated re-export shims → `evaluation/`) |
| `evaluation/` | Shared evaluation primitives used by **both** training-time held-out eval and post-training analysis: `heldout_eval_config.py`, `common.py`, `{disrnn,gru,baseline_rl}_evaluation.py`, `plotting.py` |
| `post_training_analysis/` | `generative_analysis.py`, `embedding_space_analysis.py`, `likelihood_comparison.py`, `likelihood_advantage_analysis.py`, `baseline_rl_analysis.py`, `heldout_finetuning.py` — see **[POST_TRAINING_ANALYSIS.md](POST_TRAINING_ANALYSIS.md)** |
| `run_analysis.py` | **Unified post-training-analysis CLI** (see [POST_TRAINING_ANALYSIS.md](POST_TRAINING_ANALYSIS.md)) |
| `run_heldout_subject_finetuning.py`, `run_embedding_space_analysis.py` | Older single-purpose analysis CLIs (superseded by `run_analysis.py`) |
| `run_capsule-test_*.py` | **Deprecated** scratch scripts — now stubs pointing at `run_analysis.py` |
| `tests/` | `unittest` suites (see [§9 Testing](#9-testing)) |
| `load_mice_data.py` | Pull/cache mouse data snapshots from the database |

---

## 3. Running a training job

### In Code Ocean (capsule)
`run` executes `python -u run_capsule.py`. At runtime `run_capsule` finds the
active config via `utils.run_helpers.find_hydra_config()`, which globs
`/data/jobs/**/config.yaml` (injected by the capsule/pipeline). The files in
`configs/` are the **maintained templates** you copy/select from; pick one model
type and one data source by editing the active blocks.

Outputs are written to `/results/` (see [§7 Outputs](#7-outputs--weights--biases)).

### Locally / ad hoc
Point `run_capsule` at a config by placing it under `/data/jobs/<name>/config.yaml`,
or import and drive the trainers directly (see how `tests/` construct
`GruTrainer`/`DisrnnTrainer` and call `.fit(bundle)`).

### Selecting what runs
The `run` script has commented alternates (baseline-RL, fine-tuning, analysis,
data loading). Uncomment the desired line, or call the corresponding
`run_capsule-test_*.py` / `run_*.py` script.

---

## 4. Configuration

A config has four top-level sections; **`configs/config_gru.yaml` and
`configs/config_disrnn.yaml` document every key inline** — treat them as the
reference. High level:

```yaml
data:    # which dataset + how to split it      (type, _target_, selection, eval_every_n, ...)
model:   # architecture + training schedule      (type, _target_, architecture, training, ...)
seed: 43
wandb:   # entity / project / dir / name
job_id: 0
```

- `${...}` are OmegaConf interpolations resolved at load time (e.g. `${seed}`,
  `${.architecture.hidden_size}`, `${.penalties.beta}`).
- Exactly **one** `data:` block and **one** `model:` block are active; the others
  are kept as commented, ready-to-use presets.
- disRNN penalty multipliers (`<name>_multiplier`) are resolved into effective
  penalties before training (`utils.run_helpers.resolve_disrnn_penalties`).
- Multisubject mode auto-appends `_multisubject` to the W&B run name.

---

## 5. Models & trainers

### Model types (`model.type`)
| `type` | `_target_` | Notes |
|--------|-----------|-------|
| `gru` | `model_trainers.gru_trainer.GruTrainer` | GRU baseline; `num_layers` must be 1 |
| `disrnn` | `model_trainers.disrnn_trainer.DisrnnTrainer` | Disentangled RNN with information-bottleneck penalties + warmup phase |
| `baseline_rl` | `model_trainers.baseline_rl_trainer.BaselineRLTrainer` | Classic RL agents (e.g. `ForagerQLearning`) fit by differential evolution |

### Shared base class
`GruTrainer` and `DisrnnTrainer` both subclass **`BaseMultisubjectTrainer`**
(`model_trainers/base_multisubject_trainer.py`), which owns the logic that used to
be duplicated between them: the checkpoint/initialization snapshot pipeline
(`_evaluate_initialization_snapshot`), per-split example plotting
(`_generate_split_examples`), subject-embedding / session-context state-space
plots, media cleanup, W&B held-out logging, and loss plotting. Subclasses set
`_MODEL_LABEL` / `_TRAINER_CONTEXT_NAME` (for log text) and override small hooks
(`_resolve_n_action_logits`, `_add_model_results`, `_plot_examples_for_split`,
`_plot_model_specific_diagnostics`, `_evaluate_heldout_subjects`,
`_snapshot_extra_summary_fields`). **When adding shared behavior, put it on the
base; only model-specific bits belong in the hooks.** `BaselineRLTrainer` is
independent (different fit/analysis).

### Multisubject personalization
With `architecture.multisubject: true`, the model learns a per-subject embedding
(a learned table, `models/subject_embedding_initialization.py`) concatenated into
the network input. disRNN can additionally apply an information bottleneck to the
subject embedding (`use_global_subject_bottleneck`) and per-network subject
penalties. The data loader prepends `subject_index` as input feature 0.

### Session conditioning (multisubject only)
A learned, session-indexed perturbation (a "session-delta" MLP) is added on top of
each subject embedding, gated by a curriculum schedule (0 during pretrain, linear
ramp over warmup, 1 after). Controlled by `architecture.session_encoding_type`
(`none|scalar|fourier`), `session_integration_type`, `session_fourier_k`,
`session_delta_n_layers/hidden_size`, and `session_n_pretrain_steps/n_warmup_steps`
(null → ~30%/~20% of total steps). Optional zero-mean regularization via
`training.lambda_reg_session`. Core math: `models/session_conditioning.py`. When
enabled, input feature 1 is `session_index`.

### disRNN specifics
- **Penalties** (`model.penalties`): KL information-bottleneck weights on the
  latent state, choice-net / update-net inputs, and (multisubject) subject context.
  `beta` is the default for any unset penalty.
- **Warmup**: `training.n_warmup_steps` trains a *noiseless* copy of the network
  with all penalties zeroed (penalty-free warmup) before the penalized phase.
- **Distillation** (`model.distillation`): optional GRU→disRNN knowledge
  distillation — train the disRNN student toward the temperature-softened mean of
  one or more trained GRU teachers. v1: `aggregation=mean_probs`, `loss=kl`.

### baseline_rl specifics
Classic RL agents (`agent_class`, e.g. `ForagerQLearning`) fit by differential
evolution — **one agent fit per subject**, independently (no shared model/embedding).
- **Database (`mice_snapshot`) loading is supported** for both single- and
  multi-subject runs. Multisubject fits one agent per training subject
  (parallelized by `multisubject_subject_workers`) and reports per-subject +
  pooled train/eval-split likelihood (`train_likelihood`/`eval_likelihood`, also
  surfaced as `checkpoint/train_likelihood`/`checkpoint/eval_likelihood` for
  cross-model W&B parity).
- **Held-out subjects** are *re-fit*, not transferred: when
  `model.heldout_refit.enabled` (default on for `config_baseline_rl.yaml`) and
  held-out eval applies, `run_capsule` loads the reserved held-out subjects as a
  multisubject bundle and `BaselineRLTrainer.fit_heldout` fits a fresh agent per
  held-out subject (train/eval split), logging aggregate
  `heldout/train_likelihood` + `heldout/eval_likelihood` into the same W&B run.
  Set `model.heldout_refit.skip_train_fit=true` for held-out-only RL baselines
  that should not run the main training-subject fit.
  (parity with the NN models) and writing artifacts under `outputs/heldout_test/`.
  This replaces the older transfer-based held-out eval (apply trained params to
  held-out), which is retired from the run path.

---

## 6. Data loaders & splits

### Sources (`data.type`)
| `type` | Class | Source |
|--------|-------|--------|
| `mice_snapshot` | `MiceSnapshotDatasetLoader` | Foraging **database** snapshot (the standard path) |
| `mice` | `MiceDatasetLoader` | docDB per-subject query |
| `synthetic_task_trained_rnn` | `TaskTrainedRNNDatasetLoader` | Synthetic task-trained RNN logs |
| (tests) | `SyntheticCognitiveAgents` | Synthetic RL agents — used by the test suite |

### Subject selection (`mice_snapshot`)
Subjects with ≥ `min_sessions` sessions are assigned their most-common task as a
curriculum and ranked per curriculum by session count. A fixed ~20% **held-out**
test set is reserved (every `heldout_every_n`-th in rank order); the training set
is a seeded `subject_ratio` sample of the remaining ~80% per curriculum. Pass an
explicit `subject_ids` list to bypass the pipeline entirely.

### Train/eval split
Within the selected subjects, every `eval_every_n`-th session goes to the eval
split (applied **per subject** in multisubject mode via
`utils.multisubject.compute_train_eval_session_ids`, then merged with
`subject_index` prepended). `ignore_policy` controls no-response trials
(`exclude` → 2 classes, `include` → 3).

---

## 7. Outputs & Weights & Biases

A run writes to `/results/`:
- `/results/inputs.yaml` — the fully resolved config (backup).
- `/results/outputs/params.json` — trained parameters.
- `/results/outputs/{gru,disrnn}_config.json` — resolved model config.
- `/results/outputs/output_summary.json` — final metrics + metadata.
- `/results/outputs/subject_index_map.json`, `session_context_map.json`,
  `multisubject_metadata.json` — multisubject artifacts (required by downstream
  analysis / fine-tuning).
- `/results/outputs/checkpoints/` — per-checkpoint params + `index.json`.
- figures + (optional) `output_df.csv`.

**W&B keys** (logged to the run named `${data.run_name_component}_${model.run_name_component}`):
- Training subjects: `checkpoint/train_likelihood`, `checkpoint/eval_likelihood`,
  `checkpoint/step`, plus `final/*` summary and example/diagnostic images.
- Held-out subjects (multisubject auto fine-tuning, see §8): `heldout/train_likelihood`,
  `heldout/eval_likelihood`, `heldout/train_loss`, `heldout/eval_loss`,
  `heldout/step`, and `heldout/final/*` — logged into the **same** run, on a step
  axis offset past the training steps.

---

## 8. Held-out test subjects

The ~20% reserved held-out subjects (see §6) are handled differently by mode:

- **Single-subject runs:** held-out evaluation auto-enables for database
  (`mice_snapshot`) runs (`HeldoutEvalConfig.enabled`; disable with
  `data.heldout_eval: false`). `run_capsule` evaluates the trained model directly
  on the held-out subjects at checkpoints and at the end.

- **Multisubject GRU/disRNN runs:** the model has no embedding for an unseen
  subject, so it can't be evaluated zero-shot. Held-out generalization is measured
  by **fine-tuning a fresh embedding per held-out subject** (the rest of the model
  frozen) and reporting likelihood.

- **baseline_rl runs (single- or multi-subject):** RL agents fit per subject, so
  held-out subjects are simply **re-fit** (a fresh agent per held-out subject,
  train/eval split) — gated by `model.heldout_refit.enabled` (see §5). Reports
  `heldout/train_likelihood` + `heldout/eval_likelihood` like the NN models. No
  embedding transfer/fine-tuning is involved.

### Automatic multisubject held-out fine-tuning + evaluation
Controlled by `model.training.auto_heldout_finetune` (enabled by default in the
active GRU/disRNN configs). At the end of a multisubject GRU/disRNN run,
`run_capsule` automatically:
1. resolves the just-finished run at `/results` (fine-tuning from the `final`
   checkpoint by default),
2. loads the reserved held-out subjects and splits **each** subject's sessions
   into train/eval with the same `eval_every_n` as training subjects,
3. expands the subject-embedding table with a new row per held-out subject and
   fine-tunes **only those rows**,
4. logs aggregate held-out **train/eval split likelihoods** into the same W&B run
   under `heldout/*` (mirroring the training-subject metrics).

**Logging/plotting cadence mirrors training.** `checkpoint_every_n_steps` (default
10) controls how often the held-out fine-tune evaluates and logs
`heldout/train_likelihood`/`heldout/eval_likelihood` (so you get a curve, not just
endpoints); `checkpoint_plot_split_examples_every_n` (default 10) controls
split-example plotting. State-space figures and the loss/likelihood-over-checkpoints
curves are logged as images under `heldout/fig/*` (parity with the training
`fig/validation_loss_curve`). Set the cadence knobs to 0 for endpoints-only / no plots.

It is guarded so a fine-tuning failure never fails an otherwise-successful run
(records `output["heldout_finetune"]` with the error instead). Tune via
`auto_heldout_finetune.{n_steps,lr,checkpoint_policy,checkpoint_every_n_steps,checkpoint_plot_split_examples_every_n,...}`;
set `enabled: false` to skip.

### Standalone fine-tuning CLI
The same pipeline can be run manually against any trained run:
```
python run_heldout_subject_finetuning.py --config configs/config_heldout_subject_finetuning.yaml
```
This writes its own run dir under `output.output_root` and (optionally) its own
W&B run. Implementation: `post_training_analysis/heldout_finetuning.py`.

> Note: held-out **generative/rollout post-training analysis** (the
> `generative_analysis` path) remains unsupported for multisubject runs — see
> [README.md](../README.md). The auto fine-tuning above is the supported way to get
> held-out *likelihood* numbers for multisubject models.

---

## 9. Testing

`unittest` suites live in `tests/`. Run them with `code/` on the path:
```bash
cd code
PYTHONPATH=/root/capsule/code python -m unittest tests.test_gru_trainer -v
# or several:
PYTHONPATH=/root/capsule/code python -m unittest \
  tests.test_disrnn_trainer tests.test_heldout_finetuning tests.test_run_helpers
```
Key suites: `test_gru_trainer`, `test_disrnn_trainer`, `test_disrnn_distillation`,
`test_heldout_finetuning`, `test_multisubject_utils`, `test_run_helpers`, plus the
post-training-analysis suites. Suites self-skip if `jax`/`haiku`/
`disentangled_rnns`/`aind_disrnn_utils` aren't importable — confirm those import
before trusting a green result.

**Known environment-dependent failures** (present on a clean checkout, not caused
by app code): some disRNN multisubject training tests fail with
`NaN value for non-ignored trial`, and some analysis tests need a
`ex_model_dir-*` fixture that isn't present in every environment. Treat the
*delta* against a baseline run, not absolute green, when working in such an
environment.

---

## 10. Extending the codebase

- **New model trainer:** subclass `BaseMultisubjectTrainer` (if multisubject/
  session-conditioning applies) or `base.interfaces.ModelTrainer`; implement
  `fit(bundle, loggers) -> dict`; set the hook methods; add a `config_<model>.yaml`
  and a `tests/test_<model>_trainer.py`. Reuse the base snapshot/plot/eval pipeline
  rather than re-implementing it.
- **New data loader:** subclass `base.interfaces.DatasetLoader`; return a
  `DatasetBundle` (raw df, train_set, eval_set, metadata); add a `data:` preset.
- **New config key:** read it with `getattr(cfg, "key", default)` so older configs
  stay valid, and document it inline in the config templates.
- **Always:** add/extend a test, run the affected suites, and **update this file +
  the Changelog**.

---

## Changelog

> Add a dated entry (newest first) whenever you add or change a feature.

### 2026-07-04
- **Threshold-free bottleneck-sparsity metrics.** `compute_bottleneck_sparsity_metrics`
  now also logs, per family: `n_eff_open` + `n_eff_open_frac` (normalized participation
  ratio of channel openness `1−σ`, comparable across families of different size),
  `total_openness` (Σ(1−σ)), `sigma_p10/median/p90`, and a multi-threshold `frac_open`
  curve (σ<0.03/0.1/0.5/0.9/0.97). Motivated by the single-threshold `frac_open`
  saturating to 0 (σ<0.1) while the dashboard figure reads "open" (σ<0.97) — see §1.5.
- **Docs: new §1.5 "Run lifecycle & key switches (read first)".** Consolidates the
  four run phases, the two *different* held-out switches
  (`checkpoint_run_heldout_eval` vs `auto_heldout_finetune.enabled`), and
  checkpoints/resumability/extendability upfront, after two sessions misread the
  per-checkpoint-eval skip log line as "no held-out at all."
- **Clarified the held-out skip log messages** in `gru_trainer`, `disrnn_trainer`,
  and `training_runner` to name *which* held-out path is skipped and state that the
  end-of-training `auto_heldout_finetune` still runs (with `heldout/*` metrics).

### 2026-06-18
- **Held-out auto fine-tune mirrors training's logging/plotting cadence.** The
  `auto_heldout_finetune` cadence knobs now default non-zero
  (`checkpoint_every_n_steps: 10`, `checkpoint_plot_split_examples_every_n: 10`) so
  the held-out fine-tune logs a `heldout/train_likelihood`/`heldout/eval_likelihood`
  curve + split-example plots at a controllable interval, and the
  loss/likelihood-over-checkpoints curves are now logged to W&B under
  `heldout/fig/loss_curve` + `heldout/fig/likelihood_curve` (parity with the training
  `fig/validation_loss_curve`). The inner per-step loss logging was coarsened from
  every step to every 10 steps (`log_losses_every=10`) to match training's cadence.
- **baseline_rl fits + reports likelihood on held-out subjects.** Added
  `BaselineRLTrainer.fit_heldout` — re-fits a fresh RL agent per reserved held-out
  subject (train/eval split) and logs aggregate `heldout/train_likelihood` +
  `heldout/eval_likelihood` into the same W&B run (parity with GRU/disRNN), gated by
  `model.heldout_refit.enabled` (default on in `config_baseline_rl.yaml`). Training
  subjects also surface `checkpoint/train_likelihood`/`checkpoint/eval_likelihood`.
  `run_capsule` now drives this for both single- and multi-subject baseline_rl,
  replacing the retired transfer-based held-out eval. Database (`mice_snapshot`)
  loading works for both subject groups.
- **Post-training evaluation reorganized into `evaluation/` + unified CLI.** Eval
  primitives moved from `utils/` into the new `evaluation/` package (transparent
  re-export shims left behind, so training imports are unchanged); shared
  `HeldoutEvalConfig`/helpers split into `evaluation/heldout_eval_config.py` and
  `evaluation/common.py`. Added `run_analysis.py` (one sub-command per analysis) and
  deprecated the `run_capsule-test_*.py` scratch scripts to stubs. New living guide:
  **[POST_TRAINING_ANALYSIS.md](POST_TRAINING_ANALYSIS.md)**. No training-code changes. (commit `73fd4c2`)

### 2026-06-17
- **Automatic multisubject held-out fine-tuning + evaluation.** Added
  `model.training.auto_heldout_finetune` (on by default for GRU/disRNN); at the end
  of a multisubject run, held-out subjects are fine-tuned and their aggregate
  train/eval split likelihoods are logged into the same W&B run under `heldout/*`.
  `run_heldout_subject_finetuning_from_config` gained optional
  `wandb_run`/`wandb_key_prefix`/`wandb_step_offset` (standalone CLI behavior
  unchanged). (commit `f675ca2`)
- **Trainer modularization + bug fixes.** Extracted ~1,900 lines of duplicated
  logic from `gru_trainer.py`/`disrnn_trainer.py` into the new
  `BaseMultisubjectTrainer`; fixed disRNN step validation, empty-loss guards,
  strict logit-count resolution, and the held-out summary key. (commit `6d9b379`)
- **Config documentation.** Rewrote `config_gru.yaml`/`config_disrnn.yaml` with a
  documented comment on every key and corrected stale single-subject presets.
  (commit `29911dc`)
