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
| `utils/` | `run_helpers.py`, `multisubject.py`, `session_regularized_training.py`, `gru_evaluation.py`, `disrnn_evaluation.py`, `baseline_rl_evaluation.py`, `load_mice_database.py` |
| `post_training_analysis/` | `generative_analysis.py`, `embedding_space_analysis.py`, `likelihood_comparison.py`, `likelihood_advantage_analysis.py`, `baseline_rl_analysis.py`, `heldout_finetuning.py` |
| `run_heldout_subject_finetuning.py` | Standalone CLI for held-out subject embedding fine-tuning |
| `run_capsule-test_*.py`, `run_embedding_space_analysis.py` | Alternate entry points for analysis/experiments (toggled in `run`) |
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

- **Multisubject runs:** the model has no embedding for an unseen subject, so it
  can't be evaluated zero-shot. Held-out generalization is measured by
  **fine-tuning a fresh embedding per held-out subject** (the rest of the model
  frozen) and reporting likelihood.

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

It is guarded so a fine-tuning failure never fails an otherwise-successful run
(records `output["heldout_finetune"]` with the error instead). Tune via
`auto_heldout_finetune.{n_steps,lr,checkpoint_policy,...}`; set `enabled: false`
to skip.

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
