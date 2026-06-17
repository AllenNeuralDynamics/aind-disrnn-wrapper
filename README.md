# aind-disrnn-wrapper

The wrapper capsule in the AIND-disRNN MLOps stack:

<img width="1380" height="875" alt="image" src="https://github.com/user-attachments/assets/e029c0e3-ce47-4f65-b61f-42c8bd5b053a" />

📖 **[Training Codebase Guide](code/TRAINING.md)** — what the training codebase
includes and how to use it (models, configs, data loaders, held-out evaluation,
testing). A living document; keep it updated as features are added.

## Session-Conditioned Post-Training Analysis

Session-conditioned multisubject GRU/disRNN runs are supported across the current
post-training analysis entrypoints:

- rollout-based post-training analysis via `post_training_analysis.run_post_training_analysis`
- saved-history reanalysis via `post_training_analysis.run_post_training_analysis_from_saved_histories`
- next-trial likelihood comparison via `post_training_analysis.run_prediction_likelihood_comparison`

### Required artifacts

Session-conditioned multisubject analysis expects the training run to already
contain both of these artifacts under `outputs/`:

- `subject_index_map.json`
- `session_context_map.json`

The analysis code uses these artifacts as the source of truth for subject-index
and per-subject session-index resolution. This path does not reconstruct missing
session-conditioning artifacts for older runs.

### Supported usage

- Multisubject session-conditioned analysis is supported for `split="train"` runs.
- Session-partitioned generative analysis can optionally request
  `session_partitions=("train", "eval", "combined")`.
- Saved `resolved_run.json` and likelihood-comparison summaries now record whether
  a run was analyzed as session-conditioned.

### Unsupported cases

- Multisubject held-out post-training analysis remains unsupported.
- Multisubject held-out likelihood evaluation remains unsupported.
- Session-conditioned multisubject runs missing `outputs/session_context_map.json`
  or `outputs/subject_index_map.json` are rejected instead of being analyzed with
  inferred session ordering.
