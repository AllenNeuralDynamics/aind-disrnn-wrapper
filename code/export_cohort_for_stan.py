"""Export an HB cohort to a .npz that the Stan benchmark can fit.

Why this is a script rather than a notebook cell: the Stan-vs-NumPyro comparison
(`aind-dynamic-foraging-models` benchmarks, dispatcher#129) is only meaningful if both
frameworks see **the same trials**. The cheapest way to guarantee that is to build the arrays
with the trainer's own helpers -- `_extract_subject_sessions` and `_pad_cohort` -- rather than
re-deriving the cohort from the database and hoping the selection matches. Any change to
subject selection, ignore-trial handling or session ordering then reaches both arms together.

The Hydra config is composed exactly as `run_hpc` composes it, from the dispatcher's config
tree, so `data.subject_ratio` and `seed` select the same cohort as the corresponding rung.

Usage (compute node only -- this reads the database):

    python export_cohort_for_stan.py --subject-ratio 0.049 --seed 0 --out cohort_d29.npz

The saved arrays are what `fit_three_level_stan.py --npz` expects: `choice`, `reward`,
`n_sessions`, `n_trials`. Note `n_trials` is a COUNT per session, not a mask -- the trainer
places each session's valid trials contiguously from index 0, so a count is lossless here and
is what lets Stan loop to the true length instead of padding.
"""

import argparse
import json

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-ratio", type=float, required=True,
                        help="per-curriculum fraction of the train pool, e.g. 0.049 for D~29")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=str, default="mice_snapshot_scaling")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    from model_trainers.hb_trainer import _extract_subject_sessions, _pad_cohort
    from run_hpc import _dispatcher_config_path

    import os
    here = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.abspath(os.path.join(here, _dispatcher_config_path()))

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="config",
            overrides=[
                f"data={args.data}",
                "model=hb_hattori",
                f"data.subject_ratio={args.subject_ratio}",
                f"seed={args.seed}",
            ],
        )

    loader = instantiate(cfg.data)
    bundle = loader.load()
    if bundle.raw is None or len(bundle.raw) == 0:
        raise SystemExit("loader returned no trial-level rows")

    choices, rewards, _ = _extract_subject_sessions(bundle.raw)
    choice_arr, reward_arr, valid_mask, session_mask, subject_ids = _pad_cohort(choices, rewards)

    n_sessions = session_mask.sum(axis=1).astype(np.int32)
    n_trials = valid_mask.sum(axis=2).astype(np.int32)

    # Guard the assumption the count-not-mask export rests on: every session's valid trials
    # must be a contiguous prefix. If padding ever moves, this fails loudly here rather than
    # silently handing Stan the wrong trials.
    for s in range(valid_mask.shape[0]):
        for m in range(int(n_sessions[s])):
            k = int(n_trials[s, m])
            assert valid_mask[s, m, :k].all() and not valid_mask[s, m, k:].any(), (
                f"subject {s} session {m}: valid trials are not a contiguous prefix"
            )

    real = int(n_trials.sum())
    padded = int(np.prod(choice_arr.shape))
    print(f"subjects {choice_arr.shape[0]}, sessions <= {choice_arr.shape[1]}, "
          f"trials <= {choice_arr.shape[2]}")
    print(f"real trials {real:,} vs padded {padded:,} "
          f"({100 * (1 - real / padded):.1f}% padding)")

    np.savez_compressed(
        args.out,
        choice=choice_arr.astype(np.int32),
        reward=(reward_arr > 0).astype(np.int32),
        n_sessions=n_sessions,
        n_trials=n_trials,
    )
    meta = {
        "subject_ratio": args.subject_ratio, "seed": args.seed, "data": args.data,
        "n_subjects": int(choice_arr.shape[0]), "max_sessions": int(choice_arr.shape[1]),
        "max_trials": int(choice_arr.shape[2]), "real_trials": real, "padded_trials": padded,
        "subject_ids": [str(s) for s in subject_ids],
        "config": OmegaConf.to_container(cfg.data, resolve=True),
    }
    with open(args.out.replace(".npz", ".meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2, default=str)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
