# Beaker image build log

Registry history for the disRNN wrapper images in
`ai1/aind-dynamic-foraging-foundation-model`.

The timestamps, Beaker IDs, and sizes below were reconstructed from the live
Beaker image records on 2026-07-12. Historical build refs were not stored by
Beaker and are marked as unknown rather than inferred from image names. Runtime
jobs refresh all three repositories through `WRAPPER_REF`, `DISPATCHER_REF`, and
`FORAGING_MODELS_REF`;
the baked refs describe the dependency environment, not necessarily the code a
job executes.

## Live images

### `han-hou/dynamic-foraging-bfm-wrapper-main-20260902`

- **Status:** current; use for new studies. First image under the
  `dynamic-foraging-bfm` identity (ADR-0007) and the only one carrying the
  hierarchical-Bayes dependencies.
- **Beaker image ID:** `01M1JHDCQEEMRMH4EHQGEX7QY8`
- **Created:** 2026-09-02 19:26 PT
- **Committed:** 2026-09-02 19:33 PT
- **Size:** 6,668,394,530 bytes (6.67 GB)
- **Build host:** Mac, `linux/amd64`
- **Baked wrapper ref:** `ff9c5449a5e59c6aa06685a68b9255fd6c1bf2e5`
- **Baked dispatcher ref:** `8268c9b2cdd5a096e723982036c941b8907b0301`
- **Baked foraging-models ref:** `26ce8ebd3ce3fb1d75b4fac5df8eeb7b675510c2`
- **Baked disentangled-rnns ref:** `a9b9978831cb22d37e2a75c15805c621dfe00b1f`
- **Reason:** two changes that had to ship together. The `/workspace` clone targets
  moved to `aind-dynamic-foraging-bfm-{wrapper,dispatcher}`, so **jobs using this image
  must use the new paths** -- an old image with new paths, or the reverse, dies at
  container startup with no logs. It also carries the models `[bayes]` extra
  (numpyro 0.19.0, arviz 1.3.0, xarray, h5netcdf) that `HBTrainer` needs.
- **Notable:** first build with no pip resolver conflict. The wrapper now pins
  `aind-dynamic-foraging-models==0.14.0` (0.13.0 predates the `[bayes]` extra) and
  `aind-dynamic-foraging-database==0.2.2` from PyPI instead of a floating git URL.
  `aind_disrnn_utils` is absent -- retired and vendored per ADR-0007 -- leaving
  `disentangled_rnns` as the only VCS dependency.
- **Smoke test:** [experiment `01M1JJ1EEHB8X3WCK4C11WJ9VE`](https://beaker.org/ex/01M1JJ1EEHB8X3WCK4C11WJ9VE),
  exit code 0 on one L40S. Verified the new `/workspace` paths resolve, `jax.devices()`
  sees the GPU, numpyro/arviz/`hierarchical_bayes` import, Hydra composes across both
  repos, and ~2 min of real training runs to completion.
- **Also recorded:** the first run of that smoke
  ([`01M1JHV58VTXR73TSYH0006WKX`](https://beaker.org/ex/01M1JHV58VTXR73TSYH0006WKX))
  failed with `out-of-range subject ids`, which was a defect in the smoke spec, not the
  image: `data=synthetic` is single-agent and emits no `num_subjects`, so multisubject
  disRNN can never train on it. Training moved to `data=synthetic_hierarchical`.

### `han-hou/disrnn-wrapper-main-20260712`

- **Status:** superseded by `dynamic-foraging-bfm-wrapper-main-20260902`. Still usable
  for pre-rename runs, but it bakes the OLD `/workspace/aind-disrnn-*` paths and has no
  `[bayes]` extra, so it cannot run hierarchical Bayes and must not be paired with a
  spec using the new paths.
- **Beaker image ID:** `01KXCF2EASQ8NV463684PZJ0ZP`
- **Created:** 2026-07-12 17:47 PT
- **Committed:** 2026-07-12 17:50 PT
- **Size:** 6,703,239,968 bytes (6.70 GB)
- **Build host:** Mac, `linux/amd64`
- **Baked wrapper ref:** `a4792b042ec61193f37c2be58c44f04479cb2e9b`
- **Baked dispatcher ref:** `7c3ae59d6adaf1f16f9b9a50fda55cb286a9df23`
- **Baked foraging-models ref:** `b44b0912de8d5307debe9b3b1c570cfc6dad816e`
- **Baked disentangled-rnns ref:** `a9b9978831cb22d37e2a75c15805c621dfe00b1f`
- **Reason:** default runtime refs now target `main`; foraging-models is refreshed
  from GitHub at job startup and records its resolved commit, while
  disentangled-rnns remains image-baked
- **Smoke test:** [experiment `01KXCGK6MM6QV3AND8C7ZC1TCX`](https://beaker.org/ex/01KXCGK6MM6QV3AND8C7ZC1TCX),
  exit code 0 on one g6e GPU; runtime refs, JAX CUDA, imports, Hydra composition,
  and `SMOKE OK` verified

### `han-hou/disrnn-wrapper-pck-integration-20260630`

- **Status:** deprecated; retained for older runs
- **Beaker image ID:** `01KWDGGQ4A9BTDXDHCGWXBWG05`
- **Created:** 2026-06-30 17:16 PT
- **Committed:** 2026-06-30 17:21 PT
- **Size:** 6,525,783,072 bytes (6.53 GB)
- **Build host:** Mac, `linux/amd64`
- **Baked wrapper ref:** unknown (not recorded at build time)
- **Baked dispatcher ref:** unknown (not recorded at build time)
- **Baked foraging-models ref:** PyPI `0.13.0` (no dynamic checkout)
- **Reason:** refresh dependencies for the snapshot-backed mice data path;
  includes `aind-dynamic-foraging-database` support for
  `select_sessions(snapshot=...)`

### `han-hou/disrnn-wrapper-pck-integration`

- **Status:** deprecated; retained for older runs
- **Beaker image ID:** `01KVEHPZ76A85CHWEWBQ43R6EY`
- **Created:** 2026-06-18 16:40 PT
- **Committed:** 2026-06-18 16:43 PT
- **Size:** 6,527,125,878 bytes (6.53 GB)
- **Build host:** Mac, `linux/amd64`
- **Baked wrapper ref:** unknown (not recorded at build time)
- **Baked dispatcher ref:** unknown (not recorded at build time)
- **Baked foraging-models ref:** PyPI `0.13.0` (no dynamic checkout)
- **Reason:** AI Hub integration image before the snapshot database dependency
  update; incompatible with loaders that call `select_sessions(snapshot=...)`

## Retired images

### `han-hou/disrnn-wrapper-bayes-20260901`

- **Status:** deleted from Beaker 2026-09-02; never used by any experiment
- **Beaker image ID:** `01M1F9SCQ4PSMCHBMMGG7YXTSD`
- **Created:** 2026-09-01 13:15 PT
- **Size:** 6,660,731,426 bytes (6.66 GB)
- **Baked wrapper ref:** `c8fa371b06c2af05689f6d0b9e64137a294759d2`
- **Baked dispatcher ref:** `045b560bceff0b20833986be06ca7aa31365b264`
- **Baked foraging-models ref:** `26ce8ebd3ce3fb1d75b4fac5df8eeb7b675510c2`
- **Reason built:** first image with the models `[bayes]` extra, so `HBTrainer` could
  run on Beaker at all.
- **Reason deleted:** built one day before the repo rename landed, so it bakes the
  pre-rename `/workspace/aind-disrnn-*` paths and cannot be paired with any current
  spec. `dynamic-foraging-bfm-wrapper-main-20260902` supersedes it on both counts.
  Nothing referenced it -- not the specs, not the `studies/` launch records -- and no
  job ever ran on it.
- **Known wart (fixed in the successor):** the wrapper then pinned
  `aind-dynamic-foraging-models==0.13.0` while the Dockerfile installed the package a
  second time from git, so the build printed a resolver conflict and resolved to 0.14.0.
  The editable install won, but any later `pip install` inside the image could have
  downgraded it and silently dropped the `bayes` extra.

### `han-hou/disrnn-wrapper`

- **Status:** deleted from Beaker
- **Beaker image ID:** unknown
- **Created / committed:** unknown
- **Size:** unknown
- **Reason:** original AI Hub MVP image; superseded by the integration images

## Recording a new build

After `build_and_push.sh` succeeds, add an entry above with:

- full Beaker image name and image ID
- registry-created and committed timestamps in Seattle time
- image size
- build host and target platform
- exact baked wrapper, dispatcher, and foraging-models refs
- dependency or environment change that required the rebuild
- smoke-test experiment ID and result, when available

Read registry metadata with:

```bash
beaker image get <image-name> --format json
```

Use full commit SHAs for `--wrapper-ref`, `--dispatcher-ref`, and
`--foraging-models-ref` when building a release image so the baked dependency
environment can be reproduced.
