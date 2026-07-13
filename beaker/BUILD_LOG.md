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

### `han-hou/disrnn-wrapper-pck-integration-20260630`

- **Status:** current; use for new studies
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
