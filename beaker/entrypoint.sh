#!/usr/bin/env bash
# Runtime entrypoint for the disRNN Beaker image.
#
# Runs at CONTAINER STARTUP on every Beaker job (NOT at image build time): it
# refreshes both repos to the latest code on GitHub *before* running the job, so
# code/config edits take effect by just launching a new run — no image rebuild.
#
# Invoked from the Beaker spec's `command`, e.g.:
#   command: [bash, /workspace/aind-disrnn-wrapper/beaker/entrypoint.sh,
#             wandb, agent, --count, "1", "<SWEEP_ID>"]
# Everything after the script path is the real work, exec'd once code is fresh.
#
# WHEN YOU STILL NEED A REBUILD: only when DEPENDENCIES change (pyproject.toml /
# the pinned git deps). Plain code or Hydra-config edits never need one. If a run
# fails with ImportError/ModuleNotFound after a pull, that's the signal to rebuild.
#
# Pin a run for reproducibility by passing a commit SHA instead of the branch:
#   envVars: [{name: WRAPPER_REF, value: <sha>}, {name: DISPATCHER_REF, value: <sha>}]
set -euo pipefail

WRAPPER_REF="${WRAPPER_REF:-ai_hub}"
DISPATCHER_REF="${DISPATCHER_REF:-ai_hub}"

refresh() {  # <repo dir> <ref>
    local dir="$1" ref="$2"
    # --depth 1 keeps the shallow clone shallow; works for a branch, tag, or
    # (GitHub allows) a specific commit SHA. Detach at the fetched commit.
    git -C "$dir" fetch --depth 1 origin "$ref"
    git -C "$dir" checkout -f --detach FETCH_HEAD
    echo "[entrypoint]   $(basename "$dir") @ ${ref} -> $(git -C "$dir" rev-parse --short HEAD)"
}

echo "[entrypoint] refreshing source from GitHub before the run..."
refresh /workspace/aind-disrnn-dispatcher "$DISPATCHER_REF"
refresh /workspace/aind-disrnn-wrapper    "$WRAPPER_REF"

if [ "$#" -eq 0 ]; then
    echo "[entrypoint] no command given; nothing to run." >&2
    exit 1
fi

cd /workspace/aind-disrnn-wrapper/code
echo "[entrypoint] launching: $*"
exec "$@"
