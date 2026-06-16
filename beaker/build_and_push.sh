#!/usr/bin/env bash
# Build the disRNN wrapper GPU image and push it to Beaker's registry.
#
# The Dockerfile git-clones both repos itself (all public, no token needed), so
# there is no build context to stage and you can run this from anywhere. This
# script just bundles the x86 platform flag and the `beaker image create` push.
#
# Usage:
#   [export WORKSPACE=ai1/...]                # optional, default below (our AI Hub workspace)
#   [export IMAGE_NAME=disrnn-wrapper]        # optional, default below
#   [export DISPATCHER_REF=ai_hub] [export WRAPPER_REF=ai_hub]   # optional, branch/tag to bake in
#   bash beaker/build_and_push.sh
#
# After it prints the image id, run `beaker image get <id>` to read the
# <username>/disrnn-wrapper ref you paste into experiment_mvp.yaml.
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-disrnn-wrapper}"
WORKSPACE="${WORKSPACE:-ai1/aind-dynamic-foraging-foundation-model}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "image: $IMAGE_NAME   workspace: $WORKSPACE   refs: ${DISPATCHER_REF:-ai_hub}/${WRAPPER_REF:-ai_hub}"

# Pre-flight: Beaker image names are unique per workspace. If one already exists,
# stop BEFORE building and let the user decide — never auto-delete (a delete could
# pull the rug from under a running experiment that uses that image).
ME="$(beaker account whoami --format json | sed -n 's/.*"name": *"\([^"]*\)".*/\1/p' | head -1)"
if [ -n "$ME" ] && beaker image get "$ME/$IMAGE_NAME" >/dev/null 2>&1; then
    cat >&2 <<MSG
ERROR: image "$ME/$IMAGE_NAME" already exists in Beaker — not building.
Choose one and re-run:
  - replace it:        beaker image delete "$ME/$IMAGE_NAME"   then re-run this script
  - use another name:  IMAGE_NAME=disrnn-wrapper-2 bash beaker/build_and_push.sh
                       (then update the image ref in experiment_mvp.yaml)
MSG
    exit 1
fi

# Build (x86 for Beaker GPU nodes). CACHEBUST forces a fresh git clone + reinstall
# each build (the clone instruction is otherwise byte-identical and Docker would
# reuse a stale cached clone, baking old code).
docker build \
    --platform linux/amd64 \
    --build-arg CACHEBUST="$(date +%s)" \
    ${DISPATCHER_REF:+--build-arg DISPATCHER_REF="$DISPATCHER_REF"} \
    ${WRAPPER_REF:+--build-arg WRAPPER_REF="$WRAPPER_REF"} \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "$IMAGE_NAME" \
    "$SCRIPT_DIR"

# Push to Beaker's own registry.
beaker image create --name "$IMAGE_NAME" -w "$WORKSPACE" "$IMAGE_NAME"

echo
echo "Done. Read the image ref to paste into experiment_mvp.yaml:"
echo "  beaker image get $IMAGE_NAME"
