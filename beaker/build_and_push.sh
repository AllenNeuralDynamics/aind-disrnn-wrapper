#!/usr/bin/env bash
# Build the disRNN wrapper GPU image and push it to Beaker's registry.
#
# The Dockerfile git-clones both repos itself (all public, no token needed), so
# there is no build context to stage and you can run this from anywhere. This
# script bundles the x86 platform flag, a cache-bust, and the `beaker image
# create` push.
#
# Usage:
#   bash beaker/build_and_push.sh [options]
#
# Options:
#   --name NAME            Beaker image name (default: disrnn-wrapper)
#   --workspace WS         Beaker workspace (default: ai1/aind-dynamic-foraging-foundation-model)
#   --ref REF             Branch/tag/SHA to bake into BOTH repos (default: ai_hub)
#   --wrapper-ref REF      Override the wrapper repo ref only
#   --dispatcher-ref REF   Override the dispatcher repo ref only
#   --force-rebuild        If an image of this name already exists in Beaker,
#                          replace it (delete + re-push). Without this flag the
#                          script refuses to touch an existing image and stops.
#   -h, --help             Show this help.
#
# After it pushes, run `beaker image get <name>` to read the <username>/<name>
# ref you paste into experiment_mvp.yaml.
set -euo pipefail

# Defaults (override via the flags above — not env vars).
IMAGE_NAME="disrnn-wrapper"
WORKSPACE="ai1/aind-dynamic-foraging-foundation-model"
WRAPPER_REF="ai_hub"
DISPATCHER_REF="ai_hub"
FORCE_REBUILD=0

usage() { sed -n '9,24p' "$0"; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --name)           IMAGE_NAME="$2"; shift 2 ;;
        --workspace)      WORKSPACE="$2"; shift 2 ;;
        --ref)            WRAPPER_REF="$2"; DISPATCHER_REF="$2"; shift 2 ;;
        --wrapper-ref)    WRAPPER_REF="$2"; shift 2 ;;
        --dispatcher-ref) DISPATCHER_REF="$2"; shift 2 ;;
        --force-rebuild)  FORCE_REBUILD=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "image: $IMAGE_NAME   workspace: $WORKSPACE   refs: ${DISPATCHER_REF}/${WRAPPER_REF}   force-rebuild: $FORCE_REBUILD"

# Pre-flight: Beaker image names are unique per workspace. Decide up front whether
# we're allowed to replace an existing one — but defer the actual delete until
# AFTER a successful build, so a build failure never leaves you with no image.
ME="$(beaker account whoami --format json | sed -n 's/.*"name": *"\([^"]*\)".*/\1/p' | head -1)"
REPLACE_EXISTING=0
if [ -n "$ME" ] && beaker image get "$ME/$IMAGE_NAME" >/dev/null 2>&1; then
    if [ "$FORCE_REBUILD" -eq 1 ]; then
        REPLACE_EXISTING=1
        echo "note: $ME/$IMAGE_NAME exists; will replace it after a successful build (--force-rebuild)"
    else
        cat >&2 <<MSG
ERROR: image "$ME/$IMAGE_NAME" already exists in Beaker — not building.
Choose one and re-run:
  - replace it:       add --force-rebuild
  - use another name: --name disrnn-wrapper-2   (then update the ref in experiment_mvp.yaml)
MSG
        exit 1
    fi
fi

# Build (x86 for Beaker GPU nodes). CACHEBUST forces a fresh git clone + reinstall
# each build (the clone instruction is otherwise byte-identical and Docker would
# reuse a stale cached clone, baking old code).
docker build \
    --platform linux/amd64 \
    --build-arg CACHEBUST="$(date +%s)" \
    --build-arg WRAPPER_REF="$WRAPPER_REF" \
    --build-arg DISPATCHER_REF="$DISPATCHER_REF" \
    -f "$SCRIPT_DIR/Dockerfile" \
    -t "$IMAGE_NAME" \
    "$SCRIPT_DIR"

# Replace the old image only now that the new build succeeded.
if [ "$REPLACE_EXISTING" -eq 1 ]; then
    echo "deleting previous image $ME/$IMAGE_NAME"
    beaker image delete "$ME/$IMAGE_NAME"
fi

# Push to Beaker's own registry.
beaker image create --name "$IMAGE_NAME" -w "$WORKSPACE" "$IMAGE_NAME"

echo
echo "Done. Read the image ref to paste into experiment_mvp.yaml:"
echo "  beaker image get $IMAGE_NAME"
