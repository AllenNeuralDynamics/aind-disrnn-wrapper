#!/usr/bin/env bash
# Build the disRNN wrapper GPU image and push it to Beaker's registry.
#
# The Dockerfile git-clones all four repos itself (all public, no token needed), so
# there is no build context to stage and you can run this from anywhere.
#
# Usage:
#   bash beaker/build_and_push.sh [options]
#
# Options:
#   --name NAME             Beaker image name (default: disrnn-wrapper)
#   --workspace WS          Beaker workspace (default: ai1/aind-dynamic-foraging-foundation-model)
#   --ref REF               Branch/tag/SHA to bake into all repos (default: main)
#   --wrapper-ref REF       Override the wrapper repo ref only
#   --dispatcher-ref REF    Override the dispatcher repo ref only
#   --foraging-models-ref REF
#                           Override the aind-dynamic-foraging-models ref only
#   --disentangled-rnns-ref REF
#                           Override the aind-disentangled-rnns ref only
#   --force-rebuild         Bust Docker's cache so the build does a FRESH clone +
#                           reinstall (otherwise Docker reuses cached layers and
#                           may bake stale code/deps).
#   --force-override-beaker Replace an existing Beaker image of this name (delete
#                           + re-push). Without it, the script stops if one exists.
#   -h, --help              Show this help.
#
# After it pushes, run `beaker image get <name>` to read the <username>/<name>
# ref you paste into experiment_mvp.yaml.
set -euo pipefail

# Defaults (override via the flags above — not env vars).
IMAGE_NAME="disrnn-wrapper"
WORKSPACE="ai1/aind-dynamic-foraging-foundation-model"
WRAPPER_REF="main"
DISPATCHER_REF="main"
FORAGING_MODELS_REF="main"
DISENTANGLED_RNNS_REF="main"
FORCE_REBUILD=0
FORCE_OVERRIDE_BEAKER=0

usage() { sed -n '7,24p' "$0"; }

while [ "$#" -gt 0 ]; do
    case "$1" in
        --name)                  IMAGE_NAME="$2"; shift 2 ;;
        --workspace)             WORKSPACE="$2"; shift 2 ;;
        --ref)                   WRAPPER_REF="$2"; DISPATCHER_REF="$2"; FORAGING_MODELS_REF="$2"; DISENTANGLED_RNNS_REF="$2"; shift 2 ;;
        --wrapper-ref)           WRAPPER_REF="$2"; shift 2 ;;
        --dispatcher-ref)        DISPATCHER_REF="$2"; shift 2 ;;
        --foraging-models-ref)   FORAGING_MODELS_REF="$2"; shift 2 ;;
        --disentangled-rnns-ref) DISENTANGLED_RNNS_REF="$2"; shift 2 ;;
        --force-rebuild)         FORCE_REBUILD=1; shift ;;
        --force-override-beaker) FORCE_OVERRIDE_BEAKER=1; shift ;;
        -h|--help)               usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "image: $IMAGE_NAME   workspace: $WORKSPACE"
echo "refs: dispatcher=${DISPATCHER_REF} wrapper=${WRAPPER_REF} foraging-models=${FORAGING_MODELS_REF}"
echo "      disentangled-rnns=${DISENTANGLED_RNNS_REF}"
echo "force-rebuild: $FORCE_REBUILD   force-override-beaker: $FORCE_OVERRIDE_BEAKER"

# Pre-flight: Beaker image names are unique per workspace. If one already exists,
# only --force-override-beaker lets us replace it — and even then we defer the
# delete until AFTER a successful build, so a build failure never leaves you with
# no image.
ME="$(beaker account whoami --format json | sed -n 's/.*"name": *"\([^"]*\)".*/\1/p' | head -1)"
REPLACE_EXISTING=0
if [ -n "$ME" ] && beaker image get "$ME/$IMAGE_NAME" >/dev/null 2>&1; then
    if [ "$FORCE_OVERRIDE_BEAKER" -eq 1 ]; then
        REPLACE_EXISTING=1
        echo "note: $ME/$IMAGE_NAME exists; will replace it after a successful build (--force-override-beaker)"
    else
        cat >&2 <<MSG
ERROR: image "$ME/$IMAGE_NAME" already exists in Beaker — not building.
Choose one and re-run:
  - replace it:       add --force-override-beaker
  - use another name: --name disrnn-wrapper-2   (then update the ref in experiment_mvp.yaml)
MSG
        exit 1
    fi
fi

# --force-rebuild busts Docker's cache for the clone+install layer (its instruction
# text is otherwise byte-identical, so Docker would reuse a stale cached clone).
cachebust_arg=""
if [ "$FORCE_REBUILD" -eq 1 ]; then
    cachebust_arg="--build-arg CACHEBUST=$(date +%s)"
    echo "note: --force-rebuild set -> forcing a fresh clone + reinstall (cache busted)"
fi

# Build (x86 for Beaker GPU nodes). $cachebust_arg is intentionally unquoted so it
# expands to two tokens or none (values contain no spaces).
docker build \
    --platform linux/amd64 \
    $cachebust_arg \
    --build-arg WRAPPER_REF="$WRAPPER_REF" \
    --build-arg DISPATCHER_REF="$DISPATCHER_REF" \
    --build-arg FORAGING_MODELS_REF="$FORAGING_MODELS_REF" \
    --build-arg DISENTANGLED_RNNS_REF="$DISENTANGLED_RNNS_REF" \
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
