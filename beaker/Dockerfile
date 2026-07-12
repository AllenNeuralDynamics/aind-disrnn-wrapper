# Beaker (AI Hub) image for the disRNN wrapper — GPU.
#
# Clones the four runtime source repos from GitHub.
# The dispatcher and wrapper use the sibling layout that run_hpc.py expects:
#   /workspace/aind-disrnn-wrapper/code/run_hpc.py reads its Hydra configs from
#   ../../aind-disrnn-dispatcher/code/config  ->  /workspace/aind-disrnn-dispatcher/...
# so W&B sweep agents (which invoke `python -m run_hpc <overrides>`) can
# re-compose the config on Beaker with no code change.
#
# Because all repos are cloned from git, the build context is irrelevant — you
# can build from anywhere. All repos/deps are public, so no token is needed:
#
#   docker build \
#     --platform linux/amd64 \
#     -f beaker/Dockerfile \
#     -t disrnn-wrapper beaker/        # context only has to exist; nothing is COPYed
#
# jax[cuda12] ships the CUDA runtime as pip wheels, so a slim Python base is
# enough — Beaker nodes provide the GPU driver.

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HYDRA_FULL_ERROR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Which refs to bake in. Branches, tags, and full commit SHAs are supported.
ARG GH_ORG=https://github.com/AllenNeuralDynamics
ARG DISPATCHER_REF=main
ARG WRAPPER_REF=main
ARG FORAGING_MODELS_REF=main
ARG DISENTANGLED_RNNS_REF=main

# Clone all four repos and install the wrapper + GPU JAX. All repos
# and the wrapper's git deps (disentangled-rnns, aind-disrnn-utils) are public,
# so no credentials are needed.
#
# The baked code is only a STARTING POINT: at job startup beaker/entrypoint.sh
# re-pulls all four repos to the requested refs (default main), so the image
# rarely needs rebuilding — only when DEPENDENCIES change. The build-time clone exists
# mainly to (a) install deps from pyproject and (b) seed the repos so entrypoint.sh
# can `git fetch` them. See beaker/entrypoint.sh and the README.
#
# For release images, point all *_REF args at full commit SHAs.
#
# CACHEBUST: the clone instruction text is identical across builds, so Docker
# would otherwise reuse a stale cached clone and silently bake old code. Pass a
# changing value (build_and_push.sh sends a timestamp) to force a fresh clone +
# reinstall whenever you intentionally rebuild.
ARG CACHEBUST=0
RUN clone_ref() { \
      repo="$1"; dir="$2"; ref="$3"; \
      git clone --depth 1 "$repo" "$dir"; \
      git -C "$dir" fetch --depth 1 origin "$ref"; \
      git -C "$dir" checkout -f --detach FETCH_HEAD; \
    }; \
    echo "cachebust=$CACHEBUST" && \
    clone_ref "$GH_ORG/aind-disrnn-dispatcher.git" \
      /workspace/aind-disrnn-dispatcher "$DISPATCHER_REF" && \
    clone_ref "$GH_ORG/aind-dynamic-foraging-models.git" \
      /workspace/aind-dynamic-foraging-models "$FORAGING_MODELS_REF" && \
    clone_ref "$GH_ORG/aind-disentangled-rnns.git" \
      /workspace/aind-disentangled-rnns "$DISENTANGLED_RNNS_REF" && \
    clone_ref "$GH_ORG/aind-disrnn-wrapper.git" \
      /workspace/aind-disrnn-wrapper "$WRAPPER_REF" && \
    pip install --no-cache-dir -e "/workspace/aind-disrnn-wrapper[gpu]" && \
    pip install --no-cache-dir -e /workspace/aind-dynamic-foraging-models && \
    pip install --no-cache-dir -e /workspace/aind-disentangled-rnns

WORKDIR /workspace/aind-disrnn-wrapper/code
ENV PYTHONPATH=/workspace/aind-disrnn-wrapper/code
