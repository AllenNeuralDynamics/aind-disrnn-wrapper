# Beaker (AI Hub) image for the disRNN wrapper — GPU.
#
# Clones BOTH repos from GitHub in the sibling layout that run_hpc.py expects:
#   /workspace/aind-disrnn-wrapper/code/run_hpc.py reads its Hydra configs from
#   ../../aind-disrnn-dispatcher/code/config  ->  /workspace/aind-disrnn-dispatcher/...
# so W&B sweep agents (which invoke `python -m run_hpc <overrides>`) can
# re-compose the config on Beaker with no code change.
#
# Because both repos are cloned from git, the build context is irrelevant — you
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

# Which refs to bake in. Default to the migration branch; override at build time
# (e.g. --build-arg WRAPPER_REF=main) once merged. `-b` takes a branch or tag.
ARG GH_ORG=https://github.com/AllenNeuralDynamics
ARG DISPATCHER_REF=ai_hub
ARG WRAPPER_REF=ai_hub

# Clone both repos (sibling layout) and install the wrapper + GPU JAX. All repos
# and the wrapper's git deps (disentangled-rnns, aind-disrnn-utils) are public,
# so no credentials are needed.
#
# Trade-off: pinned to branch tips (ai_hub), so the image is reproducible only as
# far as "whatever ai_hub pointed at when built" — not a commit SHA. For release
# images, point the *_REF args at tags/SHAs (a tag works with `-b`; a bare SHA
# needs a fetch+checkout instead of --depth 1 -b).
RUN git clone --depth 1 -b "$DISPATCHER_REF" \
      "$GH_ORG/aind-disrnn-dispatcher.git" /workspace/aind-disrnn-dispatcher && \
    git clone --depth 1 -b "$WRAPPER_REF" \
      "$GH_ORG/aind-disrnn-wrapper.git" /workspace/aind-disrnn-wrapper && \
    pip install --no-cache-dir -e "/workspace/aind-disrnn-wrapper[gpu]"

WORKDIR /workspace/aind-disrnn-wrapper/code
ENV PYTHONPATH=/workspace/aind-disrnn-wrapper/code
