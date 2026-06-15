# Beaker (AI Hub) image for the disRNN wrapper — GPU.
#
# Bakes in BOTH repos in the sibling layout that run_hpc.py expects:
#   /workspace/aind-disrnn-wrapper/code/run_hpc.py reads its Hydra configs from
#   ../../aind-disrnn-dispatcher/code/config  ->  /workspace/aind-disrnn-dispatcher/...
# so W&B sweep agents (which invoke `python -m run_hpc <overrides>`) can
# re-compose the config on Beaker with no code change.
#
# Build context must be the PARENT dir holding both repos as siblings
# (e.g. /home/han.hou/code). Private git deps need a GitHub token at build time:
#
#   cd /home/han.hou/code
#   DOCKER_BUILDKIT=1 docker build \
#     --platform linux/amd64 \
#     --secret id=gh_token,env=GITHUB_TOKEN \
#     -f aind-disrnn-wrapper/beaker/Dockerfile \
#     -t disrnn-wrapper .
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

# Sibling layout (order: configs first — they change less often than wrapper code)
COPY aind-disrnn-dispatcher /workspace/aind-disrnn-dispatcher
COPY aind-disrnn-wrapper    /workspace/aind-disrnn-wrapper

# Install the wrapper + GPU JAX. The wrapper's private git deps
# (disentangled-rnns, aind-disrnn-utils) are fetched using a build secret so the
# token never lands in an image layer.
RUN --mount=type=secret,id=gh_token \
    git config --global credential.helper \
      '!f() { echo username=x-access-token; echo "password=$(cat /run/secrets/gh_token)"; }; f' && \
    pip install --no-cache-dir -e "/workspace/aind-disrnn-wrapper[gpu]" && \
    git config --global --unset credential.helper || true

WORKDIR /workspace/aind-disrnn-wrapper/code
ENV PYTHONPATH=/workspace/aind-disrnn-wrapper/code
