# hash:sha256:4f567b195522ba414263860e03ae193d69e7c4114b288c652000066eab30db33
ARG REGISTRY_HOST

# --- Base image and Jax for CPU (preferred) ---
FROM $REGISTRY_HOST/codeocean/mambaforge3:24.5.0-0-python3.12.4-ubuntu22.04
ARG JAX_PKG="jax==0.6.1"

# --- Base image and Jax for GPU ---
# FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04
# ARG JAX_PKG="jax[cuda12]==0.6.1"


ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

# --- Other dependencies ---
RUN pip install -U --no-cache-dir \
    ${JAX_PKG} \
    "git+https://github.com/AllenNeuralDynamics/aind-disentangled-rnns.git@e8fbbb8521599c5ad13a70ca9199dfdb823fcb8c#egg=disentangled-rnns" \
    "git+https://github.com/AllenNeuralDynamics/aind_disrnn_utils.git@4b8defe9ab5c67287e7af3fedf9e07a714a20993#egg=aind-disrnn-utils" \
    aind-analysis-arch-result-access==0.8.0 \
    aind-dynamic-foraging-basic-analysis==0.3.34 \
    aind-dynamic-foraging-data-utils==0.1.41 \
    aind-dynamic-foraging-multisession-analysis==0.0.18 \
    aind-behavior-gym==0.4.4 \
    aind-dynamic-foraging-models~=0.12.2 \
    wandb~=0.23.0 \
    hydra-core~=1.3.0 \
    omegaconf~=2.3.0

# for dev
RUN pip install -U --no-cache-dir \
    black==25.9.0 \
    isort==7.0.0

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
