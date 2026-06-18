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
COPY git-ask-pass /

# --- Other dependencies ---
RUN pip install -U --no-cache-dir \
    ${JAX_PKG} \
    "git+https://github.com/AllenNeuralDynamics/aind-dynamic-foraging-database.git" \
    "git+https://github.com/AllenNeuralDynamics/aind-disentangled-rnns.git@a9b9978831cb22d37e2a75c15805c621dfe00b1f#egg=disentangled-rnns" \
    "git+https://github.com/AllenNeuralDynamics/aind_disrnn_utils.git@74de874d93b951d9ef3ae6bff6453e6ae805b649#egg=aind-disrnn-utils" \
    aind-analysis-arch-result-access==0.8.0 \
    aind-dynamic-foraging-basic-analysis==0.3.34 \
    aind-dynamic-foraging-data-utils==0.1.41 \
    aind-dynamic-foraging-multisession-analysis==0.0.18 \
    aind-dynamic-foraging-models==0.13.0 \
    aind-behavior-gym==0.4.5 \
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
