# hash:sha256:4f567b195522ba414263860e03ae193d69e7c4114b288c652000066eab30db33
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/pytorch:2.4.0-cuda12.4.0-mambaforge24.5.0-0-python3.12.4-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /


RUN pip install -U --no-cache-dir \
    "jax[cuda12]==0.6.1" \
    "git+https://github.com/AllenNeuralDynamics/aind-disentangled-rnns.git@af30f2af9509043b370ca6ff2eca1d0c614eeb62#egg=disentangled-rnns" \
    "git+https://github.com/AllenNeuralDynamics/aind_disrnn_utils.git@94a1b1f23c1bed67983424d188d2bc89b10f4edb#egg=aind-disrnn-utils" \
    aind-analysis-arch-result-access==0.8.0 \
    aind-dynamic-foraging-basic-analysis==0.3.34 \
    aind-dynamic-foraging-data-utils==0.1.40 \
    aind-dynamic-foraging-multisession-analysis==0.0.18 \
    wandb==0.23.0

# for dev
RUN pip install -U --no-cache-dir \
    black==25.9.0 \
    isort==7.0.0

COPY postInstall /
RUN --mount=type=secret,id=secrets . /run/secrets/secrets \
    && /postInstall
