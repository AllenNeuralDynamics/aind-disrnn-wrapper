"""Local multisubject disRNN with an explicit subject-embedding table."""

from __future__ import annotations

import dataclasses

import haiku as hk
import jax.numpy as jnp
from disentangled_rnns.library import disrnn
from disentangled_rnns.library import multisubject_disrnn as upstream_multisubject_disrnn

from models.session_conditioning import (
    apply_session_conditioning,
    build_session_feat,
    resolve_session_conditioning_config,
)
from models.subject_embedding_initialization import make_subject_embedding_initializer


@dataclasses.dataclass
class MultisubjectDisRnnConfig(upstream_multisubject_disrnn.MultisubjectDisRnnConfig):
    """Local extension of the upstream config for wrapper-only switches."""

    use_global_subject_bottleneck: bool = True
    subject_embedding_init: str = "zeros"
    session_encoding_type: str = "none"
    session_integration_type: str = "direct"
    session_fourier_k: int = 4
    session_max_index_by_subject_index: list[int] = dataclasses.field(default_factory=list)


class MultisubjectDisRnn(upstream_multisubject_disrnn.MultisubjectDisRnn):
    """Multisubject disRNN with a learned embedding row per subject."""

    def __init__(self, config: MultisubjectDisRnnConfig):
        self._use_global_subject_bottleneck = bool(
            getattr(config, "use_global_subject_bottleneck", True)
        )
        self._subject_embedding_init = make_subject_embedding_initializer(
            subject_embedding_init=getattr(config, "subject_embedding_init", "zeros"),
            max_n_subjects=int(config.max_n_subjects),
            subject_embedding_size=int(config.subject_embedding_size),
        )
        session_cfg = resolve_session_conditioning_config(
            multisubject=True,
            session_encoding_type=getattr(config, "session_encoding_type", "none"),
            session_integration_type=getattr(config, "session_integration_type", "direct"),
            session_fourier_k=getattr(config, "session_fourier_k", 4),
            session_max_index_by_subject_index=getattr(
                config,
                "session_max_index_by_subject_index",
                (),
            ),
            max_n_subjects=int(config.max_n_subjects),
            context="Multisubject disRNN",
        )
        self._session_conditioning_enabled = bool(session_cfg["enabled"])
        self._session_encoding_type = str(session_cfg["session_encoding_type"])
        self._session_integration_type = str(session_cfg["session_integration_type"])
        self._session_fourier_k = int(session_cfg["session_fourier_k"])
        self._session_max_index_by_subject_index = tuple(
            int(value) for value in session_cfg["session_max_index_by_subject_index"]
        )
        super().__init__(config)

    def _build_subj_emb_global_bottleneck(self):
        if not self._use_global_subject_bottleneck:
            self._subj_emb_global_sigma = None
            return
        super()._build_subj_emb_global_bottleneck()

    def __call__(self, inputs: jnp.ndarray, prev_latents: jnp.ndarray):
        batch_size = inputs.shape[0]
        penalty = jnp.zeros(shape=(batch_size,))

        subject_ids = jnp.asarray(inputs[:, 0], dtype=jnp.int32)
        if self._session_conditioning_enabled:
            session_ids = jnp.asarray(inputs[:, 1], dtype=jnp.int32)
            observations = inputs[:, 2:]
        else:
            session_ids = None
            observations = inputs[:, 1:]

        subject_embeddings_table = hk.get_parameter(
            "subject_embeddings",
            (self._max_n_subjects, self._subject_embedding_size),
            init=self._subject_embedding_init,
        )
        valid_subject_ids = jnp.logical_and(
            subject_ids >= 0,
            subject_ids < self._max_n_subjects,
        )
        safe_subject_ids = jnp.where(valid_subject_ids, subject_ids, 0)
        subject_embeddings = jnp.take(subject_embeddings_table, safe_subject_ids, axis=0)
        subject_embeddings = subject_embeddings * jnp.expand_dims(
            valid_subject_ids.astype(subject_embeddings.dtype),
            axis=1,
        )
        subject_context = subject_embeddings

        if self._session_conditioning_enabled:
            session_feat, valid_session_mask = build_session_feat(
                subject_idx=subject_ids,
                session_idx=session_ids,
                session_max_index_by_subject=jnp.asarray(
                    self._session_max_index_by_subject_index,
                    dtype=jnp.int32,
                ),
                encoding_type=self._session_encoding_type,
                fourier_k=self._session_fourier_k,
            )
            subject_context = apply_session_conditioning(
                subject_emb=subject_embeddings,
                session_feat=session_feat,
                valid_session_mask=valid_session_mask,
                d_subj=self._subject_embedding_size,
                integration_type=self._session_integration_type,
            )

        if self._use_global_subject_bottleneck:
            subject_context, kl_cost = disrnn.information_bottleneck(
                inputs=subject_context,
                sigmas=self._subj_emb_global_sigma,
                noiseless_mode=self._noiseless_mode,
            )
            penalty += self._subj_penalty * kl_cost

        subj_emb_for_update_net = jnp.tile(
            jnp.expand_dims(subject_context, 2),
            (1, 1, self._latent_size),
        )
        subj_emb_for_update_net, kl_cost = disrnn.information_bottleneck(
            inputs=subj_emb_for_update_net,
            sigmas=self._update_net_subj_sigmas,
            multipliers=self._update_net_subj_multipliers,
            noiseless_mode=self._noiseless_mode,
        )
        penalty += self._update_net_subj_penalty * kl_cost

        obs_for_update_net = jnp.tile(
            jnp.expand_dims(observations, 2), (1, 1, self._latent_size)
        )
        obs_for_update_net, kl_cost = disrnn.information_bottleneck(
            inputs=obs_for_update_net,
            sigmas=self._update_net_obs_sigmas,
            multipliers=self._update_net_obs_multipliers,
            noiseless_mode=self._noiseless_mode,
        )
        penalty += self._update_net_obs_penalty * kl_cost

        prev_latents_for_update_net = jnp.tile(
            jnp.expand_dims(prev_latents, 2), (1, 1, self._latent_size)
        )
        prev_latents_for_update_net, kl_cost = disrnn.information_bottleneck(
            inputs=prev_latents_for_update_net,
            sigmas=self._update_net_latent_sigmas,
            multipliers=self._update_net_latent_multipliers,
            noiseless_mode=self._noiseless_mode,
        )
        penalty += self._update_net_latent_penalty * kl_cost

        update_net_inputs = jnp.concatenate(
            (
                subj_emb_for_update_net,
                obs_for_update_net,
                prev_latents_for_update_net,
            ),
            axis=1,
        )
        new_latents, update_net_penalty = super().update_latents(
            update_net_inputs,
            prev_latents,
        )
        penalty += update_net_penalty

        subj_emb_for_choice_net, subj_emb_kl_cost = disrnn.information_bottleneck(
            inputs=subject_context,
            sigmas=self._choice_net_subj_sigmas,
            multipliers=self._choice_net_subj_multipliers,
            noiseless_mode=self._noiseless_mode,
        )
        penalty += self._choice_net_subj_penalty * subj_emb_kl_cost

        latents_for_choice_net, latent_kl_cost = disrnn.information_bottleneck(
            inputs=new_latents,
            sigmas=self._choice_net_latent_sigmas,
            multipliers=self._choice_net_latent_multipliers,
            noiseless_mode=self._noiseless_mode,
        )
        penalty += self._choice_net_latent_penalty * latent_kl_cost

        choice_net_inputs = jnp.concatenate(
            (subj_emb_for_choice_net, latents_for_choice_net),
            axis=1,
        )
        predicted_targets, choice_net_penalty = super().predict_targets(choice_net_inputs)
        penalty += choice_net_penalty

        output = jnp.zeros((batch_size, self._output_size + 1))
        output = output.at[:, :-1].set(predicted_targets)
        output = output.at[:, -1].set(penalty)
        return output, new_latents


def get_auxiliary_metrics(*args, **kwargs):
    """Reuse the upstream multisubject auxiliary-metric implementation."""
    return upstream_multisubject_disrnn.get_auxiliary_metrics(*args, **kwargs)
