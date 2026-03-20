"""Local multisubject disRNN with an explicit subject-embedding table."""

from __future__ import annotations

import haiku as hk
import jax.numpy as jnp
import numpy as np
from disentangled_rnns.library import disrnn
from disentangled_rnns.library import multisubject_disrnn as upstream_multisubject_disrnn


MultisubjectDisRnnConfig = upstream_multisubject_disrnn.MultisubjectDisRnnConfig


class MultisubjectDisRnn(upstream_multisubject_disrnn.MultisubjectDisRnn):
    """Multisubject disRNN with a learned embedding row per subject."""

    def __call__(self, inputs: jnp.ndarray, prev_latents: jnp.ndarray):
        batch_size = inputs.shape[0]
        penalty = jnp.zeros(shape=(batch_size,))

        subject_ids = jnp.asarray(inputs[:, 0], dtype=jnp.int32)
        observations = inputs[:, 1:]

        subject_embeddings_table = hk.get_parameter(
            "subject_embeddings",
            (self._max_n_subjects, self._subject_embedding_size),
            init=hk.initializers.RandomNormal(
                stddev=1.0 / np.sqrt(max(1, self._max_n_subjects))
            ),
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

        subject_embeddings, kl_cost = disrnn.information_bottleneck(
            inputs=subject_embeddings,
            sigmas=self._subj_emb_global_sigma,
            noiseless_mode=self._noiseless_mode,
        )
        penalty += self._subj_penalty * kl_cost

        subj_emb_for_update_net = jnp.tile(
            jnp.expand_dims(subject_embeddings, 2),
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
            inputs=subject_embeddings,
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
