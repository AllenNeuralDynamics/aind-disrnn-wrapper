from __future__ import annotations

import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from omegaconf import DictConfig, OmegaConf

import aind_disrnn_utils.data_loader as dl
from aind_disrnn_utils.data_models import disRNNInputSettings, disRNNOutputSettings
from disentangled_rnns.library import disrnn, plotting, rnn_utils

from capsule_core.interfaces import ModelTrainer
from capsule_core.types import DatasetBundle, TrainerResult

logger = logging.getLogger(__name__)


def _to_dict(config: Mapping[str, Any] | DictConfig) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


class DisrnnTrainer(ModelTrainer):
    """Trainer that reproduces the legacy disRNN pipeline."""

    def __init__(
        self,
        architecture: Mapping[str, Any] | DictConfig,
        penalties: Mapping[str, Any] | DictConfig,
        training: Mapping[str, Any] | DictConfig,
        wandb: Mapping[str, Any] | DictConfig | None = None,
        output_dir: str = "/results",
        seed: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__(seed=seed)
        self.architecture = _to_dict(architecture)
        self.penalties = _to_dict(penalties)
        self.training = _to_dict(training)
        self.wandb_cfg = _to_dict(wandb) if wandb is not None else {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit(self, bundle: DatasetBundle) -> TrainerResult:
        metadata = dict(bundle.metadata)
        subject_ids = metadata.get("subject_ids", [])
        ignore_policy = metadata.get("ignore_policy", "exclude")
        features = metadata.get("features", {})
        multisubject = metadata.get("multisubject", False)
        seed = metadata.get("seed", self.seed)
        if seed is None:
            raise ValueError("Training seed must be provided via data config or trainer config.")

        dataset = bundle.extras.get("dataset") if bundle.extras else None
        if dataset is None:
            raise ValueError("Dataset bundle must include the constructed disRNN dataset.")

        dataset_train = bundle.train
        dataset_eval = bundle.eval

        output = {
            "num_trials": metadata.get("num_trials"),
            "num_sessions": metadata.get("num_sessions"),
        }

        logger.info("Dataset details: input %s, output %s", dataset._xs.shape, dataset._ys.shape)
        logger.info(
            "Train/Eval shapes: train=%s eval=%s",
            dataset_train._ys.shape,
            dataset_eval._ys.shape,
        )

        rng_keys = bundle.rng_keys or {}
        key = rng_keys.get("primary")
        if key is None:
            key = jax.random.PRNGKey(seed)
            rng_keys["primary"] = key
        warmup_key = rng_keys.get("warmup")
        training_key = rng_keys.get("training")
        if warmup_key is None or training_key is None:
            warmup_key, training_key = jax.random.split(key)
            rng_keys["warmup"] = warmup_key
            rng_keys["training"] = training_key
        output["random_key"] = [int(x) for x in np.asarray(key).reshape(-1)]

        args = disRNNInputSettings(
            multisubject=multisubject,
            subject_ids=list(subject_ids),
            ignore_policy=ignore_policy,
            features=features,
            num_latents=self.architecture["latent_size"],
            update_net_n_units_per_layer=self.architecture["update_net_n_units_per_layer"],
            update_net_n_layers=self.architecture["update_net_n_layers"],
            choice_net_n_units_per_layer=self.architecture["choice_net_n_units_per_layer"],
            choice_net_n_layers=self.architecture["choice_net_n_layers"],
            activation=self.architecture["activation"],
            latent_penalty=self.penalties["latent_penalty"],
            choice_net_latent_penalty=self.penalties["choice_net_latent_penalty"],
            update_net_obs_penalty=self.penalties["update_net_obs_penalty"],
            update_net_latent_penalty=self.penalties["update_net_latent_penalty"],
            n_steps=self.training["n_steps"],
            n_warmup_steps=self.training["n_warmup_steps"],
            learning_rate=self.training["lr"],
            loss=self.training["loss"],
            loss_param=self.training["loss_param"],
        )

        wandb_entity = self.wandb_cfg.get("entity")
        wandb_project = self.wandb_cfg.get("project", "disrnn")
        wandb_group = self.wandb_cfg.get("group") or f"subject_{subject_ids}"
        wandb_job_type = self.wandb_cfg.get("job_type", "train")
        wandb_dir = self.wandb_cfg.get("dir", str(self.output_dir))
        wandb_name = self.wandb_cfg.get("name") or f"disrnn_{subject_ids}_beta_{self.penalties['latent_penalty']}"

        run = wandb.init(
            entity=wandb_entity,
            project=wandb_project,
            name=wandb_name,
            group=wandb_group,
            job_type=wandb_job_type,
            config=args.model_dump(),
            dir=wandb_dir,
        )
        run.config.update({"CO_COMPUTATION_ID": os.environ.get("CO_COMPUTATION_ID")})

        output_size = 2 if ignore_policy == "exclude" else 3
        disrnn_config = disrnn.DisRnnConfig(
            obs_size=dataset._xs.shape[2],
            output_size=output_size,
            x_names=dataset.x_names,
            y_names=dataset.y_names,
            latent_size=args.num_latents,
            update_net_n_units_per_layer=args.update_net_n_units_per_layer,
            update_net_n_layers=args.update_net_n_layers,
            choice_net_n_units_per_layer=args.choice_net_n_units_per_layer,
            choice_net_n_layers=args.choice_net_n_layers,
            activation=args.activation,
            noiseless_mode=False,
            latent_penalty=args.latent_penalty,
            choice_net_latent_penalty=args.choice_net_latent_penalty,
            update_net_obs_penalty=args.update_net_obs_penalty,
            update_net_latent_penalty=args.update_net_latent_penalty,
        )

        noiseless_network = copy.deepcopy(disrnn_config)
        noiseless_network.latent_penalty = 0
        noiseless_network.choice_net_latent_penalty = 0
        noiseless_network.update_net_obs_penalty = 0
        noiseless_network.update_net_latent_penalty = 0
        noiseless_network.l2_scale = 0
        noiseless_network.noiseless_mode = True

        logger.info("Running warmup training phase")
        warmup_start = time.time()
        params, warmup_opt_state, warmup_losses = rnn_utils.train_network(
            lambda: disrnn.HkDisentangledRNN(noiseless_network),
            dataset_train,
            dataset_eval,
            opt=optax.adam(args.learning_rate),
            loss=args.loss,
            loss_param=args.loss_param,
            n_steps=args.n_warmup_steps,
            random_key=warmup_key,
        )
        warmup_duration = time.time() - warmup_start
        warmup_path = self._plot_losses(
            warmup_losses,
            title="Loss over warmup training",
            output_name="warmup_validation.png",
        )
        wandb.log({"fig/warmup_loss_curve": wandb.Image(str(warmup_path))})

        logger.info("Running full training phase")
        start = time.time()
        params, opt_state, losses = rnn_utils.train_network(
            lambda: disrnn.HkDisentangledRNN(disrnn_config),
            dataset_train,
            dataset_eval,
            loss=args.loss,
            loss_param=args.loss_param,
            params=params,
            opt_state=None,
            opt=optax.adam(args.learning_rate),
            n_steps=args.n_steps,
            step_offset=args.n_warmup_steps,
            do_plot=True,
            random_key=training_key,
        )
        training_time = time.time() - start
        output["training_time"] = training_time

        losses_path = self._plot_losses(
            losses,
            title="Loss over Training",
            output_name="validation.png",
        )
        wandb.log({"fig/validation_loss_curve": wandb.Image(str(losses_path))})

        bottlenecks_fig = plotting.plot_bottlenecks(params, disrnn_config, sort_latents=False)
        bottlenecks_path = self._save_figure(bottlenecks_fig, "bottlenecks.png")
        wandb.log({"fig/bottlenecks": wandb.Image(str(bottlenecks_path))})

        choice_fig = plotting.plot_choice_rule(params, disrnn_config)
        if choice_fig is not None:
            choice_path = self._save_figure(choice_fig, "choice_rule.png")
            wandb.log({"fig/choice_rule": wandb.Image(str(choice_path))})

        update_figs = plotting.plot_update_rules(params, disrnn_config)
        for index, fig in enumerate(update_figs):
            path = self._save_figure(fig, f"update_rule_{index}.png")
            wandb.log({f"fig/update_rule_{index}": wandb.Image(str(path))})

        xs, ys = next(dataset_eval)
        yhat, network_states = rnn_utils.eval_network(
            lambda: disrnn.HkDisentangledRNN(noiseless_network), params, xs
        )

        df = bundle.raw
        output_df = dl.add_model_results(
            df, network_states.__array__(), yhat, ignore_policy=ignore_policy
        )
        output_path = self.output_dir / "output_df.csv"
        output_df.to_csv(output_path, index=False)

        params_path = self.output_dir / "params.json"
        with params_path.open("w") as f:
            f.write(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))

        wandb.summary["final/val_loss"] = float(losses["validation_loss"][-1])
        wandb.summary["final/train_loss"] = float(losses["training_loss"][-1])
        wandb.summary["elapsed_seconds"] = float(training_time)
        wandb.summary["warmup_seconds"] = float(warmup_duration)

        likelihood = rnn_utils.normalized_likelihood(ys, yhat[:, :, 0:2])
        output["likelihood"] = float(likelihood)
        run.finish()

        output_settings = disRNNOutputSettings(**output)
        return TrainerResult(
            output=output_settings,
            metrics={
                "final/val_loss": float(losses["validation_loss"][-1]),
                "final/train_loss": float(losses["training_loss"][-1]),
                "likelihood": float(likelihood),
            },
            extras={
                "params_path": str(params_path),
                "output_df_path": str(output_path),
                "warmup_seconds": warmup_duration,
            },
        )

    def _plot_losses(self, losses: Mapping[str, Any], title: str, output_name: str) -> Path:
        fig = plt.figure()
        plt.semilogy(losses["training_loss"], color="black")
        plt.semilogy(losses["validation_loss"], color="tab:red", linestyle="dashed")
        plt.xlabel("Training Step")
        plt.ylabel("Mean Loss")
        plt.legend(("Training Set", "Validation Set"))
        plt.title(title)
        path = self._save_figure(fig, output_name)
        return path

    def _save_figure(self, fig: Any, filename: str) -> Path:
        path = self.output_dir / filename
        fig.savefig(path)
        plt.close(fig)
        return path
