from __future__ import annotations

import copy
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Mapping

import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from omegaconf import DictConfig, OmegaConf
import haiku as hk


import aind_disrnn_utils.data_loader as dl
from aind_disrnn_utils.data_models import disRNNInputSettings
from disentangled_rnns.library import disrnn, plotting, rnn_utils

from base.interfaces import ModelTrainer
from base.types import DatasetBundle

logger = logging.getLogger(__name__)


def _to_dict(config: Mapping[str, Any] | DictConfig) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)  # type: ignore[return-value]
    return dict(config)


class GrurnnTrainer(ModelTrainer):
    """Trainer that reproduces the legacy disRNN pipeline."""

    def __init__(
        self,
        architecture: Mapping[str, Any] | DictConfig,
        # penalties: Mapping[str, Any] | DictConfig, # i don't think we have penalties here?
        training: Mapping[str, Any] | DictConfig,
        output_dir: str = "/results/outputs",
        seed: int | None = None,
        **_: Any,
    ) -> None:
        super().__init__(seed=seed)
        self.architecture = _to_dict(architecture)
        # self.penalties = _to_dict(penalties)
        self.training = _to_dict(training)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def fit(
        self,
        bundle: DatasetBundle,
        loggers: dict[str, Any] | None = None,
    ):
        metadata = dict(bundle.metadata)
        ignore_policy = metadata.get("ignore_policy", "exclude")

        wandb_run = None
        if loggers and "wandb" in loggers:
            wandb_run = loggers["wandb"]

        dataset = bundle.extras.get("dataset") if bundle.extras else None
        if dataset is None:
            raise ValueError("Dataset bundle must include the constructed disRNN dataset.")

        dataset_train = bundle.train_set
        dataset_eval = bundle.eval_set
        if dataset_train is None or dataset_eval is None:
            raise ValueError("Dataset bundle must include train and eval splits.")

        # TODO: fix this later? somehow this is an issue with grn_baseline
        dataset._xs = dataset._xs.astype(float)
        dataset_train._xs = dataset_train._xs.astype(float)
        dataset_eval._xs = dataset_eval._xs.astype(float)
        
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

        key = jax.random.PRNGKey(self.seed)
        warmup_key, training_key = jax.random.split(key)
        output["random_key"] = [int(x) for x in np.asarray(key).reshape(-1)]


        # pull parameters (may need to make this pydiantic eventually?)
        n_hidden = self.architecture['n_hidden']
        output_size = 2 if ignore_policy == "exclude" else 3
        learning_rate = self.training['lr']
        n_steps = self.training['n_steps']
        loss = self.training['loss']
        loss_param = self.training['loss_param']
        
        
        
        # make network- i assume this should be taken out of def fit? 
        def make_network():
            model = hk.DeepRNN(
            [hk.GRU(n_hidden), hk.Linear(output_size=output_size)]
            )
            return model
       
                
        # TODO: merge this pydantic validation step into disrnn.DisRnnConfig
        # Note that I already removed data-specific fields like subject_ids here,
        # since they are not needed for ModelTrainer class
        
        # INITIALIZE THE NETWORK
        # Running rnn_utils.train_network with n_steps=0 does no training but sets up the
        # parameters and optimizer state.
        logger.info("Initializing the network")

        optimizer = optax.adam(learning_rate=learning_rate)

        params, opt_state, losses = rnn_utils.train_network(
            make_network = make_network,
            training_dataset=dataset_train,
            validation_dataset=dataset_eval,
            opt = optimizer,
            loss=loss,
            n_steps=0)


        logger.info("Running full training phase")
        start = time.time()
        params, opt_state, losses = rnn_utils.train_network(
            make_network = make_network,
            training_dataset=dataset_train,
            validation_dataset=dataset_eval,
            loss=loss,
            params=params,
            opt_state=opt_state,
            opt = optax.adam(learning_rate=learning_rate),
            loss_param = loss_param,
            n_steps=n_steps,
            do_plot = True)
        training_time = time.time() - start
        output["training_time"] = training_time

        losses_path = self._plot_losses(
            losses,
            title="Loss over Training",
            output_name="validation.png",
        )
        if wandb_run is not None:
            wandb_run.log({"fig/validation_loss_curve": wandb.Image(str(losses_path))})

        # example of plotting figs to wandb       
        # update_figs = plotting.plot_update_rules(params, disrnn_config)
        # for index, fig in enumerate(update_figs):
        #     path = self._save_figure(fig, f"update_rule_{index}.png")
        #     if wandb_run is not None:
        #         wandb_run.log({f"fig/update_rule_{index}": wandb.Image(str(path))})

        # Run forward pass on the unseen data
        xs_eval, ys_eval = dataset_eval.get_all()
        yhat, network_states = rnn_utils.eval_network(make_network, params, xs_eval)


        df = bundle.raw
        output_df = dl.add_model_results(
            df, network_states.__array__(), yhat, ignore_policy=ignore_policy
        )
        output_path = self.output_dir / "output_df.csv"
        output_df.to_csv(output_path, index=False)

        params_path = self.output_dir / "params.json"
        with params_path.open("w") as f:
            f.write(json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder))

        # Compute normalized likelihood
        likelihood = rnn_utils.normalized_likelihood(ys_eval, yhat)
        output["likelihood"] = float(likelihood)

        # save output to json
        with open(self.output_dir / "output_summary.json", "w") as f:
            json.dump(output, f, indent=4)

        if wandb_run is not None:
            wandb_run.summary["final/val_loss"] = float(losses["validation_loss"][-1])
            wandb_run.summary["final/train_loss"] = float(losses["training_loss"][-1])
            wandb_run.summary["likelihood"] = float(likelihood)
            wandb_run.summary["elapsed_seconds"] = float(training_time)
            
            # Upload the whole /results/output folder as an artifact
            # Here I'm using the random id as the name, meaning each run will has its own artifact.
            artifact_name = f"grurnn-output-{getattr(wandb_run, 'id', None) or 'latest'}"
            artifact = wandb.Artifact(artifact_name, type="training-output")
            artifact.add_dir(str(self.output_dir))
            wandb_run.log_artifact(artifact)

        return output

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
