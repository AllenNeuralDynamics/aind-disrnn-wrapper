"""top level run script"""

import copy
import json
import logging
import os
import sys
import time
from types import SimpleNamespace
from pathlib import Path

import s3fs
import aind_disrnn_utils.data_loader as dl
import aind_dynamic_foraging_data_utils.code_ocean_utils as co
import aind_dynamic_foraging_multisession_analysis.multisession_load as ms_load
import jax
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd
from omegaconf import OmegaConf
from aind_disrnn_utils.data_models import disRNNOutputSettings, disRNNInputSettings
from disentangled_rnns.library import disrnn, plotting, rnn_utils

import wandb
import shutil

logger = logging.getLogger(__name__)


def find_hydra_config(logger):
    """
    Search for config.yaml under /data/jobs/.
    Returns the first found config.yaml path, or None if not found. Logs warnings if missing or multiple.
    """
    config_candidates = list(Path("/data/jobs").rglob("config.yaml"))
    if not config_candidates:
        logger.warning("No config.yaml found under /data/jobs/")
        return None
    elif len(config_candidates) > 1:
        logger.warning(
            f"Multiple config.yaml files found: {config_candidates}. Using the first one."
        )
        return config_candidates[0]
    else:
        return config_candidates[0]


if __name__ == "__main__":
    # set up a logger
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(levelname)s:%(asctime)s:%(filename)s:%(lineno)d:    %(message)s",
        datefmt="%Y-%m-%d %H-%M-%S",
    )

    # Find config path
    hydra_config = find_hydra_config(logger)
    if find_hydra_config is None:
        logger.error("No config.yaml found. Exiting.")
        sys.exit(1)

    # Copy input hydra folder to results for record-keeping
    source_dir = hydra_config.resolve().parents[1]
    destination_root = Path("/results/input")
    destination_root.mkdir(parents=True, exist_ok=True)
    destination_dir = destination_root / source_dir.name
    logger.info("Copying %s to %s", source_dir, destination_dir)
    shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)

    # Load Hydra config with OmegaConf
    logger.info("loading Hydra config from %s", hydra_config)
    hydra_config = OmegaConf.load(hydra_config)
    data_cfg = hydra_config.data
    model_cfg = hydra_config.model
    arch_cfg = model_cfg.architecture
    penalties_cfg = model_cfg.penalties
    training_cfg = model_cfg.training

    args = disRNNInputSettings(  # To match Alex's args structure
        multisubject=data_cfg.multisubject,
        subject_ids=list(data_cfg.subject_ids),
        ignore_policy=data_cfg.ignore_policy,
        features=data_cfg.features,
        # eval_every_n=data_cfg.eval_every_n,
        num_latents=arch_cfg.latent_size,
        update_net_n_units_per_layer=arch_cfg.update_net_n_units_per_layer,
        update_net_n_layers=arch_cfg.update_net_n_layers,
        choice_net_n_units_per_layer=arch_cfg.choice_net_n_units_per_layer,
        choice_net_n_layers=arch_cfg.choice_net_n_layers,
        activation=arch_cfg.activation,
        latent_penalty=penalties_cfg.latent_penalty,
        choice_net_latent_penalty=penalties_cfg.choice_net_latent_penalty,
        update_net_obs_penalty=penalties_cfg.update_net_obs_penalty,
        update_net_latent_penalty=penalties_cfg.update_net_latent_penalty,
        n_steps=training_cfg.n_steps,
        n_warmup_steps=training_cfg.n_warmup_steps,
        learning_rate=training_cfg.lr,
        loss=training_cfg.loss,
        loss_param=training_cfg.loss_param,
    )

    seed = hydra_config.seed if "seed" in hydra_config else None
    if seed is None:
        seed = int(time.time())
        logger.warning("No seed provided in config; using fallback seed %s", seed)
    else:
        logger.info("Using seed from config: %s", seed)

    OmegaConf.save(config=hydra_config, f="/results/inputs.yaml", resolve=True)
    logger.info("Saved resolved config to /results/inputs.yaml")

    # Start wandb run
    wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=hydra_config.wandb.entity,
        # Set the wandb project where this run will be logged.
        project="han-test",
        name=f"disrnn_{args.subject_ids}_beta_{args.latent_penalty}",
        group=f"subject_{args.subject_ids}",
        job_type="train",
        config=args,
        dir="/results",
    )

    # Log CodeOcean computation ID
    wandb.config.update({"CO_COMPUTATION_ID": os.environ.get("CO_COMPUTATION_ID")})

    # Haven't implemented multisubject rnns yet
    if args.multisubject:
        logger.error("Multisubject not yet supported")
        sys.exit()

    # Load Data
    subject_dfs = []
    results = []
    for subject in args.subject_ids:
        logger.info("Querying docDB for {}".format(subject))
        mouse_results = co.get_subject_assets(
            mouse, modality=["behavior"], stage=["STAGE_FINAL", "GRADUATED"]
        ) # TODO, should we expose these filters?
        mouse_results = mouse_results.sort_values(by="session_name").reset_index(
            drop=True
        )
        results.append(mouse_results)
    results = pd.concat(results)

    # TODO, filter sessions by performance

    logger.info('Getting s3 location')
    results = co.add_s3_location(results)

    logger.info("Loading NWBs")
    nwbs, df = ms_load.make_multisession_trials_df(results["s3_nwb_location"])

    np.random.seed(seed)

    # Create disrnn dataset
    dataset = dl.create_disrnn_dataset(
        df, ignore_policy=args.ignore_policy, features=args.features
    )
    dataset_train, dataset_eval = rnn_utils.split_dataset(
        dataset, eval_every_n=data_cfg.eval_every_n
    )

    # Setup output model
    output = {}
    output["num_trials"] = len(df)
    output["num_sessions"] = len(df["ses_idx"].unique())

    # Log details of dataset
    logger.info("Dataset details:")
    logger.info("Input size: {}".format(dataset._xs.shape))
    logger.info("Output size: {}".format(dataset._ys.shape))
    logger.info("Train output: {}".format(dataset_train._ys.shape))
    logger.info("Test output: {}".format(dataset_eval._ys.shape))

    # Set up random splits
    logger.info("setting up random splits")
    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    output["random_key"] = list(key.__array__())

    # Configure Network
    logger.info("Configuring network")
    output_size = 2 if args.ignore_policy == "exclude" else 3
    # Choose left / choose right, maybe ignore
    disrnn_config = disrnn.DisRnnConfig(
        # Dataset related
        obs_size=dataset._xs.shape[2],
        output_size=output_size,
        x_names=dataset.x_names,
        y_names=dataset.y_names,
        # Network architecture
        latent_size=args.num_latents,
        update_net_n_units_per_layer=args.update_net_n_units_per_layer,
        update_net_n_layers=args.update_net_n_layers,
        choice_net_n_units_per_layer=args.choice_net_n_units_per_layer,
        choice_net_n_layers=args.choice_net_n_layers,
        activation=args.activation,
        # Penalties
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

    # Initialize network
    logger.info("Initializing network")
    params, warmup_opt_state, warmup_losses = rnn_utils.train_network(
        lambda: disrnn.HkDisentangledRNN(noiseless_network),
        dataset_train,
        dataset_eval,
        opt=optax.adam(args.learning_rate),
        loss=args.loss,
        loss_param=args.loss_param,
        n_steps=args.n_warmup_steps,
        random_key=k1,
    )
    fig = plt.figure()
    plt.semilogy(warmup_losses["training_loss"], color="black")
    plt.semilogy(warmup_losses["validation_loss"], color="tab:red", linestyle="dashed")
    plt.xlabel("Training Step")
    plt.ylabel("Mean Loss")
    plt.legend(("Training Set", "Validation Set"))
    plt.title("Loss over warmup training")
    fig.savefig("/results/warmup_validation.png")

    wandb.log({"fig/warmup_loss_curve": wandb.Image("/results/warmup_validation.png")})

    # Iterate training
    logger.info("training network")
    start = time.time()
    params, opt_state, losses = rnn_utils.train_network(
        lambda: disrnn.HkDisentangledRNN(disrnn_config),
        dataset_train,
        dataset_eval,
        loss=args.loss,
        loss_param=args.loss_param,
        params=params,  # continue from warmup
        opt_state=None,
        opt=optax.adam(args.learning_rate),
        n_steps=args.n_steps,
        step_offset=args.n_warmup_steps,  # to continue wandb step counting
        do_plot=True,
        random_key=k2,
    )
    stop = time.time()
    logger.info(f"Elapsed time: {stop-start:.2f} seconds.")
    output["training_time"] = stop - start

    fig = plt.figure()
    plt.semilogy(losses["training_loss"], color="black")
    plt.semilogy(losses["validation_loss"], color="tab:red", linestyle="dashed")
    plt.xlabel("Training Step")
    plt.ylabel("Mean Loss")
    plt.legend(("Training Set", "Validation Set"))
    plt.title("Loss over Training")
    fig.savefig("/results/validation.png")

    wandb.log({"fig/validation_loss_curve": wandb.Image("/results/validation.png")})

    # Plot the open/closed state of the bottlenecks
    logger.info("Plotting state of bottlenecks")
    fig = plotting.plot_bottlenecks(params, disrnn_config, sort_latents=False)
    fig.savefig("/results/bottlenecks.png")

    wandb.log({"fig/bottlenecks": wandb.Image("/results/bottlenecks.png")})

    # Plot the choice rule
    logger.info("Plotting choice rule")
    fig = plotting.plot_choice_rule(params, disrnn_config)
    if fig is not None:
        fig.savefig("/results/choice_rule.png")
        wandb.log({"fig/choice_rule": wandb.Image("/results/choice_rule.png")})

    # Plot the update rules
    logger.info("Plotting update rules")
    figs = plotting.plot_update_rules(params, disrnn_config)
    for count, fig in enumerate(figs):
        fig.savefig("/results/update_rule_{}.png".format(count))
        wandb.log(
            {
                f"fig/update_rule_{count}": wandb.Image(
                    f"/results/update_rule_{count}.png"
                )
            }
        )

    # Evaluate the network
    xs, ys = next(dataset_eval)
    yhat, network_states = rnn_utils.eval_network(
        lambda: disrnn.HkDisentangledRNN(noiseless_network), params, xs
    )

    # Save results
    logger.info("Converting network outputs to dataframes")
    output_df = dl.add_model_results(
        df, network_states.__array__(), yhat, ignore_policy=args.ignore_policy
    )
    logger.info("saving model results")
    output_df.to_csv("/results/output_df.csv", index=False)

    logger.info("Saving model parameters")
    with open("/results/params.json", "w") as f:
        temp = json.dumps(params, cls=rnn_utils.NpJnpJsonEncoder)
        f.write(temp)
    # Load params like
    # with open(filepath) as f:
    #     params = rnn_utils.to_np(json.load(f))

    wandb.summary["final/val_loss"] = float(losses["validation_loss"][-1])
    wandb.summary["final/train_loss"] = float(losses["training_loss"][-1])
    wandb.summary["elapsed_seconds"] = float(stop - start)

    wandb.finish()

    logger.info("Output details:")
    logger.info(asset_name)
    logger.info("yhat size: {}".format(np.shape(yhat)))
    logger.info("state size: {}".format(np.shape(network_states)))

    likelihood = rnn_utils.normalized_likelihood(ys, yhat[:, :, 0:2])
    output["likelihood"] = likelihood
    logger.info(likelihood)
    logger.info("all done, goodbye")

    # log output model
    output = disRNNOutputSettings(**output)
    logger.info(output)
    json_string = output.model_dump_json(indent=2)
    file_path = "/results/outputs.json"
    with open(file_path, "w") as f:
        f.write(json_string)
