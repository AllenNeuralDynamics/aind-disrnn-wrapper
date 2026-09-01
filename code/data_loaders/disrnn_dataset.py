"""disRNN dataset construction from AIND dynamic-foraging trial dataframes.

VENDORED from ``aind_disrnn_utils`` (``src/aind_disrnn_utils/data_loader.py``)
at pin ``74de874d93b951d9ef3ae6bff6453e6ae805b649``, the SHA this repo depended
on before the package dependency was dropped.

Why it lives here now
---------------------
The upstream package was a two-file repo whose only consumers were this repo
and the dispatcher, with no remaining maintainer. Keeping the two functions we
actually call as a git-pinned dependency cost more than it returned: a perf fix
sat unmerged upstream for over two months while production kept running the slow
path, and a dtype bug had to be worked around in-tree
(``utils.trial_timing_features.create_disrnn_dataset_float``) rather than fixed
at the source. Its ``data_models.py`` (``disRNNInputSettings`` /
``disRNNOutputSettings``) was NOT vendored: nothing imported it — Hydra configs
are this stack's configuration surface.

The initial vendoring commit is a faithful copy: function bodies are
byte-identical to the pinned upstream, so the move is auditable by diff. Changes
since then are ordinary commits in this repo's history.

See: AllenNeuralDynamics/aind-disrnn-dispatcher#74 (identity rename), which this
unblocks by removing the third repo from its scope.
"""

import pandas as pd
import numpy as np
from disentangled_rnns.library import rnn_utils


def create_disrnn_dataset(
    df_trials,
    ignore_policy="include",
    batch_size=None,
    batch_mode="random",
    features=None,
) -> rnn_utils.DatasetRNN:
    """
    Creates a disrnn dataset object

    args:
    df_trials, a trial dataframe, created by aind_dynamic_foraging_data_utils
        must have 'ses_idx' as an column which indicates how to divide
        trials by session
    ignore_policy (str), must be "include" or "exclude", and determines
        how to use trials where the mouse did not response
    batch_size (int) input argument to disrnn dataset
    features (dict), keys must be columns in df_trials to be used as prediction
        features. values are the semantic labels for that feature. If None,
        use previous choice and previous reward
    """

    # Input checking
    if "ses_idx" not in df_trials:
        raise ValueError("df_trials must contain index of sessions ses_idx")
    if ignore_policy not in ["include", "exclude"]:
        raise ValueError('ignore_policy must be either "include" or "exclude"')

    # Copy so we can modify
    df_trials = df_trials.copy()

    # Determine the number of classes in the output prediction
    if ignore_policy == "include":
        n_classes = 3
    else:
        n_classes = 2
        # Remove trials without a response
        df_trials = df_trials[df_trials["animal_response"] != 2]

    # Format inputs
    # Make 0/1 coded reward vector
    df_trials["rewarded"] = df_trials["earned_reward"].astype(int)

    # Break down feature dictionary
    if features is None:
        features = {
            "animal_response": "prev choice",
            "rewarded": "prev reward",
        }
    feature_cols = list(features.keys())
    feature_labels = [features[x] for x in feature_cols]

    # Ensure all feature columns are in df_trials
    for feature in feature_cols:
        if feature not in df_trials.columns:
            raise ValueError(
                "input feature '{}' not in df_trials".format(feature)
            )

    # Determine size of input matrix
    # Input matrix has size [# trials, # sessions, # features]
    max_session_length = df_trials.groupby("ses_idx")["trial"].count().max()

    num_sessions = len(df_trials["ses_idx"].unique())
    num_input_features = len(feature_cols)
    # Pad trials to be ignored with -1
    xs = np.full((max_session_length, num_sessions, num_input_features), -1)

    # Load each session into xs
    for dex, ses_idx in enumerate(df_trials["ses_idx"].unique()):
        temp = df_trials.query("ses_idx == @ses_idx")
        this_xs = temp[feature_cols].to_numpy()[:-1, :]
        xs[1 : len(temp), dex, :] = this_xs  # noqa E203

    # Determine size of output matrix
    # Output matrix has size [# trials, # sessions, # features]
    num_output_features = 1
    # pad trials to be ignored with -1
    ys = np.full((max_session_length, num_sessions, num_output_features), -1)

    # Load each session into ys
    for dex, ses_idx in enumerate(df_trials["ses_idx"].unique()):
        temp = df_trials.query("ses_idx == @ses_idx")
        this_ys = temp[["animal_response"]].to_numpy()
        ys[0 : len(temp), dex, :] = this_ys  # noqa E203

    # Pack into a DatasetRNN object
    dataset = rnn_utils.DatasetRNN(
        ys=ys.astype(float),
        xs=xs.astype(float),
        y_type="categorical",
        n_classes=n_classes,
        x_names=feature_labels,
        y_names=["choice"],
        batch_size=batch_size,
        batch_mode=batch_mode,
    )
    return dataset


def add_model_results(
    df_trials, network_states, yhat, ignore_policy="exclude"
):
    """
    Integrates the network_states and y-hat predictions from a disRNN model
    into the trials dataframe so they can be analyzed.

    args:
    df_trials (dataframe), the trials dataframe from which the disrnn dataset
        was created. Must have columns `ses_idx`, `trials`, `animal_response`
    network_states (np array), the latent states of the network with dimensions
        (max_trial, sessions, num latents)
    yhat (np array), the predictions of the network with dimensions
        (max_trial, sessions, num_choices + 1)
    ignore_policy (str) "exclude" or "include"
    """
    # Make sure input is the correct size
    if len(df_trials["ses_idx"].unique()) != np.shape(yhat)[1]:
        raise Exception("number of sessions in df_trials and yhat differ")
    if (ignore_policy == "exclude") and (np.shape(yhat)[2] == 3):
        columns = ["logit(left)", "logit(right)"]
    elif (ignore_policy == "include") and (np.shape(yhat)[2] == 4):
        columns = ["logit(left)", "logit(right)", "logit(ignore)"]
    else:
        raise Exception(
            "Unknown combination of ignore_policy and yhat dimensions"
        )

    # Determine number of latents, and make column labels
    num_latents = np.shape(network_states)[2]
    columns = columns + ["latent_" + str(x + 1) for x in range(num_latents)]

    # Iterate through dimensions of yhat and load back into df_trials
    temps = []
    sessions = df_trials["ses_idx"].unique()
    for index, session in enumerate(sessions):
        temp_df = pd.DataFrame(
            np.concatenate(
                [yhat[:, index, :-1], network_states[:, index, :]], axis=1
            ),
            columns=columns,
        )
        temp_df["ses_idx"] = session
        if ignore_policy == "exclude":
            trials = np.array([-1] * len(temp_df))
            x = (
                df_trials.query("ses_idx ==@session")
                .query("animal_response in [0,1]")["trial"]
                .values
            )
            trials[: len(x)] = x
            temp_df = temp_df[trials >= 0].copy()
            temp_df["trial"] = x
        else:
            trials = np.array([-1] * len(temp_df))
            x = df_trials.query("ses_idx ==@session")["trial"].values
            trials[: len(x)] = x
            temp_df = temp_df[trials >= 0].copy()
            temp_df["trial"] = x
        temps.append(temp_df)
    temp_df = pd.concat(temps)
    df_trials = pd.merge(
        df_trials, temp_df, on=["ses_idx", "trial"], how="left"
    )

    if ignore_policy == "exclude" and np.any(
        df_trials["animal_response"] == 2
    ):
        assert (
            np.mean(
                df_trials[df_trials["logit(right)"].isnull()][
                    "animal_response"
                ].values
            )
            == 2
        ), "NaN value for non-ignored trial"
        assert (
            np.mean(
                df_trials[df_trials["logit(left)"].isnull()][
                    "animal_response"
                ].values
            )
            == 2
        ), "NaN value for non-ignored trial"
        assert np.all(
            df_trials.query("animal_response == 2")["logit(right)"]
            .isnull()
            .values
        ), "Non NaN value for ignored trial"
        assert np.all(
            df_trials.query("animal_response == 2")["logit(left)"]
            .isnull()
            .values
        ), "Non NaN value for ignored trial"
    elif ignore_policy == "include":
        assert np.sum(df_trials["logit(right)"].isnull()) == 0, "NaN values"
        assert np.sum(df_trials["logit(left)"].isnull()) == 0, "NaN values"
    return df_trials
