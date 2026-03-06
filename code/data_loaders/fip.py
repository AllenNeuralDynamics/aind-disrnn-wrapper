from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Literal, Mapping

import numpy as np
import pandas as pd

from aind_dynamic_foraging_data_utils import nwb_utils as nu
import aind_dynamic_foraging_basic_analysis.licks.annotation as a
import aind_dynamic_foraging_basic_analysis.metrics.trial_metrics as tm
import aind_dynamic_foraging_data_utils.code_ocean_utils as co
import aind_dynamic_foraging_data_utils.enrich_dfs as ed
import aind_dynamic_foraging_multisession_analysis.multisession_load as ms_load
import aind_disrnn_utils.data_loader as dl
from disentangled_rnns.library import rnn_utils

from base.interfaces import DatasetLoader
from base.types import DatasetBundle

logger = logging.getLogger(__name__)

def make_fip_multisession_trials_df(nwb_list, allow_duplicates=True):
    """
    Builds a dataframe of trials concatenated across multiple sessions
    nwb_list, a list of NWBs to concatenate. Can either be paths to the files
            or NWB files themselves

    The multisession dataframe will contain the union of the columns in the
    individual nwb.df_trials. The rows will be sorted by the session date,
    and then trial number within each session
    """
    # Hard coding these for now
    fiber = 0
    channel = 'G'
    method='dff-bright_mc-iso-IRLS'
    full_channel_name = f'{channel}_{fiber}_{method}'

    unique_sessions = set()
    nwbs = []
    crash_list = []
    for n in nwb_list:
        print(n)
        try:
            nwb = nu.load_nwb_from_filename(n)
            if (not allow_duplicates) and (nwb.session_id in unique_sessions):
                continue
            else:
                unique_sessions.add(nwb.session_id)
            nwb.df_trials = nu.create_df_trials(nwb, verbose=False)
            nwb.df_events = nu.create_df_events(nwb, verbose=False)
            nwb.df_licks = a.annotate_licks(nwb)
            nwb.df_trials = tm.compute_trial_metrics(nwb)
            nwb.df_trials = ms_load.add_side_bias(nwb)
            nwb.df_fip = nu.create_df_fip(nwb,verbose=False)
            nwb.df_fip = ed.zscore_fip(nwb.df_fip)
            nwb.df_trials = tm.get_average_signal_window(
                nwb, 
                alignment_event='goCue_start_time_in_session',
                offsets = [0,2],
                channel=full_channel_name,
                data_column = 'data_z',
                output_col='NE_FIP'
            )
            nwbs.append(nwb)
        except Exception as e:
            crash_list.append(n)
            print("Bad {}".format(n))
            print("   " + str(e))

    # Log summary of sessions with loading errors
    if len(crash_list) > 0:
        print("\n\nThe following sessions could not be loaded")
        print("\n".join(crash_list))

    # Make a dataframe of trials
    for nwb in nwbs:
        nwb.df_trials["ses_idx"] = [nwb.session_id[9:]] * len(nwb.df_trials)
    df = pd.concat([x.df_trials for x in nwbs])

    return nwbs, df



class FipDatasetLoader(DatasetLoader):
    """
    Load the foraging dataset for mice experiments with FIP data

    I am doing this as a dev project, so I'm hard coding the 
    Green channel of Fiber 0, which is Prelimbic cortex with
    GCaMP LC-NE axon imaging in the mice I'm using. In the future,
    I will need to generalize this code to accept a data model that
    can parameterize the fiber, channel, and aggregation statistics
    to use. I'm accepting this technical debt now out of urgency to explore
    """

    def __init__(
        self,
        subject_ids: Iterable[int],
        ignore_policy: str,
        features: Mapping[str, str],
        eval_every_n: int,
        multisubject: bool = False,
        batch_size: int | None = None,
        batch_mode: Literal["single", "rolling", "random"] = "random",
        seed: int | None = None,
        **extras: object,
    ) -> None:
        super().__init__(seed=seed)
        self.subject_ids = list(subject_ids)
        self.ignore_policy = ignore_policy
        self.features = dict(features)
        self.eval_every_n = eval_every_n
        self.multisubject = multisubject
        self.batch_size = batch_size
        self.batch_mode = batch_mode
        self.extras = extras


    def load(self) -> DatasetBundle:
        
        # Fix numpy random seed (will affect batch_mode="random")
        if self.seed is not None:
            np.random.seed(self.seed)
        
        if self.multisubject:
            raise NotImplementedError("Multisubject loading is not yet supported.")

        subject = self.subject_ids[0]
        new_mice = [774212, 779531, 781173, 781162, 778077]
        old_mice = []
        if subject in new_mice:
            results = []
            for subject in new_mice: # TODO hack because data is missing on docdb
                logger.info("Querying docDB for {}".format(subject))
                subject_results = co.get_subject_assets(
                    subject, modality=["behavior","fib"], stage=["STAGE_FINAL", "GRADUATED"]
                )  
                subject_results = subject_results.sort_values(
                    by="session_name"
                ).reset_index(drop=True)
                results.append(subject_results)
            results = pd.concat(results)
            if len(results) == 0:
                raise Exception("No data found for subject ids")
            logger.info("Getting s3 location")
            results = co.add_s3_location(results)
            logger.info("Loading NWBs")
            nwbs, df = make_fip_multisession_trials_df(results["s3_nwb_location"])
        elif subject in old_mice:
            raise Exception('not implemented yet')

        dataset = dl.create_disrnn_dataset(
            df,
            ignore_policy=self.ignore_policy,
            features=self.features,
            batch_size=self.batch_size,
            batch_mode=self.batch_mode,
        )
        dataset_train, dataset_eval = rnn_utils.split_dataset(
            dataset, eval_every_n=self.eval_every_n
        )
        metadata = {
            "subject_ids": self.subject_ids,
            "ignore_policy": self.ignore_policy,
            "features": self.features,
            "eval_every_n": self.eval_every_n,
            "num_trials": len(df),
            "num_sessions": int(df["ses_idx"].nunique()) if "ses_idx" in df else None,
            "multisubject": self.multisubject,
            "batch_size": self.batch_size,
            "batch_mode": self.batch_mode,
        }
        metadata.update(self.extras)

        extras = {"dataset": dataset}
        return DatasetBundle(
            raw=df,
            train_set=dataset_train,
            eval_set=dataset_eval,
            metadata=metadata,
            extras=extras,
        )
