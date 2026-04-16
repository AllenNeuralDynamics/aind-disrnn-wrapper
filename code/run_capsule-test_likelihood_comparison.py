import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
    force=True,
)


from post_training_analysis import run_prediction_likelihood_comparison

result = run_prediction_likelihood_comparison(
    [
        "/data/mice_multisubject_train10-baseline_rl_Bari-260414/1",
        "/data/mice_multisubject_train10-baseline_rl_Hattori-260414/1",
        "/data/mice_multisubject_train10-baseline_rl_CTT-260415/1",
        "/data/mice_multisubject_train10-gru-260323/13",
        "/data/mice_train10_test3-gru-260324/14",
        "/data/mice_multisubject_train10-disrnn-260323/3",
    ],
    checkpoint_policy="best_eval",
    output_dir="/results",            # optional
    model_labels=[
        "Bari model",
        "Hattoril model",
        "Foraging_model",
        "GRU_subj_emb",
        "GRU_no_subj_emb",
        "disrnn_subj_emb",
    ],          # optional
    precomputed_session_metrics_path=None,
    include_heldout=False,       # optional
)

print(result)