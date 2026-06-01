import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
    force=True,
)

from post_training_analysis import (
    run_likelihood_advantage_analysis,
    run_rnn_state_space_condition_analysis,
)

# result = run_likelihood_advantage_analysis(
#     "/data/mice_multisubject_train10_all_stages-gru_session_conditioning-260505-14-H64_lr5e_06-b97c06b79f07",
#     "/data/mice_multisubject_train10_all_stages-baseline_rl_Bari-260520/1",
#     split="combined",   # or "train", "eval", "heldout_test"
#     checkpoint_policy="best_eval",   # used for GRU/disRNN model1, "best_eval", "final"
#     output_dir="/results",
#     history_warmup=10,
#     top_k_variables=3,
#     jitter_seed=0,
#     include_rnn_state_space=True,
#     pca_seed=0,
#     pca_fit_fraction=0.5,
# )

print(result)

standalone_state_space_result = run_rnn_state_space_condition_analysis(
    "/data/trial_advantage.pkl",
    # condition_columns=["switch_x_prev_outcome", "trial_position"],
    output_dir="/results/figures/rnn_state_space_standalone",
    pca_seed=0,
)

print(standalone_state_space_result)
