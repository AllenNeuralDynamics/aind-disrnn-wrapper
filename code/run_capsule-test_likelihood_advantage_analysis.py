import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
    force=True,
)

from post_training_analysis import (
    run_baseline_q_space_subject_analysis,
    run_likelihood_advantage_analysis,
    run_rnn_state_space_condition_analysis,
    run_rnn_state_space_subject_analysis,
)
from post_training_analysis.likelihood_advantage_analysis import (
    run_rnn_state_space_overview_analysis,
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
#     # This baseline run does not expose recoverable policy-time Q histories.
#     # Set to True only for baseline agents that save q_rl_left/q_rl_right.
#     include_baseline_q_space=False,
# )

# print(result)

# # left_subject_state_result = run_rnn_state_space_subject_analysis(
# #     result["trial_advantage_pickle"],
# #     probability_column="p_model1_left",
# #     output_dir="/results/figures/rnn_state_space_subjects_left",
# #     pca_seed=0,
# #     pca_fit_fraction=0.5,
# # )
# # print(left_subject_state_result)

# right_subject_state_result = run_rnn_state_space_subject_analysis(
#     result["trial_advantage_pickle"],
#     probability_column="p_model1_right",
#     output_dir="/results/figures/rnn_state_space_subjects_right",
#     pca_seed=0,
#     pca_fit_fraction=0.5,
# )
# print(right_subject_state_result)

# # left_baseline_q_subject_result = run_baseline_q_space_subject_analysis(
# #     result["trial_advantage_pickle"],
# #     probability_column="p_rl_left",
# #     output_dir="/results/figures/baseline_q_space_subjects_left",
# # )
# # print(left_baseline_q_subject_result)

# if result.get("baseline_q_condition_plots") is not None:
#     right_baseline_q_subject_result = run_baseline_q_space_subject_analysis(
#         result["trial_advantage_pickle"],
#         probability_column="p_rl_right",
#         output_dir="/results/figures/baseline_q_space_subjects_right",
#     )
#     print(right_baseline_q_subject_result)


overview_state_space_result = run_rnn_state_space_overview_analysis(
    "/data/trial_advantage.pkl",
    output_dir="/results/figures/rnn_state_space_overview",
    pca_seed=0,
    pca_fit_fraction=0.5,
    n_plot_pcs=4,
)

print(overview_state_space_result)

# standalone_state_space_result = run_rnn_state_space_condition_analysis(
#     "/data/trial_advantage.pkl",
#     # condition_columns=["switch_x_prev_outcome", "trial_position"],
#     output_dir="/results/figures/rnn_state_space_standalone",
#     pca_seed=0,
# )

# print(standalone_state_space_result)


# result = run_rnn_state_space_subject_analysis(
#     trial_advantage_pickle="/data/trial_advantage.pkl",
#     subject_embeddings_path="/data/mice_multisubject_train10_all_stages-gru_session_conditioning-260505-14-H64_lr5e_06-b97c06b79f07/outputs/subject_embeddings.pkl",
#     probability_column="p_model1_right",  # or "p_model1_left"
#     output_dir="/results/figures/rnn_state_space_subjects_right",
#     pca_seed=0,
#     pca_fit_fraction=0.5,
# )

# print(result["subject_probability_plots"])
# print(result["subject_embedding_task_space"])
# print(result["subject_embedding_distances"])
