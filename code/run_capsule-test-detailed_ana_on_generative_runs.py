import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


from post_training_analysis import run_post_training_analysis_from_saved_histories


# # gru
# result = run_post_training_analysis_from_saved_histories(
#     simulated_session_history_path="/data/generative_runs-multisubject_train10-gru-260323-14-H64_lr1e-05-ca41751fa454-5_runs/simulated_session_history.pkl",
#     resolved_run_path="/data/generative_runs-multisubject_train10-gru-260323-14-H64_lr1e-05-ca41751fa454-5_runs/resolved_run.json",
#     output_dir="/results",
#     window_size=10,
#     save_animal_session_history=False,
#     session_partitions=("train", "eval", "combined"),
# )

# # disrnn
# result = run_post_training_analysis_from_saved_histories(
#     simulated_session_history_path="/data/generative_runs-multisubject_train10-disrnn-260323-3-beta0.0001_lr0.001-bf7b659d308d-5_runs/simulated_session_history.pkl",
#     resolved_run_path="/data/generative_runs-multisubject_train10-disrnn-260323-3-beta0.0001_lr0.001-bf7b659d308d-5_runs/resolved_run.json",
#     output_dir="/results",
#     window_size=10,
#     save_animal_session_history=False,
#     session_partitions=("train", "eval", "combined"),
# )

# gru, no subject_embedding
result = run_post_training_analysis_from_saved_histories(
    simulated_session_history_path="/data/generative_runs-train10_test3-gru-260324-14-H64_lr5e-06-574b07ba8c89-5_runs/simulated_session_history.pkl",
    resolved_run_path="/data/generative_runs-train10_test3-gru-260324-14-H64_lr5e-06-574b07ba8c89-5_runs/resolved_run.json",
    output_dir="/results",
    window_size=10,
    save_animal_session_history=False,
    session_partitions=("train", "eval", "combined"),
)



# # bari
# result = run_post_training_analysis_from_saved_histories(
#     simulated_session_history_path="/data/generative_runs-multisubject_train10-baseline_rl/QLearning_L1F1_CK1_softmax/simulated_session_history.pkl",
#     animal_session_history_path="/data/generative_runs-multisubject_train10-baseline_rl/animal_session_history.pkl",
#     output_dir="/results",
#     window_size=10,
#     save_animal_session_history=False,
# )

# # hattori
# result = run_post_training_analysis_from_saved_histories(
#     simulated_session_history_path="/data/generative_runs-multisubject_train10-baseline_rl/QLearning_L2F1_softmax/simulated_session_history.pkl",
#     animal_session_history_path="/data/generative_runs-multisubject_train10-baseline_rl/animal_session_history.pkl",
#     output_dir="/results",
#     window_size=10,
#     save_animal_session_history=False,
# )

# # CTT
# result = run_post_training_analysis_from_saved_histories(
#     simulated_session_history_path="/data/generative_runs-multisubject_train10-baseline_rl/ForagingCompareThreshold/simulated_session_history.pkl",
#     animal_session_history_path="/data/generative_runs-multisubject_train10-baseline_rl/animal_session_history.pkl",
#     output_dir="/results",
#     window_size=10,
#     save_animal_session_history=False,
# )



# print(result["model_vs_animal_quantitative_summary"])