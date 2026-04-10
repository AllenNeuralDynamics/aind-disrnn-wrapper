import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


from post_training_analysis import run_baseline_rl_post_training_analysis

result = run_baseline_rl_post_training_analysis(
    resolved_run_path="/code/resolved_run.json",
    fitting_df_path="/data/df_baseline_rl_fitting_260408/df_baseline_rl_fitting_260408.pkl",
    model_aliases=[
        "QLearning_L1F1_CK1_softmax",
        "QLearning_L2F1_softmax",
        "ForagingCompareThreshold",
    ],
    output_dir="/results",
    n_rollouts_per_session=5,
    session_id_policy="auto",
    fit_gap_policy="per_model_skip",
)

print(result["model_summary_json"])