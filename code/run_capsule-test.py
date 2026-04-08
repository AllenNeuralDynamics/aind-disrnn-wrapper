import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
    force=True,
)


from post_training_analysis import (
    run_post_training_analysis,
    resolve_model_run,
    load_animal_session_history,
    simulate_model_sessions,
    compute_switch_stats,
)

result = run_post_training_analysis(
    model_dir="/data/mice_multisubject_train10-disrnn-260323/3",
    split="train",
    checkpoint_policy="best_eval",
    rollout_mode="curriculum_matched",
    n_rollouts_per_session=5,
    window_size=10,
    save_animal_session_history=False,
    output_dir="/results",
)