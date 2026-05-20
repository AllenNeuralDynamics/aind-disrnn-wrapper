import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
    force=True,
)

from post_training_analysis import run_embedding_space_analysis

result = run_embedding_space_analysis(
    model_dir="/data/mice_multisubject_train10-gru_session_conditioning_lr_1e_5-260505/11",
    checkpoint_policy="best_eval",
    output_dir="/results",
    task_column="task",
    weekday_column="weekday",
    foraging_eff_column="foraging_eff_random_seed",
    bias_naive_column="bias_naive",
    reaction_time_column="reaction_time_median",
)

print(result)