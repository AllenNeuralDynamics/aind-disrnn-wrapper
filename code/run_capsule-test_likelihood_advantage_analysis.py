import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
    force=True,
)

from post_training_analysis import run_likelihood_advantage_analysis

result = run_likelihood_advantage_analysis(
    "/data/mice_multisubject_train10-gru-260323/13",
    "/data/mice_multisubject_train10-baseline_rl_Bari-260414/1",
    split="combined",
    checkpoint_policy="best_eval",
    output_dir="/results",
    history_warmup=10,
    top_k_variables=3,
    jitter_seed=0,
)

print(result)
