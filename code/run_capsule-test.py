import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    stream=sys.stdout,
    force=True,
)


from post_training_analysis import (
    resolve_model_run,
    load_animal_session_history,
    simulate_model_sessions,
    compute_switch_stats,
)

run = resolve_model_run(
    "/code/ex_model_dir-train10_test3-disrnn-260324/9",
    split="train",
    checkpoint_policy="best_eval",
)
animal = load_animal_session_history(run, split="train")
sim = simulate_model_sessions(run, animal)
stats = compute_switch_stats(animal, sim, window_size=10)

print(run.to_dict())
print(animal.head())
print(sim.head())
print(stats.keys())